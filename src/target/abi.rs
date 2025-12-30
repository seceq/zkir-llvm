//! Calling convention and ABI definitions for ZKIR.
//!
//! ZKIR uses a calling convention (zkir-spec v3.4):
//! - Arguments in a0-a5 (r4-r9), then stack
//! - Return values in a0-a1 (r4-r5)
//! - Caller-saved: ra, a0-a5, t0-t1
//! - Callee-saved: sp, fp, s0-s3

use super::registers::{Register, ARG_REGS, RET_REGS};
use super::config::TargetConfig;

// Re-export ABI constants from zkir-spec for single source of truth
pub use zkir_spec::abi::{REGISTER_SIZE_BYTES, PARAM_ALIGNMENT, FRAME_ALIGNMENT};

/// Calling convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CallingConv {
    /// Standard ZKIR calling convention
    #[default]
    ZKIR,
    /// Fast calling convention (all args in registers, no callee-saved)
    Fast,
}

/// Location for a function argument or return value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgLocation {
    /// Value is in a single register
    Register(Register),
    /// Value is split across multiple registers
    RegisterPair(Vec<Register>),
    /// Value is on the stack at the given offset from SP
    Stack(i32),
    /// Value is passed by reference (pointer in register)
    Indirect(Register),
}

impl ArgLocation {
    /// Get the primary register if this is a register location.
    pub fn as_register(&self) -> Option<Register> {
        match self {
            ArgLocation::Register(r) => Some(*r),
            ArgLocation::RegisterPair(regs) => regs.first().copied(),
            ArgLocation::Indirect(r) => Some(*r),
            ArgLocation::Stack(_) => None,
        }
    }

    /// Get all registers used by this location.
    pub fn registers(&self) -> Vec<Register> {
        match self {
            ArgLocation::Register(r) => vec![*r],
            ArgLocation::RegisterPair(regs) => regs.clone(),
            ArgLocation::Indirect(r) => vec![*r],
            ArgLocation::Stack(_) => vec![],
        }
    }
}

/// ABI information for a function.
#[derive(Debug, Clone)]
pub struct FunctionABI {
    /// Locations for each parameter
    pub params: Vec<ArgLocation>,
    /// Location for return value
    pub ret: Option<ArgLocation>,
    /// Stack space needed for arguments
    pub arg_stack_size: u32,
}

/// Compute argument locations for a function signature.
pub fn compute_arg_locations(
    param_bits: &[u32],
    ret_bits: Option<u32>,
    config: &TargetConfig,
) -> FunctionABI {
    let mut params = Vec::with_capacity(param_bits.len());
    let mut reg_idx = 0usize;
    let mut stack_offset = 0i32;

    for &bits in param_bits {
        let regs_needed = config.regs_for_bits(bits) as usize;

        if regs_needed == 0 {
            // Zero-sized type, no location needed
            params.push(ArgLocation::Register(Register::R0));
        } else if regs_needed == 1 && reg_idx < ARG_REGS.len() {
            // Single register
            params.push(ArgLocation::Register(ARG_REGS[reg_idx]));
            reg_idx += 1;
        } else if regs_needed <= ARG_REGS.len() - reg_idx {
            // Multiple registers
            let regs: Vec<Register> = (0..regs_needed)
                .map(|i| ARG_REGS[reg_idx + i])
                .collect();
            reg_idx += regs_needed;
            params.push(ArgLocation::RegisterPair(regs));
        } else {
            // Spill to stack
            // Align stack offset
            stack_offset = (stack_offset + PARAM_ALIGNMENT as i32 - 1) & !(PARAM_ALIGNMENT as i32 - 1);
            params.push(ArgLocation::Stack(stack_offset));

            let size = bits.div_ceil(8) as i32;
            stack_offset += size;
        }
    }

    // Return value location
    let ret = ret_bits.map(|bits| {
        let regs_needed = config.regs_for_bits(bits) as usize;
        if regs_needed == 0 {
            ArgLocation::Register(Register::R0)
        } else if regs_needed == 1 {
            ArgLocation::Register(RET_REGS[0])
        } else if regs_needed <= RET_REGS.len() {
            let regs: Vec<Register> = (0..regs_needed)
                .map(|i| RET_REGS[i])
                .collect();
            ArgLocation::RegisterPair(regs)
        } else {
            // Large return value: pointer passed in a0
            ArgLocation::Indirect(RET_REGS[0])
        }
    });

    FunctionABI {
        params,
        ret,
        arg_stack_size: stack_offset as u32,
    }
}

/// Stack frame layout.
#[derive(Debug, Clone, Default)]
pub struct StackFrame {
    /// Size of local variables
    pub locals_size: u32,
    /// Size of spill slots
    pub spill_size: u32,
    /// Size of outgoing arguments (for calls)
    pub outgoing_args_size: u32,
    /// Callee-saved registers to preserve
    pub saved_regs: Vec<Register>,
}

impl StackFrame {
    /// Total frame size (must be 16-byte aligned).
    pub fn total_size(&self) -> u32 {
        let size = self.locals_size
            + self.spill_size
            + self.outgoing_args_size
            + (self.saved_regs.len() as u32 * REGISTER_SIZE_BYTES as u32)
            + (2 * REGISTER_SIZE_BYTES as u32); // Return address + frame pointer

        // Align to FRAME_ALIGNMENT bytes
        (size + FRAME_ALIGNMENT as u32 - 1) & !(FRAME_ALIGNMENT as u32 - 1)
    }

    /// Offset of return address from new SP.
    pub fn ra_offset(&self) -> i32 {
        (self.total_size() - REGISTER_SIZE_BYTES as u32) as i32
    }

    /// Offset of saved frame pointer from new SP.
    pub fn fp_offset(&self) -> i32 {
        (self.total_size() - (2 * REGISTER_SIZE_BYTES) as u32) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_args() {
        let config = TargetConfig::default(); // 40-bit
        let abi = compute_arg_locations(&[32, 32], Some(32), &config);

        assert_eq!(abi.params.len(), 2);
        assert_eq!(abi.params[0], ArgLocation::Register(Register::R4)); // a0
        assert_eq!(abi.params[1], ArgLocation::Register(Register::R5)); // a1
        assert_eq!(abi.ret, Some(ArgLocation::Register(Register::R4))); // a0
    }

    #[test]
    fn test_many_args() {
        let config = TargetConfig::default();
        // 8 i32 arguments - 6 in regs, 2 on stack
        let abi = compute_arg_locations(&[32; 8], None, &config);

        assert_eq!(abi.params.len(), 8);
        // First 6 in registers
        for i in 0..6 {
            assert!(matches!(abi.params[i], ArgLocation::Register(_)));
        }
        // Last 2 on stack
        assert!(matches!(abi.params[6], ArgLocation::Stack(_)));
        assert!(matches!(abi.params[7], ArgLocation::Stack(_)));
    }

    #[test]
    fn test_i64_arg_on_40bit() {
        let config = TargetConfig::default(); // 40-bit
        // i64 needs 2 registers on 40-bit config
        let abi = compute_arg_locations(&[64], Some(64), &config);

        assert!(matches!(&abi.params[0], ArgLocation::RegisterPair(regs) if regs.len() == 2));
        assert!(matches!(&abi.ret, Some(ArgLocation::RegisterPair(regs)) if regs.len() == 2));
    }

    #[test]
    fn test_i64_arg_on_80bit() {
        let config = TargetConfig::DATA_80; // 80-bit
        // i64 fits in 1 register on 80-bit config
        let abi = compute_arg_locations(&[64], Some(64), &config);

        assert_eq!(abi.params[0], ArgLocation::Register(Register::R4)); // a0
        assert_eq!(abi.ret, Some(ArgLocation::Register(Register::R4))); // a0
    }

    #[test]
    fn test_stack_frame() {
        let mut frame = StackFrame::default();
        frame.locals_size = 16;
        frame.spill_size = 8;
        frame.saved_regs = vec![Register::R10, Register::R11]; // s0, s1

        // Total should be aligned to 16
        let total = frame.total_size();
        assert_eq!(total % 16, 0);
        assert!(total >= 16 + 8 + 8 + 8); // locals + spill + saved + ra/fp
    }
}
