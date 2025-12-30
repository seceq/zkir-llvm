//! Register definitions for ZKIR.
//!
//! ZKIR has 16 general-purpose registers (r0-r15) with specific ABI roles.
//! This follows the zkir-spec v3.4 calling convention.
//!
//! # Implementation Note
//!
//! This module re-exports `Register` from zkir-spec to ensure consistency with
//! the specification. Additional zkir-llvm-specific functionality (register
//! classification, calling convention helpers) is provided as extensions.

// Re-export Register from zkir-spec for single source of truth
pub use zkir_spec::Register;
pub use zkir_spec::NUM_REGISTERS;

/// zkir-llvm extensions to the Register type via trait.
///
/// Since we can't add inherent methods to types defined outside our crate,
/// we provide compatibility methods via a trait extension.
pub trait RegisterExt {
    /// Get the register number (compatibility alias for index).
    fn num(self) -> u8;

    /// Alias for num() - get the register number (compatibility alias for index).
    fn number(self) -> u8;

    /// Get the ABI name for this register (compatibility alias for name).
    fn abi_name(self) -> &'static str;

    /// Is this a caller-saved register?
    fn is_caller_saved(self) -> bool;

    /// Is this a callee-saved register?
    fn is_callee_saved(self) -> bool;

    /// Is this register allocatable for general use?
    fn is_allocatable(self) -> bool;

    /// Is this an argument register?
    fn is_arg_reg(self) -> bool;
}

impl RegisterExt for Register {
    #[inline]
    fn num(self) -> u8 {
        self.index()
    }

    #[inline]
    fn number(self) -> u8 {
        self.index()
    }

    #[inline]
    fn abi_name(self) -> &'static str {
        self.name()
    }

    #[inline]
    fn is_caller_saved(self) -> bool {
        matches!(self,
            Register::R1 |  // ra
            Register::R4 | Register::R5 |  // a0-a1
            Register::R6 | Register::R7 | Register::R8 | Register::R9 |  // a2-a5
            Register::R14 | Register::R15  // t0-t1
        )
    }

    #[inline]
    fn is_callee_saved(self) -> bool {
        matches!(self,
            Register::R2 |  // sp
            Register::R3 |  // fp
            Register::R10 | Register::R11 |  // s0-s1
            Register::R12 | Register::R13    // s2-s3
        )
    }

    #[inline]
    fn is_allocatable(self) -> bool {
        // Exclude: zero, sp, fp (reserved)
        !matches!(self,
            Register::R0 |  // zero - hardwired
            Register::R2 |  // sp - reserved
            Register::R3    // fp - reserved
        )
    }

    #[inline]
    fn is_arg_reg(self) -> bool {
        matches!(self,
            Register::R4 | Register::R5 |  // a0-a1
            Register::R6 | Register::R7 |  // a2-a3
            Register::R8 | Register::R9    // a4-a5
        )
    }
}

/// All registers in order (compatibility constant).
pub const ALL_REGISTERS: [Register; 16] = [
    Register::R0, Register::R1, Register::R2, Register::R3,
    Register::R4, Register::R5, Register::R6, Register::R7,
    Register::R8, Register::R9, Register::R10, Register::R11,
    Register::R12, Register::R13, Register::R14, Register::R15,
];

/// Create a register from its number (compatibility alias for from_index).
#[inline]
pub fn from_num(n: u8) -> Option<Register> {
    Register::from_index(n)
}

/// Get argument register by index (0-5).
pub fn arg_reg(index: usize) -> Option<Register> {
    ARG_REGS.get(index).copied()
}

// Note: Display trait is already implemented by zkir-spec

/// ABI names for registers (compatibility - use Register::name() instead).
/// This is kept for backward compatibility but delegates to zkir-spec.
pub const ABI_NAMES: [&str; 16] = [
    "zero", // r0
    "ra",   // r1
    "sp",   // r2
    "fp",   // r3
    "a0",   // r4
    "a1",   // r5
    "a2",   // r6
    "a3",   // r7
    "a4",   // r8
    "a5",   // r9
    "s0",   // r10
    "s1",   // r11
    "s2",   // r12
    "s3",   // r13
    "t0",   // r14
    "t1",   // r15
];

/// All registers (compatibility alias).
pub const REGISTERS: [Register; 16] = ALL_REGISTERS;

/// Argument registers (a0-a5, zkir-spec v3.4).
pub const ARG_REGS: [Register; 6] = [
    Register::R4, Register::R5,   // a0-a1
    Register::R6, Register::R7,   // a2-a3
    Register::R8, Register::R9,   // a4-a5
];

/// Return value registers (a0-a1).
pub const RET_REGS: [Register; 2] = [Register::R4, Register::R5];

/// Temporary registers (t0-t1).
pub const TEMP_REGS: [Register; 2] = [Register::R14, Register::R15];

/// Callee-saved registers (s0-s3, sp, fp).
pub const CALLEE_SAVED: [Register; 6] = [
    Register::R2,   // sp
    Register::R3,   // fp
    Register::R10,  // s0
    Register::R11,  // s1
    Register::R12,  // s2
    Register::R13,  // s3
];

/// Allocatable registers (for register allocation).
/// Excludes: zero, sp, fp
pub const ALLOCATABLE: [Register; 13] = [
    Register::R1,   // ra (caller-saved, but allocatable)
    Register::R4, Register::R5,   // a0-a1
    Register::R6, Register::R7,   // a2-a3
    Register::R8, Register::R9,   // a4-a5
    Register::R10, Register::R11, // s0-s1
    Register::R12, Register::R13, // s2-s3
    Register::R14, Register::R15, // t0-t1
];

/// Alias for ALLOCATABLE for compatibility.
pub const ALLOCATABLE_REGS: [Register; 13] = ALLOCATABLE;

/// Caller-saved registers (not preserved across calls).
pub const CALLER_SAVED: [Register; 9] = [
    Register::R1,   // ra
    Register::R4, Register::R5,   // a0-a1
    Register::R6, Register::R7,   // a2-a3
    Register::R8, Register::R9,   // a4-a5
    Register::R14, Register::R15, // t0-t1
];

/// Register class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterClass {
    /// General-purpose registers (all except zero)
    GPR,
    /// Argument registers only
    Arg,
    /// Temporary registers only
    Temp,
    /// Callee-saved registers only
    CalleeSaved,
}

impl RegisterClass {
    /// Get all registers in this class.
    pub fn registers(self) -> &'static [Register] {
        match self {
            RegisterClass::GPR => &ALLOCATABLE,
            RegisterClass::Arg => &ARG_REGS,
            RegisterClass::Temp => &TEMP_REGS,
            RegisterClass::CalleeSaved => &CALLEE_SAVED,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_from_num() {
        // Test compatibility alias
        assert_eq!(from_num(0), Some(Register::R0));
        assert_eq!(from_num(15), Some(Register::R15));
        assert_eq!(from_num(16), None);

        // Test zkir-spec method
        assert_eq!(Register::from_index(0), Some(Register::R0));
        assert_eq!(Register::from_index(15), Some(Register::R15));
    }

    #[test]
    fn test_abi_names() {
        // Test compatibility alias
        assert_eq!(Register::R0.abi_name(), "zero");
        assert_eq!(Register::R1.abi_name(), "ra");
        assert_eq!(Register::R3.abi_name(), "fp");
        assert_eq!(Register::R4.abi_name(), "a0");
        assert_eq!(Register::R10.abi_name(), "s0");
        assert_eq!(Register::R14.abi_name(), "t0");

        // Test zkir-spec method
        assert_eq!(Register::R0.name(), "zero");
        assert_eq!(Register::R1.name(), "ra");
        assert_eq!(Register::FP.name(), "fp");
        assert_eq!(Register::A0.name(), "a0");
        assert_eq!(Register::S0.name(), "s0");
        assert_eq!(Register::T0.name(), "t0");
    }

    #[test]
    fn test_register_constants() {
        // Verify zkir-spec constants work
        assert_eq!(Register::ZERO, Register::R0);
        assert_eq!(Register::RA, Register::R1);
        assert_eq!(Register::SP, Register::R2);
        assert_eq!(Register::FP, Register::R3);
        assert_eq!(Register::A0, Register::R4);
        assert_eq!(Register::S0, Register::R10);
        assert_eq!(Register::T0, Register::R14);
    }

    #[test]
    fn test_allocatable() {
        assert!(!Register::R0.is_allocatable());  // zero
        assert!(!Register::R2.is_allocatable());  // sp
        assert!(!Register::R3.is_allocatable());  // fp
        assert!(Register::R4.is_allocatable());   // a0
        assert!(Register::R10.is_allocatable());  // s0
        assert!(Register::R14.is_allocatable());  // t0
    }

    #[test]
    fn test_caller_callee_saved() {
        assert!(Register::R1.is_caller_saved());   // ra
        assert!(Register::R4.is_caller_saved());   // a0
        assert!(Register::R14.is_caller_saved());  // t0
        assert!(Register::R2.is_callee_saved());   // sp
        assert!(Register::R3.is_callee_saved());   // fp
        assert!(Register::R10.is_callee_saved());  // s0
    }

    #[test]
    fn test_zkir_spec_alignment() {
        // Ensure our extensions match zkir-spec behavior
        assert_eq!(NUM_REGISTERS, 16);
        assert_eq!(ALL_REGISTERS.len(), NUM_REGISTERS);

        // Verify index() and num() are equivalent
        for i in 0..16 {
            let reg = from_num(i as u8).unwrap();
            assert_eq!(reg.num(), reg.index());
            assert_eq!(reg.abi_name(), reg.name());
        }
    }
}

