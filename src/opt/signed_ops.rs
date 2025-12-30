//! Signed operation lowering pass.
//!
//! NOTE: Signed division and remainder are now handled directly during LLVM
//! lowering in `src/lower/arithmetic.rs` (lower_sdiv and lower_srem).
//! This pass is kept as a no-op for compatibility but the expansion functions
//! are available for reference and testing.

use crate::mir::MachineFunction;
use crate::target::config::TargetConfig;
use anyhow::Result;

#[cfg(test)]
use crate::mir::{MachineInst, Opcode, Operand, VReg};

/// Lower signed division and remainder operations.
///
/// NOTE: This is now a no-op pass. Signed operations are handled during
/// LLVM lowering in `src/lower/arithmetic.rs`.
pub fn lower_signed_ops(_func: &mut MachineFunction, _config: &TargetConfig) -> Result<()> {
    // Signed operations are now expanded during lowering, not as an optimization pass.
    // See lower_sdiv and lower_srem in src/lower/arithmetic.rs
    Ok(())
}

/// Generate instructions for signed division.
///
/// Algorithm:
/// 1. Compute absolute values of dividend and divisor
/// 2. Perform unsigned division
/// 3. Determine sign of result (xor of input signs)
/// 4. Apply sign to result
///
/// Note: This logic is implemented directly in lower_sdiv in arithmetic.rs.
/// This function is kept for testing and documentation purposes.
#[cfg(test)]
fn expand_signed_div(
    func: &mut MachineFunction,
    dst: VReg,
    dividend: VReg,
    divisor: VReg,
) -> Vec<MachineInst> {
    let mut result = Vec::new();

    let div_sign = func.new_vreg();
    let dvsr_sign = func.new_vreg();
    let zero = func.new_vreg();

    result.push(MachineInst::li(zero, 0));

    // div_sign = (dividend < 0) ? 1 : 0
    result.push(
        MachineInst::new(Opcode::SLT)
            .dst(Operand::VReg(div_sign))
            .src(Operand::VReg(dividend))
            .src(Operand::VReg(zero)),
    );

    // dvsr_sign = (divisor < 0) ? 1 : 0
    result.push(
        MachineInst::new(Opcode::SLT)
            .dst(Operand::VReg(dvsr_sign))
            .src(Operand::VReg(divisor))
            .src(Operand::VReg(zero)),
    );

    // Compute absolute values
    let neg_dividend = func.new_vreg();
    result.push(MachineInst::sub(neg_dividend, zero, dividend));

    let abs_dividend = func.new_vreg();
    result.push(
        MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(abs_dividend))
            .src(Operand::VReg(div_sign))
            .src(Operand::VReg(neg_dividend))
            .src(Operand::VReg(dividend)),
    );

    let neg_divisor = func.new_vreg();
    result.push(MachineInst::sub(neg_divisor, zero, divisor));

    let abs_divisor = func.new_vreg();
    result.push(
        MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(abs_divisor))
            .src(Operand::VReg(dvsr_sign))
            .src(Operand::VReg(neg_divisor))
            .src(Operand::VReg(divisor)),
    );

    // Perform unsigned division
    let unsigned_result = func.new_vreg();
    result.push(
        MachineInst::new(Opcode::DIV)
            .dst(Operand::VReg(unsigned_result))
            .src(Operand::VReg(abs_dividend))
            .src(Operand::VReg(abs_divisor)),
    );

    // result_sign = div_sign XOR dvsr_sign
    let result_sign = func.new_vreg();
    result.push(
        MachineInst::new(Opcode::XOR)
            .dst(Operand::VReg(result_sign))
            .src(Operand::VReg(div_sign))
            .src(Operand::VReg(dvsr_sign)),
    );

    // Apply sign to result
    let neg_result = func.new_vreg();
    result.push(MachineInst::sub(neg_result, zero, unsigned_result));

    result.push(
        MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(dst))
            .src(Operand::VReg(result_sign))
            .src(Operand::VReg(neg_result))
            .src(Operand::VReg(unsigned_result)),
    );

    result
}

/// Generate instructions for signed remainder.
///
/// Algorithm:
/// 1. Compute absolute values of dividend and divisor
/// 2. Perform unsigned remainder
/// 3. Apply dividend's sign to result (remainder has same sign as dividend)
///
/// Note: This logic is implemented directly in lower_srem in arithmetic.rs.
/// This function is kept for testing and documentation purposes.
#[cfg(test)]
fn expand_signed_rem(
    func: &mut MachineFunction,
    dst: VReg,
    dividend: VReg,
    divisor: VReg,
) -> Vec<MachineInst> {
    let mut result = Vec::new();

    let div_sign = func.new_vreg();
    let dvsr_sign = func.new_vreg();
    let zero = func.new_vreg();

    result.push(MachineInst::li(zero, 0));

    result.push(
        MachineInst::new(Opcode::SLT)
            .dst(Operand::VReg(div_sign))
            .src(Operand::VReg(dividend))
            .src(Operand::VReg(zero)),
    );

    result.push(
        MachineInst::new(Opcode::SLT)
            .dst(Operand::VReg(dvsr_sign))
            .src(Operand::VReg(divisor))
            .src(Operand::VReg(zero)),
    );

    let neg_dividend = func.new_vreg();
    result.push(MachineInst::sub(neg_dividend, zero, dividend));

    let abs_dividend = func.new_vreg();
    result.push(
        MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(abs_dividend))
            .src(Operand::VReg(div_sign))
            .src(Operand::VReg(neg_dividend))
            .src(Operand::VReg(dividend)),
    );

    let neg_divisor = func.new_vreg();
    result.push(MachineInst::sub(neg_divisor, zero, divisor));

    let abs_divisor = func.new_vreg();
    result.push(
        MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(abs_divisor))
            .src(Operand::VReg(dvsr_sign))
            .src(Operand::VReg(neg_divisor))
            .src(Operand::VReg(divisor)),
    );

    let unsigned_result = func.new_vreg();
    result.push(
        MachineInst::new(Opcode::REM)
            .dst(Operand::VReg(unsigned_result))
            .src(Operand::VReg(abs_dividend))
            .src(Operand::VReg(abs_divisor)),
    );

    // Remainder has same sign as dividend
    let neg_result = func.new_vreg();
    result.push(MachineInst::sub(neg_result, zero, unsigned_result));

    result.push(
        MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(dst))
            .src(Operand::VReg(div_sign))
            .src(Operand::VReg(neg_result))
            .src(Operand::VReg(unsigned_result)),
    );

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MachineBlock;

    #[test]
    fn test_signed_div_expansion() {
        let mut func = MachineFunction::new("test");
        let dst = func.new_vreg();
        let dividend = func.new_vreg();
        let divisor = func.new_vreg();

        let insts = expand_signed_div(&mut func, dst, dividend, divisor);

        // Should generate multiple instructions
        assert!(insts.len() > 5);

        // Should include SLT for sign detection
        assert!(insts.iter().any(|i| i.opcode == Opcode::SLT));

        // Should include DIV for the actual division
        assert!(insts.iter().any(|i| i.opcode == Opcode::DIV));

        // Should include XOR for sign computation
        assert!(insts.iter().any(|i| i.opcode == Opcode::XOR));
    }

    #[test]
    fn test_signed_rem_expansion() {
        let mut func = MachineFunction::new("test");
        let dst = func.new_vreg();
        let dividend = func.new_vreg();
        let divisor = func.new_vreg();

        let insts = expand_signed_rem(&mut func, dst, dividend, divisor);

        // Should generate multiple instructions
        assert!(insts.len() > 5);

        // Should include SLT for sign detection
        assert!(insts.iter().any(|i| i.opcode == Opcode::SLT));

        // Should include REM for the actual remainder
        assert!(insts.iter().any(|i| i.opcode == Opcode::REM));

        // Should NOT include XOR (remainder sign follows dividend only)
        let xor_count = insts.iter().filter(|i| i.opcode == Opcode::XOR).count();
        assert_eq!(xor_count, 0);
    }

    #[test]
    fn test_lower_signed_ops_pass() {
        let mut func = MachineFunction::new("test");
        let config = TargetConfig::default();

        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 3));
        entry.push(
            MachineInst::new(Opcode::DIV)
                .dst(Operand::VReg(v2))
                .src(Operand::VReg(v0))
                .src(Operand::VReg(v1)),
        );
        entry.push(MachineInst::ret());
        func.add_block(entry);

        // Should not panic (now a no-op pass)
        lower_signed_ops(&mut func, &config).unwrap();

        // Instructions should still be there
        let block = func.get_block("entry").unwrap();
        assert!(block.insts.len() >= 4);
    }
}
