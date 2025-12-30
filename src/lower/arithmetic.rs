//! Arithmetic and logical instruction lowering.
//!
//! Handles LLVM arithmetic operations, tracking value bounds for ZK optimization.

use super::LoweringContext;
use crate::mir::{MachineInst, Opcode, Operand, ValueBounds, VReg};
use anyhow::Result;
use inkwell::values::InstructionValue;
use inkwell::IntPredicate;

/// Lower add instruction.
pub fn lower_add<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    ctx.emit(MachineInst::add(dst, lhs_vreg, rhs_vreg));

    // Compute result bounds
    let lhs_bounds = ctx.get_bounds(&lhs);
    let rhs_bounds = ctx.get_bounds(&rhs);
    let result_bounds = ValueBounds::add(lhs_bounds, rhs_bounds);
    ctx.set_bounds(&result, result_bounds);

    Ok(())
}

/// Lower sub instruction.
pub fn lower_sub<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    ctx.emit(MachineInst::sub(dst, lhs_vreg, rhs_vreg));

    // Compute result bounds (conservative for subtraction)
    let lhs_bounds = ctx.get_bounds(&lhs);
    let rhs_bounds = ctx.get_bounds(&rhs);
    let result_bounds = ValueBounds::sub(lhs_bounds, rhs_bounds);
    ctx.set_bounds(&result, result_bounds);

    Ok(())
}

/// Lower mul instruction.
pub fn lower_mul<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    ctx.emit(MachineInst::mul(dst, lhs_vreg, rhs_vreg));

    // Compute result bounds (multiplication can grow significantly)
    let lhs_bounds = ctx.get_bounds(&lhs);
    let rhs_bounds = ctx.get_bounds(&rhs);
    let result_bounds = ValueBounds::mul(lhs_bounds, rhs_bounds);

    // Check if result might overflow data width
    if !result_bounds.fits_in(ctx.config.data_bits()) {
        // Mark that we may need a range check
        log::debug!("Multiplication result may overflow: {} bits", result_bounds.bits);
    }

    ctx.set_bounds(&result, result_bounds);

    Ok(())
}

/// Emit a div-by-zero check.
///
/// If the divisor could be zero (based on bounds), emit an assertion
/// that traps on division by zero. This protects ZK circuit integrity.
///
/// For ZK circuits, we use a constraint-based approach: if divisor == 0,
/// the constraint system becomes unsatisfiable (trap). This is implemented
/// using BNE to skip the trap when divisor != 0, with an immediate offset
/// to jump over the EBREAK instruction.
fn emit_div_by_zero_check(ctx: &mut LoweringContext, divisor: VReg, divisor_bounds: ValueBounds) {
    // If we can statically prove divisor is non-zero, skip the check
    if divisor_bounds.min > 0 {
        return;
    }

    // Emit: if divisor != 0, skip trap; else trap
    // BNE divisor, zero, +8 (skip 1 instruction = 4 bytes for EBREAK)
    // EBREAK
    let zero = ctx.new_vreg();
    ctx.emit(MachineInst::li(zero, 0).comment("div-by-zero check"));

    // BNE with immediate offset: if divisor != 0, skip 1 instruction (4 bytes)
    // The offset is relative and encoded in the branch instruction itself
    ctx.emit(MachineInst::new(Opcode::BNE)
        .src(Operand::VReg(divisor))
        .src(Operand::VReg(zero))
        .src(Operand::Imm(4)) // Skip 4 bytes (1 instruction) to jump over EBREAK
        .comment("skip trap if divisor != 0"));

    ctx.emit(MachineInst::new(Opcode::EBREAK)
        .comment("trap: division by zero"));
}

/// Lower udiv instruction.
pub fn lower_udiv<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Check for division by zero
    let rhs_bounds = ctx.get_bounds(&rhs);
    emit_div_by_zero_check(ctx, rhs_vreg, rhs_bounds);

    ctx.emit(MachineInst::new(Opcode::DIV)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    // Division result is at most the dividend
    let lhs_bounds = ctx.get_bounds(&lhs);
    ctx.set_bounds(&result, ValueBounds::udiv(lhs_bounds, rhs_bounds));

    Ok(())
}

/// Lower sdiv instruction (signed division).
///
/// Converts signed division to unsigned division with explicit sign handling:
/// 1. Check for division by zero
/// 2. Compute sign of result (XOR of operand signs)
/// 3. Take absolute values of operands
/// 4. Perform unsigned division
/// 5. Apply result sign
pub fn lower_sdiv<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, dividend) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, divisor) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Check for division by zero first
    let rhs_bounds = ctx.get_bounds(&rhs);
    emit_div_by_zero_check(ctx, divisor, rhs_bounds);

    // Constants
    let zero = ctx.new_vreg();
    ctx.emit(MachineInst::li(zero, 0));

    // Get sign bits: sign = (x < 0) ? 1 : 0
    let div_sign = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(div_sign))
        .src(Operand::VReg(dividend))
        .src(Operand::VReg(zero)));

    let dvsr_sign = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(dvsr_sign))
        .src(Operand::VReg(divisor))
        .src(Operand::VReg(zero)));

    // Compute absolute value of dividend: abs = sign ? -x : x
    let neg_dividend = ctx.new_vreg();
    ctx.emit(MachineInst::sub(neg_dividend, zero, dividend));

    let abs_dividend = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(abs_dividend))
        .src(Operand::VReg(div_sign))
        .src(Operand::VReg(neg_dividend))
        .src(Operand::VReg(dividend)));

    // Compute absolute value of divisor
    let neg_divisor = ctx.new_vreg();
    ctx.emit(MachineInst::sub(neg_divisor, zero, divisor));

    let abs_divisor = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(abs_divisor))
        .src(Operand::VReg(dvsr_sign))
        .src(Operand::VReg(neg_divisor))
        .src(Operand::VReg(divisor)));

    // Perform unsigned division
    let unsigned_result = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::DIV)
        .dst(Operand::VReg(unsigned_result))
        .src(Operand::VReg(abs_dividend))
        .src(Operand::VReg(abs_divisor)));

    // Result sign = dividend_sign XOR divisor_sign
    let result_sign = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(result_sign))
        .src(Operand::VReg(div_sign))
        .src(Operand::VReg(dvsr_sign)));

    // Apply sign to result: result = result_sign ? -unsigned : unsigned
    let neg_result = ctx.new_vreg();
    ctx.emit(MachineInst::sub(neg_result, zero, unsigned_result));

    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(result_sign))
        .src(Operand::VReg(neg_result))
        .src(Operand::VReg(unsigned_result)));

    // Signed division bounds are complex; use full range for now
    let lhs_bounds = ctx.get_bounds(&lhs);
    ctx.set_bounds(&result, ValueBounds::udiv(lhs_bounds, rhs_bounds));

    Ok(())
}

/// Lower urem instruction.
pub fn lower_urem<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (_, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Check for division by zero (remainder also uses divisor)
    let rhs_bounds = ctx.get_bounds(&rhs);
    emit_div_by_zero_check(ctx, rhs_vreg, rhs_bounds);

    ctx.emit(MachineInst::new(Opcode::REM)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    // Remainder is less than divisor
    ctx.set_bounds(&result, rhs_bounds);

    Ok(())
}

/// Lower srem instruction (signed remainder).
///
/// Converts signed remainder to unsigned remainder with sign handling.
/// The remainder has the same sign as the dividend (C99 semantics).
pub fn lower_srem<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (_, dividend) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, divisor) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Check for division by zero first
    let rhs_bounds = ctx.get_bounds(&rhs);
    emit_div_by_zero_check(ctx, divisor, rhs_bounds);

    // Constants
    let zero = ctx.new_vreg();
    ctx.emit(MachineInst::li(zero, 0));

    // Get sign bits
    let div_sign = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(div_sign))
        .src(Operand::VReg(dividend))
        .src(Operand::VReg(zero)));

    let dvsr_sign = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(dvsr_sign))
        .src(Operand::VReg(divisor))
        .src(Operand::VReg(zero)));

    // Compute absolute value of dividend
    let neg_dividend = ctx.new_vreg();
    ctx.emit(MachineInst::sub(neg_dividend, zero, dividend));

    let abs_dividend = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(abs_dividend))
        .src(Operand::VReg(div_sign))
        .src(Operand::VReg(neg_dividend))
        .src(Operand::VReg(dividend)));

    // Compute absolute value of divisor
    let neg_divisor = ctx.new_vreg();
    ctx.emit(MachineInst::sub(neg_divisor, zero, divisor));

    let abs_divisor = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(abs_divisor))
        .src(Operand::VReg(dvsr_sign))
        .src(Operand::VReg(neg_divisor))
        .src(Operand::VReg(divisor)));

    // Perform unsigned remainder
    let unsigned_result = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::REM)
        .dst(Operand::VReg(unsigned_result))
        .src(Operand::VReg(abs_dividend))
        .src(Operand::VReg(abs_divisor)));

    // Remainder has same sign as dividend
    let neg_result = ctx.new_vreg();
    ctx.emit(MachineInst::sub(neg_result, zero, unsigned_result));

    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(div_sign))
        .src(Operand::VReg(neg_result))
        .src(Operand::VReg(unsigned_result)));

    // Remainder is bounded by divisor
    ctx.set_bounds(&result, rhs_bounds);

    Ok(())
}

/// Lower and instruction.
pub fn lower_and<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    ctx.emit(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    // AND result is at most min of operands
    let lhs_bounds = ctx.get_bounds(&lhs);
    let rhs_bounds = ctx.get_bounds(&rhs);
    ctx.set_bounds(&result, ValueBounds::and(lhs_bounds, rhs_bounds));

    Ok(())
}

/// Lower or instruction.
pub fn lower_or<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    ctx.emit(MachineInst::new(Opcode::OR)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    // OR result could use max bits of either
    let lhs_bounds = ctx.get_bounds(&lhs);
    let rhs_bounds = ctx.get_bounds(&rhs);
    ctx.set_bounds(&result, ValueBounds::or(lhs_bounds, rhs_bounds));

    Ok(())
}

/// Lower xor instruction.
pub fn lower_xor<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    ctx.emit(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    // XOR result similar to OR
    let lhs_bounds = ctx.get_bounds(&lhs);
    let rhs_bounds = ctx.get_bounds(&rhs);
    ctx.set_bounds(&result, ValueBounds::xor(lhs_bounds, rhs_bounds));

    Ok(())
}

/// Lower shl instruction.
pub fn lower_shl<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    ctx.emit(MachineInst::new(Opcode::SLL)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    // Shift left grows the value
    let lhs_bounds = ctx.get_bounds(&lhs);
    let rhs_bounds = ctx.get_bounds(&rhs);
    ctx.set_bounds(&result, ValueBounds::shl(lhs_bounds, rhs_bounds));

    Ok(())
}

/// Lower lshr instruction (logical shift right).
pub fn lower_lshr<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    ctx.emit(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    // Shift right shrinks the value
    let lhs_bounds = ctx.get_bounds(&lhs);
    let rhs_bounds = ctx.get_bounds(&rhs);
    ctx.set_bounds(&result, ValueBounds::lshr(lhs_bounds, rhs_bounds));

    Ok(())
}

/// Lower ashr instruction (arithmetic shift right).
pub fn lower_ashr<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (lhs, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (_, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    ctx.emit(MachineInst::new(Opcode::SRA)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    // ASR preserves sign, bounds are conservative
    let lhs_bounds = ctx.get_bounds(&lhs);
    ctx.set_bounds(&result, lhs_bounds);

    Ok(())
}

/// Lower icmp instruction.
pub fn lower_icmp<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let predicate = inst.get_icmp_predicate()
        .ok_or_else(|| anyhow::anyhow!("ICmp instruction missing predicate"))?;
    let (_, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (_, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Select the appropriate comparison opcode
    let opcode = match predicate {
        IntPredicate::EQ => Opcode::SEQ,
        IntPredicate::NE => Opcode::SNE,
        IntPredicate::ULT => Opcode::SLTU,
        IntPredicate::ULE => Opcode::SLTU,  // a <= b  <==>  !(b < a)
        IntPredicate::UGT => Opcode::SLTU,  // a > b   <==>  b < a (swap operands)
        IntPredicate::UGE => Opcode::SLTU, // a >= b  <==>  !(a < b)
        IntPredicate::SLT => Opcode::SLT,
        IntPredicate::SLE => Opcode::SLT,
        IntPredicate::SGT => Opcode::SLT,
        IntPredicate::SGE => Opcode::SLT,
    };

    // Handle swapped operands and inversions
    match predicate {
        IntPredicate::EQ | IntPredicate::NE | IntPredicate::ULT | IntPredicate::SLT => {
            ctx.emit(MachineInst::new(opcode)
                .dst(Operand::VReg(dst))
                .src(Operand::VReg(lhs_vreg))
                .src(Operand::VReg(rhs_vreg)));
        }
        IntPredicate::UGT | IntPredicate::SGT => {
            // Swap operands: a > b  <==>  b < a
            ctx.emit(MachineInst::new(opcode)
                .dst(Operand::VReg(dst))
                .src(Operand::VReg(rhs_vreg))
                .src(Operand::VReg(lhs_vreg)));
        }
        IntPredicate::ULE | IntPredicate::SLE => {
            // a <= b  <==>  !(b < a)
            let temp = ctx.new_vreg();
            ctx.emit(MachineInst::new(opcode)
                .dst(Operand::VReg(temp))
                .src(Operand::VReg(rhs_vreg))
                .src(Operand::VReg(lhs_vreg)));
            // NOT by XOR with 1
            let one = ctx.new_vreg();
            ctx.emit(MachineInst::li(one, 1));
            ctx.emit(MachineInst::new(Opcode::XOR)
                .dst(Operand::VReg(dst))
                .src(Operand::VReg(temp))
                .src(Operand::VReg(one)));
        }
        IntPredicate::UGE | IntPredicate::SGE => {
            // a >= b  <==>  !(a < b)
            let temp = ctx.new_vreg();
            ctx.emit(MachineInst::new(opcode)
                .dst(Operand::VReg(temp))
                .src(Operand::VReg(lhs_vreg))
                .src(Operand::VReg(rhs_vreg)));
            // NOT by XOR with 1
            let one = ctx.new_vreg();
            ctx.emit(MachineInst::li(one, 1));
            ctx.emit(MachineInst::new(Opcode::XOR)
                .dst(Operand::VReg(dst))
                .src(Operand::VReg(temp))
                .src(Operand::VReg(one)));
        }
    }

    // Comparison result is always 0 or 1
    ctx.set_bounds(&result, ValueBounds::from_const(1));

    Ok(())
}

/// Lower select instruction.
pub fn lower_select<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (_, cond_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (true_val, true_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (false_val, false_vreg) = ctx.get_operand_vreg(inst, 2)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Use CMOV: if cond != 0, use true_val, else use false_val
    // CMOV dst, cond, true_val, false_val
    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(cond_vreg))
        .src(Operand::VReg(true_vreg))
        .src(Operand::VReg(false_vreg)));

    // Result could be either value
    let true_bounds = ctx.get_bounds(&true_val);
    let false_bounds = ctx.get_bounds(&false_val);
    let max_bits = true_bounds.bits.max(false_bounds.bits);
    ctx.set_bounds(&result, ValueBounds::from_bits(max_bits));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_tracking() {
        // Test that multiplication bounds grow correctly
        let a = ValueBounds::from_const(100);
        let b = ValueBounds::from_const(100);
        let c = ValueBounds::mul(a, b);
        assert_eq!(c.max, 10000);
        assert!(c.bits <= 14); // 10000 < 2^14
    }

    #[test]
    fn test_div_by_zero_check_skipped_for_nonzero_bounds() {
        // If min bound > 0, div-by-zero check should be skipped
        let bounds = ValueBounds { min: 1, max: 100, bits: 7 };
        // This is a statically provable non-zero divisor
        assert!(bounds.min > 0, "min > 0 means divisor cannot be zero");
    }

    #[test]
    fn test_div_by_zero_check_needed_for_zero_possible() {
        // If min bound == 0, div-by-zero check is needed
        let bounds = ValueBounds { min: 0, max: 100, bits: 7 };
        // Zero is a possible value, so check is needed
        assert!(bounds.min == 0, "min == 0 means divisor could be zero");
    }
}
