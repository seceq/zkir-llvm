//! Constant folding optimization pass.
//!
//! Evaluates operations on constants at compile time and replaces them
//! with load-immediate instructions.

use crate::mir::{MachineFunction, MachineInst, Opcode, Operand, VReg};
use anyhow::Result;
use std::collections::HashMap;

/// Perform constant folding on a function.
pub fn fold_constants(func: &mut MachineFunction) -> Result<u32> {
    let mut folded_count = 0;
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        if let Some(block) = func.get_block_mut(&label) {
            let count = fold_constants_in_block(&mut block.insts);
            folded_count += count;
        }
    }

    Ok(folded_count)
}

/// Fold constants in a single block.
fn fold_constants_in_block(insts: &mut [MachineInst]) -> u32 {
    // Map from vreg to known constant value
    let mut constants: HashMap<VReg, i64> = HashMap::new();
    let mut folded = 0;
    let mut replacements: Vec<(usize, MachineInst)> = Vec::new();

    for (idx, inst) in insts.iter().enumerate() {
        // Track constant definitions from LI
        if inst.opcode == Opcode::LI {
            if let (Some(Operand::VReg(dst)), Some(Operand::Imm(val))) =
                (&inst.dst, inst.srcs.first())
            {
                constants.insert(*dst, *val);
            }
            continue;
        }

        // Try to fold binary operations
        if let Some(new_inst) = try_fold_binary(inst, &constants) {
            replacements.push((idx, new_inst));
            folded += 1;
        }

        // Track constant propagation through MOV
        if inst.opcode == Opcode::MOV {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src))) =
                (&inst.dst, inst.srcs.first())
            {
                if let Some(&val) = constants.get(src) {
                    constants.insert(*dst, val);
                }
            }
        }

        // ADDI with constant base can produce a constant
        if inst.opcode == Opcode::ADDI {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), Some(Operand::Imm(imm))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if let Some(&base_val) = constants.get(src) {
                    let result = base_val.wrapping_add(*imm);
                    constants.insert(*dst, result);
                }
            }
        }
    }

    // Apply replacements
    for (idx, new_inst) in replacements {
        insts[idx] = new_inst;
    }

    folded
}

/// Try to fold a binary operation.
fn try_fold_binary(
    inst: &MachineInst,
    constants: &HashMap<VReg, i64>,
) -> Option<MachineInst> {
    let dst = inst.dst.as_ref()?.as_vreg()?;

    // Get operand values
    let (val1, val2) = match inst.opcode {
        Opcode::ADD | Opcode::SUB | Opcode::MUL | Opcode::DIV | Opcode::REM |
        Opcode::AND | Opcode::OR | Opcode::XOR |
        Opcode::SLL | Opcode::SRL | Opcode::SRA |
        Opcode::SLT | Opcode::SLTU | Opcode::SEQ | Opcode::SNE => {
            let src1 = inst.srcs.first()?.as_vreg()?;
            let src2 = inst.srcs.get(1)?.as_vreg()?;
            (constants.get(&src1)?, constants.get(&src2)?)
        }
        _ => return None,
    };

    // Evaluate the operation
    let result = match inst.opcode {
        Opcode::ADD => val1.wrapping_add(*val2),
        Opcode::SUB => val1.wrapping_sub(*val2),
        Opcode::MUL => val1.wrapping_mul(*val2),
        Opcode::DIV => {
            if *val2 == 0 {
                return None; // Don't fold division by zero
            }
            val1 / val2
        }
        Opcode::REM => {
            if *val2 == 0 {
                return None;
            }
            val1 % val2
        }
        Opcode::AND => val1 & val2,
        Opcode::OR => val1 | val2,
        Opcode::XOR => val1 ^ val2,
        Opcode::SLL => val1.wrapping_shl(*val2 as u32),
        Opcode::SRL => (*val1 as u64).wrapping_shr(*val2 as u32) as i64,
        Opcode::SRA => val1.wrapping_shr(*val2 as u32),
        Opcode::SLT => if val1 < val2 { 1 } else { 0 },
        Opcode::SLTU => if (*val1 as u64) < (*val2 as u64) { 1 } else { 0 },
        Opcode::SEQ => if val1 == val2 { 1 } else { 0 },
        Opcode::SNE => if val1 != val2 { 1 } else { 0 },
        _ => return None,
    };

    // Replace with LI
    Some(MachineInst::li(dst, result).comment("const folded"))
}

/// Evaluate a unary operation at compile time.
#[allow(dead_code)]
fn try_fold_unary(
    inst: &MachineInst,
    constants: &HashMap<VReg, i64>,
) -> Option<MachineInst> {
    let dst = inst.dst.as_ref()?.as_vreg()?;

    match inst.opcode {
        Opcode::NOT => {
            let src = inst.srcs.first()?.as_vreg()?;
            let val = constants.get(&src)?;
            let result = !val;
            Some(MachineInst::li(dst, result).comment("const folded"))
        }
        _ => None,
    }
}

/// Algebraic simplifications (identities).
pub fn simplify_algebra(func: &mut MachineFunction) -> Result<u32> {
    let mut simplified = 0;
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        if let Some(block) = func.get_block_mut(&label) {
            // Track which vregs are known to be zero (first pass)
            let mut zeros: HashMap<VReg, bool> = HashMap::new();
            for inst in block.insts.iter() {
                if inst.opcode == Opcode::LI {
                    if let (Some(Operand::VReg(dst)), Some(Operand::Imm(0))) =
                        (&inst.dst, inst.srcs.first())
                    {
                        zeros.insert(*dst, true);
                    }
                }
            }

            // Second pass: apply simplifications
            for inst in &mut block.insts {
                if let Some(new_inst) = simplify_inst(inst, &zeros) {
                    *inst = new_inst;
                    simplified += 1;
                }
            }
        }
    }

    Ok(simplified)
}

/// Simplify a single instruction using algebraic identities.
fn simplify_inst(
    inst: &MachineInst,
    zeros: &HashMap<VReg, bool>,
) -> Option<MachineInst> {
    let dst = inst.dst.as_ref()?.as_vreg()?;

    match inst.opcode {
        // x + 0 = x
        Opcode::ADD => {
            let src1 = inst.srcs.first()?.as_vreg()?;
            let src2 = inst.srcs.get(1)?.as_vreg()?;

            if zeros.get(&src2).is_some() {
                return Some(MachineInst::mov(dst, src1).comment("x + 0 = x"));
            }
            if zeros.get(&src1).is_some() {
                return Some(MachineInst::mov(dst, src2).comment("0 + x = x"));
            }
            None
        }

        // x - 0 = x
        Opcode::SUB => {
            let src1 = inst.srcs.first()?.as_vreg()?;
            let src2 = inst.srcs.get(1)?.as_vreg()?;

            if zeros.get(&src2).is_some() {
                return Some(MachineInst::mov(dst, src1).comment("x - 0 = x"));
            }
            // x - x = 0
            if src1 == src2 {
                return Some(MachineInst::li(dst, 0).comment("x - x = 0"));
            }
            None
        }

        // x * 0 = 0, x * 1 = x
        Opcode::MUL => {
            let src1 = inst.srcs.first()?.as_vreg()?;
            let src2 = inst.srcs.get(1)?.as_vreg()?;

            if zeros.get(&src1).is_some() || zeros.get(&src2).is_some() {
                return Some(MachineInst::li(dst, 0).comment("x * 0 = 0"));
            }
            None
        }

        // x & 0 = 0
        Opcode::AND => {
            let src1 = inst.srcs.first()?.as_vreg()?;
            let src2 = inst.srcs.get(1)?.as_vreg()?;

            if zeros.get(&src1).is_some() || zeros.get(&src2).is_some() {
                return Some(MachineInst::li(dst, 0).comment("x & 0 = 0"));
            }
            // x & x = x
            if src1 == src2 {
                return Some(MachineInst::mov(dst, src1).comment("x & x = x"));
            }
            None
        }

        // x | 0 = x
        Opcode::OR => {
            let src1 = inst.srcs.first()?.as_vreg()?;
            let src2 = inst.srcs.get(1)?.as_vreg()?;

            if zeros.get(&src2).is_some() {
                return Some(MachineInst::mov(dst, src1).comment("x | 0 = x"));
            }
            if zeros.get(&src1).is_some() {
                return Some(MachineInst::mov(dst, src2).comment("0 | x = x"));
            }
            // x | x = x
            if src1 == src2 {
                return Some(MachineInst::mov(dst, src1).comment("x | x = x"));
            }
            None
        }

        // x ^ 0 = x, x ^ x = 0
        Opcode::XOR => {
            let src1 = inst.srcs.first()?.as_vreg()?;
            let src2 = inst.srcs.get(1)?.as_vreg()?;

            if zeros.get(&src2).is_some() {
                return Some(MachineInst::mov(dst, src1).comment("x ^ 0 = x"));
            }
            if zeros.get(&src1).is_some() {
                return Some(MachineInst::mov(dst, src2).comment("0 ^ x = x"));
            }
            if src1 == src2 {
                return Some(MachineInst::li(dst, 0).comment("x ^ x = 0"));
            }
            None
        }

        // x << 0 = x, x >> 0 = x
        Opcode::SLL | Opcode::SRL | Opcode::SRA => {
            let src1 = inst.srcs.first()?.as_vreg()?;
            let src2 = inst.srcs.get(1)?.as_vreg()?;

            if zeros.get(&src2).is_some() {
                return Some(MachineInst::mov(dst, src1).comment("x << 0 = x"));
            }
            None
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MachineBlock;

    #[test]
    fn test_fold_add() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 20));
        entry.push(MachineInst::add(v2, v0, v1)); // Should fold to li(v2, 30)
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let folded = fold_constants(&mut func).unwrap();
        assert_eq!(folded, 1);

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[2].opcode, Opcode::LI);
        assert!(
            matches!(block.insts[2].srcs.first(), Some(Operand::Imm(30))),
            "Expected Imm(30), got {:?}",
            block.insts[2].srcs.first()
        );
    }

    #[test]
    fn test_fold_mul() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 7));
        entry.push(MachineInst::li(v1, 6));
        entry.push(MachineInst::mul(v2, v0, v1)); // Should fold to li(v2, 42)
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let folded = fold_constants(&mut func).unwrap();
        assert_eq!(folded, 1);

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[2].opcode, Opcode::LI);
        assert!(
            matches!(block.insts[2].srcs.first(), Some(Operand::Imm(42))),
            "Expected Imm(42), got {:?}",
            block.insts[2].srcs.first()
        );
    }

    #[test]
    fn test_simplify_add_zero() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::li(v1, 0));
        entry.push(MachineInst::add(v2, v0, v1)); // Should simplify to mov(v2, v0)
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let simplified = simplify_algebra(&mut func).unwrap();
        assert_eq!(simplified, 1);

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[2].opcode, Opcode::MOV);
    }

    #[test]
    fn test_simplify_xor_self() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::xor(v1, v0, v0)); // Should simplify to li(v1, 0)
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let simplified = simplify_algebra(&mut func).unwrap();
        assert_eq!(simplified, 1);

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[1].opcode, Opcode::LI);
        if let Some(Operand::Imm(val)) = block.insts[1].srcs.first() {
            assert_eq!(*val, 0);
        }
    }
}
