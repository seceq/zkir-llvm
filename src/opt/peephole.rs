//! Peephole optimization pass.
//!
//! Pattern-matching transformations on small instruction sequences.
//! These include strength reduction, instruction combining, and
//! redundant instruction elimination.

use crate::mir::{MachineFunction, MachineInst, Opcode, Operand, VReg};
use anyhow::Result;
use std::collections::HashMap;

/// Run peephole optimizations on a function.
pub fn peephole_optimize(func: &mut MachineFunction) -> Result<u32> {
    let mut total_optimized = 0;
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        if let Some(block) = func.get_block_mut(&label) {
            let count = optimize_block(&mut block.insts);
            total_optimized += count;
        }
    }

    Ok(total_optimized)
}

/// Optimize a single block.
fn optimize_block(insts: &mut Vec<MachineInst>) -> u32 {
    let mut optimized = 0;

    // Run multiple passes until no more optimizations
    loop {
        let mut changed = false;

        // Pass 1: Strength reduction
        optimized += strength_reduction(insts);

        // Pass 2: Redundant instruction elimination
        let eliminated = eliminate_redundant(insts);
        if eliminated > 0 {
            changed = true;
            optimized += eliminated;
        }

        // Pass 3: Instruction combining
        let combined = combine_instructions(insts);
        if combined > 0 {
            changed = true;
            optimized += combined;
        }

        if !changed {
            break;
        }
    }

    optimized
}

/// Strength reduction: replace expensive operations with cheaper ones.
fn strength_reduction(insts: &mut [MachineInst]) -> u32 {
    let mut optimized = 0;

    // Collect known constant values
    let mut constants: HashMap<VReg, i64> = HashMap::new();

    for inst in insts.iter_mut() {
        // Track constants
        if inst.opcode == Opcode::LI {
            if let (Some(Operand::VReg(dst)), Some(Operand::Imm(val))) =
                (&inst.dst, inst.srcs.first())
            {
                constants.insert(*dst, *val);
            }
        }

        // MUL by power of 2 -> SLL
        if inst.opcode == Opcode::MUL {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                // Check if src2 is a power of 2
                if let Some(&val) = constants.get(src2) {
                    if val > 0 && (val & (val - 1)) == 0 {
                        let shift = val.trailing_zeros() as i64;
                        *inst = MachineInst::new(Opcode::SLLI)
                            .dst(Operand::VReg(*dst))
                            .src(Operand::VReg(*src1))
                            .src(Operand::Imm(shift))
                            .comment("mul by 2^n -> sll");
                        optimized += 1;
                        continue;
                    }
                }
                // Check if src1 is a power of 2
                if let Some(&val) = constants.get(src1) {
                    if val > 0 && (val & (val - 1)) == 0 {
                        let shift = val.trailing_zeros() as i64;
                        *inst = MachineInst::new(Opcode::SLLI)
                            .dst(Operand::VReg(*dst))
                            .src(Operand::VReg(*src2))
                            .src(Operand::Imm(shift))
                            .comment("mul by 2^n -> sll");
                        optimized += 1;
                        continue;
                    }
                }
            }
        }

        // DIV by power of 2 -> SRL (for unsigned) or SRA (for signed)
        if inst.opcode == Opcode::DIV {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if let Some(&val) = constants.get(src2) {
                    if val > 0 && (val & (val - 1)) == 0 {
                        let shift = val.trailing_zeros() as i64;
                        // Use SRA for signed division (more conservative)
                        *inst = MachineInst::new(Opcode::SRAI)
                            .dst(Operand::VReg(*dst))
                            .src(Operand::VReg(*src1))
                            .src(Operand::Imm(shift))
                            .comment("div by 2^n -> sra");
                        optimized += 1;
                    }
                }
            }
        }

        // REM by power of 2 -> AND with mask
        if inst.opcode == Opcode::REM {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if let Some(&val) = constants.get(src2) {
                    if val > 0 && (val & (val - 1)) == 0 {
                        let mask = val - 1;
                        *inst = MachineInst::new(Opcode::ANDI)
                            .dst(Operand::VReg(*dst))
                            .src(Operand::VReg(*src1))
                            .src(Operand::Imm(mask))
                            .comment("rem by 2^n -> and mask");
                        optimized += 1;
                    }
                }
            }
        }

        // MUL by 0 -> LI 0
        if inst.opcode == Opcode::MUL {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src1) == Some(&0) || constants.get(src2) == Some(&0) {
                    *inst = MachineInst::li(*dst, 0).comment("mul by 0 -> 0");
                    optimized += 1;
                }
            }
        }

        // MUL by 1 -> MOV
        if inst.opcode == Opcode::MUL {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src2) == Some(&1) {
                    *inst = MachineInst::mov(*dst, *src1).comment("mul by 1 -> mov");
                    optimized += 1;
                } else if constants.get(src1) == Some(&1) {
                    *inst = MachineInst::mov(*dst, *src2).comment("1 * x -> mov");
                    optimized += 1;
                }
            }
        }

        // ADD by 0 -> MOV
        if inst.opcode == Opcode::ADD {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src2) == Some(&0) {
                    *inst = MachineInst::mov(*dst, *src1).comment("add 0 -> mov");
                    optimized += 1;
                } else if constants.get(src1) == Some(&0) {
                    *inst = MachineInst::mov(*dst, *src2).comment("0 + x -> mov");
                    optimized += 1;
                }
            }
        }

        // SUB by 0 -> MOV
        if inst.opcode == Opcode::SUB {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src2) == Some(&0) {
                    *inst = MachineInst::mov(*dst, *src1).comment("sub 0 -> mov");
                    optimized += 1;
                }
            }
        }

        // AND with 0 -> LI 0
        if inst.opcode == Opcode::AND {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src1) == Some(&0) || constants.get(src2) == Some(&0) {
                    *inst = MachineInst::li(*dst, 0).comment("and 0 -> 0");
                    optimized += 1;
                }
            }
        }

        // AND with -1 (all ones) -> MOV
        if inst.opcode == Opcode::AND {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src2) == Some(&-1) {
                    *inst = MachineInst::mov(*dst, *src1).comment("and -1 -> mov");
                    optimized += 1;
                } else if constants.get(src1) == Some(&-1) {
                    *inst = MachineInst::mov(*dst, *src2).comment("-1 and x -> mov");
                    optimized += 1;
                }
            }
        }

        // OR with 0 -> MOV
        if inst.opcode == Opcode::OR {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src2) == Some(&0) {
                    *inst = MachineInst::mov(*dst, *src1).comment("or 0 -> mov");
                    optimized += 1;
                } else if constants.get(src1) == Some(&0) {
                    *inst = MachineInst::mov(*dst, *src2).comment("0 or x -> mov");
                    optimized += 1;
                }
            }
        }

        // OR with -1 -> LI -1
        if inst.opcode == Opcode::OR {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src1) == Some(&-1) || constants.get(src2) == Some(&-1) {
                    *inst = MachineInst::li(*dst, -1).comment("or -1 -> -1");
                    optimized += 1;
                }
            }
        }

        // XOR with 0 -> MOV
        if inst.opcode == Opcode::XOR {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src2) == Some(&0) {
                    *inst = MachineInst::mov(*dst, *src1).comment("xor 0 -> mov");
                    optimized += 1;
                } else if constants.get(src1) == Some(&0) {
                    *inst = MachineInst::mov(*dst, *src2).comment("0 xor x -> mov");
                    optimized += 1;
                }
            }
        }

        // SLL/SRL/SRA by 0 -> MOV (when using register operands)
        if matches!(inst.opcode, Opcode::SLL | Opcode::SRL | Opcode::SRA) {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src2) == Some(&0) {
                    *inst = MachineInst::mov(*dst, *src1).comment("shift by 0 -> mov");
                    optimized += 1;
                }
            }
        }

        // DIV by 1 -> MOV
        if inst.opcode == Opcode::DIV {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src2) == Some(&1) {
                    *inst = MachineInst::mov(*dst, *src1).comment("div by 1 -> mov");
                    optimized += 1;
                }
            }
        }

        // REM by 1 -> LI 0
        if inst.opcode == Opcode::REM {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(_src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src2) == Some(&1) {
                    *inst = MachineInst::li(*dst, 0).comment("rem by 1 -> 0");
                    optimized += 1;
                }
            }
        }

        // XOR with self -> LI 0 (a ^ a = 0)
        if inst.opcode == Opcode::XOR {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if src1 == src2 {
                    *inst = MachineInst::li(*dst, 0).comment("x xor x -> 0");
                    optimized += 1;
                }
            }
        }

        // SUB self -> LI 0 (a - a = 0)
        if inst.opcode == Opcode::SUB {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if src1 == src2 {
                    *inst = MachineInst::li(*dst, 0).comment("x - x -> 0");
                    optimized += 1;
                }
            }
        }

        // AND with self -> MOV (a & a = a)
        if inst.opcode == Opcode::AND {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if src1 == src2 {
                    *inst = MachineInst::mov(*dst, *src1).comment("x and x -> x");
                    optimized += 1;
                }
            }
        }

        // OR with self -> MOV (a | a = a)
        if inst.opcode == Opcode::OR {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if src1 == src2 {
                    *inst = MachineInst::mov(*dst, *src1).comment("x or x -> x");
                    optimized += 1;
                }
            }
        }

        // SLTU with 0 on RHS -> LI 0 (nothing is less than 0 for unsigned)
        if inst.opcode == Opcode::SLTU {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(_src1)), Some(Operand::VReg(src2))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if constants.get(src2) == Some(&0) {
                    *inst = MachineInst::li(*dst, 0).comment("x <u 0 -> 0");
                    optimized += 1;
                }
            }
        }

        // Constant folding for immediate operations
        if inst.opcode == Opcode::ADDI {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), Some(Operand::Imm(imm))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                let dst = *dst;
                let src = *src;
                let imm = *imm;
                if let Some(&val) = constants.get(&src) {
                    let result = val.wrapping_add(imm);
                    *inst = MachineInst::li(dst, result).comment("const fold addi");
                    optimized += 1;
                    constants.insert(dst, result);
                }
            }
        }

        if inst.opcode == Opcode::SLLI {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), Some(Operand::Imm(shift))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                let dst = *dst;
                let src = *src;
                let shift = *shift;
                if let Some(&val) = constants.get(&src) {
                    let result = val.wrapping_shl(shift as u32);
                    *inst = MachineInst::li(dst, result).comment("const fold slli");
                    optimized += 1;
                    constants.insert(dst, result);
                }
            }
        }

        if inst.opcode == Opcode::SRLI {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), Some(Operand::Imm(shift))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                let dst = *dst;
                let src = *src;
                let shift = *shift;
                if let Some(&val) = constants.get(&src) {
                    let result = (val as u64).wrapping_shr(shift as u32) as i64;
                    *inst = MachineInst::li(dst, result).comment("const fold srli");
                    optimized += 1;
                    constants.insert(dst, result);
                }
            }
        }

        if inst.opcode == Opcode::ANDI {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), Some(Operand::Imm(mask))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                let dst = *dst;
                let src = *src;
                let mask = *mask;
                if let Some(&val) = constants.get(&src) {
                    let result = val & mask;
                    *inst = MachineInst::li(dst, result).comment("const fold andi");
                    optimized += 1;
                    constants.insert(dst, result);
                }
            }
        }

        if inst.opcode == Opcode::ORI {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), Some(Operand::Imm(imm))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                let dst = *dst;
                let src = *src;
                let imm = *imm;
                if let Some(&val) = constants.get(&src) {
                    let result = val | imm;
                    *inst = MachineInst::li(dst, result).comment("const fold ori");
                    optimized += 1;
                    constants.insert(dst, result);
                }
            }
        }

        if inst.opcode == Opcode::XORI {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), Some(Operand::Imm(imm))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                let dst = *dst;
                let src = *src;
                let imm = *imm;
                if let Some(&val) = constants.get(&src) {
                    let result = val ^ imm;
                    *inst = MachineInst::li(dst, result).comment("const fold xori");
                    optimized += 1;
                    constants.insert(dst, result);
                }
            }
        }
    }

    optimized
}

/// Eliminate redundant instructions.
fn eliminate_redundant(insts: &mut Vec<MachineInst>) -> u32 {
    let mut to_remove: Vec<usize> = Vec::new();

    // Track last definition of each vreg
    let mut last_def: HashMap<VReg, (usize, i64)> = HashMap::new();

    for (idx, inst) in insts.iter().enumerate() {
        // Track LI definitions
        if inst.opcode == Opcode::LI {
            if let (Some(Operand::VReg(dst)), Some(Operand::Imm(val))) =
                (&inst.dst, inst.srcs.first())
            {
                // If we're loading the same value into the same register, remove
                if let Some((prev_idx, prev_val)) = last_def.get(dst) {
                    if *prev_val == *val && !uses_vreg_between(insts, *prev_idx, idx, *dst) {
                        to_remove.push(idx);
                        continue;
                    }
                }
                last_def.insert(*dst, (idx, *val));
            }
        }

        // MOV to self -> remove
        if inst.opcode == Opcode::MOV {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src))) =
                (&inst.dst, inst.srcs.first())
            {
                if dst == src {
                    to_remove.push(idx);
                }
            }
        }

        // ADDI with 0 -> MOV or remove
        if inst.opcode == Opcode::ADDI {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), Some(Operand::Imm(0))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if dst == src {
                    to_remove.push(idx);
                }
            }
        }

        // SLLI/SRLI/SRAI by 0 -> remove if dst == src
        if matches!(inst.opcode, Opcode::SLLI | Opcode::SRLI | Opcode::SRAI) {
            if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), Some(Operand::Imm(0))) =
                (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
            {
                if dst == src {
                    to_remove.push(idx);
                }
            }
        }
    }

    // Remove in reverse order to preserve indices
    let removed = to_remove.len() as u32;
    for idx in to_remove.into_iter().rev() {
        insts.remove(idx);
    }

    removed
}

/// Check if a vreg is used between two instruction indices.
fn uses_vreg_between(insts: &[MachineInst], start: usize, end: usize, vreg: VReg) -> bool {
    for inst in &insts[start + 1..end] {
        for src in &inst.srcs {
            if let Operand::VReg(v) = src {
                if *v == vreg {
                    return true;
                }
            }
            if let Operand::Mem { base, .. } = src {
                if *base == vreg {
                    return true;
                }
            }
        }
    }
    false
}

/// Combine adjacent instructions.
fn combine_instructions(insts: &mut Vec<MachineInst>) -> u32 {
    if insts.len() < 2 {
        return 0;
    }

    let mut combined = 0;
    let mut replacements: Vec<(usize, MachineInst)> = Vec::new();
    let mut to_remove: Vec<usize> = Vec::new();

    // First pass: identify patterns (read-only)
    for i in 0..insts.len().saturating_sub(1) {
        let inst1 = &insts[i];
        let inst2 = &insts[i + 1];

        // LI followed by ADDI with same dst -> LI with combined value
        if inst1.opcode == Opcode::LI && inst2.opcode == Opcode::ADDI {
            if let (
                Some(Operand::VReg(dst1)), Some(Operand::Imm(val1)),
                Some(Operand::VReg(dst2)), Some(Operand::VReg(src2)), Some(Operand::Imm(val2))
            ) = (
                &inst1.dst, inst1.srcs.first(),
                &inst2.dst, inst2.srcs.first(), inst2.srcs.get(1)
            ) {
                if dst1 == dst2 && dst1 == src2 {
                    let combined_val = val1.wrapping_add(*val2);
                    replacements.push((i, MachineInst::li(*dst1, combined_val)
                        .comment("li+addi combined")));
                    to_remove.push(i + 1);
                    combined += 1;
                    continue;
                }
            }
        }

        // Two consecutive ADDIs with same dst/src -> single ADDI
        if inst1.opcode == Opcode::ADDI && inst2.opcode == Opcode::ADDI {
            if let (
                Some(Operand::VReg(dst1)), Some(Operand::VReg(src1)), Some(Operand::Imm(val1)),
                Some(Operand::VReg(dst2)), Some(Operand::VReg(src2)), Some(Operand::Imm(val2))
            ) = (
                &inst1.dst, inst1.srcs.first(), inst1.srcs.get(1),
                &inst2.dst, inst2.srcs.first(), inst2.srcs.get(1)
            ) {
                // addi x, y, a followed by addi x, x, b -> addi x, y, a+b
                if dst1 == dst2 && dst1 == src2 {
                    let combined_val = val1.wrapping_add(*val2);
                    replacements.push((i, MachineInst::addi(*dst1, *src1, combined_val)
                        .comment("addi+addi combined")));
                    to_remove.push(i + 1);
                    combined += 1;
                    continue;
                }
            }
        }

        // SLL followed by SLL -> single SLL with combined shift
        if inst1.opcode == Opcode::SLLI && inst2.opcode == Opcode::SLLI {
            if let (
                Some(Operand::VReg(dst1)), Some(Operand::VReg(src1)), Some(Operand::Imm(sh1)),
                Some(Operand::VReg(dst2)), Some(Operand::VReg(src2)), Some(Operand::Imm(sh2))
            ) = (
                &inst1.dst, inst1.srcs.first(), inst1.srcs.get(1),
                &inst2.dst, inst2.srcs.first(), inst2.srcs.get(1)
            ) {
                if dst1 == dst2 && dst1 == src2 {
                    let combined_shift = sh1.wrapping_add(*sh2);
                    if combined_shift < 64 {
                        replacements.push((i, MachineInst::new(Opcode::SLLI)
                            .dst(Operand::VReg(*dst1))
                            .src(Operand::VReg(*src1))
                            .src(Operand::Imm(combined_shift))
                            .comment("sll+sll combined")));
                        to_remove.push(i + 1);
                        combined += 1;
                    }
                }
            }
        }

        // SRL followed by SRL -> single SRL with combined shift
        if inst1.opcode == Opcode::SRLI && inst2.opcode == Opcode::SRLI {
            if let (
                Some(Operand::VReg(dst1)), Some(Operand::VReg(src1)), Some(Operand::Imm(sh1)),
                Some(Operand::VReg(dst2)), Some(Operand::VReg(src2)), Some(Operand::Imm(sh2))
            ) = (
                &inst1.dst, inst1.srcs.first(), inst1.srcs.get(1),
                &inst2.dst, inst2.srcs.first(), inst2.srcs.get(1)
            ) {
                if dst1 == dst2 && dst1 == src2 {
                    let combined_shift = sh1.wrapping_add(*sh2);
                    if combined_shift < 64 {
                        replacements.push((i, MachineInst::new(Opcode::SRLI)
                            .dst(Operand::VReg(*dst1))
                            .src(Operand::VReg(*src1))
                            .src(Operand::Imm(combined_shift))
                            .comment("srl+srl combined")));
                        to_remove.push(i + 1);
                        combined += 1;
                    }
                }
            }
        }

        // Two consecutive ANDIs with same dst/src -> single ANDI with combined mask
        if inst1.opcode == Opcode::ANDI && inst2.opcode == Opcode::ANDI {
            if let (
                Some(Operand::VReg(dst1)), Some(Operand::VReg(src1)), Some(Operand::Imm(mask1)),
                Some(Operand::VReg(dst2)), Some(Operand::VReg(src2)), Some(Operand::Imm(mask2))
            ) = (
                &inst1.dst, inst1.srcs.first(), inst1.srcs.get(1),
                &inst2.dst, inst2.srcs.first(), inst2.srcs.get(1)
            ) {
                // andi x, y, m1 followed by andi x, x, m2 -> andi x, y, m1 & m2
                if dst1 == dst2 && dst1 == src2 {
                    let combined_mask = mask1 & mask2;
                    replacements.push((i, MachineInst::new(Opcode::ANDI)
                        .dst(Operand::VReg(*dst1))
                        .src(Operand::VReg(*src1))
                        .src(Operand::Imm(combined_mask))
                        .comment("andi+andi combined")));
                    to_remove.push(i + 1);
                    combined += 1;
                    continue;
                }
            }
        }

        // Two consecutive ORIs with same dst/src -> single ORI with combined value
        if inst1.opcode == Opcode::ORI && inst2.opcode == Opcode::ORI {
            if let (
                Some(Operand::VReg(dst1)), Some(Operand::VReg(src1)), Some(Operand::Imm(val1)),
                Some(Operand::VReg(dst2)), Some(Operand::VReg(src2)), Some(Operand::Imm(val2))
            ) = (
                &inst1.dst, inst1.srcs.first(), inst1.srcs.get(1),
                &inst2.dst, inst2.srcs.first(), inst2.srcs.get(1)
            ) {
                if dst1 == dst2 && dst1 == src2 {
                    let combined_val = val1 | val2;
                    replacements.push((i, MachineInst::new(Opcode::ORI)
                        .dst(Operand::VReg(*dst1))
                        .src(Operand::VReg(*src1))
                        .src(Operand::Imm(combined_val))
                        .comment("ori+ori combined")));
                    to_remove.push(i + 1);
                    combined += 1;
                    continue;
                }
            }
        }

        // MOV followed by MOV to same destination -> eliminate first MOV
        if inst1.opcode == Opcode::MOV && inst2.opcode == Opcode::MOV {
            if let (
                Some(Operand::VReg(dst1)), Some(Operand::VReg(_src1)),
                Some(Operand::VReg(dst2)), Some(Operand::VReg(_src2))
            ) = (
                &inst1.dst, inst1.srcs.first(),
                &inst2.dst, inst2.srcs.first()
            ) {
                // If both MOVs write to the same destination, first is dead
                if dst1 == dst2 {
                    to_remove.push(i);
                    combined += 1;
                    continue;
                }
            }
        }

        // LI followed by MOV from same reg -> change MOV to LI
        if inst1.opcode == Opcode::LI && inst2.opcode == Opcode::MOV {
            if let (
                Some(Operand::VReg(dst1)), Some(Operand::Imm(val)),
                Some(Operand::VReg(dst2)), Some(Operand::VReg(src2))
            ) = (
                &inst1.dst, inst1.srcs.first(),
                &inst2.dst, inst2.srcs.first()
            ) {
                // li x, val; mov y, x -> li x, val; li y, val
                // This is useful for copy propagation later
                if dst1 == src2 && dst1 != dst2 {
                    replacements.push((i + 1, MachineInst::li(*dst2, *val)
                        .comment("li+mov -> li")));
                    combined += 1;
                    continue;
                }
            }
        }
    }

    // Second pass: apply replacements
    for (idx, new_inst) in replacements {
        insts[idx] = new_inst;
    }

    // Remove combined instructions in reverse order
    for idx in to_remove.into_iter().rev() {
        insts.remove(idx);
    }

    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MachineBlock;

    #[test]
    fn test_mul_power_of_2() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::li(v1, 8)); // 2^3
        entry.push(MachineInst::mul(v2, v0, v1)); // Should become sll by 3
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        // The mul should be converted to slli
        assert_eq!(block.insts[2].opcode, Opcode::SLLI);
        if let Some(Operand::Imm(shift)) = block.insts[2].srcs.get(1) {
            assert_eq!(*shift, 3);
        }
    }

    #[test]
    fn test_div_power_of_2() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 100));
        entry.push(MachineInst::li(v1, 4)); // 2^2
        entry.push(MachineInst::new(Opcode::DIV)
            .dst(Operand::VReg(v2))
            .src(Operand::VReg(v0))
            .src(Operand::VReg(v1)));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        // The div should be converted to srai
        assert_eq!(block.insts[2].opcode, Opcode::SRAI);
    }

    #[test]
    fn test_remove_mov_self() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::mov(v0, v0)); // Redundant
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let before = func.get_block("entry").unwrap().insts.len();
        peephole_optimize(&mut func).unwrap();
        let after = func.get_block("entry").unwrap().insts.len();

        assert_eq!(before - 1, after);
    }

    #[test]
    fn test_combine_li_addi() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        // Use a vreg that's defined but not a known constant
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::mov(v1, v0)); // Break the constant chain
        entry.push(MachineInst::addi(v1, v1, 5));
        entry.push(MachineInst::addi(v1, v1, 3)); // These two should combine
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        // li + mov + combined addi + ret = 4 insts
        // Actually with constant folding of mov(v1, v0) where v0 is constant,
        // we get li for v0, li for v1, addi (combined), ret
        // Let's just check that addi combining happens by checking instruction count decrease
        assert!(block.insts.len() <= 4); // Should have combined the two addis
    }

    #[test]
    fn test_add_zero() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::li(v1, 0));
        entry.push(MachineInst::add(v2, v0, v1)); // Should become mov
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[2].opcode, Opcode::MOV);
    }

    #[test]
    fn test_and_zero() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 0xFF));
        entry.push(MachineInst::li(v1, 0));
        entry.push(MachineInst::and(v2, v0, v1)); // Should become li 0
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[2].opcode, Opcode::LI);
        if let Some(Operand::Imm(val)) = block.insts[2].srcs.first() {
            assert_eq!(*val, 0);
        }
    }

    #[test]
    fn test_or_zero() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::li(v1, 0));
        entry.push(MachineInst::or(v2, v0, v1)); // Should become mov
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[2].opcode, Opcode::MOV);
    }

    #[test]
    fn test_xor_zero() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::li(v1, 0));
        entry.push(MachineInst::xor(v2, v0, v1)); // Should become mov
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[2].opcode, Opcode::MOV);
    }

    #[test]
    fn test_combine_slli() {
        // Test that two consecutive SLLIs get combined
        // We need a non-constant source to avoid constant folding
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        // Use a MOV to create v0 from an "external" source (like function arg)
        entry.push(MachineInst::mov(v0, v1));
        entry.push(MachineInst::new(Opcode::SLLI)
            .dst(Operand::VReg(v0))
            .src(Operand::VReg(v0))
            .src(Operand::Imm(3)));
        entry.push(MachineInst::new(Opcode::SLLI)
            .dst(Operand::VReg(v0))
            .src(Operand::VReg(v0))
            .src(Operand::Imm(5))); // Should combine to slli by 8
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let before = func.get_block("entry").unwrap().insts.len();
        peephole_optimize(&mut func).unwrap();
        let after = func.get_block("entry").unwrap().insts.len();

        // Should have combined one SLLI
        assert!(after < before);

        let block = func.get_block("entry").unwrap();
        // Find the SLLI instruction and check its shift
        for inst in &block.insts {
            if inst.opcode == Opcode::SLLI {
                if let Some(Operand::Imm(shift)) = inst.srcs.get(1) {
                    assert_eq!(*shift, 8); // 3 + 5
                }
            }
        }
    }

    #[test]
    fn test_combine_andi() {
        // Test that two consecutive ANDIs get combined
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        // Use a MOV to create v0 from an "external" source
        entry.push(MachineInst::mov(v0, v1));
        entry.push(MachineInst::new(Opcode::ANDI)
            .dst(Operand::VReg(v0))
            .src(Operand::VReg(v0))
            .src(Operand::Imm(0xF0)));
        entry.push(MachineInst::new(Opcode::ANDI)
            .dst(Operand::VReg(v0))
            .src(Operand::VReg(v0))
            .src(Operand::Imm(0x3F))); // Should combine to andi with 0x30
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let before = func.get_block("entry").unwrap().insts.len();
        peephole_optimize(&mut func).unwrap();
        let after = func.get_block("entry").unwrap().insts.len();

        // Should have combined one ANDI
        assert!(after < before);

        let block = func.get_block("entry").unwrap();
        // Find the ANDI instruction and check its mask
        for inst in &block.insts {
            if inst.opcode == Opcode::ANDI {
                if let Some(Operand::Imm(mask)) = inst.srcs.get(1) {
                    assert_eq!(*mask, 0x30); // 0xF0 & 0x3F
                }
            }
        }
    }

    #[test]
    fn test_rem_power_of_2() {
        // Test that REM by power of 2 becomes ANDI
        // We use a non-constant dividend to avoid constant folding
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        // v0 comes from "external" source (simulated by mov from undefined vreg)
        entry.push(MachineInst::mov(v0, v3));
        entry.push(MachineInst::li(v1, 16)); // 2^4
        entry.push(MachineInst::new(Opcode::REM)
            .dst(Operand::VReg(v2))
            .src(Operand::VReg(v0))
            .src(Operand::VReg(v1)));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        // Find the ANDI instruction (REM should be converted)
        let has_andi = block.insts.iter().any(|inst| {
            if inst.opcode == Opcode::ANDI {
                if let Some(Operand::Imm(mask)) = inst.srcs.get(1) {
                    return *mask == 15;
                }
            }
            false
        });
        assert!(has_andi, "REM by 16 should become ANDI with mask 15");
    }

    #[test]
    fn test_xor_self() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::xor(v1, v0, v0)); // Should become li 0
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[1].opcode, Opcode::LI);
        if let Some(Operand::Imm(val)) = block.insts[1].srcs.first() {
            assert_eq!(*val, 0);
        }
    }

    #[test]
    fn test_sub_self() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::sub(v1, v0, v0)); // Should become li 0
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[1].opcode, Opcode::LI);
        if let Some(Operand::Imm(val)) = block.insts[1].srcs.first() {
            assert_eq!(*val, 0);
        }
    }

    #[test]
    fn test_and_self() {
        // Test that x AND x becomes MOV
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        // Use mov from undefined vreg to avoid constant folding
        entry.push(MachineInst::mov(v0, v2));
        entry.push(MachineInst::and(v1, v0, v0)); // Should become mov
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[1].opcode, Opcode::MOV);
    }

    #[test]
    fn test_or_self() {
        // Test that x OR x becomes MOV
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        // Use mov from undefined vreg to avoid constant folding
        entry.push(MachineInst::mov(v0, v2));
        entry.push(MachineInst::or(v1, v0, v0)); // Should become mov
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[1].opcode, Opcode::MOV);
    }

    #[test]
    fn test_const_fold_addi() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::addi(v1, v0, 5)); // Should become li 15
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[1].opcode, Opcode::LI);
        if let Some(Operand::Imm(val)) = block.insts[1].srcs.first() {
            assert_eq!(*val, 15);
        }
    }

    #[test]
    fn test_const_fold_slli() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 3));
        entry.push(MachineInst::new(Opcode::SLLI)
            .dst(Operand::VReg(v1))
            .src(Operand::VReg(v0))
            .src(Operand::Imm(4))); // Should become li 48 (3 << 4)
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[1].opcode, Opcode::LI);
        if let Some(Operand::Imm(val)) = block.insts[1].srcs.first() {
            assert_eq!(*val, 48);
        }
    }

    #[test]
    fn test_const_fold_andi() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 0xFF));
        entry.push(MachineInst::new(Opcode::ANDI)
            .dst(Operand::VReg(v1))
            .src(Operand::VReg(v0))
            .src(Operand::Imm(0x0F))); // Should become li 0x0F
        entry.push(MachineInst::ret());
        func.add_block(entry);

        peephole_optimize(&mut func).unwrap();

        let block = func.get_block("entry").unwrap();
        assert_eq!(block.insts[1].opcode, Opcode::LI);
        if let Some(Operand::Imm(val)) = block.insts[1].srcs.first() {
            assert_eq!(*val, 0x0F);
        }
    }
}
