//! Integer splitting pass for large integer support.
//!
//! When the target data width is insufficient for the source type (e.g., i64 on
//! 40-bit config), this pass splits operations into multiple register pairs.

use crate::mir::{MachineFunction, MachineInst, Opcode, Operand, VReg};
use crate::target::config::TargetConfig;
use anyhow::Result;
use std::collections::HashMap;

/// Split large integers that don't fit in a single register.
///
/// For i64 on 40-bit (2×20-bit limbs), splits into two 40-bit halves:
/// - lo: lower 32 bits (fits in 40-bit with headroom)
/// - hi: upper 32 bits
pub fn split_large_integers(func: &mut MachineFunction, config: &TargetConfig) -> Result<()> {
    // Check if splitting is needed
    if !config.needs_split(64) {
        return Ok(()); // i64 fits, no splitting needed
    }

    // Identify wide values and collect instructions to split
    let wide_vregs = identify_wide_values(func);

    if wide_vregs.is_empty() {
        return Ok(()); // No wide values to split
    }

    // Create split pairs map
    let mut split_map: HashMap<VReg, (VReg, VReg)> = HashMap::new();

    // Pre-allocate split pairs for all wide vregs
    for &vreg in &wide_vregs {
        let lo = func.new_vreg();
        let hi = func.new_vreg();
        split_map.insert(vreg, (lo, hi));
    }

    // Split instructions block by block
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        split_block(func, &label, &wide_vregs, &mut split_map)?;
    }

    Ok(())
}

/// Identify values that need splitting.
fn identify_wide_values(func: &MachineFunction) -> std::collections::HashSet<VReg> {
    let mut wide_vregs = std::collections::HashSet::new();

    for block in func.iter_blocks() {
        for inst in &block.insts {
            // Check for explicit 64-bit operations
            match inst.opcode {
                // 64-bit loads/stores
                Opcode::LD | Opcode::SD => {
                    if let Some(def) = inst.def() {
                        wide_vregs.insert(def);
                    }
                    for vreg in inst.uses() {
                        wide_vregs.insert(vreg);
                    }
                }
                // MUL can produce 64-bit result from 32-bit inputs
                Opcode::MUL => {
                    // Mark result as potentially wide
                    if let Some(def) = inst.def() {
                        wide_vregs.insert(def);
                    }
                }
                _ => {}
            }
        }
    }

    wide_vregs
}

/// Check if an instruction needs splitting.
fn needs_splitting(inst: &MachineInst, wide_vregs: &std::collections::HashSet<VReg>) -> bool {
    // Check if any operand is a wide value
    if let Some(def) = inst.def() {
        if wide_vregs.contains(&def) {
            return true;
        }
    }

    for vreg in inst.uses() {
        if wide_vregs.contains(&vreg) {
            return true;
        }
    }

    false
}

/// Get or create split pair for a vreg.
fn get_split_pair(
    vreg: VReg,
    func: &mut MachineFunction,
    split_map: &mut HashMap<VReg, (VReg, VReg)>,
) -> (VReg, VReg) {
    if let Some(&pair) = split_map.get(&vreg) {
        return pair;
    }

    let lo = func.new_vreg();
    let hi = func.new_vreg();
    split_map.insert(vreg, (lo, hi));
    (lo, hi)
}

/// Split instructions in a block.
fn split_block(
    func: &mut MachineFunction,
    label: &str,
    wide_vregs: &std::collections::HashSet<VReg>,
    split_map: &mut HashMap<VReg, (VReg, VReg)>,
) -> Result<()> {
    // Collect current instructions
    let current_insts: Vec<MachineInst> = match func.get_block(label) {
        Some(b) => b.insts.clone(),
        None => return Ok(()),
    };

    let mut new_insts = Vec::new();

    for inst in current_insts {
        if needs_splitting(&inst, wide_vregs) {
            let split_insts = split_instruction(&inst, func, split_map)?;
            new_insts.extend(split_insts);
        } else {
            new_insts.push(inst);
        }
    }

    // Update the block
    if let Some(block) = func.get_block_mut(label) {
        block.insts = new_insts;
    }

    Ok(())
}

/// Split a single instruction into multiple instructions.
fn split_instruction(
    inst: &MachineInst,
    func: &mut MachineFunction,
    split_map: &mut HashMap<VReg, (VReg, VReg)>,
) -> Result<Vec<MachineInst>> {
    match inst.opcode {
        Opcode::ADD => split_add(inst, func, split_map),
        Opcode::SUB => split_sub(inst, func, split_map),
        Opcode::MUL => split_mul(inst, func, split_map),
        Opcode::LD => split_load(inst, func, split_map),
        Opcode::SD => split_store(inst, func, split_map),
        Opcode::LI => split_li(inst, func, split_map),
        Opcode::MOV => split_mov(inst, func, split_map),
        _ => {
            // Non-splittable ops: keep as-is
            Ok(vec![inst.clone()])
        }
    }
}

/// Split ADD into add-with-carry sequence.
fn split_add(
    inst: &MachineInst,
    func: &mut MachineFunction,
    split_map: &mut HashMap<VReg, (VReg, VReg)>,
) -> Result<Vec<MachineInst>> {
    let mut result = Vec::new();

    let dst = match inst.def() {
        Some(d) => d,
        None => return Ok(vec![inst.clone()]),
    };
    let (dst_lo, dst_hi) = get_split_pair(dst, func, split_map);

    // Get source operands
    let uses = inst.uses();
    if uses.len() < 2 {
        return Ok(vec![inst.clone()]);
    }

    let (src1_lo, src1_hi) = get_split_pair(uses[0], func, split_map);
    let (src2_lo, src2_hi) = get_split_pair(uses[1], func, split_map);

    // dst_lo = src1_lo + src2_lo
    result.push(MachineInst::add(dst_lo, src1_lo, src2_lo));

    // carry = (dst_lo < src1_lo) ? 1 : 0  (using SLTU)
    let carry = func.new_vreg();
    result.push(
        MachineInst::new(Opcode::SLTU)
            .dst(Operand::VReg(carry))
            .src(Operand::VReg(dst_lo))
            .src(Operand::VReg(src1_lo)),
    );

    // tmp = src1_hi + src2_hi
    let tmp = func.new_vreg();
    result.push(MachineInst::add(tmp, src1_hi, src2_hi));

    // dst_hi = tmp + carry
    result.push(MachineInst::add(dst_hi, tmp, carry));

    Ok(result)
}

/// Split SUB into subtract-with-borrow sequence.
fn split_sub(
    inst: &MachineInst,
    func: &mut MachineFunction,
    split_map: &mut HashMap<VReg, (VReg, VReg)>,
) -> Result<Vec<MachineInst>> {
    let mut result = Vec::new();

    let dst = match inst.def() {
        Some(d) => d,
        None => return Ok(vec![inst.clone()]),
    };
    let (dst_lo, dst_hi) = get_split_pair(dst, func, split_map);

    let uses = inst.uses();
    if uses.len() < 2 {
        return Ok(vec![inst.clone()]);
    }

    let (src1_lo, src1_hi) = get_split_pair(uses[0], func, split_map);
    let (src2_lo, src2_hi) = get_split_pair(uses[1], func, split_map);

    // borrow = (src1_lo < src2_lo) ? 1 : 0
    let borrow = func.new_vreg();
    result.push(
        MachineInst::new(Opcode::SLTU)
            .dst(Operand::VReg(borrow))
            .src(Operand::VReg(src1_lo))
            .src(Operand::VReg(src2_lo)),
    );

    // dst_lo = src1_lo - src2_lo
    result.push(MachineInst::sub(dst_lo, src1_lo, src2_lo));

    // tmp = src1_hi - src2_hi
    let tmp = func.new_vreg();
    result.push(MachineInst::sub(tmp, src1_hi, src2_hi));

    // dst_hi = tmp - borrow
    result.push(MachineInst::sub(dst_hi, tmp, borrow));

    Ok(result)
}

/// Split MUL into multi-part multiplication.
///
/// For 64-bit multiply: (a_hi:a_lo) × (b_hi:b_lo)
/// = a_lo×b_lo + (a_lo×b_hi + a_hi×b_lo)<<32 + (a_hi×b_hi)<<64
///
/// For a simple case keeping only lower 64 bits:
/// result_lo = a_lo × b_lo (lower 32 bits)
/// result_hi = a_lo×b_lo>>32 + a_lo×b_hi + a_hi×b_lo (with carries)
fn split_mul(
    inst: &MachineInst,
    func: &mut MachineFunction,
    split_map: &mut HashMap<VReg, (VReg, VReg)>,
) -> Result<Vec<MachineInst>> {
    let mut result = Vec::new();

    let dst = match inst.def() {
        Some(d) => d,
        None => return Ok(vec![inst.clone()]),
    };
    let (dst_lo, dst_hi) = get_split_pair(dst, func, split_map);

    let uses = inst.uses();
    if uses.len() < 2 {
        return Ok(vec![inst.clone()]);
    }

    let (a_lo, a_hi) = get_split_pair(uses[0], func, split_map);
    let (b_lo, b_hi) = get_split_pair(uses[1], func, split_map);

    // Simplified 64-bit multiply keeping lower 64 bits:
    // result = (a_hi:a_lo) × (b_hi:b_lo) mod 2^64
    //
    // We compute:
    // p0 = a_lo × b_lo (full product, we need both halves)
    // p1 = a_lo × b_hi (only low half matters for high result)
    // p2 = a_hi × b_lo (only low half matters for high result)
    //
    // result_lo = p0 (low part)
    // result_hi = p0>>32 + p1 + p2 (simplified - ignoring upper carry)

    // p0 = a_lo × b_lo
    let p0 = func.new_vreg();
    result.push(MachineInst::mul(p0, a_lo, b_lo));

    // For a proper implementation, we'd need to extract the upper bits of p0.
    // Since we don't have MULHU, we'll use a shift-based approach.
    // p0_hi = p0 >> 32 (arithmetic right shift for upper bits)
    let p0_hi = func.new_vreg();
    result.push(
        MachineInst::new(Opcode::SRAI)
            .dst(Operand::VReg(p0_hi))
            .src(Operand::VReg(p0))
            .src(Operand::Imm(32)),
    );

    // p1 = a_lo × b_hi
    let p1 = func.new_vreg();
    result.push(MachineInst::mul(p1, a_lo, b_hi));

    // p2 = a_hi × b_lo
    let p2 = func.new_vreg();
    result.push(MachineInst::mul(p2, a_hi, b_lo));

    // dst_lo = p0 (truncated to lower bits)
    result.push(MachineInst::mov(dst_lo, p0));

    // dst_hi = p0_hi + p1 + p2
    let tmp1 = func.new_vreg();
    result.push(MachineInst::add(tmp1, p0_hi, p1));
    result.push(MachineInst::add(dst_hi, tmp1, p2));

    Ok(result)
}

/// Split 64-bit load into two 32-bit loads.
fn split_load(
    inst: &MachineInst,
    func: &mut MachineFunction,
    split_map: &mut HashMap<VReg, (VReg, VReg)>,
) -> Result<Vec<MachineInst>> {
    let mut result = Vec::new();

    let dst = match inst.def() {
        Some(d) => d,
        None => return Ok(vec![inst.clone()]),
    };
    let (dst_lo, dst_hi) = get_split_pair(dst, func, split_map);

    // Get the base address
    let (base, offset) = match inst.srcs.first() {
        Some(Operand::Mem { base, offset }) => (*base, *offset),
        Some(Operand::VReg(r)) => (*r, 0),
        _ => return Ok(vec![inst.clone()]),
    };

    // Load lower 32 bits (offset + 0)
    result.push(
        MachineInst::new(Opcode::LW)
            .dst(Operand::VReg(dst_lo))
            .src(Operand::Mem { base, offset }),
    );

    // Load upper 32 bits (offset + 4)
    result.push(
        MachineInst::new(Opcode::LW)
            .dst(Operand::VReg(dst_hi))
            .src(Operand::Mem { base, offset: offset + 4 }),
    );

    Ok(result)
}

/// Split 64-bit store into two 32-bit stores.
fn split_store(
    inst: &MachineInst,
    func: &mut MachineFunction,
    split_map: &mut HashMap<VReg, (VReg, VReg)>,
) -> Result<Vec<MachineInst>> {
    let mut result = Vec::new();

    // Get the value to store
    let value = match inst.srcs.first() {
        Some(Operand::VReg(r)) => *r,
        _ => return Ok(vec![inst.clone()]),
    };

    let (val_lo, val_hi) = get_split_pair(value, func, split_map);

    // Get the base address
    let (base, offset) = match inst.srcs.get(1) {
        Some(Operand::Mem { base, offset }) => (*base, *offset),
        Some(Operand::VReg(r)) => (*r, 0),
        _ => return Ok(vec![inst.clone()]),
    };

    // Store lower 32 bits
    result.push(
        MachineInst::new(Opcode::SW)
            .src(Operand::VReg(val_lo))
            .src(Operand::Mem { base, offset }),
    );

    // Store upper 32 bits
    result.push(
        MachineInst::new(Opcode::SW)
            .src(Operand::VReg(val_hi))
            .src(Operand::Mem { base, offset: offset + 4 }),
    );

    Ok(result)
}

/// Split LI for 64-bit immediate.
fn split_li(
    inst: &MachineInst,
    func: &mut MachineFunction,
    split_map: &mut HashMap<VReg, (VReg, VReg)>,
) -> Result<Vec<MachineInst>> {
    let mut result = Vec::new();

    let dst = match inst.def() {
        Some(d) => d,
        None => return Ok(vec![inst.clone()]),
    };
    let (dst_lo, dst_hi) = get_split_pair(dst, func, split_map);

    // Get the immediate value
    let imm = match inst.srcs.first() {
        Some(Operand::Imm(i)) => *i,
        _ => return Ok(vec![inst.clone()]),
    };

    // Split into lo (lower 32 bits) and hi (upper 32 bits)
    let lo = (imm as u64) & 0xFFFFFFFF;
    let hi = ((imm as u64) >> 32) & 0xFFFFFFFF;

    result.push(MachineInst::li(dst_lo, lo as i64));
    result.push(MachineInst::li(dst_hi, hi as i64));

    Ok(result)
}

/// Split MOV for 64-bit values.
fn split_mov(
    inst: &MachineInst,
    func: &mut MachineFunction,
    split_map: &mut HashMap<VReg, (VReg, VReg)>,
) -> Result<Vec<MachineInst>> {
    let mut result = Vec::new();

    let dst = match inst.def() {
        Some(d) => d,
        None => return Ok(vec![inst.clone()]),
    };
    let (dst_lo, dst_hi) = get_split_pair(dst, func, split_map);

    let src = match inst.srcs.first() {
        Some(Operand::VReg(r)) => *r,
        _ => return Ok(vec![inst.clone()]),
    };

    let (src_lo, src_hi) = get_split_pair(src, func, split_map);

    result.push(MachineInst::mov(dst_lo, src_lo));
    result.push(MachineInst::mov(dst_hi, src_hi));

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MachineBlock;

    #[test]
    fn test_no_split_on_80bit() {
        let mut func = MachineFunction::new("test");
        let config = TargetConfig::DATA_80; // 80-bit can hold i64

        let v0 = func.new_vreg();
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 0x123456789));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let before_count = func.get_block("entry").unwrap().insts.len();
        split_large_integers(&mut func, &config).unwrap();
        let after_count = func.get_block("entry").unwrap().insts.len();

        // No splitting should occur
        assert_eq!(before_count, after_count);
    }

    #[test]
    fn test_split_map_creation() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let mut split_map: HashMap<VReg, (VReg, VReg)> = HashMap::new();

        let (lo1, hi1) = get_split_pair(v0, &mut func, &mut split_map);
        let (lo2, hi2) = get_split_pair(v0, &mut func, &mut split_map);

        // Same vreg should return same split pair
        assert_eq!(lo1, lo2);
        assert_eq!(hi1, hi2);

        // Lo and hi should be different
        assert_ne!(lo1, hi1);
    }
}
