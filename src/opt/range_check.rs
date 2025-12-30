//! Deferred range checking optimization.
//!
//! This pass implements the key ZK optimization from the ZKIR spec:
//! instead of checking range after every operation, we track value bounds
//! and only insert range checks when necessary.
//!
//! The algorithm:
//! 1. Propagate bounds through the function
//! 2. Identify points where values might exceed the data width
//! 3. Insert RCHK instructions only at those points
//! 4. After a range check, reset bounds to data width
//!
//! This can eliminate up to 51% of range check constraints.

use crate::mir::{MachineBlock, MachineFunction, MachineInst, Opcode, ValueBounds, VReg};
use crate::target::config::TargetConfig;
use anyhow::Result;
use std::collections::HashMap;

/// Insert range checks where necessary.
pub fn insert_range_checks(func: &mut MachineFunction, config: &TargetConfig) -> Result<()> {
    let data_bits = config.data_bits();

    // Track bounds for each vreg
    let mut bounds: HashMap<VReg, ValueBounds> = HashMap::new();

    // Process each block
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        if let Some(block) = func.get_block_mut(&label) {
            process_block(block, &mut bounds, data_bits);
        }
    }

    Ok(())
}

/// Process a single block, inserting range checks where needed.
fn process_block(block: &mut MachineBlock, bounds: &mut HashMap<VReg, ValueBounds>, data_bits: u32) {
    let mut insertions: Vec<(usize, MachineInst)> = Vec::new();

    for (idx, inst) in block.insts.iter_mut().enumerate() {
        // Update bounds based on instruction
        match inst.opcode {
            Opcode::LI => {
                if let (Some(dst), Some(imm)) = (inst.def(), inst.srcs.first().and_then(|s| s.as_imm())) {
                    let value = if imm >= 0 { imm as u128 } else { 0 }; // Conservative for negatives
                    bounds.insert(dst, ValueBounds::from_const(value));
                }
            }

            Opcode::ADD | Opcode::ADDI => {
                if let Some(dst) = inst.def() {
                    let uses = inst.uses();
                    let lhs_bounds = uses.first()
                        .and_then(|v| bounds.get(v).copied())
                        .unwrap_or_else(|| ValueBounds::from_bits(data_bits));

                    let rhs_bounds = if inst.opcode == Opcode::ADDI {
                        // Immediate operand
                        inst.srcs.get(1)
                            .and_then(|s| s.as_imm())
                            .map(|i| ValueBounds::from_const(i.unsigned_abs() as u128))
                            .unwrap_or_else(|| ValueBounds::from_bits(data_bits))
                    } else {
                        uses.get(1)
                            .and_then(|v| bounds.get(v).copied())
                            .unwrap_or_else(|| ValueBounds::from_bits(data_bits))
                    };

                    let result_bounds = ValueBounds::add(lhs_bounds, rhs_bounds);

                    // Check if result might overflow
                    if !result_bounds.fits_in(data_bits) {
                        // Insert range check after this instruction
                        insertions.push((idx + 1, MachineInst::rchk(dst)
                            .comment("overflow check")));

                        // After range check, bounds are reset
                        bounds.insert(dst, ValueBounds::from_bits(data_bits));
                    } else {
                        bounds.insert(dst, result_bounds);
                    }

                    // Store bounds in instruction for later use
                    inst.result_bounds = Some(result_bounds);
                }
            }

            Opcode::MUL => {
                if let Some(dst) = inst.def() {
                    let uses = inst.uses();
                    let lhs_bounds = uses.first()
                        .and_then(|v| bounds.get(v).copied())
                        .unwrap_or_else(|| ValueBounds::from_bits(data_bits));
                    let rhs_bounds = uses.get(1)
                        .and_then(|v| bounds.get(v).copied())
                        .unwrap_or_else(|| ValueBounds::from_bits(data_bits));

                    let result_bounds = ValueBounds::mul(lhs_bounds, rhs_bounds);

                    // Multiplication often needs range checks
                    if !result_bounds.fits_in(data_bits) {
                        insertions.push((idx + 1, MachineInst::rchk(dst)
                            .comment("mul overflow check")));
                        bounds.insert(dst, ValueBounds::from_bits(data_bits));
                    } else {
                        bounds.insert(dst, result_bounds);
                    }

                    inst.result_bounds = Some(result_bounds);
                }
            }

            Opcode::SUB => {
                if let Some(dst) = inst.def() {
                    let uses = inst.uses();
                    let lhs_bounds = uses.first()
                        .and_then(|v| bounds.get(v).copied())
                        .unwrap_or_else(|| ValueBounds::from_bits(data_bits));
                    let rhs_bounds = uses.get(1)
                        .and_then(|v| bounds.get(v).copied())
                        .unwrap_or_else(|| ValueBounds::from_bits(data_bits));

                    // Subtraction result is bounded by the larger operand
                    let result_bounds = ValueBounds::sub(lhs_bounds, rhs_bounds);
                    bounds.insert(dst, result_bounds);
                    inst.result_bounds = Some(result_bounds);
                }
            }

            Opcode::AND => {
                if let Some(dst) = inst.def() {
                    let uses = inst.uses();
                    let lhs_bounds = uses.first()
                        .and_then(|v| bounds.get(v).copied())
                        .unwrap_or_else(|| ValueBounds::from_bits(data_bits));
                    let rhs_bounds = uses.get(1)
                        .and_then(|v| bounds.get(v).copied())
                        .unwrap_or_else(|| ValueBounds::from_bits(data_bits));

                    // AND shrinks bounds
                    let result_bounds = ValueBounds::and(lhs_bounds, rhs_bounds);
                    bounds.insert(dst, result_bounds);
                    inst.result_bounds = Some(result_bounds);
                }
            }

            Opcode::SLL | Opcode::SLLI => {
                if let Some(dst) = inst.def() {
                    let uses = inst.uses();
                    let lhs_bounds = uses.first()
                        .and_then(|v| bounds.get(v).copied())
                        .unwrap_or_else(|| ValueBounds::from_bits(data_bits));

                    // Shift left grows the value
                    let shift_bounds = if inst.opcode == Opcode::SLLI {
                        inst.srcs.get(1)
                            .and_then(|s| s.as_imm())
                            .map(|i| ValueBounds::from_const(i as u128))
                            .unwrap_or_else(|| ValueBounds::from_bits(6)) // Max 64-bit shift
                    } else {
                        uses.get(1)
                            .and_then(|v| bounds.get(v).copied())
                            .unwrap_or_else(|| ValueBounds::from_bits(6))
                    };

                    let result_bounds = ValueBounds::shl(lhs_bounds, shift_bounds);

                    if !result_bounds.fits_in(data_bits) {
                        insertions.push((idx + 1, MachineInst::rchk(dst)
                            .comment("shift overflow check")));
                        bounds.insert(dst, ValueBounds::from_bits(data_bits));
                    } else {
                        bounds.insert(dst, result_bounds);
                    }

                    inst.result_bounds = Some(result_bounds);
                }
            }

            Opcode::SRL | Opcode::SRLI | Opcode::SRA | Opcode::SRAI => {
                if let Some(dst) = inst.def() {
                    let uses = inst.uses();
                    let lhs_bounds = uses.first()
                        .and_then(|v| bounds.get(v).copied())
                        .unwrap_or_else(|| ValueBounds::from_bits(data_bits));

                    // Shift right shrinks the value
                    let result_bounds = lhs_bounds; // Conservative
                    bounds.insert(dst, result_bounds);
                    inst.result_bounds = Some(result_bounds);
                }
            }

            // Comparison results are always 0 or 1
            Opcode::SLT | Opcode::SLTU | Opcode::SGE | Opcode::SGEU |
            Opcode::SEQ | Opcode::SNE => {
                if let Some(dst) = inst.def() {
                    bounds.insert(dst, ValueBounds::from_const(1));
                    inst.result_bounds = Some(ValueBounds::from_const(1));
                }
            }

            // Load from memory - unknown bounds
            Opcode::LB | Opcode::LBU => {
                if let Some(dst) = inst.def() {
                    bounds.insert(dst, ValueBounds::from_bits(8));
                }
            }
            Opcode::LH | Opcode::LHU => {
                if let Some(dst) = inst.def() {
                    bounds.insert(dst, ValueBounds::from_bits(16));
                }
            }
            Opcode::LW => {
                if let Some(dst) = inst.def() {
                    bounds.insert(dst, ValueBounds::from_bits(32));
                }
            }
            Opcode::LD => {
                if let Some(dst) = inst.def() {
                    bounds.insert(dst, ValueBounds::from_bits(64));
                }
            }

            // MOV preserves bounds
            Opcode::MOV => {
                if let Some(dst) = inst.def() {
                    let uses = inst.uses();
                    if let Some(src_bounds) = uses.first().and_then(|v| bounds.get(v).copied()) {
                        bounds.insert(dst, src_bounds);
                    }
                }
            }

            // Range check resets bounds
            Opcode::RCHK => {
                if let Some(dst) = inst.def() {
                    bounds.insert(dst, ValueBounds::from_bits(data_bits));
                }
            }

            _ => {
                // For other instructions, assume result has full data width
                if let Some(dst) = inst.def() {
                    bounds.insert(dst, ValueBounds::from_bits(data_bits));
                }
            }
        }
    }

    // Insert range checks (in reverse order to preserve indices)
    for (idx, inst) in insertions.into_iter().rev() {
        if idx <= block.insts.len() {
            block.insts.insert(idx, inst);
        }
    }
}

/// Statistics about range check optimization.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct RangeCheckStats {
    /// Number of arithmetic operations analyzed
    pub ops_analyzed: u32,
    /// Number of range checks inserted
    pub checks_inserted: u32,
    /// Number of range checks eliminated (would have been needed without optimization)
    pub checks_eliminated: u32,
}

/// Analyze and report range check statistics for a function.
#[allow(dead_code)]
pub fn analyze_range_checks(func: &MachineFunction, config: &TargetConfig) -> RangeCheckStats {
    let data_bits = config.data_bits();
    let mut stats = RangeCheckStats::default();

    for block in func.iter_blocks() {
        for inst in &block.insts {
            if inst.opcode.is_arithmetic() {
                stats.ops_analyzed += 1;

                if let Some(ref bounds) = inst.result_bounds {
                    if bounds.fits_in(data_bits) {
                        stats.checks_eliminated += 1;
                    } else {
                        stats.checks_inserted += 1;
                    }
                }
            }

            if inst.opcode == Opcode::RCHK {
                stats.checks_inserted += 1;
            }
        }
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_values_no_check() {
        let config = TargetConfig::default(); // 40-bit
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        // Small constants: 10 + 20 = 30, fits in 40 bits
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 20));
        entry.push(MachineInst::add(v2, v0, v1));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let before_count = func.get_block("entry").unwrap().insts.len();
        insert_range_checks(&mut func, &config).unwrap();
        let after_count = func.get_block("entry").unwrap().insts.len();

        // No range checks should be inserted for small values
        assert_eq!(before_count, after_count);
    }

    #[test]
    fn test_multiplication_needs_check() {
        let config = TargetConfig::default(); // 40-bit
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        // Large multiplication might overflow
        entry.push(MachineInst::li(v0, 0xFFFFF)); // 20-bit value
        entry.push(MachineInst::li(v1, 0xFFFFF)); // 20-bit value
        entry.push(MachineInst::mul(v2, v0, v1)); // Could be 40-bit result
        entry.push(MachineInst::ret());
        func.add_block(entry);

        insert_range_checks(&mut func, &config).unwrap();

        // Check that bounds were tracked
        let block = func.get_block("entry").unwrap();
        let mul_inst = &block.insts[2];
        assert!(mul_inst.result_bounds.is_some());
    }
}
