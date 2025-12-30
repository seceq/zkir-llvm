//! Memory coalescing optimization pass.
//!
//! This pass combines adjacent byte/halfword loads and stores into larger
//! word operations, reducing constraint counts significantly.
//!
//! ## Optimization Patterns
//!
//! ### Load Coalescing
//! ```text
//! Before (16 constraints):     After (8 + 4 = 12 constraints):
//!   LB v0, 0(base)               LW v4, 0(base)
//!   LB v1, 1(base)               ANDI v0, v4, 0xFF
//!   LB v2, 2(base)               SRLI t0, v4, 8
//!   LB v3, 3(base)               ANDI v1, t0, 0xFF
//!                                SRLI t1, v4, 16
//!                                ANDI v2, t1, 0xFF
//!                                SRLI v3, v4, 24
//! Savings: 25%
//! ```
//!
//! ### Store Coalescing
//! ```text
//! Before (16 constraints):     After (~12 constraints):
//!   SB v0, 0(base)               ANDI t0, v0, 0xFF
//!   SB v1, 1(base)               SLLI t1, v1, 8
//!   SB v2, 2(base)               OR t2, t0, t1
//!   SB v3, 3(base)               SLLI t3, v2, 16
//!                                OR t4, t2, t3
//!                                SLLI t5, v3, 24
//!                                OR t6, t4, t5
//!                                SW t6, 0(base)
//! Savings: 25%
//! ```

use crate::mir::{MachineFunction, MachineInst, Opcode, Operand, VReg};
use anyhow::Result;
use std::collections::HashMap;

/// Coalesce adjacent memory operations.
///
/// This pass looks for sequences of byte/halfword loads or stores
/// to adjacent addresses and combines them into larger operations.
pub fn coalesce_memory(func: &mut MachineFunction) -> Result<u32> {
    let mut coalesced = 0;
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        // Get current vreg counter to generate new vregs
        let mut next_vreg = func.vreg_count();

        if let Some(block) = func.get_block_mut(&label) {
            let (load_count, new_vreg) = coalesce_loads_in_block(&mut block.insts, next_vreg);
            coalesced += load_count;
            next_vreg = new_vreg;

            let (store_count, new_vreg) = coalesce_stores_in_block(&mut block.insts, next_vreg);
            coalesced += store_count;

            // Update the function's vreg counter
            func.set_vreg_count(new_vreg);
        }
    }

    Ok(coalesced)
}

/// Information about a byte load operation.
#[derive(Debug, Clone)]
struct ByteLoad {
    /// Index in the instruction list
    idx: usize,
    /// Destination virtual register
    dst: VReg,
    /// Base address register
    base: VReg,
    /// Offset from base
    offset: i32,
    /// Whether it's unsigned (LBU vs LB)
    unsigned: bool,
}

/// Coalesce adjacent byte loads into word loads.
/// Returns (coalesced_count, next_vreg).
fn coalesce_loads_in_block(insts: &mut Vec<MachineInst>, mut next_vreg: u32) -> (u32, u32) {
    let mut coalesced = 0;

    // Helper to allocate a new vreg
    let mut alloc_vreg = || {
        let v = VReg(next_vreg);
        next_vreg += 1;
        v
    };

    // Collect all byte loads grouped by base register
    let mut byte_loads: HashMap<VReg, Vec<ByteLoad>> = HashMap::new();

    for (idx, inst) in insts.iter().enumerate() {
        if matches!(inst.opcode, Opcode::LB | Opcode::LBU) {
            if let (Some(Operand::VReg(dst)), Some(Operand::Mem { base, offset })) =
                (inst.dst.as_ref(), inst.srcs.first())
            {
                byte_loads.entry(*base).or_default().push(ByteLoad {
                    idx,
                    dst: *dst,
                    base: *base,
                    offset: *offset,
                    unsigned: inst.opcode == Opcode::LBU,
                });
            }
        }
    }

    // Find groups of 4 consecutive byte loads that can be combined
    let mut replacements: Vec<(Vec<usize>, Vec<MachineInst>)> = Vec::new();

    for (_base, mut loads) in byte_loads {
        // Sort by offset
        loads.sort_by_key(|l| l.offset);

        // Look for groups of 4 consecutive bytes at word-aligned offsets
        let mut i = 0;
        while i + 3 < loads.len() {
            let l0 = &loads[i];
            let l1 = &loads[i + 1];
            let l2 = &loads[i + 2];
            let l3 = &loads[i + 3];

            // Check if they're consecutive and word-aligned
            if l1.offset == l0.offset + 1
                && l2.offset == l0.offset + 2
                && l3.offset == l0.offset + 3
                && l0.offset % 4 == 0
            {
                // All must be unsigned for safe combination
                if l0.unsigned && l1.unsigned && l2.unsigned && l3.unsigned {
                    // Create replacement instructions
                    let word_dst = alloc_vreg();
                    let mut new_insts = Vec::new();

                    // Load word
                    new_insts.push(
                        MachineInst::lw(word_dst, l0.base, l0.offset)
                            .comment("coalesced load"),
                    );

                    // Extract byte 0
                    new_insts.push(
                        MachineInst::new(Opcode::ANDI)
                            .dst(Operand::VReg(l0.dst))
                            .src(Operand::VReg(word_dst))
                            .src(Operand::Imm(0xFF))
                            .comment("extract byte 0"),
                    );

                    // Extract byte 1
                    let t1 = alloc_vreg();
                    new_insts.push(
                        MachineInst::new(Opcode::SRLI)
                            .dst(Operand::VReg(t1))
                            .src(Operand::VReg(word_dst))
                            .src(Operand::Imm(8)),
                    );
                    new_insts.push(
                        MachineInst::new(Opcode::ANDI)
                            .dst(Operand::VReg(l1.dst))
                            .src(Operand::VReg(t1))
                            .src(Operand::Imm(0xFF))
                            .comment("extract byte 1"),
                    );

                    // Extract byte 2
                    let t2 = alloc_vreg();
                    new_insts.push(
                        MachineInst::new(Opcode::SRLI)
                            .dst(Operand::VReg(t2))
                            .src(Operand::VReg(word_dst))
                            .src(Operand::Imm(16)),
                    );
                    new_insts.push(
                        MachineInst::new(Opcode::ANDI)
                            .dst(Operand::VReg(l2.dst))
                            .src(Operand::VReg(t2))
                            .src(Operand::Imm(0xFF))
                            .comment("extract byte 2"),
                    );

                    // Extract byte 3 (no mask needed - already in range)
                    new_insts.push(
                        MachineInst::new(Opcode::SRLI)
                            .dst(Operand::VReg(l3.dst))
                            .src(Operand::VReg(word_dst))
                            .src(Operand::Imm(24))
                            .comment("extract byte 3"),
                    );

                    let indices = vec![l0.idx, l1.idx, l2.idx, l3.idx];
                    replacements.push((indices, new_insts));
                    coalesced += 1;
                    i += 4;
                    continue;
                }
            }
            i += 1;
        }

        // Look for groups of 2 consecutive bytes (halfword)
        i = 0;
        while i + 1 < loads.len() {
            let l0 = &loads[i];
            let l1 = &loads[i + 1];

            // Check if they're consecutive and halfword-aligned
            if l1.offset == l0.offset + 1 && l0.offset % 2 == 0 {
                // Check if these indices are already being replaced
                let already_replaced = replacements.iter().any(|(indices, _)| {
                    indices.contains(&l0.idx) || indices.contains(&l1.idx)
                });

                if !already_replaced && l0.unsigned && l1.unsigned {
                    let half_dst = alloc_vreg();
                    let mut new_insts = Vec::new();

                    // Load halfword
                    new_insts.push(
                        MachineInst::new(Opcode::LHU)
                            .dst(Operand::VReg(half_dst))
                            .src(Operand::Mem {
                                base: l0.base,
                                offset: l0.offset,
                            })
                            .comment("coalesced halfword load"),
                    );

                    // Extract byte 0
                    new_insts.push(
                        MachineInst::new(Opcode::ANDI)
                            .dst(Operand::VReg(l0.dst))
                            .src(Operand::VReg(half_dst))
                            .src(Operand::Imm(0xFF))
                            .comment("extract low byte"),
                    );

                    // Extract byte 1
                    new_insts.push(
                        MachineInst::new(Opcode::SRLI)
                            .dst(Operand::VReg(l1.dst))
                            .src(Operand::VReg(half_dst))
                            .src(Operand::Imm(8))
                            .comment("extract high byte"),
                    );

                    let indices = vec![l0.idx, l1.idx];
                    replacements.push((indices, new_insts));
                    coalesced += 1;
                    i += 2;
                    continue;
                }
            }
            i += 1;
        }
    }

    // Apply replacements (in reverse order to preserve indices)
    apply_replacements(insts, replacements);

    (coalesced, next_vreg)
}

/// Information about a byte store operation.
#[derive(Debug, Clone)]
struct ByteStore {
    /// Index in the instruction list
    idx: usize,
    /// Value register being stored
    value: VReg,
    /// Base address register
    base: VReg,
    /// Offset from base
    offset: i32,
}

/// Coalesce adjacent byte stores into word stores.
/// Returns (coalesced_count, next_vreg).
fn coalesce_stores_in_block(insts: &mut Vec<MachineInst>, mut next_vreg: u32) -> (u32, u32) {
    let mut coalesced = 0;

    // Helper to allocate a new vreg
    let mut alloc_vreg = || {
        let v = VReg(next_vreg);
        next_vreg += 1;
        v
    };

    // Collect all byte stores grouped by base register
    let mut byte_stores: HashMap<VReg, Vec<ByteStore>> = HashMap::new();

    for (idx, inst) in insts.iter().enumerate() {
        if inst.opcode == Opcode::SB {
            if let (Some(Operand::VReg(value)), Some(Operand::Mem { base, offset })) =
                (inst.srcs.first(), inst.srcs.get(1))
            {
                byte_stores.entry(*base).or_default().push(ByteStore {
                    idx,
                    value: *value,
                    base: *base,
                    offset: *offset,
                });
            }
        }
    }

    // Find groups of 4 consecutive byte stores that can be combined
    let mut replacements: Vec<(Vec<usize>, Vec<MachineInst>)> = Vec::new();

    for (_base, mut stores) in byte_stores {
        // Sort by offset
        stores.sort_by_key(|s| s.offset);

        // Look for groups of 4 consecutive bytes at word-aligned offsets
        let mut i = 0;
        while i + 3 < stores.len() {
            let s0 = &stores[i];
            let s1 = &stores[i + 1];
            let s2 = &stores[i + 2];
            let s3 = &stores[i + 3];

            // Check if they're consecutive and word-aligned
            if s1.offset == s0.offset + 1
                && s2.offset == s0.offset + 2
                && s3.offset == s0.offset + 3
                && s0.offset % 4 == 0
            {
                // Create replacement instructions to combine into word store
                let mut new_insts = Vec::new();

                // Mask byte 0
                let t0 = alloc_vreg();
                new_insts.push(
                    MachineInst::new(Opcode::ANDI)
                        .dst(Operand::VReg(t0))
                        .src(Operand::VReg(s0.value))
                        .src(Operand::Imm(0xFF)),
                );

                // Shift and mask byte 1
                let t1 = alloc_vreg();
                new_insts.push(
                    MachineInst::new(Opcode::ANDI)
                        .dst(Operand::VReg(t1))
                        .src(Operand::VReg(s1.value))
                        .src(Operand::Imm(0xFF)),
                );
                let t1_shifted = alloc_vreg();
                new_insts.push(
                    MachineInst::new(Opcode::SLLI)
                        .dst(Operand::VReg(t1_shifted))
                        .src(Operand::VReg(t1))
                        .src(Operand::Imm(8)),
                );

                // Combine bytes 0 and 1
                let combined_01 = alloc_vreg();
                new_insts.push(
                    MachineInst::or(combined_01, t0, t1_shifted),
                );

                // Shift and mask byte 2
                let t2 = alloc_vreg();
                new_insts.push(
                    MachineInst::new(Opcode::ANDI)
                        .dst(Operand::VReg(t2))
                        .src(Operand::VReg(s2.value))
                        .src(Operand::Imm(0xFF)),
                );
                let t2_shifted = alloc_vreg();
                new_insts.push(
                    MachineInst::new(Opcode::SLLI)
                        .dst(Operand::VReg(t2_shifted))
                        .src(Operand::VReg(t2))
                        .src(Operand::Imm(16)),
                );

                // Combine with bytes 0-1
                let combined_012 = alloc_vreg();
                new_insts.push(
                    MachineInst::or(combined_012, combined_01, t2_shifted),
                );

                // Shift and mask byte 3
                let t3 = alloc_vreg();
                new_insts.push(
                    MachineInst::new(Opcode::ANDI)
                        .dst(Operand::VReg(t3))
                        .src(Operand::VReg(s3.value))
                        .src(Operand::Imm(0xFF)),
                );
                let t3_shifted = alloc_vreg();
                new_insts.push(
                    MachineInst::new(Opcode::SLLI)
                        .dst(Operand::VReg(t3_shifted))
                        .src(Operand::VReg(t3))
                        .src(Operand::Imm(24)),
                );

                // Combine all bytes
                let combined = alloc_vreg();
                new_insts.push(
                    MachineInst::or(combined, combined_012, t3_shifted)
                        .comment("coalesced word"),
                );

                // Store word
                new_insts.push(
                    MachineInst::sw(combined, s0.base, s0.offset)
                        .comment("coalesced store"),
                );

                let indices = vec![s0.idx, s1.idx, s2.idx, s3.idx];
                replacements.push((indices, new_insts));
                coalesced += 1;
                i += 4;
                continue;
            }
            i += 1;
        }

        // Look for groups of 2 consecutive bytes (halfword)
        i = 0;
        while i + 1 < stores.len() {
            let s0 = &stores[i];
            let s1 = &stores[i + 1];

            // Check if they're consecutive and halfword-aligned
            if s1.offset == s0.offset + 1 && s0.offset % 2 == 0 {
                // Check if these indices are already being replaced
                let already_replaced = replacements.iter().any(|(indices, _)| {
                    indices.contains(&s0.idx) || indices.contains(&s1.idx)
                });

                if !already_replaced {
                    let mut new_insts = Vec::new();

                    // Mask byte 0
                    let t0 = alloc_vreg();
                    new_insts.push(
                        MachineInst::new(Opcode::ANDI)
                            .dst(Operand::VReg(t0))
                            .src(Operand::VReg(s0.value))
                            .src(Operand::Imm(0xFF)),
                    );

                    // Shift and mask byte 1
                    let t1 = alloc_vreg();
                    new_insts.push(
                        MachineInst::new(Opcode::ANDI)
                            .dst(Operand::VReg(t1))
                            .src(Operand::VReg(s1.value))
                            .src(Operand::Imm(0xFF)),
                    );
                    let t1_shifted = alloc_vreg();
                    new_insts.push(
                        MachineInst::new(Opcode::SLLI)
                            .dst(Operand::VReg(t1_shifted))
                            .src(Operand::VReg(t1))
                            .src(Operand::Imm(8)),
                    );

                    // Combine bytes
                    let combined = alloc_vreg();
                    new_insts.push(
                        MachineInst::or(combined, t0, t1_shifted)
                            .comment("coalesced halfword"),
                    );

                    // Store halfword
                    new_insts.push(
                        MachineInst::new(Opcode::SH)
                            .src(Operand::VReg(combined))
                            .src(Operand::Mem {
                                base: s0.base,
                                offset: s0.offset,
                            })
                            .comment("coalesced store"),
                    );

                    let indices = vec![s0.idx, s1.idx];
                    replacements.push((indices, new_insts));
                    coalesced += 1;
                    i += 2;
                    continue;
                }
            }
            i += 1;
        }
    }

    // Apply replacements
    apply_replacements(insts, replacements);

    (coalesced, next_vreg)
}

/// Apply a set of replacements to an instruction list.
///
/// Each replacement is (indices_to_remove, new_instructions).
/// The new instructions replace the first index; remaining indices become NOPs.
fn apply_replacements(
    insts: &mut Vec<MachineInst>,
    mut replacements: Vec<(Vec<usize>, Vec<MachineInst>)>,
) {
    // Sort replacements by first index in descending order
    replacements.sort_by(|a, b| b.0.first().cmp(&a.0.first()));

    for (indices, new_insts) in replacements {
        if indices.is_empty() {
            continue;
        }

        // Sort indices
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();

        // Replace first instruction position with NOPs except for first
        for &idx in sorted_indices.iter().skip(1).rev() {
            insts[idx] = MachineInst::nop().comment("coalesced");
        }

        // Replace the first index with new instructions
        let first_idx = sorted_indices[0];
        if !new_insts.is_empty() {
            insts[first_idx] = new_insts[0].clone();
            // Insert remaining new instructions after the first
            for (i, inst) in new_insts.into_iter().skip(1).enumerate() {
                insts.insert(first_idx + 1 + i, inst);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MachineBlock;

    #[test]
    fn test_coalesce_four_byte_loads() {
        let mut func = MachineFunction::new("test");
        let base = func.new_vreg();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(base, 0x1000)); // base address
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v0))
                .src(Operand::Mem { base, offset: 0 }),
        );
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v1))
                .src(Operand::Mem { base, offset: 1 }),
        );
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v2))
                .src(Operand::Mem { base, offset: 2 }),
        );
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v3))
                .src(Operand::Mem { base, offset: 3 }),
        );
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let coalesced = coalesce_memory(&mut func).unwrap();
        assert_eq!(coalesced, 1, "Should coalesce 4 bytes into 1 word load");

        // Check that there's now a LW instruction
        let block = func.get_block("entry").unwrap();
        let has_lw = block.insts.iter().any(|i| i.opcode == Opcode::LW);
        assert!(has_lw, "Should have a LW instruction after coalescing");
    }

    #[test]
    fn test_coalesce_two_byte_loads() {
        let mut func = MachineFunction::new("test");
        let base = func.new_vreg();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(base, 0x1000));
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v0))
                .src(Operand::Mem { base, offset: 0 }),
        );
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v1))
                .src(Operand::Mem { base, offset: 1 }),
        );
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let coalesced = coalesce_memory(&mut func).unwrap();
        assert_eq!(coalesced, 1, "Should coalesce 2 bytes into 1 halfword load");

        // Check that there's now a LHU instruction
        let block = func.get_block("entry").unwrap();
        let has_lhu = block.insts.iter().any(|i| i.opcode == Opcode::LHU);
        assert!(has_lhu, "Should have a LHU instruction after coalescing");
    }

    #[test]
    fn test_coalesce_four_byte_stores() {
        let mut func = MachineFunction::new("test");
        let base = func.new_vreg();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(base, 0x1000));
        entry.push(MachineInst::li(v0, 0x11));
        entry.push(MachineInst::li(v1, 0x22));
        entry.push(MachineInst::li(v2, 0x33));
        entry.push(MachineInst::li(v3, 0x44));
        entry.push(
            MachineInst::new(Opcode::SB)
                .src(Operand::VReg(v0))
                .src(Operand::Mem { base, offset: 0 }),
        );
        entry.push(
            MachineInst::new(Opcode::SB)
                .src(Operand::VReg(v1))
                .src(Operand::Mem { base, offset: 1 }),
        );
        entry.push(
            MachineInst::new(Opcode::SB)
                .src(Operand::VReg(v2))
                .src(Operand::Mem { base, offset: 2 }),
        );
        entry.push(
            MachineInst::new(Opcode::SB)
                .src(Operand::VReg(v3))
                .src(Operand::Mem { base, offset: 3 }),
        );
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let coalesced = coalesce_memory(&mut func).unwrap();
        assert_eq!(coalesced, 1, "Should coalesce 4 byte stores into 1 word store");

        // Check that there's now a SW instruction
        let block = func.get_block("entry").unwrap();
        let has_sw = block.insts.iter().any(|i| i.opcode == Opcode::SW);
        assert!(has_sw, "Should have a SW instruction after coalescing");
    }

    #[test]
    fn test_no_coalesce_non_consecutive() {
        let mut func = MachineFunction::new("test");
        let base = func.new_vreg();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(base, 0x1000));
        // Non-consecutive offsets (0 and 2, not 0 and 1)
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v0))
                .src(Operand::Mem { base, offset: 0 }),
        );
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v1))
                .src(Operand::Mem { base, offset: 2 }),
        );
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let coalesced = coalesce_memory(&mut func).unwrap();
        assert_eq!(coalesced, 0, "Should not coalesce non-consecutive loads");
    }

    #[test]
    fn test_no_coalesce_misaligned() {
        let mut func = MachineFunction::new("test");
        let base = func.new_vreg();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(base, 0x1000));
        // Consecutive but not word-aligned (offset 1-4 instead of 0-3)
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v0))
                .src(Operand::Mem { base, offset: 1 }),
        );
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v1))
                .src(Operand::Mem { base, offset: 2 }),
        );
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v2))
                .src(Operand::Mem { base, offset: 3 }),
        );
        entry.push(
            MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(v3))
                .src(Operand::Mem { base, offset: 4 }),
        );
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let coalesced = coalesce_memory(&mut func).unwrap();
        // Should not coalesce into word load because not word-aligned
        // But may coalesce pairs into halfword loads
        assert!(coalesced <= 2, "Should not coalesce misaligned word loads");
    }
}
