//! PHI node elimination (SSA destruction).
//!
//! Converts SSA form to conventional form by inserting parallel copies
//! at the end of predecessor blocks for each PHI node.
//!
//! The algorithm:
//! 1. For each PHI node in a block, collect (predecessor, value) pairs
//! 2. Insert MOV instructions at the end of each predecessor block
//!    (before the terminator)
//! 3. Remove the PHI pseudo-instructions
//!
//! Note: This simple algorithm may create critical edges that need
//! splitting in complex CFGs. For now, we handle the common cases.

use crate::mir::{MachineBlock, MachineFunction, MachineInst, Opcode, Operand, VReg};
use anyhow::Result;
use std::collections::HashMap;

/// PHI information collected from a block.
#[derive(Debug)]
struct PhiInfo {
    /// Destination virtual register
    dst: VReg,
    /// Map from predecessor label to source vreg
    sources: HashMap<String, VReg>,
}

/// Eliminate PHI nodes from a function.
pub fn eliminate_phis(func: &mut MachineFunction) -> Result<()> {
    // Collect PHI information from all blocks
    let mut phis_by_block: HashMap<String, Vec<PhiInfo>> = HashMap::new();

    for block in func.iter_blocks() {
        let phis = collect_phis(block);
        if !phis.is_empty() {
            phis_by_block.insert(block.label.clone(), phis);
        }
    }

    if phis_by_block.is_empty() {
        return Ok(());
    }

    // For each PHI, insert copies in predecessor blocks
    let mut copies_to_insert: HashMap<String, Vec<(VReg, VReg)>> = HashMap::new();

    for (block_label, phis) in &phis_by_block {
        if let Some(_block) = func.get_block(block_label) {
            for phi in phis {
                for (pred_label, src_vreg) in &phi.sources {
                    copies_to_insert
                        .entry(pred_label.clone())
                        .or_default()
                        .push((*src_vreg, phi.dst));
                }
            }
        }
    }

    // Insert copies before terminators in predecessor blocks
    for (pred_label, copies) in copies_to_insert {
        if let Some(block) = func.get_block_mut(&pred_label) {
            insert_copies_before_terminator(block, &copies);
        }
    }

    // Remove PHI nodes
    for (block_label, _) in phis_by_block {
        if let Some(block) = func.get_block_mut(&block_label) {
            remove_phi_nodes(block);
        }
    }

    Ok(())
}

/// Collect PHI nodes from a block.
fn collect_phis(block: &MachineBlock) -> Vec<PhiInfo> {
    let mut phis = Vec::new();

    for inst in &block.insts {
        if inst.opcode == Opcode::PHI {
            if let Some(dst) = inst.def() {
                // Get incoming values from the PHI instruction
                let sources: HashMap<String, VReg> = inst.phi_incomings()
                    .into_iter()
                    .collect();

                phis.push(PhiInfo {
                    dst,
                    sources,
                });
            }
        }
    }

    phis
}

/// Insert parallel copies before the terminator instruction.
fn insert_copies_before_terminator(block: &mut MachineBlock, copies: &[(VReg, VReg)]) {
    // Find the terminator position
    let term_pos = block.insts.iter().position(|inst| inst.is_terminator());

    // Create MOV instructions for each copy
    let movs: Vec<MachineInst> = copies
        .iter()
        .map(|(src, dst)| {
            MachineInst::mov(*dst, *src).comment("phi copy")
        })
        .collect();

    // Insert before terminator
    if let Some(pos) = term_pos {
        // Insert in reverse to maintain order
        for mov in movs.into_iter().rev() {
            block.insts.insert(pos, mov);
        }
    } else {
        // No terminator, append at end
        for mov in movs {
            block.push(mov);
        }
    }
}

/// Remove PHI instructions from a block.
fn remove_phi_nodes(block: &mut MachineBlock) {
    block.insts.retain(|inst| inst.opcode != Opcode::PHI);
}

/// Detect and split critical edges in the CFG.
///
/// A critical edge is an edge from a block with multiple successors
/// to a block with multiple predecessors. PHI elimination requires
/// splitting these edges.
#[allow(dead_code)]
pub fn split_critical_edges(func: &mut MachineFunction) -> Result<u32> {
    let mut edges_split = 0;
    let mut new_blocks: Vec<MachineBlock> = Vec::new();

    // Collect critical edges
    let mut critical_edges: Vec<(String, String)> = Vec::new();

    for block in func.iter_blocks() {
        if block.succs.len() > 1 {
            for succ_label in &block.succs {
                if let Some(succ) = func.get_block(succ_label) {
                    if succ.preds.len() > 1 {
                        critical_edges.push((block.label.clone(), succ_label.clone()));
                    }
                }
            }
        }
    }

    // Split each critical edge by inserting a new block
    for (from_label, to_label) in critical_edges {
        let split_label = format!("{}_to_{}_split", from_label, to_label);

        // Create new block with just a jump to the successor
        let mut split_block = MachineBlock::new(&split_label);
        let zero = func.new_vreg();
        split_block.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(zero))
            .src(Operand::Label(to_label.clone()))
            .comment("critical edge split"));

        new_blocks.push(split_block);

        // Update the source block's terminator to jump to split block
        if let Some(from_block) = func.get_block_mut(&from_label) {
            for inst in &mut from_block.insts {
                if inst.is_terminator() {
                    for src in &mut inst.srcs {
                        if let Operand::Label(label) = src {
                            if label == &to_label {
                                *label = split_label.clone();
                            }
                        }
                    }
                }
            }
        }

        edges_split += 1;
    }

    // Add new blocks to function
    for block in new_blocks {
        func.add_block(block);
    }

    Ok(edges_split)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::Operand;

    #[test]
    fn test_insert_copies() {
        let mut block = MachineBlock::new("test");
        block.push(MachineInst::li(VReg(0), 10));
        block.push(MachineInst::ret());

        let copies = vec![(VReg(1), VReg(2)), (VReg(3), VReg(4))];
        insert_copies_before_terminator(&mut block, &copies);

        // Should have: li, mov, mov, ret
        assert_eq!(block.insts.len(), 4);
        assert_eq!(block.insts[0].opcode, Opcode::LI);
        assert_eq!(block.insts[1].opcode, Opcode::MOV);
        assert_eq!(block.insts[2].opcode, Opcode::MOV);
        assert_eq!(block.insts[3].opcode, Opcode::RET);
    }

    #[test]
    fn test_eliminate_empty_phis() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        // Should not fail on function with no PHIs
        eliminate_phis(&mut func).unwrap();
        assert_eq!(func.get_block("entry").unwrap().insts.len(), 2);
    }

    #[test]
    fn test_phi_instruction_creation() {
        // Test that PHI instructions are created correctly
        let phi = MachineInst::phi(VReg(0))
            .phi_incoming("bb1", VReg(1))
            .phi_incoming("bb2", VReg(2));

        assert_eq!(phi.opcode, Opcode::PHI);
        assert_eq!(phi.def(), Some(VReg(0)));

        let incomings = phi.phi_incomings();
        assert_eq!(incomings.len(), 2);
        assert!(incomings.contains(&("bb1".to_string(), VReg(1))));
        assert!(incomings.contains(&("bb2".to_string(), VReg(2))));
    }

    #[test]
    fn test_collect_phis() {
        let mut block = MachineBlock::new("merge");
        block.push(MachineInst::phi(VReg(0))
            .phi_incoming("left", VReg(1))
            .phi_incoming("right", VReg(2)));
        block.push(MachineInst::ret());

        let phis = collect_phis(&block);
        assert_eq!(phis.len(), 1);
        assert_eq!(phis[0].dst, VReg(0));
        assert_eq!(phis[0].sources.len(), 2);
        assert_eq!(phis[0].sources.get("left"), Some(&VReg(1)));
        assert_eq!(phis[0].sources.get("right"), Some(&VReg(2)));
    }

    #[test]
    fn test_eliminate_simple_phi() {
        // Build a diamond CFG:
        //       entry
        //       /   \
        //    left   right
        //       \   /
        //       merge (with PHI)
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg(); // condition
        let v1 = func.new_vreg(); // left value
        let v2 = func.new_vreg(); // right value
        let v3 = func.new_vreg(); // PHI result

        // Entry block
        let mut entry = MachineBlock::new("entry");
        let zero = func.new_vreg();
        entry.push(MachineInst::li(v0, 1));
        entry.push(MachineInst::li(zero, 0));
        entry.push(MachineInst::bne(v0, zero, "left"));
        entry.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(func.new_vreg()))
            .src(Operand::Label("right".to_string())));
        func.add_block(entry);

        // Left block
        let mut left = MachineBlock::new("left");
        left.push(MachineInst::li(v1, 10));
        left.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(func.new_vreg()))
            .src(Operand::Label("merge".to_string())));
        func.add_block(left);

        // Right block
        let mut right = MachineBlock::new("right");
        right.push(MachineInst::li(v2, 20));
        right.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(func.new_vreg()))
            .src(Operand::Label("merge".to_string())));
        func.add_block(right);

        // Merge block with PHI
        let mut merge = MachineBlock::new("merge");
        merge.push(MachineInst::phi(v3)
            .phi_incoming("left", v1)
            .phi_incoming("right", v2));
        merge.push(MachineInst::ret());
        func.add_block(merge);

        // Run PHI elimination
        eliminate_phis(&mut func).unwrap();

        // Verify PHI is removed from merge block
        let merge_block = func.get_block("merge").unwrap();
        assert!(!merge_block.insts.iter().any(|i| i.opcode == Opcode::PHI));

        // Verify MOV was inserted in left block before terminator
        let left_block = func.get_block("left").unwrap();
        let has_mov = left_block.insts.iter().any(|i| {
            i.opcode == Opcode::MOV &&
            i.def() == Some(v3) &&
            i.uses().contains(&v1)
        });
        assert!(has_mov, "left block should have MOV v3, v1");

        // Verify MOV was inserted in right block before terminator
        let right_block = func.get_block("right").unwrap();
        let has_mov = right_block.insts.iter().any(|i| {
            i.opcode == Opcode::MOV &&
            i.def() == Some(v3) &&
            i.uses().contains(&v2)
        });
        assert!(has_mov, "right block should have MOV v3, v2");
    }

    #[test]
    fn test_multiple_phis_same_block() {
        // Test multiple PHIs in the same merge block
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();
        let v4 = func.new_vreg(); // PHI result 1
        let v5 = func.new_vreg(); // PHI result 2

        // Left block
        let mut left = MachineBlock::new("left");
        left.push(MachineInst::li(v0, 10));
        left.push(MachineInst::li(v1, 100));
        left.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(func.new_vreg()))
            .src(Operand::Label("merge".to_string())));
        func.add_block(left);

        // Right block
        let mut right = MachineBlock::new("right");
        right.push(MachineInst::li(v2, 20));
        right.push(MachineInst::li(v3, 200));
        right.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(func.new_vreg()))
            .src(Operand::Label("merge".to_string())));
        func.add_block(right);

        // Merge block with two PHIs
        let mut merge = MachineBlock::new("merge");
        merge.push(MachineInst::phi(v4)
            .phi_incoming("left", v0)
            .phi_incoming("right", v2));
        merge.push(MachineInst::phi(v5)
            .phi_incoming("left", v1)
            .phi_incoming("right", v3));
        merge.push(MachineInst::ret());
        func.add_block(merge);

        // Run PHI elimination
        eliminate_phis(&mut func).unwrap();

        // Verify both PHIs are removed
        let merge_block = func.get_block("merge").unwrap();
        assert!(!merge_block.insts.iter().any(|i| i.opcode == Opcode::PHI));
        assert_eq!(merge_block.insts.len(), 1); // Only RET remains

        // Left block should have 2 MOVs
        let left_block = func.get_block("left").unwrap();
        let mov_count = left_block.insts.iter().filter(|i| i.opcode == Opcode::MOV).count();
        assert_eq!(mov_count, 2, "left block should have 2 MOVs");

        // Right block should have 2 MOVs
        let right_block = func.get_block("right").unwrap();
        let mov_count = right_block.insts.iter().filter(|i| i.opcode == Opcode::MOV).count();
        assert_eq!(mov_count, 2, "right block should have 2 MOVs");
    }
}
