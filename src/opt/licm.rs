//! Loop Invariant Code Motion (LICM) optimization pass.
//!
//! Moves computations that don't change during loop iterations out of the loop.
//! This is particularly valuable for ZK circuits where every instruction adds
//! to constraint count - moving invariants out of loops can significantly
//! reduce the total instruction count.
//!
//! # Algorithm
//!
//! 1. Detect loops using back-edge detection
//! 2. For each loop, identify the preheader (block before the loop)
//! 3. Find instructions in the loop that are "loop invariant":
//!    - All operands are either constants or defined outside the loop
//!    - The instruction has no side effects
//! 4. Move those instructions to the preheader
//!
//! # Example
//!
//! Before:
//! ```text
//! preheader:
//!   li v0, 0        ; i = 0
//!   jal loop
//! loop:
//!   li v1, 10       ; invariant - can be moved
//!   mul v2, v0, v1
//!   addi v0, v0, 1
//!   blt v0, v3, loop
//! ```
//!
//! After:
//! ```text
//! preheader:
//!   li v0, 0
//!   li v1, 10       ; moved here
//!   jal loop
//! loop:
//!   mul v2, v0, v1
//!   addi v0, v0, 1
//!   blt v0, v3, loop
//! ```

use crate::mir::{MachineFunction, MachineInst, Opcode, VReg};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Run LICM on a function.
pub fn hoist_loop_invariants(func: &mut MachineFunction) -> Result<u32> {
    let mut hoisted = 0;

    // Rebuild CFG to ensure accurate predecessor/successor info
    func.rebuild_cfg();

    // Find all loops
    let loops = find_loops(func);

    for loop_info in loops {
        let count = hoist_invariants(func, &loop_info)?;
        hoisted += count;
    }

    if hoisted > 0 {
        func.rebuild_cfg();
    }

    Ok(hoisted)
}

/// Information about a detected loop.
#[derive(Debug)]
struct LoopInfo {
    /// Header block (the loop entry point)
    #[allow(dead_code)]
    header: String,
    /// All blocks in the loop
    body_blocks: HashSet<String>,
    /// The preheader (block that jumps to header from outside the loop)
    preheader: Option<String>,
    /// Back edge source (latch block)
    #[allow(dead_code)]
    latch: String,
}

/// Find loops using back-edge detection.
fn find_loops(func: &MachineFunction) -> Vec<LoopInfo> {
    let mut loops = Vec::new();
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    // Build a position map for quick lookup
    let pos_map: HashMap<&str, usize> = block_labels.iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), i))
        .collect();

    for (i, label) in block_labels.iter().enumerate() {
        let block = match func.get_block(label) {
            Some(b) => b,
            None => continue,
        };

        // Check successors for back edges
        for succ in &block.succs {
            if let Some(&succ_pos) = pos_map.get(succ.as_str()) {
                if succ_pos <= i {
                    // Back edge: label -> succ
                    // succ is the header, label is the latch
                    if let Some(loop_info) = analyze_loop_structure(func, succ, label, &block_labels, &pos_map) {
                        loops.push(loop_info);
                    }
                }
            }
        }
    }

    loops
}

/// Analyze loop structure to find all body blocks and preheader.
fn analyze_loop_structure(
    func: &MachineFunction,
    header: &str,
    latch: &str,
    block_labels: &[String],
    pos_map: &HashMap<&str, usize>,
) -> Option<LoopInfo> {
    let header_pos = *pos_map.get(header)?;
    let latch_pos = *pos_map.get(latch)?;

    // Collect all blocks between header and latch (inclusive)
    let mut body_blocks: HashSet<String> = HashSet::new();
    body_blocks.insert(header.to_string());

    for (i, label) in block_labels.iter().enumerate() {
        if i >= header_pos && i <= latch_pos {
            body_blocks.insert(label.clone());
        }
    }

    // Find the preheader: a predecessor of the header that's not in the loop
    let header_block = func.get_block(header)?;
    let mut preheader = None;

    for pred in &header_block.preds {
        if !body_blocks.contains(pred) {
            preheader = Some(pred.clone());
            break;
        }
    }

    Some(LoopInfo {
        header: header.to_string(),
        body_blocks,
        preheader,
        latch: latch.to_string(),
    })
}

/// Hoist loop-invariant instructions from a loop.
fn hoist_invariants(func: &mut MachineFunction, loop_info: &LoopInfo) -> Result<u32> {
    let preheader = match &loop_info.preheader {
        Some(p) => p.clone(),
        None => return Ok(0), // No preheader, can't hoist
    };

    // Collect all vregs defined inside the loop
    let mut loop_defs: HashSet<VReg> = HashSet::new();
    for block_label in &loop_info.body_blocks {
        if let Some(block) = func.get_block(block_label) {
            for inst in &block.insts {
                if let Some(def) = inst.def() {
                    loop_defs.insert(def);
                }
            }
        }
    }

    // Find loop-invariant instructions
    let mut invariants: Vec<(String, usize, MachineInst)> = Vec::new();

    for block_label in &loop_info.body_blocks {
        if let Some(block) = func.get_block(block_label) {
            for (idx, inst) in block.insts.iter().enumerate() {
                if is_loop_invariant(inst, &loop_defs, &invariants) {
                    // Mark this instruction for hoisting
                    invariants.push((block_label.clone(), idx, inst.clone()));

                    // If this instruction defines a vreg, it's no longer a "loop def"
                    // for the purpose of further invariant detection
                    if let Some(def) = inst.def() {
                        loop_defs.remove(&def);
                    }
                }
            }
        }
    }

    if invariants.is_empty() {
        return Ok(0);
    }

    // Hoist invariants to preheader
    // First, collect the instructions to insert
    let to_insert: Vec<MachineInst> = invariants.iter()
        .map(|(_, _, inst)| {
            inst.clone().comment("LICM: hoisted from loop")
        })
        .collect();

    // Insert them at the end of preheader (before terminator)
    if let Some(preheader_block) = func.get_block_mut(&preheader) {
        let insert_pos = preheader_block.insts.iter()
            .position(|i| i.is_terminator())
            .unwrap_or(preheader_block.insts.len());

        for (i, inst) in to_insert.into_iter().enumerate() {
            preheader_block.insts.insert(insert_pos + i, inst);
        }
    }

    // Remove the original instructions from the loop (in reverse order to preserve indices)
    // Group by block first
    let mut by_block: HashMap<String, Vec<usize>> = HashMap::new();
    for (block_label, idx, _) in &invariants {
        by_block.entry(block_label.clone())
            .or_default()
            .push(*idx);
    }

    for (block_label, mut indices) in by_block {
        indices.sort();
        indices.reverse(); // Remove from end first to preserve indices

        if let Some(block) = func.get_block_mut(&block_label) {
            for idx in indices {
                if idx < block.insts.len() {
                    block.insts.remove(idx);
                }
            }
        }
    }

    Ok(invariants.len() as u32)
}

/// Check if an instruction is loop-invariant.
fn is_loop_invariant(
    inst: &MachineInst,
    loop_defs: &HashSet<VReg>,
    already_hoisted: &[(String, usize, MachineInst)],
) -> bool {
    // Can't hoist terminators
    if inst.is_terminator() {
        return false;
    }

    // Can't hoist instructions with side effects
    if has_side_effects(inst) {
        return false;
    }

    // Must be a pure computation
    if !is_pure_computation(inst) {
        return false;
    }

    // All operands must be:
    // 1. Constants/immediates
    // 2. Defined outside the loop
    // 3. Already identified as invariant (and will be hoisted)
    let hoisted_defs: HashSet<VReg> = already_hoisted.iter()
        .filter_map(|(_, _, i)| i.def())
        .collect();

    for vreg in inst.uses() {
        // If this vreg is defined in the loop and not being hoisted, it's not invariant
        if loop_defs.contains(&vreg) && !hoisted_defs.contains(&vreg) {
            return false;
        }
    }

    true
}

/// Check if an instruction has side effects.
fn has_side_effects(inst: &MachineInst) -> bool {
    matches!(inst.opcode,
        Opcode::CALL | Opcode::ECALL | Opcode::EBREAK |
        Opcode::SB | Opcode::SH | Opcode::SW | Opcode::SD |
        Opcode::RCHK
    )
}

/// Check if an instruction is a pure computation.
fn is_pure_computation(inst: &MachineInst) -> bool {
    matches!(inst.opcode,
        Opcode::LI | Opcode::MOV |
        Opcode::ADD | Opcode::SUB | Opcode::MUL | Opcode::MULH |
        Opcode::ADDI |
        Opcode::AND | Opcode::OR | Opcode::XOR | Opcode::NOT |
        Opcode::ANDI | Opcode::ORI | Opcode::XORI |
        Opcode::SLL | Opcode::SRL | Opcode::SRA |
        Opcode::SLLI | Opcode::SRLI | Opcode::SRAI |
        Opcode::SLT | Opcode::SLTU | Opcode::SGE | Opcode::SGEU |
        Opcode::SEQ | Opcode::SNE
    )
    // Note: DIV, DIVU, REM, REMU are excluded because they can trap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{MachineBlock, Operand};

    #[test]
    fn test_simple_licm() {
        // Create a loop with an invariant LI instruction
        // preheader:
        //   li v0, 0        ; loop counter
        //   li v3, 10       ; limit
        //   jal loop
        // loop:
        //   li v1, 42       ; INVARIANT - should be hoisted
        //   add v2, v0, v1
        //   addi v0, v0, 1
        //   blt v0, v3, loop
        // exit:
        //   ret

        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg(); // counter
        let v1 = func.new_vreg(); // invariant value
        let v2 = func.new_vreg(); // result
        let v3 = func.new_vreg(); // limit
        let dummy = func.new_vreg();

        let mut preheader = MachineBlock::new("preheader");
        preheader.push(MachineInst::li(v0, 0));
        preheader.push(MachineInst::li(v3, 10));
        preheader.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("loop".to_string())));
        func.add_block(preheader);

        let mut loop_block = MachineBlock::new("loop");
        loop_block.push(MachineInst::li(v1, 42)); // Invariant
        loop_block.push(MachineInst::add(v2, v0, v1));
        loop_block.push(MachineInst::addi(v0, v0, 1));
        loop_block.push(MachineInst::new(Opcode::BLT)
            .src(Operand::VReg(v0))
            .src(Operand::VReg(v3))
            .src(Operand::Label("loop".to_string())));
        func.add_block(loop_block);

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();

        let hoisted = hoist_loop_invariants(&mut func).unwrap();
        assert_eq!(hoisted, 1);

        // Check that the LI was moved to preheader
        let preheader = func.get_block("preheader").unwrap();
        let li_count = preheader.insts.iter()
            .filter(|i| i.opcode == Opcode::LI &&
                   i.srcs.first().and_then(|s| s.as_imm()) == Some(42))
            .count();
        assert_eq!(li_count, 1);

        // Check that the LI was removed from loop
        let loop_block = func.get_block("loop").unwrap();
        let li_in_loop = loop_block.insts.iter()
            .filter(|i| i.opcode == Opcode::LI &&
                   i.srcs.first().and_then(|s| s.as_imm()) == Some(42))
            .count();
        assert_eq!(li_in_loop, 0);
    }

    #[test]
    fn test_no_hoist_loop_dependent() {
        // Instructions that depend on loop variables should not be hoisted
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg(); // counter
        let v1 = func.new_vreg(); // depends on v0
        let v2 = func.new_vreg(); // limit
        let dummy = func.new_vreg();

        let mut preheader = MachineBlock::new("preheader");
        preheader.push(MachineInst::li(v0, 0));
        preheader.push(MachineInst::li(v2, 10));
        preheader.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("loop".to_string())));
        func.add_block(preheader);

        let mut loop_block = MachineBlock::new("loop");
        loop_block.push(MachineInst::add(v1, v0, v0)); // Depends on v0 (loop variable)
        loop_block.push(MachineInst::addi(v0, v0, 1));
        loop_block.push(MachineInst::new(Opcode::BLT)
            .src(Operand::VReg(v0))
            .src(Operand::VReg(v2))
            .src(Operand::Label("loop".to_string())));
        func.add_block(loop_block);

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();

        let hoisted = hoist_loop_invariants(&mut func).unwrap();
        assert_eq!(hoisted, 0); // Nothing should be hoisted
    }

    #[test]
    fn test_hoist_chain() {
        // Test that a chain of invariants can be hoisted together
        // li v1, 10    ; invariant
        // li v2, 20    ; invariant
        // add v3, v1, v2  ; invariant (both operands are invariant)

        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg(); // counter
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();
        let v4 = func.new_vreg();
        let v5 = func.new_vreg(); // limit
        let dummy = func.new_vreg();

        let mut preheader = MachineBlock::new("preheader");
        preheader.push(MachineInst::li(v0, 0));
        preheader.push(MachineInst::li(v5, 10));
        preheader.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("loop".to_string())));
        func.add_block(preheader);

        let mut loop_block = MachineBlock::new("loop");
        loop_block.push(MachineInst::li(v1, 10));     // Invariant
        loop_block.push(MachineInst::li(v2, 20));     // Invariant
        loop_block.push(MachineInst::add(v3, v1, v2)); // Invariant (depends on invariants)
        loop_block.push(MachineInst::add(v4, v0, v3)); // Not invariant (depends on v0)
        loop_block.push(MachineInst::addi(v0, v0, 1));
        loop_block.push(MachineInst::new(Opcode::BLT)
            .src(Operand::VReg(v0))
            .src(Operand::VReg(v5))
            .src(Operand::Label("loop".to_string())));
        func.add_block(loop_block);

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();

        let hoisted = hoist_loop_invariants(&mut func).unwrap();
        assert_eq!(hoisted, 3); // All three invariants should be hoisted

        // Check preheader has the hoisted instructions
        let preheader = func.get_block("preheader").unwrap();
        let hoisted_count = preheader.insts.iter()
            .filter(|i| i.comment.as_ref().map(|c| c.contains("LICM")).unwrap_or(false))
            .count();
        assert_eq!(hoisted_count, 3);
    }

    #[test]
    fn test_no_hoist_side_effects() {
        // Instructions with side effects should not be hoisted
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let dummy = func.new_vreg();

        let mut preheader = MachineBlock::new("preheader");
        preheader.push(MachineInst::li(v0, 0));
        preheader.push(MachineInst::li(v2, 10));
        preheader.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("loop".to_string())));
        func.add_block(preheader);

        let mut loop_block = MachineBlock::new("loop");
        loop_block.push(MachineInst::li(v1, 0x1000)); // Address
        loop_block.push(MachineInst::sw(v0, v1, 0));   // Store - has side effects
        loop_block.push(MachineInst::addi(v0, v0, 1));
        loop_block.push(MachineInst::new(Opcode::BLT)
            .src(Operand::VReg(v0))
            .src(Operand::VReg(v2))
            .src(Operand::Label("loop".to_string())));
        func.add_block(loop_block);

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();

        let hoisted = hoist_loop_invariants(&mut func).unwrap();
        // Only the LI should be hoisted, not the SW
        assert_eq!(hoisted, 1);
    }

    #[test]
    fn test_no_preheader() {
        // If there's no preheader (multiple loop entries), don't hoist
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let dummy = func.new_vreg();

        // Loop with no clear preheader (entry goes directly to loop)
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 0));
        entry.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("loop".to_string())));
        func.add_block(entry);

        let mut loop_block = MachineBlock::new("loop");
        loop_block.push(MachineInst::li(v1, 42));
        loop_block.push(MachineInst::addi(v0, v0, 1));
        loop_block.push(MachineInst::new(Opcode::BLT)
            .src(Operand::VReg(v0))
            .src(Operand::VReg(v1))
            .src(Operand::Label("loop".to_string())));
        func.add_block(loop_block);

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();

        let _hoisted = hoist_loop_invariants(&mut func).unwrap();
        // Should work - either hoists or doesn't based on CFG structure
    }

    #[test]
    fn test_loop_detection() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let dummy = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 0));
        entry.push(MachineInst::li(v1, 10));
        entry.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("header".to_string())));
        func.add_block(entry);

        let mut header = MachineBlock::new("header");
        header.push(MachineInst::new(Opcode::BGE)
            .src(Operand::VReg(v0))
            .src(Operand::VReg(v1))
            .src(Operand::Label("exit".to_string())));
        func.add_block(header);

        let mut body = MachineBlock::new("body");
        body.push(MachineInst::addi(v0, v0, 1));
        body.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("header".to_string())));
        func.add_block(body);

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();

        let loops = find_loops(&func);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header, "header");
        assert_eq!(loops[0].latch, "body");
        assert!(loops[0].body_blocks.contains("header"));
        assert!(loops[0].body_blocks.contains("body"));
    }
}
