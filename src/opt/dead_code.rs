//! Dead code elimination.
//!
//! Removes instructions whose results are never used.

use crate::mir::{MachineFunction, Opcode, VReg};
use anyhow::Result;
use std::collections::HashSet;

/// Eliminate dead code from a function.
pub fn eliminate_dead_code(func: &mut MachineFunction) -> Result<()> {
    // Iterate until no more changes
    let mut changed = true;
    while changed {
        changed = eliminate_dead_code_pass(func);
    }
    Ok(())
}

/// Single pass of dead code elimination. Returns true if any code was removed.
fn eliminate_dead_code_pass(func: &mut MachineFunction) -> bool {
    // Collect all used vregs
    let mut used: HashSet<VReg> = HashSet::new();

    // First pass: find all uses
    for block in func.iter_blocks() {
        for inst in &block.insts {
            // Uses in source operands
            for vreg in inst.uses() {
                used.insert(vreg);
            }

            // Memory base registers are uses too
            for src in &inst.srcs {
                match src {
                    crate::mir::Operand::Mem { base, .. } => {
                        used.insert(*base);
                    }
                    _ => {}
                }
            }
        }
    }

    // Also mark parameter vregs as used (they're live-in)
    for param in &func.params {
        used.insert(*param);
    }

    // Mark return vreg as used
    if let Some(ret) = func.ret_vreg {
        used.insert(ret);
    }

    // Second pass: remove dead definitions
    let mut changed = false;
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        if let Some(block) = func.get_block_mut(&label) {
            let original_len = block.insts.len();

            block.insts.retain(|inst| {
                // Never remove terminators, calls, stores, or side-effectful instructions
                if inst.is_terminator() ||
                   inst.opcode == Opcode::CALL ||
                   inst.opcode.is_store() ||
                   inst.opcode == Opcode::ECALL ||
                   inst.opcode == Opcode::EBREAK ||
                   inst.opcode == Opcode::RCHK
                {
                    return true;
                }

                // Keep if no definition or definition is used
                match inst.def() {
                    Some(def) => used.contains(&def),
                    None => true, // No def, might have side effects
                }
            });

            if block.insts.len() != original_len {
                changed = true;
            }
        }
    }

    changed
}

/// Count dead instructions without removing them.
#[allow(dead_code)]
pub fn count_dead_instructions(func: &MachineFunction) -> u32 {
    let mut used: HashSet<VReg> = HashSet::new();

    // Collect uses
    for block in func.iter_blocks() {
        for inst in &block.insts {
            for vreg in inst.uses() {
                used.insert(vreg);
            }
        }
    }

    // Count dead defs
    let mut dead_count = 0u32;

    for block in func.iter_blocks() {
        for inst in &block.insts {
            if !inst.is_terminator() &&
               inst.opcode != Opcode::CALL &&
               !inst.opcode.is_store()
            {
                if let Some(def) = inst.def() {
                    if !used.contains(&def) {
                        dead_count += 1;
                    }
                }
            }
        }
    }

    dead_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{MachineBlock, MachineInst};

    #[test]
    fn test_eliminate_unused_def() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg(); // This will be unused
        let v2 = func.new_vreg();
        func.ret_vreg = Some(v2); // Mark v2 as return value (used)

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10)); // used by add
        entry.push(MachineInst::li(v1, 20)); // dead - v1 never used
        entry.push(MachineInst::addi(v2, v0, 5)); // uses v0, defines return value
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let before = func.get_block("entry").unwrap().insts.len();
        eliminate_dead_code(&mut func).unwrap();
        let after = func.get_block("entry").unwrap().insts.len();

        assert_eq!(before, 4);
        assert_eq!(after, 3); // li(v1) should be removed
    }

    #[test]
    fn test_keep_used_def() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        func.ret_vreg = Some(v2);

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 20));
        entry.push(MachineInst::add(v2, v0, v1)); // v2 is return value
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let before = func.get_block("entry").unwrap().insts.len();
        eliminate_dead_code(&mut func).unwrap();
        let after = func.get_block("entry").unwrap().insts.len();

        // All instructions should be kept
        assert_eq!(before, after);
    }

    #[test]
    fn test_keep_stores() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 0x1000)); // address
        entry.push(MachineInst::sw(v0, v1, 0)); // store - has side effects
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let before = func.get_block("entry").unwrap().insts.len();
        eliminate_dead_code(&mut func).unwrap();
        let after = func.get_block("entry").unwrap().insts.len();

        // Store should not be removed
        assert_eq!(before, after);
    }
}
