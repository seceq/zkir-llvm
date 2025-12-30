//! Machine functions.

use super::block::MachineBlock;
use super::value::VReg;
use crate::target::abi::StackFrame;
use indexmap::IndexMap;
use std::fmt;

/// A machine function.
#[derive(Debug, Clone)]
pub struct MachineFunction {
    /// Function name
    pub name: String,
    /// Basic blocks (ordered)
    pub blocks: IndexMap<String, MachineBlock>,
    /// Entry block label
    pub entry: String,
    /// Stack frame information
    pub frame: StackFrame,
    /// Next virtual register ID
    next_vreg: u32,
    /// Parameter virtual registers
    pub params: Vec<VReg>,
    /// Return value virtual register (if any)
    pub ret_vreg: Option<VReg>,
}

impl MachineFunction {
    /// Create a new function.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            name,
            blocks: IndexMap::new(),
            entry: String::new(),
            frame: StackFrame::default(),
            next_vreg: 0,
            params: Vec::new(),
            ret_vreg: None,
        }
    }

    /// Allocate a new virtual register.
    pub fn new_vreg(&mut self) -> VReg {
        let vreg = VReg::new(self.next_vreg);
        self.next_vreg += 1;
        vreg
    }

    /// Get the number of virtual registers allocated.
    pub fn num_vregs(&self) -> u32 {
        self.next_vreg
    }

    /// Get the current vreg counter value.
    pub fn vreg_count(&self) -> u32 {
        self.next_vreg
    }

    /// Set the vreg counter (used after external vreg allocation).
    pub fn set_vreg_count(&mut self, count: u32) {
        self.next_vreg = count;
    }

    /// Add a basic block.
    pub fn add_block(&mut self, block: MachineBlock) {
        if self.entry.is_empty() {
            self.entry = block.label.clone();
        }
        self.blocks.insert(block.label.clone(), block);
    }

    /// Get a block by label.
    pub fn get_block(&self, label: &str) -> Option<&MachineBlock> {
        self.blocks.get(label)
    }

    /// Get a mutable block by label.
    pub fn get_block_mut(&mut self, label: &str) -> Option<&mut MachineBlock> {
        self.blocks.get_mut(label)
    }

    /// Get the entry block.
    pub fn entry_block(&self) -> Option<&MachineBlock> {
        self.blocks.get(&self.entry)
    }

    /// Get the entry block mutably.
    pub fn entry_block_mut(&mut self) -> Option<&mut MachineBlock> {
        let entry = self.entry.clone();
        self.blocks.get_mut(&entry)
    }

    /// Iterate over all blocks.
    pub fn iter_blocks(&self) -> impl Iterator<Item = &MachineBlock> {
        self.blocks.values()
    }

    /// Iterate over all blocks mutably.
    pub fn iter_blocks_mut(&mut self) -> impl Iterator<Item = &mut MachineBlock> {
        self.blocks.values_mut()
    }

    /// Get block labels in order.
    pub fn block_labels(&self) -> Vec<&str> {
        self.blocks.keys().map(|s| s.as_str()).collect()
    }

    /// Update CFG edges (predecessors/successors) based on terminators.
    pub fn rebuild_cfg(&mut self) {
        // Collect edges first
        let mut edges: Vec<(String, String)> = Vec::new();

        for block in self.blocks.values() {
            if let Some(term) = block.terminator() {
                // Look for label operands
                for src in &term.srcs {
                    if let crate::mir::Operand::Label(target) = src {
                        edges.push((block.label.clone(), target.clone()));
                    }
                }
            }
        }

        // Clear existing edges
        for block in self.blocks.values_mut() {
            block.preds.clear();
            block.succs.clear();
        }

        // Add edges
        for (from, to) in edges {
            if let Some(from_block) = self.blocks.get_mut(&from) {
                from_block.add_succ(&to);
            }
            if let Some(to_block) = self.blocks.get_mut(&to) {
                to_block.add_pred(&from);
            }
        }
    }

    /// Get all virtual registers defined in this function.
    pub fn all_vregs(&self) -> Vec<VReg> {
        let mut vregs = Vec::new();
        for block in self.blocks.values() {
            for inst in &block.insts {
                if let Some(vreg) = inst.def() {
                    if !vregs.contains(&vreg) {
                        vregs.push(vreg);
                    }
                }
                for vreg in inst.uses() {
                    if !vregs.contains(&vreg) {
                        vregs.push(vreg);
                    }
                }
            }
        }
        vregs
    }
}

impl fmt::Display for MachineFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "function {}:", self.name)?;
        writeln!(f, "  ; {} vregs, entry: {}", self.next_vreg, self.entry)?;

        for block in self.blocks.values() {
            writeln!(f)?;
            write!(f, "{}", block)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MachineInst;

    #[test]
    fn test_function_basic() {
        let mut func = MachineFunction::new("test");

        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        assert_eq!(v0.id(), 0);
        assert_eq!(v1.id(), 1);
        assert_eq!(func.num_vregs(), 2);

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::add(v0, v0, v1));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        assert_eq!(func.entry, "entry");
        assert!(func.get_block("entry").is_some());
    }
}
