//! Machine basic blocks.

use super::instruction::MachineInst;
use std::fmt;

/// A machine basic block.
#[derive(Debug, Clone)]
pub struct MachineBlock {
    /// Block label/name
    pub label: String,
    /// Instructions in this block
    pub insts: Vec<MachineInst>,
    /// Predecessor block labels
    pub preds: Vec<String>,
    /// Successor block labels
    pub succs: Vec<String>,
}

impl MachineBlock {
    /// Create a new empty block.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            insts: Vec::new(),
            preds: Vec::new(),
            succs: Vec::new(),
        }
    }

    /// Add an instruction to the block.
    pub fn push(&mut self, inst: MachineInst) {
        self.insts.push(inst);
    }

    /// Insert an instruction at the given index.
    pub fn insert(&mut self, index: usize, inst: MachineInst) {
        self.insts.insert(index, inst);
    }

    /// Get the terminator instruction (last instruction if it's a terminator).
    pub fn terminator(&self) -> Option<&MachineInst> {
        self.insts.last().filter(|inst| inst.is_terminator())
    }

    /// Get a mutable reference to the terminator.
    pub fn terminator_mut(&mut self) -> Option<&mut MachineInst> {
        if self.insts.last().map(|i| i.is_terminator()).unwrap_or(false) {
            self.insts.last_mut()
        } else {
            None
        }
    }

    /// Check if this block has a terminator.
    pub fn has_terminator(&self) -> bool {
        self.terminator().is_some()
    }

    /// Is this block empty?
    pub fn is_empty(&self) -> bool {
        self.insts.is_empty()
    }

    /// Number of instructions.
    pub fn len(&self) -> usize {
        self.insts.len()
    }

    /// Iterate over instructions.
    pub fn iter(&self) -> impl Iterator<Item = &MachineInst> {
        self.insts.iter()
    }

    /// Iterate over instructions mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut MachineInst> {
        self.insts.iter_mut()
    }

    /// Add a predecessor.
    pub fn add_pred(&mut self, label: impl Into<String>) {
        let label = label.into();
        if !self.preds.contains(&label) {
            self.preds.push(label);
        }
    }

    /// Add a successor.
    pub fn add_succ(&mut self, label: impl Into<String>) {
        let label = label.into();
        if !self.succs.contains(&label) {
            self.succs.push(label);
        }
    }
}

impl fmt::Display for MachineBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.label)?;
        for inst in &self.insts {
            writeln!(f, "    {}", inst)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::VReg;

    #[test]
    fn test_block_basic() {
        let mut block = MachineBlock::new("entry");
        assert!(block.is_empty());

        block.push(MachineInst::add(VReg(0), VReg(1), VReg(2)));
        assert_eq!(block.len(), 1);
        assert!(!block.has_terminator());

        block.push(MachineInst::ret());
        assert!(block.has_terminator());
    }
}
