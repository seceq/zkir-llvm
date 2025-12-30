//! Liveness analysis for register allocation.
//!
//! Computes live ranges for each virtual register to guide allocation.

use crate::mir::{MachineFunction, VReg};
use std::collections::{HashMap, HashSet};

/// Live range for a virtual register.
#[derive(Debug, Clone)]
pub struct LiveRange {
    /// First use/def position
    pub start: usize,
    /// Last use position
    pub end: usize,
    /// Blocks where the register is live
    pub live_blocks: HashSet<String>,
}

impl LiveRange {
    /// Create a new live range.
    pub fn new(start: usize) -> Self {
        Self {
            start,
            end: start,
            live_blocks: HashSet::new(),
        }
    }

    /// Extend the range to include a new position.
    pub fn extend(&mut self, pos: usize) {
        self.start = self.start.min(pos);
        self.end = self.end.max(pos);
    }

    /// Check if this range overlaps with another.
    pub fn overlaps(&self, other: &LiveRange) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Get the length of this range.
    pub fn length(&self) -> usize {
        self.end - self.start + 1
    }
}

/// Liveness information for a function.
#[derive(Debug)]
pub struct LivenessInfo {
    /// Live ranges for each virtual register
    pub ranges: HashMap<VReg, LiveRange>,
    /// Registers live at each instruction
    pub live_at: Vec<HashSet<VReg>>,
    /// Registers live-in to each block
    pub live_in: HashMap<String, HashSet<VReg>>,
    /// Registers live-out of each block
    pub live_out: HashMap<String, HashSet<VReg>>,
}

impl LivenessInfo {
    /// Check if a register is live at a given position.
    pub fn is_live_at(&self, vreg: VReg, pos: usize) -> bool {
        if let Some(range) = self.ranges.get(&vreg) {
            pos >= range.start && pos <= range.end
        } else {
            false
        }
    }

    /// Get registers live at a position.
    pub fn live_at_position(&self, pos: usize) -> HashSet<VReg> {
        self.live_at.get(pos).cloned().unwrap_or_default()
    }

    /// Get all virtual registers used in the function.
    pub fn all_vregs(&self) -> Vec<VReg> {
        self.ranges.keys().copied().collect()
    }
}

/// Compute liveness information for a function.
pub fn compute_liveness(func: &MachineFunction) -> LivenessInfo {
    let mut ranges: HashMap<VReg, LiveRange> = HashMap::new();
    let mut live_in: HashMap<String, HashSet<VReg>> = HashMap::new();
    let mut live_out: HashMap<String, HashSet<VReg>> = HashMap::new();

    // Initialize empty sets for each block
    for label in func.block_labels() {
        live_in.insert(label.to_string(), HashSet::new());
        live_out.insert(label.to_string(), HashSet::new());
    }

    // Number instructions globally
    let mut inst_positions: HashMap<(String, usize), usize> = HashMap::new();
    let mut position = 0;

    for block in func.iter_blocks() {
        for (i, _inst) in block.insts.iter().enumerate() {
            inst_positions.insert((block.label.clone(), i), position);
            position += 1;
        }
    }

    let total_instructions = position;
    let mut live_at = vec![HashSet::new(); total_instructions];

    // Backward dataflow analysis
    let mut changed = true;
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    while changed {
        changed = false;

        // Process blocks in reverse order
        for label in block_labels.iter().rev() {
            let block = func.get_block(label).unwrap();

            // Start with live-out
            let mut live: HashSet<VReg> = live_out.get(label).cloned().unwrap_or_default();

            // Process instructions in reverse order
            for (i, inst) in block.insts.iter().enumerate().rev() {
                let pos = *inst_positions.get(&(label.clone(), i)).unwrap();

                // Record what's live at this position
                live_at[pos] = live.clone();

                // Remove defined register (it's not live before this point)
                if let Some(def) = inst.def() {
                    live.remove(&def);

                    // Update or create live range
                    if let Some(range) = ranges.get_mut(&def) {
                        range.extend(pos);
                        range.live_blocks.insert(label.clone());
                    } else {
                        let mut range = LiveRange::new(pos);
                        range.live_blocks.insert(label.clone());
                        ranges.insert(def, range);
                    }
                }

                // Add used registers (they're live before this point)
                for use_vreg in inst.uses() {
                    live.insert(use_vreg);

                    // Update live range
                    if let Some(range) = ranges.get_mut(&use_vreg) {
                        range.extend(pos);
                        range.live_blocks.insert(label.clone());
                    } else {
                        let mut range = LiveRange::new(pos);
                        range.live_blocks.insert(label.clone());
                        ranges.insert(use_vreg, range);
                    }
                }
            }

            // Update live-in
            let old_live_in = live_in.get(label).cloned().unwrap_or_default();
            if live != old_live_in {
                changed = true;
                live_in.insert(label.clone(), live.clone());

                // Propagate to predecessors' live-out
                for pred in &block.preds {
                    let pred_out = live_out.entry(pred.clone()).or_default();
                    for vreg in &live {
                        pred_out.insert(*vreg);
                    }
                }
            }
        }
    }

    // Post-process: extend ranges across blocks
    for (vreg, range) in ranges.iter_mut() {
        for label in &range.live_blocks.clone() {
            if let Some(li) = live_in.get(label) {
                if li.contains(vreg) {
                    // Register is live-in, extend range to start of block
                    if let Some(block) = func.get_block(label) {
                        if !block.insts.is_empty() {
                            if let Some(&pos) = inst_positions.get(&(label.clone(), 0)) {
                                range.start = range.start.min(pos);
                            }
                        }
                    }
                }
            }
            if let Some(lo) = live_out.get(label) {
                if lo.contains(vreg) {
                    // Register is live-out, extend range to end of block
                    if let Some(block) = func.get_block(label) {
                        let last_idx = block.insts.len().saturating_sub(1);
                        if let Some(&pos) = inst_positions.get(&(label.clone(), last_idx)) {
                            range.end = range.end.max(pos);
                        }
                    }
                }
            }
        }
    }

    LivenessInfo {
        ranges,
        live_at,
        live_in,
        live_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{MachineBlock, MachineInst};

    #[test]
    fn test_simple_liveness() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));      // pos 0: def v0
        entry.push(MachineInst::li(v1, 20));      // pos 1: def v1
        entry.push(MachineInst::add(v2, v0, v1)); // pos 2: def v2, use v0, v1
        entry.push(MachineInst::ret());            // pos 3
        func.add_block(entry);

        let liveness = compute_liveness(&func);

        // v0 is live from 0 to 2
        assert!(liveness.ranges.contains_key(&v0));
        assert_eq!(liveness.ranges[&v0].start, 0);
        assert_eq!(liveness.ranges[&v0].end, 2);

        // v1 is live from 1 to 2
        assert!(liveness.ranges.contains_key(&v1));
        assert_eq!(liveness.ranges[&v1].start, 1);
        assert_eq!(liveness.ranges[&v1].end, 2);

        // v2 is live at 2 only
        assert!(liveness.ranges.contains_key(&v2));
        assert_eq!(liveness.ranges[&v2].start, 2);
    }

    #[test]
    fn test_range_overlap() {
        let r1 = LiveRange { start: 0, end: 5, live_blocks: HashSet::new() };
        let r2 = LiveRange { start: 3, end: 8, live_blocks: HashSet::new() };
        let r3 = LiveRange { start: 6, end: 10, live_blocks: HashSet::new() };

        assert!(r1.overlaps(&r2)); // 0-5 overlaps 3-8
        assert!(!r1.overlaps(&r3)); // 0-5 doesn't overlap 6-10
        assert!(r2.overlaps(&r3)); // 3-8 overlaps 6-10
    }
}
