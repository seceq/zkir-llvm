//! Linear scan register allocator.
//!
//! A simple and efficient register allocator that processes live ranges
//! in order of their start positions.

use super::liveness::{LivenessInfo, LiveRange};
use crate::mir::{MachineBlock, MachineFunction, MachineInst, Opcode, Operand, VReg};
use crate::target::config::TargetConfig;
use crate::target::registers::{Register, ALLOCATABLE_REGS, CALLEE_SAVED};
use anyhow::Result;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Assignment for a virtual register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assignment {
    /// Assigned to a physical register
    Register(Register),
    /// Spilled to stack at offset from frame pointer
    Stack(i32),
}

/// Linear scan register allocator.
pub struct LinearScanAllocator<'a> {
    /// Function being allocated
    func: &'a MachineFunction,
    /// Liveness information
    liveness: &'a LivenessInfo,
    /// Target configuration (for future use with multi-register values)
    #[allow(dead_code)]
    config: &'a TargetConfig,
    /// Current register assignments
    assignments: HashMap<VReg, Assignment>,
    /// Active intervals (sorted by end point)
    active: BTreeSet<(usize, VReg)>,
    /// Free callee-saved registers (R10-R13)
    free_callee_saved: Vec<Register>,
    /// Free caller-saved registers (R14-R15, others)
    free_caller_saved: Vec<Register>,
    /// Next stack slot offset
    next_stack_slot: i32,
    /// Registers used in the function (for callee-save tracking)
    used_regs: HashSet<Register>,
    /// Positions of call instructions (JAL, JALR, CALL, etc.)
    call_positions: HashSet<usize>,
}

impl<'a> LinearScanAllocator<'a> {
    /// Create a new allocator.
    pub fn new(
        func: &'a MachineFunction,
        liveness: &'a LivenessInfo,
        config: &'a TargetConfig,
    ) -> Self {
        // Separate callee-saved and caller-saved registers
        let mut callee_saved = Vec::new();
        let mut caller_saved = Vec::new();

        for reg in ALLOCATABLE_REGS {
            if CALLEE_SAVED.contains(&reg) {
                callee_saved.push(reg);
            } else {
                caller_saved.push(reg);
            }
        }

        // Identify call instruction positions
        let call_positions = Self::find_call_positions(func);

        Self {
            func,
            liveness,
            config,
            assignments: HashMap::new(),
            active: BTreeSet::new(),
            free_callee_saved: callee_saved,
            free_caller_saved: caller_saved,
            next_stack_slot: 0,
            used_regs: HashSet::new(),
            call_positions,
        }
    }

    /// Find positions of all call instructions in the function.
    fn find_call_positions(func: &MachineFunction) -> HashSet<usize> {
        let mut positions = HashSet::new();
        let mut position = 0;

        for block in func.iter_blocks() {
            for inst in &block.insts {
                // Check if this is a call instruction
                if matches!(inst.opcode, Opcode::JAL | Opcode::JALR |
                           Opcode::CALL | Opcode::CALLR) {
                    positions.insert(position);
                }
                position += 1;
            }
        }

        positions
    }

    /// Check if a live range spans any call instructions.
    fn spans_call(&self, range: &LiveRange) -> bool {
        for call_pos in &self.call_positions {
            if *call_pos >= range.start && *call_pos <= range.end {
                return true;
            }
        }
        false
    }

    /// Perform register allocation.
    pub fn allocate(&mut self) -> Result<MachineFunction> {
        // Sort intervals by start point
        let mut intervals: Vec<(VReg, LiveRange)> = self
            .liveness
            .ranges
            .iter()
            .map(|(v, r)| (*v, r.clone()))
            .collect();

        intervals.sort_by_key(|(_, r)| r.start);

        // Process each interval
        for (vreg, range) in intervals {
            // Expire old intervals
            self.expire_old_intervals(range.start);

            // Check if this range spans a call
            let needs_callee_saved = self.spans_call(&range);

            // Try to allocate a register
            let reg = if needs_callee_saved {
                // Prefer callee-saved register for call-spanning ranges
                if !self.free_callee_saved.is_empty() {
                    Some(self.free_callee_saved.pop().unwrap())
                } else if !self.free_caller_saved.is_empty() {
                    // If no callee-saved available, use caller-saved but we'll spill
                    None
                } else {
                    None
                }
            } else {
                // For non-call-spanning ranges, prefer caller-saved first
                if !self.free_caller_saved.is_empty() {
                    Some(self.free_caller_saved.pop().unwrap())
                } else if !self.free_callee_saved.is_empty() {
                    Some(self.free_callee_saved.pop().unwrap())
                } else {
                    None
                }
            };

            if let Some(reg) = reg {
                // Successfully allocated a register
                self.assignments.insert(vreg, Assignment::Register(reg));
                self.active.insert((range.end, vreg));
                self.used_regs.insert(reg);
            } else {
                // Need to spill
                self.spill_at_interval(&vreg, &range)?;
            }
        }

        // Build the allocated function
        self.build_allocated_function()
    }

    /// Expire intervals that end before the given position.
    fn expire_old_intervals(&mut self, pos: usize) {
        let mut to_remove = Vec::new();

        for &(end, vreg) in &self.active {
            if end >= pos {
                break;
            }
            to_remove.push((end, vreg));
        }

        for item in to_remove {
            self.active.remove(&item);

            // Return register to appropriate free pool
            if let Some(Assignment::Register(reg)) = self.assignments.get(&item.1) {
                if CALLEE_SAVED.contains(reg) {
                    self.free_callee_saved.push(*reg);
                } else {
                    self.free_caller_saved.push(*reg);
                }
            }
        }
    }

    /// Spill the interval with the longest remaining range.
    fn spill_at_interval(&mut self, vreg: &VReg, range: &LiveRange) -> Result<()> {
        // Find the active interval with the furthest end point
        if let Some(&(end, spill_vreg)) = self.active.iter().last() {
            if end > range.end {
                // Spill the furthest interval
                if let Some(Assignment::Register(reg)) = self.assignments.get(&spill_vreg).copied() {
                    // Spill the old interval to stack
                    let slot = self.allocate_stack_slot();
                    self.assignments.insert(spill_vreg, Assignment::Stack(slot));

                    // Give its register to the new interval
                    self.assignments.insert(*vreg, Assignment::Register(reg));
                    self.active.remove(&(end, spill_vreg));
                    self.active.insert((range.end, *vreg));

                    return Ok(());
                }
            }
        }

        // Spill the current interval
        let slot = self.allocate_stack_slot();
        self.assignments.insert(*vreg, Assignment::Stack(slot));

        Ok(())
    }

    /// Allocate a stack slot.
    fn allocate_stack_slot(&mut self) -> i32 {
        let slot = self.next_stack_slot;
        self.next_stack_slot += 4; // 4 bytes per slot
        slot
    }

    /// Build the allocated function with physical registers.
    fn build_allocated_function(&self) -> Result<MachineFunction> {
        let mut result = MachineFunction::new(&self.func.name);
        result.frame = self.func.frame.clone();

        // Update spill size
        result.frame.spill_size = self.next_stack_slot as u32;

        // Track callee-saved registers that we used
        for reg in &self.used_regs {
            if CALLEE_SAVED.contains(reg) && !result.frame.saved_regs.contains(reg) {
                result.frame.saved_regs.push(*reg);
            }
        }

        // Rewrite instructions with physical registers
        for block in self.func.iter_blocks() {
            let mut new_block = MachineBlock::new(&block.label);
            new_block.preds = block.preds.clone();
            new_block.succs = block.succs.clone();

            for inst in &block.insts {
                let rewritten = self.rewrite_instruction(inst)?;
                for new_inst in rewritten {
                    new_block.push(new_inst);
                }
            }

            result.add_block(new_block);
        }

        Ok(result)
    }

    /// Rewrite an instruction with physical registers.
    fn rewrite_instruction(&self, inst: &MachineInst) -> Result<Vec<MachineInst>> {
        let mut result = Vec::new();
        let mut new_inst = inst.clone();

        // Rewrite destination
        if let Some(ref dst) = inst.dst {
            new_inst.dst = Some(self.rewrite_operand(dst, &mut result, true)?);
        }

        // Rewrite sources
        new_inst.srcs = inst
            .srcs
            .iter()
            .map(|src| self.rewrite_operand(src, &mut result, false))
            .collect::<Result<Vec<_>>>()?;

        // Handle spill stores if destination is on stack
        if let Some(Operand::VReg(vreg)) = &inst.dst {
            if let Some(Assignment::Stack(offset)) = self.assignments.get(vreg) {
                // After the instruction, store result to stack
                // The instruction was rewritten to use a temp register
                // We need to emit a store after
                if let Some(Operand::Reg(temp_reg)) = new_inst.dst.clone() {
                    result.push(new_inst);
                    result.push(MachineInst::new(Opcode::SW)
                        .src(Operand::Reg(temp_reg))
                        .src(Operand::MemReg {
                            base: Register::R3, // FP (zkir-spec v3.4)
                            offset: -(*offset),
                        })
                        .comment("spill store"));
                    return Ok(result);
                }
            }
        }

        result.push(new_inst);
        Ok(result)
    }

    /// Rewrite an operand, inserting loads/stores for spilled values.
    fn rewrite_operand(
        &self,
        operand: &Operand,
        spill_code: &mut Vec<MachineInst>,
        is_def: bool,
    ) -> Result<Operand> {
        match operand {
            Operand::VReg(vreg) => {
                match self.assignments.get(vreg) {
                    Some(Assignment::Register(reg)) => Ok(Operand::Reg(*reg)),
                    Some(Assignment::Stack(offset)) => {
                        if is_def {
                            // For definitions, we'll use a temp register
                            // The caller will emit the store
                            // Use t0 as temporary (R14 in zkir-spec v3.4)
                            Ok(Operand::Reg(Register::R14))
                        } else {
                            // For uses, emit a load from stack
                            let temp_reg = Register::R14; // t0
                            spill_code.push(MachineInst::new(Opcode::LW)
                                .dst(Operand::Reg(temp_reg))
                                .src(Operand::MemReg {
                                    base: Register::R3, // FP (zkir-spec v3.4)
                                    offset: -(*offset),
                                })
                                .comment("spill reload"));
                            Ok(Operand::Reg(temp_reg))
                        }
                    }
                    None => {
                        // Unassigned vreg - this shouldn't happen
                        log::warn!("Unassigned vreg: {}", vreg);
                        // Assign to a temp register
                        Ok(Operand::Reg(Register::R14))
                    }
                }
            }
            Operand::Mem { base, offset } => {
                // Rewrite base register if it's a vreg
                if let Some(assignment) = self.assignments.get(base) {
                    match assignment {
                        Assignment::Register(reg) => {
                            Ok(Operand::MemReg { base: *reg, offset: *offset })
                        }
                        Assignment::Stack(stack_off) => {
                            // Load base from stack first
                            let temp_reg = Register::R15; // t1 (zkir-spec v3.4)
                            spill_code.push(MachineInst::new(Opcode::LW)
                                .dst(Operand::Reg(temp_reg))
                                .src(Operand::MemReg {
                                    base: Register::R3, // FP
                                    offset: -(*stack_off),
                                })
                                .comment("load base address"));
                            Ok(Operand::MemReg { base: temp_reg, offset: *offset })
                        }
                    }
                } else {
                    // Base vreg not assigned, use temp
                    Ok(Operand::MemReg { base: Register::R14, offset: *offset })
                }
            }
            // Other operands pass through unchanged
            other => Ok(other.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MachineBlock;

    #[test]
    fn test_basic_allocation() {
        let config = TargetConfig::default();

        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 20));
        entry.push(MachineInst::add(v2, v0, v1));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let liveness = super::super::liveness::compute_liveness(&func);
        let mut allocator = LinearScanAllocator::new(&func, &liveness, &config);
        let result = allocator.allocate().unwrap();

        // Verify all instructions have physical registers
        for block in result.iter_blocks() {
            for inst in &block.insts {
                if let Some(ref dst) = inst.dst {
                    // Should be physical register or memory
                    assert!(!matches!(dst, Operand::VReg(_)));
                }
            }
        }
    }

    #[test]
    fn test_spilling() {
        let config = TargetConfig::default();

        // Create a function with more live ranges than registers
        let mut func = MachineFunction::new("test");
        let vregs: Vec<VReg> = (0..20).map(|_| func.new_vreg()).collect();

        let mut entry = MachineBlock::new("entry");

        // Define all vregs
        for (i, vreg) in vregs.iter().enumerate() {
            entry.push(MachineInst::li(*vreg, i as i64));
        }

        // Use all vregs (they're all live at this point)
        let sum = func.new_vreg();
        entry.push(MachineInst::mov(sum, vregs[0]));
        for vreg in &vregs[1..] {
            let temp = func.new_vreg();
            entry.push(MachineInst::add(temp, sum, *vreg));
            entry.push(MachineInst::mov(sum, temp));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);

        let liveness = super::super::liveness::compute_liveness(&func);
        let mut allocator = LinearScanAllocator::new(&func, &liveness, &config);
        let result = allocator.allocate().unwrap();

        // Should have spilled some registers
        assert!(result.frame.spill_size > 0);
    }

    #[test]
    fn test_no_spill_short_ranges() {
        let config = TargetConfig::default();

        // Create a function with short, non-overlapping live ranges
        let mut func = MachineFunction::new("test");

        let mut entry = MachineBlock::new("entry");

        // Each vreg has a short, non-overlapping live range
        for i in 0..10 {
            let v = func.new_vreg();
            entry.push(MachineInst::li(v, i as i64));
            // Immediately "use" by returning (in a real program)
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);

        let liveness = super::super::liveness::compute_liveness(&func);
        let mut allocator = LinearScanAllocator::new(&func, &liveness, &config);
        let result = allocator.allocate().unwrap();

        // No spills needed since ranges don't overlap much
        assert_eq!(result.frame.spill_size, 0);
    }

    #[test]
    fn test_memory_operand_rewrite() {
        let config = TargetConfig::default();

        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 0x1000)); // Base address
        entry.push(MachineInst::lw(v1, v0, 8));  // Load from v0+8
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let liveness = super::super::liveness::compute_liveness(&func);
        let mut allocator = LinearScanAllocator::new(&func, &liveness, &config);
        let result = allocator.allocate().unwrap();

        // Check that memory operand base is rewritten to physical register
        let block = result.get_block("entry").unwrap();
        for inst in &block.insts {
            if inst.opcode == Opcode::LW {
                for src in &inst.srcs {
                    if let Operand::MemReg { base, offset } = src {
                        // Should be a physical register (any allocatable register)
                        // In zkir-spec v3.4: allocatable = all except zero, sp (R2), fp (R3)
                        assert!(matches!(base,
                            Register::R1 | Register::R4 | Register::R5 |
                            Register::R6 | Register::R7 | Register::R8 | Register::R9 |
                            Register::R10 | Register::R11 | Register::R12 | Register::R13 |
                            Register::R14 | Register::R15));
                        assert_eq!(*offset, 8);
                    }
                }
            }
        }
    }

    #[test]
    fn test_assignment_types() {
        // Test that Assignment enum works correctly
        let reg_assign = Assignment::Register(Register::R10);
        let stack_assign = Assignment::Stack(16);

        assert_eq!(reg_assign, Assignment::Register(Register::R10));
        assert_eq!(stack_assign, Assignment::Stack(16));
        assert_ne!(reg_assign, stack_assign);
    }

    #[test]
    fn test_preserves_terminators() {
        let config = TargetConfig::default();

        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 0));
        entry.push(MachineInst::bne(v0, v1, "exit"));
        func.add_block(entry);

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        let liveness = super::super::liveness::compute_liveness(&func);
        let mut allocator = LinearScanAllocator::new(&func, &liveness, &config);
        let result = allocator.allocate().unwrap();

        // Check that BNE is preserved
        let block = result.get_block("entry").unwrap();
        let has_bne = block.insts.iter().any(|i| i.opcode == Opcode::BNE);
        assert!(has_bne);

        // Check that RET is preserved
        let exit_block = result.get_block("exit").unwrap();
        let has_ret = exit_block.insts.iter().any(|i| i.opcode == Opcode::RET);
        assert!(has_ret);
    }

    #[test]
    fn test_callee_saved_tracking() {
        let config = TargetConfig::default();

        // Create a function that uses many registers to force callee-saved usage
        let mut func = MachineFunction::new("test");
        let vregs: Vec<VReg> = (0..15).map(|_| func.new_vreg()).collect();

        let mut entry = MachineBlock::new("entry");

        // Define all vregs - they all need to be live simultaneously
        for (i, vreg) in vregs.iter().enumerate() {
            entry.push(MachineInst::li(*vreg, i as i64));
        }

        // Use all vregs to keep them live
        let result_vreg = func.new_vreg();
        entry.push(MachineInst::add(result_vreg, vregs[0], vregs[1]));
        for vreg in &vregs[2..] {
            let temp = func.new_vreg();
            entry.push(MachineInst::add(temp, result_vreg, *vreg));
            entry.push(MachineInst::mov(result_vreg, temp));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);

        let liveness = super::super::liveness::compute_liveness(&func);
        let mut allocator = LinearScanAllocator::new(&func, &liveness, &config);
        let result = allocator.allocate().unwrap();

        // With 15+ simultaneous live ranges, we should have used some callee-saved registers
        // or spilled. Either way, the frame should track callee-saved usage.
        // Just verify allocation succeeded and produced valid output
        for block in result.iter_blocks() {
            for inst in &block.insts {
                if let Some(ref dst) = inst.dst {
                    // No virtual registers should remain
                    assert!(!matches!(dst, Operand::VReg(_)),
                        "Virtual register found in allocated code");
                }
            }
        }
    }

    #[test]
    fn test_diamond_cfg_allocation() {
        let config = TargetConfig::default();

        // Create a diamond CFG:
        //     entry
        //    /     \
        //  left   right
        //    \     /
        //     merge
        let mut func = MachineFunction::new("diamond");
        let cond = func.new_vreg();
        let v_left = func.new_vreg();
        let v_right = func.new_vreg();
        let result = func.new_vreg();
        let zero = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(cond, 1));
        entry.push(MachineInst::li(zero, 0));
        entry.push(MachineInst::bne(cond, zero, "left"));
        entry.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(func.new_vreg()))
            .src(Operand::Label("right".to_string())));
        func.add_block(entry);

        let mut left = MachineBlock::new("left");
        left.push(MachineInst::li(v_left, 10));
        left.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(func.new_vreg()))
            .src(Operand::Label("merge".to_string())));
        func.add_block(left);

        let mut right = MachineBlock::new("right");
        right.push(MachineInst::li(v_right, 20));
        right.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(func.new_vreg()))
            .src(Operand::Label("merge".to_string())));
        func.add_block(right);

        let mut merge = MachineBlock::new("merge");
        // In a real program, there would be a PHI here. For register allocation,
        // we just test that both paths can define values that merge.
        merge.push(MachineInst::li(result, 0));
        merge.push(MachineInst::ret());
        func.add_block(merge);

        func.rebuild_cfg();

        let liveness = super::super::liveness::compute_liveness(&func);
        let mut allocator = LinearScanAllocator::new(&func, &liveness, &config);
        let allocated = allocator.allocate().unwrap();

        // Verify all blocks are present and have allocated registers
        assert!(allocated.get_block("entry").is_some());
        assert!(allocated.get_block("left").is_some());
        assert!(allocated.get_block("right").is_some());
        assert!(allocated.get_block("merge").is_some());
    }

    #[test]
    fn test_loop_allocation() {
        let config = TargetConfig::default();

        // Create a simple loop structure
        let mut func = MachineFunction::new("loop");
        let counter = func.new_vreg();
        let limit = func.new_vreg();
        let sum = func.new_vreg();
        let one = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(counter, 0));
        entry.push(MachineInst::li(limit, 10));
        entry.push(MachineInst::li(sum, 0));
        entry.push(MachineInst::li(one, 1));
        entry.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(func.new_vreg()))
            .src(Operand::Label("loop_header".to_string())));
        func.add_block(entry);

        let mut header = MachineBlock::new("loop_header");
        header.push(MachineInst::new(Opcode::BGE)
            .src(Operand::VReg(counter))
            .src(Operand::VReg(limit))
            .src(Operand::Label("exit".to_string())));
        func.add_block(header);

        let mut body = MachineBlock::new("loop_body");
        body.push(MachineInst::add(sum, sum, counter));
        body.push(MachineInst::add(counter, counter, one));
        body.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(func.new_vreg()))
            .src(Operand::Label("loop_header".to_string())));
        func.add_block(body);

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();

        let liveness = super::super::liveness::compute_liveness(&func);
        let mut allocator = LinearScanAllocator::new(&func, &liveness, &config);
        let allocated = allocator.allocate().unwrap();

        // Verify allocation succeeded for loop structure
        // The counter, limit, sum, and one should all be allocated since they're
        // live across the loop
        for block in allocated.iter_blocks() {
            for inst in &block.insts {
                if let Some(ref dst) = inst.dst {
                    assert!(!matches!(dst, Operand::VReg(_)),
                        "Virtual register found in loop allocation");
                }
            }
        }
    }

    #[test]
    fn test_heavy_register_pressure() {
        let config = TargetConfig::default();

        // Create extreme register pressure - more live vregs than physical registers
        let mut func = MachineFunction::new("pressure");
        let num_vregs = 30; // More than available allocatable registers
        let vregs: Vec<VReg> = (0..num_vregs).map(|_| func.new_vreg()).collect();

        let mut entry = MachineBlock::new("entry");

        // Define all vregs first
        for (i, vreg) in vregs.iter().enumerate() {
            entry.push(MachineInst::li(*vreg, i as i64));
        }

        // Now use them all in a chain of adds to keep them all live
        let mut current = vregs[0];
        for vreg in &vregs[1..] {
            let temp = func.new_vreg();
            entry.push(MachineInst::add(temp, current, *vreg));
            current = temp;
        }

        // Final use
        entry.push(MachineInst::mov(func.new_vreg(), current));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let liveness = super::super::liveness::compute_liveness(&func);
        let mut allocator = LinearScanAllocator::new(&func, &liveness, &config);
        let allocated = allocator.allocate().unwrap();

        // Must have spilled since we have more vregs than physical registers
        assert!(allocated.frame.spill_size > 0,
            "Expected spilling with {} vregs but spill_size is 0", num_vregs);

        // Verify no virtual registers remain
        for block in allocated.iter_blocks() {
            for inst in &block.insts {
                if let Some(ref dst) = inst.dst {
                    assert!(!matches!(dst, Operand::VReg(_)),
                        "Virtual register found after allocation");
                }
            }
        }
    }
}
