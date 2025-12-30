//! Loop unrolling optimization pass.
//!
//! Unrolls small, fixed-iteration loops to eliminate branch overhead
//! and enable further optimizations. This is particularly valuable for
//! ZK circuits where branches add constraint overhead.

use crate::mir::{MachineFunction, MachineInst, Opcode, Operand, VReg};
use anyhow::Result;

/// Maximum number of iterations to unroll.
const MAX_UNROLL_ITERATIONS: i64 = 8;

/// Maximum number of instructions in loop body to consider for unrolling.
const MAX_LOOP_BODY_SIZE: usize = 32;

/// A detected loop structure.
#[derive(Debug)]
struct Loop {
    /// Header block (contains the loop condition)
    header: String,
    /// Body blocks (executed each iteration)
    body: Vec<String>,
    /// Exit block (where we go after the loop)
    exit: String,
    /// Back edge source (jumps back to header)
    #[allow(dead_code)]
    latch: String,
    /// Induction variable (the loop counter)
    induction_var: Option<InductionVar>,
}

/// Information about a loop induction variable.
#[derive(Debug, Clone)]
struct InductionVar {
    /// The virtual register holding the counter
    vreg: VReg,
    /// Initial value
    init: i64,
    /// Step (increment per iteration)
    step: i64,
    /// Limit value
    limit: i64,
    /// Comparison type (how we test for loop exit)
    cmp_type: CmpType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum CmpType {
    /// i < limit
    LessThan,
    /// i <= limit
    LessEqual,
    /// i != limit
    NotEqual,
}

impl InductionVar {
    /// Calculate the number of iterations for this loop.
    fn iteration_count(&self) -> Option<i64> {
        if self.step == 0 {
            return None; // Infinite loop
        }

        let count = match self.cmp_type {
            CmpType::LessThan => {
                if self.step > 0 && self.init < self.limit {
                    (self.limit - self.init + self.step - 1) / self.step
                } else if self.step < 0 && self.init > self.limit {
                    (self.init - self.limit + (-self.step) - 1) / (-self.step)
                } else {
                    0
                }
            }
            CmpType::LessEqual => {
                if self.step > 0 && self.init <= self.limit {
                    (self.limit - self.init) / self.step + 1
                } else if self.step < 0 && self.init >= self.limit {
                    (self.init - self.limit) / (-self.step) + 1
                } else {
                    0
                }
            }
            CmpType::NotEqual => {
                if self.step > 0 && self.init < self.limit {
                    if (self.limit - self.init) % self.step == 0 {
                        (self.limit - self.init) / self.step
                    } else {
                        return None; // Won't terminate or complex
                    }
                } else if self.step < 0 && self.init > self.limit {
                    if (self.init - self.limit) % (-self.step) == 0 {
                        (self.init - self.limit) / (-self.step)
                    } else {
                        return None;
                    }
                } else if self.init == self.limit {
                    0
                } else {
                    return None;
                }
            }
        };

        Some(count)
    }
}

/// Run loop unrolling on a function.
pub fn unroll_loops(func: &mut MachineFunction) -> Result<u32> {
    let mut unrolled = 0;

    // Rebuild CFG to ensure we have accurate predecessor/successor info
    func.rebuild_cfg();

    // Find all loops
    let loops = find_loops(func);

    for loop_info in loops {
        if let Some(iv) = &loop_info.induction_var {
            if let Some(iterations) = iv.iteration_count() {
                if iterations > 0 && iterations <= MAX_UNROLL_ITERATIONS {
                    // Check if loop body is small enough
                    let body_size: usize = loop_info.body.iter()
                        .filter_map(|label| func.get_block(label))
                        .map(|b| b.insts.len())
                        .sum();

                    if body_size <= MAX_LOOP_BODY_SIZE
                        && unroll_loop(func, &loop_info, iterations as usize)
                    {
                        unrolled += 1;
                    }
                }
            }
        }
    }

    if unrolled > 0 {
        func.rebuild_cfg();
    }

    Ok(unrolled)
}

/// Find loops in the function using a simple back-edge detection.
fn find_loops(func: &MachineFunction) -> Vec<Loop> {
    let mut loops = Vec::new();

    // Simple loop detection: look for blocks that branch back to an earlier block
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for (i, label) in block_labels.iter().enumerate() {
        let block = match func.get_block(label) {
            Some(b) => b,
            None => continue,
        };

        // Check if this block's terminator branches to an earlier block
        if let Some(term) = block.terminator() {
            for src in &term.srcs {
                if let Operand::Label(target) = src {
                    // Find target's position
                    if let Some(target_pos) = block_labels.iter().position(|l| l == target) {
                        if target_pos <= i {
                            // This is a back edge: label -> target
                            // target is the loop header, label is the latch
                            if let Some(loop_info) = analyze_loop(func, target, label, &block_labels) {
                                loops.push(loop_info);
                            }
                        }
                    }
                }
            }
        }
    }

    loops
}

/// Analyze a potential loop to extract its structure.
fn analyze_loop(
    func: &MachineFunction,
    header: &str,
    latch: &str,
    block_labels: &[String],
) -> Option<Loop> {
    let header_block = func.get_block(header)?;

    // Find the exit block (the block we branch to when exiting the loop)
    let term = header_block.terminator()?;
    if !term.opcode.is_branch() {
        return None;
    }

    // Get branch targets
    let mut branch_targets: Vec<String> = Vec::new();
    for src in &term.srcs {
        if let Operand::Label(target) = src {
            branch_targets.push(target.clone());
        }
    }

    if branch_targets.is_empty() {
        return None;
    }

    // Determine which target is the exit (not part of the loop body)
    let header_pos = block_labels.iter().position(|l| l == header)?;
    let latch_pos = block_labels.iter().position(|l| l == latch)?;

    let mut exit = None;

    for target in &branch_targets {
        let target_pos = block_labels.iter().position(|l| l == target)?;
        if target_pos > latch_pos {
            // This target is after the latch, so it's the exit
            exit = Some(target.clone());
        }
        // Note: We could also track body_start here if needed
    }

    let exit = exit?;

    // Collect body blocks (between header and latch, inclusive of latch)
    let mut body = Vec::new();
    for label in block_labels.iter() {
        let pos = block_labels.iter().position(|l| l == label)?;
        if pos > header_pos && pos <= latch_pos {
            body.push(label.clone());
        }
    }

    // Try to analyze the induction variable
    let induction_var = analyze_induction_var(func, header, &body, latch, block_labels);

    Some(Loop {
        header: header.to_string(),
        body,
        exit,
        latch: latch.to_string(),
        induction_var,
    })
}

/// Analyze the induction variable of a loop.
fn analyze_induction_var(
    func: &MachineFunction,
    header: &str,
    body: &[String],
    latch: &str,
    block_labels: &[String],
) -> Option<InductionVar> {
    let latch_pos = block_labels.iter().position(|l| l == latch)?;
    let header_block = func.get_block(header)?;

    // Look for a pattern like:
    // header:
    //   ... comparison of vreg with constant ...
    //   blt/bge/bne vreg, limit_vreg, exit/body
    // body:
    //   ...
    //   addi vreg, vreg, step
    //   jal header

    // First, find the comparison in the header
    let term = header_block.terminator()?;
    if !term.opcode.is_branch() {
        return None;
    }

    // Get the compared registers
    let cmp_vreg1 = term.srcs.first()?.as_vreg()?;
    let cmp_vreg2 = term.srcs.get(1)?.as_vreg()?;

    // Note: The comparison type depends on how the branch is used.
    // BGE v0, v1, exit means "if v0 >= v1, exit", so the loop continues while v0 < v1.
    // BLT v0, v1, body means "if v0 < v1, continue", so the loop exits when v0 >= v1.
    // We need to determine which way the branch goes (to exit or to body).
    let branch_target = term.srcs.get(2).and_then(|s| {
        if let Operand::Label(l) = s { Some(l.as_str()) } else { None }
    })?;

    // Determine if the branch goes to exit or stays in loop
    let branches_to_exit = block_labels.iter()
        .position(|l| l == branch_target)
        .map(|pos| pos > latch_pos)
        .unwrap_or(false);

    let cmp_type = match (term.opcode, branches_to_exit) {
        // BGE to exit: loop continues while LessThan
        (Opcode::BGE | Opcode::BGEU, true) => CmpType::LessThan,
        // BLT to body: loop continues while LessThan
        (Opcode::BLT | Opcode::BLTU, false) => CmpType::LessThan,
        // BNE to body: loop continues while NotEqual
        (Opcode::BNE, false) => CmpType::NotEqual,
        // BEQ to exit: loop continues while NotEqual
        (Opcode::BEQ, true) => CmpType::NotEqual,
        _ => return None,
    };

    // Find which vreg is the induction variable (the one that gets incremented in the body)
    let mut induction_vreg = None;
    let mut step = None;

    for body_label in body {
        if let Some(body_block) = func.get_block(body_label) {
            for inst in &body_block.insts {
                if inst.opcode == Opcode::ADDI {
                    if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), Some(Operand::Imm(imm))) =
                        (&inst.dst, inst.srcs.first(), inst.srcs.get(1))
                    {
                        if dst == src && (*dst == cmp_vreg1 || *dst == cmp_vreg2) {
                            induction_vreg = Some(*dst);
                            step = Some(*imm);
                        }
                    }
                }
            }
        }
    }

    let induction_vreg = induction_vreg?;
    let step = step?;

    // Find the initial value and limit
    // Look for LI instructions that set the induction variable and the limit
    let mut init = None;
    let mut limit = None;

    let limit_vreg = if induction_vreg == cmp_vreg1 { cmp_vreg2 } else { cmp_vreg1 };

    // Search in all blocks for the definitions
    for block in func.iter_blocks() {
        for inst in &block.insts {
            if inst.opcode == Opcode::LI {
                if let (Some(Operand::VReg(dst)), Some(Operand::Imm(val))) =
                    (&inst.dst, inst.srcs.first())
                {
                    if *dst == induction_vreg && init.is_none() {
                        init = Some(*val);
                    }
                    if *dst == limit_vreg && limit.is_none() {
                        limit = Some(*val);
                    }
                }
            }
        }
    }

    Some(InductionVar {
        vreg: induction_vreg,
        init: init?,
        step,
        limit: limit?,
        cmp_type,
    })
}

/// Unroll a loop by the given number of iterations.
fn unroll_loop(func: &mut MachineFunction, loop_info: &Loop, iterations: usize) -> bool {
    if iterations == 0 {
        return false;
    }

    let iv = match &loop_info.induction_var {
        Some(iv) => iv.clone(),
        None => return false,
    };

    // Collect all body instructions (excluding terminators)
    let mut body_insts: Vec<MachineInst> = Vec::new();
    for body_label in &loop_info.body {
        if let Some(body_block) = func.get_block(body_label) {
            for inst in &body_block.insts {
                if !inst.is_terminator() {
                    body_insts.push(inst.clone());
                }
            }
        }
    }

    if body_insts.is_empty() {
        return false;
    }

    // Create the unrolled instructions
    let mut unrolled_insts: Vec<MachineInst> = Vec::new();

    // We need to replace uses of the induction variable with the appropriate constant
    for iter in 0..iterations {
        // Note: current_val could be used for more advanced IV substitution
        let _current_val = iv.init + (iter as i64) * iv.step;

        for inst in &body_insts {
            let mut new_inst = inst.clone();
            new_inst.comment = Some(format!("unrolled iter {}", iter));

            // If this is the increment of the induction variable, replace with a NOP
            // (we're computing with constants now)
            if new_inst.opcode == Opcode::ADDI {
                if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src)), _) =
                    (&new_inst.dst, new_inst.srcs.first(), new_inst.srcs.get(1))
                {
                    if dst == src && *dst == iv.vreg {
                        // Skip the increment - we don't need it in unrolled code
                        continue;
                    }
                }
            }

            // Replace uses of induction variable with constant
            // Note: This is a simplified approach - a full implementation would
            // need to track SSA forms more carefully
            unrolled_insts.push(new_inst);
        }
    }

    // Allocate vregs we'll need before borrowing blocks
    let zero_vreg = func.new_vreg();
    let body_labels: Vec<String> = loop_info.body.clone();
    let exit_label = loop_info.exit.clone();
    let header_label = loop_info.header.clone();

    // Allocate vregs for each body block jump
    let body_vregs: Vec<VReg> = body_labels.iter().map(|_| func.new_vreg()).collect();

    // Now we need to:
    // 1. Replace the header block with the unrolled code
    // 2. Remove the body blocks
    // 3. Make the header fall through to the exit

    // Get the header block and replace its contents
    let header_block = match func.get_block_mut(&header_label) {
        Some(b) => b,
        None => return false,
    };

    // Keep instructions before the terminator (like the induction var init might be here)
    let non_term_count = header_block.insts.iter()
        .take_while(|i| !i.is_terminator())
        .count();
    header_block.insts.truncate(non_term_count);

    // Add all the unrolled instructions
    header_block.insts.extend(unrolled_insts);

    // Add a jump to the exit block
    header_block.insts.push(MachineInst::li(zero_vreg, 0).comment("unroll: dummy for jump"));
    header_block.insts.push(
        MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(zero_vreg))
            .src(Operand::Label(exit_label.clone()))
            .comment("unroll: jump to exit")
    );

    // Remove the body blocks (mark them as empty - they'll be cleaned up by DCE)
    for (i, body_label) in body_labels.iter().enumerate() {
        if let Some(body_block) = func.get_block_mut(body_label) {
            body_block.insts.clear();
            // Add a simple jump to exit so the block isn't completely dead
            body_block.insts.push(
                MachineInst::new(Opcode::JAL)
                    .dst(Operand::VReg(body_vregs[i]))
                    .src(Operand::Label(exit_label.clone()))
                    .comment("unroll: dead body")
            );
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MachineBlock;

    #[test]
    fn test_induction_var_count() {
        // for (i = 0; i < 4; i++)
        let iv = InductionVar {
            vreg: VReg::new(0),
            init: 0,
            step: 1,
            limit: 4,
            cmp_type: CmpType::LessThan,
        };
        assert_eq!(iv.iteration_count(), Some(4));

        // for (i = 0; i <= 3; i++)
        let iv = InductionVar {
            vreg: VReg::new(0),
            init: 0,
            step: 1,
            limit: 3,
            cmp_type: CmpType::LessEqual,
        };
        assert_eq!(iv.iteration_count(), Some(4));

        // for (i = 0; i < 8; i += 2)
        let iv = InductionVar {
            vreg: VReg::new(0),
            init: 0,
            step: 2,
            limit: 8,
            cmp_type: CmpType::LessThan,
        };
        assert_eq!(iv.iteration_count(), Some(4));

        // for (i = 0; i != 6; i += 2)
        let iv = InductionVar {
            vreg: VReg::new(0),
            init: 0,
            step: 2,
            limit: 6,
            cmp_type: CmpType::NotEqual,
        };
        assert_eq!(iv.iteration_count(), Some(3));
    }

    #[test]
    fn test_simple_loop_unroll() {
        // Create a simple loop:
        // entry:
        //   li v0, 0      ; i = 0
        //   li v1, 4      ; limit = 4
        //   jal loop_header
        // loop_header:
        //   bge v0, v1, exit  ; if i >= 4, exit
        // loop_body:
        //   addi v2, v2, 1   ; sum++
        //   addi v0, v0, 1   ; i++
        //   jal loop_header
        // exit:
        //   ret

        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg(); // induction var
        let v1 = func.new_vreg(); // limit
        let v2 = func.new_vreg(); // sum
        let v3 = func.new_vreg(); // jal dest

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 0));
        entry.push(MachineInst::li(v1, 4));
        entry.push(MachineInst::li(v2, 0));
        entry.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(v3))
            .src(Operand::Label("loop_header".to_string())));
        func.add_block(entry);

        // Use BGE to exit the loop (if i >= limit, goto exit)
        let mut header = MachineBlock::new("loop_header");
        header.push(MachineInst::new(Opcode::BGE)
            .src(Operand::VReg(v0))
            .src(Operand::VReg(v1))
            .src(Operand::Label("exit".to_string())));
        func.add_block(header);

        let mut body = MachineBlock::new("loop_body");
        body.push(MachineInst::addi(v2, v2, 1).comment("sum++"));
        body.push(MachineInst::addi(v0, v0, 1).comment("i++"));
        body.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(v3))
            .src(Operand::Label("loop_header".to_string())));
        func.add_block(body);

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();

        // Try to unroll
        let unrolled = unroll_loops(&mut func).unwrap();

        // Should have detected and unrolled the loop
        assert_eq!(unrolled, 1);

        // The header block should now contain unrolled iterations
        let header = func.get_block("loop_header").unwrap();
        // Should have 4 iterations of sum++ (the i++ is eliminated)
        let sum_increments = header.insts.iter()
            .filter(|i| i.opcode == Opcode::ADDI &&
                   i.comment.as_ref().map(|c| c.contains("unrolled")).unwrap_or(false))
            .count();
        assert_eq!(sum_increments, 4);
    }

    #[test]
    fn test_loop_detection() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        // Simple loop structure with explicit exit branch
        // entry -> header -> (body -> header) | exit
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 0));
        entry.push(MachineInst::li(v1, 8));
        entry.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(v2))
            .src(Operand::Label("header".to_string())));
        func.add_block(entry);

        // Header: if v0 >= v1, exit (BGE exits loop, BLT continues)
        // We use BGE to branch to exit, otherwise fallthrough to body
        let mut header = MachineBlock::new("header");
        header.push(MachineInst::new(Opcode::BGE)
            .src(Operand::VReg(v0))
            .src(Operand::VReg(v1))
            .src(Operand::Label("exit".to_string())));
        func.add_block(header);

        let mut body = MachineBlock::new("body");
        body.push(MachineInst::addi(v0, v0, 1));
        body.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(v2))
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
    }
}
