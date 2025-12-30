//! Common Subexpression Elimination (CSE) optimization pass.
//!
//! Identifies and eliminates redundant computations by reusing
//! previously computed values. This is particularly valuable for
//! ZK circuits where every instruction adds to constraint count.

use crate::mir::{MachineFunction, MachineInst, Opcode, Operand, VReg};
use anyhow::Result;
use std::collections::HashMap;

/// A normalized expression for CSE matching.
/// We normalize by sorting commutative operands to catch more matches.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Expression {
    opcode: Opcode,
    operands: Vec<NormalizedOperand>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NormalizedOperand {
    VReg(u32),
    Imm(i64),
}

impl Expression {
    /// Create an expression from an instruction.
    fn from_inst(inst: &MachineInst) -> Option<Self> {
        // Only handle pure computational instructions (no side effects)
        if !is_pure_computation(inst.opcode) {
            return None;
        }

        let mut operands: Vec<NormalizedOperand> = inst.srcs.iter()
            .filter_map(|op| match op {
                Operand::VReg(v) => Some(NormalizedOperand::VReg(v.id())),
                Operand::Imm(i) => Some(NormalizedOperand::Imm(*i)),
                _ => None,
            })
            .collect();

        // Normalize commutative operations by sorting operands
        if is_commutative(inst.opcode) && operands.len() == 2 {
            operands.sort_by(|a, b| {
                match (a, b) {
                    (NormalizedOperand::VReg(v1), NormalizedOperand::VReg(v2)) => v1.cmp(v2),
                    (NormalizedOperand::Imm(i1), NormalizedOperand::Imm(i2)) => i1.cmp(i2),
                    (NormalizedOperand::VReg(_), NormalizedOperand::Imm(_)) => std::cmp::Ordering::Less,
                    (NormalizedOperand::Imm(_), NormalizedOperand::VReg(_)) => std::cmp::Ordering::Greater,
                }
            });
        }

        Some(Expression {
            opcode: inst.opcode,
            operands,
        })
    }
}

/// Check if an opcode is a pure computation (no side effects, deterministic).
fn is_pure_computation(opcode: Opcode) -> bool {
    matches!(opcode,
        Opcode::ADD | Opcode::SUB | Opcode::MUL | Opcode::MULH | Opcode::ADDI |
        Opcode::AND | Opcode::OR | Opcode::XOR | Opcode::NOT |
        Opcode::ANDI | Opcode::ORI | Opcode::XORI |
        Opcode::SLL | Opcode::SRL | Opcode::SRA |
        Opcode::SLLI | Opcode::SRLI | Opcode::SRAI |
        Opcode::SLT | Opcode::SLTU | Opcode::SGE | Opcode::SGEU |
        Opcode::SEQ | Opcode::SNE
    )
    // Note: DIV, DIVU, REM, REMU are excluded because they can trap on division by zero
}

/// Check if an operation is commutative.
fn is_commutative(opcode: Opcode) -> bool {
    matches!(opcode,
        Opcode::ADD | Opcode::MUL |
        Opcode::AND | Opcode::OR | Opcode::XOR |
        Opcode::SEQ | Opcode::SNE
    )
}

/// Run CSE on a function.
pub fn eliminate_common_subexpressions(func: &mut MachineFunction) -> Result<u32> {
    let mut eliminated = 0;
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        if let Some(block) = func.get_block_mut(&label) {
            let count = cse_in_block(&mut block.insts);
            eliminated += count;
        }
    }

    Ok(eliminated)
}

/// Perform CSE within a single block.
fn cse_in_block(insts: &mut [MachineInst]) -> u32 {
    let mut eliminated = 0;

    // Map from expression to the vreg holding its value
    let mut expr_map: HashMap<Expression, VReg> = HashMap::new();

    // Track which vregs have been redefined (invalidates expressions using them)
    let mut defined: std::collections::HashSet<VReg> = std::collections::HashSet::new();

    // Collect replacements: (instruction index, new instruction)
    let mut replacements: Vec<(usize, MachineInst)> = Vec::new();

    for (idx, inst) in insts.iter().enumerate() {
        // If this instruction defines a vreg, invalidate any expressions using it
        if let Some(def_vreg) = inst.def() {
            // Remove expressions that use this vreg
            expr_map.retain(|expr, _| {
                !expr.operands.iter().any(|op| {
                    matches!(op, NormalizedOperand::VReg(v) if *v == def_vreg.id())
                })
            });
            defined.insert(def_vreg);
        }

        // Try to create an expression from this instruction
        if let Some(expr) = Expression::from_inst(inst) {
            if let Some(dst) = inst.def() {
                // Check if we've seen this expression before
                if let Some(&existing_vreg) = expr_map.get(&expr) {
                    // Replace with a MOV from the existing result
                    let new_inst = MachineInst::mov(dst, existing_vreg)
                        .comment("CSE: reuse previous result");
                    replacements.push((idx, new_inst));
                    eliminated += 1;

                    // Also record that dst now has the same value
                    // (for potential further CSE)
                } else {
                    // Record this expression
                    expr_map.insert(expr, dst);
                }
            }
        }
    }

    // Apply replacements
    for (idx, new_inst) in replacements {
        insts[idx] = new_inst;
    }

    eliminated
}

/// Local value numbering - a more aggressive form of CSE that also
/// tracks equivalences through MOV instructions.
#[allow(dead_code)]
pub fn local_value_numbering(func: &mut MachineFunction) -> Result<u32> {
    let mut eliminated = 0;
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        if let Some(block) = func.get_block_mut(&label) {
            let count = lvn_in_block(&mut block.insts);
            eliminated += count;
        }
    }

    Ok(eliminated)
}

/// Perform LVN within a single block.
fn lvn_in_block(insts: &mut [MachineInst]) -> u32 {
    let mut eliminated = 0;

    // Value number -> canonical vreg
    let mut value_to_vreg: HashMap<u32, VReg> = HashMap::new();
    // VReg -> value number
    let mut vreg_to_value: HashMap<VReg, u32> = HashMap::new();
    // Expression -> value number
    let mut expr_to_value: HashMap<Expression, u32> = HashMap::new();

    let mut next_value: u32 = 0;

    let mut replacements: Vec<(usize, MachineInst)> = Vec::new();

    for (idx, inst) in insts.iter().enumerate() {
        match inst.opcode {
            Opcode::LI => {
                // Constants get unique value numbers based on their value
                if let (Some(Operand::VReg(dst)), Some(Operand::Imm(val))) =
                    (&inst.dst, inst.srcs.first())
                {
                    // Create a pseudo-expression for constants
                    let expr = Expression {
                        opcode: Opcode::LI,
                        operands: vec![NormalizedOperand::Imm(*val)],
                    };

                    if let Some(&existing_value) = expr_to_value.get(&expr) {
                        // We've seen this constant before
                        if let Some(&canonical) = value_to_vreg.get(&existing_value) {
                            if canonical != *dst {
                                replacements.push((idx, MachineInst::mov(*dst, canonical)
                                    .comment("LVN: reuse constant")));
                                eliminated += 1;
                            }
                        }
                        vreg_to_value.insert(*dst, existing_value);
                    } else {
                        // New constant
                        let value = next_value;
                        next_value += 1;
                        expr_to_value.insert(expr, value);
                        value_to_vreg.insert(value, *dst);
                        vreg_to_value.insert(*dst, value);
                    }
                }
            }
            Opcode::MOV => {
                // MOV propagates value numbers
                if let (Some(Operand::VReg(dst)), Some(Operand::VReg(src))) =
                    (&inst.dst, inst.srcs.first())
                {
                    if let Some(&value) = vreg_to_value.get(src) {
                        vreg_to_value.insert(*dst, value);
                    } else {
                        // Source has no value number, assign new one
                        let value = next_value;
                        next_value += 1;
                        vreg_to_value.insert(*src, value);
                        vreg_to_value.insert(*dst, value);
                        value_to_vreg.insert(value, *src);
                    }
                }
            }
            _ => {
                if let Some(expr) = Expression::from_inst(inst) {
                    if let Some(dst) = inst.def() {
                        // Normalize the expression using value numbers
                        let normalized = normalize_with_values(&expr, &vreg_to_value);

                        if let Some(&existing_value) = expr_to_value.get(&normalized) {
                            if let Some(&canonical) = value_to_vreg.get(&existing_value) {
                                replacements.push((idx, MachineInst::mov(dst, canonical)
                                    .comment("LVN: reuse computation")));
                                eliminated += 1;
                                vreg_to_value.insert(dst, existing_value);
                                continue;
                            }
                        }

                        // New expression
                        let value = next_value;
                        next_value += 1;
                        expr_to_value.insert(normalized, value);
                        value_to_vreg.insert(value, dst);
                        vreg_to_value.insert(dst, value);
                    }
                } else if let Some(dst) = inst.def() {
                    // Non-pure instruction, assign fresh value number
                    let value = next_value;
                    next_value += 1;
                    vreg_to_value.insert(dst, value);
                    value_to_vreg.insert(value, dst);
                }
            }
        }
    }

    // Apply replacements
    for (idx, new_inst) in replacements {
        insts[idx] = new_inst;
    }

    eliminated
}

/// Normalize an expression using value numbers instead of vreg ids.
fn normalize_with_values(
    expr: &Expression,
    vreg_to_value: &HashMap<VReg, u32>,
) -> Expression {
    let mut operands: Vec<NormalizedOperand> = expr.operands.iter()
        .map(|op| match op {
            NormalizedOperand::VReg(id) => {
                let vreg = VReg::new(*id);
                if let Some(&value) = vreg_to_value.get(&vreg) {
                    NormalizedOperand::VReg(value)
                } else {
                    op.clone()
                }
            }
            _ => op.clone(),
        })
        .collect();

    // Sort if commutative
    if is_commutative(expr.opcode) && operands.len() == 2 {
        operands.sort_by(|a, b| {
            match (a, b) {
                (NormalizedOperand::VReg(v1), NormalizedOperand::VReg(v2)) => v1.cmp(v2),
                (NormalizedOperand::Imm(i1), NormalizedOperand::Imm(i2)) => i1.cmp(i2),
                (NormalizedOperand::VReg(_), NormalizedOperand::Imm(_)) => std::cmp::Ordering::Less,
                (NormalizedOperand::Imm(_), NormalizedOperand::VReg(_)) => std::cmp::Ordering::Greater,
            }
        });
    }

    Expression {
        opcode: expr.opcode,
        operands,
    }
}

/// Global CSE across basic blocks.
/// Uses a forward dataflow analysis to propagate available expressions.
pub fn global_cse(func: &mut MachineFunction) -> Result<u32> {
    let mut eliminated = 0;

    // Build CFG edges
    func.rebuild_cfg();

    // Get blocks in RPO (reverse post-order) for forward dataflow
    let rpo = compute_rpo(func);

    // Available expressions at entry of each block
    // Maps block label -> (expression -> defining vreg)
    let mut available_in: HashMap<String, HashMap<Expression, VReg>> = HashMap::new();

    // Initialize all blocks with empty sets
    for label in func.block_labels() {
        available_in.insert(label.to_string(), HashMap::new());
    }

    // Iterate until fixed point
    let mut changed = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 100;

    while changed && iterations < MAX_ITERATIONS {
        changed = false;
        iterations += 1;

        for label in &rpo {
            // Compute available_in as intersection of all predecessors' available_out
            let preds: Vec<String> = func.get_block(label)
                .map(|b| b.preds.clone())
                .unwrap_or_default();

            let new_in = if preds.is_empty() {
                HashMap::new()
            } else {
                // Start with first predecessor's out set
                let first_pred = &preds[0];
                let first_out = compute_available_out(func, first_pred, &available_in);

                // Intersect with other predecessors
                let mut result = first_out;
                for pred in preds.iter().skip(1) {
                    let pred_out = compute_available_out(func, pred, &available_in);
                    result.retain(|expr, vreg| {
                        pred_out.get(expr) == Some(vreg)
                    });
                }
                result
            };

            // Check if changed
            if available_in.get(label) != Some(&new_in) {
                available_in.insert(label.clone(), new_in);
                changed = true;
            }
        }
    }

    // Now use the available expressions to eliminate redundant computations
    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        if let Some(avail) = available_in.get(&label).cloned() {
            if let Some(block) = func.get_block_mut(&label) {
                let count = apply_global_cse(&mut block.insts, avail);
                eliminated += count;
            }
        }
    }

    Ok(eliminated)
}

/// Compute reverse post-order of blocks.
fn compute_rpo(func: &MachineFunction) -> Vec<String> {
    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut post_order: Vec<String> = Vec::new();

    fn dfs(
        func: &MachineFunction,
        label: &str,
        visited: &mut std::collections::HashSet<String>,
        post_order: &mut Vec<String>,
    ) {
        if visited.contains(label) {
            return;
        }
        visited.insert(label.to_string());

        if let Some(block) = func.get_block(label) {
            for succ in &block.succs {
                dfs(func, succ, visited, post_order);
            }
        }
        post_order.push(label.to_string());
    }

    dfs(func, &func.entry, &mut visited, &mut post_order);
    post_order.reverse();
    post_order
}

/// Compute available expressions at the exit of a block.
fn compute_available_out(
    func: &MachineFunction,
    label: &str,
    available_in: &HashMap<String, HashMap<Expression, VReg>>,
) -> HashMap<Expression, VReg> {
    let mut available = available_in.get(label).cloned().unwrap_or_default();

    if let Some(block) = func.get_block(label) {
        for inst in &block.insts {
            // If this instruction defines a vreg, kill expressions using it
            if let Some(def_vreg) = inst.def() {
                available.retain(|expr, _| {
                    !expr.operands.iter().any(|op| {
                        matches!(op, NormalizedOperand::VReg(v) if *v == def_vreg.id())
                    })
                });

                // Generate: add this expression if pure
                if let Some(expr) = Expression::from_inst(inst) {
                    available.insert(expr, def_vreg);
                }
            }
        }
    }

    available
}

/// Apply global CSE using available expressions from predecessors.
fn apply_global_cse(insts: &mut [MachineInst], mut available: HashMap<Expression, VReg>) -> u32 {
    let mut eliminated = 0;
    let mut replacements: Vec<(usize, MachineInst)> = Vec::new();

    for (idx, inst) in insts.iter().enumerate() {
        // Update available set for kills
        if let Some(def_vreg) = inst.def() {
            available.retain(|expr, _| {
                !expr.operands.iter().any(|op| {
                    matches!(op, NormalizedOperand::VReg(v) if *v == def_vreg.id())
                })
            });
        }

        // Try to find this expression in available set
        if let Some(expr) = Expression::from_inst(inst) {
            if let Some(dst) = inst.def() {
                if let Some(&existing_vreg) = available.get(&expr) {
                    // Replace with MOV
                    let new_inst = MachineInst::mov(dst, existing_vreg)
                        .comment("Global CSE: reuse from predecessor");
                    replacements.push((idx, new_inst));
                    eliminated += 1;
                } else {
                    // Record this expression
                    available.insert(expr, dst);
                }
            }
        }
    }

    // Apply replacements
    for (idx, new_inst) in replacements {
        insts[idx] = new_inst;
    }

    eliminated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MachineBlock;

    #[test]
    fn test_simple_cse() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 20));
        entry.push(MachineInst::add(v2, v0, v1)); // First computation
        entry.push(MachineInst::add(v3, v0, v1)); // Same computation - should be eliminated
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let eliminated = eliminate_common_subexpressions(&mut func).unwrap();
        assert_eq!(eliminated, 1);

        let block = func.get_block("entry").unwrap();
        // The second add should now be a MOV
        assert_eq!(block.insts[3].opcode, Opcode::MOV);
    }

    #[test]
    fn test_commutative_cse() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 20));
        entry.push(MachineInst::add(v2, v0, v1)); // a + b
        entry.push(MachineInst::add(v3, v1, v0)); // b + a - same due to commutativity
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let eliminated = eliminate_common_subexpressions(&mut func).unwrap();
        assert_eq!(eliminated, 1);
    }

    #[test]
    fn test_no_cse_after_redefinition() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 20));
        entry.push(MachineInst::add(v2, v0, v1)); // First computation
        entry.push(MachineInst::li(v0, 30));      // v0 redefined!
        entry.push(MachineInst::add(v3, v0, v1)); // Different computation (new v0)
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let eliminated = eliminate_common_subexpressions(&mut func).unwrap();
        assert_eq!(eliminated, 0); // Should NOT eliminate

        let block = func.get_block("entry").unwrap();
        // Both adds should remain
        assert_eq!(block.insts[2].opcode, Opcode::ADD);
        assert_eq!(block.insts[4].opcode, Opcode::ADD);
    }

    #[test]
    fn test_cse_with_immediate() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 100));
        entry.push(MachineInst::addi(v1, v0, 5)); // v1 = v0 + 5
        entry.push(MachineInst::addi(v2, v0, 5)); // Same - should be eliminated
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let eliminated = eliminate_common_subexpressions(&mut func).unwrap();
        assert_eq!(eliminated, 1);
    }

    #[test]
    fn test_lvn_constant_reuse() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::li(v1, 42)); // Same constant - can reuse
        entry.push(MachineInst::add(v2, v0, v1));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let eliminated = local_value_numbering(&mut func).unwrap();
        assert_eq!(eliminated, 1);

        let block = func.get_block("entry").unwrap();
        // The second li should now be a MOV
        assert_eq!(block.insts[1].opcode, Opcode::MOV);
    }

    #[test]
    fn test_global_cse_across_blocks() {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();
        let dummy = func.new_vreg();

        // Entry block: compute v0 + v1
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 20));
        entry.push(MachineInst::add(v2, v0, v1)); // First computation
        entry.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("next".to_string())));
        func.add_block(entry);

        // Next block: same computation should be eliminated
        let mut next = MachineBlock::new("next");
        next.push(MachineInst::add(v3, v0, v1)); // Same computation - should use v2
        next.push(MachineInst::ret());
        func.add_block(next);

        func.rebuild_cfg();

        let eliminated = global_cse(&mut func).unwrap();
        assert_eq!(eliminated, 1);

        let block = func.get_block("next").unwrap();
        // The add should now be a MOV from v2
        assert_eq!(block.insts[0].opcode, Opcode::MOV);
    }

    #[test]
    fn test_global_cse_diamond() {
        // Test CSE at merge point (diamond CFG)
        // In this test we check that expression from entry is available at merge
        // via both paths (left and right both preserve v0 and v1)
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();
        let v4 = func.new_vreg();
        let dummy = func.new_vreg();

        // Entry: compute v0 + v1, then jump to left
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 20));
        entry.push(MachineInst::add(v2, v0, v1)); // First computation
        entry.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("left".to_string())));
        func.add_block(entry);

        // Left branch - preserves v0, v1
        let mut left = MachineBlock::new("left");
        left.push(MachineInst::li(v3, 100));
        left.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("merge".to_string())));
        func.add_block(left);

        // Merge: same computation as entry - should be eliminated
        let mut merge = MachineBlock::new("merge");
        merge.push(MachineInst::add(v4, v0, v1)); // Same - available from entry via left
        merge.push(MachineInst::ret());
        func.add_block(merge);

        func.rebuild_cfg();

        let eliminated = global_cse(&mut func).unwrap();
        assert_eq!(eliminated, 1);

        let block = func.get_block("merge").unwrap();
        assert_eq!(block.insts[0].opcode, Opcode::MOV);
    }
}
