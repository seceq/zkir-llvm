//! Function inlining optimization pass.
//!
//! Inlines small functions at their call sites. This eliminates
//! call/return overhead (branches are expensive in ZK) and enables
//! cross-function optimizations like CSE and constant folding.

use crate::mir::{MachineFunction, MachineInst, Module, Opcode, Operand, VReg};
use anyhow::Result;
use std::collections::HashMap;

/// Maximum size of a function to inline (in instructions).
const MAX_INLINE_SIZE: usize = 32;

/// Maximum number of calls to inline per function to prevent blowup.
const MAX_INLINES_PER_FUNCTION: usize = 16;

/// Inline small functions in a module.
pub fn inline_functions(module: &mut Module) -> Result<u32> {
    let mut total_inlined = 0;

    // First, identify which functions are inlinable
    let inlinable: HashMap<String, bool> = module
        .functions
        .iter()
        .map(|(name, func)| {
            let can_inline = is_inlinable(func);
            (name.clone(), can_inline)
        })
        .collect();

    // Clone function data for functions that will be inlined
    let inline_data: HashMap<String, FunctionInlineData> = module
        .functions
        .iter()
        .filter(|(name, _)| inlinable.get(*name).copied().unwrap_or(false))
        .map(|(name, func)| (name.clone(), FunctionInlineData::from_function(func)))
        .collect();

    // Process each function
    let func_names: Vec<String> = module.functions.keys().cloned().collect();

    for name in func_names {
        if let Some(func) = module.functions.get_mut(&name) {
            let count = inline_in_function(func, &inline_data)?;
            total_inlined += count;
        }
    }

    Ok(total_inlined)
}

/// Check if a function is eligible for inlining.
fn is_inlinable(func: &MachineFunction) -> bool {
    // Count total instructions
    let total_insts: usize = func.iter_blocks().map(|b| b.insts.len()).sum();

    if total_insts > MAX_INLINE_SIZE {
        return false;
    }

    // Check for unsupported features
    for block in func.iter_blocks() {
        for inst in &block.insts {
            match inst.opcode {
                // Don't inline functions that call other functions (could cause recursion)
                Opcode::CALL => return false,
                // Don't inline functions with complex control flow
                // (single block or simple conditionals are OK)
                _ => {}
            }
        }
    }

    // Must have at most 2 blocks (entry and optional exit)
    if func.blocks.len() > 2 {
        return false;
    }

    true
}

/// Data needed to inline a function.
#[derive(Clone)]
struct FunctionInlineData {
    /// Parameter vregs in the callee
    params: Vec<VReg>,
    /// Instructions in the function body (excluding RET)
    insts: Vec<MachineInst>,
    /// Return value vreg (if any)
    ret_vreg: Option<VReg>,
}

impl FunctionInlineData {
    fn from_function(func: &MachineFunction) -> Self {
        let mut insts = Vec::new();
        let mut ret_vreg = None;

        for block in func.iter_blocks() {
            for inst in &block.insts {
                if inst.opcode == Opcode::RET {
                    // Capture return value if present
                    if let Some(Operand::VReg(v)) = inst.srcs.first() {
                        ret_vreg = Some(*v);
                    }
                } else {
                    insts.push(inst.clone());
                }
            }
        }

        FunctionInlineData {
            params: func.params.clone(),
            insts,
            ret_vreg,
        }
    }
}

/// Inline function calls within a single function.
fn inline_in_function(
    func: &mut MachineFunction,
    inline_data: &HashMap<String, FunctionInlineData>,
) -> Result<u32> {
    let mut inlined = 0;

    let block_labels: Vec<String> = func.block_labels().iter().map(|s| s.to_string()).collect();

    for label in block_labels {
        if inlined >= MAX_INLINES_PER_FUNCTION as u32 {
            break;
        }

        // Find CALL instructions in this block
        let call_sites: Vec<(usize, String, Vec<VReg>, Option<VReg>)> = {
            let block = match func.get_block(&label) {
                Some(b) => b,
                None => continue,
            };

            block
                .insts
                .iter()
                .enumerate()
                .filter_map(|(idx, inst)| {
                    if inst.opcode == Opcode::CALL {
                        // Extract callee name
                        let callee = inst.srcs.iter().find_map(|op| {
                            if let Operand::Label(l) = op {
                                Some(l.clone())
                            } else {
                                None
                            }
                        })?;

                        // Check if callee is inlinable
                        if !inline_data.contains_key(&callee) {
                            return None;
                        }

                        // Extract arguments
                        let args: Vec<VReg> = inst
                            .srcs
                            .iter()
                            .filter_map(|op| {
                                if let Operand::VReg(v) = op {
                                    Some(*v)
                                } else {
                                    None
                                }
                            })
                            .collect();

                        // Extract destination
                        let dst = inst.def();

                        Some((idx, callee, args, dst))
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Process call sites in reverse order to maintain indices
        for (idx, callee, args, dst) in call_sites.into_iter().rev() {
            if inlined >= MAX_INLINES_PER_FUNCTION as u32 {
                break;
            }

            let data = match inline_data.get(&callee) {
                Some(d) => d,
                None => continue,
            };

            // Create vreg mapping: callee vreg -> caller vreg
            let mut vreg_map: HashMap<VReg, VReg> = HashMap::new();

            // Map parameters to arguments
            for (param, arg) in data.params.iter().zip(args.iter()) {
                vreg_map.insert(*param, *arg);
            }

            // Allocate new vregs for callee-defined values
            for inst in &data.insts {
                if let Some(def) = inst.def() {
                    vreg_map.entry(def).or_insert_with(|| func.new_vreg());
                }
            }

            // Also map the return vreg
            if let Some(ret) = data.ret_vreg {
                vreg_map.entry(ret).or_insert_with(|| {
                    if let Some(d) = dst {
                        d
                    } else {
                        func.new_vreg()
                    }
                });
            }

            // Create inlined instructions
            let mut inlined_insts: Vec<MachineInst> = Vec::new();

            for inst in &data.insts {
                let mut new_inst = inst.clone();

                // Remap destination
                if let Some(Operand::VReg(v)) = &mut new_inst.dst {
                    if let Some(&mapped) = vreg_map.get(v) {
                        *v = mapped;
                    }
                }

                // Remap sources
                for src in &mut new_inst.srcs {
                    if let Operand::VReg(v) = src {
                        if let Some(&mapped) = vreg_map.get(v) {
                            *v = mapped;
                        }
                    }
                }

                new_inst = new_inst.comment(&format!("inlined from {}", callee));
                inlined_insts.push(new_inst);
            }

            // Add MOV for return value if needed
            if let (Some(ret), Some(d)) = (data.ret_vreg, dst) {
                let ret_mapped = vreg_map.get(&ret).copied().unwrap_or(ret);
                if ret_mapped != d {
                    inlined_insts.push(
                        MachineInst::mov(d, ret_mapped)
                            .comment(&format!("inlined return from {}", callee)),
                    );
                }
            }

            // Replace the CALL instruction with inlined code
            if let Some(block) = func.get_block_mut(&label) {
                // Remove the CALL
                block.insts.remove(idx);

                // Insert inlined instructions
                for (i, inst) in inlined_insts.into_iter().enumerate() {
                    block.insts.insert(idx + i, inst);
                }
            }

            inlined += 1;
        }
    }

    Ok(inlined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MachineBlock;

    #[test]
    fn test_simple_inline() {
        let mut module = Module::new("test");

        // Create a small function to inline: add_one(x) = x + 1
        let mut add_one = MachineFunction::new("add_one");
        let param = add_one.new_vreg();
        add_one.params.push(param);
        let result = add_one.new_vreg();
        add_one.ret_vreg = Some(result);

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::addi(result, param, 1));
        entry.push(MachineInst::new(Opcode::RET).src(Operand::VReg(result)));
        add_one.add_block(entry);

        module.add_function(add_one);

        // Create caller function
        let mut caller = MachineFunction::new("caller");
        let v0 = caller.new_vreg();
        let v1 = caller.new_vreg();

        let mut caller_entry = MachineBlock::new("entry");
        caller_entry.push(MachineInst::li(v0, 10));
        caller_entry.push(
            MachineInst::new(Opcode::CALL)
                .dst(Operand::VReg(v1))
                .src(Operand::Label("add_one".to_string()))
                .src(Operand::VReg(v0)),
        );
        caller_entry.push(MachineInst::new(Opcode::RET).src(Operand::VReg(v1)));
        caller.add_block(caller_entry);

        module.add_function(caller);

        // Inline
        let inlined = inline_functions(&mut module).unwrap();
        assert_eq!(inlined, 1);

        // Check that the call was replaced
        let caller = module.get_function("caller").unwrap();
        let block = caller.get_block("entry").unwrap();

        // Should have: li, addi, (optional mov), ret
        // No CALL instruction
        assert!(!block.insts.iter().any(|i| i.opcode == Opcode::CALL));
        assert!(block.insts.iter().any(|i| i.opcode == Opcode::ADDI));
    }

    #[test]
    fn test_no_inline_recursive() {
        let mut module = Module::new("test");

        // Create a recursive function - should not be inlined
        let mut recursive = MachineFunction::new("recursive");
        let param = recursive.new_vreg();
        recursive.params.push(param);
        let result = recursive.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(
            MachineInst::new(Opcode::CALL)
                .dst(Operand::VReg(result))
                .src(Operand::Label("recursive".to_string()))
                .src(Operand::VReg(param)),
        );
        entry.push(MachineInst::new(Opcode::RET).src(Operand::VReg(result)));
        recursive.add_block(entry);

        module.add_function(recursive);

        // Create caller
        let mut caller = MachineFunction::new("caller");
        let v0 = caller.new_vreg();
        let v1 = caller.new_vreg();

        let mut caller_entry = MachineBlock::new("entry");
        caller_entry.push(MachineInst::li(v0, 5));
        caller_entry.push(
            MachineInst::new(Opcode::CALL)
                .dst(Operand::VReg(v1))
                .src(Operand::Label("recursive".to_string()))
                .src(Operand::VReg(v0)),
        );
        caller_entry.push(MachineInst::new(Opcode::RET).src(Operand::VReg(v1)));
        caller.add_block(caller_entry);

        module.add_function(caller);

        // Should not inline because recursive function contains CALL
        let inlined = inline_functions(&mut module).unwrap();
        assert_eq!(inlined, 0);
    }

    #[test]
    fn test_inline_multiple_args() {
        let mut module = Module::new("test");

        // Create add(a, b) = a + b
        let mut add = MachineFunction::new("add");
        let a = add.new_vreg();
        let b = add.new_vreg();
        add.params.push(a);
        add.params.push(b);
        let result = add.new_vreg();
        add.ret_vreg = Some(result);

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::add(result, a, b));
        entry.push(MachineInst::new(Opcode::RET).src(Operand::VReg(result)));
        add.add_block(entry);

        module.add_function(add);

        // Create caller
        let mut caller = MachineFunction::new("caller");
        let v0 = caller.new_vreg();
        let v1 = caller.new_vreg();
        let v2 = caller.new_vreg();

        let mut caller_entry = MachineBlock::new("entry");
        caller_entry.push(MachineInst::li(v0, 10));
        caller_entry.push(MachineInst::li(v1, 20));
        caller_entry.push(
            MachineInst::new(Opcode::CALL)
                .dst(Operand::VReg(v2))
                .src(Operand::Label("add".to_string()))
                .src(Operand::VReg(v0))
                .src(Operand::VReg(v1)),
        );
        caller_entry.push(MachineInst::new(Opcode::RET).src(Operand::VReg(v2)));
        caller.add_block(caller_entry);

        module.add_function(caller);

        // Inline
        let inlined = inline_functions(&mut module).unwrap();
        assert_eq!(inlined, 1);

        // Check result
        let caller = module.get_function("caller").unwrap();
        let block = caller.get_block("entry").unwrap();

        // Should have ADD instruction
        assert!(block.insts.iter().any(|i| i.opcode == Opcode::ADD));
        assert!(!block.insts.iter().any(|i| i.opcode == Opcode::CALL));
    }
}
