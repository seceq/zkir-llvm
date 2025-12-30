//! Control flow instruction lowering.
//!
//! Handles LLVM control flow: branches, returns, calls, phi nodes.

use super::LoweringContext;
use super::types;
use crate::mir::{MachineInst, Opcode, Operand, ValueBounds, VReg};
use crate::target::abi::{compute_arg_locations, ArgLocation};
use crate::target::registers::{Register, RET_REGS};
use anyhow::Result;
use inkwell::values::{InstructionValue, PhiValue, AsValueRef, BasicValueEnum, BasicValue, AnyValueEnum};
use inkwell::types::{AnyTypeEnum, BasicTypeEnum};

/// Lower unconditional or conditional branch.
pub fn lower_br<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let num_operands = inst.get_num_operands();

    if num_operands == 1 {
        // Unconditional branch: br label %dest
        let dest_op = inst.get_operand(0)
            .ok_or_else(|| anyhow::anyhow!("Branch instruction missing destination operand"))?;
        let bb = dest_op.block()
            .ok_or_else(|| anyhow::anyhow!("Unconditional branch operand is not a basic block"))?;
        let label = bb.get_name().to_str().unwrap_or("bb").to_string();

        // Use JAL with zero destination (effectively a jump)
        let zero = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(zero))
            .src(Operand::Label(label)));
    } else if num_operands == 3 {
        // Conditional branch: br i1 %cond, label %true, label %false
        let cond_op = inst.get_operand(0)
            .ok_or_else(|| anyhow::anyhow!("Conditional branch missing condition operand"))?;
        let cond = cond_op.value()
            .ok_or_else(|| anyhow::anyhow!("Branch condition is not a value (expected i1)"))?;
        let cond_vreg = ctx.get_vreg(&cond);

        let true_bb = inst.get_operand(1)
            .ok_or_else(|| anyhow::anyhow!("Conditional branch missing true destination"))?
            .block()
            .ok_or_else(|| anyhow::anyhow!("True destination is not a basic block"))?;
        let false_bb = inst.get_operand(2)
            .ok_or_else(|| anyhow::anyhow!("Conditional branch missing false destination"))?
            .block()
            .ok_or_else(|| anyhow::anyhow!("False destination is not a basic block"))?;

        let true_label = true_bb.get_name().to_str().unwrap_or("bb").to_string();
        let false_label = false_bb.get_name().to_str().unwrap_or("bb").to_string();

        // BNE cond, zero, true_label (branch if cond != 0)
        let zero = ctx.new_vreg();
        ctx.emit(MachineInst::li(zero, 0));
        ctx.emit(MachineInst::bne(cond_vreg, zero, &true_label));

        // Fall through or jump to false label
        let link = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(link))
            .src(Operand::Label(false_label)));
    } else {
        return Err(anyhow::anyhow!(
            "Invalid branch instruction: expected 1 or 3 operands, got {}",
            num_operands
        ));
    }

    Ok(())
}

/// Lower switch instruction.
pub fn lower_switch<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    // Switch is lowered to a series of comparisons and branches
    let cond_op = inst.get_operand(0)
        .ok_or_else(|| anyhow::anyhow!("Switch instruction missing condition operand"))?;
    let cond = cond_op.value()
        .ok_or_else(|| anyhow::anyhow!("Switch condition is not a value"))?;
    let cond_vreg = ctx.get_vreg(&cond);

    // Get default destination
    let default_op = inst.get_operand(1)
        .ok_or_else(|| anyhow::anyhow!("Switch instruction missing default destination"))?;
    let default_bb = default_op.block()
        .ok_or_else(|| anyhow::anyhow!("Switch default destination is not a basic block"))?;
    let default_label = default_bb.get_name().to_str().unwrap_or("bb").to_string();

    // Get switch cases (pairs of value, destination)
    let num_operands = inst.get_num_operands();
    let num_cases = (num_operands - 2) / 2;

    for i in 0..num_cases {
        let case_val_op = inst.get_operand(2 + i * 2);
        let case_dest_op = inst.get_operand(3 + i * 2);

        if let (Some(case_val), Some(case_dest)) = (
            case_val_op.and_then(|op| op.value()),
            case_dest_op.and_then(|op| op.block()),
        ) {
            let case_label = case_dest.get_name().to_str().unwrap_or("bb").to_string();
            let case_vreg = ctx.get_vreg(&case_val);

            // Compare and branch if equal
            let eq_result = ctx.new_vreg();
            ctx.emit(MachineInst::new(Opcode::SEQ)
                .dst(Operand::VReg(eq_result))
                .src(Operand::VReg(cond_vreg))
                .src(Operand::VReg(case_vreg)));

            let zero = ctx.new_vreg();
            ctx.emit(MachineInst::li(zero, 0));
            ctx.emit(MachineInst::bne(eq_result, zero, &case_label));
        }
    }

    // Jump to default
    let link = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::JAL)
        .dst(Operand::VReg(link))
        .src(Operand::Label(default_label)));

    Ok(())
}

/// Lower return instruction.
pub fn lower_return<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    // Check if we have a return value
    if let Some(ret_val) = inst.get_operand(0) {
        if let Some(value) = ret_val.value() {
            let ret_vreg = ctx.get_vreg(&value);

            // Move return value to the designated return register (a0/r10)
            // This will be handled by the register allocator, but we emit a MOV
            // to the function's return vreg
            if let Some(func_ret_vreg) = ctx.func.ret_vreg {
                ctx.emit(MachineInst::mov(func_ret_vreg, ret_vreg)
                    .comment("return value"));
            }
        }
    }

    // Emit return
    ctx.emit(MachineInst::ret());

    Ok(())
}

/// Lower call instruction.
pub fn lower_call<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    // Get the callee
    let num_operands = inst.get_num_operands();
    let callee_operand = inst.get_operand(num_operands - 1);

    // Determine if this is a direct or indirect call
    let (callee_name, is_indirect, callee_vreg) = if let Some(operand) = callee_operand {
        if let Some(func) = operand.value() {
            // Try to extract pointer value - the callee is typically a pointer to a function
            if func.is_pointer_value() {
                let ptr = func.into_pointer_value();
                let name = ptr.get_name().to_str().ok().map(|s| s.to_string());

                // Check if this is a named function (direct call) or a computed pointer (indirect call)
                // A direct call has a non-empty function name. An indirect call uses a value.
                if let Some(ref n) = name {
                    if !n.is_empty() {
                        // Direct call to a named function
                        (n.clone(), false, None)
                    } else {
                        // Indirect call - need to reload func since we consumed it
                        // Use a fresh vreg for the function pointer
                        let vreg = ctx.new_vreg();
                        ("".to_string(), true, Some(vreg))
                    }
                } else {
                    // Indirect call
                    let vreg = ctx.new_vreg();
                    ("".to_string(), true, Some(vreg))
                }
            } else {
                // Not a pointer value - treat as indirect call through the value
                let vreg = ctx.get_vreg(&func);
                ("".to_string(), true, Some(vreg))
            }
        } else {
            ("unknown".to_string(), false, None)
        }
    } else {
        ("unknown".to_string(), false, None)
    };

    // Get argument count (all operands except the last one which is the callee)
    let num_args = num_operands - 1;

    // Lower arguments - place them in argument registers or stack
    let mut arg_bits: Vec<u32> = Vec::with_capacity(num_args as usize);
    let mut arg_vregs: Vec<VReg> = Vec::with_capacity(num_args as usize);

    for i in 0..(num_args as u32) {
        if let Some(arg) = inst.get_operand(i) {
            if let Some(value) = arg.value() {
                let vreg = ctx.get_vreg(&value);
                arg_vregs.push(vreg);

                let bits = types::type_bits(&value.get_type(), ctx.config);
                arg_bits.push(bits);
            }
        }
    }

    // Compute argument locations
    let ret_bits = {
        let call_type = inst.get_type();
        match call_type {
            AnyTypeEnum::IntType(int_ty) => Some(int_ty.get_bit_width()),
            AnyTypeEnum::PointerType(_) => Some(ctx.config.addr_bits()),
            _ => None,
        }
    };

    let abi = compute_arg_locations(&arg_bits, ret_bits, ctx.config);

    // Move arguments to their designated locations
    for (i, (vreg, loc)) in arg_vregs.iter().zip(abi.params.iter()).enumerate() {
        match loc {
            ArgLocation::Register(reg) => {
                // Move to argument register
                // We'll emit this as a MOV that the allocator will coalesce
                ctx.emit(MachineInst::new(Opcode::MOV)
                    .dst(Operand::Reg(*reg))
                    .src(Operand::VReg(*vreg))
                    .comment(format!("arg{}", i)));
            }
            ArgLocation::RegisterPair(regs) => {
                // Multi-register argument for split values (e.g., i64 on 40-bit config)
                // Move the source vreg to all registers in the pair.
                // The int_splitting pass (which runs after lowering) will handle
                // the actual lo/hi split of values that don't fit in a single register.
                for (j, reg) in regs.iter().enumerate() {
                    ctx.emit(MachineInst::new(Opcode::MOV)
                        .dst(Operand::Reg(*reg))
                        .src(Operand::VReg(*vreg))
                        .comment(format!("arg{} part{}", i, j)));
                }
            }
            ArgLocation::Stack(offset) => {
                // Store to stack
                ctx.emit(MachineInst::new(Opcode::SW)
                    .src(Operand::VReg(*vreg))
                    .src(Operand::MemReg {
                        base: Register::R2, // SP
                        offset: *offset,
                    })
                    .comment(format!("arg{} (stack)", i)));
            }
            ArgLocation::Indirect(reg) => {
                // Pass by reference
                ctx.emit(MachineInst::new(Opcode::MOV)
                    .dst(Operand::Reg(*reg))
                    .src(Operand::VReg(*vreg))
                    .comment(format!("arg{} (indirect)", i)));
            }
        }
    }

    // Update outgoing args size
    ctx.func.frame.outgoing_args_size = ctx.func.frame.outgoing_args_size.max(abi.arg_stack_size);

    // Emit the call (direct or indirect)
    if is_indirect {
        // Indirect call through register
        if let Some(target_vreg) = callee_vreg {
            ctx.emit(MachineInst::callr(target_vreg)
                .comment("indirect call"));
        } else {
            return Err(anyhow::anyhow!("Indirect call missing target register"));
        }
    } else {
        // Direct call to named function
        ctx.emit(MachineInst::new(Opcode::CALL)
            .src(Operand::Label(callee_name)));
    }

    // Handle return value
    if let Some(ret_loc) = abi.ret {
        let dst = ctx.new_vreg();

        match ret_loc {
            ArgLocation::Register(reg) => {
                ctx.emit(MachineInst::new(Opcode::MOV)
                    .dst(Operand::VReg(dst))
                    .src(Operand::Reg(reg))
                    .comment("call result"));
            }
            ArgLocation::RegisterPair(regs) => {
                // Multi-register return for split values (e.g., i64 on 40-bit)
                // For now, just use the first register as the result.
                // The int_splitting pass will handle combining lo/hi if needed.
                if let Some(&reg) = regs.first() {
                    ctx.emit(MachineInst::new(Opcode::MOV)
                        .dst(Operand::VReg(dst))
                        .src(Operand::Reg(reg))
                        .comment("call result (lo)"));
                }
            }
            _ => {
                // Stack or indirect return - use default return register
                ctx.emit(MachineInst::new(Opcode::MOV)
                    .dst(Operand::VReg(dst))
                    .src(Operand::Reg(RET_REGS[0]))
                    .comment("call result"));
            }
        }

        // Map result - use the instruction's result as a basic value
        // In inkwell 0.5, we get the value from the instruction operand result
        let result_val = unsafe {
            BasicValueEnum::new(inst.as_value_ref())
        };
        ctx.map_value(&result_val, dst);

        // Set bounds from return type
        if let Some(bits) = ret_bits {
            ctx.set_bounds(&result_val, ValueBounds::from_bits(bits));
        }
    }

    Ok(())
}

/// Lower unreachable instruction.
pub fn lower_unreachable<'a>(ctx: &mut LoweringContext<'a>, _inst: &InstructionValue<'a>) -> Result<()> {
    // Emit EBREAK to trap
    ctx.emit(MachineInst::new(Opcode::EBREAK));
    Ok(())
}

/// Lower phi instruction.
pub fn lower_phi<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    // PHI nodes are lowered to a special PHI pseudo-instruction that carries
    // all the (predecessor, value) pairs. The phi_elimination pass will later
    // convert this to MOV instructions in predecessor blocks.

    // Convert InstructionValue to PhiValue
    let phi: PhiValue<'a> = unsafe {
        PhiValue::new(inst.as_value_ref())
    };

    let dst = ctx.new_vreg();

    // Get the type bits for bounds tracking
    let ty = phi.as_basic_value().get_type();
    let bits = match ty {
        BasicTypeEnum::IntType(int_ty) => int_ty.get_bit_width(),
        BasicTypeEnum::PointerType(_) => ctx.config.addr_bits(),
        _ => 32,
    };

    // Map result
    let result = phi.as_basic_value();
    ctx.map_value(&result, dst);
    ctx.set_bounds(&result, ValueBounds::from_bits(bits));

    // Create PHI instruction with all incoming values
    let mut phi_inst = MachineInst::phi(dst);

    let num_incoming = phi.count_incoming();
    for i in 0..num_incoming {
        let (value, block) = phi.get_incoming(i)
            .ok_or_else(|| anyhow::anyhow!("PHI incoming {} not found", i))?;
        let src_vreg = ctx.get_vreg(&value);
        let pred_label = block.get_name().to_str().unwrap_or("bb").to_string();
        phi_inst = phi_inst.phi_incoming(&pred_label, src_vreg);
        log::debug!("PHI: {} = {} from {}", dst, src_vreg, pred_label);
    }

    ctx.emit(phi_inst);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_opcodes() {
        // Just verify opcodes exist
        assert!(matches!(Opcode::BEQ, Opcode::BEQ));
        assert!(matches!(Opcode::BNE, Opcode::BNE));
        assert!(matches!(Opcode::JAL, Opcode::JAL));
    }
}
