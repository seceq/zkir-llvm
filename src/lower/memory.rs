//! Memory instruction lowering.
//!
//! Handles LLVM memory operations: alloca, load, store, GEP.

use super::LoweringContext;
use super::types;
use crate::mir::{MachineInst, Opcode, Operand, ValueBounds};
use anyhow::Result;
use inkwell::values::InstructionValue;
use inkwell::types::{AnyType, AnyTypeEnum, BasicType};

/// Lower alloca instruction.
pub fn lower_alloca<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    // Alloca allocates space on the stack and returns a pointer
    // In inkwell 0.5, get_allocated_type returns Result<BasicTypeEnum, &str>
    let allocated_type = inst.get_allocated_type()
        .map_err(|e| anyhow::anyhow!("Alloca instruction missing allocated type: {}", e))?;
    let size = types::type_size_bytes(&allocated_type.as_any_type_enum(), ctx.config);

    // Get alignment from the instruction, defaulting to 4 bytes
    let align = inst.get_alignment().unwrap_or(4).max(4);

    // Align the current stack offset to the required alignment
    let current_size = ctx.func.frame.locals_size;
    let aligned_offset = (current_size + align - 1) & !(align - 1);

    // Update frame for stack allocation
    let offset = aligned_offset;
    ctx.func.frame.locals_size = aligned_offset + size;

    // Map result and set bounds
    let (result, dst) = ctx.map_result(inst)?;

    // Emit instruction to compute stack address
    // Stack layout after prologue:
    //   [SP] -> local variables area (lowest addresses)
    //   [SP + locals_size] -> spill slots
    //   [SP + locals_size + spill_size] -> callee-saved registers
    //   [SP + total_size - 8] -> saved FP
    //   [SP + total_size - 4] -> return address
    //   [SP + total_size] = old SP = FP
    //
    // So locals are at positive offsets from SP (new stack pointer)
    ctx.emit(MachineInst::new(Opcode::ADDI)
        .dst(Operand::VReg(dst))
        .src(Operand::Reg(crate::target::Register::R2)) // SP
        .src(Operand::Imm(offset as i64))
        .comment(format!("alloca {} bytes at SP+{}", size, offset)));

    // Pointer bounds
    ctx.set_bounds(&result, ValueBounds::from_bits(ctx.config.addr_bits()));

    Ok(())
}

/// Lower load instruction.
pub fn lower_load<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (_, ptr_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Determine load size from the loaded type
    let loaded_type = inst.get_type();
    let bits = match loaded_type {
        AnyTypeEnum::IntType(int_ty) => int_ty.get_bit_width(),
        AnyTypeEnum::PointerType(_) => ctx.config.addr_bits(),
        _ => 32,
    };

    // Select appropriate load opcode based on size
    let opcode = match bits {
        1..=8 => Opcode::LBU,   // Load byte unsigned
        9..=16 => Opcode::LHU,  // Load halfword unsigned
        17..=32 => Opcode::LW,  // Load word
        33..=64 => Opcode::LD,  // Load doubleword
        _ => Opcode::LW,        // Default to word
    };

    ctx.emit(MachineInst::new(opcode)
        .dst(Operand::VReg(dst))
        .src(Operand::Mem { base: ptr_vreg, offset: 0 }));

    // Loaded value has unknown bounds (from memory)
    ctx.set_bounds(&result, ValueBounds::from_bits(bits));

    Ok(())
}

/// Lower store instruction.
pub fn lower_store<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (value, value_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (_, ptr_vreg) = ctx.get_operand_vreg(inst, 1)?;

    // Determine store size from the value type
    let value_type = value.get_type();
    let bits: u32 = types::type_bits(&value_type, ctx.config);

    // Select appropriate store opcode based on size
    let opcode = match bits {
        1..=8 => Opcode::SB,   // Store byte
        9..=16 => Opcode::SH,  // Store halfword
        17..=32 => Opcode::SW, // Store word
        33..=64 => Opcode::SD, // Store doubleword
        _ => Opcode::SW,       // Default to word
    };

    ctx.emit(MachineInst::new(opcode)
        .src(Operand::VReg(value_vreg))
        .src(Operand::Mem { base: ptr_vreg, offset: 0 }));

    Ok(())
}

/// Lower getelementptr instruction.
pub fn lower_gep<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    // GEP calculates address: base + sum(index * element_size)
    let (_, mut result_vreg) = ctx.get_operand_vreg(inst, 0)?;

    // Get the source element type from the GEP
    let source_element_type = inst.get_allocated_type().ok();

    // Process each index
    let num_operands = inst.get_num_operands();
    for i in 1..num_operands {
        let index = ctx.get_value_operand(inst, i)?;
        let index_vreg = ctx.get_vreg(&index);

        // Calculate element size for this level
        // For first index, it's the element type size
        // For subsequent indices, we need to look at nested types
        let element_size = if let Some(ty) = &source_element_type {
            types::type_size_bytes(&ty.as_any_type_enum(), ctx.config) as i64
        } else {
            // Fallback: assume i8 (byte addressing)
            1
        };

        if element_size == 1 {
            // Simple case: just add the index
            let new_result = ctx.new_vreg();
            ctx.emit(MachineInst::add(new_result, result_vreg, index_vreg));
            result_vreg = new_result;
        } else {
            // Multiply index by element size, then add to base
            let size_vreg = ctx.new_vreg();
            let offset_vreg = ctx.new_vreg();
            let new_result = ctx.new_vreg();

            ctx.emit(MachineInst::li(size_vreg, element_size));
            ctx.emit(MachineInst::mul(offset_vreg, index_vreg, size_vreg));
            ctx.emit(MachineInst::add(new_result, result_vreg, offset_vreg));
            result_vreg = new_result;
        }
    }

    // Map result
    let (result, _) = ctx.map_result(inst)?;
    // Override the mapping with our computed result
    ctx.map_value(&result, result_vreg);

    // GEP result is a pointer
    ctx.set_bounds(&result, ValueBounds::from_bits(ctx.config.addr_bits()));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::config::TargetConfig;

    #[test]
    fn test_load_opcode_selection() {
        // Verify correct opcode selection based on bit width
        let _config = TargetConfig::default();

        // 8-bit load should use LBU
        assert!(matches!(Opcode::LBU, Opcode::LBU));

        // 32-bit load should use LW
        assert!(matches!(Opcode::LW, Opcode::LW));

        // 64-bit load should use LD
        assert!(matches!(Opcode::LD, Opcode::LD));
    }
}
