//! Type lowering and conversions.
//!
//! Handles LLVM type sizes, alignments, and conversion instructions.

use super::LoweringContext;
use crate::mir::{MachineInst, Opcode, Operand, ValueBounds};
use crate::target::config::TargetConfig;
use anyhow::Result;
use inkwell::types::{AnyType, AnyTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::InstructionValue;

/// Get the size of a type in bits.
pub fn type_bits(ty: &BasicTypeEnum, config: &TargetConfig) -> u32 {
    match ty {
        BasicTypeEnum::IntType(int_ty) => int_ty.get_bit_width(),
        BasicTypeEnum::FloatType(_) => 32,
        BasicTypeEnum::PointerType(_) => config.addr_bits(),
        BasicTypeEnum::ArrayType(arr_ty) => {
            let elem_ty = arr_ty.get_element_type();
            let elem_bits = type_bits(&elem_ty, config);
            elem_bits * arr_ty.len()
        }
        BasicTypeEnum::StructType(struct_ty) => {
            let mut total = 0u32;
            for i in 0..struct_ty.count_fields() {
                if let Some(field_ty) = struct_ty.get_field_type_at_index(i) {
                    total += type_bits(&field_ty, config);
                }
            }
            total
        }
        BasicTypeEnum::VectorType(vec_ty) => {
            let elem_ty = vec_ty.get_element_type();
            let elem_bits = type_bits(&elem_ty, config);
            elem_bits * vec_ty.get_size()
        }
        BasicTypeEnum::ScalableVectorType(vec_ty) => {
            // Scalable vectors have a minimum size that scales at runtime
            // For ZK, we use the minimum guaranteed size
            let elem_ty = vec_ty.get_element_type();
            let elem_bits = type_bits(&elem_ty, config);
            elem_bits * vec_ty.get_size() // Minimum elements
        }
    }
}

/// Get the size of a type in bytes.
pub fn type_size_bytes(ty: &AnyTypeEnum, config: &TargetConfig) -> u32 {
    let bits = match ty {
        AnyTypeEnum::IntType(int_ty) => int_ty.get_bit_width(),
        AnyTypeEnum::FloatType(_) => 32,
        AnyTypeEnum::PointerType(_) => config.addr_bits(),
        AnyTypeEnum::ArrayType(arr_ty) => {
            let elem_ty = arr_ty.get_element_type().as_any_type_enum();
            let elem_bytes = type_size_bytes(&elem_ty, config);
            elem_bytes * arr_ty.len() * 8
        }
        AnyTypeEnum::StructType(struct_ty) => {
            let mut total = 0u32;
            for i in 0..struct_ty.count_fields() {
                if let Some(field_ty) = struct_ty.get_field_type_at_index(i) {
                    let field_ty = field_ty.as_any_type_enum();
                    total += type_size_bytes(&field_ty, config);
                }
            }
            return total;
        }
        AnyTypeEnum::VectorType(vec_ty) => {
            let elem_ty = vec_ty.get_element_type().as_any_type_enum();
            let elem_bytes = type_size_bytes(&elem_ty, config);
            elem_bytes * vec_ty.get_size() * 8
        }
        AnyTypeEnum::VoidType(_) => 0,
        AnyTypeEnum::FunctionType(_) => config.addr_bits(),
        AnyTypeEnum::ScalableVectorType(vec_ty) => {
            let elem_ty = vec_ty.get_element_type().as_any_type_enum();
            let elem_bytes = type_size_bytes(&elem_ty, config);
            elem_bytes * vec_ty.get_size() * 8 // Minimum elements
        }
    };
    (bits + 7) / 8
}

/// Get the alignment of a type in bytes.
pub fn type_align(ty: &AnyTypeEnum, config: &TargetConfig) -> u32 {
    match ty {
        AnyTypeEnum::IntType(int_ty) => {
            let bits = int_ty.get_bit_width();
            ((bits + 7) / 8).min(8) // Max 8-byte alignment
        }
        AnyTypeEnum::FloatType(_) => 4,
        AnyTypeEnum::PointerType(_) => (config.addr_bits() + 7) / 8,
        AnyTypeEnum::ArrayType(arr_ty) => {
            let elem_ty = arr_ty.get_element_type().as_any_type_enum();
            type_align(&elem_ty, config)
        }
        AnyTypeEnum::StructType(struct_ty) => {
            let mut max_align = 1u32;
            for i in 0..struct_ty.count_fields() {
                if let Some(field_ty) = struct_ty.get_field_type_at_index(i) {
                    let field_ty = field_ty.as_any_type_enum();
                    max_align = max_align.max(type_align(&field_ty, config));
                }
            }
            max_align
        }
        AnyTypeEnum::VectorType(vec_ty) => {
            let elem_ty = vec_ty.get_element_type().as_any_type_enum();
            type_align(&elem_ty, config)
        }
        AnyTypeEnum::ScalableVectorType(vec_ty) => {
            let elem_ty = vec_ty.get_element_type().as_any_type_enum();
            type_align(&elem_ty, config)
        }
        AnyTypeEnum::VoidType(_) => 1,
        AnyTypeEnum::FunctionType(_) => 1,
    }
}

/// Lower LLVM types to Machine IR type representation.
pub fn lower_type(ty: &BasicTypeEnum, config: &TargetConfig) -> TypeInfo {
    let bits = type_bits(ty, config);
    let regs_needed = config.regs_for_bits(bits);

    TypeInfo {
        bits,
        regs_needed,
        is_pointer: matches!(ty, BasicTypeEnum::PointerType(_)),
        is_aggregate: matches!(ty, BasicTypeEnum::ArrayType(_) | BasicTypeEnum::StructType(_)),
    }
}

/// Information about a lowered type.
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// Total bits in the type
    pub bits: u32,
    /// Number of registers needed
    pub regs_needed: u32,
    /// Is this a pointer type
    pub is_pointer: bool,
    /// Is this an aggregate (array/struct)
    pub is_aggregate: bool,
}

/// Lower trunc instruction.
pub fn lower_trunc<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (operand, src) = ctx.get_operand_vreg(inst, 0)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Get destination type bit width
    let dst_ty = inst.get_type();
    let dst_bits = if let AnyTypeEnum::IntType(int_ty) = dst_ty {
        int_ty.get_bit_width()
    } else {
        32
    };

    // Truncation can be done with a mask
    let mask = if dst_bits >= 64 {
        -1i64
    } else {
        (1i64 << dst_bits) - 1
    };

    // AND with mask to truncate
    let mask_vreg = ctx.new_vreg();
    ctx.emit(MachineInst::li(mask_vreg, mask));
    ctx.emit(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(src))
        .src(Operand::VReg(mask_vreg)));

    // Update bounds
    let src_bounds = ctx.get_bounds(&operand);
    ctx.set_bounds(&result, src_bounds.trunc(dst_bits));

    Ok(())
}

/// Lower zext instruction.
pub fn lower_zext<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (operand, src) = ctx.get_operand_vreg(inst, 0)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Get source type bit width for masking
    let src_ty = operand.get_type();
    let src_bits = type_bits(&src_ty, ctx.config);

    // Zero extension: mask to source width to ensure high bits are zero
    if src_bits < 64 {
        let mask = (1i64 << src_bits) - 1;
        let mask_vreg = ctx.new_vreg();
        ctx.emit(MachineInst::li(mask_vreg, mask));
        ctx.emit(MachineInst::new(Opcode::AND)
            .dst(Operand::VReg(dst))
            .src(Operand::VReg(src))
            .src(Operand::VReg(mask_vreg)));
    } else {
        // Just move, value is already zero-extended
        ctx.emit(MachineInst::mov(dst, src));
    }

    // Bounds don't change for zero extension
    let src_bounds = ctx.get_bounds(&operand);
    let dst_ty = inst.get_type();
    let dst_bits = if let AnyTypeEnum::IntType(int_ty) = dst_ty {
        int_ty.get_bit_width()
    } else {
        ctx.config.data_bits()
    };
    ctx.set_bounds(&result, src_bounds.zext(dst_bits));

    Ok(())
}

/// Lower sext instruction.
pub fn lower_sext<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (operand, src) = ctx.get_operand_vreg(inst, 0)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Get source type bit width
    let src_ty = operand.get_type();
    let src_bits = type_bits(&src_ty, ctx.config);

    // Sign extension sequence:
    // 1. Shift left to put sign bit at MSB
    // 2. Arithmetic shift right to sign-extend

    let data_bits = ctx.config.data_bits();
    let shift_amount = (data_bits - src_bits) as i64;

    if shift_amount > 0 {
        let shift_vreg = ctx.new_vreg();
        let temp = ctx.new_vreg();
        ctx.emit(MachineInst::li(shift_vreg, shift_amount));

        // Shift left
        ctx.emit(MachineInst::new(Opcode::SLL)
            .dst(Operand::VReg(temp))
            .src(Operand::VReg(src))
            .src(Operand::VReg(shift_vreg)));

        // Arithmetic shift right
        ctx.emit(MachineInst::new(Opcode::SRA)
            .dst(Operand::VReg(dst))
            .src(Operand::VReg(temp))
            .src(Operand::VReg(shift_vreg)));
    } else {
        // No shift needed
        ctx.emit(MachineInst::mov(dst, src));
    }

    // After sign extension, we don't know the bounds precisely (could be negative)
    let dst_ty = inst.get_type();
    let dst_bits = if let AnyTypeEnum::IntType(int_ty) = dst_ty {
        int_ty.get_bit_width()
    } else {
        ctx.config.data_bits()
    };
    ctx.set_bounds(&result, ValueBounds::from_bits(dst_bits));

    Ok(())
}

/// Lower ptrtoint instruction.
pub fn lower_ptrtoint<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (_, src) = ctx.get_operand_vreg(inst, 0)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Pointer to integer is just a move (pointers are just integers)
    ctx.emit(MachineInst::mov(dst, src));

    // Set bounds to address width
    ctx.set_bounds(&result, ValueBounds::from_bits(ctx.config.addr_bits()));

    Ok(())
}

/// Lower inttoptr instruction.
pub fn lower_inttoptr<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (_, src) = ctx.get_operand_vreg(inst, 0)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Integer to pointer is just a move
    ctx.emit(MachineInst::mov(dst, src));

    // Set bounds to address width
    ctx.set_bounds(&result, ValueBounds::from_bits(ctx.config.addr_bits()));

    Ok(())
}

/// Lower bitcast instruction.
pub fn lower_bitcast<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (operand, src) = ctx.get_operand_vreg(inst, 0)?;
    let (result, dst) = ctx.map_result(inst)?;

    // Bitcast is a no-op at the machine level (just reinterprets bits)
    ctx.emit(MachineInst::mov(dst, src));

    // Preserve bounds from source
    let src_bounds = ctx.get_bounds(&operand);
    ctx.set_bounds(&result, src_bounds);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_bits() {
        use inkwell::context::Context;
        let context = Context::create();
        let config = TargetConfig::default();

        let i32_ty = BasicTypeEnum::IntType(context.i32_type());
        assert_eq!(type_bits(&i32_ty, &config), 32);

        let i64_ty = BasicTypeEnum::IntType(context.i64_type());
        assert_eq!(type_bits(&i64_ty, &config), 64);

        let i8_ty = BasicTypeEnum::IntType(context.i8_type());
        assert_eq!(type_bits(&i8_ty, &config), 8);
    }

    #[test]
    fn test_type_info() {
        use inkwell::context::Context;
        let context = Context::create();
        let config = TargetConfig::default(); // 40-bit

        let i32_ty = BasicTypeEnum::IntType(context.i32_type());
        let info = lower_type(&i32_ty, &config);
        assert_eq!(info.bits, 32);
        assert_eq!(info.regs_needed, 1); // 32 bits fits in 40-bit data width
        assert!(!info.is_pointer);

        let i64_ty = BasicTypeEnum::IntType(context.i64_type());
        let info = lower_type(&i64_ty, &config);
        assert_eq!(info.bits, 64);
        assert_eq!(info.regs_needed, 2); // 64 bits needs 2 registers in 40-bit mode
    }
}
