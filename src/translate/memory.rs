//! Memory instruction translation

use super::{context::TranslationContext, TranslateResult};
use crate::ir::{Type, Value};
use zkir_spec::{Instruction, Register};

/// Translate load instruction
pub fn translate_load(
    ctx: &mut TranslationContext,
    result: &str,
    ty: &Type,
    ptr: &Value,
) -> TranslateResult<()> {
    let ptr_reg = ctx.load_value(ptr)?;
    let bits = ty.bit_width();

    match bits {
        8 => {
            let rd = ctx.alloc_temp()?;
            ctx.emit(Instruction::Lbu {
                rd,
                rs1: ptr_reg,
                imm: 0,
            });
            ctx.bind(result, super::context::Location::Reg(rd));
        }
        16 => {
            let rd = ctx.alloc_temp()?;
            ctx.emit(Instruction::Lhu {
                rd,
                rs1: ptr_reg,
                imm: 0,
            });
            ctx.bind(result, super::context::Location::Reg(rd));
        }
        32 => {
            let rd = ctx.alloc_temp()?;
            ctx.emit(Instruction::Lw {
                rd,
                rs1: ptr_reg,
                imm: 0,
            });
            ctx.bind(result, super::context::Location::Reg(rd));
        }
        64 => {
            let (rd_lo, rd_hi) = ctx.alloc_reg_pair()?;
            ctx.emit(Instruction::Lw {
                rd: rd_lo,
                rs1: ptr_reg,
                imm: 0,
            });
            ctx.emit(Instruction::Lw {
                rd: rd_hi,
                rs1: ptr_reg,
                imm: 4,
            });
            ctx.bind(
                result,
                super::context::Location::RegPair {
                    lo: rd_lo,
                    hi: rd_hi,
                },
            );
        }
        _ => {
            return Err(super::TranslateError::UnsupportedWidth(bits));
        }
    }

    Ok(())
}

/// Translate store instruction
pub fn translate_store(
    ctx: &mut TranslationContext,
    value: &Value,
    ty: &Type,
    ptr: &Value,
) -> TranslateResult<()> {
    let ptr_reg = ctx.load_value(ptr)?;
    let bits = ty.bit_width();

    match bits {
        8 => {
            let val_reg = ctx.load_value(value)?;
            ctx.emit(Instruction::Sb {
                rs1: ptr_reg,
                rs2: val_reg,
                imm: 0,
            });
        }
        16 => {
            let val_reg = ctx.load_value(value)?;
            ctx.emit(Instruction::Sh {
                rs1: ptr_reg,
                rs2: val_reg,
                imm: 0,
            });
        }
        32 => {
            let val_reg = ctx.load_value(value)?;
            ctx.emit(Instruction::Sw {
                rs1: ptr_reg,
                rs2: val_reg,
                imm: 0,
            });
        }
        64 => {
            let (val_lo, val_hi) = ctx.load_value_pair(value)?;
            ctx.emit(Instruction::Sw {
                rs1: ptr_reg,
                rs2: val_lo,
                imm: 0,
            });
            ctx.emit(Instruction::Sw {
                rs1: ptr_reg,
                rs2: val_hi,
                imm: 4,
            });
        }
        _ => {
            return Err(super::TranslateError::UnsupportedWidth(bits));
        }
    }

    Ok(())
}

/// Translate alloca instruction
pub fn translate_alloca(
    ctx: &mut TranslationContext,
    result: &str,
    ty: &Type,
) -> TranslateResult<()> {
    // Allocate stack space
    let size = ty.size_in_bytes() as i32;
    let offset = ctx.alloc_stack(size);

    // Compute address: fp + offset
    let rd = ctx.alloc_temp()?;
    ctx.emit(Instruction::Addi {
        rd,
        rs1: Register::FP,
        imm: offset as i16,
    });

    ctx.bind(result, super::context::Location::Reg(rd));

    Ok(())
}
