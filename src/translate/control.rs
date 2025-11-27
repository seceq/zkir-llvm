//! Control flow instruction translation

use super::{context::TranslationContext, TranslateResult};
use crate::ir::{instruction::ICmpPredicate, Type, Value};
use zkir_spec::{Instruction, Register};

/// Translate unconditional branch
pub fn translate_br(ctx: &mut TranslationContext, dest: &str) -> TranslateResult<()> {
    // JAL zero, offset (unconditional jump)
    ctx.emit(Instruction::Jal {
        rd: Register::ZERO,
        imm: 0, // Will be fixed up later
    });
    ctx.add_fixup(dest);

    Ok(())
}

/// Translate conditional branch
pub fn translate_cond_br(
    ctx: &mut TranslationContext,
    cond: &Value,
    true_dest: &str,
    false_dest: &str,
) -> TranslateResult<()> {
    let cond_reg = ctx.load_value(cond)?;

    // BNE cond, zero, true_dest
    ctx.emit(Instruction::Bne {
        rs1: cond_reg,
        rs2: Register::ZERO,
        imm: 0, // Will be fixed up
    });
    ctx.add_fixup(true_dest);

    // JAL zero, false_dest
    ctx.emit(Instruction::Jal {
        rd: Register::ZERO,
        imm: 0, // Will be fixed up
    });
    ctx.add_fixup(false_dest);

    Ok(())
}

/// Translate return instruction
pub fn translate_ret(ctx: &mut TranslationContext, value: Option<&Value>) -> TranslateResult<()> {
    // Move return value to rv (r1) if present
    if let Some(val) = value {
        let reg = ctx.load_value(val)?;
        if reg != Register::RV {
            ctx.emit(Instruction::Add {
                rd: Register::RV,
                rs1: reg,
                rs2: Register::ZERO,
            });
        }
    }

    // Return: JALR zero, ra, 0
    ctx.emit(Instruction::Jalr {
        rd: Register::ZERO,
        rs1: Register::RA,
        imm: 0,
    });

    Ok(())
}

/// Translate comparison instruction
pub fn translate_icmp(
    ctx: &mut TranslationContext,
    result: &str,
    pred: ICmpPredicate,
    ty: &Type,
    lhs: &Value,
    rhs: &Value,
) -> TranslateResult<()> {
    let rs1 = ctx.load_value(lhs)?;
    let rs2 = ctx.load_value(rhs)?;
    let rd = ctx.alloc_temp()?;

    match pred {
        ICmpPredicate::Eq => {
            // rd = (rs1 == rs2)
            // SUB tmp, rs1, rs2; SLTIU rd, tmp, 1
            let tmp = ctx.alloc_temp()?;
            ctx.emit(Instruction::Sub {
                rd: tmp,
                rs1,
                rs2,
            });
            ctx.emit(Instruction::Sltiu {
                rd,
                rs1: tmp,
                imm: 1,
            });
        }
        ICmpPredicate::Ne => {
            // rd = (rs1 != rs2)
            let tmp = ctx.alloc_temp()?;
            ctx.emit(Instruction::Sub {
                rd: tmp,
                rs1,
                rs2,
            });
            ctx.emit(Instruction::Sltu {
                rd,
                rs1: Register::ZERO,
                rs2: tmp,
            });
        }
        ICmpPredicate::Slt => {
            // rd = (rs1 < rs2) signed
            ctx.emit(Instruction::Slt { rd, rs1, rs2 });
        }
        ICmpPredicate::Sle => {
            // rd = !(rs2 < rs1)
            let tmp = ctx.alloc_temp()?;
            ctx.emit(Instruction::Slt {
                rd: tmp,
                rs1: rs2,
                rs2: rs1,
            });
            ctx.emit(Instruction::Xori {
                rd,
                rs1: tmp,
                imm: 1,
            });
        }
        ICmpPredicate::Sgt => {
            // rd = (rs2 < rs1)
            ctx.emit(Instruction::Slt {
                rd,
                rs1: rs2,
                rs2: rs1,
            });
        }
        ICmpPredicate::Sge => {
            // rd = !(rs1 < rs2)
            let tmp = ctx.alloc_temp()?;
            ctx.emit(Instruction::Slt {
                rd: tmp,
                rs1,
                rs2,
            });
            ctx.emit(Instruction::Xori {
                rd,
                rs1: tmp,
                imm: 1,
            });
        }
        ICmpPredicate::Ult => {
            // rd = (rs1 < rs2) unsigned
            ctx.emit(Instruction::Sltu { rd, rs1, rs2 });
        }
        ICmpPredicate::Ule => {
            // rd = !(rs2 < rs1) unsigned
            let tmp = ctx.alloc_temp()?;
            ctx.emit(Instruction::Sltu {
                rd: tmp,
                rs1: rs2,
                rs2: rs1,
            });
            ctx.emit(Instruction::Xori {
                rd,
                rs1: tmp,
                imm: 1,
            });
        }
        ICmpPredicate::Ugt => {
            // rd = (rs2 < rs1) unsigned
            ctx.emit(Instruction::Sltu {
                rd,
                rs1: rs2,
                rs2: rs1,
            });
        }
        ICmpPredicate::Uge => {
            // rd = !(rs1 < rs2) unsigned
            let tmp = ctx.alloc_temp()?;
            ctx.emit(Instruction::Sltu {
                rd: tmp,
                rs1,
                rs2,
            });
            ctx.emit(Instruction::Xori {
                rd,
                rs1: tmp,
                imm: 1,
            });
        }
    }

    ctx.bind(result, super::context::Location::Reg(rd));

    Ok(())
}

/// Translate function call
pub fn translate_call(
    ctx: &mut TranslationContext,
    result: Option<&str>,
    callee: &str,
    args: &[Value],
    ret_ty: &Type,
) -> TranslateResult<()> {
    // Place arguments in a0-a3, rest on stack
    for (i, arg) in args.iter().enumerate() {
        if i < 4 {
            let src = ctx.load_value(arg)?;
            let dst = Register::from_index(4 + i).unwrap(); // a0-a3
            if src != dst {
                ctx.emit(Instruction::Add {
                    rd: dst,
                    rs1: src,
                    rs2: Register::ZERO,
                });
            }
        } else {
            // Push to stack
            let src = ctx.load_value(arg)?;
            let offset = -((i - 3) as i16 * 4);
            ctx.emit(Instruction::Sw {
                rs1: Register::SP,
                rs2: src,
                imm: offset,
            });
        }
    }

    // Call function
    ctx.emit(Instruction::Jal {
        rd: Register::RA,
        imm: 0, // Will be fixed up
    });
    ctx.add_fixup(callee);

    // Handle return value
    if let Some(result) = result {
        let bits = ret_ty.bit_width();
        if bits <= 32 {
            ctx.bind(result, super::context::Location::Reg(Register::RV));
        } else if bits == 64 {
            ctx.bind(
                result,
                super::context::Location::RegPair {
                    lo: Register::RV,
                    hi: Register::A0,
                },
            );
        }
    }

    Ok(())
}
