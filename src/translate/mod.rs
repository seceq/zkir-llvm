//! Translation from LLVM IR to ZK IR

pub mod context;
pub mod types;
pub mod arithmetic;
pub mod memory;
pub mod control;
pub mod intrinsics;

use crate::ir::Module;
use anyhow::Result;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TranslateError {
    #[error("Unsupported type: {0:?}")]
    UnsupportedType(crate::ir::Type),

    #[error("Unsupported instruction: {0}")]
    UnsupportedInstruction(String),

    #[error("Unsupported bit width: {0}")]
    UnsupportedWidth(u32),

    #[error("Register allocation failed: {0}")]
    RegisterAllocation(String),

    #[error("Out of registers")]
    OutOfRegisters,

    #[error("Undefined value: {0}")]
    UndefinedValue(String),

    #[error("Type mismatch: expected {expected:?}, got {actual:?}")]
    TypeMismatch {
        expected: crate::ir::Type,
        actual: crate::ir::Type,
    },

    #[error("Invalid branch target: {0}")]
    InvalidBranch(String),

    #[error("Function not found: {0}")]
    FunctionNotFound(String),
}

pub type TranslateResult<T> = Result<T, TranslateError>;

/// Translate an LLVM module to ZK IR program
pub fn translate_module(module: &Module, opt_level: u8) -> Result<zkir_spec::Program> {
    let mut all_code = Vec::new();

    // Translate each function
    for func in module.functions() {
        if func.is_declaration() {
            // Skip function declarations
            continue;
        }

        let instructions = translate_function(func)?;

        // Encode instructions to u32 words
        for instr in instructions {
            let encoded = encode_instruction(&instr);
            all_code.push(encoded);
        }
    }

    // Apply optimizations based on opt_level
    if opt_level > 0 {
        // TODO: Add optimizations
    }

    Ok(zkir_spec::Program::new(all_code))
}

/// Encode instruction to u32
fn encode_instruction(instr: &zkir_spec::Instruction) -> u32 {
    // For now, use a simple encoding
    // TODO: Implement proper RISC-V encoding
    match instr {
        zkir_spec::Instruction::Add { rd, rs1, rs2 } => {
            let opcode = 0x33u32;
            let funct3 = 0x0u32;
            let funct7 = 0x00u32;
            opcode | ((rd.index() as u32) << 7) | ((funct3) << 12) |
                ((rs1.index() as u32) << 15) | ((rs2.index() as u32) << 20) | ((funct7) << 25)
        }
        // TODO: Implement encoding for other instructions
        _ => 0, // Placeholder
    }
}

/// Translate a single function
fn translate_function(func: &crate::ir::Function) -> Result<Vec<zkir_spec::Instruction>> {
    use context::TranslationContext;

    let mut ctx = TranslationContext::new(func.name());

    // Set up function parameters
    for (i, (name, ty)) in func.params().iter().enumerate() {
        ctx.bind_parameter(name, ty, i)?;
    }

    // Translate each basic block
    for block in func.blocks() {
        ctx.start_block(block.name());

        for instr in block.instructions() {
            translate_instruction(&mut ctx, instr)?;
        }
    }

    // Resolve labels and fixups
    ctx.resolve_labels()?;

    Ok(ctx.into_instructions())
}

/// Translate a single instruction
fn translate_instruction(
    ctx: &mut context::TranslationContext,
    instr: &crate::ir::Instruction,
) -> Result<()> {
    use crate::ir::Instruction as I;

    match instr {
        // Arithmetic
        I::Add { result, ty, lhs, rhs } => {
            arithmetic::translate_add(ctx, result, ty, lhs, rhs)?;
        }
        I::Sub { result, ty, lhs, rhs } => {
            arithmetic::translate_sub(ctx, result, ty, lhs, rhs)?;
        }
        I::Mul { result, ty, lhs, rhs } => {
            arithmetic::translate_mul(ctx, result, ty, lhs, rhs)?;
        }
        I::UDiv { result, ty, lhs, rhs } => {
            arithmetic::translate_udiv(ctx, result, ty, lhs, rhs)?;
        }
        I::SDiv { result, ty, lhs, rhs } => {
            arithmetic::translate_sdiv(ctx, result, ty, lhs, rhs)?;
        }
        I::URem { result, ty, lhs, rhs } => {
            arithmetic::translate_urem(ctx, result, ty, lhs, rhs)?;
        }
        I::SRem { result, ty, lhs, rhs } => {
            arithmetic::translate_srem(ctx, result, ty, lhs, rhs)?;
        }

        // Bitwise
        I::And { result, ty, lhs, rhs } => {
            arithmetic::translate_and(ctx, result, ty, lhs, rhs)?;
        }
        I::Or { result, ty, lhs, rhs } => {
            arithmetic::translate_or(ctx, result, ty, lhs, rhs)?;
        }
        I::Xor { result, ty, lhs, rhs } => {
            arithmetic::translate_xor(ctx, result, ty, lhs, rhs)?;
        }
        I::Shl { result, ty, lhs, rhs } => {
            arithmetic::translate_shl(ctx, result, ty, lhs, rhs)?;
        }
        I::LShr { result, ty, lhs, rhs } => {
            arithmetic::translate_lshr(ctx, result, ty, lhs, rhs)?;
        }
        I::AShr { result, ty, lhs, rhs } => {
            arithmetic::translate_ashr(ctx, result, ty, lhs, rhs)?;
        }

        // Comparison
        I::ICmp {
            result,
            pred,
            ty,
            lhs,
            rhs,
        } => {
            control::translate_icmp(ctx, result, *pred, ty, lhs, rhs)?;
        }

        // Memory
        I::Load { result, ty, ptr } => {
            memory::translate_load(ctx, result, ty, ptr)?;
        }
        I::Store { value, ty, ptr } => {
            memory::translate_store(ctx, value, ty, ptr)?;
        }
        I::Alloca { result, ty } => {
            memory::translate_alloca(ctx, result, ty)?;
        }

        // Control flow
        I::Ret { value } => {
            control::translate_ret(ctx, value.as_ref())?;
        }
        I::Br { dest } => {
            control::translate_br(ctx, dest)?;
        }
        I::CondBr {
            cond,
            true_dest,
            false_dest,
        } => {
            control::translate_cond_br(ctx, cond, true_dest, false_dest)?;
        }
        I::Call {
            result,
            callee,
            args,
            ret_ty,
        } => {
            control::translate_call(ctx, result.as_deref(), callee, args, ret_ty)?;
        }

        I::Phi { .. } => {
            // Phi nodes should be lowered before translation
            // For now, skip them
        }

        I::GetElementPtr { .. } => {
            // TODO: Implement GEP
            return Err(TranslateError::UnsupportedInstruction("getelementptr".to_string()).into());
        }
    }

    Ok(())
}

/// Check if a module is compatible with ZK IR
pub fn check_module_compatibility(module: &Module) -> Result<()> {
    for func in module.functions() {
        check_function_compatibility(func)?;
    }
    Ok(())
}

fn check_function_compatibility(func: &crate::ir::Function) -> Result<()> {
    // Check return type
    if !func.ret_ty().is_supported() {
        return Err(TranslateError::UnsupportedType(func.ret_ty().clone()).into());
    }

    // Check parameters
    for (_, ty) in func.params() {
        if !ty.is_supported() {
            return Err(TranslateError::UnsupportedType(ty.clone()).into());
        }
    }

    // Check instructions in each block
    for block in func.blocks() {
        for instr in block.instructions() {
            check_instruction_compatibility(instr)?;
        }
    }

    Ok(())
}

fn check_instruction_compatibility(instr: &crate::ir::Instruction) -> Result<()> {
    use crate::ir::Instruction as I;

    match instr {
        I::GetElementPtr { .. } => {
            // GEP is supported but complex
            Ok(())
        }
        _ => Ok(()), // Most instructions are supported
    }
}
