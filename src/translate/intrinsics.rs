//! LLVM intrinsic translation

use super::{context::TranslationContext, TranslateResult};
use crate::ir::Value;

/// Translate LLVM intrinsic function call
pub fn translate_intrinsic(
    ctx: &mut TranslationContext,
    name: &str,
    args: &[Value],
    result: Option<&str>,
) -> TranslateResult<()> {
    match name {
        // Memory intrinsics
        "llvm.memcpy" | "llvm.memmove" | "llvm.memset" => {
            // TODO: Implement memory intrinsics
            Err(super::TranslateError::UnsupportedInstruction(format!(
                "intrinsic {}",
                name
            )))
        }

        // Math intrinsics
        "llvm.sqrt" | "llvm.sin" | "llvm.cos" | "llvm.exp" | "llvm.log" => {
            // Math intrinsics not supported in ZK IR
            Err(super::TranslateError::UnsupportedInstruction(format!(
                "intrinsic {}",
                name
            )))
        }

        // Overflow intrinsics
        "llvm.sadd.with.overflow" | "llvm.uadd.with.overflow" | "llvm.ssub.with.overflow"
        | "llvm.usub.with.overflow" => {
            // TODO: Implement overflow checking
            Err(super::TranslateError::UnsupportedInstruction(format!(
                "intrinsic {}",
                name
            )))
        }

        // Debug intrinsics (ignore)
        "llvm.dbg.declare" | "llvm.dbg.value" | "llvm.dbg.label" => {
            // Ignore debug intrinsics
            Ok(())
        }

        // Lifetime intrinsics (ignore)
        "llvm.lifetime.start" | "llvm.lifetime.end" => {
            // Ignore lifetime intrinsics
            Ok(())
        }

        _ => Err(super::TranslateError::UnsupportedInstruction(format!(
            "intrinsic {}",
            name
        ))),
    }
}
