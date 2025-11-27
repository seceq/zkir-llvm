//! LLVM IR to ZK IR translator
//!
//! This crate provides functionality to translate LLVM IR into ZK IR bytecode,
//! enabling compilation of Rust, C, and C++ programs to zero-knowledge provable code.

pub mod parser;
pub mod ir;
pub mod translate;
pub mod regalloc;
pub mod emit;

pub use emit::emit_program;

use anyhow::Result;

/// Translate LLVM IR source code to ZK IR bytecode
pub fn translate_llvm_ir(source: &str, opt_level: u8) -> Result<zkir_spec::Program> {
    // Parse LLVM IR
    let module = parser::parse(source)?;

    // Translate to ZK IR
    let program = translate::translate_module(&module, opt_level)?;

    Ok(program)
}

/// Check if LLVM IR is compatible with ZK IR
pub fn check_compatibility(source: &str) -> Result<()> {
    let module = parser::parse(source)?;
    translate::check_module_compatibility(&module)?;
    Ok(())
}
