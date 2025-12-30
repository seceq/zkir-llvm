//! ZKIR LLVM Backend
//!
//! This crate provides an LLVM backend for ZKIR, enabling compilation of
//! C, C++, and Rust programs to ZKIR bytecode for zero-knowledge proof generation.
//!
//! # Architecture
//!
//! The backend follows a multi-stage pipeline:
//!
//! ```text
//! LLVM Bitcode (.bc)
//!        │
//!        ▼ (inkwell)
//! ┌──────────────┐
//! │  LLVM IR     │
//! └──────┬───────┘
//!        │ (lower)
//!        ▼
//! ┌──────────────┐
//! │  Machine IR  │  ← ZK metadata (bounds, range checks)
//! └──────┬───────┘
//!        │ (opt)
//!        ▼
//! ┌──────────────┐
//! │  Optimized   │  ← PHI elimination, range check insertion
//! └──────┬───────┘
//!        │ (regalloc)
//!        ▼
//! ┌──────────────┐
//! │  Allocated   │
//! └──────┬───────┘
//!        │ (emit)
//!        ▼
//! ZKIR Bytecode (.zkir)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use zkir_llvm::{compile, TargetConfig};
//!
//! let config = TargetConfig::default(); // 20-bit × 2 limbs
//! let bytecode = compile("program.bc", &config)?;
//! std::fs::write("program.zkir", bytecode)?;
//! ```
//!
//! # Building
//!
//! This crate requires LLVM to be installed. Enable the appropriate feature:
//!
//! ```bash
//! cargo build --features llvm17-0
//! ```

pub mod target;
pub mod mir;
#[cfg(feature = "inkwell")]
pub mod lower;
pub mod opt;
pub mod regalloc;
pub mod emit;
pub mod stats;
pub mod debug;

pub use target::TargetConfig;
pub use stats::CompileStats;
pub use debug::{DebugInfo, SourceLoc, LineTable};
pub use opt::OptLevel;
pub use emit::OutputMode;

use anyhow::Result;
use std::path::Path;

/// Compile LLVM bitcode to ZKIR bytecode.
///
/// # Arguments
///
/// * `input` - Path to LLVM bitcode file (.bc)
/// * `config` - Target configuration (limb size, data limbs, etc.)
///
/// # Returns
///
/// The compiled ZKIR bytecode as a byte vector.
#[cfg(feature = "inkwell")]
pub fn compile<P: AsRef<Path>>(input: P, config: &TargetConfig) -> Result<Vec<u8>> {
    let (bytecode, _stats) = compile_with_stats(input, config)?;
    Ok(bytecode)
}

/// Compile LLVM bitcode to ZKIR bytecode with statistics.
///
/// Like `compile`, but also returns compilation statistics.
#[cfg(feature = "inkwell")]
pub fn compile_with_stats<P: AsRef<Path>>(input: P, config: &TargetConfig) -> Result<(Vec<u8>, CompileStats)> {
    compile_with_opt(input, config, OptLevel::default())
}

/// Compile LLVM bitcode to ZKIR bytecode with a specific optimization level.
///
/// # Arguments
///
/// * `input` - Path to LLVM bitcode file (.bc)
/// * `config` - Target configuration (limb size, data limbs, etc.)
/// * `opt` - Optimization level (O0, O1, O2, O3, Os)
///
/// # Returns
///
/// The compiled ZKIR bytecode as a byte vector and compilation statistics.
#[cfg(feature = "inkwell")]
pub fn compile_with_opt<P: AsRef<Path>>(
    input: P,
    config: &TargetConfig,
    opt: OptLevel,
) -> Result<(Vec<u8>, CompileStats)> {
    compile_with_mode(input, config, opt, OutputMode::Release)
}

/// Compile LLVM bitcode to ZKIR bytecode with debug symbols.
///
/// Debug mode includes function and global symbol tables for disassembly.
/// Note: Debug output is NOT compatible with zkir-spec's Program::from_bytes().
#[cfg(feature = "inkwell")]
pub fn compile_debug<P: AsRef<Path>>(
    input: P,
    config: &TargetConfig,
    opt: OptLevel,
) -> Result<(Vec<u8>, CompileStats)> {
    compile_with_mode(input, config, opt, OutputMode::Debug)
}

/// Compile LLVM bitcode to ZKIR bytecode with a specific output mode.
///
/// # Arguments
///
/// * `input` - Path to LLVM bitcode file (.bc)
/// * `config` - Target configuration (limb size, data limbs, etc.)
/// * `opt` - Optimization level (O0, O1, O2, O3, Os)
/// * `mode` - Output mode (Release for zkir-spec compatible, Debug for symbol tables)
///
/// # Returns
///
/// The compiled ZKIR bytecode as a byte vector and compilation statistics.
#[cfg(feature = "inkwell")]
pub fn compile_with_mode<P: AsRef<Path>>(
    input: P,
    config: &TargetConfig,
    opt: OptLevel,
    mode: OutputMode,
) -> Result<(Vec<u8>, CompileStats)> {
    use inkwell::context::Context;
    use inkwell::module::Module;
    use stats::Timer;

    let mut stats = CompileStats::new();
    let total_timer = Timer::start();

    // Create LLVM context and load module
    let context = Context::create();
    let module = Module::parse_bitcode_from_path(input.as_ref(), &context)
        .map_err(|e| anyhow::anyhow!("Failed to parse bitcode: {:?}", e))?;

    // Lower LLVM IR to Machine IR
    let lower_timer = Timer::start();
    let mut mir_module = lower::lower_module(&module, config)?;
    stats.lower_time = lower_timer.stop();

    // Collect pre-optimization stats
    stats.num_functions = mir_module.functions.len();
    stats.num_blocks = stats::count_blocks(&mir_module);
    stats.num_insts_before = stats::count_instructions(&mir_module);

    // Run ZK optimization passes
    let opt_timer = Timer::start();
    opt::optimize_with_level(&mut mir_module, config, opt)?;
    stats.opt_time = opt_timer.stop();

    // Collect post-optimization stats
    stats.num_insts_after = stats::count_instructions(&mir_module);
    stats.num_range_checks = stats::count_range_checks(&mir_module);
    stats.num_vregs = stats::count_vregs(&mir_module);

    // Allocate registers
    let regalloc_timer = Timer::start();
    let allocated = regalloc::allocate(&mir_module, config)?;
    stats.regalloc_time = regalloc_timer.stop();

    // Emit ZKIR bytecode
    let emit_timer = Timer::start();
    let bytecode = emit::emit_with_mode(&allocated, config, mode)?;
    stats.emit_time = emit_timer.stop();

    stats.output_size = bytecode.len();
    stats.total_time = total_timer.stop();

    Ok((bytecode, stats))
}

/// Compile and return the Machine IR for debugging.
#[cfg(feature = "inkwell")]
pub fn compile_to_mir<P: AsRef<Path>>(input: P, config: &TargetConfig) -> Result<mir::Module> {
    use inkwell::context::Context;
    use inkwell::module::Module;

    let context = Context::create();
    let module = Module::parse_bitcode_from_path(input.as_ref(), &context)
        .map_err(|e| anyhow::anyhow!("Failed to parse bitcode: {:?}", e))?;

    let mut mir_module = lower::lower_module(&module, config)?;

    // Run optimization passes for debugging output
    opt::optimize(&mut mir_module, config)?;

    Ok(mir_module)
}

/// Stub compile function when LLVM is not available.
#[cfg(not(feature = "inkwell"))]
pub fn compile<P: AsRef<Path>>(_input: P, _config: &TargetConfig) -> Result<Vec<u8>> {
    Err(anyhow::anyhow!(
        "LLVM support not enabled. Build with --features llvm17-0 (or appropriate version)"
    ))
}

/// Stub compile_with_stats function when LLVM is not available.
#[cfg(not(feature = "inkwell"))]
pub fn compile_with_stats<P: AsRef<Path>>(_input: P, _config: &TargetConfig) -> Result<(Vec<u8>, CompileStats)> {
    Err(anyhow::anyhow!(
        "LLVM support not enabled. Build with --features llvm17-0 (or appropriate version)"
    ))
}

/// Stub compile_with_opt function when LLVM is not available.
#[cfg(not(feature = "inkwell"))]
pub fn compile_with_opt<P: AsRef<Path>>(
    _input: P,
    _config: &TargetConfig,
    _opt: OptLevel,
) -> Result<(Vec<u8>, CompileStats)> {
    Err(anyhow::anyhow!(
        "LLVM support not enabled. Build with --features llvm17-0 (or appropriate version)"
    ))
}

/// Stub compile_debug function when LLVM is not available.
#[cfg(not(feature = "inkwell"))]
pub fn compile_debug<P: AsRef<Path>>(
    _input: P,
    _config: &TargetConfig,
    _opt: OptLevel,
) -> Result<(Vec<u8>, CompileStats)> {
    Err(anyhow::anyhow!(
        "LLVM support not enabled. Build with --features llvm17-0 (or appropriate version)"
    ))
}

/// Stub compile_with_mode function when LLVM is not available.
#[cfg(not(feature = "inkwell"))]
pub fn compile_with_mode<P: AsRef<Path>>(
    _input: P,
    _config: &TargetConfig,
    _opt: OptLevel,
    _mode: OutputMode,
) -> Result<(Vec<u8>, CompileStats)> {
    Err(anyhow::anyhow!(
        "LLVM support not enabled. Build with --features llvm17-0 (or appropriate version)"
    ))
}

/// Stub compile_to_mir function when LLVM is not available.
#[cfg(not(feature = "inkwell"))]
pub fn compile_to_mir<P: AsRef<Path>>(_input: P, _config: &TargetConfig) -> Result<mir::Module> {
    Err(anyhow::anyhow!(
        "LLVM support not enabled. Build with --features llvm17-0 (or appropriate version)"
    ))
}
