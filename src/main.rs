//! ZKIR LLVM Backend CLI
//!
//! Compiles LLVM bitcode (.bc) to ZKIR bytecode (.zkir).
//!
//! # Usage
//!
//! ```bash
//! # Basic usage
//! zkir-llvm input.bc -o output.zkir
//!
//! # With custom configuration
//! zkir-llvm input.bc -o output.zkir --limb-bits 20 --data-limbs 3
//!
//! # Dump Machine IR for debugging
//! zkir-llvm input.bc --emit-mir
//! ```

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EmitType {
    /// Emit ZKIR bytecode (default)
    Zkir,
    /// Emit Machine IR for debugging
    Mir,
    /// Emit assembly-like text representation
    Asm,
    /// Disassemble existing ZKIR bytecode
    Disasm,
}

#[derive(Parser, Debug)]
#[command(
    name = "zkir-llvm",
    version,
    about = "LLVM backend for ZKIR - compiles LLVM bitcode to ZK IR bytecode",
    long_about = r#"
ZKIR LLVM Backend

Compiles LLVM bitcode (.bc) files to ZKIR bytecode (.zkir) for
zero-knowledge proof generation.

WORKFLOW:
  1. Compile source to LLVM bitcode:
     clang -O2 -emit-llvm -c program.c -o program.bc

  2. Run this backend:
     zkir-llvm program.bc -o program.zkir

OPTIMIZATION LEVELS:
  -O0  No optimizations (fastest compilation)
  -O1  Basic optimizations (CSE, copy propagation)
  -O2  Standard optimizations (default - LICM, loop unrolling, global CSE)
  -O3  Aggressive optimizations (all O2 + more aggressive settings)
  -Os  Size-optimized (minimize constraint count)

CONFIGURATION:
  ZKIR supports variable limb sizes for different use cases:

  --limb-bits 20 --data-limbs 2  (40-bit, good for i32)
  --limb-bits 20 --data-limbs 3  (60-bit, good for i64)
  --limb-bits 20 --data-limbs 4  (80-bit, i64 with headroom)
  --limb-bits 30 --data-limbs 2  (60-bit, large limbs)

PRESETS:
  --preset default      40-bit (good for i32)
  --preset 60bit        60-bit (good for i64)
  --preset 80bit        80-bit (i64 with headroom)
  --preset i32-compact  36-bit (minimal for i32)
  --preset large-limb   60-bit (30-bit limbs, fewer range checks)
"#
)]
struct Args {
    /// Input LLVM bitcode file (.bc)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file path
    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Use a preset configuration (overrides limb-bits/data-limbs/addr-limbs)
    #[arg(long, value_name = "NAME")]
    preset: Option<String>,

    /// Bits per limb (16, 18, 20, 22, 24, 26, 28, or 30)
    #[arg(long, default_value = "20", value_parser = clap::value_parser!(u8).range(16..=30))]
    limb_bits: u8,

    /// Number of limbs for data values (1, 2, 3, or 4)
    #[arg(long, default_value = "2", value_parser = clap::value_parser!(u8).range(1..=4))]
    data_limbs: u8,

    /// Number of limbs for addresses (1 or 2)
    #[arg(long, default_value = "2", value_parser = clap::value_parser!(u8).range(1..=2))]
    addr_limbs: u8,

    /// Output type
    #[arg(long, value_enum, default_value = "zkir")]
    emit: EmitType,

    /// Optimization level (0, 1, 2, 3, or s)
    #[arg(short = 'O', long = "opt-level", default_value = "2")]
    opt_level: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Suppress configuration warnings
    #[arg(long)]
    no_warnings: bool,

    /// Emit debug symbols (function/global tables for disassembly)
    /// Note: Debug output is NOT compatible with zkir-spec's Program::from_bytes()
    #[arg(long, short = 'g')]
    debug: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    }

    // Parse optimization level
    let opt_level = zkir_llvm::OptLevel::from_str(&args.opt_level)
        .ok_or_else(|| anyhow::anyhow!(
            "Invalid optimization level '{}'. Use 0, 1, 2, 3, or s.",
            args.opt_level
        ))?;

    // Build target configuration (preset overrides individual settings)
    let config = if let Some(preset_name) = &args.preset {
        zkir_llvm::TargetConfig::preset(preset_name)
            .ok_or_else(|| anyhow::anyhow!(
                "Unknown preset '{}'. Available presets: {}",
                preset_name,
                zkir_llvm::TargetConfig::preset_names().join(", ")
            ))?
    } else {
        // Validate limb_bits is even
        if args.limb_bits % 2 != 0 {
            anyhow::bail!("limb-bits must be even (got {})", args.limb_bits);
        }

        zkir_llvm::TargetConfig {
            limb_bits: args.limb_bits,
            data_limbs: args.data_limbs,
            addr_limbs: args.addr_limbs,
        }
    };

    // Validate configuration
    config.validate()
        .map_err(|e| anyhow::anyhow!("Invalid configuration: {}", e))?;

    // Show configuration warnings unless suppressed
    if !args.no_warnings {
        for warning in config.check_warnings() {
            eprintln!("warning: {}", warning);
        }
    }

    if args.verbose {
        eprintln!("Configuration:");
        if let Some(preset) = &args.preset {
            eprintln!("  preset:     {}", preset);
        }
        eprintln!("  opt_level:  {}", opt_level);
        eprintln!("  limb_bits:  {}", config.limb_bits);
        eprintln!("  data_limbs: {}", config.data_limbs);
        eprintln!("  addr_limbs: {}", config.addr_limbs);
        eprintln!("  data_bits:  {}", config.data_bits());
        eprintln!("  addr_bits:  {}", config.addr_bits());
        eprintln!("  headroom:   {} bits (for i32)", config.headroom(32));
    }

    // Verify input file exists
    if !args.input.exists() {
        anyhow::bail!("Input file not found: {}", args.input.display());
    }

    match args.emit {
        EmitType::Zkir => {
            let mode = if args.debug {
                zkir_llvm::OutputMode::Debug
            } else {
                zkir_llvm::OutputMode::Release
            };

            let (bytecode, stats) = zkir_llvm::compile_with_mode(&args.input, &config, opt_level, mode)
                .with_context(|| format!("Failed to compile {}", args.input.display()))?;

            let output = args.output.unwrap_or_else(|| {
                args.input.with_extension("zkir")
            });

            std::fs::write(&output, &bytecode)
                .with_context(|| format!("Failed to write {}", output.display()))?;

            if args.verbose {
                stats.display();
                eprintln!("\nWrote {} bytes to {}", bytecode.len(), output.display());
            } else {
                println!("{}", output.display());
            }
        }

        EmitType::Mir => {
            let mir = zkir_llvm::compile_to_mir(&args.input, &config)
                .with_context(|| format!("Failed to compile {}", args.input.display()))?;

            // Print MIR to stdout or file
            let mir_text = format!("{:#?}", mir);

            if let Some(output) = args.output {
                std::fs::write(&output, &mir_text)
                    .with_context(|| format!("Failed to write {}", output.display()))?;
            } else {
                println!("{}", mir_text);
            }
        }

        EmitType::Asm => {
            let mir = zkir_llvm::compile_to_mir(&args.input, &config)
                .with_context(|| format!("Failed to compile {}", args.input.display()))?;

            let asm_text = zkir_llvm::emit::format_asm(&mir);

            if let Some(output) = args.output {
                std::fs::write(&output, &asm_text)
                    .with_context(|| format!("Failed to write {}", output.display()))?;
            } else {
                println!("{}", asm_text);
            }
        }

        EmitType::Disasm => {
            // Read bytecode file and disassemble
            let bytecode = std::fs::read(&args.input)
                .with_context(|| format!("Failed to read {}", args.input.display()))?;

            let disasm_text = zkir_llvm::emit::disassemble(&bytecode)
                .with_context(|| "Failed to disassemble bytecode")?;

            if let Some(output) = args.output {
                std::fs::write(&output, &disasm_text)
                    .with_context(|| format!("Failed to write {}", output.display()))?;
            } else {
                println!("{}", disasm_text);
            }
        }
    }

    Ok(())
}
