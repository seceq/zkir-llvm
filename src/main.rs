//! CLI for zkir-llvm translator

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;

#[derive(Parser)]
#[command(name = "zkir-llvm")]
#[command(about = "LLVM IR to ZK IR translator")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Translate LLVM IR to ZK IR bytecode
    Translate {
        /// Input file (.ll or .bc)
        input: PathBuf,

        /// Output file (.zkbc)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Optimization level (0-3)
        #[arg(short = 'O', default_value = "1")]
        opt_level: u8,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Check LLVM IR for ZK IR compatibility
    Check {
        /// Input file (.ll)
        input: PathBuf,
    },

    /// Dump translated assembly (for debugging)
    Dump {
        /// Input file (.ll)
        input: PathBuf,
    },
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Translate { input, output, opt_level, verbose } => {
            let output = output.unwrap_or_else(|| input.with_extension("zkbc"));

            if verbose {
                println!("Reading LLVM IR from {:?}...", input);
            }

            let source = std::fs::read_to_string(&input)?;
            let program = zkir_llvm::translate_llvm_ir(&source, opt_level)?;

            if verbose {
                println!("Generated {} instructions", program.code.len());
            }

            // Save bytecode
            use zkir_llvm::emit::emit_program;
            let bytecode = emit_program(&program)?;
            std::fs::write(&output, bytecode)?;

            println!("Wrote {:?}", output);
        }

        Commands::Check { input } => {
            let source = std::fs::read_to_string(&input)?;
            zkir_llvm::check_compatibility(&source)?;
            println!("Compatible with ZK IR");
        }

        Commands::Dump { input } => {
            let source = std::fs::read_to_string(&input)?;
            let program = zkir_llvm::translate_llvm_ir(&source, 0)?;

            println!("ZK IR Assembly:\n");
            for (i, instr) in program.code.iter().enumerate() {
                println!("{:04x}: {:?}", i * 4 + 0x1000, instr);
            }
        }
    }

    Ok(())
}
