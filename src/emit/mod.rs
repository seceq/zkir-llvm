//! Code emission for ZKIR bytecode.
//!
//! Converts Machine IR (after register allocation) to ZKIR bytecode format.

mod encode;
mod binary;
mod disasm;

use crate::mir::Module;
use crate::target::config::TargetConfig;
use anyhow::Result;

pub use encode::{InstructionEncoder, ValidationError, validate_no_vregs};
pub use binary::{BinaryWriter, OutputMode};
pub use disasm::{Disassembler, DisasmInst, disassemble, disassemble_code};

/// Emit a module to ZKIR bytecode (release mode - zkir-spec compatible).
pub fn emit(module: &Module, config: &TargetConfig) -> Result<Vec<u8>> {
    emit_with_mode(module, config, OutputMode::Release)
}

/// Emit a module to ZKIR bytecode (debug mode - with symbol tables).
pub fn emit_debug(module: &Module, config: &TargetConfig) -> Result<Vec<u8>> {
    emit_with_mode(module, config, OutputMode::Debug)
}

/// Emit a module to ZKIR bytecode with the specified output mode.
pub fn emit_with_mode(module: &Module, config: &TargetConfig, mode: OutputMode) -> Result<Vec<u8>> {
    let mut writer = BinaryWriter::with_mode(config, mode);
    writer.emit_module(module)?;
    Ok(writer.finish())
}

/// Format a module as assembly-like text.
pub fn format_asm(module: &Module) -> String {
    format_asm_with_options(module, &AsmFormatOptions::default())
}

/// Options for assembly formatting.
#[derive(Debug, Clone)]
pub struct AsmFormatOptions {
    /// Show instruction addresses/indices
    pub show_addresses: bool,
    /// Include comments in output
    pub show_comments: bool,
    /// Include frame info in function headers
    pub show_frame_info: bool,
    /// Instruction mnemonic width for alignment
    pub mnemonic_width: usize,
}

impl Default for AsmFormatOptions {
    fn default() -> Self {
        Self {
            show_addresses: false,
            show_comments: true,
            show_frame_info: true,
            mnemonic_width: 8,
        }
    }
}

/// Format a module as assembly-like text with options.
pub fn format_asm_with_options(module: &Module, opts: &AsmFormatOptions) -> String {
    use std::fmt::Write;
    let mut output = String::new();

    // Header
    writeln!(output, "; ZKIR Assembly").unwrap();
    writeln!(output, "; Module: {}", module.name).unwrap();
    writeln!(output).unwrap();

    // Data section (globals)
    if !module.globals.is_empty() {
        writeln!(output, ".section .data").unwrap();
        for (name, global) in &module.globals {
            let visibility = if global.is_const { "const" } else { "global" };
            writeln!(output, "    .{:<8} {}  ; {} bytes, align {}",
                visibility, name, global.size, global.align).unwrap();

            // Show initializer if present
            if let Some(ref init_data) = global.init {
                if !init_data.is_empty() {
                    write!(output, "    .bytes    ").unwrap();
                    for (i, byte) in init_data.iter().take(16).enumerate() {
                        if i > 0 {
                            write!(output, ", ").unwrap();
                        }
                        write!(output, "0x{:02x}", byte).unwrap();
                    }
                    if init_data.len() > 16 {
                        write!(output, ", ... ({} more)", init_data.len() - 16).unwrap();
                    }
                    writeln!(output).unwrap();
                }
            }
        }
        writeln!(output).unwrap();
    }

    // Text section (functions)
    writeln!(output, ".section .text").unwrap();
    writeln!(output).unwrap();

    for func in module.functions.values() {
        // Function header
        writeln!(output, ".global {}", func.name).unwrap();
        writeln!(output, "{}:", func.name).unwrap();

        // Frame info as comments
        if opts.show_frame_info {
            let total_frame = func.frame.locals_size + func.frame.spill_size + func.frame.outgoing_args_size;
            if total_frame > 0 {
                writeln!(output, "    ; frame: {} bytes (locals={}, spill={}, outgoing={})",
                    total_frame,
                    func.frame.locals_size,
                    func.frame.spill_size,
                    func.frame.outgoing_args_size).unwrap();
            }
            if !func.frame.saved_regs.is_empty() {
                let regs: Vec<String> = func.frame.saved_regs.iter()
                    .map(|r| format!("{}", r))
                    .collect();
                writeln!(output, "    ; callee-saved: {}", regs.join(", ")).unwrap();
            }
        }

        // Parameters
        if !func.params.is_empty() {
            let params: Vec<String> = func.params.iter()
                .map(|v| format!("{}", v))
                .collect();
            writeln!(output, "    ; params: {}", params.join(", ")).unwrap();
        }

        // Return register
        if let Some(ret) = func.ret_vreg {
            writeln!(output, "    ; returns: {}", ret).unwrap();
        }

        // Basic blocks
        let mut inst_idx = 0u32;
        for block in func.iter_blocks() {
            writeln!(output, ".L{}:", block.label).unwrap();

            for inst in &block.insts {
                // Format instruction with proper alignment
                let inst_str = format_instruction(inst, opts);

                if opts.show_addresses {
                    write!(output, "{:4}: ", inst_idx).unwrap();
                }

                writeln!(output, "    {}", inst_str).unwrap();
                inst_idx += 1;
            }
        }
        writeln!(output).unwrap();
    }

    output
}

/// Format a single instruction with proper alignment.
fn format_instruction(inst: &crate::mir::MachineInst, opts: &AsmFormatOptions) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    // Opcode with fixed width
    write!(out, "{:<width$}", inst.opcode, width = opts.mnemonic_width).unwrap();

    // Operands
    let mut operands = String::new();
    if let Some(dst) = &inst.dst {
        write!(operands, "{}", dst).unwrap();
    }

    for (i, src) in inst.srcs.iter().enumerate() {
        if i == 0 && inst.dst.is_some() {
            write!(operands, ", {}", src).unwrap();
        } else if i == 0 {
            write!(operands, "{}", src).unwrap();
        } else {
            write!(operands, ", {}", src).unwrap();
        }
    }

    out.push_str(&operands);

    // Comment
    if opts.show_comments {
        if let Some(comment) = &inst.comment {
            // Pad to align comments
            let pad_len = 32usize.saturating_sub(out.len());
            for _ in 0..pad_len {
                out.push(' ');
            }
            write!(out, " ; {}", comment).unwrap();
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{MachineBlock, MachineFunction, MachineInst};

    #[test]
    fn test_format_asm() {
        let mut module = Module::new("test");

        let mut func = MachineFunction::new("main");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::mov(v1, v0));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        module.add_function(func);

        let asm = format_asm(&module);
        assert!(asm.contains("main:"));
        assert!(asm.contains(".Lentry:"));  // Updated label format
        assert!(asm.contains("li"));
        assert!(asm.contains("42"));
        assert!(asm.contains(".section .text"));
    }

    #[test]
    fn test_format_asm_with_globals() {
        let mut module = Module::new("test");

        // Add a global
        module.globals.insert(
            "my_data".to_string(),
            crate::mir::GlobalVar {
                name: "my_data".to_string(),
                size: 16,
                align: 4,
                is_const: true,
                init: Some(vec![0x01, 0x02, 0x03, 0x04]),
            },
        );

        let asm = format_asm(&module);
        assert!(asm.contains(".section .data"));
        assert!(asm.contains("my_data"));
        assert!(asm.contains("const"));
        assert!(asm.contains("0x01"));
    }

    #[test]
    fn test_format_asm_options() {
        let mut module = Module::new("test");

        let mut func = MachineFunction::new("main");
        let v0 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42).comment("load value"));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        module.add_function(func);

        // Test with addresses
        let opts = AsmFormatOptions {
            show_addresses: true,
            ..Default::default()
        };
        let asm = format_asm_with_options(&module, &opts);
        assert!(asm.contains("   0:")); // First instruction at index 0

        // Test without comments
        let opts = AsmFormatOptions {
            show_comments: false,
            ..Default::default()
        };
        let asm = format_asm_with_options(&module, &opts);
        assert!(!asm.contains("load value"));
    }
}
