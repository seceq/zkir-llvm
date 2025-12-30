//! ZKIR binary format writer.
//!
//! Produces the final ZKIR bytecode file format using zkir-spec v3.4.
//!
//! ## Output Modes
//!
//! - **Release mode** (default): Matches zkir-spec format exactly.
//!   Layout: `[header][code][data]`
//!   Compatible with `Program::from_bytes()` for execution and proving.
//!
//! - **Debug mode**: Includes symbol tables for debugging/disassembly.
//!   Layout: `[header][globals metadata][function table][code]`
//!   Preserves function names, sizes, and global metadata.

use super::encode::InstructionEncoder;
use crate::mir::{Module, MachineFunction, Opcode, Operand};
use crate::target::config::TargetConfig;
use crate::target::registers::RegisterExt; // For .number() method
use anyhow::Result;
use std::collections::HashMap;

// Re-export from zkir-spec for backwards compatibility
pub use zkir_spec::program::{MAGIC as ZKIR_MAGIC_U32, VERSION as ZKIR_VERSION_U32, ProgramHeader};

/// Output format mode for the binary writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMode {
    /// Release mode: matches zkir-spec format exactly.
    /// Layout: [header][code][data]
    /// Compatible with Program::from_bytes() for execution.
    #[default]
    Release,

    /// Debug mode: includes symbol tables for debugging.
    /// Layout: [header][globals metadata][function table][code]
    /// Useful for disassembly and debugging.
    Debug,
}

/// Binary writer for ZKIR bytecode.
pub struct BinaryWriter<'a> {
    config: &'a TargetConfig,
    buffer: Vec<u8>,
    /// Function offsets for linking
    function_offsets: HashMap<String, u32>,
    /// Global variable offsets in data section
    global_offsets: HashMap<String, u32>,
    /// Label offsets within current function
    label_offsets: HashMap<String, u32>,
    /// Relocations to fix up
    relocations: Vec<Relocation>,
    /// Output mode (release or debug)
    mode: OutputMode,
}

/// A relocation entry.
#[derive(Debug)]
struct Relocation {
    /// Offset in the buffer where the fixup is needed
    offset: u32,
    /// Target symbol (function or label name)
    target: String,
    /// Type of relocation
    kind: RelocKind,
}

/// Relocation type.
#[derive(Debug, Clone, Copy)]
enum RelocKind {
    /// Absolute function address (J-type, 21-bit offset)
    FunctionAbs,
    /// Relative branch offset (B-type, 17-bit immediate)
    BranchRel,
    /// Relative jump offset (J-type, 21-bit offset) - for JAL to local labels
    JumpRel,
    /// Global variable address
    GlobalAbs,
}

impl<'a> BinaryWriter<'a> {
    /// Create a new binary writer in release mode (default).
    pub fn new(config: &'a TargetConfig) -> Self {
        Self::with_mode(config, OutputMode::Release)
    }

    /// Create a new binary writer with the specified output mode.
    pub fn with_mode(config: &'a TargetConfig, mode: OutputMode) -> Self {
        Self {
            config,
            buffer: Vec::new(),
            function_offsets: HashMap::new(),
            global_offsets: HashMap::new(),
            label_offsets: HashMap::new(),
            relocations: Vec::new(),
            mode,
        }
    }

    /// Emit a module to the buffer.
    pub fn emit_module(&mut self, module: &Module) -> Result<()> {
        match self.mode {
            OutputMode::Release => self.emit_module_release(module),
            OutputMode::Debug => self.emit_module_debug(module),
        }
    }

    /// Emit module in release mode (zkir-spec compatible).
    ///
    /// Layout: [32-byte header][code section][data section]
    /// - Code starts immediately after header (offset 32)
    /// - entry_point is CODE_BASE (0x1000) + offset to main within code section
    fn emit_module_release(&mut self, module: &Module) -> Result<()> {
        self.emit_header(module)?;

        // Emit code section first (right after header, at offset 32)
        let code_start = self.buffer.len();
        self.emit_functions_release(module)?;
        let code_size = (self.buffer.len() - code_start) as u32;

        // Emit data section after code
        let data_start = self.buffer.len();
        self.emit_globals_release(module)?;
        let data_size = (self.buffer.len() - data_start) as u32;

        self.apply_relocations()?;

        // Calculate entry point as CODE_BASE + offset to _start
        // _start is always emitted first, so it's at the beginning of the code section
        let start_offset = self.function_offsets.get("_start")
            .copied()
            .unwrap_or(ProgramHeader::SIZE as u32);
        let entry_point = 0x1000 + (start_offset - ProgramHeader::SIZE as u32);

        // Update header fields with actual sizes
        self.update_header_sizes(entry_point, code_size, data_size)?;

        Ok(())
    }

    /// Emit module in debug mode (with symbol tables).
    ///
    /// Layout: [32-byte header][globals metadata][function table][code]
    /// - Includes function names and sizes for disassembly
    /// - Includes global variable metadata
    fn emit_module_debug(&mut self, module: &Module) -> Result<()> {
        self.emit_header(module)?;

        let data_start = self.buffer.len();
        self.emit_globals_debug(module)?;
        let data_size = (self.buffer.len() - data_start) as u32;

        let code_start = self.buffer.len();
        self.emit_functions_debug(module)?;
        let code_size = (self.buffer.len() - code_start) as u32;

        self.apply_relocations()?;

        // In debug mode, entry_point is the file offset where code starts
        self.update_header_sizes(code_start as u32, code_size, data_size)?;

        Ok(())
    }

    /// Update header fields after emitting code and data.
    fn update_header_sizes(&mut self, entry_point: u32, code_size: u32, data_size: u32) -> Result<()> {
        // ProgramHeader field offsets (from zkir-spec):
        // entry_point at offset 0x0C (bytes 12-15)
        // code_size at offset 0x10 (bytes 16-19)
        // data_size at offset 0x14 (bytes 20-23)
        self.buffer[12..16].copy_from_slice(&entry_point.to_le_bytes());
        self.buffer[16..20].copy_from_slice(&code_size.to_le_bytes());
        self.buffer[20..24].copy_from_slice(&data_size.to_le_bytes());
        Ok(())
    }

    /// Finish writing and return the buffer.
    pub fn finish(self) -> Vec<u8> {
        self.buffer
    }

    /// Emit the file header using zkir-spec v3.4 ProgramHeader (32 bytes).
    fn emit_header(&mut self, _module: &Module) -> Result<()> {
        // Create a zkir-spec ProgramHeader with our configuration
        let header = ProgramHeader {
            magic: ZKIR_MAGIC_U32,
            version: ZKIR_VERSION_U32,
            limb_bits: self.config.limb_bits,
            data_limbs: self.config.data_limbs,
            addr_limbs: self.config.addr_limbs,
            flags: 0,
            entry_point: 0, // Will be updated after emitting code
            code_size: 0,   // Will be updated after emitting code
            data_size: 0,   // Will be updated after emitting globals
            bss_size: 0,
            stack_size: zkir_spec::memory::DEFAULT_STACK_SIZE as u32,
        };

        // Emit the 32-byte header
        self.buffer.extend(header.to_bytes());

        Ok(())
    }

    /// Emit global variable section (release mode).
    ///
    /// Emits only raw initialized data to match zkir-spec format.
    /// Uninitialized (BSS) data is not emitted, only tracked in header.bss_size.
    fn emit_globals_release(&mut self, module: &Module) -> Result<()> {
        // Calculate data section layout - compute offsets for each global
        //
        // Memory layout at runtime:
        //   CODE_BASE (0x1000) -> code section
        //   CODE_BASE + code_size -> data section (loaded right after code)
        //
        // File layout:
        //   Offset 0-31: Header (32 bytes)
        //   Offset 32: Code section starts
        //   Offset 32 + code_size: Data section starts
        //
        // Runtime address = CODE_BASE + (file_offset - HEADER_SIZE)
        // For data at file offset F: runtime_addr = 0x1000 + F - 32
        //
        // Since data_section_start is the file offset where data begins,
        // and current_offset is the offset within data section:
        // runtime_addr = CODE_BASE + (data_section_start + current_offset - HEADER_SIZE)
        //              = 0x1000 + data_section_start + current_offset - 32

        let data_section_start = self.buffer.len() as u32;
        let header_size = ProgramHeader::SIZE as u32;  // 32 bytes
        let code_base: u32 = 0x1000;
        let mut current_offset = 0u32;

        // Collect globals sorted by name for deterministic output
        let mut globals: Vec<_> = module.globals.iter().collect();
        globals.sort_by_key(|(name, _)| *name);

        // First pass: calculate offsets with proper alignment
        for (name, global) in &globals {
            // Align current offset
            let align = global.align.max(1);
            current_offset = (current_offset + align - 1) & !(align - 1);

            // Compute runtime address: CODE_BASE + (file_offset - HEADER_SIZE)
            // file_offset = data_section_start + current_offset
            let runtime_addr = code_base + data_section_start + current_offset - header_size;
            self.global_offsets.insert((*name).clone(), runtime_addr);

            // Advance by size
            current_offset += global.size;
        }

        // Second pass: emit raw initialized data only
        current_offset = 0;
        for (_name, global) in &globals {
            // Align with padding bytes
            let align = global.align.max(1);
            let aligned_offset = (current_offset + align - 1) & !(align - 1);
            let padding = aligned_offset - current_offset;
            self.buffer.extend(std::iter::repeat(0u8).take(padding as usize));
            current_offset = aligned_offset;

            // Emit initial value or zeros
            if let Some(ref init) = global.init {
                self.buffer.extend(init);
                // Pad to full size if init is smaller
                if init.len() < global.size as usize {
                    self.buffer.extend(std::iter::repeat(0u8).take(global.size as usize - init.len()));
                }
            } else {
                // Uninitialized global - emit zeros for data section
                // (BSS would skip this, but we emit for simplicity)
                self.buffer.extend(std::iter::repeat(0u8).take(global.size as usize));
            }

            current_offset += global.size;
        }

        Ok(())
    }

    /// Emit function section (release mode).
    ///
    /// Emits code directly without a function table to match zkir-spec format.
    /// Layout: [_start stub][main][other functions...]
    ///
    /// The _start stub initializes SP and calls main, then terminates with EBREAK.
    /// This makes .zkir files self-contained without requiring load-time patching.
    fn emit_functions_release(&mut self, module: &Module) -> Result<()> {
        // Emit _start stub first (entry point)
        self.emit_start_stub()?;

        // Collect functions, putting main first (after _start)
        let mut funcs: Vec<&MachineFunction> = module.functions.values().collect();
        funcs.sort_by(|a, b| {
            match (a.name.as_str(), b.name.as_str()) {
                ("main", _) => std::cmp::Ordering::Less,
                (_, "main") => std::cmp::Ordering::Greater,
                (a_name, b_name) => a_name.cmp(b_name),
            }
        });

        // Emit function code directly
        for func in funcs {
            let code_start = self.buffer.len() as u32;
            self.function_offsets.insert(func.name.clone(), code_start);

            self.emit_function(func)?;
        }

        Ok(())
    }

    /// Emit the _start stub that initializes the runtime and calls main.
    ///
    /// Generated code:
    /// ```asm
    /// _start:
    ///     addi sp, zero, 0x8000    # SP = 32768
    ///     addi sp, sp, 0x8000      # SP = 65536 (64KB stack)
    ///     jal  ra, main            # Call main (sets return address)
    ///     ebreak                   # Terminate after main returns
    /// ```
    ///
    /// This makes .zkir files self-contained:
    /// - SP is properly initialized before any code runs
    /// - main() can use standard calling convention (return via JALR ra)
    /// - Program terminates cleanly when main() returns
    fn emit_start_stub(&mut self) -> Result<()> {
        use zkir_spec::encoding::encode_itype;
        use zkir_spec::Opcode as SpecOpcode;
        use zkir_spec::Register;

        let start_offset = self.buffer.len() as u32;
        self.function_offsets.insert("_start".to_string(), start_offset);

        let sp = Register::SP.index() as u32;  // R2
        let ra = Register::RA.index() as u32;  // R1
        let zero = Register::ZERO.index() as u32;  // R0

        // ADDI sp, zero, 0x8000  (SP = 32768)
        let addi_sp_1 = encode_itype(SpecOpcode::Addi, sp, zero, 0x8000);
        self.buffer.extend(addi_sp_1.to_le_bytes());

        // ADDI sp, sp, 0x8000  (SP = 65536)
        let addi_sp_2 = encode_itype(SpecOpcode::Addi, sp, sp, 0x8000);
        self.buffer.extend(addi_sp_2.to_le_bytes());

        // JAL ra, main  (call main, to be relocated)
        // Encode with placeholder offset, will be fixed by relocation
        let jal_offset = self.buffer.len() as u32;
        let jal_main = zkir_spec::encoding::encode_jtype(SpecOpcode::Jal, ra, 0);
        self.buffer.extend(jal_main.to_le_bytes());

        // Record relocation for JAL to main
        self.relocations.push(Relocation {
            offset: jal_offset,
            target: "main".to_string(),
            kind: RelocKind::FunctionAbs,
        });

        // EBREAK  (terminate)
        let ebreak = SpecOpcode::Ebreak as u32;
        self.buffer.extend(ebreak.to_le_bytes());

        Ok(())
    }

    /// Emit global variable section (debug mode).
    ///
    /// Includes metadata (name, size, alignment, flags) for debugging.
    fn emit_globals_debug(&mut self, module: &Module) -> Result<()> {
        // Calculate data section layout - compute offsets for each global
        let data_section_start = self.buffer.len() as u32;
        let mut current_offset = 0u32;

        // First pass: calculate offsets with proper alignment
        for (name, global) in &module.globals {
            // Align current offset
            let align = global.align.max(1);
            current_offset = (current_offset + align - 1) & !(align - 1);

            // Record offset (relative to data section start)
            self.global_offsets.insert(name.clone(), data_section_start + current_offset);

            // Advance by size
            current_offset += global.size;
        }

        // Second pass: emit global metadata and data
        for (name, global) in &module.globals {
            // Name length and name
            let name_bytes = name.as_bytes();
            self.buffer.push(name_bytes.len() as u8);
            self.buffer.extend(name_bytes);

            // Size
            self.buffer.extend(global.size.to_le_bytes());

            // Alignment
            self.buffer.extend(global.align.to_le_bytes());

            // Flags (is_const)
            self.buffer.push(if global.is_const { 1 } else { 0 });

            // Initial value
            if let Some(ref init) = global.init {
                self.buffer.extend((init.len() as u32).to_le_bytes());
                self.buffer.extend(init);
            } else {
                self.buffer.extend(0u32.to_le_bytes());
            }
        }

        Ok(())
    }

    /// Emit function section (debug mode).
    ///
    /// Includes function table with names, offsets, and sizes for debugging.
    fn emit_functions_debug(&mut self, module: &Module) -> Result<()> {
        // First pass: collect function offsets
        let _functions_start = self.buffer.len();

        // Emit function table (name, offset, size placeholders)
        let mut function_table: Vec<(String, usize)> = Vec::new();

        for func in module.functions.values() {
            let name_bytes = func.name.as_bytes();
            self.buffer.push(name_bytes.len() as u8);
            self.buffer.extend(name_bytes);

            // Placeholder for code offset and size
            let offset_pos = self.buffer.len();
            self.buffer.extend(0u32.to_le_bytes()); // offset
            self.buffer.extend(0u32.to_le_bytes()); // size

            function_table.push((func.name.clone(), offset_pos));
        }

        // Second pass: emit function code
        for (func_name, offset_pos) in &function_table {
            let func = module.functions.get(func_name).unwrap();
            let code_start = self.buffer.len() as u32;
            self.function_offsets.insert(func_name.clone(), code_start);

            self.emit_function(func)?;

            let code_size = (self.buffer.len() as u32) - code_start;

            // Fix up the offset and size in the table
            self.buffer[*offset_pos..*offset_pos + 4].copy_from_slice(&code_start.to_le_bytes());
            self.buffer[*offset_pos + 4..*offset_pos + 8].copy_from_slice(&code_size.to_le_bytes());
        }

        Ok(())
    }

    /// Emit a single function.
    fn emit_function(&mut self, func: &MachineFunction) -> Result<()> {
        // Validate that all virtual registers have been allocated
        if let Err(errors) = super::encode::validate_no_vregs(func) {
            for err in &errors {
                log::error!("{}", err);
            }
            return Err(anyhow::anyhow!(
                "Function {} has {} unallocated virtual registers",
                func.name,
                errors.len()
            ));
        }

        // NOTE: We do NOT clear label_offsets here. Relocations are applied AFTER
        // all functions are emitted, so we need labels from all functions to remain
        // in the map. Labels are prefixed with function name to avoid collisions.

        // Emit prologue first
        self.emit_prologue(func)?;

        // Calculate label positions AFTER prologue, using absolute file offsets
        // This ensures branch relocations compute correct relative offsets
        let prologue_end = self.buffer.len() as u32;
        let mut code_offset = prologue_end;
        for block in func.iter_blocks() {
            // Store absolute file offset for each label, prefixed with function name
            let qualified_label = format!("{}::{}", func.name, block.label);
            self.label_offsets.insert(qualified_label, code_offset);
            for inst in &block.insts {
                let encoder = InstructionEncoder::new(self.config);
                let bytes = encoder.encode(inst)?;
                code_offset += bytes.len() as u32;
            }
        }

        // Store current function name for emit_instruction to use
        let func_name = func.name.clone();

        // Emit blocks
        for block in func.iter_blocks() {
            for inst in &block.insts {
                // Emit epilogue before RET instructions
                if inst.opcode == Opcode::RET {
                    self.emit_epilogue(func)?;
                }
                self.emit_instruction_with_context(inst, &func_name)?;
            }
        }

        Ok(())
    }

    /// Emit function epilogue to restore callee-saved registers.
    fn emit_epilogue(&mut self, func: &MachineFunction) -> Result<()> {
        use zkir_spec::encoding::encode_itype;
        use zkir_spec::Opcode as SpecOpcode;
        use zkir_spec::Register;

        let frame_size = func.frame.total_size() as i32;

        if frame_size > 0 {
            let sp = Register::SP.index() as u32;
            let ra = Register::RA.index() as u32;
            let fp = Register::FP.index() as u32;

            // Restore callee-saved registers (in reverse order from prologue)
            // In prologue, we saved starting at fp_offset - 4, decrementing by 4 for each
            // So first saved reg is at fp_offset - 4, second at fp_offset - 8, etc.
            // Restore in reverse order: last saved first
            let num_saved = func.frame.saved_regs.len() as i32;
            let mut offset = func.frame.fp_offset() - 4 - (num_saved - 1) * 4;
            for reg in func.frame.saved_regs.iter().rev() {
                let reg_num = reg.number() as u32;
                let off = (offset as u32) & zkir_spec::encoding::IMM_MASK;
                let lw_reg = encode_itype(SpecOpcode::Lw, reg_num, sp, off);
                self.buffer.extend(lw_reg.to_le_bytes());
                offset += 4;
            }

            // Restore frame pointer: LW fp, fp_offset(sp)
            let fp_offset = (func.frame.fp_offset() as u32) & zkir_spec::encoding::IMM_MASK;
            let lw_fp = encode_itype(SpecOpcode::Lw, fp, sp, fp_offset);
            self.buffer.extend(lw_fp.to_le_bytes());

            // Restore return address: LW ra, ra_offset(sp)
            let ra_offset = (func.frame.ra_offset() as u32) & zkir_spec::encoding::IMM_MASK;
            let lw_ra = encode_itype(SpecOpcode::Lw, ra, sp, ra_offset);
            self.buffer.extend(lw_ra.to_le_bytes());

            // Restore stack pointer: ADDI sp, sp, frame_size
            let pos_frame = (frame_size as u32) & zkir_spec::encoding::IMM_MASK;
            let addi_sp = encode_itype(SpecOpcode::Addi, sp, sp, pos_frame);
            self.buffer.extend(addi_sp.to_le_bytes());
        }

        Ok(())
    }

    /// Emit function prologue using zkir-spec v3.4 32-bit instruction format.
    fn emit_prologue(&mut self, func: &MachineFunction) -> Result<()> {
        use zkir_spec::encoding::{encode_itype, encode_stype};
        use zkir_spec::Opcode as SpecOpcode;
        use zkir_spec::Register;

        let frame_size = func.frame.total_size() as i32;

        if frame_size > 0 {
            let sp = Register::SP.index() as u32;
            let ra = Register::RA.index() as u32;
            let fp = Register::FP.index() as u32;

            // ADDI sp, sp, -frame_size
            let neg_frame = ((-frame_size) as u32) & zkir_spec::encoding::IMM_MASK;
            let addi_sp = encode_itype(SpecOpcode::Addi, sp, sp, neg_frame);
            self.buffer.extend(addi_sp.to_le_bytes());

            // Save return address: SW ra, ra_offset(sp)
            let ra_offset = (func.frame.ra_offset() as u32) & zkir_spec::encoding::IMM_MASK;
            let sw_ra = encode_stype(SpecOpcode::Sw, sp, ra, ra_offset);
            self.buffer.extend(sw_ra.to_le_bytes());

            // Save frame pointer: SW fp, fp_offset(sp)
            let fp_offset = (func.frame.fp_offset() as u32) & zkir_spec::encoding::IMM_MASK;
            let sw_fp = encode_stype(SpecOpcode::Sw, sp, fp, fp_offset);
            self.buffer.extend(sw_fp.to_le_bytes());

            // Set up frame pointer: ADDI fp, sp, frame_size
            let pos_frame = (frame_size as u32) & zkir_spec::encoding::IMM_MASK;
            let addi_fp = encode_itype(SpecOpcode::Addi, fp, sp, pos_frame);
            self.buffer.extend(addi_fp.to_le_bytes());

            // Save callee-saved registers
            let mut offset = func.frame.fp_offset() - 4;
            for reg in &func.frame.saved_regs {
                let reg_num = reg.number() as u32;
                let off = (offset as u32) & zkir_spec::encoding::IMM_MASK;
                let sw_reg = encode_stype(SpecOpcode::Sw, sp, reg_num, off);
                self.buffer.extend(sw_reg.to_le_bytes());
                offset -= 4;
            }
        }

        Ok(())
    }

    /// Emit a single instruction with function context for qualified label lookup.
    fn emit_instruction_with_context(&mut self, inst: &crate::mir::MachineInst, func_name: &str) -> Result<()> {
        // Record the instruction start position for relocations
        let inst_offset = self.buffer.len() as u32;

        // Handle branch/call instructions that need relocation
        if inst.opcode.is_branch() || inst.opcode == Opcode::JAL || inst.opcode == Opcode::CALL {
            // Check for label operand
            for src in &inst.srcs {
                if let Operand::Label(label) = src {
                    // Determine relocation type:
                    // - CALL always targets a function (use FunctionAbs)
                    // - JAL can target either a function or a local block label
                    // - Branches (BEQ, BNE, etc.) target local block labels (use BranchRel)

                    // Check if this is a local label using qualified name
                    let qualified_label = format!("{}::{}", func_name, label);
                    let is_local_label = self.label_offsets.contains_key(&qualified_label);

                    if inst.opcode == Opcode::CALL {
                        // CALL pseudo-op always targets a function
                        self.relocations.push(Relocation {
                            offset: inst_offset,
                            target: label.clone(),
                            kind: RelocKind::FunctionAbs,
                        });
                    } else if inst.opcode == Opcode::JAL && !is_local_label {
                        // JAL to a function (not a local label)
                        self.relocations.push(Relocation {
                            offset: inst_offset,
                            target: label.clone(),
                            kind: RelocKind::FunctionAbs,
                        });
                    } else if inst.opcode == Opcode::JAL {
                        // JAL to a local label (J-type, 21-bit offset)
                        // Use qualified label name for relocation target
                        self.relocations.push(Relocation {
                            offset: inst_offset,
                            target: qualified_label,
                            kind: RelocKind::JumpRel,
                        });
                    } else {
                        // Branches (BEQ, BNE, etc.) use B-type with 17-bit offset
                        // Use qualified label name for relocation target
                        self.relocations.push(Relocation {
                            offset: inst_offset,
                            target: qualified_label,
                            kind: RelocKind::BranchRel,
                        });
                    }
                }
            }
        }

        // Handle GlobalAddr operands that need relocation
        for src in &inst.srcs {
            if let Operand::GlobalAddr(name) = src {
                // I-type: immediate in bits 15-31, stored at instruction start
                self.relocations.push(Relocation {
                    offset: inst_offset,
                    target: name.clone(),
                    kind: RelocKind::GlobalAbs,
                });
            }
        }

        // Encode and emit
        let encoder = InstructionEncoder::new(self.config);
        let bytes = encoder.encode(inst)?;
        self.buffer.extend(bytes);

        Ok(())
    }

    /// Apply all relocations for zkir-spec v3.4 32-bit instructions.
    fn apply_relocations(&mut self) -> Result<()> {
        use zkir_spec::encoding::{IMM_SHIFT, IMM_MASK, OFFSET_SHIFT, OFFSET_MASK};

        for reloc in &self.relocations {
            let target_value: i32 = match reloc.kind {
                RelocKind::FunctionAbs => {
                    // JAL to a function uses PC-relative addressing
                    // Offset = target_address - instruction_address
                    let func_offset = self.function_offsets.get(&reloc.target).copied();
                    if func_offset.is_none() {
                        log::warn!("FunctionAbs relocation: target '{}' not found in function_offsets", reloc.target);
                        log::warn!("  Available functions: {:?}", self.function_offsets.keys().collect::<Vec<_>>());
                    }
                    let func_offset = func_offset.unwrap_or(0) as i32;
                    let inst_pos = reloc.offset as i32;
                    func_offset - inst_pos
                }
                RelocKind::BranchRel | RelocKind::JumpRel => {
                    if let Some(&label_offset) = self.label_offsets.get(&reloc.target) {
                        // Relative offset from instruction position
                        let inst_pos = reloc.offset as i32;
                        (label_offset as i32) - inst_pos
                    } else {
                        // Try function offset (also PC-relative)
                        let func_offset = self.function_offsets.get(&reloc.target).copied();
                        if func_offset.is_none() {
                            log::warn!("BranchRel/JumpRel relocation: target '{}' not found", reloc.target);
                            log::warn!("  Available labels: {:?}", self.label_offsets.keys().collect::<Vec<_>>());
                            log::warn!("  Available functions: {:?}", self.function_offsets.keys().collect::<Vec<_>>());
                        }
                        let func_offset = func_offset.unwrap_or(0) as i32;
                        let inst_pos = reloc.offset as i32;
                        func_offset - inst_pos
                    }
                }
                RelocKind::GlobalAbs => {
                    self.global_offsets.get(&reloc.target).copied().unwrap_or_else(|| {
                        log::warn!("Unknown global variable in relocation: {}", reloc.target);
                        0
                    }) as i32
                }
            };

            // Read the existing instruction
            let offset = reloc.offset as usize;
            if offset + 4 > self.buffer.len() {
                continue;
            }

            let mut inst = u32::from_le_bytes([
                self.buffer[offset],
                self.buffer[offset + 1],
                self.buffer[offset + 2],
                self.buffer[offset + 3],
            ]);

            // Patch the appropriate field based on relocation type
            match reloc.kind {
                RelocKind::FunctionAbs | RelocKind::JumpRel => {
                    // J-type: patch offset field (bits 11-31, 21 bits)
                    let masked = (target_value as u32) & OFFSET_MASK;
                    inst = (inst & !((OFFSET_MASK) << OFFSET_SHIFT)) | (masked << OFFSET_SHIFT);
                }
                RelocKind::BranchRel => {
                    // B-type: patch immediate field (bits 15-31, 17 bits)
                    let masked = (target_value as u32) & IMM_MASK;
                    inst = (inst & !((IMM_MASK) << IMM_SHIFT)) | (masked << IMM_SHIFT);
                }
                RelocKind::GlobalAbs => {
                    // I-type: patch immediate field (bits 15-31, 17 bits)
                    let masked = (target_value as u32) & IMM_MASK;
                    inst = (inst & !((IMM_MASK) << IMM_SHIFT)) | (masked << IMM_SHIFT);
                }
            }

            // Write the patched instruction back
            self.buffer[offset..offset + 4].copy_from_slice(&inst.to_le_bytes());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{MachineBlock, MachineFunction, MachineInst};

    #[test]
    fn test_header() {
        let config = TargetConfig::default();
        let mut writer = BinaryWriter::new(&config);

        let module = Module::new("test");
        writer.emit_header(&module).unwrap();

        // zkir-spec v3.4 header is 32 bytes
        assert_eq!(writer.buffer.len(), ProgramHeader::SIZE);

        // Check magic (bytes 0-3, little-endian u32)
        let magic = u32::from_le_bytes([
            writer.buffer[0], writer.buffer[1],
            writer.buffer[2], writer.buffer[3]
        ]);
        assert_eq!(magic, ZKIR_MAGIC_U32);

        // Check version (bytes 4-7, little-endian u32)
        let version = u32::from_le_bytes([
            writer.buffer[4], writer.buffer[5],
            writer.buffer[6], writer.buffer[7]
        ]);
        assert_eq!(version, ZKIR_VERSION_U32);

        // Check configuration (bytes 8-10)
        assert_eq!(writer.buffer[8], config.limb_bits);
        assert_eq!(writer.buffer[9], config.data_limbs);
        assert_eq!(writer.buffer[10], config.addr_limbs);
    }

    #[test]
    fn test_emit_simple_function() {
        let config = TargetConfig::default();
        let mut writer = BinaryWriter::new(&config);

        let mut module = Module::new("test");
        let mut func = MachineFunction::new("main");

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::ret());
        func.add_block(entry);

        module.add_function(func);

        writer.emit_module(&module).unwrap();

        let bytes = writer.finish();
        assert!(!bytes.is_empty());

        // Check magic at start
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, ZKIR_MAGIC_U32);
    }
}
