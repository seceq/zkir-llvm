//! ZKIR binary format writer.
//!
//! Produces the final ZKIR bytecode file format using zkir-spec v3.4.

use super::encode::InstructionEncoder;
use crate::mir::{Module, MachineFunction, Opcode, Operand};
use crate::target::config::TargetConfig;
use crate::target::registers::RegisterExt; // For .number() method
use anyhow::Result;
use std::collections::HashMap;

// Re-export from zkir-spec for backwards compatibility
pub use zkir_spec::program::{MAGIC as ZKIR_MAGIC_U32, VERSION as ZKIR_VERSION_U32, ProgramHeader};

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
    /// Absolute function address
    FunctionAbs,
    /// Relative branch offset
    BranchRel,
    /// Global variable address
    GlobalAbs,
}

impl<'a> BinaryWriter<'a> {
    /// Create a new binary writer.
    pub fn new(config: &'a TargetConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
            function_offsets: HashMap::new(),
            global_offsets: HashMap::new(),
            label_offsets: HashMap::new(),
            relocations: Vec::new(),
        }
    }

    /// Emit a module to the buffer.
    pub fn emit_module(&mut self, module: &Module) -> Result<()> {
        self.emit_header(module)?;

        let data_start = self.buffer.len();
        self.emit_globals(module)?;
        let data_size = (self.buffer.len() - data_start) as u32;

        let code_start = self.buffer.len();
        self.emit_functions(module)?;
        let code_size = (self.buffer.len() - code_start) as u32;

        self.apply_relocations()?;

        // Update header fields with actual sizes
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

    /// Emit global variable section.
    fn emit_globals(&mut self, module: &Module) -> Result<()> {
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

    /// Emit function section.
    fn emit_functions(&mut self, module: &Module) -> Result<()> {
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

        // Clear label offsets for this function
        self.label_offsets.clear();

        // Emit prologue
        self.emit_prologue(func)?;

        // Collect label positions first
        let mut code_offset = 0u32;
        for block in func.iter_blocks() {
            self.label_offsets.insert(block.label.clone(), code_offset);
            for inst in &block.insts {
                let encoder = InstructionEncoder::new(self.config);
                let bytes = encoder.encode(inst)?;
                code_offset += bytes.len() as u32;
            }
        }

        // Emit blocks
        for block in func.iter_blocks() {
            for inst in &block.insts {
                self.emit_instruction(inst)?;
            }
        }

        // Emit epilogue (handled by RET instructions)

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

    /// Emit a single instruction.
    fn emit_instruction(&mut self, inst: &crate::mir::MachineInst) -> Result<()> {
        // Record the instruction start position for relocations
        let inst_offset = self.buffer.len() as u32;

        // Handle branch/call instructions that need relocation
        if inst.opcode.is_branch() || inst.opcode == Opcode::JAL || inst.opcode == Opcode::CALL {
            // Check for label operand
            for src in &inst.srcs {
                if let Operand::Label(label) = src {
                    if inst.opcode == Opcode::CALL || inst.opcode == Opcode::JAL {
                        // J-type: offset in bits 11-31, stored at instruction start
                        self.relocations.push(Relocation {
                            offset: inst_offset,
                            target: label.clone(),
                            kind: RelocKind::FunctionAbs,
                        });
                    } else {
                        // B-type: offset in bits 15-31, stored at instruction start
                        self.relocations.push(Relocation {
                            offset: inst_offset,
                            target: label.clone(),
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
                    self.function_offsets.get(&reloc.target).copied().unwrap_or(0) as i32
                }
                RelocKind::BranchRel => {
                    if let Some(&label_offset) = self.label_offsets.get(&reloc.target) {
                        // Relative offset from instruction position
                        let inst_pos = reloc.offset as i32;
                        (label_offset as i32) - inst_pos
                    } else {
                        // Try function offset
                        self.function_offsets.get(&reloc.target).copied().unwrap_or(0) as i32
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
                RelocKind::FunctionAbs => {
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
