//! ZKIR bytecode disassembler.
//!
//! Decodes ZKIR bytecode back to human-readable assembly for debugging.
//! Uses zkir-spec v3.4 32-bit instruction format.

use super::binary::{ZKIR_MAGIC_U32, ZKIR_VERSION_U32, ProgramHeader};
use crate::mir::Opcode;
use anyhow::{anyhow, Result};
use std::fmt::Write;

// Import zkir-spec extraction functions
use zkir_spec::encoding::{
    extract_opcode, extract_rd, extract_rs1, extract_rs2,
    extract_imm_signed, extract_offset_signed,
    extract_stype_rs1, extract_stype_rs2, extract_stype_imm,
};

/// Disassembled instruction.
#[derive(Debug, Clone)]
pub struct DisasmInst {
    /// Offset in bytecode
    pub offset: u32,
    /// Raw bytes
    pub bytes: Vec<u8>,
    /// Opcode
    pub opcode: Opcode,
    /// Formatted operands string
    pub operands: String,
    /// Optional comment
    pub comment: Option<String>,
}

impl std::fmt::Display for DisasmInst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:08x}:  ", self.offset)?;

        // Print bytes (up to 8)
        for (i, byte) in self.bytes.iter().take(8).enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{:02x}", byte)?;
        }

        // Pad to fixed width
        for _ in self.bytes.len()..4 {
            write!(f, "   ")?;
        }

        write!(f, "    {:8} {}", self.opcode, self.operands)?;

        if let Some(ref comment) = self.comment {
            write!(f, "  ; {}", comment)?;
        }

        Ok(())
    }
}

/// ZKIR bytecode disassembler.
pub struct Disassembler<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Disassembler<'a> {
    /// Create a new disassembler.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Disassemble the entire bytecode file.
    pub fn disassemble(&mut self) -> Result<String> {
        let mut output = String::new();

        // Parse and display header
        let num_globals = self.parse_header(&mut output)?;

        // Parse globals
        self.parse_globals(&mut output, num_globals)?;

        // Parse functions
        self.parse_functions(&mut output)?;

        Ok(output)
    }

    /// Disassemble just the code section.
    pub fn disassemble_code(&mut self, code: &'a [u8]) -> Result<Vec<DisasmInst>> {
        self.bytes = code;
        self.pos = 0;

        let mut instructions = Vec::new();

        while self.pos < self.bytes.len() {
            if let Some(inst) = self.decode_instruction()? {
                instructions.push(inst);
            } else {
                break;
            }
        }

        Ok(instructions)
    }

    /// Parse the file header using zkir-spec v3.4 format (32 bytes).
    /// Returns 0 for num_globals since the new format doesn't store global count in header.
    fn parse_header(&mut self, output: &mut String) -> Result<u16> {
        if self.bytes.len() < ProgramHeader::SIZE {
            return Err(anyhow!("File too small for header (need {} bytes)", ProgramHeader::SIZE));
        }

        // Magic (u32 at offset 0)
        let magic = u32::from_le_bytes([
            self.bytes[0], self.bytes[1], self.bytes[2], self.bytes[3]
        ]);
        if magic != ZKIR_MAGIC_U32 {
            return Err(anyhow!("Invalid magic: expected {:#010x}, got {:#010x}", ZKIR_MAGIC_U32, magic));
        }

        // Version (u32 at offset 4)
        let version = u32::from_le_bytes([
            self.bytes[4], self.bytes[5], self.bytes[6], self.bytes[7]
        ]);
        if version != ZKIR_VERSION_U32 {
            writeln!(output, "; Warning: version {:#010x} (expected {:#010x})", version, ZKIR_VERSION_U32)?;
        }

        // Config (bytes 8-10)
        let limb_bits = self.bytes[8];
        let data_limbs = self.bytes[9];
        let addr_limbs = self.bytes[10];
        let _flags = self.bytes[11];

        // Entry point (u32 at offset 12)
        let entry_point = u32::from_le_bytes([
            self.bytes[12], self.bytes[13], self.bytes[14], self.bytes[15]
        ]);

        // Code size (u32 at offset 16)
        let code_size = u32::from_le_bytes([
            self.bytes[16], self.bytes[17], self.bytes[18], self.bytes[19]
        ]);

        // Data size (u32 at offset 20)
        let data_size = u32::from_le_bytes([
            self.bytes[20], self.bytes[21], self.bytes[22], self.bytes[23]
        ]);

        // BSS size (u32 at offset 24)
        let bss_size = u32::from_le_bytes([
            self.bytes[24], self.bytes[25], self.bytes[26], self.bytes[27]
        ]);

        // Stack size (u32 at offset 28)
        let stack_size = u32::from_le_bytes([
            self.bytes[28], self.bytes[29], self.bytes[30], self.bytes[31]
        ]);

        let version_major = (version >> 16) & 0xFFFF;
        let version_minor = version & 0xFFFF;
        writeln!(output, "; ZKIR bytecode v{}.{}", version_major, version_minor)?;
        writeln!(output, "; Config: {} bits × {} data limbs, {} addr limbs",
                 limb_bits, data_limbs, addr_limbs)?;
        writeln!(output, "; Entry: {:#010x}, Code: {} bytes, Data: {} bytes",
                 entry_point, code_size, data_size)?;
        writeln!(output, "; BSS: {} bytes, Stack: {} bytes", bss_size, stack_size)?;

        self.pos = ProgramHeader::SIZE;

        writeln!(output)?;
        // The v3.4 header doesn't include global count - globals are in data section
        // Return 0 to skip globals parsing (they're embedded in data)
        Ok(0)
    }

    /// Parse globals section.
    fn parse_globals(&mut self, output: &mut String, num_globals: u16) -> Result<()> {
        if num_globals == 0 {
            return Ok(());
        }

        writeln!(output, ".data")?;

        for _ in 0..num_globals {
            // Name length and name
            if self.pos >= self.bytes.len() {
                writeln!(output, "; truncated global section")?;
                break;
            }

            let name_len = self.bytes[self.pos] as usize;
            self.pos += 1;

            if self.pos + name_len > self.bytes.len() {
                writeln!(output, "; truncated global name")?;
                break;
            }

            let name = String::from_utf8_lossy(&self.bytes[self.pos..self.pos + name_len]).to_string();
            self.pos += name_len;

            // Size (u32 LE)
            if self.pos + 4 > self.bytes.len() {
                writeln!(output, "; truncated global size")?;
                break;
            }
            let size = u32::from_le_bytes([
                self.bytes[self.pos], self.bytes[self.pos + 1],
                self.bytes[self.pos + 2], self.bytes[self.pos + 3]
            ]);
            self.pos += 4;

            // Alignment (u32 LE)
            if self.pos + 4 > self.bytes.len() {
                writeln!(output, "; truncated global alignment")?;
                break;
            }
            let align = u32::from_le_bytes([
                self.bytes[self.pos], self.bytes[self.pos + 1],
                self.bytes[self.pos + 2], self.bytes[self.pos + 3]
            ]);
            self.pos += 4;

            // Flags
            if self.pos >= self.bytes.len() {
                writeln!(output, "; truncated global flags")?;
                break;
            }
            let is_const = self.bytes[self.pos] != 0;
            self.pos += 1;

            // Init data length
            if self.pos + 4 > self.bytes.len() {
                writeln!(output, "; truncated global init length")?;
                break;
            }
            let init_len = u32::from_le_bytes([
                self.bytes[self.pos], self.bytes[self.pos + 1],
                self.bytes[self.pos + 2], self.bytes[self.pos + 3]
            ]) as usize;
            self.pos += 4;

            // Format output
            let section = if is_const { ".rodata" } else { ".data" };
            writeln!(output, "  {}:", name)?;
            writeln!(output, "    .section {}", section)?;
            writeln!(output, "    .align {}", align)?;
            writeln!(output, "    .size {}", size)?;

            // Init data
            if init_len > 0 {
                if self.pos + init_len > self.bytes.len() {
                    writeln!(output, "    ; truncated init data")?;
                    break;
                }

                let init_data = &self.bytes[self.pos..self.pos + init_len];
                self.pos += init_len;

                // Format init data as hex bytes (up to 16 per line)
                write!(output, "    .byte ")?;
                for (i, byte) in init_data.iter().enumerate() {
                    if i > 0 && i % 16 == 0 {
                        writeln!(output)?;
                        write!(output, "    .byte ")?;
                    } else if i > 0 {
                        write!(output, ", ")?;
                    }
                    write!(output, "0x{:02x}", byte)?;
                }
                writeln!(output)?;
            } else {
                writeln!(output, "    .zero {}", size)?;
            }

            writeln!(output)?;
        }

        Ok(())
    }

    /// Parse functions section.
    fn parse_functions(&mut self, output: &mut String) -> Result<()> {
        writeln!(output, ".text")?;

        // Parse function table and code
        while self.pos < self.bytes.len() {
            // Try to parse a function entry
            if self.pos + 9 > self.bytes.len() {
                break;
            }

            let name_len = self.bytes[self.pos] as usize;
            self.pos += 1;

            if self.pos + name_len + 8 > self.bytes.len() {
                break;
            }

            let name = String::from_utf8_lossy(&self.bytes[self.pos..self.pos + name_len]).to_string();
            self.pos += name_len;

            let code_offset = u32::from_le_bytes([
                self.bytes[self.pos], self.bytes[self.pos + 1],
                self.bytes[self.pos + 2], self.bytes[self.pos + 3]
            ]);
            self.pos += 4;

            let code_size = u32::from_le_bytes([
                self.bytes[self.pos], self.bytes[self.pos + 1],
                self.bytes[self.pos + 2], self.bytes[self.pos + 3]
            ]);
            self.pos += 4;

            writeln!(output)?;
            writeln!(output, "{}:  ; offset={:#x}, size={}", name, code_offset, code_size)?;

            // Disassemble function code
            let code_start = code_offset as usize;
            let code_end = code_start + code_size as usize;

            if code_end <= self.bytes.len() {
                let code = &self.bytes[code_start..code_end];
                let mut code_disasm = Disassembler::new(code);
                let insts = code_disasm.disassemble_code(code)?;

                for inst in insts {
                    writeln!(output, "  {}", inst)?;
                }
            }
        }

        Ok(())
    }

    /// Decode a single 32-bit instruction using zkir-spec v3.4 format.
    fn decode_instruction(&mut self) -> Result<Option<DisasmInst>> {
        if self.pos + 4 > self.bytes.len() {
            return Ok(None);
        }

        let offset = self.pos as u32;

        // Read 32-bit instruction (little-endian)
        let inst = u32::from_le_bytes([
            self.bytes[self.pos],
            self.bytes[self.pos + 1],
            self.bytes[self.pos + 2],
            self.bytes[self.pos + 3],
        ]);
        self.pos += 4;

        let bytes = inst.to_le_bytes().to_vec();
        let opcode_val = extract_opcode(inst) as u8;

        let opcode = match Self::byte_to_opcode(opcode_val) {
            Some(op) => op,
            None => {
                return Ok(Some(DisasmInst {
                    offset,
                    bytes,
                    opcode: Opcode::NOP,
                    operands: format!(".word {:#010x}", inst),
                    comment: Some(format!("unknown opcode {:#04x}", opcode_val)),
                }));
            }
        };

        // Decode based on instruction format
        let (operands, comment) = self.decode_operands_v3(opcode, inst)?;

        Ok(Some(DisasmInst {
            offset,
            bytes,
            opcode,
            operands,
            comment,
        }))
    }

    /// Decode operands for a 32-bit instruction using zkir-spec v3.4 format.
    fn decode_operands_v3(&self, opcode: Opcode, inst: u32) -> Result<(String, Option<String>)> {
        let operands = match opcode {
            // R-type: [opcode:7][rd:4][rs1:4][rs2:4][funct:13]
            Opcode::ADD | Opcode::SUB | Opcode::MUL | Opcode::MULH |
            Opcode::DIV | Opcode::DIVU | Opcode::REM | Opcode::REMU |
            Opcode::AND | Opcode::OR | Opcode::XOR |
            Opcode::SLL | Opcode::SRL | Opcode::SRA |
            Opcode::SLT | Opcode::SLTU | Opcode::SGE | Opcode::SGEU |
            Opcode::SEQ | Opcode::SNE => {
                let rd = extract_rd(inst);
                let rs1 = extract_rs1(inst);
                let rs2 = extract_rs2(inst);
                format!("r{}, r{}, r{}", rd, rs1, rs2)
            }

            // I-type: [opcode:7][rd:4][rs1:4][imm:17]
            Opcode::ADDI | Opcode::ANDI | Opcode::ORI | Opcode::XORI |
            Opcode::SLLI | Opcode::SRLI | Opcode::SRAI => {
                let rd = extract_rd(inst);
                let rs1 = extract_rs1(inst);
                let imm = extract_imm_signed(inst);
                format!("r{}, r{}, {}", rd, rs1, imm)
            }

            // Load (I-type): [opcode:7][rd:4][rs1:4][imm:17]
            Opcode::LB | Opcode::LBU | Opcode::LH | Opcode::LHU |
            Opcode::LW | Opcode::LD => {
                let rd = extract_rd(inst);
                let base = extract_rs1(inst);
                let offset = extract_imm_signed(inst);
                format!("r{}, {}(r{})", rd, offset, base)
            }

            // Store (S-type): [opcode:7][rs1:4][rs2:4][imm:17]
            Opcode::SB | Opcode::SH | Opcode::SW | Opcode::SD => {
                let base = extract_stype_rs1(inst);
                let rs = extract_stype_rs2(inst);
                let offset = extract_stype_imm(inst) as i32;
                // Sign extend if bit 16 is set
                let offset = if offset & (1 << 16) != 0 {
                    offset - (1 << 17)
                } else {
                    offset
                };
                format!("r{}, {}(r{})", rs, offset, base)
            }

            // Branch (B-type): [opcode:7][rs1:4][rs2:4][offset:17]
            Opcode::BEQ | Opcode::BNE | Opcode::BLT | Opcode::BGE |
            Opcode::BLTU | Opcode::BGEU => {
                let rs1 = extract_stype_rs1(inst);
                let rs2 = extract_stype_rs2(inst);
                let offset = extract_imm_signed(inst);
                format!("r{}, r{}, {:+}", rs1, rs2, offset)
            }

            // JAL (J-type): [opcode:7][rd:4][offset:21]
            Opcode::JAL => {
                let rd = extract_rd(inst);
                let offset = extract_offset_signed(inst);
                format!("r{}, {:+}", rd, offset)
            }

            // JALR (I-type): [opcode:7][rd:4][rs1:4][imm:17]
            Opcode::JALR => {
                let rd = extract_rd(inst);
                let rs1 = extract_rs1(inst);
                let offset = extract_imm_signed(inst);
                format!("r{}, r{}, {}", rd, rs1, offset)
            }

            // CMOV (R-type with funct): [opcode:7][rd:4][rs1:4][rs2:4][cond:13]
            Opcode::CMOV | Opcode::CMOVZ | Opcode::CMOVNZ => {
                let rd = extract_rd(inst);
                let rs1 = extract_rs1(inst);
                let rs2 = extract_rs2(inst);
                let cond = zkir_spec::encoding::extract_funct(inst);
                format!("r{}, r{}, r{}, r{}", rd, cond, rs1, rs2)
            }

            // System
            Opcode::ECALL | Opcode::EBREAK => String::new(),

            // Pseudo-ops that may appear in bytecode (for debugging)
            Opcode::MOV => {
                // MOV is ADDI rd, rs, 0
                let rd = extract_rd(inst);
                let rs = extract_rs1(inst);
                format!("r{}, r{}", rd, rs)
            }

            Opcode::NOT => {
                // NOT is XORI rd, rs, -1
                let rd = extract_rd(inst);
                let rs = extract_rs1(inst);
                format!("r{}, r{}", rd, rs)
            }

            Opcode::LI => {
                // LI is ADDI rd, zero, imm
                let rd = extract_rd(inst);
                let imm = extract_imm_signed(inst);
                format!("r{}, {}", rd, imm)
            }

            Opcode::NOP => String::new(),

            Opcode::RET => String::new(),

            Opcode::CALL => {
                // CALL is JAL ra, offset
                let offset = extract_offset_signed(inst);
                format!("{:+}", offset)
            }

            Opcode::RCHK => {
                let rd = extract_rd(inst);
                format!("r{}", rd)
            }

            Opcode::PHI => {
                let rd = extract_rd(inst);
                format!("r{} ; PHI (should not appear in bytecode)", rd)
            }

            Opcode::CALLR => {
                // CALLR is JALR ra, rs, 0
                let rs = extract_rs1(inst);
                format!("r{} ; indirect call", rs)
            }
        };

        Ok((operands, None))
    }

    /// Convert byte to opcode.
    /// Values aligned with zkir-spec canonical definitions.
    fn byte_to_opcode(byte: u8) -> Option<Opcode> {
        use zkir_spec::Opcode as SpecOpcode;

        // Try real opcodes from zkir-spec first (0x00-0x51)
        if let Some(spec_opcode) = SpecOpcode::from_u8(byte) {
            // Convert zkir-spec PascalCase to MIR SCREAMING_SNAKE_CASE
            return Some(match spec_opcode {
                // Arithmetic (0x00-0x08)
                SpecOpcode::Add => Opcode::ADD,
                SpecOpcode::Sub => Opcode::SUB,
                SpecOpcode::Mul => Opcode::MUL,
                SpecOpcode::Mulh => Opcode::MULH,
                SpecOpcode::Divu => Opcode::DIVU,
                SpecOpcode::Remu => Opcode::REMU,
                SpecOpcode::Div => Opcode::DIV,
                SpecOpcode::Rem => Opcode::REM,
                SpecOpcode::Addi => Opcode::ADDI,

                // Logical (0x10-0x15)
                SpecOpcode::And => Opcode::AND,
                SpecOpcode::Or => Opcode::OR,
                SpecOpcode::Xor => Opcode::XOR,
                SpecOpcode::Andi => Opcode::ANDI,
                SpecOpcode::Ori => Opcode::ORI,
                SpecOpcode::Xori => Opcode::XORI,

                // Shift (0x18-0x1D)
                SpecOpcode::Sll => Opcode::SLL,
                SpecOpcode::Srl => Opcode::SRL,
                SpecOpcode::Sra => Opcode::SRA,
                SpecOpcode::Slli => Opcode::SLLI,
                SpecOpcode::Srli => Opcode::SRLI,
                SpecOpcode::Srai => Opcode::SRAI,

                // Compare (0x20-0x25)
                SpecOpcode::Sltu => Opcode::SLTU,
                SpecOpcode::Sgeu => Opcode::SGEU,
                SpecOpcode::Slt => Opcode::SLT,
                SpecOpcode::Sge => Opcode::SGE,
                SpecOpcode::Seq => Opcode::SEQ,
                SpecOpcode::Sne => Opcode::SNE,

                // Conditional Move (0x26-0x28)
                SpecOpcode::Cmov => Opcode::CMOV,
                SpecOpcode::Cmovz => Opcode::CMOVZ,
                SpecOpcode::Cmovnz => Opcode::CMOVNZ,

                // Load (0x30-0x35)
                SpecOpcode::Lb => Opcode::LB,
                SpecOpcode::Lbu => Opcode::LBU,
                SpecOpcode::Lh => Opcode::LH,
                SpecOpcode::Lhu => Opcode::LHU,
                SpecOpcode::Lw => Opcode::LW,
                SpecOpcode::Ld => Opcode::LD,

                // Store (0x38-0x3B)
                SpecOpcode::Sb => Opcode::SB,
                SpecOpcode::Sh => Opcode::SH,
                SpecOpcode::Sw => Opcode::SW,
                SpecOpcode::Sd => Opcode::SD,

                // Branch (0x40-0x45)
                SpecOpcode::Beq => Opcode::BEQ,
                SpecOpcode::Bne => Opcode::BNE,
                SpecOpcode::Blt => Opcode::BLT,
                SpecOpcode::Bge => Opcode::BGE,
                SpecOpcode::Bltu => Opcode::BLTU,
                SpecOpcode::Bgeu => Opcode::BGEU,

                // Jump (0x48-0x49)
                SpecOpcode::Jal => Opcode::JAL,
                SpecOpcode::Jalr => Opcode::JALR,

                // System (0x50-0x51)
                SpecOpcode::Ecall => Opcode::ECALL,
                SpecOpcode::Ebreak => Opcode::EBREAK,
            });
        }

        // Handle pseudo-ops (0xF0+) - compiler internal, shouldn't appear in final bytecode
        match byte {
            0xF0 => Some(Opcode::MOV),
            0xF1 => Some(Opcode::LI),
            0xF2 => Some(Opcode::NOP),
            0xF3 => Some(Opcode::RET),
            0xF4 => Some(Opcode::CALL),
            0xF5 => Some(Opcode::RCHK),
            0xF6 => Some(Opcode::PHI),
            0xF7 => Some(Opcode::CALLR),
            0xF8 => Some(Opcode::NOT),
            _ => None,
        }
    }
}

/// Disassemble ZKIR bytecode to a string.
pub fn disassemble(bytes: &[u8]) -> Result<String> {
    let mut disasm = Disassembler::new(bytes);
    disasm.disassemble()
}

/// Disassemble raw code bytes (no header).
pub fn disassemble_code(code: &[u8]) -> Result<Vec<DisasmInst>> {
    let mut disasm = Disassembler::new(code);
    disasm.disassemble_code(code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkir_spec::encoding::{encode_rtype, encode_itype};
    use zkir_spec::Opcode as SpecOpcode;

    #[test]
    fn test_byte_to_opcode() {
        // Test opcodes aligned with zkir-spec values
        assert_eq!(Disassembler::byte_to_opcode(0x00), Some(Opcode::ADD));
        assert_eq!(Disassembler::byte_to_opcode(0x08), Some(Opcode::ADDI));  // ADDI = 0x08
        assert_eq!(Disassembler::byte_to_opcode(0x34), Some(Opcode::LW));    // LW = 0x34
        assert_eq!(Disassembler::byte_to_opcode(0x49), Some(Opcode::JALR));  // JALR = 0x49
        assert_eq!(Disassembler::byte_to_opcode(0xF3), Some(Opcode::RET));
        assert_eq!(Disassembler::byte_to_opcode(0xFF), None);
    }

    #[test]
    fn test_disassemble_add() {
        // ADD r10, r11, r12 using zkir-spec v3.4 encoding
        let inst = encode_rtype(SpecOpcode::Add, 10, 11, 12, 0);
        let code = inst.to_le_bytes().to_vec();
        let insts = disassemble_code(&code).unwrap();

        assert_eq!(insts.len(), 1);
        assert_eq!(insts[0].opcode, Opcode::ADD);
        assert_eq!(insts[0].operands, "r10, r11, r12");
    }

    #[test]
    fn test_disassemble_addi() {
        // ADDI r5, r6, 100 using zkir-spec v3.4 encoding
        let inst = encode_itype(SpecOpcode::Addi, 5, 6, 100);
        let code = inst.to_le_bytes().to_vec();
        let insts = disassemble_code(&code).unwrap();

        assert_eq!(insts.len(), 1);
        assert_eq!(insts[0].opcode, Opcode::ADDI);
        assert_eq!(insts[0].operands, "r5, r6, 100");
    }

    #[test]
    fn test_disassemble_load() {
        // LW r3, 8(r4) using zkir-spec v3.4 encoding (I-type)
        let inst = encode_itype(SpecOpcode::Lw, 3, 4, 8);
        let code = inst.to_le_bytes().to_vec();
        let insts = disassemble_code(&code).unwrap();

        assert_eq!(insts.len(), 1);
        assert_eq!(insts[0].opcode, Opcode::LW);
        assert_eq!(insts[0].operands, "r3, 8(r4)");
    }

    #[test]
    fn test_disassemble_ret() {
        // RET (expands to JALR zero, ra, 0)
        let inst = encode_itype(SpecOpcode::Jalr, 0, 1, 0);
        let code = inst.to_le_bytes().to_vec();
        let insts = disassemble_code(&code).unwrap();

        assert_eq!(insts.len(), 1);
        assert_eq!(insts[0].opcode, Opcode::JALR);
        assert_eq!(insts[0].operands, "r0, r1, 0");
    }

    #[test]
    fn test_disassemble_sequence() {
        // Multiple instructions with zkir-spec v3.4 encoding
        let mut code = Vec::new();

        // ADDI r10, r0, 10
        code.extend(encode_itype(SpecOpcode::Addi, 10, 0, 10).to_le_bytes());
        // ADDI r11, r0, 20
        code.extend(encode_itype(SpecOpcode::Addi, 11, 0, 20).to_le_bytes());
        // ADD r12, r10, r11
        code.extend(encode_rtype(SpecOpcode::Add, 12, 10, 11, 0).to_le_bytes());
        // JALR r0, r1, 0 (RET)
        code.extend(encode_itype(SpecOpcode::Jalr, 0, 1, 0).to_le_bytes());

        let insts = disassemble_code(&code).unwrap();
        assert_eq!(insts.len(), 4);
        assert_eq!(insts[0].opcode, Opcode::ADDI);
        assert_eq!(insts[1].opcode, Opcode::ADDI);
        assert_eq!(insts[2].opcode, Opcode::ADD);
        assert_eq!(insts[3].opcode, Opcode::JALR);
    }

    #[test]
    fn test_disasm_inst_display() {
        let inst = encode_rtype(SpecOpcode::Add, 10, 11, 12, 0);
        let bytes = inst.to_le_bytes().to_vec();

        let disasm_inst = DisasmInst {
            offset: 0x100,
            bytes,
            opcode: Opcode::ADD,
            operands: "r10, r11, r12".to_string(),
            comment: None,
        };

        let display = format!("{}", disasm_inst);
        assert!(display.contains("00000100:"));
        assert!(display.contains("add"));
        assert!(display.contains("r10, r11, r12"));
    }

    #[test]
    fn test_disassemble_header_v3() {
        // Build a valid zkir-spec v3.4 header (32 bytes)
        let header = ProgramHeader {
            magic: ZKIR_MAGIC_U32,
            version: ZKIR_VERSION_U32,
            limb_bits: 20,
            data_limbs: 2,
            addr_limbs: 2,
            flags: 0,
            entry_point: zkir_spec::memory::CODE_BASE as u32,
            code_size: 0,
            data_size: 0,
            bss_size: 0,
            stack_size: zkir_spec::memory::DEFAULT_STACK_SIZE as u32,
        };

        let bytes = header.to_bytes().to_vec();
        let output = disassemble(&bytes).unwrap();

        assert!(output.contains("v3.4"), "Should show version 3.4");
        assert!(output.contains("20 bits"), "Should show limb bits");
        assert!(output.contains("2 data limbs"), "Should show data limbs");
    }
}
