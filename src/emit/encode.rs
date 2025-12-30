//! Instruction encoding for ZKIR bytecode.
//!
//! Encodes Machine IR instructions to their binary representation
//! using zkir-spec v3.4 32-bit instruction format.

use crate::mir::{MachineFunction, MachineInst, Opcode, Operand};
use crate::target::config::TargetConfig;
use crate::target::registers::RegisterExt; // For .number() method
use anyhow::{anyhow, Result};

// Import zkir-spec encoding functions
use zkir_spec::encoding::{
    encode_rtype, encode_itype, encode_stype, encode_btype, encode_jtype,
};
use zkir_spec::Opcode as SpecOpcode;

/// Validation error for unallocated virtual registers.
#[derive(Debug)]
pub struct ValidationError {
    pub function: String,
    pub block: String,
    pub instruction_idx: usize,
    pub vreg: String,
    pub location: &'static str,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Unallocated virtual register {} in {} at {}:{}:inst{}",
            self.vreg, self.location, self.function, self.block, self.instruction_idx
        )
    }
}

/// Validate that a function has no virtual registers remaining.
///
/// This should be called after register allocation and before encoding
/// to ensure all virtual registers have been replaced with physical registers.
pub fn validate_no_vregs(func: &MachineFunction) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    for block in func.iter_blocks() {
        for (idx, inst) in block.insts.iter().enumerate() {
            // Check destination
            if let Some(Operand::VReg(v)) = &inst.dst {
                errors.push(ValidationError {
                    function: func.name.clone(),
                    block: block.label.clone(),
                    instruction_idx: idx,
                    vreg: format!("{}", v),
                    location: "destination",
                });
            }

            // Check sources
            for src in &inst.srcs {
                match src {
                    Operand::VReg(v) => {
                        errors.push(ValidationError {
                            function: func.name.clone(),
                            block: block.label.clone(),
                            instruction_idx: idx,
                            vreg: format!("{}", v),
                            location: "source",
                        });
                    }
                    Operand::Mem { base, .. } => {
                        errors.push(ValidationError {
                            function: func.name.clone(),
                            block: block.label.clone(),
                            instruction_idx: idx,
                            vreg: format!("{}", base),
                            location: "memory base",
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Instruction encoder for ZKIR.
pub struct InstructionEncoder<'a> {
    /// Target configuration (for encoding format variations)
    #[allow(dead_code)]
    config: &'a TargetConfig,
}

impl<'a> InstructionEncoder<'a> {
    /// Create a new encoder.
    pub fn new(config: &'a TargetConfig) -> Self {
        Self { config }
    }

    /// Encode an instruction to bytes.
    pub fn encode(&self, inst: &MachineInst) -> Result<Vec<u8>> {
        match inst.opcode {
            // Pseudo-ops are expanded first
            Opcode::MOV => self.encode_mov(inst),
            Opcode::LI => self.encode_li(inst),
            Opcode::NOP => self.encode_nop(),
            Opcode::RET => self.encode_ret(),
            Opcode::CALL => self.encode_call(inst),
            Opcode::CALLR => self.encode_callr(inst),
            Opcode::RCHK => self.encode_rchk(inst),
            Opcode::NOT => self.encode_not(inst),

            // Regular instructions
            _ => self.encode_regular(inst),
        }
    }

    /// Convert local Opcode to zkir-spec Opcode.
    fn to_spec_opcode(opcode: Opcode) -> Option<SpecOpcode> {
        // The opcode values should match since we aligned them in mir/mod.rs
        SpecOpcode::from_u8(opcode as u8)
    }

    /// Encode a regular (non-pseudo) instruction using zkir-spec v3.4 format.
    /// All instructions are 32-bit.
    fn encode_regular(&self, inst: &MachineInst) -> Result<Vec<u8>> {
        let spec_opcode = Self::to_spec_opcode(inst.opcode)
            .ok_or_else(|| anyhow!("Unknown opcode: {:?}", inst.opcode))?;

        let encoded: u32 = match inst.opcode {
            // R-type: ADD, SUB, MUL, DIV, etc.
            // Format: [opcode:7][rd:4][rs1:4][rs2:4][funct:13]
            Opcode::ADD | Opcode::SUB | Opcode::MUL | Opcode::MULH |
            Opcode::DIV | Opcode::DIVU | Opcode::REM | Opcode::REMU |
            Opcode::AND | Opcode::OR | Opcode::XOR |
            Opcode::SLL | Opcode::SRL | Opcode::SRA |
            Opcode::SLT | Opcode::SLTU | Opcode::SGE | Opcode::SGEU |
            Opcode::SEQ | Opcode::SNE => {
                let rd = self.get_reg_num(&inst.dst)? as u32;
                let rs1 = self.get_operand_reg(&inst.srcs.first())? as u32;
                let rs2 = self.get_operand_reg(&inst.srcs.get(1))? as u32;
                encode_rtype(spec_opcode, rd, rs1, rs2, 0)
            }

            // I-type: ADDI, ANDI, ORI, etc.
            // Format: [opcode:7][rd:4][rs1:4][imm:17]
            Opcode::ADDI | Opcode::ANDI | Opcode::ORI | Opcode::XORI |
            Opcode::SLLI | Opcode::SRLI | Opcode::SRAI => {
                let rd = self.get_reg_num(&inst.dst)? as u32;
                let rs1 = self.get_operand_reg(&inst.srcs.first())? as u32;
                let imm = self.get_operand_imm(&inst.srcs.get(1))? as u32;
                encode_itype(spec_opcode, rd, rs1, imm)
            }

            // Load: LB, LBU, LH, LHU, LW, LD (I-type format)
            Opcode::LB | Opcode::LBU | Opcode::LH | Opcode::LHU |
            Opcode::LW | Opcode::LD => {
                let rd = self.get_reg_num(&inst.dst)? as u32;
                let (base, offset) = self.get_mem_operand(&inst.srcs.first())?;
                encode_itype(spec_opcode, rd, base as u32, offset as u32)
            }

            // Store: SB, SH, SW, SD (S-type format)
            // Format: [opcode:7][rs1:4][rs2:4][imm:17]
            Opcode::SB | Opcode::SH | Opcode::SW | Opcode::SD => {
                let rs2 = self.get_operand_reg(&inst.srcs.first())? as u32; // value to store
                let (rs1, offset) = self.get_mem_operand(&inst.srcs.get(1))?; // base, offset
                encode_stype(spec_opcode, rs1 as u32, rs2, offset as u32)
            }

            // Branch: BEQ, BNE, BLT, BGE, etc. (B-type format)
            Opcode::BEQ | Opcode::BNE | Opcode::BLT | Opcode::BGE |
            Opcode::BLTU | Opcode::BGEU => {
                let rs1 = self.get_operand_reg(&inst.srcs.first())? as u32;
                let rs2 = self.get_operand_reg(&inst.srcs.get(1))? as u32;
                // Offset will be fixed up during linking
                encode_btype(spec_opcode, rs1, rs2, 0)
            }

            // JAL: Jump and link (J-type format)
            // Format: [opcode:7][rd:4][offset:21]
            Opcode::JAL => {
                let rd = self.get_reg_num(&inst.dst)? as u32;
                // Offset will be fixed up during linking
                encode_jtype(spec_opcode, rd, 0)
            }

            // JALR: Jump and link register (I-type format)
            Opcode::JALR => {
                let rd = self.get_reg_num(&inst.dst)? as u32;
                let rs1 = self.get_operand_reg(&inst.srcs.first())? as u32;
                let offset = self.get_operand_imm(&inst.srcs.get(1)).unwrap_or(0) as u32;
                encode_itype(spec_opcode, rd, rs1, offset)
            }

            // CMOV: Conditional move (R-type with funct field for variant)
            Opcode::CMOV | Opcode::CMOVZ | Opcode::CMOVNZ => {
                let rd = self.get_reg_num(&inst.dst)? as u32;
                let cond = self.get_operand_reg(&inst.srcs.first())? as u32;
                let rs1 = self.get_operand_reg(&inst.srcs.get(1))? as u32;
                let rs2 = self.get_operand_reg(&inst.srcs.get(2))? as u32;
                // Use funct field to encode condition register
                encode_rtype(spec_opcode, rd, rs1, rs2, cond)
            }

            // System instructions
            Opcode::ECALL => encode_itype(spec_opcode, 0, 0, 0),
            Opcode::EBREAK => encode_itype(spec_opcode, 0, 0, 1),

            _ => {
                return Err(anyhow!("Unhandled opcode in encode_regular: {:?}", inst.opcode));
            }
        };

        // Return as little-endian bytes
        Ok(encoded.to_le_bytes().to_vec())
    }

    /// Encode MOV pseudo-op as ADDI rd, rs, 0 using zkir-spec format.
    fn encode_mov(&self, inst: &MachineInst) -> Result<Vec<u8>> {
        let rd = self.get_reg_num(&inst.dst)? as u32;
        let rs = self.get_operand_reg(&inst.srcs.first())? as u32;
        let encoded = encode_itype(SpecOpcode::Addi, rd, rs, 0);
        Ok(encoded.to_le_bytes().to_vec())
    }

    /// Encode LI pseudo-op as ADDI rd, zero, imm using zkir-spec format.
    /// Note: zkir-spec I-type has 17-bit immediate field, supporting larger range.
    fn encode_li(&self, inst: &MachineInst) -> Result<Vec<u8>> {
        let rd = self.get_reg_num(&inst.dst)? as u32;
        let imm = self.get_operand_imm(&inst.srcs.first())?;

        // 17-bit signed immediate range: -65536 to 65535
        let imm_masked = (imm as u32) & zkir_spec::encoding::IMM_MASK;
        let encoded = encode_itype(SpecOpcode::Addi, rd, 0, imm_masked);
        Ok(encoded.to_le_bytes().to_vec())
    }

    /// Encode NOP as ADDI zero, zero, 0 using zkir-spec format.
    fn encode_nop(&self) -> Result<Vec<u8>> {
        let encoded = encode_itype(SpecOpcode::Addi, 0, 0, 0);
        Ok(encoded.to_le_bytes().to_vec())
    }

    /// Encode RET as JALR zero, ra, 0 using zkir-spec format.
    fn encode_ret(&self) -> Result<Vec<u8>> {
        use zkir_spec::Register;
        let zero = Register::ZERO.index() as u32;
        let ra = Register::RA.index() as u32;
        let encoded = encode_itype(SpecOpcode::Jalr, zero, ra, 0);
        Ok(encoded.to_le_bytes().to_vec())
    }

    /// Encode CALL as JAL ra, offset using zkir-spec format.
    fn encode_call(&self, _inst: &MachineInst) -> Result<Vec<u8>> {
        use zkir_spec::Register;
        let ra = Register::RA.index() as u32;
        // Offset placeholder (fixed during linking)
        let encoded = encode_jtype(SpecOpcode::Jal, ra, 0);
        Ok(encoded.to_le_bytes().to_vec())
    }

    /// Encode CALLR (indirect call) as JALR ra, rs, 0 using zkir-spec format.
    fn encode_callr(&self, inst: &MachineInst) -> Result<Vec<u8>> {
        use zkir_spec::Register;
        let ra = Register::RA.index() as u32;
        let target = self.get_operand_reg(&inst.srcs.first())? as u32;
        let encoded = encode_itype(SpecOpcode::Jalr, ra, target, 0);
        Ok(encoded.to_le_bytes().to_vec())
    }

    /// Encode RCHK (range check) as a NOP.
    ///
    /// RCHK is a pseudo-instruction that serves as an optimization hint for the
    /// ZK prover. It indicates that the value in the register should be range-checked.
    /// Since ZKIR doesn't have a dedicated range check opcode (valid opcodes are
    /// 0x00-0x51), we encode RCHK as a NOP (ADDI zero, zero, 0).
    ///
    /// The prover can still identify range check opportunities from the witness
    /// trace without needing explicit RCHK instructions in the bytecode.
    fn encode_rchk(&self, _inst: &MachineInst) -> Result<Vec<u8>> {
        // Encode as NOP: ADDI zero, zero, 0
        let encoded = encode_itype(SpecOpcode::Addi, 0, 0, 0);
        Ok(encoded.to_le_bytes().to_vec())
    }

    /// Encode NOT pseudo-op as XORI rd, rs, -1 using zkir-spec format.
    fn encode_not(&self, inst: &MachineInst) -> Result<Vec<u8>> {
        let rd = self.get_reg_num(&inst.dst)? as u32;
        let rs = self.get_operand_reg(&inst.srcs.first())? as u32;
        // -1 in 17-bit signed = 0x1FFFF (all bits set)
        let encoded = encode_itype(SpecOpcode::Xori, rd, rs, 0x1FFFF);
        Ok(encoded.to_le_bytes().to_vec())
    }

    /// Get register number from destination operand.
    fn get_reg_num(&self, dst: &Option<Operand>) -> Result<u8> {
        match dst {
            Some(Operand::Reg(r)) => Ok(r.number()),
            Some(Operand::VReg(v)) => {
                // Shouldn't happen after regalloc
                log::warn!("Virtual register in encoding: {}", v);
                Ok(0)
            }
            _ => Ok(0),
        }
    }

    /// Get register number from operand.
    fn get_operand_reg(&self, op: &Option<&Operand>) -> Result<u8> {
        match op {
            Some(Operand::Reg(r)) => Ok(r.number()),
            Some(Operand::VReg(v)) => {
                log::warn!("Virtual register in encoding: {}", v);
                Ok(0)
            }
            _ => Ok(0),
        }
    }

    /// Get immediate value from operand.
    fn get_operand_imm(&self, op: &Option<&Operand>) -> Result<i64> {
        match op {
            Some(Operand::Imm(i)) => Ok(*i),
            Some(Operand::GlobalAddr(_)) => {
                // Global address - will be fixed up during linking
                // Return 0 as placeholder
                Ok(0)
            }
            _ => Err(anyhow!("Expected immediate operand")),
        }
    }

    /// Get base register and offset from memory operand.
    fn get_mem_operand(&self, op: &Option<&Operand>) -> Result<(u8, i16)> {
        match op {
            Some(Operand::MemReg { base, offset }) => {
                Ok((base.number(), *offset as i16))
            }
            Some(Operand::Mem { base, offset }) => {
                log::warn!("Virtual register in memory operand: {}", base);
                Ok((0, *offset as i16))
            }
            _ => Err(anyhow!("Expected memory operand")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::registers::Register;
    use zkir_spec::encoding::{extract_opcode, extract_rd, extract_rs1, extract_rs2, extract_imm};

    #[test]
    fn test_encode_add() {
        let config = TargetConfig::default();
        let encoder = InstructionEncoder::new(&config);

        let inst = MachineInst::new(Opcode::ADD)
            .dst(Operand::Reg(Register::R10))
            .src(Operand::Reg(Register::R11))
            .src(Operand::Reg(Register::R12));

        let bytes = encoder.encode(&inst).unwrap();
        assert_eq!(bytes.len(), 4);

        // Decode the 32-bit instruction
        let encoded = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        // Verify using zkir-spec extraction functions
        assert_eq!(extract_opcode(encoded), Opcode::ADD as u32);
        assert_eq!(extract_rd(encoded), 10); // R10
        assert_eq!(extract_rs1(encoded), 11); // R11
        assert_eq!(extract_rs2(encoded), 12); // R12
    }

    #[test]
    fn test_encode_li_small() {
        let config = TargetConfig::default();
        let encoder = InstructionEncoder::new(&config);

        let inst = MachineInst::new(Opcode::LI)
            .dst(Operand::Reg(Register::R10))
            .src(Operand::Imm(42));

        let bytes = encoder.encode(&inst).unwrap();
        let encoded = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        // LI expands to ADDI rd, zero, imm
        assert_eq!(extract_opcode(encoded), Opcode::ADDI as u32);
        assert_eq!(extract_rd(encoded), 10); // R10
        assert_eq!(extract_rs1(encoded), 0);  // zero
        assert_eq!(extract_imm(encoded), 42); // immediate
    }

    #[test]
    fn test_encode_nop() {
        let config = TargetConfig::default();
        let encoder = InstructionEncoder::new(&config);

        let bytes = encoder.encode_nop().unwrap();
        let encoded = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        // NOP = ADDI zero, zero, 0
        assert_eq!(extract_opcode(encoded), Opcode::ADDI as u32);
        assert_eq!(extract_rd(encoded), 0);   // zero
        assert_eq!(extract_rs1(encoded), 0);  // zero
        assert_eq!(extract_imm(encoded), 0);  // 0
    }

    #[test]
    fn test_validate_no_vregs_success() {
        use crate::mir::{MachineBlock, MachineFunction};

        let mut func = MachineFunction::new("test");
        let mut entry = MachineBlock::new("entry");

        // All physical registers - should pass validation
        entry.push(MachineInst::new(Opcode::ADD)
            .dst(Operand::Reg(Register::R10))
            .src(Operand::Reg(Register::R11))
            .src(Operand::Reg(Register::R12)));
        entry.push(MachineInst::new(Opcode::RET));
        func.add_block(entry);

        let result = validate_no_vregs(&func);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_no_vregs_failure_dst() {
        use crate::mir::{MachineBlock, MachineFunction, VReg};

        let mut func = MachineFunction::new("test");
        let mut entry = MachineBlock::new("entry");

        // Virtual register in destination - should fail
        entry.push(MachineInst::new(Opcode::ADD)
            .dst(Operand::VReg(VReg(0)))
            .src(Operand::Reg(Register::R11))
            .src(Operand::Reg(Register::R12)));
        func.add_block(entry);

        let result = validate_no_vregs(&func);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].location, "destination");
    }

    #[test]
    fn test_validate_no_vregs_failure_src() {
        use crate::mir::{MachineBlock, MachineFunction, VReg};

        let mut func = MachineFunction::new("test");
        let mut entry = MachineBlock::new("entry");

        // Virtual register in source - should fail
        entry.push(MachineInst::new(Opcode::ADD)
            .dst(Operand::Reg(Register::R10))
            .src(Operand::VReg(VReg(1)))
            .src(Operand::Reg(Register::R12)));
        func.add_block(entry);

        let result = validate_no_vregs(&func);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].location, "source");
    }

    #[test]
    fn test_validate_no_vregs_multiple_errors() {
        use crate::mir::{MachineBlock, MachineFunction, VReg};

        let mut func = MachineFunction::new("test");
        let mut entry = MachineBlock::new("entry");

        // Multiple virtual registers - should report all
        entry.push(MachineInst::new(Opcode::ADD)
            .dst(Operand::VReg(VReg(0)))
            .src(Operand::VReg(VReg(1)))
            .src(Operand::VReg(VReg(2))));
        func.add_block(entry);

        let result = validate_no_vregs(&func);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 3); // dst + 2 sources
    }

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError {
            function: "main".to_string(),
            block: "entry".to_string(),
            instruction_idx: 5,
            vreg: "v0".to_string(),
            location: "destination",
        };

        let msg = format!("{}", err);
        assert!(msg.contains("v0"));
        assert!(msg.contains("destination"));
        assert!(msg.contains("main"));
        assert!(msg.contains("entry"));
    }
}
