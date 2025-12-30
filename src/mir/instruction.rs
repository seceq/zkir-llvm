//! Machine instructions for ZKIR.
//!
//! This module defines the Machine IR (MIR) instruction set used by zkir-llvm during
//! compilation. These instructions represent the compiler's internal representation
//! before final code emission.
//!
//! # Opcode Design Philosophy
//!
//! ## Why Duplicate the Opcode Enum?
//!
//! zkir-llvm maintains its own `Opcode` enum rather than directly using `zkir_spec::Opcode`
//! for several important reasons:
//!
//! ### 1. Separation of Concerns
//!
//! - **MIR vs Bytecode**: The MIR represents compiler-internal instructions which may
//!   include pseudo-operations (MOV, LI, CALL, RET, etc.) that don't exist in the final
//!   bytecode. These pseudo-ops simplify earlier compiler passes but are lowered to
//!   real instructions during code emission.
//!
//! - **Compiler Flexibility**: The compiler needs freedom to add internal opcodes for
//!   optimization passes (PHI nodes, NOT, RCHK) without polluting the specification.
//!
//! ### 2. Different Responsibilities
//!
//! - **zkir-spec::Opcode**: Defines the canonical ISA specification that all ZKIR
//!   implementations must follow. It represents what can be encoded in bytecode.
//!
//! - **zkir-llvm::Opcode**: Represents the compiler's internal instruction set,
//!   including both real instructions and compiler-specific pseudo-operations.
//!
//! ### 3. Validation Strategy
//!
//! Rather than forcing a 1:1 correspondence, we:
//! - Keep real opcodes (ADD, SUB, LW, etc.) aligned with zkir-spec values
//! - Add compile-time validation tests (see `test_opcode_alignment_with_zkir_spec`)
//! - Reserve high opcode values (0xF0+) for pseudo-operations
//! - Fail fast if alignment drifts via automated testing
//!
//! ## Pseudo-Operations
//!
//! The following opcodes are zkir-llvm specific and don't appear in zkir-spec:
//!
//! - **MOV** (0xF0): Register-to-register move (lowered to ADDI rd, rs, 0)
//! - **LI** (0xF1): Load immediate (lowered to ADDI or LUI+ADDI)
//! - **NOP** (0xF2): No operation (lowered to ADDI zero, zero, 0)
//! - **RET** (0xF3): Return from function (lowered to JALR zero, ra, 0)
//! - **CALL** (0xF4): Call function (lowered to JAL)
//! - **CALLR** (0xF5): Call via register (lowered to JALR)
//! - **RCHK** (0xF6): Range check marker (optimization hint)
//! - **NOT** (0xF7): Bitwise NOT (lowered to XORI rd, rs, -1)
//! - **PHI** (0xF8): SSA phi node (eliminated before emission)
//!
//! These pseudo-ops exist to:
//! - Simplify IR generation from high-level languages
//! - Enable more aggressive optimizations
//! - Provide clearer debugging information
//! - Defer instruction selection decisions
//!
//! ## Alignment Guarantee
//!
//! All real opcodes (non-pseudo) maintain strict alignment with zkir-spec values.
//! This is verified by `test_opcode_alignment_with_zkir_spec()` which will fail
//! if any opcode value drifts from the specification.
//!
//! ## Example
//!
//! ```rust
//! use zkir_llvm::mir::Opcode;
//!
//! // Real opcodes match zkir-spec
//! assert_eq!(Opcode::ADD as u8, 0x00);
//! assert_eq!(Opcode::SUB as u8, 0x01);
//! assert_eq!(Opcode::LW as u8, 0x34);
//!
//! // Pseudo-ops are in the 0xF0+ range
//! assert_eq!(Opcode::MOV as u8, 0xF0);
//! assert_eq!(Opcode::RET as u8, 0xF3);
//! ```

use super::value::{VReg, ValueBounds};
use crate::debug::SourceLoc;
use crate::target::Register;
use serde::{Deserialize, Serialize};
use std::fmt;

/// ZKIR opcodes - values aligned with zkir-spec canonical definitions.
///
/// See zkir-spec/src/opcode.rs for the authoritative opcode encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Opcode {
    // ========== Arithmetic (0x00-0x08) - matches zkir_spec::Opcode ==========
    /// ADD: rd = rs1 + rs2
    ADD = 0x00,
    /// SUB: rd = rs1 - rs2
    SUB = 0x01,
    /// MUL: rd = (rs1 * rs2) lower bits
    MUL = 0x02,
    /// MULH: rd = (rs1 * rs2) upper bits
    MULH = 0x03,
    /// DIVU: rd = rs1 / rs2 (unsigned)
    DIVU = 0x04,
    /// REMU: rd = rs1 % rs2 (unsigned)
    REMU = 0x05,
    /// DIV: rd = rs1 / rs2 (signed)
    DIV = 0x06,
    /// REM: rd = rs1 % rs2 (signed)
    REM = 0x07,
    /// ADDI: rd = rs1 + imm
    ADDI = 0x08,

    // ========== Logical (0x10-0x15) - matches zkir_spec::Opcode ==========
    /// AND: rd = rs1 & rs2
    AND = 0x10,
    /// OR: rd = rs1 | rs2
    OR = 0x11,
    /// XOR: rd = rs1 ^ rs2
    XOR = 0x12,
    /// ANDI: rd = rs1 & imm
    ANDI = 0x13,
    /// ORI: rd = rs1 | imm
    ORI = 0x14,
    /// XORI: rd = rs1 ^ imm
    XORI = 0x15,

    // ========== Shift (0x18-0x1D) - matches zkir_spec::Opcode ==========
    /// SLL: rd = rs1 << rs2
    SLL = 0x18,
    /// SRL: rd = rs1 >> rs2 (logical)
    SRL = 0x19,
    /// SRA: rd = rs1 >> rs2 (arithmetic)
    SRA = 0x1A,
    /// SLLI: rd = rs1 << shamt
    SLLI = 0x1B,
    /// SRLI: rd = rs1 >> shamt (logical)
    SRLI = 0x1C,
    /// SRAI: rd = rs1 >> shamt (arithmetic)
    SRAI = 0x1D,

    // ========== Compare (0x20-0x25) - matches zkir_spec::Opcode ==========
    /// SLTU: rd = (rs1 < rs2) ? 1 : 0 (unsigned)
    SLTU = 0x20,
    /// SGEU: rd = (rs1 >= rs2) ? 1 : 0 (unsigned)
    SGEU = 0x21,
    /// SLT: rd = (rs1 < rs2) ? 1 : 0 (signed)
    SLT = 0x22,
    /// SGE: rd = (rs1 >= rs2) ? 1 : 0 (signed)
    SGE = 0x23,
    /// SEQ: rd = (rs1 == rs2) ? 1 : 0
    SEQ = 0x24,
    /// SNE: rd = (rs1 != rs2) ? 1 : 0
    SNE = 0x25,

    // ========== Conditional Move (0x26-0x28) - matches zkir_spec::Opcode ==========
    /// CMOV: rd = (rs2 != 0) ? rs1 : rd
    CMOV = 0x26,
    /// CMOVZ: rd = (rs2 == 0) ? rs1 : rd
    CMOVZ = 0x27,
    /// CMOVNZ: rd = (rs2 != 0) ? rs1 : rd
    CMOVNZ = 0x28,

    // ========== Load (0x30-0x35) - matches zkir_spec::Opcode ==========
    /// LB: rd = sign_extend(mem[rs1 + imm][7:0])
    LB = 0x30,
    /// LBU: rd = zero_extend(mem[rs1 + imm][7:0])
    LBU = 0x31,
    /// LH: rd = sign_extend(mem[rs1 + imm][15:0])
    LH = 0x32,
    /// LHU: rd = zero_extend(mem[rs1 + imm][15:0])
    LHU = 0x33,
    /// LW: rd = sign_extend(mem[rs1 + imm][31:0])
    LW = 0x34,
    /// LD: rd = mem[rs1 + imm][59:0]
    LD = 0x35,

    // ========== Store (0x38-0x3B) - matches zkir_spec::Opcode ==========
    /// SB: mem[rs1 + imm][7:0] = rs2[7:0]
    SB = 0x38,
    /// SH: mem[rs1 + imm][15:0] = rs2[15:0]
    SH = 0x39,
    /// SW: mem[rs1 + imm][31:0] = rs2[31:0]
    SW = 0x3A,
    /// SD: mem[rs1 + imm][59:0] = rs2[59:0]
    SD = 0x3B,

    // ========== Branch (0x40-0x45) - matches zkir_spec::Opcode ==========
    /// BEQ: if (rs1 == rs2) PC += offset
    BEQ = 0x40,
    /// BNE: if (rs1 != rs2) PC += offset
    BNE = 0x41,
    /// BLT: if (rs1 < rs2) PC += offset (signed)
    BLT = 0x42,
    /// BGE: if (rs1 >= rs2) PC += offset (signed)
    BGE = 0x43,
    /// BLTU: if (rs1 < rs2) PC += offset (unsigned)
    BLTU = 0x44,
    /// BGEU: if (rs1 >= rs2) PC += offset (unsigned)
    BGEU = 0x45,

    // ========== Jump (0x48-0x49) - matches zkir_spec::Opcode ==========
    /// JAL: rd = PC + 4; PC += offset
    JAL = 0x48,
    /// JALR: rd = PC + 4; PC = (rs1 + imm) & ~1
    JALR = 0x49,

    // ========== System (0x50-0x51) - matches zkir_spec::Opcode ==========
    /// ECALL: System call
    ECALL = 0x50,
    /// EBREAK: Breakpoint
    EBREAK = 0x51,

    // ========== Pseudo-ops (0xF0+) - compiler-internal, expanded during emission ==========
    // These are NOT part of zkir-spec and are expanded before final bytecode emission
    /// Pseudo: MOV rd, rs -> expands to ADDI rd, rs, 0
    MOV = 0xF0,
    /// Pseudo: LI rd, imm -> load immediate
    LI = 0xF1,
    /// Pseudo: NOP -> expands to ADDI zero, zero, 0
    NOP = 0xF2,
    /// Pseudo: RET -> expands to JALR zero, ra, 0
    RET = 0xF3,
    /// Pseudo: CALL label -> expands to JAL ra, offset
    CALL = 0xF4,
    /// Pseudo: RCHK rd -> range check (ZK-specific)
    RCHK = 0xF5,
    /// Pseudo: PHI rd -> PHI node (SSA merge point)
    PHI = 0xF6,
    /// Pseudo: CALLR rs -> indirect call through register
    CALLR = 0xF7,
    /// Pseudo: NOT rd, rs -> expands to XORI rd, rs, -1
    NOT = 0xF8,
}

impl Opcode {
    /// Is this an arithmetic operation?
    pub fn is_arithmetic(self) -> bool {
        matches!(self, Opcode::ADD | Opcode::SUB | Opcode::MUL | Opcode::MULH |
                      Opcode::DIV | Opcode::DIVU | Opcode::REM | Opcode::REMU | Opcode::ADDI)
    }

    /// Is this a logical operation?
    pub fn is_logical(self) -> bool {
        matches!(self, Opcode::AND | Opcode::OR | Opcode::XOR |
                      Opcode::ANDI | Opcode::ORI | Opcode::XORI | Opcode::NOT)
    }

    /// Is this a shift operation?
    pub fn is_shift(self) -> bool {
        matches!(self, Opcode::SLL | Opcode::SRL | Opcode::SRA |
                      Opcode::SLLI | Opcode::SRLI | Opcode::SRAI)
    }

    /// Is this a compare operation?
    pub fn is_compare(self) -> bool {
        matches!(self, Opcode::SLTU | Opcode::SGEU | Opcode::SLT |
                      Opcode::SGE | Opcode::SEQ | Opcode::SNE)
    }

    /// Is this a conditional move operation?
    pub fn is_cmov(self) -> bool {
        matches!(self, Opcode::CMOV | Opcode::CMOVZ | Opcode::CMOVNZ)
    }

    /// Is this a branch instruction?
    pub fn is_branch(self) -> bool {
        matches!(self, Opcode::BEQ | Opcode::BNE | Opcode::BLT |
                      Opcode::BGE | Opcode::BLTU | Opcode::BGEU)
    }

    /// Is this a jump instruction?
    pub fn is_jump(self) -> bool {
        matches!(self, Opcode::JAL | Opcode::JALR)
    }

    /// Is this a system instruction?
    pub fn is_system(self) -> bool {
        matches!(self, Opcode::ECALL | Opcode::EBREAK)
    }

    /// Is this a terminator instruction?
    pub fn is_terminator(self) -> bool {
        self.is_branch() || matches!(self, Opcode::JAL | Opcode::JALR |
                                          Opcode::RET | Opcode::EBREAK)
    }

    /// Is this a memory operation?
    pub fn is_memory(self) -> bool {
        self.is_load() || self.is_store()
    }

    /// Is this a load operation?
    pub fn is_load(self) -> bool {
        matches!(self, Opcode::LB | Opcode::LBU | Opcode::LH | Opcode::LHU |
                      Opcode::LW | Opcode::LD)
    }

    /// Is this a store operation?
    pub fn is_store(self) -> bool {
        matches!(self, Opcode::SB | Opcode::SH | Opcode::SW | Opcode::SD)
    }

    /// Is this a pseudo-op that will be expanded?
    pub fn is_pseudo(self) -> bool {
        matches!(self, Opcode::MOV | Opcode::LI | Opcode::NOP | Opcode::NOT |
                      Opcode::RET | Opcode::CALL | Opcode::CALLR | Opcode::RCHK | Opcode::PHI)
    }

    /// Is this a PHI node?
    pub fn is_phi(self) -> bool {
        matches!(self, Opcode::PHI)
    }

    /// Check if this opcode uses an immediate value.
    pub fn uses_immediate(self) -> bool {
        matches!(self, Opcode::ADDI | Opcode::ANDI | Opcode::ORI | Opcode::XORI |
                      Opcode::SLLI | Opcode::SRLI | Opcode::SRAI |
                      Opcode::LB | Opcode::LBU | Opcode::LH | Opcode::LHU |
                      Opcode::LW | Opcode::LD |
                      Opcode::SB | Opcode::SH | Opcode::SW | Opcode::SD |
                      Opcode::JALR)
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            // Arithmetic
            Opcode::ADD => "add",
            Opcode::SUB => "sub",
            Opcode::MUL => "mul",
            Opcode::MULH => "mulh",
            Opcode::DIVU => "divu",
            Opcode::REMU => "remu",
            Opcode::DIV => "div",
            Opcode::REM => "rem",
            Opcode::ADDI => "addi",
            // Logical
            Opcode::AND => "and",
            Opcode::OR => "or",
            Opcode::XOR => "xor",
            Opcode::ANDI => "andi",
            Opcode::ORI => "ori",
            Opcode::XORI => "xori",
            // Shift
            Opcode::SLL => "sll",
            Opcode::SRL => "srl",
            Opcode::SRA => "sra",
            Opcode::SLLI => "slli",
            Opcode::SRLI => "srli",
            Opcode::SRAI => "srai",
            // Compare
            Opcode::SLTU => "sltu",
            Opcode::SGEU => "sgeu",
            Opcode::SLT => "slt",
            Opcode::SGE => "sge",
            Opcode::SEQ => "seq",
            Opcode::SNE => "sne",
            // Conditional move
            Opcode::CMOV => "cmov",
            Opcode::CMOVZ => "cmovz",
            Opcode::CMOVNZ => "cmovnz",
            // Load
            Opcode::LB => "lb",
            Opcode::LBU => "lbu",
            Opcode::LH => "lh",
            Opcode::LHU => "lhu",
            Opcode::LW => "lw",
            Opcode::LD => "ld",
            // Store
            Opcode::SB => "sb",
            Opcode::SH => "sh",
            Opcode::SW => "sw",
            Opcode::SD => "sd",
            // Branch
            Opcode::BEQ => "beq",
            Opcode::BNE => "bne",
            Opcode::BLT => "blt",
            Opcode::BGE => "bge",
            Opcode::BLTU => "bltu",
            Opcode::BGEU => "bgeu",
            // Jump
            Opcode::JAL => "jal",
            Opcode::JALR => "jalr",
            // System
            Opcode::ECALL => "ecall",
            Opcode::EBREAK => "ebreak",
            // Pseudo-ops
            Opcode::MOV => "mov",
            Opcode::LI => "li",
            Opcode::NOP => "nop",
            Opcode::RET => "ret",
            Opcode::CALL => "call",
            Opcode::CALLR => "callr",
            Opcode::RCHK => "rchk",
            Opcode::PHI => "phi",
            Opcode::NOT => "not",
        };
        write!(f, "{}", name)
    }
}

/// An operand for a machine instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// Virtual register (before register allocation)
    VReg(VReg),
    /// Physical register
    Reg(Register),
    /// Immediate value
    Imm(i64),
    /// Label (block name or function name)
    Label(String),
    /// Memory operand: base register + offset
    Mem { base: VReg, offset: i32 },
    /// Memory operand with physical register
    MemReg { base: Register, offset: i32 },
    /// Global variable address reference
    /// Will be resolved to an absolute address during linking/emission
    GlobalAddr(String),
    /// Memory operand with global base + offset
    GlobalMem { name: String, offset: i32 },
}

impl Operand {
    /// Is this a virtual register?
    pub fn is_vreg(&self) -> bool {
        matches!(self, Operand::VReg(_))
    }

    /// Get as virtual register if applicable.
    pub fn as_vreg(&self) -> Option<VReg> {
        match self {
            Operand::VReg(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as physical register if applicable.
    pub fn as_reg(&self) -> Option<Register> {
        match self {
            Operand::Reg(r) => Some(*r),
            _ => None,
        }
    }

    /// Get as immediate if applicable.
    pub fn as_imm(&self) -> Option<i64> {
        match self {
            Operand::Imm(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as global address name if applicable.
    pub fn as_global(&self) -> Option<&str> {
        match self {
            Operand::GlobalAddr(name) => Some(name),
            Operand::GlobalMem { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Check if this operand references a global.
    pub fn is_global(&self) -> bool {
        matches!(self, Operand::GlobalAddr(_) | Operand::GlobalMem { .. })
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::VReg(v) => write!(f, "{}", v),
            Operand::Reg(r) => write!(f, "{}", r),
            Operand::Imm(i) => write!(f, "{}", i),
            Operand::Label(l) => write!(f, "{}", l),
            Operand::Mem { base, offset } => write!(f, "{}({})", offset, base),
            Operand::MemReg { base, offset } => write!(f, "{}({})", offset, base),
            Operand::GlobalAddr(name) => write!(f, "%{}", name),
            Operand::GlobalMem { name, offset } => write!(f, "{}(%{})", offset, name),
        }
    }
}

/// A machine instruction.
#[derive(Debug, Clone)]
pub struct MachineInst {
    /// Opcode
    pub opcode: Opcode,
    /// Destination operand (if any)
    pub dst: Option<Operand>,
    /// Source operands
    pub srcs: Vec<Operand>,
    /// Bounds of the result (for ZK optimization)
    pub result_bounds: Option<ValueBounds>,
    /// Comment for debugging
    pub comment: Option<String>,
    /// Source location (for debugging)
    pub source_loc: Option<SourceLoc>,
}

impl MachineInst {
    /// Create a new instruction.
    pub fn new(opcode: Opcode) -> Self {
        Self {
            opcode,
            dst: None,
            srcs: Vec::new(),
            result_bounds: None,
            comment: None,
            source_loc: None,
        }
    }

    /// Set the destination.
    pub fn dst(mut self, dst: Operand) -> Self {
        self.dst = Some(dst);
        self
    }

    /// Add a source operand.
    pub fn src(mut self, src: Operand) -> Self {
        self.srcs.push(src);
        self
    }

    /// Set result bounds.
    pub fn bounds(mut self, bounds: ValueBounds) -> Self {
        self.result_bounds = Some(bounds);
        self
    }

    /// Add a comment.
    pub fn comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Set source location for debugging.
    pub fn at(mut self, loc: SourceLoc) -> Self {
        self.source_loc = Some(loc);
        self
    }

    /// Set source location from file and line.
    pub fn at_line(mut self, file: impl Into<String>, line: u32) -> Self {
        self.source_loc = Some(SourceLoc::new(file, line, 0));
        self
    }

    /// Create ADD rd, rs1, rs2
    pub fn add(rd: VReg, rs1: VReg, rs2: VReg) -> Self {
        Self::new(Opcode::ADD)
            .dst(Operand::VReg(rd))
            .src(Operand::VReg(rs1))
            .src(Operand::VReg(rs2))
    }

    /// Create ADDI rd, rs1, imm
    pub fn addi(rd: VReg, rs1: VReg, imm: i64) -> Self {
        Self::new(Opcode::ADDI)
            .dst(Operand::VReg(rd))
            .src(Operand::VReg(rs1))
            .src(Operand::Imm(imm))
    }

    /// Create SUB rd, rs1, rs2
    pub fn sub(rd: VReg, rs1: VReg, rs2: VReg) -> Self {
        Self::new(Opcode::SUB)
            .dst(Operand::VReg(rd))
            .src(Operand::VReg(rs1))
            .src(Operand::VReg(rs2))
    }

    /// Create MUL rd, rs1, rs2
    pub fn mul(rd: VReg, rs1: VReg, rs2: VReg) -> Self {
        Self::new(Opcode::MUL)
            .dst(Operand::VReg(rd))
            .src(Operand::VReg(rs1))
            .src(Operand::VReg(rs2))
    }

    /// Create MOV rd, rs (pseudo-op)
    pub fn mov(rd: VReg, rs: VReg) -> Self {
        Self::new(Opcode::MOV)
            .dst(Operand::VReg(rd))
            .src(Operand::VReg(rs))
    }

    /// Create LI rd, imm (pseudo-op)
    pub fn li(rd: VReg, imm: i64) -> Self {
        Self::new(Opcode::LI)
            .dst(Operand::VReg(rd))
            .src(Operand::Imm(imm))
    }

    /// Create LW rd, offset(rs)
    pub fn lw(rd: VReg, base: VReg, offset: i32) -> Self {
        Self::new(Opcode::LW)
            .dst(Operand::VReg(rd))
            .src(Operand::Mem { base, offset })
    }

    /// Create SW rs, offset(base)
    pub fn sw(rs: VReg, base: VReg, offset: i32) -> Self {
        Self::new(Opcode::SW)
            .src(Operand::VReg(rs))
            .src(Operand::Mem { base, offset })
    }

    /// Create BEQ rs1, rs2, label
    pub fn beq(rs1: VReg, rs2: VReg, label: impl Into<String>) -> Self {
        Self::new(Opcode::BEQ)
            .src(Operand::VReg(rs1))
            .src(Operand::VReg(rs2))
            .src(Operand::Label(label.into()))
    }

    /// Create BNE rs1, rs2, label
    pub fn bne(rs1: VReg, rs2: VReg, label: impl Into<String>) -> Self {
        Self::new(Opcode::BNE)
            .src(Operand::VReg(rs1))
            .src(Operand::VReg(rs2))
            .src(Operand::Label(label.into()))
    }

    /// Create JAL rd, label
    pub fn jal(rd: VReg, label: impl Into<String>) -> Self {
        Self::new(Opcode::JAL)
            .dst(Operand::VReg(rd))
            .src(Operand::Label(label.into()))
    }

    /// Create RET (pseudo-op)
    pub fn ret() -> Self {
        Self::new(Opcode::RET)
    }

    /// Create CALLR rd (indirect call through register, pseudo-op)
    /// The target address is in the source register.
    pub fn callr(target: VReg) -> Self {
        Self::new(Opcode::CALLR)
            .src(Operand::VReg(target))
    }

    /// Create NOP
    pub fn nop() -> Self {
        Self::new(Opcode::NOP)
    }

    /// Create XOR rd, rs1, rs2
    pub fn xor(rd: VReg, rs1: VReg, rs2: VReg) -> Self {
        Self::new(Opcode::XOR)
            .dst(Operand::VReg(rd))
            .src(Operand::VReg(rs1))
            .src(Operand::VReg(rs2))
    }

    /// Create AND rd, rs1, rs2
    pub fn and(rd: VReg, rs1: VReg, rs2: VReg) -> Self {
        Self::new(Opcode::AND)
            .dst(Operand::VReg(rd))
            .src(Operand::VReg(rs1))
            .src(Operand::VReg(rs2))
    }

    /// Create OR rd, rs1, rs2
    pub fn or(rd: VReg, rs1: VReg, rs2: VReg) -> Self {
        Self::new(Opcode::OR)
            .dst(Operand::VReg(rd))
            .src(Operand::VReg(rs1))
            .src(Operand::VReg(rs2))
    }

    /// Create RCHK rd (range check, pseudo-op)
    pub fn rchk(rd: VReg) -> Self {
        Self::new(Opcode::RCHK)
            .dst(Operand::VReg(rd))
    }

    /// Create PHI rd (phi node, pseudo-op)
    /// Incoming values are added with phi_incoming() method.
    pub fn phi(rd: VReg) -> Self {
        Self::new(Opcode::PHI)
            .dst(Operand::VReg(rd))
    }

    /// Add an incoming value to a PHI node.
    /// The incoming values are stored as pairs: (Label(pred), VReg(value))
    pub fn phi_incoming(mut self, pred_label: impl Into<String>, value: VReg) -> Self {
        self.srcs.push(Operand::Label(pred_label.into()));
        self.srcs.push(Operand::VReg(value));
        self
    }

    /// Get PHI incoming values as (predecessor_label, value_vreg) pairs.
    pub fn phi_incomings(&self) -> Vec<(String, VReg)> {
        if self.opcode != Opcode::PHI {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut i = 0;
        while i + 1 < self.srcs.len() {
            if let (Operand::Label(label), Operand::VReg(vreg)) = (&self.srcs[i], &self.srcs[i + 1]) {
                result.push((label.clone(), *vreg));
            }
            i += 2;
        }
        result
    }

    /// Get the destination virtual register if applicable.
    pub fn def(&self) -> Option<VReg> {
        self.dst.as_ref().and_then(|op| op.as_vreg())
    }

    /// Get all used virtual registers.
    pub fn uses(&self) -> Vec<VReg> {
        let mut uses = Vec::new();
        for src in &self.srcs {
            if let Operand::VReg(v) = src {
                uses.push(*v);
            } else if let Operand::Mem { base, .. } = src {
                uses.push(*base);
            }
        }
        uses
    }

    /// Is this a terminator instruction?
    pub fn is_terminator(&self) -> bool {
        self.opcode.is_terminator()
    }
}

impl fmt::Display for MachineInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.opcode)?;

        if let Some(dst) = &self.dst {
            write!(f, " {}", dst)?;
        }

        for (i, src) in self.srcs.iter().enumerate() {
            if i == 0 && self.dst.is_some() {
                write!(f, ", {}", src)?;
            } else if i == 0 {
                write!(f, " {}", src)?;
            } else {
                write!(f, ", {}", src)?;
            }
        }

        if let Some(comment) = &self.comment {
            write!(f, "  # {}", comment)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_display() {
        let inst = MachineInst::add(VReg(0), VReg(1), VReg(2));
        assert_eq!(format!("{}", inst), "add v0, v1, v2");

        let inst = MachineInst::addi(VReg(0), VReg(1), 42);
        assert_eq!(format!("{}", inst), "addi v0, v1, 42");

        let inst = MachineInst::lw(VReg(0), VReg(1), 8);
        assert_eq!(format!("{}", inst), "lw v0, 8(v1)");
    }

    #[test]
    fn test_def_use() {
        let inst = MachineInst::add(VReg(0), VReg(1), VReg(2));
        assert_eq!(inst.def(), Some(VReg(0)));
        assert_eq!(inst.uses(), vec![VReg(1), VReg(2)]);
    }

    /// Verify that our Opcode enum values match zkir-spec's canonical definitions.
    ///
    /// This test ensures that we stay aligned with zkir-spec and will catch
    /// any drift if the specification changes.
    #[test]
    fn test_opcode_alignment_with_zkir_spec() {
        use zkir_spec::Opcode as SpecOpcode;

        // Arithmetic opcodes (0x00-0x08)
        assert_eq!(Opcode::ADD as u8, SpecOpcode::Add.to_u8(), "ADD opcode mismatch");
        assert_eq!(Opcode::SUB as u8, SpecOpcode::Sub.to_u8(), "SUB opcode mismatch");
        assert_eq!(Opcode::MUL as u8, SpecOpcode::Mul.to_u8(), "MUL opcode mismatch");
        assert_eq!(Opcode::MULH as u8, SpecOpcode::Mulh.to_u8(), "MULH opcode mismatch");
        assert_eq!(Opcode::DIVU as u8, SpecOpcode::Divu.to_u8(), "DIVU opcode mismatch");
        assert_eq!(Opcode::REMU as u8, SpecOpcode::Remu.to_u8(), "REMU opcode mismatch");
        assert_eq!(Opcode::DIV as u8, SpecOpcode::Div.to_u8(), "DIV opcode mismatch");
        assert_eq!(Opcode::REM as u8, SpecOpcode::Rem.to_u8(), "REM opcode mismatch");
        assert_eq!(Opcode::ADDI as u8, SpecOpcode::Addi.to_u8(), "ADDI opcode mismatch");

        // Logical opcodes (0x10-0x15)
        assert_eq!(Opcode::AND as u8, SpecOpcode::And.to_u8(), "AND opcode mismatch");
        assert_eq!(Opcode::OR as u8, SpecOpcode::Or.to_u8(), "OR opcode mismatch");
        assert_eq!(Opcode::XOR as u8, SpecOpcode::Xor.to_u8(), "XOR opcode mismatch");
        assert_eq!(Opcode::ANDI as u8, SpecOpcode::Andi.to_u8(), "ANDI opcode mismatch");
        assert_eq!(Opcode::ORI as u8, SpecOpcode::Ori.to_u8(), "ORI opcode mismatch");
        assert_eq!(Opcode::XORI as u8, SpecOpcode::Xori.to_u8(), "XORI opcode mismatch");

        // Shift opcodes (0x18-0x1D)
        assert_eq!(Opcode::SLL as u8, SpecOpcode::Sll.to_u8(), "SLL opcode mismatch");
        assert_eq!(Opcode::SRL as u8, SpecOpcode::Srl.to_u8(), "SRL opcode mismatch");
        assert_eq!(Opcode::SRA as u8, SpecOpcode::Sra.to_u8(), "SRA opcode mismatch");
        assert_eq!(Opcode::SLLI as u8, SpecOpcode::Slli.to_u8(), "SLLI opcode mismatch");
        assert_eq!(Opcode::SRLI as u8, SpecOpcode::Srli.to_u8(), "SRLI opcode mismatch");
        assert_eq!(Opcode::SRAI as u8, SpecOpcode::Srai.to_u8(), "SRAI opcode mismatch");

        // Compare opcodes (0x20-0x25)
        assert_eq!(Opcode::SLTU as u8, SpecOpcode::Sltu.to_u8(), "SLTU opcode mismatch");
        assert_eq!(Opcode::SGEU as u8, SpecOpcode::Sgeu.to_u8(), "SGEU opcode mismatch");
        assert_eq!(Opcode::SLT as u8, SpecOpcode::Slt.to_u8(), "SLT opcode mismatch");
        assert_eq!(Opcode::SGE as u8, SpecOpcode::Sge.to_u8(), "SGE opcode mismatch");
        assert_eq!(Opcode::SEQ as u8, SpecOpcode::Seq.to_u8(), "SEQ opcode mismatch");
        assert_eq!(Opcode::SNE as u8, SpecOpcode::Sne.to_u8(), "SNE opcode mismatch");

        // Conditional move opcodes (0x26-0x28)
        assert_eq!(Opcode::CMOV as u8, SpecOpcode::Cmov.to_u8(), "CMOV opcode mismatch");
        assert_eq!(Opcode::CMOVZ as u8, SpecOpcode::Cmovz.to_u8(), "CMOVZ opcode mismatch");
        assert_eq!(Opcode::CMOVNZ as u8, SpecOpcode::Cmovnz.to_u8(), "CMOVNZ opcode mismatch");

        // Load opcodes (0x30-0x35)
        assert_eq!(Opcode::LB as u8, SpecOpcode::Lb.to_u8(), "LB opcode mismatch");
        assert_eq!(Opcode::LBU as u8, SpecOpcode::Lbu.to_u8(), "LBU opcode mismatch");
        assert_eq!(Opcode::LH as u8, SpecOpcode::Lh.to_u8(), "LH opcode mismatch");
        assert_eq!(Opcode::LHU as u8, SpecOpcode::Lhu.to_u8(), "LHU opcode mismatch");
        assert_eq!(Opcode::LW as u8, SpecOpcode::Lw.to_u8(), "LW opcode mismatch");
        assert_eq!(Opcode::LD as u8, SpecOpcode::Ld.to_u8(), "LD opcode mismatch");

        // Store opcodes (0x38-0x3B)
        assert_eq!(Opcode::SB as u8, SpecOpcode::Sb.to_u8(), "SB opcode mismatch");
        assert_eq!(Opcode::SH as u8, SpecOpcode::Sh.to_u8(), "SH opcode mismatch");
        assert_eq!(Opcode::SW as u8, SpecOpcode::Sw.to_u8(), "SW opcode mismatch");
        assert_eq!(Opcode::SD as u8, SpecOpcode::Sd.to_u8(), "SD opcode mismatch");

        // Branch opcodes (0x40-0x45)
        assert_eq!(Opcode::BEQ as u8, SpecOpcode::Beq.to_u8(), "BEQ opcode mismatch");
        assert_eq!(Opcode::BNE as u8, SpecOpcode::Bne.to_u8(), "BNE opcode mismatch");
        assert_eq!(Opcode::BLT as u8, SpecOpcode::Blt.to_u8(), "BLT opcode mismatch");
        assert_eq!(Opcode::BGE as u8, SpecOpcode::Bge.to_u8(), "BGE opcode mismatch");
        assert_eq!(Opcode::BLTU as u8, SpecOpcode::Bltu.to_u8(), "BLTU opcode mismatch");
        assert_eq!(Opcode::BGEU as u8, SpecOpcode::Bgeu.to_u8(), "BGEU opcode mismatch");

        // Jump opcodes (0x48-0x49)
        assert_eq!(Opcode::JAL as u8, SpecOpcode::Jal.to_u8(), "JAL opcode mismatch");
        assert_eq!(Opcode::JALR as u8, SpecOpcode::Jalr.to_u8(), "JALR opcode mismatch");

        // Pseudo-ops (0x80+) are zkir-llvm specific and don't need alignment
        // MOV, LI, NOP, RET, CALL, CALLR, RCHK, NOT, PHI
    }

    /// Verify that converting between our Opcode and zkir-spec Opcode works correctly.
    #[test]
    fn test_opcode_roundtrip_conversion() {
        use zkir_spec::Opcode as SpecOpcode;

        // Test that we can convert from our opcode to zkir-spec and back
        let test_opcodes = vec![
            Opcode::ADD, Opcode::SUB, Opcode::MUL, Opcode::AND, Opcode::OR,
            Opcode::XOR, Opcode::SLL, Opcode::BEQ, Opcode::JAL, Opcode::LW, Opcode::SW,
        ];

        for opcode in test_opcodes {
            let opcode_byte = opcode as u8;
            let spec_opcode = SpecOpcode::from_u8(opcode_byte).expect("Failed to convert to spec opcode");
            assert_eq!(spec_opcode.to_u8(), opcode_byte, "Roundtrip conversion failed for {:?}", opcode);
        }
    }
}
