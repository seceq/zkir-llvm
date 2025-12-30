//! LLVM intrinsic handling.
//!
//! Handles special LLVM intrinsics that need custom lowering.

use super::{LoweringContext, LoweringError};
use crate::mir::{MachineInst, Opcode, Operand, VReg};
use anyhow::Result;
use inkwell::values::InstructionValue;

/// Known LLVM intrinsics that we handle specially.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intrinsic {
    /// llvm.memcpy
    Memcpy,
    /// llvm.memmove
    Memmove,
    /// llvm.memset
    Memset,
    /// llvm.ctlz (count leading zeros)
    Ctlz,
    /// llvm.cttz (count trailing zeros)
    Cttz,
    /// llvm.ctpop (population count)
    Ctpop,
    /// llvm.bswap (byte swap)
    Bswap,
    /// llvm.abs
    Abs,
    /// llvm.smin
    Smin,
    /// llvm.smax
    Smax,
    /// llvm.umin
    Umin,
    /// llvm.umax
    Umax,
    /// llvm.sadd.with.overflow (signed add with overflow detection)
    SaddWithOverflow,
    /// llvm.uadd.with.overflow (unsigned add with overflow detection)
    UaddWithOverflow,
    /// llvm.ssub.with.overflow (signed sub with overflow detection)
    SsubWithOverflow,
    /// llvm.usub.with.overflow (unsigned sub with overflow detection)
    UsubWithOverflow,
    /// llvm.smul.with.overflow (signed mul with overflow detection)
    SmulWithOverflow,
    /// llvm.umul.with.overflow (unsigned mul with overflow detection)
    UmulWithOverflow,
    /// llvm.sadd.sat (signed saturating add)
    SaddSat,
    /// llvm.uadd.sat (unsigned saturating add)
    UaddSat,
    /// llvm.ssub.sat (signed saturating sub)
    SsubSat,
    /// llvm.usub.sat (unsigned saturating sub)
    UsubSat,
    /// llvm.expect (branch prediction hint - ignored)
    Expect,
    /// llvm.assume (optimization hint - ignored)
    Assume,
    /// llvm.lifetime.start (ignored)
    LifetimeStart,
    /// llvm.lifetime.end (ignored)
    LifetimeEnd,
    /// llvm.dbg.* (debug info - ignored)
    Debug,
    // Cryptographic intrinsics (ZKIR-specific)
    /// ZKIR SHA-256 hash
    Sha256,
    /// ZKIR Keccak-256 hash
    Keccak256,
    /// ZKIR Poseidon2 hash
    Poseidon2,
    /// ZKIR Blake3 hash
    Blake3,
    /// Unknown intrinsic
    Unknown,
}

/// Cryptographic algorithm types with their semantic widths.
///
/// Per ZKIR_SPEC_V3.4 §6, each crypto algorithm has:
/// - algorithm_bits: The semantic bit width required by the algorithm
/// - min_internal_bits: Minimum internal representation for zero intermediate range checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CryptoType {
    /// SHA-256: 32-bit algorithm, 44-bit minimum internal
    Sha256,
    /// Keccak-256: 64-bit algorithm, 80-bit minimum internal
    Keccak256,
    /// Poseidon2: 31-bit algorithm, 40-bit minimum internal
    Poseidon2,
    /// Blake3: 32-bit algorithm, 44-bit minimum internal
    Blake3,
}

impl CryptoType {
    /// Algorithm bit width (semantic width).
    pub fn algorithm_bits(&self) -> u32 {
        match self {
            CryptoType::Sha256 => 32,
            CryptoType::Keccak256 => 64,
            CryptoType::Poseidon2 => 31,
            CryptoType::Blake3 => 32,
        }
    }

    /// Minimum internal representation for zero intermediate range checks.
    pub fn min_internal_bits(&self) -> u32 {
        match self {
            CryptoType::Sha256 => 44,
            CryptoType::Keccak256 => 80,
            CryptoType::Poseidon2 => 40,
            CryptoType::Blake3 => 44,
        }
    }

    /// Adaptive internal width based on program configuration.
    ///
    /// crypto_internal = max(min_internal_bits, program_bits)
    pub fn adaptive_internal_bits(&self, program_bits: u32) -> u32 {
        self.min_internal_bits().max(program_bits)
    }

    /// Headroom available DURING crypto execution.
    ///
    /// internal_headroom = adaptive_internal - algorithm_bits
    pub fn internal_headroom(&self, program_bits: u32) -> u32 {
        self.adaptive_internal_bits(program_bits).saturating_sub(self.algorithm_bits())
    }

    /// Headroom available AFTER crypto output (post-crypto headroom).
    ///
    /// post_crypto_headroom = program_bits - algorithm_bits
    /// This enables deferred operations on crypto outputs.
    pub fn post_crypto_headroom(&self, program_bits: u32) -> u32 {
        program_bits.saturating_sub(self.algorithm_bits())
    }

    /// Whether range check is needed after crypto output.
    ///
    /// Range check needed ONLY when algorithm_bits > program_bits.
    pub fn needs_range_check_after_output(&self, program_bits: u32) -> bool {
        self.algorithm_bits() > program_bits
    }

    /// Maximum deferred operations after crypto output.
    pub fn max_deferred_ops_after(&self, program_bits: u32) -> u64 {
        let headroom = self.post_crypto_headroom(program_bits);
        if headroom >= 64 {
            u64::MAX
        } else {
            1u64 << headroom
        }
    }

    /// Output word count for this crypto type.
    pub fn output_words(&self) -> usize {
        match self {
            CryptoType::Sha256 => 8,     // 256 bits / 32 bits = 8 words
            CryptoType::Keccak256 => 4,  // 256 bits / 64 bits = 4 words
            CryptoType::Poseidon2 => 12, // 372 bits (12 × 31-bit elements)
            CryptoType::Blake3 => 8,     // 256 bits / 32 bits = 8 words
        }
    }

    /// Syscall number for this crypto type.
    pub fn syscall_number(&self) -> u32 {
        match self {
            CryptoType::Sha256 => 100,
            CryptoType::Keccak256 => 101,
            CryptoType::Poseidon2 => 102,
            CryptoType::Blake3 => 103,
        }
    }
}

/// Result from a crypto operation with tracked bounds.
///
/// Per ZKIR_SPEC_V3.4 §6.8, crypto results carry their semantic width
/// to enable bound-aware optimization and deferred range checking.
#[derive(Debug, Clone)]
pub struct CryptoResult {
    /// Virtual registers containing output words
    pub vregs: Vec<VReg>,
    /// Crypto type for bound tracking
    pub crypto_type: CryptoType,
    /// Bits per output word (algorithm_bits)
    pub bits_per_word: u32,
}

impl CryptoResult {
    /// Create a new crypto result.
    pub fn new(vregs: Vec<VReg>, crypto_type: CryptoType) -> Self {
        Self {
            vregs,
            crypto_type,
            bits_per_word: crypto_type.algorithm_bits(),
        }
    }

    /// Headroom available for deferred operations on this result.
    pub fn headroom(&self, program_bits: u32) -> u32 {
        self.crypto_type.post_crypto_headroom(program_bits)
    }

    /// Whether this result needs a range check to fit in program registers.
    pub fn needs_range_check(&self, program_bits: u32) -> bool {
        self.crypto_type.needs_range_check_after_output(program_bits)
    }

    /// Get the value bounds for this crypto output.
    ///
    /// The bounds are based on the algorithm's semantic width,
    /// NOT the program's data width. This enables:
    /// 1. Skipping range checks when algorithm fits in program
    /// 2. Computing post-crypto headroom for deferred ops
    pub fn value_bounds(&self) -> crate::mir::ValueBounds {
        crate::mir::ValueBounds::from_bits(self.bits_per_word)
    }
}

impl Intrinsic {
    /// Identify an intrinsic from its name.
    pub fn from_name(name: &str) -> Self {
        if name.starts_with("llvm.memcpy") {
            Intrinsic::Memcpy
        } else if name.starts_with("llvm.memmove") {
            Intrinsic::Memmove
        } else if name.starts_with("llvm.memset") {
            Intrinsic::Memset
        } else if name.starts_with("llvm.ctlz") {
            Intrinsic::Ctlz
        } else if name.starts_with("llvm.cttz") {
            Intrinsic::Cttz
        } else if name.starts_with("llvm.ctpop") {
            Intrinsic::Ctpop
        } else if name.starts_with("llvm.bswap") {
            Intrinsic::Bswap
        } else if name.starts_with("llvm.abs") {
            Intrinsic::Abs
        } else if name.starts_with("llvm.smin") {
            Intrinsic::Smin
        } else if name.starts_with("llvm.smax") {
            Intrinsic::Smax
        } else if name.starts_with("llvm.umin") {
            Intrinsic::Umin
        } else if name.starts_with("llvm.umax") {
            Intrinsic::Umax
        } else if name.starts_with("llvm.sadd.with.overflow") {
            Intrinsic::SaddWithOverflow
        } else if name.starts_with("llvm.uadd.with.overflow") {
            Intrinsic::UaddWithOverflow
        } else if name.starts_with("llvm.ssub.with.overflow") {
            Intrinsic::SsubWithOverflow
        } else if name.starts_with("llvm.usub.with.overflow") {
            Intrinsic::UsubWithOverflow
        } else if name.starts_with("llvm.smul.with.overflow") {
            Intrinsic::SmulWithOverflow
        } else if name.starts_with("llvm.umul.with.overflow") {
            Intrinsic::UmulWithOverflow
        } else if name.starts_with("llvm.sadd.sat") {
            Intrinsic::SaddSat
        } else if name.starts_with("llvm.uadd.sat") {
            Intrinsic::UaddSat
        } else if name.starts_with("llvm.ssub.sat") {
            Intrinsic::SsubSat
        } else if name.starts_with("llvm.usub.sat") {
            Intrinsic::UsubSat
        } else if name.starts_with("llvm.expect") {
            Intrinsic::Expect
        } else if name.starts_with("llvm.assume") {
            Intrinsic::Assume
        } else if name.starts_with("llvm.lifetime.start") {
            Intrinsic::LifetimeStart
        } else if name.starts_with("llvm.lifetime.end") {
            Intrinsic::LifetimeEnd
        } else if name.starts_with("llvm.dbg.") {
            Intrinsic::Debug
        // ZKIR cryptographic intrinsics
        } else if name.starts_with("zkir.sha256") || name.starts_with("__zkir_sha256") {
            Intrinsic::Sha256
        } else if name.starts_with("zkir.keccak256") || name.starts_with("__zkir_keccak256") {
            Intrinsic::Keccak256
        } else if name.starts_with("zkir.poseidon2") || name.starts_with("__zkir_poseidon2") {
            Intrinsic::Poseidon2
        } else if name.starts_with("zkir.blake3") || name.starts_with("__zkir_blake3") {
            Intrinsic::Blake3
        } else {
            Intrinsic::Unknown
        }
    }

    /// Convert to crypto type if this is a crypto intrinsic.
    pub fn as_crypto_type(&self) -> Option<CryptoType> {
        match self {
            Intrinsic::Sha256 => Some(CryptoType::Sha256),
            Intrinsic::Keccak256 => Some(CryptoType::Keccak256),
            Intrinsic::Poseidon2 => Some(CryptoType::Poseidon2),
            Intrinsic::Blake3 => Some(CryptoType::Blake3),
            _ => None,
        }
    }
}

/// Lower an intrinsic call.
///
/// Returns the name of the intrinsic for error reporting if needed.
pub fn lower_intrinsic<'a>(
    ctx: &mut LoweringContext<'a>,
    inst: &InstructionValue<'a>,
    intrinsic: Intrinsic,
    intrinsic_name: &str,
) -> Result<()> {
    match intrinsic {
        Intrinsic::Memcpy | Intrinsic::Memmove => lower_memcpy(ctx, inst),
        Intrinsic::Memset => lower_memset(ctx, inst),
        Intrinsic::Abs => lower_abs(ctx, inst),
        Intrinsic::Smin | Intrinsic::Umin => lower_min(ctx, inst, intrinsic == Intrinsic::Smin),
        Intrinsic::Smax | Intrinsic::Umax => lower_max(ctx, inst, intrinsic == Intrinsic::Smax),
        Intrinsic::Ctlz => lower_ctlz(ctx, inst),
        Intrinsic::Cttz => lower_cttz(ctx, inst),
        Intrinsic::Ctpop => lower_ctpop(ctx, inst),
        Intrinsic::Bswap => lower_bswap(ctx, inst),
        // Overflow intrinsics
        Intrinsic::SaddWithOverflow => lower_add_with_overflow(ctx, inst, true),
        Intrinsic::UaddWithOverflow => lower_add_with_overflow(ctx, inst, false),
        Intrinsic::SsubWithOverflow => lower_sub_with_overflow(ctx, inst, true),
        Intrinsic::UsubWithOverflow => lower_sub_with_overflow(ctx, inst, false),
        Intrinsic::SmulWithOverflow => lower_mul_with_overflow(ctx, inst, true),
        Intrinsic::UmulWithOverflow => lower_mul_with_overflow(ctx, inst, false),
        // Saturating arithmetic
        Intrinsic::SaddSat => lower_add_sat(ctx, inst, true),
        Intrinsic::UaddSat => lower_add_sat(ctx, inst, false),
        Intrinsic::SsubSat => lower_sub_sat(ctx, inst, true),
        Intrinsic::UsubSat => lower_sub_sat(ctx, inst, false),
        // Cryptographic intrinsics with bound tracking
        Intrinsic::Sha256 => lower_crypto_hash(ctx, inst, CryptoType::Sha256),
        Intrinsic::Keccak256 => lower_crypto_hash(ctx, inst, CryptoType::Keccak256),
        Intrinsic::Poseidon2 => lower_crypto_hash(ctx, inst, CryptoType::Poseidon2),
        Intrinsic::Blake3 => lower_crypto_hash(ctx, inst, CryptoType::Blake3),
        // Ignored intrinsics
        Intrinsic::Expect | Intrinsic::Assume |
        Intrinsic::LifetimeStart | Intrinsic::LifetimeEnd |
        Intrinsic::Debug => Ok(()),
        // Unknown intrinsic - emit warning with name
        Intrinsic::Unknown => {
            log::warn!("Unknown intrinsic '{}' - treating as no-op. \
                        This may cause incorrect behavior.", intrinsic_name);
            // If the intrinsic has a return value, we need to provide one
            // For now, return zero
            if let Ok((_, dst)) = ctx.map_result(inst) {
                ctx.emit(MachineInst::li(dst, 0)
                    .comment(format!("unknown intrinsic {} placeholder", intrinsic_name)));
            }
            Ok(())
        }
    }
}

/// Lower memcpy/memmove intrinsic.
///
/// For ZK circuits, we must fully expand memcpy at compile time since
/// runtime loops are not supported. For constant sizes, we unroll the copy.
/// For dynamic sizes, we emit an error as ZK circuits need deterministic
/// instruction counts.
fn lower_memcpy<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    // memcpy(dest, src, len)
    let (_, dest_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (_, src_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let len = ctx.get_value_operand(inst, 2)?;

    // For constant lengths, unroll the copy
    if let Some(const_len) = len.into_int_value().get_zero_extended_constant() {
        // Maximum unroll size (prevents code explosion)
        const MAX_UNROLL: u64 = 1024;

        if const_len > MAX_UNROLL {
            log::warn!("memcpy size {} exceeds max unroll {}, truncating", const_len, MAX_UNROLL);
        }

        let copy_len = const_len.min(MAX_UNROLL);

        // Use word-sized copies when possible for efficiency
        let word_size = 4u64; // 32-bit words
        let num_words = copy_len / word_size;
        let remaining_bytes = copy_len % word_size;

        // Copy full words
        for i in 0..num_words {
            let offset = (i * word_size) as i32;
            let tmp = ctx.new_vreg();
            ctx.emit(MachineInst::new(Opcode::LW)
                .dst(Operand::VReg(tmp))
                .src(Operand::Mem { base: src_vreg, offset }));
            ctx.emit(MachineInst::new(Opcode::SW)
                .src(Operand::VReg(tmp))
                .src(Operand::Mem { base: dest_vreg, offset }));
        }

        // Copy remaining bytes
        let base_offset = (num_words * word_size) as i32;
        for i in 0..remaining_bytes {
            let offset = base_offset + i as i32;
            let tmp = ctx.new_vreg();
            ctx.emit(MachineInst::new(Opcode::LBU)
                .dst(Operand::VReg(tmp))
                .src(Operand::Mem { base: src_vreg, offset }));
            ctx.emit(MachineInst::new(Opcode::SB)
                .src(Operand::VReg(tmp))
                .src(Operand::Mem { base: dest_vreg, offset }));
        }

        return Ok(());
    }

    // Dynamic length - not supported for ZK circuits
    // ZK circuits require deterministic instruction counts at compile time
    Err(ctx.err_zk_constraint(
        "memcpy with dynamic (non-constant) length is not supported in ZK circuits. \
         ZK circuits require deterministic instruction counts at compile time. \
         Consider using a constant size or restructuring your code."
    ).into())
}

/// Lower memset intrinsic.
///
/// For ZK circuits, we must fully expand memset at compile time since
/// runtime loops are not supported. For constant sizes, we unroll the set.
/// For dynamic sizes, we emit an error as ZK circuits need deterministic
/// instruction counts.
fn lower_memset<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    // memset(dest, val, len)
    let (_, dest_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (_, val_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let len = ctx.get_value_operand(inst, 2)?;

    // For constant lengths, unroll the memset
    if let Some(const_len) = len.into_int_value().get_zero_extended_constant() {
        // Maximum unroll size (prevents code explosion)
        const MAX_UNROLL: u64 = 1024;

        if const_len > MAX_UNROLL {
            log::warn!("memset size {} exceeds max unroll {}, truncating", const_len, MAX_UNROLL);
        }

        let set_len = const_len.min(MAX_UNROLL);

        // For efficiency, replicate the byte value to create a word pattern
        // val_word = val | (val << 8) | (val << 16) | (val << 24)
        let word_size = 4u64; // 32-bit words
        let num_words = set_len / word_size;
        let remaining_bytes = set_len % word_size;

        // Build the word pattern if we have words to write
        let word_val = if num_words > 0 {
            let word_vreg = ctx.new_vreg();
            let tmp1 = ctx.new_vreg();
            let tmp2 = ctx.new_vreg();
            let shifted1 = ctx.new_vreg();
            let shifted2 = ctx.new_vreg();

            // Mask value to 8 bits
            let mask_ff = ctx.new_vreg();
            ctx.emit(MachineInst::li(mask_ff, 0xFF));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(tmp1))
                .src(Operand::VReg(val_vreg))
                .src(Operand::VReg(mask_ff)));

            // shifted1 = val << 8
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(shifted1))
                .src(Operand::VReg(tmp1))
                .src(Operand::Imm(8)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(tmp2))
                .src(Operand::VReg(tmp1))
                .src(Operand::VReg(shifted1)));

            // shifted2 = (val | val<<8) << 16
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(shifted2))
                .src(Operand::VReg(tmp2))
                .src(Operand::Imm(16)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(word_vreg))
                .src(Operand::VReg(tmp2))
                .src(Operand::VReg(shifted2)));

            Some(word_vreg)
        } else {
            None
        };

        // Write full words
        if let Some(word_vreg) = word_val {
            for i in 0..num_words {
                let offset = (i * word_size) as i32;
                ctx.emit(MachineInst::new(Opcode::SW)
                    .src(Operand::VReg(word_vreg))
                    .src(Operand::Mem { base: dest_vreg, offset }));
            }
        }

        // Write remaining bytes
        let base_offset = (num_words * word_size) as i32;
        for i in 0..remaining_bytes {
            let offset = base_offset + i as i32;
            ctx.emit(MachineInst::new(Opcode::SB)
                .src(Operand::VReg(val_vreg))
                .src(Operand::Mem { base: dest_vreg, offset }));
        }

        return Ok(());
    }

    // Dynamic length - not supported for ZK circuits
    // ZK circuits require deterministic instruction counts at compile time
    Err(ctx.err_zk_constraint(
        "memset with dynamic (non-constant) length is not supported in ZK circuits. \
         ZK circuits require deterministic instruction counts at compile time. \
         Consider using a constant size or restructuring your code."
    ).into())
}

/// Lower abs intrinsic.
fn lower_abs<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let (_, src) = ctx.get_operand_vreg(inst, 0)?;
    let (_, dst) = ctx.map_result(inst)?;

    // abs(x) = x < 0 ? -x : x
    // Implemented as: abs = (x ^ (x >> 31)) - (x >> 31) for 32-bit

    let shift = ctx.new_vreg();
    let sign = ctx.new_vreg();
    let xored = ctx.new_vreg();

    // shift = x >> 31 (arithmetic shift to get sign extension)
    ctx.emit(MachineInst::li(shift, 31));
    ctx.emit(MachineInst::new(Opcode::SRA)
        .dst(Operand::VReg(sign))
        .src(Operand::VReg(src))
        .src(Operand::VReg(shift)));

    // xored = x ^ sign
    ctx.emit(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(xored))
        .src(Operand::VReg(src))
        .src(Operand::VReg(sign)));

    // dst = xored - sign
    ctx.emit(MachineInst::sub(dst, xored, sign));

    Ok(())
}

/// Lower min intrinsic.
fn lower_min<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>, signed: bool) -> Result<()> {
    let (_, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (_, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (_, dst) = ctx.map_result(inst)?;

    // min(a, b) = a < b ? a : b
    let cmp_result = ctx.new_vreg();
    let cmp_op = if signed { Opcode::SLT } else { Opcode::SLTU };

    ctx.emit(MachineInst::new(cmp_op)
        .dst(Operand::VReg(cmp_result))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    // CMOV dst, cmp_result, lhs, rhs
    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(cmp_result))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    Ok(())
}

/// Lower max intrinsic.
fn lower_max<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>, signed: bool) -> Result<()> {
    let (_, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (_, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (_, dst) = ctx.map_result(inst)?;

    // max(a, b) = a > b ? a : b = b < a ? a : b
    let cmp_result = ctx.new_vreg();
    let cmp_op = if signed { Opcode::SLT } else { Opcode::SLTU };

    // Compare rhs < lhs (swapped order)
    ctx.emit(MachineInst::new(cmp_op)
        .dst(Operand::VReg(cmp_result))
        .src(Operand::VReg(rhs_vreg))
        .src(Operand::VReg(lhs_vreg)));

    // CMOV dst, cmp_result, lhs, rhs
    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(cmp_result))
        .src(Operand::VReg(lhs_vreg))
        .src(Operand::VReg(rhs_vreg)));

    Ok(())
}

/// Lower ctlz (count leading zeros) intrinsic.
///
/// Uses a binary search approach for efficiency. For 32-bit values,
/// this generates ~30 instructions instead of ~200+ for the bit-by-bit approach.
///
/// Algorithm:
/// 1. If upper 16 bits are zero → add 16 to result, shift left by 16
/// 2. If upper 8 bits are zero → add 8 to result, shift left by 8
/// 3. Continue halving...
fn lower_ctlz<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let operand = ctx.get_value_operand(inst, 0)?;
    let src = ctx.get_vreg(&operand);
    let (_, dst) = ctx.map_result(inst)?;

    let bits = operand.get_type().into_int_type().get_bit_width();

    let result = ctx.new_vreg();
    let tmp = ctx.new_vreg();
    let zero = ctx.new_vreg();

    ctx.emit(MachineInst::li(result, 0));
    ctx.emit(MachineInst::li(zero, 0));
    ctx.emit(MachineInst::mov(tmp, src));

    // Binary search steps for different bit widths
    let steps: &[(u32, i64)] = match bits {
        32 => &[
            (16, 0xFFFF0000u32 as i64),
            (8, 0xFF000000u32 as i64),
            (4, 0xF0000000u32 as i64),
            (2, 0xC0000000u32 as i64),
            (1, 0x80000000u32 as i64),
        ],
        64 => &[
            (32, 0xFFFFFFFF00000000u64 as i64),
            (16, 0xFFFF000000000000u64 as i64),
            (8, 0xFF00000000000000u64 as i64),
            (4, 0xF000000000000000u64 as i64),
            (2, 0xC000000000000000u64 as i64),
            (1, 0x8000000000000000u64 as i64),
        ],
        16 => &[
            (8, 0xFF00u32 as i64),
            (4, 0xF000u32 as i64),
            (2, 0xC000u32 as i64),
            (1, 0x8000u32 as i64),
        ],
        8 => &[
            (4, 0xF0i64),
            (2, 0xC0i64),
            (1, 0x80i64),
        ],
        _ => &[], // Unsupported bit width
    };

    for &(shift_amt, mask_val) in steps {
        let mask = ctx.new_vreg();
        let masked = ctx.new_vreg();
        let is_zero = ctx.new_vreg();
        let add_val = ctx.new_vreg();
        let new_result = ctx.new_vreg();
        let shifted = ctx.new_vreg();
        let shift_reg = ctx.new_vreg();
        let new_tmp = ctx.new_vreg();

        // Check if upper bits are zero
        ctx.emit(MachineInst::li(mask, mask_val));
        ctx.emit(MachineInst::new(Opcode::AND)
            .dst(Operand::VReg(masked))
            .src(Operand::VReg(tmp))
            .src(Operand::VReg(mask)));
        ctx.emit(MachineInst::new(Opcode::SEQ)
            .dst(Operand::VReg(is_zero))
            .src(Operand::VReg(masked))
            .src(Operand::VReg(zero)));

        // If upper bits are zero, add shift_amt to result
        ctx.emit(MachineInst::li(add_val, shift_amt as i64));
        ctx.emit(MachineInst::add(new_result, result, add_val));
        ctx.emit(MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(result))
            .src(Operand::VReg(is_zero))
            .src(Operand::VReg(new_result))
            .src(Operand::VReg(result)));

        // If upper bits are zero, shift tmp left by shift_amt
        ctx.emit(MachineInst::li(shift_reg, shift_amt as i64));
        ctx.emit(MachineInst::new(Opcode::SLL)
            .dst(Operand::VReg(shifted))
            .src(Operand::VReg(tmp))
            .src(Operand::VReg(shift_reg)));
        ctx.emit(MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(new_tmp))
            .src(Operand::VReg(is_zero))
            .src(Operand::VReg(shifted))
            .src(Operand::VReg(tmp)));
        ctx.emit(MachineInst::mov(tmp, new_tmp));
    }

    // Handle all-zeros case: if original value was 0, result should be bits
    let orig_is_zero = ctx.new_vreg();
    let full_count = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::SEQ)
        .dst(Operand::VReg(orig_is_zero))
        .src(Operand::VReg(src))
        .src(Operand::VReg(zero)));
    ctx.emit(MachineInst::li(full_count, bits as i64));
    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(orig_is_zero))
        .src(Operand::VReg(full_count))
        .src(Operand::VReg(result)));

    Ok(())
}

/// Lower cttz (count trailing zeros) intrinsic.
///
/// Uses a binary search approach for efficiency. For 32-bit values,
/// this generates ~30 instructions instead of ~200+ for the bit-by-bit approach.
///
/// Algorithm:
/// 1. If lower 16 bits are zero → add 16 to result, shift right by 16
/// 2. If lower 8 bits are zero → add 8 to result, shift right by 8
/// 3. Continue halving...
fn lower_cttz<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let operand = ctx.get_value_operand(inst, 0)?;
    let src = ctx.get_vreg(&operand);
    let (_, dst) = ctx.map_result(inst)?;

    let bits = operand.get_type().into_int_type().get_bit_width();

    let result = ctx.new_vreg();
    let tmp = ctx.new_vreg();
    let zero = ctx.new_vreg();

    ctx.emit(MachineInst::li(result, 0));
    ctx.emit(MachineInst::li(zero, 0));
    ctx.emit(MachineInst::mov(tmp, src));

    // Binary search steps for different bit widths
    // For cttz, we check lower bits and shift right
    let steps: &[(u32, i64)] = match bits {
        32 => &[
            (16, 0x0000FFFFi64),
            (8, 0x000000FFi64),
            (4, 0x0000000Fi64),
            (2, 0x00000003i64),
            (1, 0x00000001i64),
        ],
        64 => &[
            (32, 0x00000000FFFFFFFFi64),
            (16, 0x000000000000FFFFi64),
            (8, 0x00000000000000FFi64),
            (4, 0x000000000000000Fi64),
            (2, 0x0000000000000003i64),
            (1, 0x0000000000000001i64),
        ],
        16 => &[
            (8, 0x00FFi64),
            (4, 0x000Fi64),
            (2, 0x0003i64),
            (1, 0x0001i64),
        ],
        8 => &[
            (4, 0x0Fi64),
            (2, 0x03i64),
            (1, 0x01i64),
        ],
        _ => &[], // Unsupported bit width
    };

    for &(shift_amt, mask_val) in steps {
        let mask = ctx.new_vreg();
        let masked = ctx.new_vreg();
        let is_zero = ctx.new_vreg();
        let add_val = ctx.new_vreg();
        let new_result = ctx.new_vreg();
        let shifted = ctx.new_vreg();
        let shift_reg = ctx.new_vreg();
        let new_tmp = ctx.new_vreg();

        // Check if lower bits are zero
        ctx.emit(MachineInst::li(mask, mask_val));
        ctx.emit(MachineInst::new(Opcode::AND)
            .dst(Operand::VReg(masked))
            .src(Operand::VReg(tmp))
            .src(Operand::VReg(mask)));
        ctx.emit(MachineInst::new(Opcode::SEQ)
            .dst(Operand::VReg(is_zero))
            .src(Operand::VReg(masked))
            .src(Operand::VReg(zero)));

        // If lower bits are zero, add shift_amt to result
        ctx.emit(MachineInst::li(add_val, shift_amt as i64));
        ctx.emit(MachineInst::add(new_result, result, add_val));
        ctx.emit(MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(result))
            .src(Operand::VReg(is_zero))
            .src(Operand::VReg(new_result))
            .src(Operand::VReg(result)));

        // If lower bits are zero, shift tmp right by shift_amt
        ctx.emit(MachineInst::li(shift_reg, shift_amt as i64));
        ctx.emit(MachineInst::new(Opcode::SRL)
            .dst(Operand::VReg(shifted))
            .src(Operand::VReg(tmp))
            .src(Operand::VReg(shift_reg)));
        ctx.emit(MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(new_tmp))
            .src(Operand::VReg(is_zero))
            .src(Operand::VReg(shifted))
            .src(Operand::VReg(tmp)));
        ctx.emit(MachineInst::mov(tmp, new_tmp));
    }

    // Handle all-zeros case: if original value was 0, result should be bits
    let orig_is_zero = ctx.new_vreg();
    let full_count = ctx.new_vreg();
    ctx.emit(MachineInst::new(Opcode::SEQ)
        .dst(Operand::VReg(orig_is_zero))
        .src(Operand::VReg(src))
        .src(Operand::VReg(zero)));
    ctx.emit(MachineInst::li(full_count, bits as i64));
    ctx.emit(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(orig_is_zero))
        .src(Operand::VReg(full_count))
        .src(Operand::VReg(result)));

    Ok(())
}

/// Lower ctpop (population count / number of set bits) intrinsic.
fn lower_ctpop<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let operand = ctx.get_value_operand(inst, 0)?;
    let src = ctx.get_vreg(&operand);
    let (_, dst) = ctx.map_result(inst)?;

    let bits = operand.get_type().into_int_type().get_bit_width();

    let result = ctx.new_vreg();
    let zero = ctx.new_vreg();

    ctx.emit(MachineInst::li(result, 0));
    ctx.emit(MachineInst::li(zero, 0));

    // Count each bit
    if bits <= 32 {
        for i in 0..bits {
            let mask = ctx.new_vreg();
            let masked = ctx.new_vreg();
            let is_set = ctx.new_vreg();
            let one = ctx.new_vreg();
            let incremented = ctx.new_vreg();

            ctx.emit(MachineInst::li(mask, 1i64 << i));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(masked))
                .src(Operand::VReg(src))
                .src(Operand::VReg(mask)));
            ctx.emit(MachineInst::new(Opcode::SNE)
                .dst(Operand::VReg(is_set))
                .src(Operand::VReg(masked))
                .src(Operand::VReg(zero)));

            ctx.emit(MachineInst::li(one, 1));
            ctx.emit(MachineInst::add(incremented, result, one));
            ctx.emit(MachineInst::new(Opcode::CMOV)
                .dst(Operand::VReg(result))
                .src(Operand::VReg(is_set))
                .src(Operand::VReg(incremented))
                .src(Operand::VReg(result))
                .comment(&format!("ctpop bit {}", i)));
        }
    } else if bits == 64 {
        // For 64-bit: count each of the 64 bits
        for i in 0..64u32 {
            let mask = ctx.new_vreg();
            let masked = ctx.new_vreg();
            let is_set = ctx.new_vreg();
            let one = ctx.new_vreg();
            let incremented = ctx.new_vreg();

            ctx.emit(MachineInst::li(mask, 1i64 << i));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(masked))
                .src(Operand::VReg(src))
                .src(Operand::VReg(mask)));
            ctx.emit(MachineInst::new(Opcode::SNE)
                .dst(Operand::VReg(is_set))
                .src(Operand::VReg(masked))
                .src(Operand::VReg(zero)));

            ctx.emit(MachineInst::li(one, 1));
            ctx.emit(MachineInst::add(incremented, result, one));
            ctx.emit(MachineInst::new(Opcode::CMOV)
                .dst(Operand::VReg(result))
                .src(Operand::VReg(is_set))
                .src(Operand::VReg(incremented))
                .src(Operand::VReg(result))
                .comment(&format!("ctpop64 bit {}", i)));
        }
    } else {
        // Unsupported bit width
        ctx.emit(MachineInst::li(result, 0)
            .comment(&format!("ctpop {}bit unsupported", bits)));
    }

    ctx.emit(MachineInst::mov(dst, result));

    Ok(())
}

/// Lower bswap (byte swap) intrinsic.
fn lower_bswap<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    let operand = ctx.get_value_operand(inst, 0)?;
    let src = ctx.get_vreg(&operand);
    let (_, dst) = ctx.map_result(inst)?;

    let bits = operand.get_type().into_int_type().get_bit_width();

    match bits {
        16 => {
            // Swap 2 bytes: (x >> 8) | (x << 8)
            let hi = ctx.new_vreg();
            let lo = ctx.new_vreg();
            let eight = ctx.new_vreg();

            ctx.emit(MachineInst::li(eight, 8));
            ctx.emit(MachineInst::new(Opcode::SRL)
                .dst(Operand::VReg(lo))
                .src(Operand::VReg(src))
                .src(Operand::VReg(eight)));
            ctx.emit(MachineInst::new(Opcode::SLL)
                .dst(Operand::VReg(hi))
                .src(Operand::VReg(src))
                .src(Operand::VReg(eight)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(dst))
                .src(Operand::VReg(hi))
                .src(Operand::VReg(lo)));

            // Mask to 16 bits
            let mask = ctx.new_vreg();
            ctx.emit(MachineInst::li(mask, 0xFFFF));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(dst))
                .src(Operand::VReg(dst))
                .src(Operand::VReg(mask)));
        }
        32 => {
            // Swap 4 bytes
            let b0 = ctx.new_vreg();
            let b1 = ctx.new_vreg();
            let b2 = ctx.new_vreg();
            let b3 = ctx.new_vreg();
            let tmp1 = ctx.new_vreg();
            let tmp2 = ctx.new_vreg();

            // Extract each byte and shift to new position
            let mask_ff = ctx.new_vreg();
            ctx.emit(MachineInst::li(mask_ff, 0xFF));

            // b0 = (src & 0xFF) << 24
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b0))
                .src(Operand::VReg(src))
                .src(Operand::VReg(mask_ff)));
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(b0))
                .src(Operand::VReg(b0))
                .src(Operand::Imm(24)));

            // b1 = ((src >> 8) & 0xFF) << 16
            ctx.emit(MachineInst::new(Opcode::SRLI)
                .dst(Operand::VReg(b1))
                .src(Operand::VReg(src))
                .src(Operand::Imm(8)));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b1))
                .src(Operand::VReg(b1))
                .src(Operand::VReg(mask_ff)));
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(b1))
                .src(Operand::VReg(b1))
                .src(Operand::Imm(16)));

            // b2 = ((src >> 16) & 0xFF) << 8
            ctx.emit(MachineInst::new(Opcode::SRLI)
                .dst(Operand::VReg(b2))
                .src(Operand::VReg(src))
                .src(Operand::Imm(16)));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b2))
                .src(Operand::VReg(b2))
                .src(Operand::VReg(mask_ff)));
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(b2))
                .src(Operand::VReg(b2))
                .src(Operand::Imm(8)));

            // b3 = (src >> 24) & 0xFF
            ctx.emit(MachineInst::new(Opcode::SRLI)
                .dst(Operand::VReg(b3))
                .src(Operand::VReg(src))
                .src(Operand::Imm(24)));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b3))
                .src(Operand::VReg(b3))
                .src(Operand::VReg(mask_ff)));

            // Combine: b0 | b1 | b2 | b3
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(tmp1))
                .src(Operand::VReg(b0))
                .src(Operand::VReg(b1)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(tmp2))
                .src(Operand::VReg(b2))
                .src(Operand::VReg(b3)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(dst))
                .src(Operand::VReg(tmp1))
                .src(Operand::VReg(tmp2)));
        }
        64 => {
            // Swap 8 bytes for 64-bit
            let b0 = ctx.new_vreg();
            let b1 = ctx.new_vreg();
            let b2 = ctx.new_vreg();
            let b3 = ctx.new_vreg();
            let b4 = ctx.new_vreg();
            let b5 = ctx.new_vreg();
            let b6 = ctx.new_vreg();
            let b7 = ctx.new_vreg();
            let tmp1 = ctx.new_vreg();
            let tmp2 = ctx.new_vreg();
            let tmp3 = ctx.new_vreg();
            let tmp4 = ctx.new_vreg();
            let tmp5 = ctx.new_vreg();
            let tmp6 = ctx.new_vreg();

            let mask_ff = ctx.new_vreg();
            ctx.emit(MachineInst::li(mask_ff, 0xFF));

            // b0 = (src & 0xFF) << 56
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b0))
                .src(Operand::VReg(src))
                .src(Operand::VReg(mask_ff)));
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(b0))
                .src(Operand::VReg(b0))
                .src(Operand::Imm(56)));

            // b1 = ((src >> 8) & 0xFF) << 48
            ctx.emit(MachineInst::new(Opcode::SRLI)
                .dst(Operand::VReg(b1))
                .src(Operand::VReg(src))
                .src(Operand::Imm(8)));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b1))
                .src(Operand::VReg(b1))
                .src(Operand::VReg(mask_ff)));
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(b1))
                .src(Operand::VReg(b1))
                .src(Operand::Imm(48)));

            // b2 = ((src >> 16) & 0xFF) << 40
            ctx.emit(MachineInst::new(Opcode::SRLI)
                .dst(Operand::VReg(b2))
                .src(Operand::VReg(src))
                .src(Operand::Imm(16)));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b2))
                .src(Operand::VReg(b2))
                .src(Operand::VReg(mask_ff)));
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(b2))
                .src(Operand::VReg(b2))
                .src(Operand::Imm(40)));

            // b3 = ((src >> 24) & 0xFF) << 32
            ctx.emit(MachineInst::new(Opcode::SRLI)
                .dst(Operand::VReg(b3))
                .src(Operand::VReg(src))
                .src(Operand::Imm(24)));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b3))
                .src(Operand::VReg(b3))
                .src(Operand::VReg(mask_ff)));
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(b3))
                .src(Operand::VReg(b3))
                .src(Operand::Imm(32)));

            // b4 = ((src >> 32) & 0xFF) << 24
            ctx.emit(MachineInst::new(Opcode::SRLI)
                .dst(Operand::VReg(b4))
                .src(Operand::VReg(src))
                .src(Operand::Imm(32)));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b4))
                .src(Operand::VReg(b4))
                .src(Operand::VReg(mask_ff)));
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(b4))
                .src(Operand::VReg(b4))
                .src(Operand::Imm(24)));

            // b5 = ((src >> 40) & 0xFF) << 16
            ctx.emit(MachineInst::new(Opcode::SRLI)
                .dst(Operand::VReg(b5))
                .src(Operand::VReg(src))
                .src(Operand::Imm(40)));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b5))
                .src(Operand::VReg(b5))
                .src(Operand::VReg(mask_ff)));
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(b5))
                .src(Operand::VReg(b5))
                .src(Operand::Imm(16)));

            // b6 = ((src >> 48) & 0xFF) << 8
            ctx.emit(MachineInst::new(Opcode::SRLI)
                .dst(Operand::VReg(b6))
                .src(Operand::VReg(src))
                .src(Operand::Imm(48)));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b6))
                .src(Operand::VReg(b6))
                .src(Operand::VReg(mask_ff)));
            ctx.emit(MachineInst::new(Opcode::SLLI)
                .dst(Operand::VReg(b6))
                .src(Operand::VReg(b6))
                .src(Operand::Imm(8)));

            // b7 = (src >> 56) & 0xFF
            ctx.emit(MachineInst::new(Opcode::SRLI)
                .dst(Operand::VReg(b7))
                .src(Operand::VReg(src))
                .src(Operand::Imm(56)));
            ctx.emit(MachineInst::new(Opcode::AND)
                .dst(Operand::VReg(b7))
                .src(Operand::VReg(b7))
                .src(Operand::VReg(mask_ff)));

            // Combine all bytes: b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(tmp1))
                .src(Operand::VReg(b0))
                .src(Operand::VReg(b1)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(tmp2))
                .src(Operand::VReg(b2))
                .src(Operand::VReg(b3)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(tmp3))
                .src(Operand::VReg(b4))
                .src(Operand::VReg(b5)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(tmp4))
                .src(Operand::VReg(b6))
                .src(Operand::VReg(b7)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(tmp5))
                .src(Operand::VReg(tmp1))
                .src(Operand::VReg(tmp2)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(tmp6))
                .src(Operand::VReg(tmp3))
                .src(Operand::VReg(tmp4)));
            ctx.emit(MachineInst::new(Opcode::OR)
                .dst(Operand::VReg(dst))
                .src(Operand::VReg(tmp5))
                .src(Operand::VReg(tmp6)));
        }
        _ => {
            ctx.emit(MachineInst::mov(dst, src)
                .comment(&format!("bswap {}bit unsupported", bits)));
        }
    }

    Ok(())
}

/// Lower add with overflow intrinsic.
///
/// Returns a struct {result, overflow_flag}.
/// For LLVM this returns {iN, i1} where overflow is 1 if overflow occurred.
fn lower_add_with_overflow<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>, signed: bool) -> Result<()> {
    let (_, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (_, dst) = ctx.map_result(inst)?;

    let bits = rhs.get_type().into_int_type().get_bit_width();

    // Compute the sum
    let sum = ctx.new_vreg();
    ctx.emit(MachineInst::add(sum, lhs_vreg, rhs_vreg));

    // Compute overflow flag
    let overflow = ctx.new_vreg();
    let zero = ctx.new_vreg();
    ctx.emit(MachineInst::li(zero, 0));

    if signed {
        // Signed overflow: (lhs > 0 && rhs > 0 && sum < 0) || (lhs < 0 && rhs < 0 && sum > 0)
        // Simplified: (lhs ^ sum) & (rhs ^ sum) has MSB set when overflow
        let xor1 = ctx.new_vreg();
        let xor2 = ctx.new_vreg();
        let anded = ctx.new_vreg();
        let sign_bit = ctx.new_vreg();

        ctx.emit(MachineInst::xor(xor1, lhs_vreg, sum));
        ctx.emit(MachineInst::xor(xor2, rhs_vreg, sum));
        ctx.emit(MachineInst::and(anded, xor1, xor2));

        // Extract sign bit
        let shift_amt = ctx.new_vreg();
        ctx.emit(MachineInst::li(shift_amt, (bits - 1) as i64));
        ctx.emit(MachineInst::new(Opcode::SRL)
            .dst(Operand::VReg(sign_bit))
            .src(Operand::VReg(anded))
            .src(Operand::VReg(shift_amt)));

        // Mask to 1 bit
        let one = ctx.new_vreg();
        ctx.emit(MachineInst::li(one, 1));
        ctx.emit(MachineInst::and(overflow, sign_bit, one));
    } else {
        // Unsigned overflow: sum < lhs (wrapping occurred)
        ctx.emit(MachineInst::new(Opcode::SLTU)
            .dst(Operand::VReg(overflow))
            .src(Operand::VReg(sum))
            .src(Operand::VReg(lhs_vreg)));
    }

    // Result struct is (sum, overflow) - for now just return sum
    // The caller will extract the struct fields
    ctx.emit(MachineInst::mov(dst, sum)
        .comment(if signed { "sadd.with.overflow result" } else { "uadd.with.overflow result" }));

    Ok(())
}

/// Lower sub with overflow intrinsic.
fn lower_sub_with_overflow<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>, signed: bool) -> Result<()> {
    let (_, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (_, dst) = ctx.map_result(inst)?;

    let bits = rhs.get_type().into_int_type().get_bit_width();

    // Compute the difference
    let diff = ctx.new_vreg();
    ctx.emit(MachineInst::sub(diff, lhs_vreg, rhs_vreg));

    // Compute overflow flag
    let overflow = ctx.new_vreg();

    if signed {
        // Signed overflow: (lhs >= 0 && rhs < 0 && diff < 0) || (lhs < 0 && rhs >= 0 && diff >= 0)
        // Simplified: (lhs ^ rhs) & (lhs ^ diff) has MSB set when overflow
        let xor1 = ctx.new_vreg();
        let xor2 = ctx.new_vreg();
        let anded = ctx.new_vreg();
        let sign_bit = ctx.new_vreg();

        ctx.emit(MachineInst::xor(xor1, lhs_vreg, rhs_vreg));
        ctx.emit(MachineInst::xor(xor2, lhs_vreg, diff));
        ctx.emit(MachineInst::and(anded, xor1, xor2));

        // Extract sign bit
        let shift_amt = ctx.new_vreg();
        ctx.emit(MachineInst::li(shift_amt, (bits - 1) as i64));
        ctx.emit(MachineInst::new(Opcode::SRL)
            .dst(Operand::VReg(sign_bit))
            .src(Operand::VReg(anded))
            .src(Operand::VReg(shift_amt)));

        // Mask to 1 bit
        let one = ctx.new_vreg();
        ctx.emit(MachineInst::li(one, 1));
        ctx.emit(MachineInst::and(overflow, sign_bit, one));
    } else {
        // Unsigned underflow: lhs < rhs
        ctx.emit(MachineInst::new(Opcode::SLTU)
            .dst(Operand::VReg(overflow))
            .src(Operand::VReg(lhs_vreg))
            .src(Operand::VReg(rhs_vreg)));
    }

    ctx.emit(MachineInst::mov(dst, diff)
        .comment(if signed { "ssub.with.overflow result" } else { "usub.with.overflow result" }));

    Ok(())
}

/// Lower mul with overflow intrinsic.
fn lower_mul_with_overflow<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>, signed: bool) -> Result<()> {
    let (_, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (_, dst) = ctx.map_result(inst)?;

    let bits = rhs.get_type().into_int_type().get_bit_width();

    // Compute the product (low bits)
    let prod = ctx.new_vreg();
    ctx.emit(MachineInst::mul(prod, lhs_vreg, rhs_vreg));

    // For overflow detection, we need to check if the full-width result
    // differs from the sign/zero-extended low result
    // This is expensive but necessary for correctness
    let overflow = ctx.new_vreg();
    let zero = ctx.new_vreg();
    ctx.emit(MachineInst::li(zero, 0));

    if bits <= 32 {
        // For 32-bit or smaller, we can compute high bits and check
        // Simplified approach: divide result by one operand, compare to other
        // If rhs != 0 and prod / rhs != lhs, overflow occurred
        let is_zero = ctx.new_vreg();
        let quot = ctx.new_vreg();
        let eq = ctx.new_vreg();
        let not_eq = ctx.new_vreg();

        // Check if rhs is zero
        ctx.emit(MachineInst::new(Opcode::SEQ)
            .dst(Operand::VReg(is_zero))
            .src(Operand::VReg(rhs_vreg))
            .src(Operand::VReg(zero)));

        // Compute quot = prod / rhs (assuming rhs != 0)
        ctx.emit(MachineInst::new(Opcode::DIV)
            .dst(Operand::VReg(quot))
            .src(Operand::VReg(prod))
            .src(Operand::VReg(rhs_vreg)));

        // Check if quot == lhs
        ctx.emit(MachineInst::new(Opcode::SEQ)
            .dst(Operand::VReg(eq))
            .src(Operand::VReg(quot))
            .src(Operand::VReg(lhs_vreg)));

        // overflow = !is_zero && !eq
        ctx.emit(MachineInst::new(Opcode::NOT)
            .dst(Operand::VReg(not_eq))
            .src(Operand::VReg(eq)));

        let one = ctx.new_vreg();
        ctx.emit(MachineInst::li(one, 1));
        ctx.emit(MachineInst::and(not_eq, not_eq, one));

        // If rhs was zero, no overflow
        let not_zero = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::NOT)
            .dst(Operand::VReg(not_zero))
            .src(Operand::VReg(is_zero)));
        ctx.emit(MachineInst::and(not_zero, not_zero, one));

        ctx.emit(MachineInst::and(overflow, not_eq, not_zero));
    } else {
        // For larger types, this gets complex - for now assume no overflow
        ctx.emit(MachineInst::li(overflow, 0)
            .comment("mul overflow detection for large types not implemented"));
    }

    ctx.emit(MachineInst::mov(dst, prod)
        .comment(if signed { "smul.with.overflow result" } else { "umul.with.overflow result" }));

    Ok(())
}

/// Lower saturating add intrinsic.
fn lower_add_sat<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>, signed: bool) -> Result<()> {
    let (_, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (_, dst) = ctx.map_result(inst)?;

    let bits = rhs.get_type().into_int_type().get_bit_width();

    // Compute sum
    let sum = ctx.new_vreg();
    ctx.emit(MachineInst::add(sum, lhs_vreg, rhs_vreg));

    if signed {
        // For signed saturation, clamp to [MIN, MAX]
        let max_val = (1i64 << (bits - 1)) - 1;
        let min_val = -(1i64 << (bits - 1));

        // Check for positive overflow: both positive, sum negative
        let xor1 = ctx.new_vreg();
        let xor2 = ctx.new_vreg();
        let anded = ctx.new_vreg();
        let overflow_check = ctx.new_vreg();

        ctx.emit(MachineInst::xor(xor1, lhs_vreg, sum));
        ctx.emit(MachineInst::xor(xor2, rhs_vreg, sum));
        ctx.emit(MachineInst::and(anded, xor1, xor2));

        // Extract sign bit
        let shift_amt = ctx.new_vreg();
        ctx.emit(MachineInst::li(shift_amt, (bits - 1) as i64));
        ctx.emit(MachineInst::new(Opcode::SRA)
            .dst(Operand::VReg(overflow_check))
            .src(Operand::VReg(anded))
            .src(Operand::VReg(shift_amt)));

        // Determine if positive or negative overflow
        let lhs_sign = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::SRA)
            .dst(Operand::VReg(lhs_sign))
            .src(Operand::VReg(lhs_vreg))
            .src(Operand::VReg(shift_amt)));

        // If overflow and lhs positive -> MAX, if overflow and lhs negative -> MIN
        let max_reg = ctx.new_vreg();
        let min_reg = ctx.new_vreg();
        ctx.emit(MachineInst::li(max_reg, max_val));
        ctx.emit(MachineInst::li(min_reg, min_val));

        // sat_val = lhs_sign == 0 ? max : min
        let zero = ctx.new_vreg();
        ctx.emit(MachineInst::li(zero, 0));
        let is_pos = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::SEQ)
            .dst(Operand::VReg(is_pos))
            .src(Operand::VReg(lhs_sign))
            .src(Operand::VReg(zero)));

        let sat_val = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(sat_val))
            .src(Operand::VReg(is_pos))
            .src(Operand::VReg(max_reg))
            .src(Operand::VReg(min_reg)));

        // result = overflow ? sat_val : sum
        let one = ctx.new_vreg();
        ctx.emit(MachineInst::li(one, 1));
        let has_overflow = ctx.new_vreg();
        ctx.emit(MachineInst::and(has_overflow, overflow_check, one));

        ctx.emit(MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(dst))
            .src(Operand::VReg(has_overflow))
            .src(Operand::VReg(sat_val))
            .src(Operand::VReg(sum)));
    } else {
        // For unsigned saturation, clamp to MAX if overflow
        let max_val = (1u64 << bits).wrapping_sub(1) as i64;

        // Check for overflow: sum < lhs
        let overflow = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::SLTU)
            .dst(Operand::VReg(overflow))
            .src(Operand::VReg(sum))
            .src(Operand::VReg(lhs_vreg)));

        let max_reg = ctx.new_vreg();
        ctx.emit(MachineInst::li(max_reg, max_val));

        // result = overflow ? max : sum
        ctx.emit(MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(dst))
            .src(Operand::VReg(overflow))
            .src(Operand::VReg(max_reg))
            .src(Operand::VReg(sum)));
    }

    Ok(())
}

/// Lower saturating sub intrinsic.
fn lower_sub_sat<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>, signed: bool) -> Result<()> {
    let (_, lhs_vreg) = ctx.get_operand_vreg(inst, 0)?;
    let (rhs, rhs_vreg) = ctx.get_operand_vreg(inst, 1)?;
    let (_, dst) = ctx.map_result(inst)?;

    let bits = rhs.get_type().into_int_type().get_bit_width();

    // Compute difference
    let diff = ctx.new_vreg();
    ctx.emit(MachineInst::sub(diff, lhs_vreg, rhs_vreg));

    if signed {
        // For signed saturation, clamp to [MIN, MAX]
        let max_val = (1i64 << (bits - 1)) - 1;
        let min_val = -(1i64 << (bits - 1));

        // Check for overflow: (lhs ^ rhs) & (lhs ^ diff) has MSB set
        let xor1 = ctx.new_vreg();
        let xor2 = ctx.new_vreg();
        let anded = ctx.new_vreg();
        let overflow_check = ctx.new_vreg();

        ctx.emit(MachineInst::xor(xor1, lhs_vreg, rhs_vreg));
        ctx.emit(MachineInst::xor(xor2, lhs_vreg, diff));
        ctx.emit(MachineInst::and(anded, xor1, xor2));

        // Extract sign bit
        let shift_amt = ctx.new_vreg();
        ctx.emit(MachineInst::li(shift_amt, (bits - 1) as i64));
        ctx.emit(MachineInst::new(Opcode::SRA)
            .dst(Operand::VReg(overflow_check))
            .src(Operand::VReg(anded))
            .src(Operand::VReg(shift_amt)));

        // Determine if positive or negative overflow
        let lhs_sign = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::SRA)
            .dst(Operand::VReg(lhs_sign))
            .src(Operand::VReg(lhs_vreg))
            .src(Operand::VReg(shift_amt)));

        // If overflow and lhs positive -> MAX, if overflow and lhs negative -> MIN
        let max_reg = ctx.new_vreg();
        let min_reg = ctx.new_vreg();
        ctx.emit(MachineInst::li(max_reg, max_val));
        ctx.emit(MachineInst::li(min_reg, min_val));

        let zero = ctx.new_vreg();
        ctx.emit(MachineInst::li(zero, 0));
        let is_pos = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::SEQ)
            .dst(Operand::VReg(is_pos))
            .src(Operand::VReg(lhs_sign))
            .src(Operand::VReg(zero)));

        let sat_val = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(sat_val))
            .src(Operand::VReg(is_pos))
            .src(Operand::VReg(max_reg))
            .src(Operand::VReg(min_reg)));

        let one = ctx.new_vreg();
        ctx.emit(MachineInst::li(one, 1));
        let has_overflow = ctx.new_vreg();
        ctx.emit(MachineInst::and(has_overflow, overflow_check, one));

        ctx.emit(MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(dst))
            .src(Operand::VReg(has_overflow))
            .src(Operand::VReg(sat_val))
            .src(Operand::VReg(diff)));
    } else {
        // For unsigned saturation, clamp to 0 if underflow
        // Check for underflow: lhs < rhs
        let underflow = ctx.new_vreg();
        ctx.emit(MachineInst::new(Opcode::SLTU)
            .dst(Operand::VReg(underflow))
            .src(Operand::VReg(lhs_vreg))
            .src(Operand::VReg(rhs_vreg)));

        let zero = ctx.new_vreg();
        ctx.emit(MachineInst::li(zero, 0));

        // result = underflow ? 0 : diff
        ctx.emit(MachineInst::new(Opcode::CMOV)
            .dst(Operand::VReg(dst))
            .src(Operand::VReg(underflow))
            .src(Operand::VReg(zero))
            .src(Operand::VReg(diff)));
    }

    Ok(())
}

/// Lower a cryptographic hash intrinsic with bound tracking.
///
/// Per ZKIR_SPEC_V3.4 §6, crypto intrinsics:
/// 1. Use adaptive internal representation: max(min_internal, program_bits)
/// 2. Execute with zero intermediate range checks (sufficient headroom)
/// 3. Output with tracked bounds based on algorithm_bits
/// 4. Skip range check if algorithm_bits <= program_bits
///
/// This enables bound-aware optimization after crypto operations.
fn lower_crypto_hash<'a>(
    ctx: &mut LoweringContext<'a>,
    inst: &InstructionValue<'a>,
    crypto_type: CryptoType,
) -> Result<()> {
    // Get program bits for adaptive representation
    let program_bits = ctx.config.data_bits();

    // Compute adaptive internal representation
    let internal_bits = crypto_type.adaptive_internal_bits(program_bits);
    let internal_headroom = crypto_type.internal_headroom(program_bits);
    let post_crypto_headroom = crypto_type.post_crypto_headroom(program_bits);

    log::debug!(
        "Crypto {}: algorithm={}bit, internal={}bit (headroom={}), post-crypto headroom={}",
        match crypto_type {
            CryptoType::Sha256 => "SHA-256",
            CryptoType::Keccak256 => "Keccak-256",
            CryptoType::Poseidon2 => "Poseidon2",
            CryptoType::Blake3 => "Blake3",
        },
        crypto_type.algorithm_bits(),
        internal_bits,
        internal_headroom,
        post_crypto_headroom
    );

    // Get input buffer pointer (operand 0) and length (operand 1)
    let (_, input_ptr) = ctx.get_operand_vreg(inst, 0)?;
    let (_, input_len) = ctx.get_operand_vreg(inst, 1)?;

    // Get output buffer pointer (operand 2) for hash output
    let (_, output_ptr) = ctx.get_operand_vreg(inst, 2)?;

    // Syscall number for this crypto operation
    let syscall_num = crypto_type.syscall_number();

    // Set up syscall arguments:
    // a0 = syscall number
    // a1 = input pointer
    // a2 = input length
    // a3 = output pointer
    use crate::target::registers::Register;

    let syscall_reg = ctx.new_vreg();
    ctx.emit(MachineInst::li(syscall_reg, syscall_num as i64)
        .comment(format!("crypto syscall {}", syscall_num)));

    // Move arguments to ABI registers (zkir-spec v3.4: a0=R4, a1=R5, a2=R6, a3=R7)
    ctx.emit(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R4)) // a0
        .src(Operand::VReg(syscall_reg)));
    ctx.emit(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R5)) // a1
        .src(Operand::VReg(input_ptr)));
    ctx.emit(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R6)) // a2
        .src(Operand::VReg(input_len)));
    ctx.emit(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R7)) // a3
        .src(Operand::VReg(output_ptr)));

    // Emit ECALL for crypto syscall
    ctx.emit(MachineInst::new(Opcode::ECALL)
        .comment(format!("{} hash", match crypto_type {
            CryptoType::Sha256 => "SHA-256",
            CryptoType::Keccak256 => "Keccak-256",
            CryptoType::Poseidon2 => "Poseidon2",
            CryptoType::Blake3 => "Blake3",
        })));

    // The syscall writes output to the output buffer.
    // Now we need to set bounds on the output values.
    //
    // Per ZKIR_SPEC_V3.4 §6.6:
    // - Range check needed ONLY when algorithm_bits > program_bits
    // - If algorithm fits, outputs are already bounded (no check needed)

    let needs_range_check = crypto_type.needs_range_check_after_output(program_bits);
    let output_words = crypto_type.output_words();
    let algorithm_bits = crypto_type.algorithm_bits();

    if needs_range_check {
        // Algorithm output doesn't fit in program registers
        // Need to emit range checks when reading output values
        log::debug!(
            "Crypto output needs range check: {}bit algorithm > {}bit program",
            algorithm_bits, program_bits
        );

        // Emit range check instructions for each output word
        // This is handled when the caller reads from output_ptr
        // For now, we annotate the instruction
        ctx.emit(MachineInst::nop()
            .comment(format!("crypto output needs range check ({}bit > {}bit)",
                algorithm_bits, program_bits)));
    } else {
        // Algorithm output fits in program registers - NO range check needed!
        // This is a significant optimization per ZKIR_SPEC_V3.4 §6.4
        log::debug!(
            "Crypto output NO range check: {}bit algorithm <= {}bit program, {} deferred ops available",
            algorithm_bits, program_bits, crypto_type.max_deferred_ops_after(program_bits)
        );
    }

    // Create CryptoResult for bound tracking (used by optimization passes)
    let mut output_vregs = Vec::with_capacity(output_words);
    let word_bits = match crypto_type {
        CryptoType::Keccak256 => 64,
        _ => 32,
    };

    // Load output words with proper bounds annotation
    for i in 0..output_words {
        let word_vreg = ctx.new_vreg();
        let offset = (i * (word_bits / 8)) as i32;

        // Load based on word size
        let load_opcode = if word_bits == 64 { Opcode::LD } else { Opcode::LW };
        ctx.emit(MachineInst::new(load_opcode)
            .dst(Operand::VReg(word_vreg))
            .src(Operand::Mem { base: output_ptr, offset })
            .comment(format!("crypto output[{}] ({}bit bounded)", i, algorithm_bits)));

        // Set bounds on this vreg based on algorithm width, not program width
        // This is the key insight from ZKIR_SPEC_V3.4 §6.4
        let bounds = crate::mir::ValueBounds::from_bits(algorithm_bits);
        ctx.set_vreg_bounds(word_vreg, bounds);

        output_vregs.push(word_vreg);

        // If range check is needed, emit it now
        if needs_range_check {
            ctx.emit(MachineInst::rchk(word_vreg)
                .comment(format!("range check crypto output[{}]", i)));
        }
    }

    // Store crypto result metadata for downstream optimization
    // The CryptoResult tracks that these values have known bounds
    let _crypto_result = CryptoResult::new(output_vregs, crypto_type);

    // Map the instruction result (if any) to the first output vreg
    // Many crypto intrinsics return void, but some may return a status
    if let Ok((_, dst)) = ctx.map_result(inst) {
        // Return 0 for success
        ctx.emit(MachineInst::li(dst, 0)
            .comment("crypto success"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intrinsic_detection() {
        assert_eq!(Intrinsic::from_name("llvm.memcpy.p0i8.p0i8.i64"), Intrinsic::Memcpy);
        assert_eq!(Intrinsic::from_name("llvm.memset.p0i8.i64"), Intrinsic::Memset);
        assert_eq!(Intrinsic::from_name("llvm.abs.i32"), Intrinsic::Abs);
        assert_eq!(Intrinsic::from_name("llvm.lifetime.start.p0i8"), Intrinsic::LifetimeStart);
        assert_eq!(Intrinsic::from_name("some.other.function"), Intrinsic::Unknown);
    }

    #[test]
    fn test_overflow_intrinsic_detection() {
        assert_eq!(Intrinsic::from_name("llvm.sadd.with.overflow.i32"), Intrinsic::SaddWithOverflow);
        assert_eq!(Intrinsic::from_name("llvm.uadd.with.overflow.i64"), Intrinsic::UaddWithOverflow);
        assert_eq!(Intrinsic::from_name("llvm.ssub.with.overflow.i32"), Intrinsic::SsubWithOverflow);
        assert_eq!(Intrinsic::from_name("llvm.usub.with.overflow.i16"), Intrinsic::UsubWithOverflow);
        assert_eq!(Intrinsic::from_name("llvm.smul.with.overflow.i32"), Intrinsic::SmulWithOverflow);
        assert_eq!(Intrinsic::from_name("llvm.umul.with.overflow.i64"), Intrinsic::UmulWithOverflow);
    }

    #[test]
    fn test_saturating_intrinsic_detection() {
        assert_eq!(Intrinsic::from_name("llvm.sadd.sat.i32"), Intrinsic::SaddSat);
        assert_eq!(Intrinsic::from_name("llvm.uadd.sat.i64"), Intrinsic::UaddSat);
        assert_eq!(Intrinsic::from_name("llvm.ssub.sat.i32"), Intrinsic::SsubSat);
        assert_eq!(Intrinsic::from_name("llvm.usub.sat.i16"), Intrinsic::UsubSat);
    }

    #[test]
    fn test_debug_intrinsic_detection() {
        assert_eq!(Intrinsic::from_name("llvm.dbg.value"), Intrinsic::Debug);
        assert_eq!(Intrinsic::from_name("llvm.dbg.declare"), Intrinsic::Debug);
        assert_eq!(Intrinsic::from_name("llvm.dbg.addr"), Intrinsic::Debug);
    }

    #[test]
    fn test_crypto_intrinsic_detection() {
        // ZKIR crypto intrinsics
        assert_eq!(Intrinsic::from_name("zkir.sha256"), Intrinsic::Sha256);
        assert_eq!(Intrinsic::from_name("zkir.keccak256"), Intrinsic::Keccak256);
        assert_eq!(Intrinsic::from_name("zkir.poseidon2"), Intrinsic::Poseidon2);
        assert_eq!(Intrinsic::from_name("zkir.blake3"), Intrinsic::Blake3);

        // Alternative naming convention
        assert_eq!(Intrinsic::from_name("__zkir_sha256"), Intrinsic::Sha256);
        assert_eq!(Intrinsic::from_name("__zkir_keccak256"), Intrinsic::Keccak256);
        assert_eq!(Intrinsic::from_name("__zkir_poseidon2"), Intrinsic::Poseidon2);
        assert_eq!(Intrinsic::from_name("__zkir_blake3"), Intrinsic::Blake3);
    }

    #[test]
    fn test_crypto_type_algorithm_bits() {
        assert_eq!(CryptoType::Sha256.algorithm_bits(), 32);
        assert_eq!(CryptoType::Keccak256.algorithm_bits(), 64);
        assert_eq!(CryptoType::Poseidon2.algorithm_bits(), 31);
        assert_eq!(CryptoType::Blake3.algorithm_bits(), 32);
    }

    #[test]
    fn test_crypto_type_min_internal_bits() {
        assert_eq!(CryptoType::Sha256.min_internal_bits(), 44);
        assert_eq!(CryptoType::Keccak256.min_internal_bits(), 80);
        assert_eq!(CryptoType::Poseidon2.min_internal_bits(), 40);
        assert_eq!(CryptoType::Blake3.min_internal_bits(), 44);
    }

    #[test]
    fn test_crypto_type_adaptive_internal_bits() {
        // When program_bits < min_internal, use min_internal
        assert_eq!(CryptoType::Sha256.adaptive_internal_bits(40), 44);
        assert_eq!(CryptoType::Keccak256.adaptive_internal_bits(40), 80);

        // When program_bits >= min_internal, use program_bits
        assert_eq!(CryptoType::Sha256.adaptive_internal_bits(60), 60);
        assert_eq!(CryptoType::Poseidon2.adaptive_internal_bits(60), 60);
        assert_eq!(CryptoType::Keccak256.adaptive_internal_bits(80), 80);
    }

    #[test]
    fn test_crypto_type_internal_headroom() {
        // SHA-256: 40-bit program -> 44-bit internal -> 12 bit headroom
        assert_eq!(CryptoType::Sha256.internal_headroom(40), 12);
        // SHA-256: 60-bit program -> 60-bit internal -> 28 bit headroom
        assert_eq!(CryptoType::Sha256.internal_headroom(60), 28);

        // Keccak-256: 40-bit program -> 80-bit internal -> 16 bit headroom
        assert_eq!(CryptoType::Keccak256.internal_headroom(40), 16);
        // Keccak-256: 80-bit program -> 80-bit internal -> 16 bit headroom
        assert_eq!(CryptoType::Keccak256.internal_headroom(80), 16);
    }

    #[test]
    fn test_crypto_type_post_crypto_headroom() {
        // Post-crypto headroom = program_bits - algorithm_bits
        // SHA-256 (32-bit) in 40-bit program: 8 bit headroom
        assert_eq!(CryptoType::Sha256.post_crypto_headroom(40), 8);
        // SHA-256 in 60-bit program: 28 bit headroom
        assert_eq!(CryptoType::Sha256.post_crypto_headroom(60), 28);

        // Keccak-256 (64-bit) in 40-bit program: 0 bit headroom (saturates)
        assert_eq!(CryptoType::Keccak256.post_crypto_headroom(40), 0);
        // Keccak-256 in 80-bit program: 16 bit headroom
        assert_eq!(CryptoType::Keccak256.post_crypto_headroom(80), 16);
    }

    #[test]
    fn test_crypto_type_needs_range_check() {
        // SHA-256 (32-bit) in 40-bit program: NO range check
        assert!(!CryptoType::Sha256.needs_range_check_after_output(40));
        assert!(!CryptoType::Sha256.needs_range_check_after_output(32));
        // SHA-256 in 30-bit program: NEEDS range check
        assert!(CryptoType::Sha256.needs_range_check_after_output(30));

        // Keccak-256 (64-bit) in 40-bit program: NEEDS range check
        assert!(CryptoType::Keccak256.needs_range_check_after_output(40));
        assert!(CryptoType::Keccak256.needs_range_check_after_output(60));
        // Keccak-256 in 64+ bit program: NO range check
        assert!(!CryptoType::Keccak256.needs_range_check_after_output(64));
        assert!(!CryptoType::Keccak256.needs_range_check_after_output(80));

        // Poseidon2 (31-bit) in 31+ bit program: NO range check
        assert!(!CryptoType::Poseidon2.needs_range_check_after_output(31));
        assert!(!CryptoType::Poseidon2.needs_range_check_after_output(40));
        // Poseidon2 in 30-bit program: NEEDS range check
        assert!(CryptoType::Poseidon2.needs_range_check_after_output(30));
    }

    #[test]
    fn test_crypto_type_max_deferred_ops() {
        // SHA-256 in 40-bit: 8 bit headroom = 256 deferred ops
        assert_eq!(CryptoType::Sha256.max_deferred_ops_after(40), 256);
        // SHA-256 in 60-bit: 28 bit headroom = 268M deferred ops
        assert_eq!(CryptoType::Sha256.max_deferred_ops_after(60), 1 << 28);

        // Keccak-256 in 40-bit: 0 bit headroom = 1 op (no deferring)
        assert_eq!(CryptoType::Keccak256.max_deferred_ops_after(40), 1);
        // Keccak-256 in 80-bit: 16 bit headroom = 65K deferred ops
        assert_eq!(CryptoType::Keccak256.max_deferred_ops_after(80), 1 << 16);
    }

    #[test]
    fn test_crypto_type_output_words() {
        assert_eq!(CryptoType::Sha256.output_words(), 8);
        assert_eq!(CryptoType::Keccak256.output_words(), 4);
        assert_eq!(CryptoType::Poseidon2.output_words(), 12);
        assert_eq!(CryptoType::Blake3.output_words(), 8);
    }

    #[test]
    fn test_crypto_type_syscall_numbers() {
        assert_eq!(CryptoType::Sha256.syscall_number(), 100);
        assert_eq!(CryptoType::Keccak256.syscall_number(), 101);
        assert_eq!(CryptoType::Poseidon2.syscall_number(), 102);
        assert_eq!(CryptoType::Blake3.syscall_number(), 103);
    }

    #[test]
    fn test_intrinsic_to_crypto_type() {
        assert_eq!(Intrinsic::Sha256.as_crypto_type(), Some(CryptoType::Sha256));
        assert_eq!(Intrinsic::Keccak256.as_crypto_type(), Some(CryptoType::Keccak256));
        assert_eq!(Intrinsic::Poseidon2.as_crypto_type(), Some(CryptoType::Poseidon2));
        assert_eq!(Intrinsic::Blake3.as_crypto_type(), Some(CryptoType::Blake3));
        assert_eq!(Intrinsic::Memcpy.as_crypto_type(), None);
        assert_eq!(Intrinsic::Unknown.as_crypto_type(), None);
    }

    #[test]
    fn test_crypto_result_bounds() {
        let result = CryptoResult::new(vec![VReg(0), VReg(1)], CryptoType::Sha256);

        // Bounds should be based on algorithm bits (32), not program bits
        let bounds = result.value_bounds();
        assert_eq!(bounds.bits, 32);
        assert_eq!(bounds.max, (1u128 << 32) - 1);

        // Headroom depends on program bits
        assert_eq!(result.headroom(40), 8);
        assert_eq!(result.headroom(60), 28);

        // Range check depends on program bits
        assert!(!result.needs_range_check(40)); // 32 <= 40
        assert!(!result.needs_range_check(32)); // 32 <= 32
        assert!(result.needs_range_check(30));  // 32 > 30
    }
}
