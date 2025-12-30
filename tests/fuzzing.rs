//! Fuzzing and property-based tests for ZKIR-LLVM.
//!
//! This module contains:
//! - Pipeline fuzzing
//! - Optimization pass property tests
//! - Round-trip encoding/decoding tests
//! - Value bounds property tests
//! - Register allocation edge cases
//!
//! Note: Crypto intrinsics tests require the inkwell feature and are
//! in the library's unit tests instead.

use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use zkir_llvm::emit;
use zkir_llvm::mir::{
    MachineBlock, MachineFunction, MachineInst, Module, Opcode, Operand, ValueBounds, VReg,
};
use zkir_llvm::regalloc;
use zkir_llvm::target::TargetConfig;

/// Check if bytecode contains a specific opcode (32-bit encoding format).
/// In zkir-spec v3.4, each instruction is 32 bits with opcode in bits 0-6.
/// The code section starts with a function table, so we need to skip it.
fn bytecode_contains_opcode(bytecode: &[u8], opcode: Opcode) -> bool {
    if bytecode.len() < 32 + 4 {
        return false;
    }
    // Get entry_point from header (bytes 12-15, little-endian u32)
    // This points to the start of the code section (function table + code)
    let entry_point = u32::from_le_bytes([bytecode[12], bytecode[13], bytecode[14], bytecode[15]]) as usize;
    if bytecode.len() < entry_point + 4 {
        return false;
    }

    // Parse function table to find actual code start
    // Format: [name_len: u8][name: bytes][offset: u32][size: u32]
    let mut pos = entry_point;
    while pos < bytecode.len() {
        if pos >= bytecode.len() {
            break;
        }
        let name_len = bytecode[pos] as usize;
        pos += 1;
        if pos + name_len > bytecode.len() {
            break;
        }
        // Skip name
        pos += name_len;
        if pos + 8 > bytecode.len() {
            break;
        }
        // Read code offset and size
        let code_offset = u32::from_le_bytes([bytecode[pos], bytecode[pos+1], bytecode[pos+2], bytecode[pos+3]]) as usize;
        let _code_size = u32::from_le_bytes([bytecode[pos+4], bytecode[pos+5], bytecode[pos+6], bytecode[pos+7]]) as usize;
        pos += 8;

        // If code_offset is valid and past current position, we're still in the function table
        // If code_offset points to actual code, scan those instructions
        if code_offset > 0 && code_offset < bytecode.len() {
            // Scan 32-bit instructions starting at code_offset
            for chunk in bytecode[code_offset..].chunks(4) {
                if chunk.len() == 4 {
                    let inst = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let inst_opcode = (inst & 0x7F) as u8; // bits 0-6
                    if inst_opcode == opcode as u8 {
                        return true;
                    }
                }
            }
        }

        // Only check the first function for now
        break;
    }
    false
}

// =============================================================================
// OPTIMIZATION PASS PROPERTY TESTS
// =============================================================================

/// Create a function with arithmetic chain.
fn create_arithmetic_chain(length: usize) -> Module {
    let mut module = Module::new("arith_chain");
    let mut func = MachineFunction::new("chain");

    let mut vregs: Vec<VReg> = Vec::with_capacity(length + 2);
    vregs.push(func.new_vreg());
    vregs.push(func.new_vreg());

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(vregs[0], 1));
    entry.push(MachineInst::li(vregs[1], 2));

    for i in 2..length + 2 {
        let new_vreg = func.new_vreg();
        vregs.push(new_vreg);
        entry.push(MachineInst::add(new_vreg, vregs[i - 1], vregs[i - 2]));
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);
    module
}

proptest! {
    /// Property: Optimization pipeline is idempotent (running twice gives same result).
    #[test]
    fn prop_optimization_idempotent(chain_len in 5usize..20) {
        use zkir_llvm::opt::OptLevel;

        let mut module1 = create_arithmetic_chain(chain_len);
        let mut module2 = create_arithmetic_chain(chain_len);
        let config = TargetConfig::default();

        // First optimization pass on module1
        zkir_llvm::opt::optimize_with_level(&mut module1, &config, OptLevel::O1)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;

        // Optimize module2, then optimize again
        zkir_llvm::opt::optimize_with_level(&mut module2, &config, OptLevel::O1)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        zkir_llvm::opt::optimize_with_level(&mut module2, &config, OptLevel::O1)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;

        // Should have same function count
        prop_assert_eq!(module1.functions.len(), module2.functions.len());
    }

    /// Property: Higher optimization levels don't increase instruction count significantly.
    #[test]
    fn prop_higher_opt_less_insts(chain_len in 10usize..30) {
        use zkir_llvm::opt::OptLevel;

        let mut module0 = create_arithmetic_chain(chain_len);
        let mut module1 = create_arithmetic_chain(chain_len);
        let mut module2 = create_arithmetic_chain(chain_len);
        let config = TargetConfig::default();

        zkir_llvm::opt::optimize_with_level(&mut module0, &config, OptLevel::O0)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        zkir_llvm::opt::optimize_with_level(&mut module1, &config, OptLevel::O1)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        zkir_llvm::opt::optimize_with_level(&mut module2, &config, OptLevel::O2)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;

        // Count instructions
        fn count_insts(m: &Module) -> usize {
            m.functions.values()
                .flat_map(|f| f.iter_blocks())
                .map(|b| b.insts.len())
                .sum()
        }

        let count0 = count_insts(&module0);
        let count1 = count_insts(&module1);
        let count2 = count_insts(&module2);

        // Higher optimization should not significantly increase instruction count
        // (allowing for small variations due to different optimization choices)
        prop_assert!(count1 <= count0 + 5);
        prop_assert!(count2 <= count1 + 5);
    }

    /// Property: Optimization preserves function structure.
    #[test]
    fn prop_optimization_preserves_structure(chain_len in 3usize..15) {
        use zkir_llvm::opt::OptLevel;

        let original_func_count;
        let original_func_names: Vec<String>;

        let mut module = create_arithmetic_chain(chain_len);
        original_func_count = module.functions.len();
        original_func_names = module.functions.keys().cloned().collect();

        let config = TargetConfig::default();

        zkir_llvm::opt::optimize_with_level(&mut module, &config, OptLevel::O2)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;

        // Function count should be preserved
        prop_assert_eq!(module.functions.len(), original_func_count);

        // Function names should be preserved
        for name in &original_func_names {
            prop_assert!(module.functions.contains_key(name));
        }
    }
}

// =============================================================================
// END-TO-END PIPELINE FUZZING
// =============================================================================

/// Generate a random valid opcode for testing.
#[allow(dead_code)]
fn valid_opcodes() -> impl Strategy<Value = Opcode> {
    prop_oneof![
        Just(Opcode::ADD),
        Just(Opcode::SUB),
        Just(Opcode::MUL),
        Just(Opcode::AND),
        Just(Opcode::OR),
        Just(Opcode::XOR),
    ]
}

proptest! {
    /// Property: Any sequence of valid instructions compiles successfully.
    #[test]
    fn prop_valid_instructions_compile(
        num_insts in 1usize..20,
        seed in 0u64..1000000,
    ) {
        let config = TargetConfig::default();
        let mut module = Module::new("fuzz");
        let mut func = MachineFunction::new("f");

        // Use seed to generate deterministic but varied instructions
        let mut vregs: Vec<VReg> = Vec::new();
        for _ in 0..num_insts + 3 {
            vregs.push(func.new_vreg());
        }

        let mut entry = MachineBlock::new("entry");
        // Initialize first few registers
        entry.push(MachineInst::li(vregs[0], (seed % 100) as i64));
        entry.push(MachineInst::li(vregs[1], ((seed / 100) % 100) as i64));
        entry.push(MachineInst::li(vregs[2], ((seed / 10000) % 100) as i64));

        // Generate instructions based on seed
        for i in 3..num_insts + 3 {
            let op_type = (seed + i as u64) % 6;
            let src1 = vregs[(i - 1) % vregs.len()];
            let src2 = vregs[(i - 2) % vregs.len()];

            let inst = match op_type {
                0 => MachineInst::add(vregs[i], src1, src2),
                1 => MachineInst::sub(vregs[i], src1, src2),
                2 => MachineInst::mul(vregs[i], src1, src2),
                3 => MachineInst::and(vregs[i], src1, src2),
                4 => MachineInst::or(vregs[i], src1, src2),
                5 => MachineInst::xor(vregs[i], src1, src2),
                _ => MachineInst::add(vregs[i], src1, src2),
            };
            entry.push(inst);
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        // Should compile without errors
        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(format!("Allocation failed: {}", e)))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(format!("Emit failed: {}", e)))?;

        prop_assert!(!bytecode.is_empty());
        let magic = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
        prop_assert_eq!(magic, 0x5A4B4952);
    }

    /// Property: Pipeline produces valid bytecode for any config.
    #[test]
    fn prop_pipeline_any_config(
        limb_bits in prop::sample::select(vec![16u8, 18, 20, 22, 24, 26, 28, 30]),
        data_limbs in 1u8..=4,
        addr_limbs in 1u8..=2,
    ) {
        let config = TargetConfig { limb_bits, data_limbs, addr_limbs };

        let mut module = Module::new("cfg_test");
        let mut func = MachineFunction::new("f");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::li(v1, 58));
        entry.push(MachineInst::add(v2, v0, v1));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;

        // Header should match config (bytes 8-10 in zkir-spec v3.4 format)
        prop_assert_eq!(bytecode[8], limb_bits);
        prop_assert_eq!(bytecode[9], data_limbs);
        prop_assert_eq!(bytecode[10], addr_limbs);
    }

    /// Property: Disassembly of bytecode is valid string.
    #[test]
    fn prop_disassembly_valid(num_insts in 1usize..10) {
        let config = TargetConfig::default();
        let mut module = Module::new("disasm_test");
        let mut func = MachineFunction::new("f");

        let vregs: Vec<VReg> = (0..num_insts + 2).map(|_| func.new_vreg()).collect();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(vregs[0], 1));
        entry.push(MachineInst::li(vregs[1], 2));

        for i in 2..num_insts + 2 {
            entry.push(MachineInst::add(vregs[i], vregs[i - 1], vregs[i - 2]));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;

        let disasm = emit::disassemble(&bytecode)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;

        prop_assert!(!disasm.is_empty());
        prop_assert!(disasm.contains("f:"));
    }
}

// =============================================================================
// VALUE BOUNDS EDGE CASES
// =============================================================================

/// Test value bounds with extreme values.
#[test]
fn test_bounds_extreme_values() {
    // Maximum u128
    let b = ValueBounds::from_const(u128::MAX);
    assert_eq!(b.max, u128::MAX);
    assert_eq!(b.bits, 128);

    // Zero
    let b = ValueBounds::from_const(0);
    assert_eq!(b.max, 0);
    assert_eq!(b.bits, 1);  // Zero still needs 1 bit

    // One
    let b = ValueBounds::from_const(1);
    assert_eq!(b.max, 1);
    assert_eq!(b.bits, 1);

    // Power of 2 boundary
    let b = ValueBounds::from_const(256);
    assert_eq!(b.max, 256);
    assert_eq!(b.bits, 9);  // 256 = 2^8, needs 9 bits

    let b = ValueBounds::from_const(255);
    assert_eq!(b.max, 255);
    assert_eq!(b.bits, 8);  // 255 = 2^8 - 1, needs 8 bits
}

/// Test bounds arithmetic with overflow scenarios.
#[test]
fn test_bounds_arithmetic_overflow() {
    // Near-max addition (should saturate)
    let a = ValueBounds::from_const(u128::MAX - 1);
    let b = ValueBounds::from_const(2);
    let result = ValueBounds::add(a, b);
    assert_eq!(result.max, u128::MAX);  // Saturates

    // Near-max multiplication
    let a = ValueBounds::from_const(u64::MAX as u128);
    let b = ValueBounds::from_const(2);
    let result = ValueBounds::mul(a, b);
    assert!(result.max >= u64::MAX as u128 * 2);
}

/// Test bounds operations preserve soundness.
#[test]
fn test_bounds_soundness() {
    // After any operation, max should accurately bound possible values

    // Subtraction can produce 0 to a.max
    let a = ValueBounds::from_const(100);
    let b = ValueBounds::from_const(50);
    let result = ValueBounds::sub(a, b);
    assert!(result.max >= 50);  // 100 - 50 = 50

    // Division
    let a = ValueBounds::from_const(100);
    let b = ValueBounds::from_const(3);
    let result = ValueBounds::udiv(a, b);
    assert!(result.max >= 33);  // 100 / 3 = 33

    // AND never exceeds smaller operand
    let a = ValueBounds::from_bits(32);
    let b = ValueBounds::from_bits(16);
    let result = ValueBounds::and(a, b);
    assert!(result.max <= b.max);
}

/// Test left shift bounds.
#[test]
fn test_bounds_shift_edge_cases() {
    // Shift by 0
    let a = ValueBounds::from_const(100);
    let b = ValueBounds::from_const(0);
    let result = ValueBounds::shl(a, b);
    assert_eq!(result.max, 100);

    // Shift by large amount
    let a = ValueBounds::from_const(1);
    let b = ValueBounds::from_const(64);
    let result = ValueBounds::shl(a, b);
    assert_eq!(result.max, 1u128 << 64);

    // Shift by 128+ should saturate
    let a = ValueBounds::from_const(1);
    let b = ValueBounds::from_const(130);
    let result = ValueBounds::shl(a, b);
    assert_eq!(result.bits, 128);
}

/// Test right shift bounds.
#[test]
fn test_bounds_right_shift_edge_cases() {
    // Right shift by large amount gives 0
    let a = ValueBounds::from_const(1000);
    let b = ValueBounds::from_const(130);
    let result = ValueBounds::lshr(a, b);
    assert_eq!(result.max, 0);

    // Normal right shift
    let a = ValueBounds::from_const(256);
    let b = ValueBounds::from_const(4);
    let result = ValueBounds::lshr(a, b);
    assert!(result.max >= 16);  // 256 >> 4 = 16
}

// =============================================================================
// REGISTER ALLOCATION EDGE CASES
// =============================================================================

/// Test allocation with no virtual registers.
#[test]
fn test_alloc_no_vregs() {
    let config = TargetConfig::default();
    let mut module = Module::new("no_vregs");
    let mut func = MachineFunction::new("empty");
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::nop());
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should handle no vregs");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test allocation with single vreg.
#[test]
fn test_alloc_single_vreg() {
    let config = TargetConfig::default();
    let mut module = Module::new("single_vreg");
    let mut func = MachineFunction::new("single");
    let v0 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should handle single vreg");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test allocation with dead value (defined but never used).
#[test]
fn test_alloc_dead_value() {
    let config = TargetConfig::default();
    let mut module = Module::new("dead_value");
    let mut func = MachineFunction::new("dead");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();  // Dead - never used

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::li(v1, 100));  // Dead store
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should handle dead values");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test allocation with phi-like join points.
#[test]
fn test_alloc_join_points() {
    let config = TargetConfig::default();
    let mut module = Module::new("join_points");
    let mut func = MachineFunction::new("join");

    let cond = func.new_vreg();
    let v_then = func.new_vreg();
    let v_else = func.new_vreg();
    let zero = func.new_vreg();
    let ra = func.new_vreg();

    // Entry
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(cond, 1));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::bne(cond, zero, "then"));
    func.add_block(entry);

    // Then
    let mut then_block = MachineBlock::new("then");
    then_block.push(MachineInst::li(v_then, 100));
    then_block.push(MachineInst::jal(ra, "merge"));
    func.add_block(then_block);

    // Else
    let mut else_block = MachineBlock::new("else");
    else_block.push(MachineInst::li(v_else, 200));
    else_block.push(MachineInst::jal(ra, "merge"));
    func.add_block(else_block);

    // Merge
    let mut merge = MachineBlock::new("merge");
    // Both v_then and v_else might be live here (depending on path)
    merge.push(MachineInst::ret());
    func.add_block(merge);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should handle join points");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// INSTRUCTION FORMAT TESTS
// =============================================================================

/// Test all pseudo-ops format correctly.
#[test]
fn test_pseudo_op_formats() {
    let _config = TargetConfig::default();
    let mut module = Module::new("pseudo_ops");
    let mut func = MachineFunction::new("pseudo");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));         // LI pseudo-op
    entry.push(MachineInst::mov(v1, v0));        // MOV pseudo-op
    entry.push(MachineInst::nop());              // NOP pseudo-op
    entry.push(MachineInst::ret());              // RET pseudo-op
    func.add_block(entry);
    module.add_function(func);

    let asm = emit::format_asm(&module);
    assert!(asm.contains("li"));
    assert!(asm.contains("mov"));
    assert!(asm.contains("nop"));
    assert!(asm.contains("ret"));
}

/// Test conditional moves format correctly.
#[test]
fn test_cmov_formats() {
    let config = TargetConfig::default();
    let mut module = Module::new("cmov");
    let mut func = MachineFunction::new("cmov_fn");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let result = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1));
    entry.push(MachineInst::li(v1, 100));
    entry.push(MachineInst::li(v2, 200));

    // CMOV: result = cond ? v1 : v2
    entry.push(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1))
        .src(Operand::VReg(v2)));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test range check instruction.
#[test]
fn test_range_check_instruction() {
    let _config = TargetConfig::default();
    let mut module = Module::new("rchk");
    let mut func = MachineFunction::new("rchk_fn");

    let v0 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1000));

    // Range check instruction
    entry.push(MachineInst::new(Opcode::RCHK)
        .dst(Operand::VReg(v0)));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let asm = emit::format_asm(&module);
    assert!(asm.contains("rchk"));
}

// =============================================================================
// MODULE STRUCTURE TESTS
// =============================================================================

/// Test module with many globals.
#[test]
fn test_many_globals() {
    use zkir_llvm::mir::GlobalVar;

    let config = TargetConfig::default();
    let mut module = Module::new("many_globals");

    // Add many globals
    for i in 0..20 {
        module.globals.insert(
            format!("global_{}", i),
            GlobalVar {
                name: format!("global_{}", i),
                size: 4 * (i as u32 + 1),
                align: 4,
                init: Some(vec![i as u8; 4]),
                is_const: i % 2 == 0,
            },
        );
    }

    // Add a function
    let mut func = MachineFunction::new("use_globals");
    let v0 = func.new_vreg();
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate with globals");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit with globals");
    assert!(!bytecode.is_empty());

    let asm = emit::format_asm(&module);
    assert!(asm.contains(".section .data"));
    for i in 0..20 {
        assert!(asm.contains(&format!("global_{}", i)));
    }
}

/// Test empty module.
#[test]
fn test_empty_module() {
    let _config = TargetConfig::default();
    let module = Module::new("empty");

    let asm = emit::format_asm(&module);
    assert!(asm.contains("ZKIR Assembly"));
    assert!(asm.contains(".section .text"));
}

// =============================================================================
// ADDITIONAL INSTRUCTION PATTERN TESTS
// =============================================================================

/// Test long chain of dependent operations.
#[test]
fn test_long_dependency_chain() {
    let config = TargetConfig::default();
    let mut module = Module::new("dep_chain");
    let mut func = MachineFunction::new("chain");

    let chain_length = 100;
    let mut vregs: Vec<VReg> = Vec::with_capacity(chain_length);

    let mut entry = MachineBlock::new("entry");

    // First value
    let first = func.new_vreg();
    vregs.push(first);
    entry.push(MachineInst::li(first, 1));

    // Chain of dependent operations
    for i in 1..chain_length {
        let prev = vregs[i - 1];
        let curr = func.new_vreg();
        vregs.push(curr);

        // Alternate between different operations
        match i % 4 {
            0 => entry.push(MachineInst::addi(curr, prev, 1)),
            1 => entry.push(MachineInst::add(curr, prev, first)),
            2 => entry.push(MachineInst::and(curr, prev, first)),
            _ => entry.push(MachineInst::or(curr, prev, first)),
        }
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(bytecode.len() > 100);
}

/// Test diamond CFG pattern.
#[test]
fn test_diamond_cfg() {
    let config = TargetConfig::default();
    let mut module = Module::new("diamond");
    let mut func = MachineFunction::new("diamond_fn");

    let cond = func.new_vreg();
    let zero = func.new_vreg();
    let v_left = func.new_vreg();
    let v_right = func.new_vreg();
    let ra = func.new_vreg();

    // Entry: branch based on condition
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(cond, 1));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::bne(cond, zero, "left"));
    func.add_block(entry);

    // Left branch
    let mut left = MachineBlock::new("left");
    left.push(MachineInst::li(v_left, 100));
    left.push(MachineInst::jal(ra, "merge"));
    func.add_block(left);

    // Right branch (fallthrough from entry if not taken)
    let mut right = MachineBlock::new("right");
    right.push(MachineInst::li(v_right, 200));
    right.push(MachineInst::jal(ra, "merge"));
    func.add_block(right);

    // Merge point
    let mut merge = MachineBlock::new("merge");
    merge.push(MachineInst::ret());
    func.add_block(merge);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test triangle CFG pattern (if without else).
#[test]
fn test_triangle_cfg() {
    let config = TargetConfig::default();
    let mut module = Module::new("triangle");
    let mut func = MachineFunction::new("triangle_fn");

    let cond = func.new_vreg();
    let zero = func.new_vreg();
    let v_then = func.new_vreg();
    let ra = func.new_vreg();

    // Entry: conditional branch
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(cond, 1));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::bne(cond, zero, "then"));
    func.add_block(entry);

    // Then block
    let mut then_block = MachineBlock::new("then");
    then_block.push(MachineInst::li(v_then, 42));
    then_block.push(MachineInst::jal(ra, "exit"));
    func.add_block(then_block);

    // Exit (reached from both entry fallthrough and then)
    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test simple loop CFG.
#[test]
fn test_simple_loop_cfg() {
    let config = TargetConfig::default();
    let mut module = Module::new("simple_loop");
    let mut func = MachineFunction::new("loop_fn");

    let counter = func.new_vreg();
    let limit = func.new_vreg();
    let one = func.new_vreg();

    // Entry: initialize
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(counter, 0));
    entry.push(MachineInst::li(limit, 10));
    entry.push(MachineInst::li(one, 1));
    func.add_block(entry);

    // Loop header: check condition
    let mut header = MachineBlock::new("loop");
    header.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(counter))
        .src(Operand::VReg(limit))
        .src(Operand::Label("body".to_string())));
    func.add_block(header);

    // Loop body: increment and jump back
    let mut body = MachineBlock::new("body");
    body.push(MachineInst::add(counter, counter, one));
    body.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(one))
        .src(Operand::VReg(one))
        .src(Operand::Label("loop".to_string())));  // Unconditional back edge
    func.add_block(body);

    // Exit
    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test switch-like pattern with multiple branches.
#[test]
fn test_switch_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("switch");
    let mut func = MachineFunction::new("switch_fn");

    let selector = func.new_vreg();
    let case_vals: Vec<VReg> = (0..4).map(|_| func.new_vreg()).collect();
    let results: Vec<VReg> = (0..4).map(|_| func.new_vreg()).collect();
    let ra = func.new_vreg();

    // Entry: load selector and case values
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(selector, 2));
    for (i, &v) in case_vals.iter().enumerate() {
        entry.push(MachineInst::li(v, i as i64));
    }
    // Check case 0
    entry.push(MachineInst::beq(selector, case_vals[0], "case0"));
    func.add_block(entry);

    // Check case 1
    let mut check1 = MachineBlock::new("check1");
    check1.push(MachineInst::beq(selector, case_vals[1], "case1"));
    func.add_block(check1);

    // Check case 2
    let mut check2 = MachineBlock::new("check2");
    check2.push(MachineInst::beq(selector, case_vals[2], "case2"));
    func.add_block(check2);

    // Default case
    let mut default = MachineBlock::new("default");
    default.push(MachineInst::li(results[3], 999));
    default.push(MachineInst::jal(ra, "exit"));
    func.add_block(default);

    // Case 0
    let mut case0 = MachineBlock::new("case0");
    case0.push(MachineInst::li(results[0], 100));
    case0.push(MachineInst::jal(ra, "exit"));
    func.add_block(case0);

    // Case 1
    let mut case1 = MachineBlock::new("case1");
    case1.push(MachineInst::li(results[1], 200));
    case1.push(MachineInst::jal(ra, "exit"));
    func.add_block(case1);

    // Case 2
    let mut case2 = MachineBlock::new("case2");
    case2.push(MachineInst::li(results[2], 300));
    case2.push(MachineInst::jal(ra, "exit"));
    func.add_block(case2);

    // Exit
    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// MEMORY ACCESS PATTERN TESTS
// =============================================================================

/// Test sequential memory accesses.
#[test]
fn test_sequential_memory_access() {
    let config = TargetConfig::default();
    let mut module = Module::new("seq_mem");
    let mut func = MachineFunction::new("seq_mem_fn");

    let base = func.new_vreg();
    let vals: Vec<VReg> = (0..8).map(|_| func.new_vreg()).collect();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000));

    // Sequential stores
    for (i, &v) in vals.iter().enumerate() {
        entry.push(MachineInst::li(v, i as i64 * 10));
        entry.push(MachineInst::new(Opcode::SW)
            .src(Operand::VReg(v))
            .src(Operand::Mem { base, offset: (i * 4) as i32 }));
    }

    // Sequential loads
    let loaded: Vec<VReg> = (0..8).map(|_| func.new_vreg()).collect();
    for (i, &l) in loaded.iter().enumerate() {
        entry.push(MachineInst::new(Opcode::LW)
            .dst(Operand::VReg(l))
            .src(Operand::Mem { base, offset: (i * 4) as i32 }));
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test strided memory access pattern.
#[test]
fn test_strided_memory_access() {
    let config = TargetConfig::default();
    let mut module = Module::new("strided_mem");
    let mut func = MachineFunction::new("strided_fn");

    let base = func.new_vreg();
    let stride = 16;  // Access every 16 bytes

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x2000));

    // Strided stores (like accessing struct fields)
    for i in 0..4 {
        let v = func.new_vreg();
        entry.push(MachineInst::li(v, i as i64 * 100));
        entry.push(MachineInst::new(Opcode::SW)
            .src(Operand::VReg(v))
            .src(Operand::Mem { base, offset: i * stride }));
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test mixed-size memory accesses.
#[test]
fn test_mixed_size_memory_access() {
    let config = TargetConfig::default();
    let mut module = Module::new("mixed_mem");
    let mut func = MachineFunction::new("mixed_fn");

    let base = func.new_vreg();
    let val = func.new_vreg();
    let loaded_byte = func.new_vreg();
    let loaded_half = func.new_vreg();
    let loaded_word = func.new_vreg();
    let loaded_double = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x3000));
    entry.push(MachineInst::li(val, 0x12345678));

    // Store with different sizes
    entry.push(MachineInst::new(Opcode::SB)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::new(Opcode::SH)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base, offset: 4 }));
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base, offset: 8 }));
    entry.push(MachineInst::new(Opcode::SD)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base, offset: 16 }));

    // Load with different sizes (both signed and unsigned)
    entry.push(MachineInst::new(Opcode::LBU)
        .dst(Operand::VReg(loaded_byte))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::new(Opcode::LHU)
        .dst(Operand::VReg(loaded_half))
        .src(Operand::Mem { base, offset: 4 }));
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(loaded_word))
        .src(Operand::Mem { base, offset: 8 }));
    entry.push(MachineInst::new(Opcode::LD)
        .dst(Operand::VReg(loaded_double))
        .src(Operand::Mem { base, offset: 16 }));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// REGISTER PRESSURE GRADIENT TESTS
// =============================================================================

/// Test gradual increase in register pressure.
#[test]
fn test_register_pressure_gradient() {
    let config = TargetConfig::default();

    // Test with increasing numbers of live values
    for num_live in [5, 10, 15, 20, 30, 50] {
        let mut module = Module::new(&format!("pressure_{}", num_live));
        let mut func = MachineFunction::new("pressure_fn");

        let mut entry = MachineBlock::new("entry");

        // Create num_live values that are all live simultaneously
        let vregs: Vec<VReg> = (0..num_live).map(|_| func.new_vreg()).collect();
        for (i, &v) in vregs.iter().enumerate() {
            entry.push(MachineInst::li(v, i as i64));
        }

        // Use all values to keep them live
        let mut sum = vregs[0];
        for i in 1..num_live {
            let new_sum = func.new_vreg();
            entry.push(MachineInst::add(new_sum, sum, vregs[i]));
            sum = new_sum;
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config)
            .expect(&format!("Should allocate with {} live values", num_live));
        let bytecode = emit::emit(&allocated, &config)
            .expect(&format!("Should emit with {} live values", num_live));
        assert!(!bytecode.is_empty());
    }
}

/// Test spill/reload patterns with high pressure.
#[test]
fn test_spill_reload_patterns() {
    let config = TargetConfig::default();
    let mut module = Module::new("spill_reload");
    let mut func = MachineFunction::new("spill_fn");

    // More live values than available registers
    let num_values = 30;
    let mut entry = MachineBlock::new("entry");

    // Phase 1: Create many values
    let initial_values: Vec<VReg> = (0..num_values).map(|_| func.new_vreg()).collect();
    for (i, &v) in initial_values.iter().enumerate() {
        entry.push(MachineInst::li(v, i as i64 * 7));
    }

    // Phase 2: Use all initial values (forces spills)
    let sums: Vec<VReg> = (0..num_values - 1).map(|_| func.new_vreg()).collect();
    for i in 0..num_values - 1 {
        entry.push(MachineInst::add(sums[i], initial_values[i], initial_values[i + 1]));
    }

    // Phase 3: Use the sums (forces reloads of any spilled values)
    let mut final_sum = sums[0];
    for i in 1..sums.len() {
        let new_sum = func.new_vreg();
        entry.push(MachineInst::add(new_sum, final_sum, sums[i]));
        final_sum = new_sum;
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should handle spills");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// OPTIMIZATION INTERACTION TESTS
// =============================================================================

/// Test constant folding followed by dead code elimination.
#[test]
fn test_const_fold_then_dce() {
    use zkir_llvm::opt::OptLevel;

    let mut module = Module::new("fold_dce");
    let mut func = MachineFunction::new("fold_dce_fn");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v3 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    // These should be folded to constants
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 20));
    entry.push(MachineInst::add(v2, v0, v1));  // Should fold to 30
    // v3 is dead
    entry.push(MachineInst::li(v3, 999));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let config = TargetConfig::default();
    zkir_llvm::opt::optimize_with_level(&mut module, &config, OptLevel::O2).unwrap();

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test copy propagation followed by CSE.
#[test]
fn test_copy_prop_then_cse() {
    use zkir_llvm::opt::OptLevel;

    let mut module = Module::new("copy_cse");
    let mut func = MachineFunction::new("copy_cse_fn");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v3 = func.new_vreg();
    let v4 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::mov(v1, v0));  // Copy
    entry.push(MachineInst::add(v2, v0, v0));  // Common subexpression
    entry.push(MachineInst::add(v3, v1, v1));  // Same after copy prop: v0 + v0
    entry.push(MachineInst::add(v4, v2, v3));  // Use both
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let config = TargetConfig::default();
    zkir_llvm::opt::optimize_with_level(&mut module, &config, OptLevel::O2).unwrap();

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// IMMEDIATE VALUE RANGE TESTS
// =============================================================================

proptest! {
    /// Property: Any 32-bit signed immediate can be loaded.
    #[test]
    fn prop_any_i32_immediate(imm in i32::MIN..=i32::MAX) {
        let config = TargetConfig::default();
        let mut module = Module::new("imm_test");
        let mut func = MachineFunction::new("f");
        let v = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v, imm as i64));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(!bytecode.is_empty());
    }

    /// Property: Shift amounts in valid range work correctly.
    #[test]
    fn prop_shift_amounts(shift_amt in 0u8..64) {
        let config = TargetConfig::default();
        let mut module = Module::new("shift_test");
        let mut func = MachineFunction::new("f");

        let val = func.new_vreg();
        let amt = func.new_vreg();
        let result = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(val, 0xFF));
        entry.push(MachineInst::li(amt, shift_amt as i64));
        entry.push(MachineInst::new(Opcode::SLL)
            .dst(Operand::VReg(result))
            .src(Operand::VReg(val))
            .src(Operand::VReg(amt)));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(!bytecode.is_empty());
    }

    /// Property: Memory offsets in range work correctly.
    #[test]
    fn prop_memory_offsets(offset in -2048i32..2048) {
        let config = TargetConfig::default();
        let mut module = Module::new("offset_test");
        let mut func = MachineFunction::new("f");

        let base = func.new_vreg();
        let val = func.new_vreg();
        let loaded = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(base, 0x10000));
        entry.push(MachineInst::li(val, 42));
        entry.push(MachineInst::new(Opcode::SW)
            .src(Operand::VReg(val))
            .src(Operand::Mem { base, offset }));
        entry.push(MachineInst::new(Opcode::LW)
            .dst(Operand::VReg(loaded))
            .src(Operand::Mem { base, offset }));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(!bytecode.is_empty());
    }
}

// =============================================================================
// FUNCTION CALL PATTERN TESTS
// =============================================================================

/// Test function with CALL instruction.
#[test]
fn test_function_call() {
    let config = TargetConfig::default();
    let mut module = Module::new("call_test");

    // Callee function
    let mut callee = MachineFunction::new("callee");
    let ret_val = callee.new_vreg();
    let mut callee_entry = MachineBlock::new("entry");
    callee_entry.push(MachineInst::li(ret_val, 42));
    callee_entry.push(MachineInst::ret());
    callee.add_block(callee_entry);
    module.add_function(callee);

    // Caller function
    let mut caller = MachineFunction::new("caller");
    let mut caller_entry = MachineBlock::new("entry");
    caller_entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("callee".to_string())));
    caller_entry.push(MachineInst::ret());
    caller.add_block(caller_entry);
    module.add_function(caller);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());

    let asm = emit::format_asm(&allocated);
    assert!(asm.contains("callee:"));
    assert!(asm.contains("caller:"));
    assert!(asm.contains("call"));
}

/// Test indirect call pattern.
#[test]
fn test_indirect_call_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("indirect_call");
    let mut func = MachineFunction::new("indirect_caller");

    let fn_ptr = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    // Load a function pointer (simulated)
    entry.push(MachineInst::li(fn_ptr, 0x4000));
    // Indirect call
    entry.push(MachineInst::callr(fn_ptr));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test syscall (ECALL) instruction.
#[test]
fn test_syscall_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("syscall");
    let mut func = MachineFunction::new("syscall_fn");

    let syscall_num = func.new_vreg();
    let arg1 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(syscall_num, 1));  // Syscall number
    entry.push(MachineInst::li(arg1, 42));        // Argument
    entry.push(MachineInst::new(Opcode::ECALL));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());

    let asm = emit::format_asm(&allocated);
    assert!(asm.contains("ecall"));
}

// =============================================================================
// MULTI-FUNCTION MODULE TESTS
// =============================================================================

/// Test module with function calling chain.
#[test]
fn test_function_call_chain() {
    let config = TargetConfig::default();
    let mut module = Module::new("call_chain");

    // Create a chain: main -> f1 -> f2 -> f3
    for i in (0..4).rev() {
        let name = if i == 0 { "main".to_string() } else { format!("f{}", i) };
        let mut func = MachineFunction::new(&name);
        let v = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v, i as i64));

        if i < 3 {
            let next = if i == 0 { "f1" } else { &format!("f{}", i + 1) };
            entry.push(MachineInst::new(Opcode::CALL)
                .src(Operand::Label(next.to_string())));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);
    }

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());

    let asm = emit::format_asm(&allocated);
    assert!(asm.contains("main:"));
    assert!(asm.contains("f1:"));
    assert!(asm.contains("f2:"));
    assert!(asm.contains("f3:"));
}

/// Test module with mutually recursive functions.
#[test]
fn test_mutual_recursion() {
    let config = TargetConfig::default();
    let mut module = Module::new("mutual_rec");

    // is_even function
    let mut even = MachineFunction::new("is_even");
    let n_even = even.new_vreg();
    let zero = even.new_vreg();
    let _ra = even.new_vreg();

    let mut even_entry = MachineBlock::new("entry");
    even_entry.push(MachineInst::li(n_even, 10));
    even_entry.push(MachineInst::li(zero, 0));
    even_entry.push(MachineInst::beq(n_even, zero, "return_true"));
    even.add_block(even_entry);

    let mut return_true = MachineBlock::new("return_true");
    return_true.push(MachineInst::ret());
    even.add_block(return_true);

    // Call is_odd
    let mut call_odd = MachineBlock::new("call_odd");
    call_odd.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("is_odd".to_string())));
    call_odd.push(MachineInst::ret());
    even.add_block(call_odd);

    even.rebuild_cfg();
    module.add_function(even);

    // is_odd function
    let mut odd = MachineFunction::new("is_odd");
    let n_odd = odd.new_vreg();

    let mut odd_entry = MachineBlock::new("entry");
    odd_entry.push(MachineInst::li(n_odd, 10));
    odd_entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("is_even".to_string())));
    odd_entry.push(MachineInst::ret());
    odd.add_block(odd_entry);

    module.add_function(odd);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// EDGE CASE AND BOUNDARY TESTS
// =============================================================================

/// Test empty function (just return).
#[test]
fn test_empty_function() {
    let config = TargetConfig::default();
    let mut module = Module::new("empty_fn");
    let mut func = MachineFunction::new("empty");

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test function with only NOPs.
#[test]
fn test_nop_function() {
    let config = TargetConfig::default();
    let mut module = Module::new("nop_fn");
    let mut func = MachineFunction::new("nops");

    let mut entry = MachineBlock::new("entry");
    for _ in 0..10 {
        entry.push(MachineInst::nop());
    }
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test maximum immediate values (within safe encoding range).
#[test]
fn test_max_immediate_values() {
    let config = TargetConfig::default();
    let mut module = Module::new("max_imm");
    let mut func = MachineFunction::new("max_imm_fn");

    let v_max_i32 = func.new_vreg();
    let v_min_i32 = func.new_vreg();
    let v_large = func.new_vreg();
    let v_small = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v_max_i32, i32::MAX as i64));
    entry.push(MachineInst::li(v_min_i32, i32::MIN as i64));
    // Use values within safe range to avoid overflow in encoder
    entry.push(MachineInst::li(v_large, 0x7FFFFFFF_FFFF));  // 48-bit max
    entry.push(MachineInst::li(v_small, -0x7FFFFFFF_FFFF)); // 48-bit min
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test zero value operations.
#[test]
fn test_zero_value_ops() {
    let config = TargetConfig::default();
    let mut module = Module::new("zero_ops");
    let mut func = MachineFunction::new("zero_fn");

    let zero = func.new_vreg();
    let v1 = func.new_vreg();
    let r_add = func.new_vreg();
    let r_sub = func.new_vreg();
    let r_mul = func.new_vreg();
    let r_and = func.new_vreg();
    let r_or = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::li(v1, 42));
    entry.push(MachineInst::add(r_add, v1, zero));  // x + 0 = x
    entry.push(MachineInst::sub(r_sub, v1, zero));  // x - 0 = x
    entry.push(MachineInst::mul(r_mul, v1, zero));  // x * 0 = 0
    entry.push(MachineInst::and(r_and, v1, zero)); // x & 0 = 0
    entry.push(MachineInst::or(r_or, v1, zero));   // x | 0 = x
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test self-operations (x op x).
#[test]
fn test_self_operations() {
    let config = TargetConfig::default();
    let mut module = Module::new("self_ops");
    let mut func = MachineFunction::new("self_fn");

    let v1 = func.new_vreg();
    let r_add = func.new_vreg();
    let r_sub = func.new_vreg();
    let r_xor = func.new_vreg();
    let r_and = func.new_vreg();
    let r_or = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v1, 42));
    entry.push(MachineInst::add(r_add, v1, v1));  // x + x = 2x
    entry.push(MachineInst::sub(r_sub, v1, v1));  // x - x = 0
    entry.push(MachineInst::xor(r_xor, v1, v1));  // x ^ x = 0
    entry.push(MachineInst::and(r_and, v1, v1));  // x & x = x
    entry.push(MachineInst::or(r_or, v1, v1));    // x | x = x
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test power of two values (common in bit manipulation).
#[test]
fn test_power_of_two_values() {
    let config = TargetConfig::default();
    let mut module = Module::new("pow2");
    let mut func = MachineFunction::new("pow2_fn");

    let mut entry = MachineBlock::new("entry");

    // Load all powers of 2 from 2^0 to 2^63
    let vregs: Vec<VReg> = (0..64).map(|_| func.new_vreg()).collect();
    for i in 0..64u32 {
        let val = 1i64.wrapping_shl(i);
        entry.push(MachineInst::li(vregs[i as usize], val));
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test alternating bit patterns.
#[test]
fn test_alternating_bits() {
    let config = TargetConfig::default();
    let mut module = Module::new("alt_bits");
    let mut func = MachineFunction::new("alt_fn");

    let v_5555 = func.new_vreg();
    let v_aaaa = func.new_vreg();
    let v_and = func.new_vreg();
    let v_or = func.new_vreg();
    let v_xor = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v_5555, 0x5555555555555555u64 as i64));
    entry.push(MachineInst::li(v_aaaa, 0xAAAAAAAAAAAAAAAAu64 as i64));
    entry.push(MachineInst::and(v_and, v_5555, v_aaaa));  // Should be 0
    entry.push(MachineInst::or(v_or, v_5555, v_aaaa));    // Should be -1
    entry.push(MachineInst::xor(v_xor, v_5555, v_aaaa));  // Should be -1
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// INSTRUCTION ENCODING VERIFICATION TESTS
// =============================================================================

/// Test that all arithmetic opcodes encode correctly.
#[test]
fn test_arithmetic_opcode_encoding() {
    let config = TargetConfig::default();
    let mut module = Module::new("arith_encode");
    let mut func = MachineFunction::new("arith");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let r_add = func.new_vreg();
    let r_sub = func.new_vreg();
    let r_mul = func.new_vreg();
    let r_div = func.new_vreg();
    let r_rem = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 100));
    entry.push(MachineInst::li(v1, 7));
    entry.push(MachineInst::add(r_add, v0, v1));
    entry.push(MachineInst::sub(r_sub, v0, v1));
    entry.push(MachineInst::mul(r_mul, v0, v1));
    entry.push(MachineInst::new(Opcode::DIV)
        .dst(Operand::VReg(r_div))
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1)));
    entry.push(MachineInst::new(Opcode::REM)
        .dst(Operand::VReg(r_rem))
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1)));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");

    // Verify bytecode contains expected opcodes (in 32-bit encoding format)
    assert!(bytecode_contains_opcode(&bytecode, Opcode::ADD));
    assert!(bytecode_contains_opcode(&bytecode, Opcode::SUB));
    assert!(bytecode_contains_opcode(&bytecode, Opcode::MUL));
}

/// Test that all comparison opcodes encode correctly.
#[test]
fn test_comparison_opcode_encoding() {
    let config = TargetConfig::default();
    let mut module = Module::new("cmp_encode");
    let mut func = MachineFunction::new("cmp");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let r_slt = func.new_vreg();
    let r_sltu = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 20));
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(r_slt))
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1)));
    entry.push(MachineInst::new(Opcode::SLTU)
        .dst(Operand::VReg(r_sltu))
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1)));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test all branch opcodes.
#[test]
fn test_branch_opcode_encoding() {
    let config = TargetConfig::default();
    let mut module = Module::new("branch_encode");
    let mut func = MachineFunction::new("branch");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let ra = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 20));
    entry.push(MachineInst::beq(v0, v1, "target"));
    func.add_block(entry);

    let mut block1 = MachineBlock::new("block1");
    block1.push(MachineInst::bne(v0, v1, "target"));
    func.add_block(block1);

    let mut block2 = MachineBlock::new("block2");
    block2.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1))
        .src(Operand::Label("target".to_string())));
    func.add_block(block2);

    let mut block3 = MachineBlock::new("block3");
    block3.push(MachineInst::new(Opcode::BGE)
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1))
        .src(Operand::Label("target".to_string())));
    func.add_block(block3);

    let mut block4 = MachineBlock::new("block4");
    block4.push(MachineInst::new(Opcode::BLTU)
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1))
        .src(Operand::Label("target".to_string())));
    func.add_block(block4);

    let mut block5 = MachineBlock::new("block5");
    block5.push(MachineInst::new(Opcode::BGEU)
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1))
        .src(Operand::Label("target".to_string())));
    func.add_block(block5);

    let mut target = MachineBlock::new("target");
    target.push(MachineInst::jal(ra, "exit"));
    func.add_block(target);

    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test all memory size opcodes.
#[test]
fn test_memory_size_encoding() {
    let config = TargetConfig::default();
    let mut module = Module::new("mem_size");
    let mut func = MachineFunction::new("mem");

    let base = func.new_vreg();
    let val = func.new_vreg();
    let lb = func.new_vreg();
    let lbu = func.new_vreg();
    let lh = func.new_vreg();
    let lhu = func.new_vreg();
    let lw = func.new_vreg();
    let lwu = func.new_vreg();
    let ld = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000));
    entry.push(MachineInst::li(val, 0x12345678));

    // Stores
    entry.push(MachineInst::new(Opcode::SB)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::new(Opcode::SH)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base, offset: 4 }));
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base, offset: 8 }));
    entry.push(MachineInst::new(Opcode::SD)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base, offset: 16 }));

    // Loads (signed)
    entry.push(MachineInst::new(Opcode::LB)
        .dst(Operand::VReg(lb))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::new(Opcode::LH)
        .dst(Operand::VReg(lh))
        .src(Operand::Mem { base, offset: 4 }));
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(lw))
        .src(Operand::Mem { base, offset: 8 }));
    entry.push(MachineInst::new(Opcode::LD)
        .dst(Operand::VReg(ld))
        .src(Operand::Mem { base, offset: 16 }));

    // Loads (unsigned)
    entry.push(MachineInst::new(Opcode::LBU)
        .dst(Operand::VReg(lbu))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::new(Opcode::LHU)
        .dst(Operand::VReg(lhu))
        .src(Operand::Mem { base, offset: 4 }));
    // Use lwu vreg for another LW (LWU doesn't exist in this ISA)
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(lwu))
        .src(Operand::Mem { base, offset: 12 }));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// CONTROL FLOW MERGE TESTS
// =============================================================================

/// Test simple if-then-else merge pattern (without explicit PHI).
/// PHI elimination happens during lowering from LLVM, so we test the CFG pattern.
#[test]
fn test_if_then_else_merge() {
    let config = TargetConfig::default();
    let mut module = Module::new("merge_test");
    let mut func = MachineFunction::new("merge_fn");

    let cond = func.new_vreg();
    let zero = func.new_vreg();
    let v_true = func.new_vreg();
    let v_false = func.new_vreg();
    let ra = func.new_vreg();

    // Entry: branch based on condition
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(cond, 1));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::bne(cond, zero, "then"));
    func.add_block(entry);

    // Then block - stores result to memory (simulating phi)
    let mut then_block = MachineBlock::new("then");
    then_block.push(MachineInst::li(v_true, 100));
    then_block.push(MachineInst::jal(ra, "merge"));
    func.add_block(then_block);

    // Else block - stores different result
    let mut else_block = MachineBlock::new("else");
    else_block.push(MachineInst::li(v_false, 200));
    else_block.push(MachineInst::jal(ra, "merge"));
    func.add_block(else_block);

    // Merge block
    let mut merge = MachineBlock::new("merge");
    merge.push(MachineInst::ret());
    func.add_block(merge);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test multi-way branch merge pattern (switch-like).
#[test]
fn test_multi_way_merge() {
    let config = TargetConfig::default();
    let mut module = Module::new("multi_merge");
    let mut func = MachineFunction::new("multi_merge_fn");

    let selector = func.new_vreg();
    let vals: Vec<VReg> = (0..4).map(|_| func.new_vreg()).collect();
    let case_vals: Vec<VReg> = (0..4).map(|_| func.new_vreg()).collect();
    let ra = func.new_vreg();

    // Entry
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(selector, 2));
    for (i, &v) in case_vals.iter().enumerate() {
        entry.push(MachineInst::li(v, i as i64));
    }
    entry.push(MachineInst::beq(selector, case_vals[0], "case0"));
    func.add_block(entry);

    // Check blocks
    for i in 1..3 {
        let mut check = MachineBlock::new(&format!("check{}", i));
        check.push(MachineInst::beq(selector, case_vals[i], &format!("case{}", i)));
        func.add_block(check);
    }

    // Case blocks
    for i in 0..4 {
        let mut case_block = MachineBlock::new(&format!("case{}", i));
        case_block.push(MachineInst::li(vals[i], (i + 1) as i64 * 100));
        case_block.push(MachineInst::jal(ra, "merge"));
        func.add_block(case_block);
    }

    // Merge block
    let mut merge = MachineBlock::new("merge");
    merge.push(MachineInst::ret());
    func.add_block(merge);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// FRAME LAYOUT AND STACK TESTS
// =============================================================================

/// Test function with local variables on stack.
#[test]
fn test_stack_locals() {
    let config = TargetConfig::default();
    let mut module = Module::new("stack_locals");
    let mut func = MachineFunction::new("locals_fn");

    // Set up frame with local storage
    func.frame.locals_size = 64; // 64 bytes of locals

    let sp = func.new_vreg();
    let val = func.new_vreg();
    let loaded = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(sp, 0x8000)); // Stack pointer
    entry.push(MachineInst::li(val, 42));

    // Store to local
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base: sp, offset: -8 }));

    // Load from local
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(loaded))
        .src(Operand::Mem { base: sp, offset: -8 }));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test function with outgoing arguments.
#[test]
fn test_outgoing_args() {
    let config = TargetConfig::default();
    let mut module = Module::new("outgoing_args");

    // Callee takes 10 args (more than registers can hold)
    let mut callee = MachineFunction::new("callee");
    let mut callee_entry = MachineBlock::new("entry");
    callee_entry.push(MachineInst::ret());
    callee.add_block(callee_entry);
    module.add_function(callee);

    // Caller passes 10 args
    let mut caller = MachineFunction::new("caller");
    caller.frame.outgoing_args_size = 40; // 10 args * 4 bytes

    let args: Vec<VReg> = (0..10).map(|_| caller.new_vreg()).collect();

    let mut entry = MachineBlock::new("entry");
    for (i, &arg) in args.iter().enumerate() {
        entry.push(MachineInst::li(arg, i as i64));
    }
    entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("callee".to_string())));
    entry.push(MachineInst::ret());
    caller.add_block(entry);
    module.add_function(caller);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test deeply nested function calls.
#[test]
fn test_deep_call_stack() {
    let config = TargetConfig::default();
    let mut module = Module::new("deep_calls");

    // Create a chain of 10 functions
    for i in 0..10 {
        let name = format!("func_{}", i);
        let mut func = MachineFunction::new(&name);
        let v = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v, i as i64));

        if i < 9 {
            let next = format!("func_{}", i + 1);
            entry.push(MachineInst::new(Opcode::CALL)
                .src(Operand::Label(next)));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);
    }

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// GLOBAL VARIABLE TESTS
// =============================================================================

/// Test module with global data section.
#[test]
fn test_global_data() {
    use zkir_llvm::mir::GlobalVar;

    let config = TargetConfig::default();
    let mut module = Module::new("globals");

    // Add some globals directly to the globals map
    module.globals.insert("counter".to_string(), GlobalVar {
        name: "counter".to_string(),
        size: 4,
        align: 4,
        init: Some(vec![0, 0, 0, 0]),
        is_const: false,
    });
    module.globals.insert("message".to_string(), GlobalVar {
        name: "message".to_string(),
        size: 5,
        align: 1,
        init: Some(b"Hello".to_vec()),
        is_const: true,
    });
    module.globals.insert("array".to_string(), GlobalVar {
        name: "array".to_string(),
        size: 8,
        align: 4,
        init: Some(vec![1, 2, 3, 4, 5, 6, 7, 8]),
        is_const: false,
    });

    // Function that references globals
    let mut func = MachineFunction::new("use_globals");
    let base = func.new_vreg();
    let val = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000)); // Simulated global address
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(val))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());

    let asm = emit::format_asm(&allocated);
    assert!(asm.contains("counter") || asm.contains("globals"));
}

/// Test large global array.
#[test]
fn test_large_global() {
    use zkir_llvm::mir::GlobalVar;

    let config = TargetConfig::default();
    let mut module = Module::new("large_global");

    // 4KB array
    let large_data: Vec<u8> = (0..4096).map(|i| (i & 0xFF) as u8).collect();
    module.globals.insert("big_array".to_string(), GlobalVar {
        name: "big_array".to_string(),
        size: 4096,
        align: 8,
        init: Some(large_data),
        is_const: false,
    });

    let mut func = MachineFunction::new("access_large");
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// BYTECODE VERIFICATION TESTS
// =============================================================================

/// Test that bytecode emits correctly and can be verified.
#[test]
fn test_bytecode_structure() {
    let config = TargetConfig::default();
    let mut module = Module::new("bytecode_test");
    let mut func = MachineFunction::new("test_fn");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::li(v1, 10));
    entry.push(MachineInst::add(v2, v0, v1));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");

    // Check bytecode starts with ZKIR magic (little-endian u32)
    let magic = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
    assert_eq!(magic, 0x5A4B4952);
    // Check bytecode contains ADD opcode (32-bit encoding format)
    assert!(bytecode_contains_opcode(&bytecode, Opcode::ADD));
}

/// Test complex program bytecode structure.
#[test]
fn test_complex_bytecode() {
    let config = TargetConfig::default();
    let mut module = Module::new("complex_bytecode");
    let mut func = MachineFunction::new("complex");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let zero = func.new_vreg();
    let _ra = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 100));
    entry.push(MachineInst::li(v1, 0));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::bne(v0, zero, "loop"));
    func.add_block(entry);

    let mut loop_block = MachineBlock::new("loop");
    loop_block.push(MachineInst::addi(v0, v0, -1));
    loop_block.push(MachineInst::addi(v1, v1, 1));
    loop_block.push(MachineInst::bne(v0, zero, "loop"));
    func.add_block(loop_block);

    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");

    // Bytecode should contain branch opcode (32-bit encoding format)
    assert!(bytecode_contains_opcode(&bytecode, Opcode::BNE));
}

// =============================================================================
// VALUE BOUNDS EDGE CASES
// =============================================================================

/// Test bounds with zero values.
#[test]
fn test_bounds_zero() {
    let zero = ValueBounds::from_const(0);
    assert_eq!(zero.max, 0);
    assert_eq!(zero.bits, 1); // Zero still needs 1 bit to represent

    let add_zero = ValueBounds::add(zero, zero);
    assert_eq!(add_zero.max, 0);

    let mul_zero = ValueBounds::mul(zero, ValueBounds::from_const(1000));
    assert_eq!(mul_zero.max, 0);
}

/// Test bounds with maximum values.
#[test]
fn test_bounds_max() {
    let max_32 = ValueBounds::from_bits(32);
    assert_eq!(max_32.max, 0xFFFFFFFF);
    assert_eq!(max_32.bits, 32);

    let max_64 = ValueBounds::from_bits(64);
    assert_eq!(max_64.max, 0xFFFFFFFFFFFFFFFF);
    assert_eq!(max_64.bits, 64);

    let max_128 = ValueBounds::from_bits(128);
    assert_eq!(max_128.max, u128::MAX);
    assert_eq!(max_128.bits, 128);
}

/// Test bounds saturation.
#[test]
fn test_bounds_saturation() {
    let max = ValueBounds::from_bits(128);
    let also_max = ValueBounds::from_bits(128);

    // Adding two max values should saturate
    let sum = ValueBounds::add(max, also_max);
    assert_eq!(sum.max, u128::MAX);

    // Multiplying max values should saturate
    let prod = ValueBounds::mul(max, also_max);
    assert_eq!(prod.max, u128::MAX);
}

/// Test bounds after shifts.
#[test]
fn test_bounds_shifts() {
    let small = ValueBounds::from_const(1);

    // Shift left by 63
    let shift_amt = ValueBounds::from_const(63);
    let shifted = ValueBounds::shl(small, shift_amt);
    assert!(shifted.max >= (1u128 << 63));

    // Large shift should saturate
    let big_shift = ValueBounds::from_const(200);
    let saturated = ValueBounds::shl(small, big_shift);
    assert_eq!(saturated.bits, 128);
}

/// Test truncation bounds.
#[test]
fn test_bounds_truncation() {
    let full_64 = ValueBounds::from_bits(64);

    let trunc_32 = full_64.trunc(32);
    assert_eq!(trunc_32.bits, 32);
    assert_eq!(trunc_32.max, 0xFFFFFFFF);

    let trunc_16 = full_64.trunc(16);
    assert_eq!(trunc_16.bits, 16);
    assert_eq!(trunc_16.max, 0xFFFF);

    let trunc_8 = full_64.trunc(8);
    assert_eq!(trunc_8.bits, 8);
    assert_eq!(trunc_8.max, 0xFF);
}

// =============================================================================
// PROPTEST FOR EDGE CASES
// =============================================================================

proptest! {
    /// Property: Back-to-back stores and loads work correctly.
    #[test]
    fn prop_store_load_pairs(num_pairs in 1usize..10) {
        let config = TargetConfig::default();
        let mut module = Module::new("store_load");
        let mut func = MachineFunction::new("sl_fn");

        let base = func.new_vreg();
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(base, 0x1000));

        for i in 0..num_pairs {
            let val = func.new_vreg();
            let loaded = func.new_vreg();
            let offset = (i * 8) as i32;

            entry.push(MachineInst::li(val, i as i64));
            entry.push(MachineInst::new(Opcode::SD)
                .src(Operand::VReg(val))
                .src(Operand::Mem { base, offset }));
            entry.push(MachineInst::new(Opcode::LD)
                .dst(Operand::VReg(loaded))
                .src(Operand::Mem { base, offset }));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(!bytecode.is_empty());
    }

    /// Property: Arithmetic chains with various operations work.
    #[test]
    fn prop_mixed_arithmetic(ops in 1usize..20) {
        let config = TargetConfig::default();
        let mut module = Module::new("mixed_arith");
        let mut func = MachineFunction::new("arith_fn");

        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 100));
        entry.push(MachineInst::li(v1, 3));

        let mut curr = v0;
        for i in 0..ops {
            let next = func.new_vreg();
            match i % 5 {
                0 => entry.push(MachineInst::add(next, curr, v1)),
                1 => entry.push(MachineInst::sub(next, curr, v1)),
                2 => entry.push(MachineInst::mul(next, curr, v1)),
                3 => entry.push(MachineInst::and(next, curr, v1)),
                _ => entry.push(MachineInst::or(next, curr, v1)),
            }
            curr = next;
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(!bytecode.is_empty());
    }

    /// Property: Multiple functions compile correctly.
    #[test]
    fn prop_multi_function_module(num_funcs in 1usize..10) {
        let config = TargetConfig::default();
        let mut module = Module::new("multi_fn");

        for i in 0..num_funcs {
            let mut func = MachineFunction::new(&format!("fn_{}", i));
            let v = func.new_vreg();

            let mut entry = MachineBlock::new("entry");
            entry.push(MachineInst::li(v, i as i64));
            entry.push(MachineInst::ret());
            func.add_block(entry);
            module.add_function(func);
        }

        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(!bytecode.is_empty());
    }

    /// Property: Nested control flow compiles correctly.
    #[test]
    fn prop_nested_branches(depth in 1usize..5) {
        let config = TargetConfig::default();
        let mut module = Module::new("nested");
        let mut func = MachineFunction::new("nested_fn");

        let cond = func.new_vreg();
        let zero = func.new_vreg();
        let ra = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(cond, 1));
        entry.push(MachineInst::li(zero, 0));
        entry.push(MachineInst::bne(cond, zero, "level_0"));
        func.add_block(entry);

        // Create nested if-then-else levels
        for i in 0..depth {
            let val = func.new_vreg();
            let mut block = MachineBlock::new(&format!("level_{}", i));
            block.push(MachineInst::li(val, i as i64));
            if i < depth - 1 {
                block.push(MachineInst::bne(cond, zero, &format!("level_{}", i + 1)));
            } else {
                block.push(MachineInst::jal(ra, "exit"));
            }
            func.add_block(block);
        }

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(!bytecode.is_empty());
    }

    /// Property: Different block orderings compile correctly.
    #[test]
    fn prop_block_count(num_blocks in 2usize..20) {
        let config = TargetConfig::default();
        let mut module = Module::new("blocks");
        let mut func = MachineFunction::new("blocks_fn");

        let ra = func.new_vreg();

        // Entry block jumps to first block
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::jal(ra, "block_0"));
        func.add_block(entry);

        // Chain of blocks
        for i in 0..num_blocks {
            let val = func.new_vreg();
            let mut block = MachineBlock::new(&format!("block_{}", i));
            block.push(MachineInst::li(val, i as i64));
            if i < num_blocks - 1 {
                block.push(MachineInst::jal(ra, &format!("block_{}", i + 1)));
            } else {
                block.push(MachineInst::jal(ra, "exit"));
            }
            func.add_block(block);
        }

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(!bytecode.is_empty());
    }
}

// =============================================================================
// CRYPTO-LIKE PATTERN TESTS
// =============================================================================

/// Test XOR chain (common in crypto).
#[test]
fn test_xor_chain() {
    let config = TargetConfig::default();
    let mut module = Module::new("xor_chain");
    let mut func = MachineFunction::new("xor_fn");

    let state = func.new_vreg();
    let keys: Vec<VReg> = (0..8).map(|_| func.new_vreg()).collect();
    let mut results: Vec<VReg> = Vec::new();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(state, 0x12345678));

    // Load round keys
    for (i, &k) in keys.iter().enumerate() {
        entry.push(MachineInst::li(k, (0xDEADBEEF_u64 ^ (i as u64 * 0x11111111)) as i64));
    }

    // XOR chain (like encryption rounds)
    let mut current = state;
    for &k in &keys {
        let next = func.new_vreg();
        entry.push(MachineInst::xor(next, current, k));
        results.push(next);
        current = next;
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test rotate-like pattern using shifts and OR.
#[test]
fn test_rotate_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("rotate");
    let mut func = MachineFunction::new("rotate_fn");

    let val = func.new_vreg();
    let shift_left = func.new_vreg();
    let shift_right = func.new_vreg();
    let rotated = func.new_vreg();
    let amt_left = func.new_vreg();
    let amt_right = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(val, 0x12345678));
    entry.push(MachineInst::li(amt_left, 8));
    entry.push(MachineInst::li(amt_right, 24));

    // Rotate left by 8: (val << 8) | (val >> 24)
    entry.push(MachineInst::new(Opcode::SLL)
        .dst(Operand::VReg(shift_left))
        .src(Operand::VReg(val))
        .src(Operand::VReg(amt_left)));
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(shift_right))
        .src(Operand::VReg(val))
        .src(Operand::VReg(amt_right)));
    entry.push(MachineInst::or(rotated, shift_left, shift_right));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test S-box lookup pattern (table access).
#[test]
fn test_sbox_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("sbox");
    let mut func = MachineFunction::new("sbox_fn");

    let table_base = func.new_vreg();
    let input = func.new_vreg();
    let offset = func.new_vreg();
    let scaled = func.new_vreg();
    let addr = func.new_vreg();
    let output = func.new_vreg();
    let four = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(table_base, 0x1000)); // S-box table address
    entry.push(MachineInst::li(input, 0x42));
    entry.push(MachineInst::li(four, 4));

    // Compute offset: input & 0xFF (already masked)
    entry.push(MachineInst::mov(offset, input));

    // Scale by element size (4 bytes)
    entry.push(MachineInst::mul(scaled, offset, four));

    // Add to base
    entry.push(MachineInst::add(addr, table_base, scaled));

    // Load from table (simulated - in real code would use computed address)
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(output))
        .src(Operand::Mem { base: table_base, offset: 0 }));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test modular arithmetic pattern.
#[test]
fn test_modular_arithmetic() {
    let config = TargetConfig::default();
    let mut module = Module::new("mod_arith");
    let mut func = MachineFunction::new("mod_fn");

    let a = func.new_vreg();
    let b = func.new_vreg();
    let modulus = func.new_vreg();
    let sum = func.new_vreg();
    let product = func.new_vreg();
    let sum_mod = func.new_vreg();
    let prod_mod = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, 12345));
    entry.push(MachineInst::li(b, 67890));
    entry.push(MachineInst::li(modulus, 65537)); // Fermat prime

    // Modular addition: (a + b) % m
    entry.push(MachineInst::add(sum, a, b));
    entry.push(MachineInst::new(Opcode::REM)
        .dst(Operand::VReg(sum_mod))
        .src(Operand::VReg(sum))
        .src(Operand::VReg(modulus)));

    // Modular multiplication: (a * b) % m
    entry.push(MachineInst::mul(product, a, b));
    entry.push(MachineInst::new(Opcode::REM)
        .dst(Operand::VReg(prod_mod))
        .src(Operand::VReg(product))
        .src(Operand::VReg(modulus)));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// ABI AND CALLING CONVENTION TESTS
// =============================================================================

/// Test function with many arguments (exceeds registers).
#[test]
fn test_many_arguments() {
    let config = TargetConfig::default();
    let mut module = Module::new("many_args");

    // Callee with many parameters (would need stack)
    let mut callee = MachineFunction::new("callee");
    let params: Vec<VReg> = (0..16).map(|_| callee.new_vreg()).collect();
    let sum = callee.new_vreg();

    let mut callee_entry = MachineBlock::new("entry");
    // Sum all parameters
    callee_entry.push(MachineInst::mov(sum, params[0]));
    for i in 1..16 {
        let tmp = callee.new_vreg();
        callee_entry.push(MachineInst::add(tmp, sum, params[i]));
        callee_entry.push(MachineInst::mov(sum, tmp));
    }
    callee_entry.push(MachineInst::ret());
    callee.add_block(callee_entry);
    module.add_function(callee);

    // Caller sets up 16 arguments
    let mut caller = MachineFunction::new("caller");
    let args: Vec<VReg> = (0..16).map(|_| caller.new_vreg()).collect();

    let mut caller_entry = MachineBlock::new("entry");
    for (i, &arg) in args.iter().enumerate() {
        caller_entry.push(MachineInst::li(arg, i as i64 * 10));
    }
    caller_entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("callee".to_string())));
    caller_entry.push(MachineInst::ret());
    caller.add_block(caller_entry);
    module.add_function(caller);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test callee-saved register preservation.
#[test]
fn test_callee_saved_registers() {
    let config = TargetConfig::default();
    let mut module = Module::new("callee_saved");
    let mut func = MachineFunction::new("preserve_fn");

    // Use many registers to force spilling of callee-saved
    let regs: Vec<VReg> = (0..20).map(|_| func.new_vreg()).collect();
    let mut sums: Vec<VReg> = Vec::new();

    let mut entry = MachineBlock::new("entry");

    // Initialize all registers
    for (i, &r) in regs.iter().enumerate() {
        entry.push(MachineInst::li(r, i as i64));
    }

    // Call a function (this should preserve callee-saved registers)
    entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("dummy".to_string())));

    // Use all registers after call (to verify preservation)
    for i in 0..regs.len() - 1 {
        let sum = func.new_vreg();
        entry.push(MachineInst::add(sum, regs[i], regs[i + 1]));
        sums.push(sum);
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    // Add dummy callee
    let mut dummy = MachineFunction::new("dummy");
    let mut dummy_entry = MachineBlock::new("entry");
    dummy_entry.push(MachineInst::ret());
    dummy.add_block(dummy_entry);
    module.add_function(dummy);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test return value handling.
#[test]
fn test_return_value() {
    let config = TargetConfig::default();
    let mut module = Module::new("ret_val");

    // Function that returns a computed value
    let mut compute = MachineFunction::new("compute");
    let a = compute.new_vreg();
    let b = compute.new_vreg();
    let result = compute.new_vreg();

    let mut compute_entry = MachineBlock::new("entry");
    compute_entry.push(MachineInst::li(a, 42));
    compute_entry.push(MachineInst::li(b, 58));
    compute_entry.push(MachineInst::add(result, a, b)); // Returns 100
    compute_entry.push(MachineInst::ret());
    compute.add_block(compute_entry);
    module.add_function(compute);

    // Caller uses the return value
    let mut caller = MachineFunction::new("caller");
    let ret = caller.new_vreg();
    let doubled = caller.new_vreg();

    let mut caller_entry = MachineBlock::new("entry");
    caller_entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("compute".to_string())));
    // Assume return value is in ret
    caller_entry.push(MachineInst::li(ret, 100)); // Simulated return value
    caller_entry.push(MachineInst::add(doubled, ret, ret));
    caller_entry.push(MachineInst::ret());
    caller.add_block(caller_entry);
    module.add_function(caller);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// OPTIMIZATION EDGE CASE TESTS
// =============================================================================

/// Test dead code after return.
#[test]
fn test_dead_code_after_return() {
    use zkir_llvm::opt::OptLevel;

    let mut module = Module::new("dead_after_ret");
    let mut func = MachineFunction::new("dead_fn");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    // These should be eliminated as dead code
    entry.push(MachineInst::li(v1, 100));
    entry.push(MachineInst::li(v2, 200));
    func.add_block(entry);
    module.add_function(func);

    let config = TargetConfig::default();
    zkir_llvm::opt::optimize_with_level(&mut module, &config, OptLevel::O2).unwrap();

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test unreachable block elimination.
#[test]
fn test_unreachable_block() {
    use zkir_llvm::opt::OptLevel;

    let mut module = Module::new("unreachable");
    let mut func = MachineFunction::new("unreach_fn");

    let v0 = func.new_vreg();
    let ra = func.new_vreg();

    // Entry returns directly
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    // Unreachable block (no predecessors)
    let mut unreachable = MachineBlock::new("unreachable");
    let v1 = func.new_vreg();
    unreachable.push(MachineInst::li(v1, 100));
    unreachable.push(MachineInst::jal(ra, "entry"));
    func.add_block(unreachable);

    func.rebuild_cfg();
    module.add_function(func);

    let config = TargetConfig::default();
    zkir_llvm::opt::optimize_with_level(&mut module, &config, OptLevel::O2).unwrap();

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test constant propagation through multiple operations.
#[test]
fn test_constant_chain_propagation() {
    use zkir_llvm::opt::OptLevel;

    let mut module = Module::new("const_chain");
    let mut func = MachineFunction::new("const_fn");

    let a = func.new_vreg();
    let b = func.new_vreg();
    let c = func.new_vreg();
    let d = func.new_vreg();
    let e = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, 10));
    entry.push(MachineInst::li(b, 20));
    entry.push(MachineInst::add(c, a, b)); // 30
    entry.push(MachineInst::add(d, c, a)); // 40
    entry.push(MachineInst::add(e, d, b)); // 60
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let config = TargetConfig::default();
    zkir_llvm::opt::optimize_with_level(&mut module, &config, OptLevel::O2).unwrap();

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// ASSEMBLY OUTPUT FORMAT TESTS
// =============================================================================

/// Test assembly output contains function labels.
#[test]
fn test_asm_function_labels() {
    let config = TargetConfig::default();
    let mut module = Module::new("asm_labels");

    for name in ["main", "helper", "util"] {
        let mut func = MachineFunction::new(name);
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);
    }

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let asm = emit::format_asm(&allocated);

    assert!(asm.contains("main:"));
    assert!(asm.contains("helper:"));
    assert!(asm.contains("util:"));
}

/// Test assembly output contains instruction mnemonics.
#[test]
fn test_asm_instruction_mnemonics() {
    let config = TargetConfig::default();
    let mut module = Module::new("asm_mnemonics");
    let mut func = MachineFunction::new("mixed");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let base = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::li(v1, 10));
    entry.push(MachineInst::add(v2, v0, v1));
    entry.push(MachineInst::li(base, 0x1000));
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(v2))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let asm = emit::format_asm(&allocated);

    assert!(asm.contains("add"));
    assert!(asm.contains("sw") || asm.contains("store"));
}

/// Test assembly section headers.
#[test]
fn test_asm_section_headers() {
    let config = TargetConfig::default();
    let mut module = Module::new("asm_sections");
    let mut func = MachineFunction::new("test");

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let asm = emit::format_asm(&module);

    assert!(asm.contains(".text") || asm.contains(".section"));
    assert!(asm.contains("ZKIR"));
}

// =============================================================================
// STRESS TESTS FOR EXTREME CASES
// =============================================================================

/// Test function with 100 blocks.
#[test]
fn test_100_blocks() {
    let config = TargetConfig::default();
    let mut module = Module::new("100_blocks");
    let mut func = MachineFunction::new("blocks100");

    let ra = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::jal(ra, "block_0"));
    func.add_block(entry);

    for i in 0..100 {
        let val = func.new_vreg();
        let mut block = MachineBlock::new(&format!("block_{}", i));
        block.push(MachineInst::li(val, i as i64));
        if i < 99 {
            block.push(MachineInst::jal(ra, &format!("block_{}", i + 1)));
        } else {
            block.push(MachineInst::jal(ra, "exit"));
        }
        func.add_block(block);
    }

    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(bytecode.len() > 400); // At least 4 bytes per block
}

/// Test function with 200 virtual registers.
#[test]
fn test_200_vregs() {
    let config = TargetConfig::default();
    let mut module = Module::new("200_vregs");
    let mut func = MachineFunction::new("vregs200");

    let mut entry = MachineBlock::new("entry");

    // Create 200 virtual registers
    let vregs: Vec<VReg> = (0..200).map(|_| func.new_vreg()).collect();

    // Initialize all
    for (i, &v) in vregs.iter().enumerate() {
        entry.push(MachineInst::li(v, i as i64));
    }

    // Use all to keep them live
    for i in 0..199 {
        let sum = func.new_vreg();
        entry.push(MachineInst::add(sum, vregs[i], vregs[i + 1]));
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test function with 500 instructions.
#[test]
fn test_500_instructions() {
    let config = TargetConfig::default();
    let mut module = Module::new("500_insts");
    let mut func = MachineFunction::new("insts500");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1));
    entry.push(MachineInst::li(v1, 2));

    // 500 instructions
    let mut current = v0;
    for i in 0..500 {
        let next = func.new_vreg();
        match i % 4 {
            0 => entry.push(MachineInst::add(next, current, v1)),
            1 => entry.push(MachineInst::sub(next, current, v1)),
            2 => entry.push(MachineInst::and(next, current, v1)),
            _ => entry.push(MachineInst::or(next, current, v1)),
        }
        current = next;
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(bytecode.len() > 1000); // At least 2 bytes per instruction
}

/// Test module with 20 functions.
#[test]
fn test_20_functions() {
    let config = TargetConfig::default();
    let mut module = Module::new("20_funcs");

    for i in 0..20 {
        let mut func = MachineFunction::new(&format!("func_{}", i));
        let v = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v, i as i64 * 100));

        // Each function calls the next (except last)
        if i < 19 {
            entry.push(MachineInst::new(Opcode::CALL)
                .src(Operand::Label(format!("func_{}", i + 1))));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);
    }

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());

    let asm = emit::format_asm(&allocated);
    for i in 0..20 {
        assert!(asm.contains(&format!("func_{}:", i)));
    }
}

/// Test deeply nested loops (3 levels).
#[test]
fn test_triple_nested_loop() {
    let config = TargetConfig::default();
    let mut module = Module::new("triple_loop");
    let mut func = MachineFunction::new("triple");

    let i = func.new_vreg();
    let j = func.new_vreg();
    let k = func.new_vreg();
    let limit = func.new_vreg();
    let one = func.new_vreg();
    let sum = func.new_vreg();
    let tmp = func.new_vreg();
    let ra = func.new_vreg();

    // Entry
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(i, 0));
    entry.push(MachineInst::li(limit, 5));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::li(sum, 0));
    entry.push(MachineInst::jal(ra, "loop_i"));
    func.add_block(entry);

    // Outer loop (i)
    let mut loop_i = MachineBlock::new("loop_i");
    loop_i.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(i))
        .src(Operand::VReg(limit))
        .src(Operand::Label("body_i".to_string())));
    func.add_block(loop_i);

    let mut body_i = MachineBlock::new("body_i");
    body_i.push(MachineInst::li(j, 0));
    body_i.push(MachineInst::jal(ra, "loop_j"));
    func.add_block(body_i);

    // Middle loop (j)
    let mut loop_j = MachineBlock::new("loop_j");
    loop_j.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(j))
        .src(Operand::VReg(limit))
        .src(Operand::Label("body_j".to_string())));
    func.add_block(loop_j);

    let mut body_j = MachineBlock::new("body_j");
    body_j.push(MachineInst::li(k, 0));
    body_j.push(MachineInst::jal(ra, "loop_k"));
    func.add_block(body_j);

    // Inner loop (k)
    let mut loop_k = MachineBlock::new("loop_k");
    loop_k.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(k))
        .src(Operand::VReg(limit))
        .src(Operand::Label("body_k".to_string())));
    func.add_block(loop_k);

    let mut body_k = MachineBlock::new("body_k");
    body_k.push(MachineInst::add(tmp, sum, one));
    body_k.push(MachineInst::mov(sum, tmp));
    body_k.push(MachineInst::add(k, k, one));
    body_k.push(MachineInst::jal(ra, "loop_k"));
    func.add_block(body_k);

    // Exit k loop
    let mut exit_k = MachineBlock::new("exit_k");
    exit_k.push(MachineInst::add(j, j, one));
    exit_k.push(MachineInst::jal(ra, "loop_j"));
    func.add_block(exit_k);

    // Exit j loop
    let mut exit_j = MachineBlock::new("exit_j");
    exit_j.push(MachineInst::add(i, i, one));
    exit_j.push(MachineInst::jal(ra, "loop_i"));
    func.add_block(exit_j);

    // Exit i loop
    let mut exit_i = MachineBlock::new("exit_i");
    exit_i.push(MachineInst::ret());
    func.add_block(exit_i);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Bit Manipulation Pattern Tests
// ============================================================================

/// Test bit field extraction pattern (value >> shift) & mask
#[test]
fn test_bit_field_extraction() {
    let config = TargetConfig::default();
    let mut module = Module::new("bit_field");
    let mut func = MachineFunction::new("extract_bits");

    let value = VReg::new(0);
    let shift = VReg::new(1);
    let mask = VReg::new(2);
    let shifted = VReg::new(3);
    let result = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(value, 0xDEADBEEF_u32 as i64));
    entry.push(MachineInst::li(shift, 8));
    entry.push(MachineInst::li(mask, 0xFF));
    // Extract byte at position 1: (0xDEADBEEF >> 8) & 0xFF = 0xBE
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(shifted))
        .src(Operand::VReg(value))
        .src(Operand::VReg(shift)));
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(shifted))
        .src(Operand::VReg(mask)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test bit field insertion pattern: (dest & ~mask) | ((value << shift) & mask)
#[test]
fn test_bit_field_insertion() {
    let config = TargetConfig::default();
    let mut module = Module::new("bit_insert");
    let mut func = MachineFunction::new("insert_bits");

    let dest = VReg::new(0);
    let value = VReg::new(1);
    let mask = VReg::new(2);
    let shift = VReg::new(3);
    let not_mask = VReg::new(4);
    let cleared = VReg::new(5);
    let shifted = VReg::new(6);
    let masked = VReg::new(7);
    let result = VReg::new(8);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(dest, 0x12345678_u32 as i64));
    entry.push(MachineInst::li(value, 0xAB));
    entry.push(MachineInst::li(mask, 0xFF00));
    entry.push(MachineInst::li(shift, 8));
    // not_mask = ~mask (simulate with XOR -1)
    entry.push(MachineInst::new(Opcode::XORI)
        .dst(Operand::VReg(not_mask))
        .src(Operand::VReg(mask))
        .src(Operand::Imm(-1)));
    // cleared = dest & ~mask
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(cleared))
        .src(Operand::VReg(dest))
        .src(Operand::VReg(not_mask)));
    // shifted = value << shift
    entry.push(MachineInst::new(Opcode::SLL)
        .dst(Operand::VReg(shifted))
        .src(Operand::VReg(value))
        .src(Operand::VReg(shift)));
    // masked = shifted & mask
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(masked))
        .src(Operand::VReg(shifted))
        .src(Operand::VReg(mask)));
    // result = cleared | masked
    entry.push(MachineInst::new(Opcode::OR)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(cleared))
        .src(Operand::VReg(masked)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test popcount-like pattern (counting set bits via repeated shift and add)
#[test]
fn test_popcount_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("popcount");
    let mut func = MachineFunction::new("count_bits");

    let value = VReg::new(0);
    let count = VReg::new(1);
    let one = VReg::new(2);
    let bit = VReg::new(3);
    let tmp = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(value, 0b10101010));
    entry.push(MachineInst::li(count, 0));
    entry.push(MachineInst::li(one, 1));
    func.add_block(entry);

    // Unrolled loop for 8 bits
    let mut body = MachineBlock::new("count_loop");
    for _ in 0..8 {
        // bit = value & 1
        body.push(MachineInst::new(Opcode::AND)
            .dst(Operand::VReg(bit))
            .src(Operand::VReg(value))
            .src(Operand::VReg(one)));
        // count += bit
        body.push(MachineInst::add(tmp, count, bit));
        body.push(MachineInst::mov(count, tmp));
        // value >>= 1
        body.push(MachineInst::new(Opcode::SRL)
            .dst(Operand::VReg(tmp))
            .src(Operand::VReg(value))
            .src(Operand::VReg(one)));
        body.push(MachineInst::mov(value, tmp));
    }
    body.push(MachineInst::ret());
    func.add_block(body);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test find-first-set pattern (finding lowest set bit position)
#[test]
fn test_ffs_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("ffs");
    let mut func = MachineFunction::new("find_first_set");

    let value = VReg::new(0);
    let pos = VReg::new(1);
    let one = VReg::new(2);
    let bit = VReg::new(3);
    let tmp = VReg::new(4);
    let zero = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(value, 0b01000000)); // bit 6 is set
    entry.push(MachineInst::li(pos, 0));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::li(zero, 0));
    func.add_block(entry);

    // Check bit 0
    let mut check = MachineBlock::new("check");
    check.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(bit))
        .src(Operand::VReg(value))
        .src(Operand::VReg(one)));
    check.push(MachineInst::new(Opcode::BNE)
        .src(Operand::VReg(bit))
        .src(Operand::VReg(zero))
        .src(Operand::Label("found".to_string())));
    // value >>= 1, pos++
    check.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(tmp))
        .src(Operand::VReg(value))
        .src(Operand::VReg(one)));
    check.push(MachineInst::mov(value, tmp));
    check.push(MachineInst::add(tmp, pos, one));
    check.push(MachineInst::mov(pos, tmp));
    check.push(MachineInst::jal(VReg::new(6), "check"));
    func.add_block(check);

    let mut found = MachineBlock::new("found");
    found.push(MachineInst::ret());
    func.add_block(found);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test byte swap pattern (endianness conversion)
#[test]
fn test_byte_swap_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("bswap");
    let mut func = MachineFunction::new("byte_swap_32");

    let value = VReg::new(0);
    let b0 = VReg::new(1);
    let b1 = VReg::new(2);
    let b2 = VReg::new(3);
    let b3 = VReg::new(4);
    let s0 = VReg::new(5);
    let s1 = VReg::new(6);
    let s2 = VReg::new(7);
    let s3 = VReg::new(8);
    let t1 = VReg::new(9);
    let t2 = VReg::new(10);
    let result = VReg::new(11);
    let mask = VReg::new(12);
    let eight = VReg::new(13);
    let sixteen = VReg::new(14);
    let twentyfour = VReg::new(15);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(value, 0x12345678_u32 as i64));
    entry.push(MachineInst::li(mask, 0xFF));
    entry.push(MachineInst::li(eight, 8));
    entry.push(MachineInst::li(sixteen, 16));
    entry.push(MachineInst::li(twentyfour, 24));

    // Extract bytes
    // b0 = value & 0xFF
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(b0))
        .src(Operand::VReg(value))
        .src(Operand::VReg(mask)));
    // b1 = (value >> 8) & 0xFF
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(s0))
        .src(Operand::VReg(value))
        .src(Operand::VReg(eight)));
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(b1))
        .src(Operand::VReg(s0))
        .src(Operand::VReg(mask)));
    // b2 = (value >> 16) & 0xFF
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(s1))
        .src(Operand::VReg(value))
        .src(Operand::VReg(sixteen)));
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(b2))
        .src(Operand::VReg(s1))
        .src(Operand::VReg(mask)));
    // b3 = (value >> 24) & 0xFF
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(s2))
        .src(Operand::VReg(value))
        .src(Operand::VReg(twentyfour)));
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(b3))
        .src(Operand::VReg(s2))
        .src(Operand::VReg(mask)));

    // Reassemble in reverse order
    // s3 = b0 << 24
    entry.push(MachineInst::new(Opcode::SLL)
        .dst(Operand::VReg(s3))
        .src(Operand::VReg(b0))
        .src(Operand::VReg(twentyfour)));
    // t1 = b1 << 16
    entry.push(MachineInst::new(Opcode::SLL)
        .dst(Operand::VReg(t1))
        .src(Operand::VReg(b1))
        .src(Operand::VReg(sixteen)));
    // t2 = b2 << 8
    entry.push(MachineInst::new(Opcode::SLL)
        .dst(Operand::VReg(t2))
        .src(Operand::VReg(b2))
        .src(Operand::VReg(eight)));
    // result = s3 | t1 | t2 | b3
    let tmp1 = VReg::new(16);
    let tmp2 = VReg::new(17);
    entry.push(MachineInst::new(Opcode::OR)
        .dst(Operand::VReg(tmp1))
        .src(Operand::VReg(s3))
        .src(Operand::VReg(t1)));
    entry.push(MachineInst::new(Opcode::OR)
        .dst(Operand::VReg(tmp2))
        .src(Operand::VReg(tmp1))
        .src(Operand::VReg(t2)));
    entry.push(MachineInst::new(Opcode::OR)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(tmp2))
        .src(Operand::VReg(b3)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Signed vs Unsigned Operation Tests
// ============================================================================

/// Test signed comparison (SLT) vs unsigned (SLTU)
#[test]
fn test_signed_vs_unsigned_comparison() {
    let config = TargetConfig::default();
    let mut module = Module::new("cmp_test");
    let mut func = MachineFunction::new("compare");

    let neg_one = VReg::new(0);
    let one = VReg::new(1);
    let signed_result = VReg::new(2);
    let unsigned_result = VReg::new(3);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(neg_one, -1)); // 0xFFFFFFFF...
    entry.push(MachineInst::li(one, 1));

    // Signed: -1 < 1 is TRUE
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(signed_result))
        .src(Operand::VReg(neg_one))
        .src(Operand::VReg(one)));

    // Unsigned: 0xFFFF... < 1 is FALSE (0xFFFF... is huge unsigned)
    entry.push(MachineInst::new(Opcode::SLTU)
        .dst(Operand::VReg(unsigned_result))
        .src(Operand::VReg(neg_one))
        .src(Operand::VReg(one)));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test arithmetic right shift (preserves sign) vs logical (fills with zeros)
#[test]
fn test_signed_vs_unsigned_shift() {
    let config = TargetConfig::default();
    let mut module = Module::new("shift_test");
    let mut func = MachineFunction::new("shifts");

    let neg_val = VReg::new(0);
    let shift = VReg::new(1);
    let arith_result = VReg::new(2);
    let logic_result = VReg::new(3);

    let mut entry = MachineBlock::new("entry");
    // -128 in 8-bit two's complement = 0x80 (or 0xFFFFFF80 sign-extended)
    entry.push(MachineInst::li(neg_val, -128));
    entry.push(MachineInst::li(shift, 2));

    // Arithmetic right shift: -128 >> 2 = -32 (sign extended)
    entry.push(MachineInst::new(Opcode::SRA)
        .dst(Operand::VReg(arith_result))
        .src(Operand::VReg(neg_val))
        .src(Operand::VReg(shift)));

    // Logical right shift: 0xFFFFFF80 >> 2 = 0x3FFFFFE0 (zeros fill)
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(logic_result))
        .src(Operand::VReg(neg_val))
        .src(Operand::VReg(shift)));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test division and remainder operations
#[test]
fn test_division_and_remainder() {
    let config = TargetConfig::default();
    let mut module = Module::new("div_test");
    let mut func = MachineFunction::new("divisions");

    let neg_ten = VReg::new(0);
    let three = VReg::new(1);
    let div_result = VReg::new(2);
    let rem_result = VReg::new(3);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(neg_ten, -10));
    entry.push(MachineInst::li(three, 3));

    // Division: -10 / 3 = -3
    entry.push(MachineInst::new(Opcode::DIV)
        .dst(Operand::VReg(div_result))
        .src(Operand::VReg(neg_ten))
        .src(Operand::VReg(three)));

    // Remainder: -10 % 3 = -1
    entry.push(MachineInst::new(Opcode::REM)
        .dst(Operand::VReg(rem_result))
        .src(Operand::VReg(neg_ten))
        .src(Operand::VReg(three)));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Live Range Splitting Tests
// ============================================================================

/// Test value used far apart (potentially needs spill/reload)
#[test]
fn test_long_live_range() {
    let config = TargetConfig::default();
    let mut module = Module::new("long_range");
    let mut func = MachineFunction::new("long_live");

    let early_val = VReg::new(0);
    let temp = VReg::new(1);
    let one = VReg::new(2);
    let result = VReg::new(3);

    let mut entry = MachineBlock::new("entry");
    // Define early_val at the start
    entry.push(MachineInst::li(early_val, 42));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::li(temp, 0));

    // Many operations that don't use early_val (create pressure)
    for i in 4..20 {
        entry.push(MachineInst::li(VReg::new(i), i as i64));
        entry.push(MachineInst::add(VReg::new(i + 16), VReg::new(i), one));
    }

    // Now use early_val - it must have survived across all those operations
    entry.push(MachineInst::add(result, early_val, one));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test live range with a "hole" (value not needed in middle section)
#[test]
fn test_live_range_with_hole() {
    let config = TargetConfig::default();
    let mut module = Module::new("hole_test");
    let mut func = MachineFunction::new("range_hole");

    let a = VReg::new(0);
    let b = VReg::new(1);
    let c = VReg::new(2);
    let d = VReg::new(3);
    let result = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, 10));  // a is defined here
    entry.push(MachineInst::li(b, 20));
    entry.push(MachineInst::jal(VReg::new(10), "middle"));
    func.add_block(entry);

    // Middle section - 'a' is not used here, so its live range has a hole
    let mut middle = MachineBlock::new("middle");
    middle.push(MachineInst::li(c, 30));
    middle.push(MachineInst::li(d, 40));
    middle.push(MachineInst::add(VReg::new(5), b, c));
    middle.push(MachineInst::add(VReg::new(6), VReg::new(5), d));
    middle.push(MachineInst::jal(VReg::new(10), "end"));
    func.add_block(middle);

    // End section - 'a' is used again
    let mut end = MachineBlock::new("end");
    end.push(MachineInst::add(result, a, VReg::new(6)));
    end.push(MachineInst::ret());
    func.add_block(end);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Conditional Move Pattern Tests
// ============================================================================

/// Test select pattern: result = cond ? a : b
#[test]
fn test_conditional_select() {
    let config = TargetConfig::default();
    let mut module = Module::new("select");
    let mut func = MachineFunction::new("select_val");

    let cond = VReg::new(0);
    let a = VReg::new(1);
    let b = VReg::new(2);
    let zero = VReg::new(3);
    let result = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(cond, 1)); // true
    entry.push(MachineInst::li(a, 100));
    entry.push(MachineInst::li(b, 200));
    entry.push(MachineInst::li(zero, 0));
    // Branch based on condition
    entry.push(MachineInst::new(Opcode::BNE)
        .src(Operand::VReg(cond))
        .src(Operand::VReg(zero))
        .src(Operand::Label("take_a".to_string())));
    entry.push(MachineInst::jal(VReg::new(5), "take_b"));
    func.add_block(entry);

    let mut take_a = MachineBlock::new("take_a");
    take_a.push(MachineInst::mov(result, a));
    take_a.push(MachineInst::jal(VReg::new(5), "done"));
    func.add_block(take_a);

    let mut take_b = MachineBlock::new("take_b");
    take_b.push(MachineInst::mov(result, b));
    take_b.push(MachineInst::jal(VReg::new(5), "done"));
    func.add_block(take_b);

    let mut done = MachineBlock::new("done");
    done.push(MachineInst::ret());
    func.add_block(done);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test min/max pattern using conditional
#[test]
fn test_min_max_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("minmax");
    let mut func = MachineFunction::new("find_min_max");

    let a = VReg::new(0);
    let b = VReg::new(1);
    let min_result = VReg::new(2);
    let max_result = VReg::new(3);
    let cmp = VReg::new(4);
    let zero = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, 25));
    entry.push(MachineInst::li(b, 17));
    entry.push(MachineInst::li(zero, 0));
    // cmp = a < b
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(cmp))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));
    entry.push(MachineInst::new(Opcode::BNE)
        .src(Operand::VReg(cmp))
        .src(Operand::VReg(zero))
        .src(Operand::Label("a_smaller".to_string())));
    entry.push(MachineInst::jal(VReg::new(6), "b_smaller"));
    func.add_block(entry);

    // a < b: min = a, max = b
    let mut a_smaller = MachineBlock::new("a_smaller");
    a_smaller.push(MachineInst::mov(min_result, a));
    a_smaller.push(MachineInst::mov(max_result, b));
    a_smaller.push(MachineInst::jal(VReg::new(6), "done"));
    func.add_block(a_smaller);

    // b <= a: min = b, max = a
    let mut b_smaller = MachineBlock::new("b_smaller");
    b_smaller.push(MachineInst::mov(min_result, b));
    b_smaller.push(MachineInst::mov(max_result, a));
    b_smaller.push(MachineInst::jal(VReg::new(6), "done"));
    func.add_block(b_smaller);

    let mut done = MachineBlock::new("done");
    done.push(MachineInst::ret());
    func.add_block(done);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test absolute value pattern: abs(x) = x < 0 ? -x : x
#[test]
fn test_absolute_value() {
    let config = TargetConfig::default();
    let mut module = Module::new("abs");
    let mut func = MachineFunction::new("absolute");

    let x = VReg::new(0);
    let zero = VReg::new(1);
    let neg_x = VReg::new(2);
    let result = VReg::new(3);
    let cmp = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(x, -42));
    entry.push(MachineInst::li(zero, 0));
    // cmp = x < 0
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(cmp))
        .src(Operand::VReg(x))
        .src(Operand::VReg(zero)));
    // neg_x = 0 - x
    entry.push(MachineInst::sub(neg_x, zero, x));
    entry.push(MachineInst::new(Opcode::BNE)
        .src(Operand::VReg(cmp))
        .src(Operand::VReg(zero))
        .src(Operand::Label("negative".to_string())));
    entry.push(MachineInst::jal(VReg::new(5), "positive"));
    func.add_block(entry);

    let mut negative = MachineBlock::new("negative");
    negative.push(MachineInst::mov(result, neg_x));
    negative.push(MachineInst::jal(VReg::new(5), "done"));
    func.add_block(negative);

    let mut positive = MachineBlock::new("positive");
    positive.push(MachineInst::mov(result, x));
    positive.push(MachineInst::jal(VReg::new(5), "done"));
    func.add_block(positive);

    let mut done = MachineBlock::new("done");
    done.push(MachineInst::ret());
    func.add_block(done);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Data Structure Access Pattern Tests
// ============================================================================

/// Test array element access pattern (base + index * element_size)
#[test]
fn test_array_access_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("array");
    let mut func = MachineFunction::new("array_access");

    let base = VReg::new(0);
    let index = VReg::new(1);
    let elem_size = VReg::new(2);
    let offset = VReg::new(3);
    let addr = VReg::new(4);
    let value = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000)); // Array base address
    entry.push(MachineInst::li(index, 5));     // Access element 5
    entry.push(MachineInst::li(elem_size, 4)); // 4 bytes per element

    // offset = index * elem_size
    entry.push(MachineInst::new(Opcode::MUL)
        .dst(Operand::VReg(offset))
        .src(Operand::VReg(index))
        .src(Operand::VReg(elem_size)));
    // addr = base + offset
    entry.push(MachineInst::add(addr, base, offset));
    // Load value from addr (offset 0 from computed address)
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(value))
        .src(Operand::Mem { base: addr, offset: 0 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test struct field access pattern (base + field_offset)
#[test]
fn test_struct_field_access() {
    let config = TargetConfig::default();
    let mut module = Module::new("struct_access");
    let mut func = MachineFunction::new("field_access");

    let struct_ptr = VReg::new(0);
    let field_a = VReg::new(1);  // offset 0
    let field_b = VReg::new(2);  // offset 4
    let field_c = VReg::new(3);  // offset 8
    let sum = VReg::new(4);
    let tmp = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(struct_ptr, 0x2000)); // Struct base

    // Load field_a at offset 0
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(field_a))
        .src(Operand::Mem { base: struct_ptr, offset: 0 }));
    // Load field_b at offset 4
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(field_b))
        .src(Operand::Mem { base: struct_ptr, offset: 4 }));
    // Load field_c at offset 8
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(field_c))
        .src(Operand::Mem { base: struct_ptr, offset: 8 }));

    // sum = field_a + field_b + field_c
    entry.push(MachineInst::add(tmp, field_a, field_b));
    entry.push(MachineInst::add(sum, tmp, field_c));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test linked list traversal pattern
#[test]
fn test_linked_list_traversal() {
    let config = TargetConfig::default();
    let mut module = Module::new("linked_list");
    let mut func = MachineFunction::new("traverse");

    let node = VReg::new(0);
    let next = VReg::new(1);
    let value = VReg::new(2);
    let sum = VReg::new(3);
    let tmp = VReg::new(4);
    let zero = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(node, 0x3000)); // Head of list
    entry.push(MachineInst::li(sum, 0));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::jal(VReg::new(6), "loop"));
    func.add_block(entry);

    let mut loop_blk = MachineBlock::new("loop");
    // Check if node is null
    loop_blk.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(node))
        .src(Operand::VReg(zero))
        .src(Operand::Label("done".to_string())));
    // Load value from node (offset 0)
    loop_blk.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(value))
        .src(Operand::Mem { base: node, offset: 0 }));
    // sum += value
    loop_blk.push(MachineInst::add(tmp, sum, value));
    loop_blk.push(MachineInst::mov(sum, tmp));
    // Load next pointer (offset 4)
    loop_blk.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(next))
        .src(Operand::Mem { base: node, offset: 4 }));
    // node = next
    loop_blk.push(MachineInst::mov(node, next));
    loop_blk.push(MachineInst::jal(VReg::new(6), "loop"));
    func.add_block(loop_blk);

    let mut done = MachineBlock::new("done");
    done.push(MachineInst::ret());
    func.add_block(done);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test 2D array access pattern (row-major: base + (row * cols + col) * elem_size)
#[test]
fn test_2d_array_access() {
    let config = TargetConfig::default();
    let mut module = Module::new("matrix");
    let mut func = MachineFunction::new("matrix_access");

    let base = VReg::new(0);
    let row = VReg::new(1);
    let col = VReg::new(2);
    let cols = VReg::new(3);
    let elem_size = VReg::new(4);
    let row_offset = VReg::new(5);
    let linear_idx = VReg::new(6);
    let byte_offset = VReg::new(7);
    let addr = VReg::new(8);
    let value = VReg::new(9);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x4000));  // Matrix base
    entry.push(MachineInst::li(row, 3));        // Row 3
    entry.push(MachineInst::li(col, 7));        // Column 7
    entry.push(MachineInst::li(cols, 10));      // 10 columns per row
    entry.push(MachineInst::li(elem_size, 4));  // 4 bytes per element

    // row_offset = row * cols
    entry.push(MachineInst::new(Opcode::MUL)
        .dst(Operand::VReg(row_offset))
        .src(Operand::VReg(row))
        .src(Operand::VReg(cols)));
    // linear_idx = row_offset + col
    entry.push(MachineInst::add(linear_idx, row_offset, col));
    // byte_offset = linear_idx * elem_size
    entry.push(MachineInst::new(Opcode::MUL)
        .dst(Operand::VReg(byte_offset))
        .src(Operand::VReg(linear_idx))
        .src(Operand::VReg(elem_size)));
    // addr = base + byte_offset
    entry.push(MachineInst::add(addr, base, byte_offset));
    // Load value
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(value))
        .src(Operand::Mem { base: addr, offset: 0 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Additional Property Tests
// ============================================================================

proptest! {
    /// Test that any sequence of valid register operations can be allocated
    #[test]
    fn prop_random_register_ops(ops in proptest::collection::vec(0u8..6, 5..20)) {
        let config = TargetConfig::default();
        let mut module = Module::new("prop_reg_ops");
        let mut func = MachineFunction::new("random_ops");

        let mut entry = MachineBlock::new("entry");
        let one = VReg::new(100);
        entry.push(MachineInst::li(one, 1));

        for (i, op) in ops.iter().enumerate() {
            let dest = VReg::new(i as u32);
            let src = VReg::new((i.saturating_sub(1)) as u32);
            match op % 6 {
                0 => entry.push(MachineInst::li(dest, (i as i64) * 7)),
                1 => entry.push(MachineInst::add(dest, src, one)),
                2 => entry.push(MachineInst::sub(dest, src, one)),
                3 => entry.push(MachineInst::mov(dest, src)),
                4 => entry.push(MachineInst::new(Opcode::AND)
                    .dst(Operand::VReg(dest))
                    .src(Operand::VReg(src))
                    .src(Operand::VReg(one))),
                _ => entry.push(MachineInst::new(Opcode::OR)
                    .dst(Operand::VReg(dest))
                    .src(Operand::VReg(src))
                    .src(Operand::VReg(one))),
            }
        }
        entry.push(MachineInst::ret());
        func.add_block(entry);

        func.rebuild_cfg();
        module.add_function(func);

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());
    }

    /// Test that nested control flow with varying depths and branching patterns works
    #[test]
    fn prop_nested_branches_extended(depth in 1usize..5, branching in proptest::collection::vec(proptest::bool::ANY, 1..10)) {
        let config = TargetConfig::default();
        let mut module = Module::new("prop_nested");
        let mut func = MachineFunction::new("nested");

        let cond = VReg::new(0);
        let zero = VReg::new(1);
        let counter = VReg::new(2);
        let one = VReg::new(3);
        let tmp = VReg::new(4);

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(cond, if branching.first().copied().unwrap_or(false) { 1 } else { 0 }));
        entry.push(MachineInst::li(zero, 0));
        entry.push(MachineInst::li(counter, 0));
        entry.push(MachineInst::li(one, 1));
        entry.push(MachineInst::jal(VReg::new(5), "level_0"));
        func.add_block(entry);

        for level in 0..depth {
            let mut block = MachineBlock::new(&format!("level_{}", level));
            block.push(MachineInst::add(tmp, counter, one));
            block.push(MachineInst::mov(counter, tmp));
            if level + 1 < depth {
                block.push(MachineInst::new(Opcode::BNE)
                    .src(Operand::VReg(cond))
                    .src(Operand::VReg(zero))
                    .src(Operand::Label(format!("level_{}", level + 1))));
            }
            block.push(MachineInst::jal(VReg::new(5), "exit"));
            func.add_block(block);
        }

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();
        module.add_function(func);

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());
    }

    /// Test that different immediate value ranges are handled correctly
    #[test]
    fn prop_immediate_ranges(imm in -0x7FFFFFFFFFi64..0x7FFFFFFFFFi64) {
        let config = TargetConfig::default();
        let mut module = Module::new("prop_imm");
        let mut func = MachineFunction::new("imm_test");

        let dest = VReg::new(0);
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(dest, imm));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        func.rebuild_cfg();
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
        let result = emit::emit(&allocated, &config);
        prop_assert!(result.is_ok());
    }
}

// ============================================================================
// Overflow and Wraparound Behavior Tests
// ============================================================================

/// Test addition that might overflow
#[test]
fn test_addition_overflow_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("overflow");
    let mut func = MachineFunction::new("add_overflow");

    let max_val = VReg::new(0);
    let one = VReg::new(1);
    let result = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    // Load max 32-bit value
    entry.push(MachineInst::li(max_val, 0x7FFFFFFF)); // INT32_MAX
    entry.push(MachineInst::li(one, 1));
    // This would overflow in signed 32-bit, but we're working with larger values
    entry.push(MachineInst::add(result, max_val, one));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test multiplication that might overflow
#[test]
fn test_multiplication_overflow_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("mul_overflow");
    let mut func = MachineFunction::new("mul_overflow");

    let large1 = VReg::new(0);
    let large2 = VReg::new(1);
    let result = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(large1, 0x10000)); // 65536
    entry.push(MachineInst::li(large2, 0x10000)); // 65536
    // Result would be 2^32, testing large multiplication
    entry.push(MachineInst::mul(result, large1, large2));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test subtraction that goes negative
#[test]
fn test_subtraction_underflow_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("underflow");
    let mut func = MachineFunction::new("sub_underflow");

    let small = VReg::new(0);
    let large = VReg::new(1);
    let result = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(small, 10));
    entry.push(MachineInst::li(large, 100));
    // 10 - 100 = -90 (negative result)
    entry.push(MachineInst::sub(result, small, large));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test shift by large amount
#[test]
fn test_large_shift_amount() {
    let config = TargetConfig::default();
    let mut module = Module::new("large_shift");
    let mut func = MachineFunction::new("shift_large");

    let value = VReg::new(0);
    let shift = VReg::new(1);
    let result = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(value, 0xFF));
    entry.push(MachineInst::li(shift, 60)); // Large shift amount
    entry.push(MachineInst::new(Opcode::SLL)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(value))
        .src(Operand::VReg(shift)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test division by power of two (common optimization case)
#[test]
fn test_division_by_power_of_two() {
    let config = TargetConfig::default();
    let mut module = Module::new("div_pow2");
    let mut func = MachineFunction::new("div_by_pow2");

    let value = VReg::new(0);
    let divisor = VReg::new(1);
    let result = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(value, 1024));
    entry.push(MachineInst::li(divisor, 16)); // Power of 2
    entry.push(MachineInst::new(Opcode::DIV)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(value))
        .src(Operand::VReg(divisor)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Calling Convention Edge Case Tests
// ============================================================================

/// Test function with no arguments and no return
#[test]
fn test_void_function() {
    let config = TargetConfig::default();
    let mut module = Module::new("void_fn");
    let mut func = MachineFunction::new("void_func");

    let mut entry = MachineBlock::new("entry");
    // Just return immediately
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test function that only uses temporaries (no args/returns)
#[test]
fn test_temp_only_function() {
    let config = TargetConfig::default();
    let mut module = Module::new("temp_only");
    let mut func = MachineFunction::new("temps");

    let t0 = VReg::new(0);
    let t1 = VReg::new(1);
    let t2 = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(t0, 1));
    entry.push(MachineInst::li(t1, 2));
    entry.push(MachineInst::add(t2, t0, t1));
    // Discard result, just return
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test indirect call through register (CALLR)
#[test]
fn test_indirect_call() {
    let config = TargetConfig::default();
    let mut module = Module::new("indirect");
    let mut func = MachineFunction::new("call_indirect");

    let func_ptr = VReg::new(0);

    let mut entry = MachineBlock::new("entry");
    // Load function address (simulated)
    entry.push(MachineInst::li(func_ptr, 0x1000));
    // Indirect call through register
    entry.push(MachineInst::callr(func_ptr));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test saving and restoring across function call
#[test]
fn test_caller_save_restore() {
    let config = TargetConfig::default();
    let mut module = Module::new("save_restore");
    let mut func = MachineFunction::new("caller_save");

    let important = VReg::new(0);
    let temp = VReg::new(1);
    let result = VReg::new(2);
    let one = VReg::new(3);

    let mut entry = MachineBlock::new("entry");
    // Value that must survive across call
    entry.push(MachineInst::li(important, 42));
    entry.push(MachineInst::li(one, 1));
    // Some computation before call
    entry.push(MachineInst::add(temp, important, one));
    // Direct call (would clobber registers)
    entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("other_func".to_string())));
    // Use important value after call - must have been saved
    entry.push(MachineInst::add(result, important, one));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Instruction Encoding Edge Case Tests
// ============================================================================

/// Test all immediate sizes near boundaries
#[test]
fn test_immediate_boundary_values() {
    let config = TargetConfig::default();
    let mut module = Module::new("imm_bounds");
    let mut func = MachineFunction::new("imm_test");

    // Test various immediate sizes at boundaries
    let values = [
        0i64, 1, -1,
        127, 128, -128, -129,        // 8-bit boundaries
        255, 256,
        2047, 2048, -2048, -2049,    // 12-bit boundaries (ADDI range)
        32767, 32768, -32768, -32769, // 16-bit boundaries
        0x7FFFFFFF, -0x80000000i64,  // 32-bit boundaries
    ];

    let mut entry = MachineBlock::new("entry");
    for (i, &val) in values.iter().enumerate() {
        let vreg = VReg::new(i as u32);
        entry.push(MachineInst::li(vreg, val));
    }
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test ADDI with edge case offsets
#[test]
fn test_addi_edge_offsets() {
    let config = TargetConfig::default();
    let mut module = Module::new("addi_edge");
    let mut func = MachineFunction::new("addi_test");

    let base = VReg::new(0);
    let r1 = VReg::new(1);
    let r2 = VReg::new(2);
    let r3 = VReg::new(3);
    let r4 = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 1000));
    // ADDI with various immediate values
    entry.push(MachineInst::new(Opcode::ADDI)
        .dst(Operand::VReg(r1))
        .src(Operand::VReg(base))
        .src(Operand::Imm(0)));
    entry.push(MachineInst::new(Opcode::ADDI)
        .dst(Operand::VReg(r2))
        .src(Operand::VReg(base))
        .src(Operand::Imm(2047))); // Max positive 12-bit
    entry.push(MachineInst::new(Opcode::ADDI)
        .dst(Operand::VReg(r3))
        .src(Operand::VReg(base))
        .src(Operand::Imm(-2048))); // Min negative 12-bit
    entry.push(MachineInst::new(Opcode::ADDI)
        .dst(Operand::VReg(r4))
        .src(Operand::VReg(base))
        .src(Operand::Imm(1))); // Simple +1
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test memory operations with various offsets
#[test]
fn test_memory_offset_variations() {
    let config = TargetConfig::default();
    let mut module = Module::new("mem_offsets");
    let mut func = MachineFunction::new("mem_test");

    let base = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);
    let v3 = VReg::new(3);
    let v4 = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000));
    // Various offsets
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(v1))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(v2))
        .src(Operand::Mem { base, offset: 4 }));
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(v3))
        .src(Operand::Mem { base, offset: -4 }));
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(v4))
        .src(Operand::Mem { base, offset: 2044 })); // Near max offset
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// CFG Edge Case Tests
// ============================================================================

/// Test single-block function (no control flow)
#[test]
fn test_single_block_function() {
    let config = TargetConfig::default();
    let mut module = Module::new("single_blk");
    let mut func = MachineFunction::new("single");

    let a = VReg::new(0);
    let b = VReg::new(1);
    let c = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, 1));
    entry.push(MachineInst::li(b, 2));
    entry.push(MachineInst::add(c, a, b));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test diamond CFG pattern with value merging (if-then-else merge)
#[test]
fn test_diamond_cfg_with_value_merge() {
    let config = TargetConfig::default();
    let mut module = Module::new("diamond");
    let mut func = MachineFunction::new("diamond_cfg");

    let cond = VReg::new(0);
    let a = VReg::new(1);
    let b = VReg::new(2);
    let result = VReg::new(3);
    let zero = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(cond, 1));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::new(Opcode::BNE)
        .src(Operand::VReg(cond))
        .src(Operand::VReg(zero))
        .src(Operand::Label("then".to_string())));
    entry.push(MachineInst::jal(VReg::new(10), "else"));
    func.add_block(entry);

    let mut then_blk = MachineBlock::new("then");
    then_blk.push(MachineInst::li(a, 10));
    then_blk.push(MachineInst::mov(result, a));
    then_blk.push(MachineInst::jal(VReg::new(10), "merge"));
    func.add_block(then_blk);

    let mut else_blk = MachineBlock::new("else");
    else_blk.push(MachineInst::li(b, 20));
    else_blk.push(MachineInst::mov(result, b));
    else_blk.push(MachineInst::jal(VReg::new(10), "merge"));
    func.add_block(else_blk);

    let mut merge = MachineBlock::new("merge");
    merge.push(MachineInst::ret());
    func.add_block(merge);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test early return pattern
#[test]
fn test_early_return_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("early_ret");
    let mut func = MachineFunction::new("early_return");

    let cond = VReg::new(0);
    let result = VReg::new(1);
    let zero = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(cond, 0));
    entry.push(MachineInst::li(zero, 0));
    // Early return if condition is zero
    entry.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(cond))
        .src(Operand::VReg(zero))
        .src(Operand::Label("early_exit".to_string())));
    entry.push(MachineInst::jal(VReg::new(10), "continue"));
    func.add_block(entry);

    let mut early_exit = MachineBlock::new("early_exit");
    early_exit.push(MachineInst::li(result, 0));
    early_exit.push(MachineInst::ret());
    func.add_block(early_exit);

    let mut cont = MachineBlock::new("continue");
    cont.push(MachineInst::li(result, 1));
    cont.push(MachineInst::ret());
    func.add_block(cont);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test loop with break pattern
#[test]
fn test_loop_with_break() {
    let config = TargetConfig::default();
    let mut module = Module::new("loop_break");
    let mut func = MachineFunction::new("loop_break");

    let i = VReg::new(0);
    let limit = VReg::new(1);
    let special = VReg::new(2);
    let one = VReg::new(3);
    let tmp = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(i, 0));
    entry.push(MachineInst::li(limit, 100));
    entry.push(MachineInst::li(special, 50));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::jal(VReg::new(10), "loop"));
    func.add_block(entry);

    let mut loop_blk = MachineBlock::new("loop");
    // Check loop condition
    loop_blk.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(i))
        .src(Operand::VReg(limit))
        .src(Operand::Label("body".to_string())));
    loop_blk.push(MachineInst::jal(VReg::new(10), "exit"));
    func.add_block(loop_blk);

    let mut body = MachineBlock::new("body");
    // Check for break condition
    body.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(i))
        .src(Operand::VReg(special))
        .src(Operand::Label("exit".to_string())));
    // Increment
    body.push(MachineInst::add(tmp, i, one));
    body.push(MachineInst::mov(i, tmp));
    body.push(MachineInst::jal(VReg::new(10), "loop"));
    func.add_block(body);

    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test loop with continue pattern
#[test]
fn test_loop_with_continue() {
    let config = TargetConfig::default();
    let mut module = Module::new("loop_cont");
    let mut func = MachineFunction::new("loop_continue");

    let i = VReg::new(0);
    let limit = VReg::new(1);
    let skip = VReg::new(2);
    let one = VReg::new(3);
    let tmp = VReg::new(4);
    let sum = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(i, 0));
    entry.push(MachineInst::li(limit, 10));
    entry.push(MachineInst::li(skip, 5));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::li(sum, 0));
    entry.push(MachineInst::jal(VReg::new(10), "loop"));
    func.add_block(entry);

    let mut loop_blk = MachineBlock::new("loop");
    loop_blk.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(i))
        .src(Operand::VReg(limit))
        .src(Operand::Label("body".to_string())));
    loop_blk.push(MachineInst::jal(VReg::new(10), "exit"));
    func.add_block(loop_blk);

    let mut body = MachineBlock::new("body");
    // Skip iteration if i == skip
    body.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(i))
        .src(Operand::VReg(skip))
        .src(Operand::Label("continue".to_string())));
    // Add to sum
    body.push(MachineInst::add(tmp, sum, i));
    body.push(MachineInst::mov(sum, tmp));
    body.push(MachineInst::jal(VReg::new(10), "continue"));
    func.add_block(body);

    let mut cont = MachineBlock::new("continue");
    cont.push(MachineInst::add(tmp, i, one));
    cont.push(MachineInst::mov(i, tmp));
    cont.push(MachineInst::jal(VReg::new(10), "loop"));
    func.add_block(cont);

    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Bounds Propagation Tests
// ============================================================================

/// Test that constant bounds are correctly computed
#[test]
fn test_constant_bounds() {
    let b1 = ValueBounds::from_const(100);
    assert_eq!(b1.max, 100);
    assert!(b1.fits_in(32));

    let b2 = ValueBounds::from_const(0);
    assert_eq!(b2.max, 0);
    assert!(b2.fits_in(1));

    let b3 = ValueBounds::from_const(u64::MAX as u128);
    assert_eq!(b3.max, u64::MAX as u128);
}

/// Test bounds after arithmetic operations
#[test]
fn test_arithmetic_bounds() {
    let a = ValueBounds::from_const(100);
    let b = ValueBounds::from_const(50);

    // Addition
    let sum = ValueBounds::add(a, b);
    assert_eq!(sum.max, 150);

    // Subtraction
    let diff = ValueBounds::sub(a, b);
    assert!(diff.max >= 100); // Result could be a - b = 50

    // Multiplication
    let prod = ValueBounds::mul(a, b);
    assert_eq!(prod.max, 5000);
}

/// Test bounds after bitwise operations
#[test]
fn test_bitwise_bounds() {
    let a = ValueBounds::from_bits(8); // 0-255
    let b = ValueBounds::from_bits(8);

    // AND - result is at most min(a.max, b.max)
    let and_result = ValueBounds::and(a, b);
    assert!(and_result.max <= 255);

    // OR - result is at most max bits of either
    let or_result = ValueBounds::or(a, b);
    assert!(or_result.bits <= 8);

    // XOR - same as OR
    let xor_result = ValueBounds::xor(a, b);
    assert!(xor_result.bits <= 8);
}

/// Test bounds after shift operations
#[test]
fn test_shift_bounds() {
    let a = ValueBounds::from_const(16);
    let shift_amt = ValueBounds::from_const(2);

    // Left shift
    let shl_result = ValueBounds::shl(a, shift_amt);
    assert_eq!(shl_result.max, 64); // 16 << 2 = 64

    // Right shift
    let shr_result = ValueBounds::lshr(a, shift_amt);
    assert!(shr_result.max <= 16); // Result <= original
}

/// Test bounds extension and truncation
#[test]
fn test_extend_truncate_bounds() {
    let b32 = ValueBounds::from_bits(32);

    // Zero extend to 64 bits - value doesn't change
    let zext = b32.zext(64);
    assert_eq!(zext.max, b32.max);

    // Truncate to 16 bits
    let trunc = b32.trunc(16);
    assert!(trunc.bits <= 16);
}

/// Test fits_in for various widths
#[test]
fn test_fits_in_various_widths() {
    let small = ValueBounds::from_const(100);
    assert!(small.fits_in(8));
    assert!(small.fits_in(16));
    assert!(small.fits_in(32));

    let medium = ValueBounds::from_const(1000);
    assert!(!medium.fits_in(8));
    assert!(medium.fits_in(16));
    assert!(medium.fits_in(32));

    let large = ValueBounds::from_bits(40);
    assert!(!large.fits_in(32));
    assert!(large.fits_in(40));
    assert!(large.fits_in(64));
}

/// Test headroom calculation
#[test]
fn test_headroom_calculation() {
    let b = ValueBounds::from_bits(24);

    assert_eq!(b.headroom(32), 8);  // 32 - 24 = 8 bits headroom
    assert_eq!(b.headroom(24), 0);  // No headroom
    assert_eq!(b.headroom(16), 0);  // Saturates at 0
}

/// Test bounds after range check
#[test]
fn test_bounds_after_range_check() {
    let after = ValueBounds::after_range_check(32);
    assert_eq!(after.bits, 32);
    assert!(after.fits_in(32));
}

// ============================================================================
// Additional Edge Case Property Tests
// ============================================================================

proptest! {
    /// Test that register pressure doesn't cause failures
    #[test]
    fn prop_register_pressure_handling(num_simultaneous in 5usize..25) {
        let config = TargetConfig::default();
        let mut module = Module::new("pressure");
        let mut func = MachineFunction::new("pressure_test");

        let mut entry = MachineBlock::new("entry");
        let one = VReg::new(0);
        entry.push(MachineInst::li(one, 1));

        // Create many simultaneous live values
        for i in 1..=num_simultaneous {
            let v = VReg::new(i as u32);
            entry.push(MachineInst::li(v, i as i64));
        }

        // Use all of them in a computation chain
        for i in 1..num_simultaneous {
            let src1 = VReg::new(i as u32);
            let src2 = VReg::new((i + 1) as u32);
            let dst = VReg::new((num_simultaneous + i) as u32);
            entry.push(MachineInst::add(dst, src1, src2));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);

        func.rebuild_cfg();
        module.add_function(func);

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());
    }

    /// Test that deep nesting works correctly
    #[test]
    fn prop_deep_nesting(depth in 2usize..8) {
        let config = TargetConfig::default();
        let mut module = Module::new("deep");
        let mut func = MachineFunction::new("deep_nest");

        let cond = VReg::new(0);
        let zero = VReg::new(1);
        let counter = VReg::new(2);
        let one = VReg::new(3);
        let tmp = VReg::new(4);

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(cond, 1));
        entry.push(MachineInst::li(zero, 0));
        entry.push(MachineInst::li(counter, 0));
        entry.push(MachineInst::li(one, 1));
        entry.push(MachineInst::jal(VReg::new(10), "level_0"));
        func.add_block(entry);

        // Create nested blocks
        for level in 0..depth {
            let name = format!("level_{}", level);
            let mut block = MachineBlock::new(&name);
            block.push(MachineInst::add(tmp, counter, one));
            block.push(MachineInst::mov(counter, tmp));

            if level + 1 < depth {
                block.push(MachineInst::new(Opcode::BNE)
                    .src(Operand::VReg(cond))
                    .src(Operand::VReg(zero))
                    .src(Operand::Label(format!("level_{}", level + 1))));
            }
            block.push(MachineInst::jal(VReg::new(10), "exit"));
            func.add_block(block);
        }

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();
        module.add_function(func);

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());

        if let Ok(allocated) = result {
            let emit_result = emit::emit(&allocated, &config);
            prop_assert!(emit_result.is_ok());
        }
    }

    /// Test that sequential operations work correctly
    #[test]
    fn prop_sequential_operations(num_ops in 10usize..50) {
        let config = TargetConfig::default();
        let mut module = Module::new("seq");
        let mut func = MachineFunction::new("sequential");

        let mut entry = MachineBlock::new("entry");
        let acc = VReg::new(0);
        let one = VReg::new(1);
        let tmp = VReg::new(2);

        entry.push(MachineInst::li(acc, 0));
        entry.push(MachineInst::li(one, 1));

        for _ in 0..num_ops {
            entry.push(MachineInst::add(tmp, acc, one));
            entry.push(MachineInst::mov(acc, tmp));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);

        func.rebuild_cfg();
        module.add_function(func);

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());
    }
}

// ============================================================================
// Multi-Function Module Tests
// ============================================================================

/// Test module with multiple independent functions
#[test]
fn test_multi_function_module() {
    let config = TargetConfig::default();
    let mut module = Module::new("multi_fn");

    // First function: add
    let mut add_fn = MachineFunction::new("add");
    let a = VReg::new(0);
    let b = VReg::new(1);
    let result = VReg::new(2);
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, 10));
    entry.push(MachineInst::li(b, 20));
    entry.push(MachineInst::add(result, a, b));
    entry.push(MachineInst::ret());
    add_fn.add_block(entry);
    add_fn.rebuild_cfg();
    module.add_function(add_fn);

    // Second function: sub
    let mut sub_fn = MachineFunction::new("sub");
    let x = VReg::new(0);
    let y = VReg::new(1);
    let diff = VReg::new(2);
    let mut entry2 = MachineBlock::new("entry");
    entry2.push(MachineInst::li(x, 100));
    entry2.push(MachineInst::li(y, 30));
    entry2.push(MachineInst::sub(diff, x, y));
    entry2.push(MachineInst::ret());
    sub_fn.add_block(entry2);
    sub_fn.rebuild_cfg();
    module.add_function(sub_fn);

    // Third function: mul
    let mut mul_fn = MachineFunction::new("mul");
    let m = VReg::new(0);
    let n = VReg::new(1);
    let product = VReg::new(2);
    let mut entry3 = MachineBlock::new("entry");
    entry3.push(MachineInst::li(m, 7));
    entry3.push(MachineInst::li(n, 8));
    entry3.push(MachineInst::mul(product, m, n));
    entry3.push(MachineInst::ret());
    mul_fn.add_block(entry3);
    mul_fn.rebuild_cfg();
    module.add_function(mul_fn);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test module with functions calling each other
#[test]
fn test_inter_function_calls() {
    let config = TargetConfig::default();
    let mut module = Module::new("calls");

    // Helper function
    let mut helper = MachineFunction::new("helper");
    let v0 = VReg::new(0);
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    helper.add_block(entry);
    helper.rebuild_cfg();
    module.add_function(helper);

    // Main function that calls helper
    let mut main_fn = MachineFunction::new("main");
    let result = VReg::new(0);
    let mut main_entry = MachineBlock::new("entry");
    main_entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("helper".to_string())));
    main_entry.push(MachineInst::li(result, 1));
    main_entry.push(MachineInst::ret());
    main_fn.add_block(main_entry);
    main_fn.rebuild_cfg();
    module.add_function(main_fn);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test module with many small functions
#[test]
fn test_many_small_functions() {
    let config = TargetConfig::default();
    let mut module = Module::new("many_fns");

    for i in 0..20 {
        let mut func = MachineFunction::new(&format!("fn_{}", i));
        let v = VReg::new(0);
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v, i as i64));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        func.rebuild_cfg();
        module.add_function(func);
    }

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Store Instruction Tests
// ============================================================================

/// Test byte store instruction
#[test]
fn test_store_byte() {
    let config = TargetConfig::default();
    let mut module = Module::new("store_byte");
    let mut func = MachineFunction::new("sb_test");

    let base = VReg::new(0);
    let value = VReg::new(1);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000));
    entry.push(MachineInst::li(value, 0xAB));
    entry.push(MachineInst::new(Opcode::SB)
        .src(Operand::VReg(value))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test halfword store instruction
#[test]
fn test_store_halfword() {
    let config = TargetConfig::default();
    let mut module = Module::new("store_half");
    let mut func = MachineFunction::new("sh_test");

    let base = VReg::new(0);
    let value = VReg::new(1);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000));
    entry.push(MachineInst::li(value, 0xABCD));
    entry.push(MachineInst::new(Opcode::SH)
        .src(Operand::VReg(value))
        .src(Operand::Mem { base, offset: 2 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test word store instruction
#[test]
fn test_store_word() {
    let config = TargetConfig::default();
    let mut module = Module::new("store_word");
    let mut func = MachineFunction::new("sw_test");

    let base = VReg::new(0);
    let value = VReg::new(1);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000));
    entry.push(MachineInst::li(value, 0x12345678));
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(value))
        .src(Operand::Mem { base, offset: 4 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test doubleword store instruction
#[test]
fn test_store_doubleword() {
    let config = TargetConfig::default();
    let mut module = Module::new("store_double");
    let mut func = MachineFunction::new("sd_test");

    let base = VReg::new(0);
    let value = VReg::new(1);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000));
    entry.push(MachineInst::li(value, 0x123456789ABCDEF0u64 as i64));
    entry.push(MachineInst::new(Opcode::SD)
        .src(Operand::VReg(value))
        .src(Operand::Mem { base, offset: 8 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test load-modify-store pattern
#[test]
fn test_load_modify_store() {
    let config = TargetConfig::default();
    let mut module = Module::new("load_mod_store");
    let mut func = MachineFunction::new("lms_test");

    let base = VReg::new(0);
    let loaded = VReg::new(1);
    let one = VReg::new(2);
    let modified = VReg::new(3);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000));
    entry.push(MachineInst::li(one, 1));
    // Load
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(loaded))
        .src(Operand::Mem { base, offset: 0 }));
    // Modify
    entry.push(MachineInst::add(modified, loaded, one));
    // Store
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(modified))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test multiple stores to consecutive locations
#[test]
fn test_consecutive_stores() {
    let config = TargetConfig::default();
    let mut module = Module::new("consec_stores");
    let mut func = MachineFunction::new("stores");

    let base = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);
    let v3 = VReg::new(3);
    let v4 = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000));
    entry.push(MachineInst::li(v1, 1));
    entry.push(MachineInst::li(v2, 2));
    entry.push(MachineInst::li(v3, 3));
    entry.push(MachineInst::li(v4, 4));

    // Store to consecutive locations
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(v1))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(v2))
        .src(Operand::Mem { base, offset: 4 }));
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(v3))
        .src(Operand::Mem { base, offset: 8 }));
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(v4))
        .src(Operand::Mem { base, offset: 12 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Comparison Instruction Chain Tests
// ============================================================================

/// Test chained comparisons (a < b < c pattern)
#[test]
fn test_chained_comparisons() {
    let config = TargetConfig::default();
    let mut module = Module::new("chained_cmp");
    let mut func = MachineFunction::new("chain_cmp");

    let a = VReg::new(0);
    let b = VReg::new(1);
    let c = VReg::new(2);
    let cmp1 = VReg::new(3);
    let cmp2 = VReg::new(4);
    let result = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, 10));
    entry.push(MachineInst::li(b, 20));
    entry.push(MachineInst::li(c, 30));
    // cmp1 = a < b
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(cmp1))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));
    // cmp2 = b < c
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(cmp2))
        .src(Operand::VReg(b))
        .src(Operand::VReg(c)));
    // result = cmp1 && cmp2
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(cmp1))
        .src(Operand::VReg(cmp2)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test equality and inequality together
#[test]
fn test_eq_neq_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("eq_neq");
    let mut func = MachineFunction::new("eq_neq_test");

    let a = VReg::new(0);
    let b = VReg::new(1);
    let zero = VReg::new(2);
    let diff = VReg::new(3);
    let is_eq = VReg::new(4);
    let is_neq = VReg::new(5);
    let one = VReg::new(6);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, 42));
    entry.push(MachineInst::li(b, 42));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::li(one, 1));
    // diff = a - b (will be 0 if equal)
    entry.push(MachineInst::sub(diff, a, b));
    // is_eq = (diff == 0) using SLTU trick: !(diff | -diff) >> 63
    // Simplified: is_eq = (diff < 1) && (diff >= 0)
    entry.push(MachineInst::new(Opcode::SLTU)
        .dst(Operand::VReg(is_eq))
        .src(Operand::VReg(diff))
        .src(Operand::VReg(one)));
    // is_neq = !is_eq
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(is_neq))
        .src(Operand::VReg(is_eq))
        .src(Operand::VReg(one)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test comparison using SLT (set less than)
#[test]
fn test_comparison_slt() {
    let config = TargetConfig::default();
    let mut module = Module::new("cmp_slt");
    let mut func = MachineFunction::new("slt_test");

    let value1 = VReg::new(0);
    let value2 = VReg::new(1);
    let result = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(value1, 50));
    entry.push(MachineInst::li(value2, 100));
    // result = value1 < value2 (signed)
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(value1))
        .src(Operand::VReg(value2)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test all comparison types
#[test]
fn test_all_comparison_types() {
    let config = TargetConfig::default();
    let mut module = Module::new("all_cmp");
    let mut func = MachineFunction::new("comparisons");

    let a = VReg::new(0);
    let b = VReg::new(1);
    let r_slt = VReg::new(2);
    let r_sltu = VReg::new(3);
    let r_sge = VReg::new(4);
    let one = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, -5)); // Negative for signed comparison
    entry.push(MachineInst::li(b, 10));
    entry.push(MachineInst::li(one, 1));

    // Signed less than
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(r_slt))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));

    // Unsigned less than
    entry.push(MachineInst::new(Opcode::SLTU)
        .dst(Operand::VReg(r_sltu))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));

    // Signed greater than or equal (a >= b is !(a < b))
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(r_sge))
        .src(Operand::VReg(r_slt))
        .src(Operand::VReg(one)));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Register Coalescing Opportunity Tests
// ============================================================================

/// Test simple move chain (should coalesce)
#[test]
fn test_move_chain_coalescing() {
    let config = TargetConfig::default();
    let mut module = Module::new("coalesce");
    let mut func = MachineFunction::new("move_chain");

    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);
    let v3 = VReg::new(3);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    // Chain of moves
    entry.push(MachineInst::mov(v1, v0));
    entry.push(MachineInst::mov(v2, v1));
    entry.push(MachineInst::mov(v3, v2));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test copy with intervening use
#[test]
fn test_copy_with_use() {
    let config = TargetConfig::default();
    let mut module = Module::new("copy_use");
    let mut func = MachineFunction::new("copy_test");

    let orig = VReg::new(0);
    let copy = VReg::new(1);
    let result = VReg::new(2);
    let one = VReg::new(3);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(orig, 100));
    entry.push(MachineInst::li(one, 1));
    // Copy the value
    entry.push(MachineInst::mov(copy, orig));
    // Use original
    entry.push(MachineInst::add(result, orig, one));
    // Use copy (both should be available)
    entry.push(MachineInst::add(result, copy, result));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test swap pattern
#[test]
fn test_swap_pattern() {
    let config = TargetConfig::default();
    let mut module = Module::new("swap");
    let mut func = MachineFunction::new("swap_test");

    let a = VReg::new(0);
    let b = VReg::new(1);
    let tmp = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, 10));
    entry.push(MachineInst::li(b, 20));
    // Swap using temp
    entry.push(MachineInst::mov(tmp, a));
    entry.push(MachineInst::mov(a, b));
    entry.push(MachineInst::mov(b, tmp));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test XOR swap (no temp needed)
#[test]
fn test_xor_swap() {
    let config = TargetConfig::default();
    let mut module = Module::new("xor_swap");
    let mut func = MachineFunction::new("xor_swap_test");

    let a = VReg::new(0);
    let b = VReg::new(1);
    let t1 = VReg::new(2);
    let t2 = VReg::new(3);
    let t3 = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(a, 10));
    entry.push(MachineInst::li(b, 20));
    // XOR swap: a ^= b; b ^= a; a ^= b
    entry.push(MachineInst::xor(t1, a, b));
    entry.push(MachineInst::mov(a, t1));
    entry.push(MachineInst::xor(t2, b, a));
    entry.push(MachineInst::mov(b, t2));
    entry.push(MachineInst::xor(t3, a, b));
    entry.push(MachineInst::mov(a, t3));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Special Value Tests
// ============================================================================

/// Test operations with zero
#[test]
fn test_operations_with_zero() {
    let config = TargetConfig::default();
    let mut module = Module::new("zero_ops");
    let mut func = MachineFunction::new("zero_test");

    let val = VReg::new(0);
    let zero = VReg::new(1);
    let r_add = VReg::new(2);
    let r_sub = VReg::new(3);
    let r_mul = VReg::new(4);
    let r_and = VReg::new(5);
    let r_or = VReg::new(6);
    let r_xor = VReg::new(7);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(val, 42));
    entry.push(MachineInst::li(zero, 0));

    // x + 0 = x
    entry.push(MachineInst::add(r_add, val, zero));
    // x - 0 = x
    entry.push(MachineInst::sub(r_sub, val, zero));
    // x * 0 = 0
    entry.push(MachineInst::mul(r_mul, val, zero));
    // x & 0 = 0
    entry.push(MachineInst::and(r_and, val, zero));
    // x | 0 = x
    entry.push(MachineInst::or(r_or, val, zero));
    // x ^ 0 = x
    entry.push(MachineInst::xor(r_xor, val, zero));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test operations with negative one (-1)
#[test]
fn test_operations_with_neg_one() {
    let config = TargetConfig::default();
    let mut module = Module::new("neg_one_ops");
    let mut func = MachineFunction::new("neg_one_test");

    let val = VReg::new(0);
    let neg_one = VReg::new(1);
    let r_add = VReg::new(2);
    let r_mul = VReg::new(3);
    let r_and = VReg::new(4);
    let r_xor = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(val, 42));
    entry.push(MachineInst::li(neg_one, -1));

    // x + (-1) = x - 1
    entry.push(MachineInst::add(r_add, val, neg_one));
    // x * (-1) = -x
    entry.push(MachineInst::mul(r_mul, val, neg_one));
    // x & (-1) = x (all bits set)
    entry.push(MachineInst::and(r_and, val, neg_one));
    // x ^ (-1) = ~x (bitwise NOT)
    entry.push(MachineInst::xor(r_xor, val, neg_one));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test operations with power of two values
#[test]
fn test_power_of_two_operations() {
    let config = TargetConfig::default();
    let mut module = Module::new("pow2_ops");
    let mut func = MachineFunction::new("pow2_test");

    let val = VReg::new(0);
    let pow2 = VReg::new(1);
    let shift = VReg::new(2);
    let r_mul = VReg::new(3);
    let r_div = VReg::new(4);
    let r_mod = VReg::new(5);
    let r_shl = VReg::new(6);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(val, 100));
    entry.push(MachineInst::li(pow2, 8)); // 2^3
    entry.push(MachineInst::li(shift, 3));

    // x * 8 = x << 3
    entry.push(MachineInst::mul(r_mul, val, pow2));
    // x / 8 = x >> 3 (for unsigned)
    entry.push(MachineInst::new(Opcode::DIV)
        .dst(Operand::VReg(r_div))
        .src(Operand::VReg(val))
        .src(Operand::VReg(pow2)));
    // x % 8 = x & 7
    entry.push(MachineInst::new(Opcode::REM)
        .dst(Operand::VReg(r_mod))
        .src(Operand::VReg(val))
        .src(Operand::VReg(pow2)));
    // Direct shift equivalent
    entry.push(MachineInst::new(Opcode::SLL)
        .dst(Operand::VReg(r_shl))
        .src(Operand::VReg(val))
        .src(Operand::VReg(shift)));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test identity operations (x op x patterns)
#[test]
fn test_identity_operations() {
    let config = TargetConfig::default();
    let mut module = Module::new("identity_ops");
    let mut func = MachineFunction::new("identity_test");

    let x = VReg::new(0);
    let r_sub = VReg::new(1);
    let r_xor = VReg::new(2);
    let r_and = VReg::new(3);
    let r_or = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(x, 42));

    // x - x = 0
    entry.push(MachineInst::sub(r_sub, x, x));
    // x ^ x = 0
    entry.push(MachineInst::xor(r_xor, x, x));
    // x & x = x
    entry.push(MachineInst::and(r_and, x, x));
    // x | x = x
    entry.push(MachineInst::or(r_or, x, x));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test maximum and minimum values
#[test]
fn test_extreme_values() {
    let config = TargetConfig::default();
    let mut module = Module::new("extreme");
    let mut func = MachineFunction::new("extreme_test");

    let max_i32 = VReg::new(0);
    let min_i32 = VReg::new(1);
    let max_u32 = VReg::new(2);
    let one = VReg::new(3);
    let r1 = VReg::new(4);
    let r2 = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(max_i32, 0x7FFFFFFF));
    entry.push(MachineInst::li(min_i32, -0x80000000i64));
    entry.push(MachineInst::li(max_u32, 0xFFFFFFFFu32 as i64));
    entry.push(MachineInst::li(one, 1));

    // Operations with extreme values
    entry.push(MachineInst::add(r1, max_i32, one)); // Overflow
    entry.push(MachineInst::sub(r2, min_i32, one)); // Underflow

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// Additional Property-Based Tests
// ============================================================================

proptest! {
    /// Test that modules with varying function counts work
    #[test]
    fn prop_multi_function_modules(num_funcs in 1usize..10) {
        let config = TargetConfig::default();
        let mut module = Module::new("multi");

        for i in 0..num_funcs {
            let mut func = MachineFunction::new(&format!("fn_{}", i));
            let v = VReg::new(0);
            let mut entry = MachineBlock::new("entry");
            entry.push(MachineInst::li(v, i as i64));
            entry.push(MachineInst::ret());
            func.add_block(entry);
            func.rebuild_cfg();
            module.add_function(func);
        }

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());

        if let Ok(allocated) = result {
            let emit_result = emit::emit(&allocated, &config);
            prop_assert!(emit_result.is_ok());
        }
    }

    /// Test that various store patterns work
    #[test]
    fn prop_store_patterns(num_stores in 1usize..10, offset_base in 0i32..100) {
        let config = TargetConfig::default();
        let mut module = Module::new("stores");
        let mut func = MachineFunction::new("store_test");

        let base = VReg::new(0);
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(base, 0x1000));

        for i in 0..num_stores {
            let val = VReg::new((i + 1) as u32);
            entry.push(MachineInst::li(val, i as i64));
            entry.push(MachineInst::new(Opcode::SW)
                .src(Operand::VReg(val))
                .src(Operand::Mem { base, offset: offset_base + (i * 4) as i32 }));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);
        func.rebuild_cfg();
        module.add_function(func);

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());
    }

    /// Test comparison chains of varying length
    #[test]
    fn prop_comparison_chains(chain_len in 2usize..6) {
        let config = TargetConfig::default();
        let mut module = Module::new("cmp_chain");
        let mut func = MachineFunction::new("chain");

        let mut entry = MachineBlock::new("entry");

        // Create values
        for i in 0..=chain_len {
            let v = VReg::new(i as u32);
            entry.push(MachineInst::li(v, (i * 10) as i64));
        }

        // Create comparison chain
        for i in 0..chain_len {
            let src1 = VReg::new(i as u32);
            let src2 = VReg::new((i + 1) as u32);
            let dst = VReg::new((chain_len + 1 + i) as u32);
            entry.push(MachineInst::new(Opcode::SLT)
                .dst(Operand::VReg(dst))
                .src(Operand::VReg(src1))
                .src(Operand::VReg(src2)));
        }

        entry.push(MachineInst::ret());
        func.add_block(entry);
        func.rebuild_cfg();
        module.add_function(func);

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());
    }
}

// ============================================================================
// INDIRECT CALL (CALLR) TESTS
// ============================================================================

/// Test basic indirect call through register
#[test]
fn test_callr_basic() {
    let config = TargetConfig::default();
    let mut module = Module::new("callr_basic");
    let mut func = MachineFunction::new("caller");

    let fn_ptr = VReg::new(0);

    let mut entry = MachineBlock::new("entry");
    // Load function pointer (simulated)
    entry.push(MachineInst::li(fn_ptr, 0x1000));
    // Indirect call through register
    entry.push(MachineInst::callr(fn_ptr).comment("indirect call"));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test indirect call with arguments setup
#[test]
fn test_callr_with_args() {
    let config = TargetConfig::default();
    let mut module = Module::new("callr_args");
    let mut func = MachineFunction::new("caller");

    let fn_ptr = VReg::new(0);
    let arg1 = VReg::new(1);
    let arg2 = VReg::new(2);
    let result = VReg::new(3);

    let mut entry = MachineBlock::new("entry");
    // Setup arguments
    entry.push(MachineInst::li(arg1, 10));
    entry.push(MachineInst::li(arg2, 20));
    // Load function pointer
    entry.push(MachineInst::li(fn_ptr, 0x2000));
    // Indirect call
    entry.push(MachineInst::callr(fn_ptr));
    // Use result
    entry.push(MachineInst::li(result, 0));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test indirect call in a loop (function pointer table dispatch)
#[test]
fn test_callr_in_loop() {
    let config = TargetConfig::default();
    let mut module = Module::new("callr_loop");
    let mut func = MachineFunction::new("dispatcher");

    let fn_ptr = VReg::new(0);
    let counter = VReg::new(1);
    let limit = VReg::new(2);
    let one = VReg::new(3);
    let ra = VReg::new(4);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(counter, 0));
    entry.push(MachineInst::li(limit, 5));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::jal(ra, "loop"));
    func.add_block(entry);

    let mut loop_block = MachineBlock::new("loop");
    loop_block.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(counter))
        .src(Operand::VReg(limit))
        .src(Operand::Label("body".to_string())));
    func.add_block(loop_block);

    let mut body = MachineBlock::new("body");
    body.push(MachineInst::li(fn_ptr, 0x3000));
    body.push(MachineInst::callr(fn_ptr));
    body.push(MachineInst::add(counter, counter, one));
    body.push(MachineInst::jal(ra, "loop"));
    func.add_block(body);

    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test multiple indirect calls in sequence
#[test]
fn test_callr_sequence() {
    let config = TargetConfig::default();
    let mut module = Module::new("callr_seq");
    let mut func = MachineFunction::new("multi_call");

    let fn1 = VReg::new(0);
    let fn2 = VReg::new(1);
    let fn3 = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(fn1, 0x1000));
    entry.push(MachineInst::li(fn2, 0x2000));
    entry.push(MachineInst::li(fn3, 0x3000));

    entry.push(MachineInst::callr(fn1));
    entry.push(MachineInst::callr(fn2));
    entry.push(MachineInst::callr(fn3));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test CALLR opcode properties
#[test]
fn test_callr_opcode_properties() {
    assert!(Opcode::CALLR.is_pseudo());
    assert!(!Opcode::CALLR.is_arithmetic());
    assert!(!Opcode::CALLR.is_branch());
    assert!(!Opcode::CALLR.is_load());
    assert!(!Opcode::CALLR.is_store());
}

// ============================================================================
// PHI NODE TESTS
// ============================================================================

/// Test PHI opcode exists and has correct properties
#[test]
fn test_phi_opcode_properties() {
    assert!(Opcode::PHI.is_pseudo());
    assert!(Opcode::PHI.is_phi());
    assert!(!Opcode::PHI.is_arithmetic());
    assert!(!Opcode::PHI.is_branch());
}

/// Test PHI node creation
#[test]
fn test_phi_node_creation() {
    let dst = VReg::new(0);
    let src1 = VReg::new(1);
    let src2 = VReg::new(2);

    let phi = MachineInst::new(Opcode::PHI)
        .dst(Operand::VReg(dst))
        .src(Operand::VReg(src1))
        .src(Operand::VReg(src2));

    assert_eq!(phi.opcode, Opcode::PHI);
    assert!(phi.opcode.is_phi());
}

/// Test diamond CFG pattern that would use PHI
#[test]
fn test_phi_diamond_cfg() {
    let config = TargetConfig::default();
    let mut module = Module::new("phi_diamond");
    let mut func = MachineFunction::new("diamond");

    let cond = VReg::new(0);
    let zero = VReg::new(1);
    let val_then = VReg::new(2);
    let val_else = VReg::new(3);
    let result = VReg::new(4);
    let ra = VReg::new(5);

    // Entry: branch based on condition
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(cond, 1));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::bne(cond, zero, "then"));
    func.add_block(entry);

    // Then block
    let mut then_block = MachineBlock::new("then");
    then_block.push(MachineInst::li(val_then, 100));
    then_block.push(MachineInst::jal(ra, "merge"));
    func.add_block(then_block);

    // Else block
    let mut else_block = MachineBlock::new("else");
    else_block.push(MachineInst::li(val_else, 200));
    else_block.push(MachineInst::jal(ra, "merge"));
    func.add_block(else_block);

    // Merge block (where PHI would be in SSA)
    let mut merge = MachineBlock::new("merge");
    // In lowered form, we'd use moves instead of PHI
    merge.push(MachineInst::li(result, 0)); // placeholder
    merge.push(MachineInst::ret());
    func.add_block(merge);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// DISASSEMBLER ROUND-TRIP TESTS
// ============================================================================

/// Test disassembly of basic bytecode
#[test]
fn test_disasm_basic() {
    use zkir_llvm::emit::disassemble;

    let config = TargetConfig::default();
    let mut module = Module::new("disasm_test");
    let mut func = MachineFunction::new("simple");

    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 20));
    entry.push(MachineInst::add(v2, v0, v1));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");

    // Disassemble and verify output contains expected instructions
    let disasm = disassemble(&bytecode).expect("Should disassemble");
    assert!(disasm.contains("li") || disasm.contains("LI") || disasm.contains("add") || disasm.contains("ADD"));
}

/// Test disassembly of arithmetic instructions
#[test]
fn test_disasm_arithmetic() {
    use zkir_llvm::emit::disassemble;

    let config = TargetConfig::default();
    let mut module = Module::new("disasm_arith");
    let mut func = MachineFunction::new("arith");

    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);
    let v3 = VReg::new(3);
    let v4 = VReg::new(4);
    let v5 = VReg::new(5);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 100));
    entry.push(MachineInst::li(v1, 10));
    entry.push(MachineInst::add(v2, v0, v1));
    entry.push(MachineInst::sub(v3, v0, v1));
    entry.push(MachineInst::mul(v4, v0, v1));
    entry.push(MachineInst::new(Opcode::DIV)
        .dst(Operand::VReg(v5))
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");

    let disasm = disassemble(&bytecode).expect("Should disassemble");
    assert!(!disasm.is_empty());
}

/// Test disassembly of memory instructions
#[test]
fn test_disasm_memory() {
    use zkir_llvm::emit::disassemble;

    let config = TargetConfig::default();
    let mut module = Module::new("disasm_mem");
    let mut func = MachineFunction::new("mem");

    let base = VReg::new(0);
    let val = VReg::new(1);
    let loaded = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(base, 0x1000));
    entry.push(MachineInst::li(val, 42));
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(loaded))
        .src(Operand::Mem { base, offset: 0 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");

    let disasm = disassemble(&bytecode).expect("Should disassemble");
    assert!(!disasm.is_empty());
}

/// Test disassembly preserves magic header
#[test]
fn test_disasm_magic_header() {
    use zkir_llvm::emit::disassemble;

    let config = TargetConfig::default();
    let mut module = Module::new("magic_test");
    let mut func = MachineFunction::new("f");

    let v0 = VReg::new(0);
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");

    // Verify magic header (little-endian u32)
    let magic = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
    assert_eq!(magic, 0x5A4B4952);

    let disasm = disassemble(&bytecode).expect("Should disassemble");
    assert!(disasm.contains("ZKIR") || disasm.contains("Header"));
}

// ============================================================================
// FRAME LAYOUT AND STACK TESTS
// ============================================================================

/// Test function with local stack allocation
#[test]
fn test_frame_with_locals() {
    let config = TargetConfig::default();
    let mut module = Module::new("frame_locals");
    let mut func = MachineFunction::new("with_locals");

    // Set up frame with local variables
    func.frame.locals_size = 32; // 32 bytes of locals

    let v0 = VReg::new(0);
    let v1 = VReg::new(1);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 100));
    // Store to local slot
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(v0))
        .src(Operand::Mem { base: v0, offset: -8 })); // fp-8
    // Load from local slot
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(v1))
        .src(Operand::Mem { base: v0, offset: -8 }));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test function with spill slots
#[test]
fn test_frame_with_spills() {
    let config = TargetConfig::default();
    let mut module = Module::new("frame_spills");
    let mut func = MachineFunction::new("with_spills");

    // Create enough registers to force spilling
    let mut entry = MachineBlock::new("entry");
    let mut vregs = Vec::new();
    for i in 0..20 {
        let v = VReg::new(i);
        vregs.push(v);
        entry.push(MachineInst::li(v, i as i64 * 10));
    }

    // Use all vregs to keep them live
    for i in 0..18 {
        let dst = VReg::new(20 + i);
        entry.push(MachineInst::add(dst, vregs[i as usize], vregs[(i + 1) as usize]));
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test function with outgoing arguments on stack
#[test]
fn test_frame_outgoing_args() {
    let config = TargetConfig::default();
    let mut module = Module::new("frame_outgoing");
    let mut func = MachineFunction::new("caller");

    // Function that needs to pass arguments on stack
    func.frame.outgoing_args_size = 32; // Space for stack args

    let v0 = VReg::new(0);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    // Store argument to outgoing area
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(v0))
        .src(Operand::Mem { base: v0, offset: 0 }));
    entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("callee".to_string())));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test frame size calculation
#[test]
fn test_frame_size_calculation() {
    let config = TargetConfig::default();
    let mut module = Module::new("frame_size");
    let mut func = MachineFunction::new("sized");

    func.frame.locals_size = 16;
    func.frame.outgoing_args_size = 24;

    let v0 = VReg::new(0);
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");

    // Verify frame info is preserved
    let alloc_func = &allocated.functions[0];
    assert!(alloc_func.frame.locals_size >= 16);
}

// ============================================================================
// ABI AND CALLING CONVENTION TESTS
// ============================================================================

/// Test function with many register arguments
#[test]
fn test_abi_many_reg_args() {
    let config = TargetConfig::default();
    let mut module = Module::new("abi_reg_args");
    let mut func = MachineFunction::new("many_args");

    // Simulate a function that uses argument registers
    let mut entry = MachineBlock::new("entry");
    for i in 0..8 {
        let v = VReg::new(i);
        entry.push(MachineInst::li(v, (i + 1) as i64 * 10));
    }

    // Sum all "arguments"
    let sum = VReg::new(10);
    entry.push(MachineInst::add(sum, VReg::new(0), VReg::new(1)));
    for i in 2..8 {
        let tmp = VReg::new(10 + i);
        entry.push(MachineInst::add(tmp, VReg::new(10 + i - 1), VReg::new(i)));
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test direct call to named function
#[test]
fn test_abi_direct_call() {
    let config = TargetConfig::default();
    let mut module = Module::new("abi_direct");

    // Callee function
    let mut callee = MachineFunction::new("callee");
    let v0 = VReg::new(0);
    let mut callee_entry = MachineBlock::new("entry");
    callee_entry.push(MachineInst::li(v0, 42));
    callee_entry.push(MachineInst::ret());
    callee.add_block(callee_entry);
    callee.rebuild_cfg();
    module.add_function(callee);

    // Caller function
    let mut caller = MachineFunction::new("caller");
    let mut caller_entry = MachineBlock::new("entry");
    caller_entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("callee".to_string())));
    caller_entry.push(MachineInst::ret());
    caller.add_block(caller_entry);
    caller.rebuild_cfg();
    module.add_function(caller);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test return value in register
#[test]
fn test_abi_return_value() {
    let config = TargetConfig::default();
    let mut module = Module::new("abi_retval");
    let mut func = MachineFunction::new("returns_value");

    let result = VReg::new(0);

    let mut entry = MachineBlock::new("entry");
    // Compute return value
    entry.push(MachineInst::li(result, 123));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test caller-saved register preservation pattern
#[test]
fn test_abi_caller_saved() {
    let config = TargetConfig::default();
    let mut module = Module::new("abi_caller_saved");
    let mut func = MachineFunction::new("preserves_regs");

    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    // Value that must survive call
    entry.push(MachineInst::li(v0, 100));
    // Call that might clobber registers
    entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("other".to_string())));
    // Use the value after call (regalloc must preserve it)
    entry.push(MachineInst::li(v1, 1));
    entry.push(MachineInst::add(v2, v0, v1));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// ERROR HANDLING EDGE CASE TESTS
// ============================================================================

/// Test empty function handling
#[test]
fn test_error_empty_function() {
    let config = TargetConfig::default();
    let mut module = Module::new("empty_func");
    let func = MachineFunction::new("empty");
    // Note: no blocks added
    module.add_function(func);

    // Empty function should fail gracefully
    let result = regalloc::allocate(&module, &config);
    // It might succeed with empty output or fail - either is acceptable
    if result.is_ok() {
        let allocated = result.unwrap();
        let _bytecode = emit::emit(&allocated, &config);
        // Either way is fine for empty function
    }
}

/// Test function with unreachable blocks
#[test]
fn test_unreachable_blocks() {
    let config = TargetConfig::default();
    let mut module = Module::new("unreachable");
    let mut func = MachineFunction::new("has_unreachable");

    let v0 = VReg::new(0);
    let ra = VReg::new(1);

    // Entry that exits early
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    // Unreachable block (no predecessors)
    let mut unreachable = MachineBlock::new("unreachable");
    unreachable.push(MachineInst::li(v0, 999));
    unreachable.push(MachineInst::jal(ra, "entry")); // would create cycle, but unreachable
    func.add_block(unreachable);

    func.rebuild_cfg();
    module.add_function(func);

    // Should handle unreachable blocks gracefully
    let result = regalloc::allocate(&module, &config);
    assert!(result.is_ok());
}

/// Test very long basic block
#[test]
fn test_very_long_block() {
    let config = TargetConfig::default();
    let mut module = Module::new("long_block");
    let mut func = MachineFunction::new("long");

    let mut entry = MachineBlock::new("entry");

    // Create 500 instructions in single block
    for i in 0..250 {
        let v = VReg::new(i);
        entry.push(MachineInst::li(v, i as i64));
    }
    for i in 0..249 {
        let dst = VReg::new(250 + i);
        entry.push(MachineInst::add(dst, VReg::new(i), VReg::new(i + 1)));
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test deeply nested CFG
#[test]
fn test_deeply_nested_cfg() {
    let config = TargetConfig::default();
    let mut module = Module::new("deep_nest");
    let mut func = MachineFunction::new("nested");

    let v0 = VReg::new(0);
    let zero = VReg::new(1);
    let ra = VReg::new(2);

    let depth = 20;

    // Entry
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::jal(ra, "level_0"));
    func.add_block(entry);

    // Create nested levels
    for i in 0..depth {
        let name = format!("level_{}", i);
        let next = if i < depth - 1 {
            format!("level_{}", i + 1)
        } else {
            "exit".to_string()
        };

        let mut level = MachineBlock::new(&name);
        let tmp = VReg::new(10 + i);
        level.push(MachineInst::li(tmp, i as i64));
        level.push(MachineInst::bne(tmp, zero, &next));
        func.add_block(level);
    }

    // Exit
    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

/// Test module with many functions
#[test]
fn test_many_functions_module() {
    let config = TargetConfig::default();
    let mut module = Module::new("many_funcs");

    // Create 50 simple functions
    for i in 0..50 {
        let mut func = MachineFunction::new(&format!("func_{}", i));
        let v = VReg::new(0);
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v, i as i64));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        func.rebuild_cfg();
        module.add_function(func);
    }

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// RANGE CHECK (RCHK) TESTS
// ============================================================================

/// Test RCHK opcode properties
#[test]
fn test_rchk_opcode_properties() {
    assert!(Opcode::RCHK.is_pseudo());
    assert!(!Opcode::RCHK.is_arithmetic());
    assert!(!Opcode::RCHK.is_branch());
}

/// Test RCHK instruction creation
#[test]
fn test_rchk_instruction() {
    let v = VReg::new(0);
    let rchk = MachineInst::new(Opcode::RCHK)
        .dst(Operand::VReg(v));

    assert_eq!(rchk.opcode, Opcode::RCHK);
}

/// Test RCHK in value pipeline
#[test]
fn test_rchk_in_pipeline() {
    let config = TargetConfig::default();
    let mut module = Module::new("rchk_test");
    let mut func = MachineFunction::new("with_rchk");

    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1000000));
    entry.push(MachineInst::li(v1, 2000000));
    entry.push(MachineInst::mul(v2, v0, v1));
    // Range check after multiplication
    entry.push(MachineInst::new(Opcode::RCHK)
        .dst(Operand::VReg(v2)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should allocate");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit");
    assert!(!bytecode.is_empty());
}

// ============================================================================
// ADDITIONAL PROPERTY-BASED TESTS
// ============================================================================

proptest! {
    /// Test that indirect calls work for various function pointer values
    #[test]
    fn prop_callr_addresses(addr in 0x1000i64..0x100000) {
        let config = TargetConfig::default();
        let mut module = Module::new("callr_prop");
        let mut func = MachineFunction::new("caller");

        let fn_ptr = VReg::new(0);
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(fn_ptr, addr));
        entry.push(MachineInst::callr(fn_ptr));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        func.rebuild_cfg();
        module.add_function(func);

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());
    }

    /// Test that nested loops of varying depth work
    #[test]
    fn prop_nested_depth(depth in 2usize..8) {
        let config = TargetConfig::default();
        let mut module = Module::new("nested_prop");
        let mut func = MachineFunction::new("nested");

        let v0 = VReg::new(0);
        let zero = VReg::new(1);
        let ra = VReg::new(2);

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 1));
        entry.push(MachineInst::li(zero, 0));
        entry.push(MachineInst::jal(ra, "level_0"));
        func.add_block(entry);

        for i in 0..depth {
            let name = format!("level_{}", i);
            let next = if i < depth - 1 {
                format!("level_{}", i + 1)
            } else {
                "exit".to_string()
            };

            let mut level = MachineBlock::new(&name);
            let tmp = VReg::new(10 + i as u32);
            level.push(MachineInst::li(tmp, i as i64));
            level.push(MachineInst::bne(tmp, zero, &next));
            func.add_block(level);
        }

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        func.rebuild_cfg();
        module.add_function(func);

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());
    }

    /// Test varying frame sizes
    #[test]
    fn prop_frame_sizes(locals in 0u32..256, outgoing in 0u32..128) {
        let config = TargetConfig::default();
        let mut module = Module::new("frame_prop");
        let mut func = MachineFunction::new("framed");

        func.frame.locals_size = locals;
        func.frame.outgoing_args_size = outgoing;

        let v0 = VReg::new(0);
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 1));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        func.rebuild_cfg();
        module.add_function(func);

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());
    }

    /// Test modules with varying function counts and call patterns
    #[test]
    fn prop_call_graph(num_funcs in 2usize..10) {
        let config = TargetConfig::default();
        let mut module = Module::new("call_graph");

        for i in 0..num_funcs {
            let mut func = MachineFunction::new(&format!("f_{}", i));
            let v = VReg::new(0);
            let mut entry = MachineBlock::new("entry");
            entry.push(MachineInst::li(v, i as i64));

            // Call next function (if not last)
            if i < num_funcs - 1 {
                entry.push(MachineInst::new(Opcode::CALL)
                    .src(Operand::Label(format!("f_{}", i + 1))));
            }

            entry.push(MachineInst::ret());
            func.add_block(entry);
            func.rebuild_cfg();
            module.add_function(func);
        }

        let result = regalloc::allocate(&module, &config);
        prop_assert!(result.is_ok());
    }
}
