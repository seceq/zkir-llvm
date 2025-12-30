//! Comprehensive test suite for ZKIR-LLVM.
//!
//! This module contains:
//! - Configuration matrix tests (all preset combinations)
//! - Property-based testing (proptest fuzzing)
//! - Edge case and boundary value tests
//! - Stress tests for register allocation
//! - End-to-end pipeline tests

use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use zkir_llvm::emit;
use zkir_llvm::mir::{
    MachineBlock, MachineFunction, MachineInst, Module, Opcode, Operand, ValueBounds, VReg,
};
use zkir_llvm::regalloc;
use zkir_llvm::target::TargetConfig;

// =============================================================================
// CONFIGURATION MATRIX TESTS
// =============================================================================

/// All valid limb bit sizes (must be even, 16-30).
const VALID_LIMB_BITS: [u8; 8] = [16, 18, 20, 22, 24, 26, 28, 30];

/// All valid data limb counts.
const VALID_DATA_LIMBS: [u8; 4] = [1, 2, 3, 4];

/// All valid address limb counts.
const VALID_ADDR_LIMBS: [u8; 2] = [1, 2];

/// Test all preset configurations compile and emit correctly.
#[test]
fn test_all_presets_compile() {
    let presets = TargetConfig::preset_names();

    for preset_name in presets {
        let config = TargetConfig::preset(preset_name).expect("Preset should exist");
        assert!(config.validate().is_ok(), "Preset {} should be valid", preset_name);

        // Create a simple test function
        let mut module = Module::new("test");
        let mut func = MachineFunction::new("test_fn");
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

        // Allocate and emit
        let allocated = regalloc::allocate(&module, &config)
            .expect(&format!("Allocation should succeed for preset {}", preset_name));
        let bytecode = emit::emit(&allocated, &config)
            .expect(&format!("Emit should succeed for preset {}", preset_name));

        assert!(bytecode.len() >= 32, "Bytecode should have header for {}", preset_name);
        // zkir-spec v3.4: Magic as u32 LE at bytes 0-3
        let magic = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
        assert_eq!(magic, 0x5A4B4952, "Magic number for {}", preset_name);
    }
}

/// Test configuration matrix - all valid limb × data_limbs × addr_limbs combinations.
#[test]
fn test_configuration_matrix() {
    let mut tested = 0;

    for &limb_bits in &VALID_LIMB_BITS {
        for &data_limbs in &VALID_DATA_LIMBS {
            for &addr_limbs in &VALID_ADDR_LIMBS {
                let config = TargetConfig { limb_bits, data_limbs, addr_limbs };

                // All combinations should be valid
                assert!(
                    config.validate().is_ok(),
                    "Config {}×{}×{} should be valid",
                    limb_bits,
                    data_limbs,
                    addr_limbs
                );

                // Verify computed values
                assert_eq!(
                    config.data_bits(),
                    limb_bits as u32 * data_limbs as u32,
                    "data_bits for {}×{}",
                    limb_bits,
                    data_limbs
                );
                assert_eq!(
                    config.addr_bits(),
                    limb_bits as u32 * addr_limbs as u32,
                    "addr_bits for {}×{}",
                    limb_bits,
                    addr_limbs
                );
                assert_eq!(
                    config.chunk_bits(),
                    limb_bits as u32 / 2,
                    "chunk_bits for {}",
                    limb_bits
                );
                assert_eq!(
                    config.table_size(),
                    1 << (limb_bits / 2),
                    "table_size for {}",
                    limb_bits
                );

                // Compile a simple function
                let mut module = Module::new("matrix_test");
                let mut func = MachineFunction::new("f");
                let v0 = func.new_vreg();
                let mut entry = MachineBlock::new("entry");
                entry.push(MachineInst::li(v0, 1));
                entry.push(MachineInst::ret());
                func.add_block(entry);
                module.add_function(func);

                let allocated = regalloc::allocate(&module, &config)
                    .expect(&format!("Alloc failed for {}×{}×{}", limb_bits, data_limbs, addr_limbs));
                let bytecode = emit::emit(&allocated, &config)
                    .expect(&format!("Emit failed for {}×{}×{}", limb_bits, data_limbs, addr_limbs));

                assert!(!bytecode.is_empty());
                tested += 1;
            }
        }
    }

    // Should test 8 × 4 × 2 = 64 configurations
    assert_eq!(tested, 64);
}

/// Test invalid configurations are rejected.
#[test]
fn test_invalid_configurations() {
    // Odd limb bits
    for bits in [15, 17, 19, 21, 23, 25, 27, 29, 31] {
        let config = TargetConfig { limb_bits: bits, data_limbs: 2, addr_limbs: 2 };
        assert!(config.validate().is_err(), "Odd limb_bits {} should be invalid", bits);
    }

    // Out of range limb bits
    for bits in [0, 2, 4, 6, 8, 10, 12, 14, 32, 34, 36, 40, 50, 64] {
        let config = TargetConfig { limb_bits: bits, data_limbs: 2, addr_limbs: 2 };
        assert!(config.validate().is_err(), "limb_bits {} should be invalid", bits);
    }

    // Invalid data_limbs
    for limbs in [0, 5, 6, 7, 8, 10, 100] {
        let config = TargetConfig { limb_bits: 20, data_limbs: limbs, addr_limbs: 2 };
        assert!(config.validate().is_err(), "data_limbs {} should be invalid", limbs);
    }

    // Invalid addr_limbs
    for limbs in [0, 3, 4, 5, 10] {
        let config = TargetConfig { limb_bits: 20, data_limbs: 2, addr_limbs: limbs };
        assert!(config.validate().is_err(), "addr_limbs {} should be invalid", limbs);
    }
}

/// Test configuration warnings are generated appropriately.
#[test]
fn test_configuration_warnings() {
    // 40-bit config should warn about i64 splitting
    let config_40 = TargetConfig::DEFAULT;
    let warnings = config_40.check_warnings();
    assert!(
        warnings.iter().any(|w| format!("{}", w).contains("i64")),
        "40-bit config should warn about i64"
    );

    // 80-bit config should NOT warn about i64 splitting
    let config_80 = TargetConfig::DATA_80;
    let warnings = config_80.check_warnings();
    assert!(
        !warnings.iter().any(|w| format!("{}", w).contains("i64") && format!("{}", w).contains("split")),
        "80-bit config should not warn about i64 splitting"
    );

    // Very small config should warn about low headroom
    let config_small = TargetConfig { limb_bits: 16, data_limbs: 2, addr_limbs: 1 };
    let warnings = config_small.check_warnings();
    assert!(
        warnings.iter().any(|w| format!("{}", w).contains("headroom")),
        "32-bit config should warn about low headroom"
    );
}

// =============================================================================
// VALUE BOUNDS PROPERTY TESTS
// =============================================================================

proptest! {
    /// Property: from_const produces bounds where max equals the input.
    #[test]
    fn prop_bounds_from_const(value in 0u128..=u64::MAX as u128) {
        let bounds = ValueBounds::from_const(value);
        prop_assert_eq!(bounds.max, value);
        prop_assert!(bounds.bits >= 1);
        prop_assert!(bounds.bits <= 128);
    }

    /// Property: from_bits produces correct max value.
    #[test]
    fn prop_bounds_from_bits(bits in 0u32..=128) {
        let bounds = ValueBounds::from_bits(bits);
        if bits == 0 {
            prop_assert_eq!(bounds.max, 0);
        } else if bits >= 128 {
            prop_assert_eq!(bounds.max, u128::MAX);
        } else {
            prop_assert_eq!(bounds.max, (1u128 << bits) - 1);
        }
    }

    /// Property: addition bounds are sound (a.max + b.max >= actual result).
    #[test]
    fn prop_bounds_add_sound(a_val in 0u64..=u32::MAX as u64, b_val in 0u64..=u32::MAX as u64) {
        let a = ValueBounds::from_const(a_val as u128);
        let b = ValueBounds::from_const(b_val as u128);
        let result = ValueBounds::add(a, b);

        // Result max should be >= actual sum
        let actual_sum = a_val as u128 + b_val as u128;
        prop_assert!(result.max >= actual_sum);
    }

    /// Property: multiplication bounds are sound.
    #[test]
    fn prop_bounds_mul_sound(a_val in 0u64..=u16::MAX as u64, b_val in 0u64..=u16::MAX as u64) {
        let a = ValueBounds::from_const(a_val as u128);
        let b = ValueBounds::from_const(b_val as u128);
        let result = ValueBounds::mul(a, b);

        let actual_product = a_val as u128 * b_val as u128;
        prop_assert!(result.max >= actual_product);
    }

    /// Property: AND bounds never exceed the smaller input.
    #[test]
    fn prop_bounds_and_sound(a_val in 0u128..=u64::MAX as u128, b_val in 0u128..=u64::MAX as u128) {
        let a = ValueBounds::from_const(a_val);
        let b = ValueBounds::from_const(b_val);
        let result = ValueBounds::and(a, b);

        prop_assert!(result.max <= a.max);
        prop_assert!(result.max <= b.max);
    }

    /// Property: fits_in is monotonic (more bits = more capacity).
    #[test]
    fn prop_fits_in_monotonic(value_bits in 1u32..=64, data_bits1 in 1u32..=128, data_bits2 in 1u32..=128) {
        let bounds = ValueBounds::from_bits(value_bits);

        if data_bits1 <= data_bits2 {
            // If it fits in smaller, it fits in larger
            if bounds.fits_in(data_bits1) {
                prop_assert!(bounds.fits_in(data_bits2));
            }
        }
    }

    /// Property: headroom is consistent with fits_in.
    #[test]
    fn prop_headroom_consistent(value_bits in 1u32..=64, data_bits in 1u32..=128) {
        let bounds = ValueBounds::from_bits(value_bits);
        let headroom = bounds.headroom(data_bits);

        if value_bits <= data_bits {
            prop_assert_eq!(headroom, data_bits - value_bits);
            prop_assert!(bounds.fits_in(data_bits));
        } else {
            prop_assert_eq!(headroom, 0);
            prop_assert!(!bounds.fits_in(data_bits));
        }
    }

    /// Property: truncation reduces bits.
    #[test]
    fn prop_trunc_reduces(value_bits in 1u32..=64, trunc_bits in 1u32..=64) {
        let bounds = ValueBounds::from_bits(value_bits);
        let truncated = bounds.trunc(trunc_bits);

        prop_assert!(truncated.bits <= bounds.bits.max(trunc_bits));
    }

    /// Property: zero extension preserves value.
    #[test]
    fn prop_zext_preserves(value_bits in 1u32..=64, ext_bits in 1u32..=128) {
        let bounds = ValueBounds::from_bits(value_bits);
        let extended = bounds.zext(ext_bits);

        // Zero extension preserves the actual value/bounds
        prop_assert_eq!(extended.max, bounds.max);
        prop_assert_eq!(extended.bits, bounds.bits);
    }
}

// =============================================================================
// INSTRUCTION ENCODING PROPERTY TESTS
// =============================================================================

/// Helper to create a simple function with given instructions.
fn create_test_function(name: &str, insts: Vec<MachineInst>) -> MachineFunction {
    let mut func = MachineFunction::new(name);
    let mut entry = MachineBlock::new("entry");
    for inst in insts {
        entry.push(inst);
    }
    entry.push(MachineInst::ret());
    func.add_block(entry);
    func
}

proptest! {
    /// Property: LI instruction encodes any valid immediate.
    #[test]
    fn prop_li_encodes(imm in -1000000i64..=1000000) {
        let config = TargetConfig::default();
        let mut module = Module::new("test");

        let mut func = MachineFunction::new("f");
        let v0 = func.new_vreg();
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, imm));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config).map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config).map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(!bytecode.is_empty());
    }

    /// Property: ADDI instruction with valid offset encodes.
    #[test]
    fn prop_addi_encodes(offset in -2048i64..=2047) {
        let config = TargetConfig::default();
        let mut module = Module::new("test");

        let mut func = MachineFunction::new("f");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 100));
        entry.push(MachineInst::new(Opcode::ADDI)
            .dst(Operand::VReg(v1))
            .src(Operand::VReg(v0))
            .src(Operand::Imm(offset)));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config).map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config).map_err(|e| TestCaseError::fail(e.to_string()))?;
        prop_assert!(!bytecode.is_empty());
    }

    /// Property: Bytecode always starts with ZKIR magic.
    #[test]
    fn prop_magic_header(num_insts in 1usize..10) {
        let config = TargetConfig::default();
        let mut module = Module::new("test");

        let mut func = MachineFunction::new("f");
        let vregs: Vec<_> = (0..num_insts + 1).map(|_| func.new_vreg()).collect();

        let mut entry = MachineBlock::new("entry");
        for i in 0..num_insts {
            entry.push(MachineInst::li(vregs[i], i as i64));
        }
        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config).map_err(|e| TestCaseError::fail(e.to_string()))?;
        let bytecode = emit::emit(&allocated, &config).map_err(|e| TestCaseError::fail(e.to_string()))?;

        let magic = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
        prop_assert_eq!(magic, 0x5A4B4952);
    }
}

// =============================================================================
// REGISTER ALLOCATION STRESS TESTS
// =============================================================================

/// Test register allocation with maximum register pressure.
#[test]
fn test_regalloc_max_pressure() {
    let config = TargetConfig::default();
    let mut module = Module::new("pressure_test");
    let mut func = MachineFunction::new("high_pressure");

    // Create many live values simultaneously (more than available registers)
    let num_values = 50; // More than the 15 allocatable registers
    let vregs: Vec<_> = (0..num_values).map(|_| func.new_vreg()).collect();

    let mut entry = MachineBlock::new("entry");

    // Load all values
    for (i, &vreg) in vregs.iter().enumerate() {
        entry.push(MachineInst::li(vreg, i as i64));
    }

    // Use all values (keeps them all live)
    let sum_vreg = func.new_vreg();
    entry.push(MachineInst::li(sum_vreg, 0));

    for &vreg in &vregs {
        let new_sum = func.new_vreg();
        entry.push(MachineInst::add(new_sum, sum_vreg, vreg));
        // Note: we're creating a chain, but each old vreg is still live for potential use
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    // Should handle spilling gracefully
    let allocated = regalloc::allocate(&module, &config).expect("Allocation should succeed");
    let bytecode = emit::emit(&allocated, &config).expect("Emit should succeed");
    assert!(!bytecode.is_empty());
}

/// Test register allocation with complex control flow.
#[test]
fn test_regalloc_complex_cfg() {
    let config = TargetConfig::default();
    let mut module = Module::new("cfg_test");
    let mut func = MachineFunction::new("complex_cfg");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v_zero = func.new_vreg();
    let v_result = func.new_vreg();
    let ra = func.new_vreg(); // Return address register

    // Entry block
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 20));
    entry.push(MachineInst::li(v_zero, 0));
    entry.push(MachineInst::bne(v0, v_zero, "then1"));
    func.add_block(entry);

    // Then block 1 - nested condition
    let mut then1 = MachineBlock::new("then1");
    then1.push(MachineInst::add(v2, v0, v1));
    then1.push(MachineInst::bne(v2, v_zero, "then2"));
    func.add_block(then1);

    // Then block 2
    let mut then2 = MachineBlock::new("then2");
    then2.push(MachineInst::mul(v_result, v2, v0));
    then2.push(MachineInst::jal(ra, "exit"));
    func.add_block(then2);

    // Else block 1
    let mut else1 = MachineBlock::new("else1");
    else1.push(MachineInst::li(v_result, 0));
    else1.push(MachineInst::jal(ra, "exit"));
    func.add_block(else1);

    // Exit block
    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Allocation should succeed");
    let bytecode = emit::emit(&allocated, &config).expect("Emit should succeed");
    assert!(!bytecode.is_empty());
}

/// Test nested loop register allocation.
#[test]
fn test_regalloc_nested_loops() {
    let config = TargetConfig::default();
    let mut module = Module::new("nested_loops");
    let mut func = MachineFunction::new("nested");

    let i = func.new_vreg();
    let j = func.new_vreg();
    let sum = func.new_vreg();
    let limit = func.new_vreg();
    let one = func.new_vreg();
    let tmp = func.new_vreg();
    let ra = func.new_vreg(); // Return address register

    // Entry
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(i, 0));
    entry.push(MachineInst::li(sum, 0));
    entry.push(MachineInst::li(limit, 10));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::jal(ra, "outer_loop"));
    func.add_block(entry);

    // Outer loop header
    let mut outer_header = MachineBlock::new("outer_loop");
    outer_header.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(i))
        .src(Operand::VReg(limit))
        .src(Operand::Label("outer_body".to_string())));
    func.add_block(outer_header);

    // Outer loop body - initialize inner loop
    let mut outer_body = MachineBlock::new("outer_body");
    outer_body.push(MachineInst::li(j, 0));
    outer_body.push(MachineInst::jal(ra, "inner_loop"));
    func.add_block(outer_body);

    // Inner loop header
    let mut inner_header = MachineBlock::new("inner_loop");
    inner_header.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(j))
        .src(Operand::VReg(limit))
        .src(Operand::Label("inner_body".to_string())));
    func.add_block(inner_header);

    // Inner loop body
    let mut inner_body = MachineBlock::new("inner_body");
    inner_body.push(MachineInst::add(tmp, sum, i));
    inner_body.push(MachineInst::add(sum, tmp, j));
    inner_body.push(MachineInst::add(j, j, one));
    inner_body.push(MachineInst::jal(ra, "inner_loop"));
    func.add_block(inner_body);

    // Inner loop exit
    let mut inner_exit = MachineBlock::new("inner_exit");
    inner_exit.push(MachineInst::add(i, i, one));
    inner_exit.push(MachineInst::jal(ra, "outer_loop"));
    func.add_block(inner_exit);

    // Exit
    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Allocation should succeed");
    let bytecode = emit::emit(&allocated, &config).expect("Emit should succeed");
    assert!(!bytecode.is_empty());
}

/// Test with all instruction types.
#[test]
fn test_all_instruction_types() {
    let config = TargetConfig::default();
    let mut module = Module::new("all_insts");
    let mut func = MachineFunction::new("all_types");

    let v: Vec<_> = (0..20).map(|_| func.new_vreg()).collect();

    let mut entry = MachineBlock::new("entry");

    // Arithmetic
    entry.push(MachineInst::li(v[0], 100));
    entry.push(MachineInst::li(v[1], 50));
    entry.push(MachineInst::add(v[2], v[0], v[1]));
    entry.push(MachineInst::sub(v[3], v[0], v[1]));
    entry.push(MachineInst::mul(v[4], v[0], v[1]));
    entry.push(MachineInst::new(Opcode::DIV)
        .dst(Operand::VReg(v[5]))
        .src(Operand::VReg(v[0]))
        .src(Operand::VReg(v[1])));
    entry.push(MachineInst::new(Opcode::REM)
        .dst(Operand::VReg(v[6]))
        .src(Operand::VReg(v[0]))
        .src(Operand::VReg(v[1])));

    // Logical
    entry.push(MachineInst::and(v[7], v[0], v[1]));
    entry.push(MachineInst::or(v[8], v[0], v[1]));
    entry.push(MachineInst::xor(v[9], v[0], v[1]));
    entry.push(MachineInst::new(Opcode::NOT)
        .dst(Operand::VReg(v[10]))
        .src(Operand::VReg(v[0])));

    // Shifts
    entry.push(MachineInst::new(Opcode::SLL)
        .dst(Operand::VReg(v[11]))
        .src(Operand::VReg(v[0]))
        .src(Operand::VReg(v[1])));
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(v[12]))
        .src(Operand::VReg(v[0]))
        .src(Operand::VReg(v[1])));
    entry.push(MachineInst::new(Opcode::SRA)
        .dst(Operand::VReg(v[13]))
        .src(Operand::VReg(v[0]))
        .src(Operand::VReg(v[1])));

    // Comparison
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(v[14]))
        .src(Operand::VReg(v[0]))
        .src(Operand::VReg(v[1])));
    entry.push(MachineInst::new(Opcode::SLTU)
        .dst(Operand::VReg(v[15]))
        .src(Operand::VReg(v[0]))
        .src(Operand::VReg(v[1])));
    entry.push(MachineInst::new(Opcode::SEQ)
        .dst(Operand::VReg(v[16]))
        .src(Operand::VReg(v[0]))
        .src(Operand::VReg(v[1])));
    entry.push(MachineInst::new(Opcode::SNE)
        .dst(Operand::VReg(v[17]))
        .src(Operand::VReg(v[0]))
        .src(Operand::VReg(v[1])));

    // Conditional move
    entry.push(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(v[18]))
        .src(Operand::VReg(v[0]))
        .src(Operand::VReg(v[1]))
        .src(Operand::VReg(v[2])));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Allocation should succeed");
    let bytecode = emit::emit(&allocated, &config).expect("Emit should succeed");
    assert!(!bytecode.is_empty());
}

// =============================================================================
// EDGE CASE AND BOUNDARY TESTS
// =============================================================================

/// Test empty function.
#[test]
fn test_empty_function() {
    let config = TargetConfig::default();
    let mut module = Module::new("empty");
    let mut func = MachineFunction::new("empty_fn");
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Should handle empty function");
    let bytecode = emit::emit(&allocated, &config).expect("Should emit empty function");
    assert!(!bytecode.is_empty());
}

/// Test function with single NOP.
#[test]
fn test_single_nop() {
    let config = TargetConfig::default();
    let mut module = Module::new("nop");
    let mut func = MachineFunction::new("nop_fn");
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::nop());
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test maximum immediate values.
#[test]
fn test_max_immediate_values() {
    let config = TargetConfig::default();
    let mut module = Module::new("max_imm");
    let mut func = MachineFunction::new("max_imm_fn");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v3 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");

    // Various extreme values
    entry.push(MachineInst::li(v0, 0));
    entry.push(MachineInst::li(v1, -1));
    entry.push(MachineInst::li(v2, i32::MAX as i64));
    entry.push(MachineInst::li(v3, i32::MIN as i64));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test boundary immediate values for ADDI.
#[test]
fn test_addi_boundary_values() {
    let config = TargetConfig::default();
    let mut module = Module::new("addi_bounds");
    let mut func = MachineFunction::new("addi_fn");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v3 = func.new_vreg();
    let v4 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1000));

    // ADDI boundary values (-2048 to 2047 for 12-bit signed)
    entry.push(MachineInst::new(Opcode::ADDI)
        .dst(Operand::VReg(v1))
        .src(Operand::VReg(v0))
        .src(Operand::Imm(0)));
    entry.push(MachineInst::new(Opcode::ADDI)
        .dst(Operand::VReg(v2))
        .src(Operand::VReg(v0))
        .src(Operand::Imm(2047)));
    entry.push(MachineInst::new(Opcode::ADDI)
        .dst(Operand::VReg(v3))
        .src(Operand::VReg(v0))
        .src(Operand::Imm(-2048)));
    entry.push(MachineInst::new(Opcode::ADDI)
        .dst(Operand::VReg(v4))
        .src(Operand::VReg(v0))
        .src(Operand::Imm(1)));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test all comparison opcodes.
#[test]
fn test_all_comparisons() {
    let config = TargetConfig::default();
    let mut module = Module::new("comparisons");
    let mut func = MachineFunction::new("cmp_fn");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let results: Vec<_> = (0..6).map(|_| func.new_vreg()).collect();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 20));

    // All comparison types
    entry.push(MachineInst::new(Opcode::SLT).dst(Operand::VReg(results[0]))
        .src(Operand::VReg(v0)).src(Operand::VReg(v1)));
    entry.push(MachineInst::new(Opcode::SLTU).dst(Operand::VReg(results[1]))
        .src(Operand::VReg(v0)).src(Operand::VReg(v1)));
    entry.push(MachineInst::new(Opcode::SEQ).dst(Operand::VReg(results[2]))
        .src(Operand::VReg(v0)).src(Operand::VReg(v1)));
    entry.push(MachineInst::new(Opcode::SNE).dst(Operand::VReg(results[3]))
        .src(Operand::VReg(v0)).src(Operand::VReg(v1)));
    // SGE and SGEU are the zkir-spec comparison variants
    entry.push(MachineInst::new(Opcode::SGE).dst(Operand::VReg(results[4]))
        .src(Operand::VReg(v0)).src(Operand::VReg(v1)));
    entry.push(MachineInst::new(Opcode::SGEU).dst(Operand::VReg(results[5]))
        .src(Operand::VReg(v0)).src(Operand::VReg(v1)));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test all branch types.
#[test]
fn test_all_branch_types() {
    let config = TargetConfig::default();
    let mut module = Module::new("branches");
    let mut func = MachineFunction::new("branch_fn");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 20));
    entry.push(MachineInst::beq(v0, v1, "beq_target"));
    func.add_block(entry);

    let mut beq_target = MachineBlock::new("beq_target");
    beq_target.push(MachineInst::bne(v0, v1, "bne_target"));
    func.add_block(beq_target);

    let mut bne_target = MachineBlock::new("bne_target");
    bne_target.push(MachineInst::new(Opcode::BLT)
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1))
        .src(Operand::Label("blt_target".to_string())));
    func.add_block(bne_target);

    let mut blt_target = MachineBlock::new("blt_target");
    blt_target.push(MachineInst::new(Opcode::BGE)
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1))
        .src(Operand::Label("bge_target".to_string())));
    func.add_block(blt_target);

    let mut bge_target = MachineBlock::new("bge_target");
    bge_target.push(MachineInst::new(Opcode::BLTU)
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1))
        .src(Operand::Label("bltu_target".to_string())));
    func.add_block(bge_target);

    let mut bltu_target = MachineBlock::new("bltu_target");
    bltu_target.push(MachineInst::new(Opcode::BGEU)
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1))
        .src(Operand::Label("exit".to_string())));
    func.add_block(bltu_target);

    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test memory operations.
#[test]
fn test_memory_operations() {
    let config = TargetConfig::default();
    let mut module = Module::new("memory");
    let mut func = MachineFunction::new("mem_fn");

    let ptr = func.new_vreg();
    let val = func.new_vreg();
    let loaded: Vec<_> = (0..6).map(|_| func.new_vreg()).collect();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(ptr, 0x1000)); // Fake pointer
    entry.push(MachineInst::li(val, 0x42));

    // Store operations (different sizes)
    entry.push(MachineInst::new(Opcode::SB)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base: ptr, offset: 0 }));
    entry.push(MachineInst::new(Opcode::SH)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base: ptr, offset: 4 }));
    entry.push(MachineInst::new(Opcode::SW)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base: ptr, offset: 8 }));
    entry.push(MachineInst::new(Opcode::SD)
        .src(Operand::VReg(val))
        .src(Operand::Mem { base: ptr, offset: 16 }));

    // Load operations (different sizes, signed and unsigned)
    entry.push(MachineInst::new(Opcode::LB)
        .dst(Operand::VReg(loaded[0]))
        .src(Operand::Mem { base: ptr, offset: 0 }));
    entry.push(MachineInst::new(Opcode::LBU)
        .dst(Operand::VReg(loaded[1]))
        .src(Operand::Mem { base: ptr, offset: 0 }));
    entry.push(MachineInst::new(Opcode::LH)
        .dst(Operand::VReg(loaded[2]))
        .src(Operand::Mem { base: ptr, offset: 4 }));
    entry.push(MachineInst::new(Opcode::LHU)
        .dst(Operand::VReg(loaded[3]))
        .src(Operand::Mem { base: ptr, offset: 4 }));
    entry.push(MachineInst::new(Opcode::LW)
        .dst(Operand::VReg(loaded[4]))
        .src(Operand::Mem { base: ptr, offset: 8 }));
    entry.push(MachineInst::new(Opcode::LD)
        .dst(Operand::VReg(loaded[5]))
        .src(Operand::Mem { base: ptr, offset: 16 }));

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test multiple functions in module.
#[test]
fn test_multiple_functions() {
    let config = TargetConfig::default();
    let mut module = Module::new("multi");

    // Function 1
    let mut func1 = MachineFunction::new("func1");
    let v0 = func1.new_vreg();
    let mut entry1 = MachineBlock::new("entry");
    entry1.push(MachineInst::li(v0, 1));
    entry1.push(MachineInst::ret());
    func1.add_block(entry1);
    module.add_function(func1);

    // Function 2
    let mut func2 = MachineFunction::new("func2");
    let v1 = func2.new_vreg();
    let mut entry2 = MachineBlock::new("entry");
    entry2.push(MachineInst::li(v1, 2));
    entry2.push(MachineInst::ret());
    func2.add_block(entry2);
    module.add_function(func2);

    // Function 3
    let mut func3 = MachineFunction::new("func3");
    let v2 = func3.new_vreg();
    let mut entry3 = MachineBlock::new("entry");
    entry3.push(MachineInst::li(v2, 3));
    entry3.push(MachineInst::ret());
    func3.add_block(entry3);
    module.add_function(func3);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // All functions should be in output
    let asm = emit::format_asm(&allocated);
    assert!(asm.contains("func1:"));
    assert!(asm.contains("func2:"));
    assert!(asm.contains("func3:"));
}

/// Test function with global variable.
#[test]
fn test_with_global_variable() {
    use zkir_llvm::mir::GlobalVar;

    let config = TargetConfig::default();
    let mut module = Module::new("globals");

    // Add global variable
    module.globals.insert(
        "my_global".to_string(),
        GlobalVar {
            name: "my_global".to_string(),
            size: 8,
            align: 4,
            init: Some(vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]),
            is_const: false,
        },
    );

    // Add function that uses the global
    let mut func = MachineFunction::new("use_global");
    let v0 = func.new_vreg();
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    let asm = emit::format_asm(&module);
    assert!(asm.contains("my_global"));
    assert!(asm.contains(".data"));
}

/// Test indirect call (CALLR).
#[test]
fn test_indirect_call() {
    let config = TargetConfig::default();
    let mut module = Module::new("indirect_call");
    let mut func = MachineFunction::new("caller");

    let fn_ptr = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(fn_ptr, 0x1000)); // Fake function pointer
    entry.push(MachineInst::callr(fn_ptr));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    let asm = emit::format_asm(&allocated);
    assert!(asm.contains("callr") || asm.contains("jalr"));
}

// =============================================================================
// OPCODE PROPERTY TESTS
// =============================================================================

proptest! {
    /// Property: All arithmetic opcodes are correctly classified.
    #[test]
    fn prop_opcode_arithmetic_classification(opcode_byte in 0u8..=255) {
        // Test that classification is consistent
        let _ = opcode_byte; // Use the value even if test is simple

        // Verify known arithmetic opcodes
        prop_assert!(Opcode::ADD.is_arithmetic());
        prop_assert!(Opcode::SUB.is_arithmetic());
        prop_assert!(Opcode::MUL.is_arithmetic());
        prop_assert!(Opcode::DIV.is_arithmetic());
        prop_assert!(Opcode::REM.is_arithmetic());
        prop_assert!(Opcode::ADDI.is_arithmetic());

        // Verify non-arithmetic opcodes
        prop_assert!(!Opcode::AND.is_arithmetic());
        prop_assert!(!Opcode::BEQ.is_arithmetic());
        prop_assert!(!Opcode::LW.is_arithmetic());
    }

    /// Property: All branch opcodes are correctly classified.
    #[test]
    fn prop_opcode_branch_classification(_dummy in 0u8..1) {
        // Branch opcodes
        prop_assert!(Opcode::BEQ.is_branch());
        prop_assert!(Opcode::BNE.is_branch());
        prop_assert!(Opcode::BLT.is_branch());
        prop_assert!(Opcode::BGE.is_branch());
        prop_assert!(Opcode::BLTU.is_branch());
        prop_assert!(Opcode::BGEU.is_branch());

        // Non-branch opcodes
        prop_assert!(!Opcode::ADD.is_branch());
        prop_assert!(!Opcode::JAL.is_branch());
        prop_assert!(!Opcode::RET.is_branch());
    }

    /// Property: All terminator opcodes are correctly classified.
    #[test]
    fn prop_opcode_terminator_classification(_dummy in 0u8..1) {
        // Terminators include branches and jumps
        prop_assert!(Opcode::BEQ.is_terminator());
        prop_assert!(Opcode::JAL.is_terminator());
        prop_assert!(Opcode::JALR.is_terminator());
        prop_assert!(Opcode::RET.is_terminator());
        prop_assert!(Opcode::EBREAK.is_terminator());

        // Non-terminators
        prop_assert!(!Opcode::ADD.is_terminator());
        prop_assert!(!Opcode::LW.is_terminator());
        prop_assert!(!Opcode::NOP.is_terminator());
    }

    /// Property: All memory opcodes are correctly classified.
    #[test]
    fn prop_opcode_memory_classification(_dummy in 0u8..1) {
        // Loads
        prop_assert!(Opcode::LB.is_memory());
        prop_assert!(Opcode::LB.is_load());
        prop_assert!(!Opcode::LB.is_store());

        prop_assert!(Opcode::LW.is_memory());
        prop_assert!(Opcode::LW.is_load());

        // Stores
        prop_assert!(Opcode::SB.is_memory());
        prop_assert!(Opcode::SB.is_store());
        prop_assert!(!Opcode::SB.is_load());

        prop_assert!(Opcode::SW.is_memory());
        prop_assert!(Opcode::SW.is_store());

        // Non-memory
        prop_assert!(!Opcode::ADD.is_memory());
        prop_assert!(!Opcode::BEQ.is_memory());
    }

    /// Property: All pseudo-ops are correctly classified.
    #[test]
    fn prop_opcode_pseudo_classification(_dummy in 0u8..1) {
        prop_assert!(Opcode::MOV.is_pseudo());
        prop_assert!(Opcode::LI.is_pseudo());
        prop_assert!(Opcode::NOP.is_pseudo());
        prop_assert!(Opcode::RET.is_pseudo());
        prop_assert!(Opcode::CALL.is_pseudo());
        prop_assert!(Opcode::CALLR.is_pseudo());
        prop_assert!(Opcode::RCHK.is_pseudo());
        prop_assert!(Opcode::PHI.is_pseudo());

        // Non-pseudo
        prop_assert!(!Opcode::ADD.is_pseudo());
        prop_assert!(!Opcode::JAL.is_pseudo());
    }
}

// =============================================================================
// ASSEMBLY OUTPUT TESTS
// =============================================================================

/// Test assembly output contains expected sections.
#[test]
fn test_asm_sections() {
    use zkir_llvm::mir::GlobalVar;

    let mut module = Module::new("asm_test");

    // Add global
    module.globals.insert(
        "data".to_string(),
        GlobalVar {
            name: "data".to_string(),
            size: 4,
            align: 4,
            init: None,
            is_const: true,
        },
    );

    // Add function
    let mut func = MachineFunction::new("main");
    let v0 = func.new_vreg();
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let asm = emit::format_asm(&module);

    // Check for expected sections
    assert!(asm.contains(".section .data"), "Should have data section");
    assert!(asm.contains(".section .text"), "Should have text section");
    assert!(asm.contains("main:"), "Should have function label");
    assert!(asm.contains(".Lentry:"), "Should have block label");
}

/// Test assembly output with frame info.
#[test]
fn test_asm_frame_info() {
    use zkir_llvm::emit::AsmFormatOptions;

    let mut module = Module::new("frame_test");
    let mut func = MachineFunction::new("with_frame");

    // Manually set frame info
    func.frame.locals_size = 16;
    func.frame.spill_size = 8;
    func.frame.outgoing_args_size = 0;

    let v0 = func.new_vreg();
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    // With frame info
    let opts = AsmFormatOptions {
        show_frame_info: true,
        ..Default::default()
    };
    let asm = emit::format_asm_with_options(&module, &opts);
    // Should show frame size info (locals=16, spill=8)
    assert!(asm.contains("locals=16") || asm.contains("frame: 24"), "Should show frame info");

    // Without frame info
    let opts = AsmFormatOptions {
        show_frame_info: false,
        ..Default::default()
    };
    let asm = emit::format_asm_with_options(&module, &opts);
    // Should NOT show locals info
    assert!(!asm.contains("locals=16"), "Should not show frame locals info");
}

/// Test assembly output with addresses.
#[test]
fn test_asm_addresses() {
    use zkir_llvm::emit::AsmFormatOptions;

    let mut module = Module::new("addr_test");
    let mut func = MachineFunction::new("with_addr");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1));
    entry.push(MachineInst::li(v1, 2));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    // With addresses
    let opts = AsmFormatOptions {
        show_addresses: true,
        ..Default::default()
    };
    let asm = emit::format_asm_with_options(&module, &opts);
    assert!(asm.contains("0:"), "Should show instruction addresses");
    assert!(asm.contains("1:"), "Should show instruction 1 address");
}

// =============================================================================
// BYTECODE FORMAT TESTS
// =============================================================================

/// Test bytecode header format using zkir-spec v3.4 32-byte header.
#[test]
fn test_bytecode_header() {
    let config = TargetConfig::default();
    let mut module = Module::new("header_test");

    let mut func = MachineFunction::new("f");
    let v0 = func.new_vreg();
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    // zkir-spec v3.4 header format (32 bytes)
    // Magic at bytes 0-3 (little-endian u32)
    let magic = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
    assert_eq!(magic, 0x5A4B4952, "Magic number"); // "ZKIR" as u32 LE

    // Version at bytes 4-7 (little-endian u32)
    let version = u32::from_le_bytes([bytecode[4], bytecode[5], bytecode[6], bytecode[7]]);
    assert_eq!(version, 0x00030004, "Version v3.4");

    // Config at bytes 8-10
    assert_eq!(bytecode[8], config.limb_bits, "Limb bits");
    assert_eq!(bytecode[9], config.data_limbs, "Data limbs");
    assert_eq!(bytecode[10], config.addr_limbs, "Addr limbs");
}

/// Test bytecode header for different configs using zkir-spec v3.4 format.
#[test]
fn test_bytecode_header_configs() {
    let configs = [
        TargetConfig::DEFAULT,
        TargetConfig::DATA_60,
        TargetConfig::DATA_80,
        TargetConfig::I32_COMPACT,
        TargetConfig::LARGE_LIMB,
    ];

    for config in configs {
        let mut module = Module::new("test");
        let mut func = MachineFunction::new("f");
        let v0 = func.new_vreg();
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 1));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);

        let allocated = regalloc::allocate(&module, &config).unwrap();
        let bytecode = emit::emit(&allocated, &config).unwrap();

        // zkir-spec v3.4: config at bytes 8-10
        assert_eq!(bytecode[8], config.limb_bits);
        assert_eq!(bytecode[9], config.data_limbs);
        assert_eq!(bytecode[10], config.addr_limbs);
    }
}

// =============================================================================
// DATA WIDTH TESTS
// =============================================================================

/// Test i32 operations across all configurations.
#[test]
fn test_i32_all_configs() {
    for &limb_bits in &VALID_LIMB_BITS {
        for &data_limbs in &VALID_DATA_LIMBS {
            let config = TargetConfig {
                limb_bits,
                data_limbs,
                addr_limbs: 2,
            };

            // Skip configs that are too small for i32
            if config.data_bits() < 32 {
                continue;
            }

            let mut module = Module::new("i32_test");
            let mut func = MachineFunction::new("i32_fn");
            let v0 = func.new_vreg();
            let v1 = func.new_vreg();
            let v2 = func.new_vreg();

            let mut entry = MachineBlock::new("entry");
            entry.push(MachineInst::li(v0, i32::MAX as i64));
            entry.push(MachineInst::li(v1, i32::MIN as i64));
            entry.push(MachineInst::add(v2, v0, v1));
            entry.push(MachineInst::ret());
            func.add_block(entry);
            module.add_function(func);

            let allocated = regalloc::allocate(&module, &config)
                .expect(&format!("i32 alloc failed for {}×{}", limb_bits, data_limbs));
            let bytecode = emit::emit(&allocated, &config)
                .expect(&format!("i32 emit failed for {}×{}", limb_bits, data_limbs));
            assert!(!bytecode.is_empty());
        }
    }
}

/// Test range check requirements across configs.
#[test]
fn test_range_check_requirements() {
    // Config with low headroom should need more range checks
    let config_32 = TargetConfig { limb_bits: 16, data_limbs: 2, addr_limbs: 2 };
    assert_eq!(config_32.data_bits(), 32);
    assert_eq!(config_32.headroom(32), 0); // No headroom for i32

    // Config with high headroom should need fewer
    let config_80 = TargetConfig::DATA_80;
    assert_eq!(config_80.data_bits(), 80);
    assert_eq!(config_80.headroom(32), 48); // Lots of headroom
    assert_eq!(config_80.headroom(64), 16); // Some headroom for i64
}

// =============================================================================
// VREG TESTS
// =============================================================================

/// Test VReg ordering and comparison.
#[test]
fn test_vreg_ordering() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);
    let v1_dup = VReg::new(1);

    assert!(v0 < v1);
    assert!(v1 < v2);
    assert_eq!(v1, v1_dup);
    assert_ne!(v0, v1);
}

/// Test VReg display.
#[test]
fn test_vreg_display() {
    assert_eq!(format!("{}", VReg::new(0)), "v0");
    assert_eq!(format!("{}", VReg::new(42)), "v42");
    assert_eq!(format!("{}", VReg::new(1000)), "v1000");
}

proptest! {
    /// Property: VReg serializes round-trip correctly.
    #[test]
    fn prop_vreg_roundtrip(id in 0u32..1000000) {
        let vreg = VReg::new(id);
        prop_assert_eq!(vreg.id(), id);
        prop_assert_eq!(vreg, VReg::new(id));
    }
}

// =============================================================================
// STRESS TESTS
// =============================================================================

/// Stress test: Many sequential operations.
#[test]
fn test_stress_sequential_ops() {
    let config = TargetConfig::default();
    let mut module = Module::new("stress_seq");
    let mut func = MachineFunction::new("seq");

    let num_ops = 500;
    let mut vregs: Vec<VReg> = Vec::with_capacity(num_ops + 2);

    // Initial values
    vregs.push(func.new_vreg());
    vregs.push(func.new_vreg());

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(vregs[0], 1));
    entry.push(MachineInst::li(vregs[1], 2));

    // Chain of operations
    for i in 2..num_ops + 2 {
        let new_vreg = func.new_vreg();
        vregs.push(new_vreg);

        let op_type = i % 4;
        match op_type {
            0 => entry.push(MachineInst::add(new_vreg, vregs[i - 1], vregs[i - 2])),
            1 => entry.push(MachineInst::sub(new_vreg, vregs[i - 1], vregs[i - 2])),
            2 => entry.push(MachineInst::and(new_vreg, vregs[i - 1], vregs[i - 2])),
            3 => entry.push(MachineInst::or(new_vreg, vregs[i - 1], vregs[i - 2])),
            _ => unreachable!(),
        }
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Stress test allocation failed");
    let bytecode = emit::emit(&allocated, &config).expect("Stress test emit failed");
    assert!(bytecode.len() > 100);
}

/// Stress test: Wide function (many parallel live values).
#[test]
fn test_stress_wide_function() {
    let config = TargetConfig::default();
    let mut module = Module::new("stress_wide");
    let mut func = MachineFunction::new("wide");

    let width = 100; // Many parallel values

    let mut entry = MachineBlock::new("entry");

    // Create many independent values
    let mut vregs: Vec<VReg> = Vec::with_capacity(width);
    for i in 0..width {
        let vreg = func.new_vreg();
        vregs.push(vreg);
        entry.push(MachineInst::li(vreg, i as i64));
    }

    // Use all of them (keeps them all live)
    let sum = func.new_vreg();
    entry.push(MachineInst::li(sum, 0));

    for vreg in &vregs {
        let new_sum = func.new_vreg();
        entry.push(MachineInst::add(new_sum, sum, *vreg));
        // Update sum for next iteration (creates chain but old vregs still matter)
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Wide function allocation failed");
    let bytecode = emit::emit(&allocated, &config).expect("Wide function emit failed");
    assert!(!bytecode.is_empty());
}

/// Stress test: Many blocks.
#[test]
fn test_stress_many_blocks() {
    let config = TargetConfig::default();
    let mut module = Module::new("stress_blocks");
    let mut func = MachineFunction::new("blocks");

    let num_blocks = 50;
    let v0 = func.new_vreg();
    let v_zero = func.new_vreg();
    let ra = func.new_vreg(); // Return address register

    // Entry block
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1));
    entry.push(MachineInst::li(v_zero, 0));
    entry.push(MachineInst::jal(ra, "block_0"));
    func.add_block(entry);

    // Chain of blocks
    for i in 0..num_blocks {
        let name = format!("block_{}", i);
        let next = if i < num_blocks - 1 {
            format!("block_{}", i + 1)
        } else {
            "exit".to_string()
        };

        let mut block = MachineBlock::new(&name);
        let tmp = func.new_vreg();
        block.push(MachineInst::add(tmp, v0, v0));
        block.push(MachineInst::jal(ra, &next));
        func.add_block(block);
    }

    // Exit block
    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).expect("Many blocks allocation failed");
    let bytecode = emit::emit(&allocated, &config).expect("Many blocks emit failed");
    assert!(!bytecode.is_empty());
}

/// Stress test: Many functions.
#[test]
fn test_stress_many_functions() {
    let config = TargetConfig::default();
    let mut module = Module::new("stress_funcs");

    let num_funcs = 50;

    for i in 0..num_funcs {
        let mut func = MachineFunction::new(&format!("func_{}", i));
        let v0 = func.new_vreg();
        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, i as i64));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        module.add_function(func);
    }

    let allocated = regalloc::allocate(&module, &config).expect("Many functions allocation failed");
    let bytecode = emit::emit(&allocated, &config).expect("Many functions emit failed");

    let asm = emit::format_asm(&allocated);
    for i in 0..num_funcs {
        assert!(asm.contains(&format!("func_{}:", i)), "Missing function {}", i);
    }
}

// =============================================================================
// DISASSEMBLY ROUND-TRIP TESTS
// =============================================================================

/// Test disassembly produces valid output.
#[test]
fn test_disassembly_basic() {
    use zkir_llvm::emit::disassemble;

    let config = TargetConfig::default();
    let mut module = Module::new("disasm_test");

    let mut func = MachineFunction::new("f");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 20));
    entry.push(MachineInst::add(v2, v0, v1));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    // Disassemble
    let disasm_output = disassemble(&bytecode).expect("Disassembly should succeed");

    // Should contain function and instructions
    assert!(disasm_output.contains("f:"), "Should have function label");
    // Should mention some kind of add or instruction
    assert!(!disasm_output.is_empty(), "Disassembly should produce output");
}
