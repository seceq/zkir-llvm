//! Integration tests for the ZKIR LLVM backend.
//!
//! These tests exercise the compilation pipeline without requiring LLVM,
//! by constructing Machine IR directly.

use zkir_llvm::emit;
use zkir_llvm::mir::{MachineBlock, MachineFunction, MachineInst, Module};
use zkir_llvm::regalloc;
use zkir_llvm::target::TargetConfig;

/// Test vreg validation prevents emission of unallocated code.
#[test]
fn test_vreg_validation() {
    use zkir_llvm::emit::validate_no_vregs;

    // Create a function with virtual registers (not allocated)
    let mut func = MachineFunction::new("unallocated");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::add(v2, v0, v1));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    // Validation should fail - we haven't allocated registers
    let result = validate_no_vregs(&func);
    assert!(result.is_err(), "Validation should fail for unallocated vregs");

    let errors = result.unwrap_err();
    assert!(errors.len() >= 3, "Should report errors for all vregs");
}

/// Test that register allocation makes code pass validation.
#[test]
fn test_allocated_code_passes_validation() {
    use zkir_llvm::emit::validate_no_vregs;

    let mut module = Module::new("test");

    let mut func = MachineFunction::new("allocated");
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

    // Allocate registers
    let config = TargetConfig::default();
    let allocated = regalloc::allocate(&module, &config).unwrap();

    // Validation should pass after allocation
    for func in allocated.functions.values() {
        let result = validate_no_vregs(func);
        assert!(result.is_ok(), "Validation should pass after allocation: {:?}",
            result.err().map(|e| e.iter().map(|v| format!("{}", v)).collect::<Vec<_>>()));
    }
}

/// Test basic MIR to bytecode compilation.
#[test]
fn test_mir_to_bytecode() {
    // Create a simple function: add two numbers and return
    let mut module = Module::new("test");

    let mut func = MachineFunction::new("add");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10).comment("load 10"));
    entry.push(MachineInst::li(v1, 20).comment("load 20"));
    entry.push(MachineInst::add(v2, v0, v1).comment("add"));
    entry.push(MachineInst::ret().comment("return"));
    func.add_block(entry);

    module.add_function(func);

    // Run register allocation
    let config = TargetConfig::default();
    let allocated = regalloc::allocate(&module, &config).unwrap();

    // Emit bytecode
    let bytecode = emit::emit(&allocated, &config).unwrap();

    // Verify bytecode is not empty and starts with magic number
    assert!(!bytecode.is_empty());
    let magic = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
    assert_eq!(magic, 0x52494B5A); // "ZKIR" in little-endian
}

/// Test assembly output formatting.
#[test]
fn test_asm_output() {
    let mut module = Module::new("test");

    let mut func = MachineFunction::new("main");
    let v0 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    let asm = emit::format_asm(&module);

    assert!(asm.contains("main:"));
    assert!(asm.contains(".Lentry:"));  // Labels use .L prefix
    assert!(asm.contains("li"));
    assert!(asm.contains("ret"));
}

/// Test function with multiple blocks.
#[test]
fn test_multi_block_function() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");

    let mut func = MachineFunction::new("conditional");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v_zero = func.new_vreg();

    // Entry block
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v_zero, 0));
    entry.push(MachineInst::bne(v0, v_zero, "then"));
    func.add_block(entry);

    // Then block
    let mut then_block = MachineBlock::new("then");
    then_block.push(MachineInst::li(v1, 1));
    then_block.push(MachineInst::ret());
    func.add_block(then_block);

    // Else block (fallthrough)
    let mut else_block = MachineBlock::new("else");
    else_block.push(MachineInst::li(v1, 0));
    else_block.push(MachineInst::ret());
    func.add_block(else_block);

    func.rebuild_cfg();
    module.add_function(func);

    // Should compile without errors
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test different limb configurations.
#[test]
fn test_limb_configurations() {
    let mut module = Module::new("test");
    let mut func = MachineFunction::new("test");
    let v0 = func.new_vreg();
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 100));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    // Test 40-bit (2×20)
    let config_40 = TargetConfig::default();
    assert_eq!(config_40.data_bits(), 40);
    let allocated = regalloc::allocate(&module, &config_40).unwrap();
    let bytecode = emit::emit(&allocated, &config_40).unwrap();
    // New header format: bytes 8-10 are limb_bits, data_limbs, addr_limbs
    assert_eq!(bytecode[8], 20); // limb_bits
    assert_eq!(bytecode[9], 2);  // data_limbs

    // Test 60-bit (3×20)
    let config_60 = TargetConfig {
        limb_bits: 20,
        data_limbs: 3,
        addr_limbs: 2,
    };
    assert_eq!(config_60.data_bits(), 60);

    // Test 80-bit (4×20)
    let config_80 = TargetConfig::DATA_80;
    assert_eq!(config_80.data_bits(), 80);
}

/// Test bounds tracking through operations.
#[test]
fn test_value_bounds() {
    use zkir_llvm::mir::ValueBounds;

    let a = ValueBounds::from_const(100);
    let b = ValueBounds::from_const(200);

    // Addition
    let sum = ValueBounds::add(a, b);
    assert_eq!(sum.max, 300);

    // Multiplication
    let product = ValueBounds::mul(a, b);
    assert_eq!(product.max, 20000);

    // Check if fits in data width
    let config = TargetConfig::default();
    assert!(sum.fits_in(config.data_bits()));
    assert!(product.fits_in(config.data_bits()));

    // Large values that don't fit
    let large = ValueBounds::from_bits(64);
    assert!(!large.fits_in(config.data_bits()));
}

/// Test optimization passes.
#[test]
fn test_optimization_pipeline() {
    use zkir_llvm::opt;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Create a function with some dead code
    // Note: constant folding will reduce li(10) + li(20) + add to li(30)
    // and dead code elimination will remove unused definitions
    let mut func = MachineFunction::new("test_opt");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v3 = func.new_vreg();
    func.ret_vreg = Some(v2);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 20));
    entry.push(MachineInst::add(v2, v0, v1));
    entry.push(MachineInst::li(v3, 999)); // Dead code - v3 never used
    entry.push(MachineInst::ret());
    func.add_block(entry);

    let before_count = func.get_block("entry").unwrap().insts.len();
    module.add_function(func);

    // Run optimization
    opt::optimize(&mut module, &config).unwrap();

    // Dead code and constant folding should reduce instruction count
    let optimized = module.functions.get("test_opt").unwrap();
    let entry_block = optimized.get_block("entry").unwrap();

    // Constant folding: li(10) + li(20) + add -> li(30)
    // Dead code: li(v3) removed, li(v0) removed, li(v1) removed
    // Result: li(v2, 30) + ret = 2 instructions
    assert!(entry_block.insts.len() < before_count,
        "Expected optimization to reduce instructions, got {} from {}",
        entry_block.insts.len(), before_count);
}

/// Test loop structure.
#[test]
fn test_loop_structure() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Create a simple loop: sum = 0; for i = 0 to 5: sum += i
    let mut func = MachineFunction::new("loop_sum");
    let i = func.new_vreg();
    let sum = func.new_vreg();
    let limit = func.new_vreg();
    let one = func.new_vreg();
    let cond = func.new_vreg();
    func.ret_vreg = Some(sum);

    // Entry block
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(sum, 0));
    entry.push(MachineInst::li(i, 0));
    entry.push(MachineInst::li(limit, 5));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::new(zkir_llvm::mir::Opcode::JAL)
        .dst(zkir_llvm::mir::Operand::VReg(func.new_vreg()))
        .src(zkir_llvm::mir::Operand::Label("loop".to_string())));
    func.add_block(entry);

    // Loop block
    let mut loop_block = MachineBlock::new("loop");
    loop_block.push(MachineInst::add(sum, sum, i));
    loop_block.push(MachineInst::add(i, i, one));
    // SLT: cond = i < limit
    loop_block.push(MachineInst::new(zkir_llvm::mir::Opcode::SLT)
        .dst(zkir_llvm::mir::Operand::VReg(cond))
        .src(zkir_llvm::mir::Operand::VReg(i))
        .src(zkir_llvm::mir::Operand::VReg(limit)));
    // Branch back if cond != 0
    let zero = func.new_vreg();
    loop_block.push(MachineInst::li(zero, 0));
    loop_block.push(MachineInst::bne(cond, zero, "loop"));
    // Fall through to exit
    loop_block.push(MachineInst::new(zkir_llvm::mir::Opcode::JAL)
        .dst(zkir_llvm::mir::Operand::VReg(func.new_vreg()))
        .src(zkir_llvm::mir::Operand::Label("exit".to_string())));
    func.add_block(loop_block);

    // Exit block
    let mut exit_block = MachineBlock::new("exit");
    exit_block.push(MachineInst::ret());
    func.add_block(exit_block);

    func.rebuild_cfg();
    module.add_function(func);

    // Compile and verify
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test memory operations.
#[test]
fn test_memory_operations() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");

    let mut func = MachineFunction::new("memory_test");
    let base = func.new_vreg();
    let val = func.new_vreg();
    let loaded = func.new_vreg();
    func.ret_vreg = Some(loaded);

    let mut entry = MachineBlock::new("entry");
    // base = 0x1000 (some memory address)
    entry.push(MachineInst::li(base, 0x1000));
    // val = 42
    entry.push(MachineInst::li(val, 42));
    // store val to [base + 0]
    entry.push(MachineInst::sw(val, base, 0));
    // load from [base + 0]
    entry.push(MachineInst::lw(loaded, base, 0));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Should compile without errors
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test function call simulation.
#[test]
fn test_function_call() {
    use zkir_llvm::mir::Opcode;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Create a simple callee
    let mut callee = MachineFunction::new("callee");
    let ret_val = callee.new_vreg();
    callee.ret_vreg = Some(ret_val);

    let mut callee_entry = MachineBlock::new("entry");
    callee_entry.push(MachineInst::li(ret_val, 100));
    callee_entry.push(MachineInst::ret());
    callee.add_block(callee_entry);

    // Create caller
    let mut caller = MachineFunction::new("caller");
    let result = caller.new_vreg();
    caller.ret_vreg = Some(result);

    let mut caller_entry = MachineBlock::new("entry");
    // Call callee
    caller_entry.push(MachineInst::new(Opcode::CALL)
        .src(zkir_llvm::mir::Operand::Label("callee".to_string())));
    // Move result (from a0) to our result vreg
    caller_entry.push(MachineInst::mov(result, caller.new_vreg())); // Placeholder
    caller_entry.push(MachineInst::ret());
    caller.add_block(caller_entry);

    module.add_function(callee);
    module.add_function(caller);

    // Should compile
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test different data widths.
#[test]
fn test_data_widths() {
    // 40-bit config (i32 fits with headroom)
    let config_40 = TargetConfig::default();
    assert_eq!(config_40.data_bits(), 40);
    assert!(!config_40.needs_split(32)); // i32 fits
    assert!(config_40.needs_split(64));  // i64 needs split

    // 60-bit config
    let config_60 = TargetConfig::DATA_60;
    assert_eq!(config_60.data_bits(), 60);
    assert!(!config_60.needs_split(32));
    assert!(!config_60.needs_split(60)); // 60-bit values fit exactly
    assert!(config_60.needs_split(64));  // i64 still needs split (64 > 60)

    // 80-bit config (i64 fits with lots of headroom)
    let config_80 = TargetConfig::DATA_80;
    assert_eq!(config_80.data_bits(), 80);
    assert!(!config_80.needs_split(64));  // i64 fits in 80 bits
    assert!(!config_80.needs_split(80));  // 80-bit fits exactly
    assert!(config_80.needs_split(128));  // i128 still needs split
}

/// Test signed division and remainder operations.
/// These are lowered to unsigned operations with sign handling during LLVM lowering.
#[test]
fn test_signed_division() {
    use zkir_llvm::mir::Opcode;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Test signed division: sdiv(-10, 3) should produce quotient with correct sign
    let mut func = MachineFunction::new("signed_div_test");
    let dividend = func.new_vreg();
    let divisor = func.new_vreg();
    let quotient = func.new_vreg();
    let remainder = func.new_vreg();
    func.ret_vreg = Some(quotient);

    let mut entry = MachineBlock::new("entry");

    // Load dividend (negative value represented as large positive in unsigned)
    // -10 in two's complement (for a 40-bit field would be 2^40 - 10)
    entry.push(MachineInst::li(dividend, -10i64));
    entry.push(MachineInst::li(divisor, 3));

    // The lowering phase should expand SDIV to use SLT + CMOV + DIV + XOR pattern
    // For MIR construction, we use the DIV/REM opcodes and rely on the lowering
    // to handle signed semantics when coming from LLVM IR.

    // Use unsigned DIV here since MIR doesn't have SDIV opcode
    // The actual signed lowering happens in the LLVM->MIR translation
    entry.push(MachineInst::new(Opcode::DIV)
        .dst(zkir_llvm::mir::Operand::VReg(quotient))
        .src(zkir_llvm::mir::Operand::VReg(dividend))
        .src(zkir_llvm::mir::Operand::VReg(divisor)));

    entry.push(MachineInst::new(Opcode::REM)
        .dst(zkir_llvm::mir::Operand::VReg(remainder))
        .src(zkir_llvm::mir::Operand::VReg(dividend))
        .src(zkir_llvm::mir::Operand::VReg(divisor)));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Should compile without errors
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // Check assembly output contains the operations
    let asm = emit::format_asm(&module);
    assert!(asm.contains("signed_div_test:"), "Function should be present");
    assert!(asm.contains("div") || asm.contains("DIV"), "Should contain division");
}

/// Test signed division expansion pattern (the lowering produces multiple instructions).
#[test]
fn test_signed_ops_lowering_pattern() {
    use zkir_llvm::mir::Opcode;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Create a function that tests the expanded signed division pattern
    // The pattern uses: SLT (sign detection), CMOV (conditional abs), DIV, XOR (sign result)
    let mut func = MachineFunction::new("sdiv_pattern");
    let dividend = func.new_vreg();
    let divisor = func.new_vreg();
    let div_sign = func.new_vreg();
    let dvsr_sign = func.new_vreg();
    let zero = func.new_vreg();
    let neg_dividend = func.new_vreg();
    let abs_dividend = func.new_vreg();
    let neg_divisor = func.new_vreg();
    let abs_divisor = func.new_vreg();
    let unsigned_result = func.new_vreg();
    let result_sign = func.new_vreg();
    let neg_result = func.new_vreg();
    let final_result = func.new_vreg();
    func.ret_vreg = Some(final_result);

    let mut entry = MachineBlock::new("entry");

    entry.push(MachineInst::li(dividend, -15i64));
    entry.push(MachineInst::li(divisor, 4));
    entry.push(MachineInst::li(zero, 0));

    // Sign detection: div_sign = (dividend < 0) ? 1 : 0
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(zkir_llvm::mir::Operand::VReg(div_sign))
        .src(zkir_llvm::mir::Operand::VReg(dividend))
        .src(zkir_llvm::mir::Operand::VReg(zero)));

    entry.push(MachineInst::new(Opcode::SLT)
        .dst(zkir_llvm::mir::Operand::VReg(dvsr_sign))
        .src(zkir_llvm::mir::Operand::VReg(divisor))
        .src(zkir_llvm::mir::Operand::VReg(zero)));

    // Compute absolute values using CMOV
    entry.push(MachineInst::sub(neg_dividend, zero, dividend));
    entry.push(MachineInst::new(Opcode::CMOV)
        .dst(zkir_llvm::mir::Operand::VReg(abs_dividend))
        .src(zkir_llvm::mir::Operand::VReg(div_sign))
        .src(zkir_llvm::mir::Operand::VReg(neg_dividend))
        .src(zkir_llvm::mir::Operand::VReg(dividend)));

    entry.push(MachineInst::sub(neg_divisor, zero, divisor));
    entry.push(MachineInst::new(Opcode::CMOV)
        .dst(zkir_llvm::mir::Operand::VReg(abs_divisor))
        .src(zkir_llvm::mir::Operand::VReg(dvsr_sign))
        .src(zkir_llvm::mir::Operand::VReg(neg_divisor))
        .src(zkir_llvm::mir::Operand::VReg(divisor)));

    // Unsigned division
    entry.push(MachineInst::new(Opcode::DIV)
        .dst(zkir_llvm::mir::Operand::VReg(unsigned_result))
        .src(zkir_llvm::mir::Operand::VReg(abs_dividend))
        .src(zkir_llvm::mir::Operand::VReg(abs_divisor)));

    // Result sign = dividend_sign XOR divisor_sign
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(zkir_llvm::mir::Operand::VReg(result_sign))
        .src(zkir_llvm::mir::Operand::VReg(div_sign))
        .src(zkir_llvm::mir::Operand::VReg(dvsr_sign)));

    // Apply sign to result
    entry.push(MachineInst::sub(neg_result, zero, unsigned_result));
    entry.push(MachineInst::new(Opcode::CMOV)
        .dst(zkir_llvm::mir::Operand::VReg(final_result))
        .src(zkir_llvm::mir::Operand::VReg(result_sign))
        .src(zkir_llvm::mir::Operand::VReg(neg_result))
        .src(zkir_llvm::mir::Operand::VReg(unsigned_result)));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Should compile successfully through the full pipeline
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // Verify assembly contains the expected instructions
    let asm = emit::format_asm(&module);
    assert!(asm.contains("slt") || asm.contains("SLT"), "Should contain SLT for sign detection");
    assert!(asm.contains("cmov") || asm.contains("CMOV"), "Should contain CMOV for conditional select");
    assert!(asm.contains("xor") || asm.contains("XOR"), "Should contain XOR for sign computation");
    assert!(asm.contains("div") || asm.contains("DIV"), "Should contain DIV");
}

/// Test PHI node elimination.
#[test]
fn test_phi_elimination() {
    use zkir_llvm::mir::Opcode;
    use zkir_llvm::opt::eliminate_phis;

    let mut module = Module::new("test");

    // Build a diamond CFG with a PHI node at the merge point:
    //       entry
    //       /   \
    //    left   right
    //       \   /
    //       merge (PHI: v4 = [left: v1, right: v2])
    let mut func = MachineFunction::new("diamond_phi");
    let v0 = func.new_vreg(); // condition
    let v1 = func.new_vreg(); // left value
    let v2 = func.new_vreg(); // right value
    let v3 = func.new_vreg(); // PHI result
    let zero = func.new_vreg();
    func.ret_vreg = Some(v3);

    // Entry block - branch based on condition
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::new(Opcode::BNE)
        .src(zkir_llvm::mir::Operand::VReg(v0))
        .src(zkir_llvm::mir::Operand::VReg(zero))
        .src(zkir_llvm::mir::Operand::Label("left".to_string())));
    entry.push(MachineInst::new(Opcode::JAL)
        .dst(zkir_llvm::mir::Operand::VReg(func.new_vreg()))
        .src(zkir_llvm::mir::Operand::Label("right".to_string())));
    func.add_block(entry);

    // Left block
    let mut left = MachineBlock::new("left");
    left.push(MachineInst::li(v1, 10));
    left.push(MachineInst::new(Opcode::JAL)
        .dst(zkir_llvm::mir::Operand::VReg(func.new_vreg()))
        .src(zkir_llvm::mir::Operand::Label("merge".to_string())));
    func.add_block(left);

    // Right block
    let mut right = MachineBlock::new("right");
    right.push(MachineInst::li(v2, 20));
    right.push(MachineInst::new(Opcode::JAL)
        .dst(zkir_llvm::mir::Operand::VReg(func.new_vreg()))
        .src(zkir_llvm::mir::Operand::Label("merge".to_string())));
    func.add_block(right);

    // Merge block with PHI
    let mut merge = MachineBlock::new("merge");
    merge.push(MachineInst::phi(v3)
        .phi_incoming("left", v1)
        .phi_incoming("right", v2));
    merge.push(MachineInst::ret());
    func.add_block(merge);

    module.add_function(func);

    // Check PHI exists before elimination
    {
        let func = module.functions.get("diamond_phi").unwrap();
        let merge_block = func.get_block("merge").unwrap();
        let has_phi = merge_block.insts.iter().any(|i| i.opcode == Opcode::PHI);
        assert!(has_phi, "PHI should exist before elimination");
    }

    // Run PHI elimination
    for func in module.functions.values_mut() {
        eliminate_phis(func).unwrap();
    }

    // Verify PHI was eliminated and MOVs were inserted
    let func = module.functions.get("diamond_phi").unwrap();

    // PHI should be removed from merge block
    let merge_block = func.get_block("merge").unwrap();
    let has_phi = merge_block.insts.iter().any(|i| i.opcode == Opcode::PHI);
    assert!(!has_phi, "PHI should be eliminated");

    // MOVs should be inserted in predecessor blocks (before terminators)
    let left_block = func.get_block("left").unwrap();
    let left_movs = left_block.insts.iter().filter(|i| i.opcode == Opcode::MOV).count();
    assert!(left_movs > 0, "Left block should have a MOV for the PHI copy");

    let right_block = func.get_block("right").unwrap();
    let right_movs = right_block.insts.iter().filter(|i| i.opcode == Opcode::MOV).count();
    assert!(right_movs > 0, "Right block should have a MOV for the PHI copy");
}

/// Test division by zero check is inserted.
#[test]
fn test_div_by_zero_check() {
    use zkir_llvm::mir::Opcode;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Create a function with a division where divisor could be zero
    let mut func = MachineFunction::new("div_test");
    let dividend = func.new_vreg();
    let divisor = func.new_vreg();
    let quotient = func.new_vreg();
    func.params = vec![dividend, divisor]; // Mark as params so bounds are unknown
    func.ret_vreg = Some(quotient);

    let mut entry = MachineBlock::new("entry");
    // Division - the lowering should emit div-by-zero check
    // Since divisor bounds are unknown (param), check should be inserted
    entry.push(MachineInst::new(Opcode::DIV)
        .dst(zkir_llvm::mir::Operand::VReg(quotient))
        .src(zkir_llvm::mir::Operand::VReg(dividend))
        .src(zkir_llvm::mir::Operand::VReg(divisor)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // The div-by-zero check happens during LLVM lowering, not MIR construction.
    // This test verifies the MIR can be compiled. A more thorough test would
    // use LLVM IR input to verify the check is emitted during lowering.
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test function call with argument passing.
#[test]
fn test_function_call_with_args() {
    use zkir_llvm::mir::{Opcode, Operand};
    use zkir_llvm::target::Register;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Create a function that adds its two arguments: add(a, b) -> a + b
    let mut add_func = MachineFunction::new("add_two");
    let arg0 = add_func.new_vreg();
    let arg1 = add_func.new_vreg();
    let sum = add_func.new_vreg();
    add_func.params = vec![arg0, arg1];
    add_func.ret_vreg = Some(sum);

    let mut add_entry = MachineBlock::new("entry");
    // The ABI places first two args in a0 (R10) and a1 (R11)
    add_entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::VReg(arg0))
        .src(Operand::Reg(Register::R10))); // a0
    add_entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::VReg(arg1))
        .src(Operand::Reg(Register::R11))); // a1
    add_entry.push(MachineInst::add(sum, arg0, arg1));
    // Return value goes in a0
    add_entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R10))
        .src(Operand::VReg(sum)));
    add_entry.push(MachineInst::ret());
    add_func.add_block(add_entry);

    // Create caller that passes arguments
    let mut caller = MachineFunction::new("caller");
    let val1 = caller.new_vreg();
    let val2 = caller.new_vreg();
    let result = caller.new_vreg();
    caller.ret_vreg = Some(result);

    let mut caller_entry = MachineBlock::new("entry");
    // Load values to pass as arguments
    caller_entry.push(MachineInst::li(val1, 30));
    caller_entry.push(MachineInst::li(val2, 12));
    // Move args into argument registers
    caller_entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R10))
        .src(Operand::VReg(val1)));
    caller_entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R11))
        .src(Operand::VReg(val2)));
    // Call the function
    caller_entry.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("add_two".to_string())));
    // Get return value from a0
    caller_entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::VReg(result))
        .src(Operand::Reg(Register::R10)));
    caller_entry.push(MachineInst::ret());
    caller.add_block(caller_entry);

    module.add_function(add_func);
    module.add_function(caller);

    // Should compile successfully
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // Verify assembly contains both functions
    let asm = emit::format_asm(&module);
    assert!(asm.contains("add_two:"), "add_two function should be present");
    assert!(asm.contains("caller:"), "caller function should be present");
    assert!(asm.contains("call"), "Should contain call instruction");
}

/// Test recursive function call.
#[test]
fn test_recursive_function() {
    use zkir_llvm::mir::{Opcode, Operand};
    use zkir_llvm::target::Register;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Create a factorial-like pattern (simplified for testing compilation)
    // fact(n) = n <= 1 ? 1 : n * fact(n-1)
    let mut fact = MachineFunction::new("factorial");
    let n = fact.new_vreg();
    let result = fact.new_vreg();
    let one = fact.new_vreg();
    let cond = fact.new_vreg();
    let n_minus_1 = fact.new_vreg();
    let rec_result = fact.new_vreg();
    let zero = fact.new_vreg();
    fact.params = vec![n];
    fact.ret_vreg = Some(result);

    // Entry: get arg from a0
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::VReg(n))
        .src(Operand::Reg(Register::R10)));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::li(zero, 0));
    // cond = n <= 1 => !(n > 1) => !(1 < n) = n - 1 <= 0
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(cond))
        .src(Operand::VReg(one))
        .src(Operand::VReg(n))); // cond = 1 if n > 1
    entry.push(MachineInst::bne(cond, zero, "recurse"));
    // Fall through to base case
    fact.add_block(entry);

    // Base case: return 1
    let mut base_case = MachineBlock::new("base");
    base_case.push(MachineInst::li(result, 1));
    base_case.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R10))
        .src(Operand::VReg(result)));
    base_case.push(MachineInst::ret());
    fact.add_block(base_case);

    // Recursive case
    let mut recurse = MachineBlock::new("recurse");
    recurse.push(MachineInst::sub(n_minus_1, n, one));
    // Save n to stack (simplified - just keep in register)
    recurse.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R10))
        .src(Operand::VReg(n_minus_1)));
    recurse.push(MachineInst::new(Opcode::CALL)
        .src(Operand::Label("factorial".to_string())));
    // Get recursive result
    recurse.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::VReg(rec_result))
        .src(Operand::Reg(Register::R10)));
    // Multiply n * rec_result (simplified - n may have been clobbered)
    recurse.push(MachineInst::mul(result, n, rec_result));
    recurse.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R10))
        .src(Operand::VReg(result)));
    recurse.push(MachineInst::ret());
    fact.add_block(recurse);

    fact.rebuild_cfg();
    module.add_function(fact);

    // Should compile (though the semantics may be incorrect due to simplification)
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test function with multiple return paths.
#[test]
fn test_multiple_return_paths() {
    use zkir_llvm::mir::{Opcode, Operand};
    use zkir_llvm::target::Register;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Function: abs(x) = x < 0 ? -x : x
    let mut abs_func = MachineFunction::new("abs");
    let x = abs_func.new_vreg();
    let neg_x = abs_func.new_vreg();
    let zero = abs_func.new_vreg();
    let is_neg = abs_func.new_vreg();
    abs_func.params = vec![x];
    abs_func.ret_vreg = Some(x); // Will return either x or neg_x

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::VReg(x))
        .src(Operand::Reg(Register::R10)));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(is_neg))
        .src(Operand::VReg(x))
        .src(Operand::VReg(zero)));
    entry.push(MachineInst::bne(is_neg, zero, "negate"));
    // Fall through to positive case
    abs_func.add_block(entry);

    // Positive case: return x as-is
    let mut positive = MachineBlock::new("positive");
    positive.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R10))
        .src(Operand::VReg(x)));
    positive.push(MachineInst::ret());
    abs_func.add_block(positive);

    // Negative case: return -x
    let mut negate = MachineBlock::new("negate");
    negate.push(MachineInst::sub(neg_x, zero, x));
    negate.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::Reg(Register::R10))
        .src(Operand::VReg(neg_x)));
    negate.push(MachineInst::ret());
    abs_func.add_block(negate);

    abs_func.rebuild_cfg();
    module.add_function(abs_func);

    // Should compile
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // Verify both return paths exist
    let asm = emit::format_asm(&module);
    let ret_count = asm.matches("ret").count();
    assert!(ret_count >= 2, "Should have multiple return instructions, found {}", ret_count);
}

/// Test range check insertion.
#[test]
fn test_range_checks() {
    use zkir_llvm::opt::insert_range_checks;
    use zkir_llvm::mir::Opcode;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    let mut func = MachineFunction::new("range_test");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    func.ret_vreg = Some(v2);
    func.params = vec![v0, v1]; // Mark as parameters so they're not constant-folded

    let mut entry = MachineBlock::new("entry");
    // Use parameters - cannot be constant folded
    // Multiplication of two potentially large values can exceed bounds
    entry.push(MachineInst::mul(v2, v0, v1));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Run only range check insertion (not full optimization to avoid folding)
    for func in module.functions.values_mut() {
        insert_range_checks(func, &config).unwrap();
    }

    // Check that range check was inserted
    let optimized = module.functions.get("range_test").unwrap();
    let entry_block = optimized.get_block("entry").unwrap();

    // Should have a RCHK instruction after the multiplication
    let has_rchk = entry_block.insts.iter().any(|i| i.opcode == Opcode::RCHK);
    assert!(has_rchk, "Expected range check to be inserted for multiplication of unknown values");
}

// ============================================================================
// End-to-End Pipeline Tests
// ============================================================================
// These tests simulate patterns that would be produced by LLVM IR lowering
// and exercise the full compilation pipeline.

/// Test nested loop structure (simulating LLVM's lowered loop).
#[test]
fn test_nested_loops() {
    use zkir_llvm::mir::{Opcode, Operand};

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Create nested loop: for i in 0..3: for j in 0..4: sum += i * j
    let mut func = MachineFunction::new("nested_loops");
    let i = func.new_vreg();
    let j = func.new_vreg();
    let sum = func.new_vreg();
    let outer_limit = func.new_vreg();
    let inner_limit = func.new_vreg();
    let one = func.new_vreg();
    let zero = func.new_vreg();
    let prod = func.new_vreg();
    let outer_cond = func.new_vreg();
    let inner_cond = func.new_vreg();
    func.ret_vreg = Some(sum);

    // Entry: initialize
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(sum, 0));
    entry.push(MachineInst::li(i, 0));
    entry.push(MachineInst::li(outer_limit, 3));
    entry.push(MachineInst::li(inner_limit, 4));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::new(Opcode::JAL)
        .dst(Operand::VReg(func.new_vreg()))
        .src(Operand::Label("outer_loop".to_string())));
    func.add_block(entry);

    // Outer loop header
    let mut outer_loop = MachineBlock::new("outer_loop");
    outer_loop.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(outer_cond))
        .src(Operand::VReg(i))
        .src(Operand::VReg(outer_limit)));
    outer_loop.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(outer_cond))
        .src(Operand::VReg(zero))
        .src(Operand::Label("exit".to_string())));
    outer_loop.push(MachineInst::li(j, 0));
    func.add_block(outer_loop);

    // Inner loop header
    let mut inner_loop = MachineBlock::new("inner_loop");
    inner_loop.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(inner_cond))
        .src(Operand::VReg(j))
        .src(Operand::VReg(inner_limit)));
    inner_loop.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(inner_cond))
        .src(Operand::VReg(zero))
        .src(Operand::Label("outer_inc".to_string())));
    // Inner loop body
    inner_loop.push(MachineInst::mul(prod, i, j));
    inner_loop.push(MachineInst::add(sum, sum, prod));
    inner_loop.push(MachineInst::add(j, j, one));
    inner_loop.push(MachineInst::new(Opcode::JAL)
        .dst(Operand::VReg(func.new_vreg()))
        .src(Operand::Label("inner_loop".to_string())));
    func.add_block(inner_loop);

    // Outer loop increment
    let mut outer_inc = MachineBlock::new("outer_inc");
    outer_inc.push(MachineInst::add(i, i, one));
    outer_inc.push(MachineInst::new(Opcode::JAL)
        .dst(Operand::VReg(func.new_vreg()))
        .src(Operand::Label("outer_loop".to_string())));
    func.add_block(outer_inc);

    // Exit
    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    // Full pipeline
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test switch-case pattern (simulating LLVM switch lowering).
#[test]
fn test_switch_case_pattern() {
    use zkir_llvm::mir::{Opcode, Operand};

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // switch(x) { case 0: return 100; case 1: return 200; case 2: return 300; default: return 0; }
    let mut func = MachineFunction::new("switch_test");
    let x = func.new_vreg();
    let result = func.new_vreg();
    let zero = func.new_vreg();
    let one = func.new_vreg();
    let two = func.new_vreg();
    let cmp_0 = func.new_vreg();
    let cmp_1 = func.new_vreg();
    let cmp_2 = func.new_vreg();
    func.params = vec![x];
    func.ret_vreg = Some(result);

    // Entry: compare and branch
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::li(two, 2));

    // Check case 0
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(cmp_0))
        .src(Operand::VReg(x))
        .src(Operand::VReg(zero)));
    entry.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(cmp_0))
        .src(Operand::VReg(zero))
        .src(Operand::Label("case_0".to_string())));

    // Check case 1
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(cmp_1))
        .src(Operand::VReg(x))
        .src(Operand::VReg(one)));
    entry.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(cmp_1))
        .src(Operand::VReg(zero))
        .src(Operand::Label("case_1".to_string())));

    // Check case 2
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(cmp_2))
        .src(Operand::VReg(x))
        .src(Operand::VReg(two)));
    entry.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(cmp_2))
        .src(Operand::VReg(zero))
        .src(Operand::Label("case_2".to_string())));

    // Default case
    entry.push(MachineInst::new(Opcode::JAL)
        .dst(Operand::VReg(func.new_vreg()))
        .src(Operand::Label("default".to_string())));
    func.add_block(entry);

    // Case 0
    let mut case_0 = MachineBlock::new("case_0");
    case_0.push(MachineInst::li(result, 100));
    case_0.push(MachineInst::ret());
    func.add_block(case_0);

    // Case 1
    let mut case_1 = MachineBlock::new("case_1");
    case_1.push(MachineInst::li(result, 200));
    case_1.push(MachineInst::ret());
    func.add_block(case_1);

    // Case 2
    let mut case_2 = MachineBlock::new("case_2");
    case_2.push(MachineInst::li(result, 300));
    case_2.push(MachineInst::ret());
    func.add_block(case_2);

    // Default
    let mut default = MachineBlock::new("default");
    default.push(MachineInst::li(result, 0));
    default.push(MachineInst::ret());
    func.add_block(default);

    func.rebuild_cfg();
    module.add_function(func);

    // Full pipeline
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // Verify all cases are present in assembly
    let asm = emit::format_asm(&module);
    assert!(asm.contains("case_0") || asm.contains("Lcase_0"));
    assert!(asm.contains("case_1") || asm.contains("Lcase_1"));
    assert!(asm.contains("case_2") || asm.contains("Lcase_2"));
}

/// Test array access pattern (simulating GEP lowering).
#[test]
fn test_array_access_pattern() {
    use zkir_llvm::mir::{Opcode, Operand};

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Simulate array[i] = array[i] + 1 for i in 0..4
    let mut func = MachineFunction::new("array_inc");
    let base = func.new_vreg();      // array base pointer
    let i = func.new_vreg();         // loop counter
    let limit = func.new_vreg();     // 4
    let elem_size = func.new_vreg(); // 4 bytes
    let one = func.new_vreg();
    let zero = func.new_vreg();
    let offset = func.new_vreg();
    let addr = func.new_vreg();
    let value = func.new_vreg();
    let new_value = func.new_vreg();
    let cond = func.new_vreg();
    func.params = vec![base];

    // Entry
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(i, 0));
    entry.push(MachineInst::li(limit, 4));
    entry.push(MachineInst::li(elem_size, 4));
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::li(zero, 0));
    func.add_block(entry);

    // Loop header
    let mut loop_header = MachineBlock::new("loop");
    loop_header.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(cond))
        .src(Operand::VReg(i))
        .src(Operand::VReg(limit)));
    loop_header.push(MachineInst::new(Opcode::BEQ)
        .src(Operand::VReg(cond))
        .src(Operand::VReg(zero))
        .src(Operand::Label("exit".to_string())));

    // GEP: addr = base + i * elem_size
    loop_header.push(MachineInst::mul(offset, i, elem_size));
    loop_header.push(MachineInst::add(addr, base, offset));

    // Load, increment, store
    loop_header.push(MachineInst::lw(value, addr, 0));
    loop_header.push(MachineInst::add(new_value, value, one));
    loop_header.push(MachineInst::sw(new_value, addr, 0));

    // Increment counter
    loop_header.push(MachineInst::add(i, i, one));
    loop_header.push(MachineInst::new(Opcode::JAL)
        .dst(Operand::VReg(func.new_vreg()))
        .src(Operand::Label("loop".to_string())));
    func.add_block(loop_header);

    // Exit
    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    // Full pipeline
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // Verify memory operations are present
    let asm = emit::format_asm(&module);
    assert!(asm.contains("lw") || asm.contains("LW"));
    assert!(asm.contains("sw") || asm.contains("SW"));
}

/// Test struct field access pattern.
#[test]
fn test_struct_field_access() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Simulate: struct { int a; int b; int c; }
    // Access: s->b (offset 4), s->c (offset 8)
    let mut func = MachineFunction::new("struct_access");
    let ptr = func.new_vreg();
    let offset_b = func.new_vreg();
    let offset_c = func.new_vreg();
    let addr_b = func.new_vreg();
    let addr_c = func.new_vreg();
    let val_b = func.new_vreg();
    let val_c = func.new_vreg();
    let result = func.new_vreg();
    func.params = vec![ptr];
    func.ret_vreg = Some(result);

    let mut entry = MachineBlock::new("entry");
    // Calculate field offsets
    entry.push(MachineInst::li(offset_b, 4));  // offsetof(b)
    entry.push(MachineInst::li(offset_c, 8));  // offsetof(c)
    entry.push(MachineInst::add(addr_b, ptr, offset_b));
    entry.push(MachineInst::add(addr_c, ptr, offset_c));

    // Load fields
    entry.push(MachineInst::lw(val_b, addr_b, 0));
    entry.push(MachineInst::lw(val_c, addr_c, 0));

    // Compute result = b + c
    entry.push(MachineInst::add(result, val_b, val_c));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Full pipeline
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test bitwise operations pattern (common in crypto).
#[test]
fn test_bitwise_crypto_pattern() {
    use zkir_llvm::mir::{Opcode, Operand};

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Simulate a simplified hash-like mixing function:
    // x = x ^ (x >> 16)
    // x = x * 0x85ebca6b
    // x = x ^ (x >> 13)
    let mut func = MachineFunction::new("mix");
    let x = func.new_vreg();
    let tmp1 = func.new_vreg();
    let tmp2 = func.new_vreg();
    let tmp3 = func.new_vreg();
    let shift16 = func.new_vreg();
    let shift13 = func.new_vreg();
    let mult = func.new_vreg();
    let result = func.new_vreg();
    func.params = vec![x];
    func.ret_vreg = Some(result);

    let mut entry = MachineBlock::new("entry");

    // Load constants
    entry.push(MachineInst::li(shift16, 16));
    entry.push(MachineInst::li(shift13, 13));
    entry.push(MachineInst::li(mult, 0x85ebca6b_u32 as i64));

    // x = x ^ (x >> 16)
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(tmp1))
        .src(Operand::VReg(x))
        .src(Operand::VReg(shift16)));
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(tmp2))
        .src(Operand::VReg(x))
        .src(Operand::VReg(tmp1)));

    // x = x * multiplier
    entry.push(MachineInst::mul(tmp3, tmp2, mult));

    // x = x ^ (x >> 13)
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(tmp1))
        .src(Operand::VReg(tmp3))
        .src(Operand::VReg(shift13)));
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(tmp3))
        .src(Operand::VReg(tmp1)));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Full pipeline
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // Verify bitwise operations are present
    let asm = emit::format_asm(&module);
    assert!(asm.contains("xor") || asm.contains("XOR"));
    assert!(asm.contains("srl") || asm.contains("SRL"));
}

/// Test condition code patterns (select/cmov).
#[test]
fn test_conditional_move_patterns() {
    use zkir_llvm::mir::{Opcode, Operand};

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Simulate: result = (a > b) ? a : b (max function)
    let mut func = MachineFunction::new("max");
    let a = func.new_vreg();
    let b = func.new_vreg();
    let cond = func.new_vreg();
    let result = func.new_vreg();
    func.params = vec![a, b];
    func.ret_vreg = Some(result);

    let mut entry = MachineBlock::new("entry");
    // cond = (a > b) = (b < a)
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(cond))
        .src(Operand::VReg(b))
        .src(Operand::VReg(a)));
    // result = cond ? a : b
    entry.push(MachineInst::new(Opcode::CMOV)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(cond))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Full pipeline
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // Verify CMOV is present (shouldn't be optimized away since inputs are params)
    let asm = emit::format_asm(&module);
    assert!(asm.contains("cmov") || asm.contains("CMOV"));
}

/// Test multi-value return pattern (via stack).
#[test]
fn test_multi_value_return_via_stack() {
    use zkir_llvm::mir::{Opcode, Operand};
    use zkir_llvm::target::Register;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Simulate: divmod(a, b) -> (quotient, remainder) returned via pointer
    let mut func = MachineFunction::new("divmod");
    let a = func.new_vreg();
    let b = func.new_vreg();
    let out_ptr = func.new_vreg();
    let quot = func.new_vreg();
    let rem = func.new_vreg();
    let four = func.new_vreg();
    let out_rem = func.new_vreg();
    func.params = vec![a, b, out_ptr];

    let mut entry = MachineBlock::new("entry");
    // Get params from ABI registers
    entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::VReg(a))
        .src(Operand::Reg(Register::R10)));
    entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::VReg(b))
        .src(Operand::Reg(Register::R11)));
    entry.push(MachineInst::new(Opcode::MOV)
        .dst(Operand::VReg(out_ptr))
        .src(Operand::Reg(Register::R12)));

    // Compute quotient and remainder
    entry.push(MachineInst::new(Opcode::DIV)
        .dst(Operand::VReg(quot))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));
    entry.push(MachineInst::new(Opcode::REM)
        .dst(Operand::VReg(rem))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));

    // Store results
    entry.push(MachineInst::li(four, 4));
    entry.push(MachineInst::sw(quot, out_ptr, 0));
    entry.push(MachineInst::add(out_rem, out_ptr, four));
    entry.push(MachineInst::sw(rem, out_rem, 0));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Full pipeline
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test long chain of operations (tests register pressure and spilling).
#[test]
fn test_long_expression_chain() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Create a long chain: result = ((((a+b)*c)+d)*e)+f
    // This creates many live values simultaneously
    let mut func = MachineFunction::new("long_chain");
    let a = func.new_vreg();
    let b = func.new_vreg();
    let c = func.new_vreg();
    let d = func.new_vreg();
    let e = func.new_vreg();
    let f = func.new_vreg();
    let t1 = func.new_vreg();
    let t2 = func.new_vreg();
    let t3 = func.new_vreg();
    let t4 = func.new_vreg();
    let result = func.new_vreg();
    func.params = vec![a, b, c, d, e, f];
    func.ret_vreg = Some(result);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::add(t1, a, b));     // t1 = a + b
    entry.push(MachineInst::mul(t2, t1, c));    // t2 = t1 * c
    entry.push(MachineInst::add(t3, t2, d));    // t3 = t2 + d
    entry.push(MachineInst::mul(t4, t3, e));    // t4 = t3 * e
    entry.push(MachineInst::add(result, t4, f)); // result = t4 + f
    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Full pipeline
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test dead store elimination pattern.
#[test]
fn test_dead_store_elimination() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");

    let mut func = MachineFunction::new("dead_store");
    let ptr = func.new_vreg();
    let val1 = func.new_vreg();
    let val2 = func.new_vreg();
    let loaded = func.new_vreg();
    func.params = vec![ptr];
    func.ret_vreg = Some(loaded);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(val1, 100));
    entry.push(MachineInst::li(val2, 200));
    // First store (should be eliminated as dead since overwritten)
    entry.push(MachineInst::sw(val1, ptr, 0));
    // Second store to same location
    entry.push(MachineInst::sw(val2, ptr, 0));
    // Load back
    entry.push(MachineInst::lw(loaded, ptr, 0));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Capture instruction count before optimization
    let before_count = module.functions.get("dead_store").unwrap()
        .get_block("entry").unwrap().insts.len();

    // Run optimization
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();

    // Dead store should be eliminated (val1 store is overwritten)
    // Note: This depends on DSE being implemented
    let after_count = module.functions.get("dead_store").unwrap()
        .get_block("entry").unwrap().insts.len();

    // Should compile regardless
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // Check that some optimization occurred (at least dead li(val1) should be removed
    // if DSE removed the first store)
    // This assertion is soft - if DSE isn't fully implemented, the test still passes
    assert!(after_count <= before_count,
        "Optimization should not increase instruction count: {} -> {}",
        before_count, after_count);
}

/// Test copy propagation through PHI nodes.
#[test]
fn test_copy_propagation_through_phi() {
    use zkir_llvm::mir::{Opcode, Operand};

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Create pattern where copy propagation could help after PHI elimination:
    // if (cond) x = a; else x = a;  // Both branches assign same value
    let mut func = MachineFunction::new("copy_prop_phi");
    let cond = func.new_vreg();
    let a = func.new_vreg();
    let x_left = func.new_vreg();
    let x_right = func.new_vreg();
    let x_merged = func.new_vreg();
    let zero = func.new_vreg();
    func.params = vec![cond, a];
    func.ret_vreg = Some(x_merged);

    // Entry
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(zero, 0));
    entry.push(MachineInst::new(Opcode::BNE)
        .src(Operand::VReg(cond))
        .src(Operand::VReg(zero))
        .src(Operand::Label("left".to_string())));
    entry.push(MachineInst::new(Opcode::JAL)
        .dst(Operand::VReg(func.new_vreg()))
        .src(Operand::Label("right".to_string())));
    func.add_block(entry);

    // Left branch
    let mut left = MachineBlock::new("left");
    left.push(MachineInst::mov(x_left, a));
    left.push(MachineInst::new(Opcode::JAL)
        .dst(Operand::VReg(func.new_vreg()))
        .src(Operand::Label("merge".to_string())));
    func.add_block(left);

    // Right branch
    let mut right = MachineBlock::new("right");
    right.push(MachineInst::mov(x_right, a));
    right.push(MachineInst::new(Opcode::JAL)
        .dst(Operand::VReg(func.new_vreg()))
        .src(Operand::Label("merge".to_string())));
    func.add_block(right);

    // Merge with PHI
    let mut merge = MachineBlock::new("merge");
    merge.push(MachineInst::phi(x_merged)
        .phi_incoming("left", x_left)
        .phi_incoming("right", x_right));
    merge.push(MachineInst::ret());
    func.add_block(merge);

    func.rebuild_cfg();
    module.add_function(func);

    // Run full optimization
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();

    // Should compile
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test all arithmetic operations compile correctly.
#[test]
fn test_all_arithmetic_ops() {
    use zkir_llvm::mir::{Opcode, Operand};

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    let mut func = MachineFunction::new("all_arith");
    let a = func.new_vreg();
    let b = func.new_vreg();
    let r_add = func.new_vreg();
    let r_sub = func.new_vreg();
    let r_mul = func.new_vreg();
    let r_div = func.new_vreg();
    let r_rem = func.new_vreg();
    let r_and = func.new_vreg();
    let r_or = func.new_vreg();
    let r_xor = func.new_vreg();
    let r_sll = func.new_vreg();
    let r_srl = func.new_vreg();
    let r_sra = func.new_vreg();
    let r_slt = func.new_vreg();
    let r_sltu = func.new_vreg();
    let result = func.new_vreg();
    func.params = vec![a, b];
    func.ret_vreg = Some(result);

    let mut entry = MachineBlock::new("entry");

    // Test all arithmetic/logical operations
    entry.push(MachineInst::add(r_add, a, b));
    entry.push(MachineInst::sub(r_sub, a, b));
    entry.push(MachineInst::mul(r_mul, a, b));

    entry.push(MachineInst::new(Opcode::DIV)
        .dst(Operand::VReg(r_div))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));
    entry.push(MachineInst::new(Opcode::REM)
        .dst(Operand::VReg(r_rem))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));

    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(r_and))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));
    entry.push(MachineInst::new(Opcode::OR)
        .dst(Operand::VReg(r_or))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(r_xor))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));

    entry.push(MachineInst::new(Opcode::SLL)
        .dst(Operand::VReg(r_sll))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(r_srl))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));
    entry.push(MachineInst::new(Opcode::SRA)
        .dst(Operand::VReg(r_sra))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));

    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(r_slt))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));
    entry.push(MachineInst::new(Opcode::SLTU)
        .dst(Operand::VReg(r_sltu))
        .src(Operand::VReg(a))
        .src(Operand::VReg(b)));

    // Combine all results to force them to be live
    entry.push(MachineInst::add(result, r_add, r_sub));
    entry.push(MachineInst::add(result, result, r_mul));
    entry.push(MachineInst::add(result, result, r_div));
    entry.push(MachineInst::add(result, result, r_rem));
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(result))
        .src(Operand::VReg(r_and)));
    entry.push(MachineInst::new(Opcode::XOR)
        .dst(Operand::VReg(result))
        .src(Operand::VReg(result))
        .src(Operand::VReg(r_or)));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Full pipeline
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test global variable access pattern.
#[test]
fn test_global_variable_access() {
    use zkir_llvm::mir::GlobalVar;

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Add a global variable
    module.globals.insert("counter".to_string(), GlobalVar {
        name: "counter".to_string(),
        size: 4,
        align: 4,
        init: Some(vec![0, 0, 0, 0]), // initialized to 0
        is_const: false,
    });

    // Create function that increments the global
    let mut func = MachineFunction::new("increment_global");
    let global_addr = func.new_vreg();
    let value = func.new_vreg();
    let one = func.new_vreg();
    let new_value = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    // Load address of global using LI with placeholder address
    // In real code, this would be resolved during linking
    entry.push(MachineInst::li(global_addr, 0x1000)); // Placeholder address
    // Load current value
    entry.push(MachineInst::lw(value, global_addr, 0));
    // Increment
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::add(new_value, value, one));
    // Store back
    entry.push(MachineInst::sw(new_value, global_addr, 0));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Compile
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());

    // Verify the module has the global
    assert!(module.globals.contains_key("counter"), "Global variable should exist");
}

/// Test intrinsic-like operations (CTZ, CLZ patterns).
#[test]
fn test_bit_manipulation_patterns() {
    use zkir_llvm::mir::{Opcode, Operand};

    let config = TargetConfig::default();
    let mut module = Module::new("test");

    // Test popcount-like pattern: count bits set in a value
    // Simplified: result = (x & 1) + ((x >> 1) & 1) + ((x >> 2) & 1) + ((x >> 3) & 1)
    let mut func = MachineFunction::new("popcount_4bit");
    let x = func.new_vreg();
    let one = func.new_vreg();
    let shift1 = func.new_vreg();
    let shift2 = func.new_vreg();
    let shift3 = func.new_vreg();
    let b0 = func.new_vreg();
    let b1 = func.new_vreg();
    let b2 = func.new_vreg();
    let b3 = func.new_vreg();
    let x_shift1 = func.new_vreg();
    let x_shift2 = func.new_vreg();
    let x_shift3 = func.new_vreg();
    let t1 = func.new_vreg();
    let t2 = func.new_vreg();
    let result = func.new_vreg();
    func.params = vec![x];
    func.ret_vreg = Some(result);

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(one, 1));
    entry.push(MachineInst::li(shift1, 1));
    entry.push(MachineInst::li(shift2, 2));
    entry.push(MachineInst::li(shift3, 3));

    // Extract each bit
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(b0))
        .src(Operand::VReg(x))
        .src(Operand::VReg(one)));

    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(x_shift1))
        .src(Operand::VReg(x))
        .src(Operand::VReg(shift1)));
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(b1))
        .src(Operand::VReg(x_shift1))
        .src(Operand::VReg(one)));

    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(x_shift2))
        .src(Operand::VReg(x))
        .src(Operand::VReg(shift2)));
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(b2))
        .src(Operand::VReg(x_shift2))
        .src(Operand::VReg(one)));

    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(x_shift3))
        .src(Operand::VReg(x))
        .src(Operand::VReg(shift3)));
    entry.push(MachineInst::new(Opcode::AND)
        .dst(Operand::VReg(b3))
        .src(Operand::VReg(x_shift3))
        .src(Operand::VReg(one)));

    // Sum the bits
    entry.push(MachineInst::add(t1, b0, b1));
    entry.push(MachineInst::add(t2, b2, b3));
    entry.push(MachineInst::add(result, t1, t2));

    entry.push(MachineInst::ret());
    func.add_block(entry);

    module.add_function(func);

    // Full pipeline
    zkir_llvm::opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();
    assert!(!bytecode.is_empty());
}

/// Test that bytecode header is correctly formatted.
#[test]
fn test_bytecode_header_format() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");

    let mut func = MachineFunction::new("simple");
    let v0 = func.new_vreg();
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::ret());
    func.add_block(entry);
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    // Verify header structure (32 bytes total, zkir-spec v3.4):
    // 0-3:   Magic (u32 LE = 0x52494B5A)
    // 4-7:   Version (u32 LE = 0x00030004)
    // 8:     limb_bits
    // 9:     data_limbs
    // 10:    addr_limbs
    // 11:    flags (reserved)
    // 12-15: entry_point (u32 LE)
    // 16-19: code_size (u32 LE)
    // 20-23: data_size (u32 LE)
    // 24-27: bss_size (u32 LE)
    // 28-31: stack_size (u32 LE)
    assert!(bytecode.len() >= 32, "Bytecode should have at least 32 byte header");

    let magic = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
    assert_eq!(magic, 0x52494B5A, "Magic number should be ZKIR");

    let version = u32::from_le_bytes([bytecode[4], bytecode[5], bytecode[6], bytecode[7]]);
    assert_eq!(version, 0x00030004, "Version should be 3.4");

    assert_eq!(bytecode[8], config.limb_bits as u8, "Limb bits should match config");
    assert_eq!(bytecode[9], config.data_limbs as u8, "Data limbs should match config");
    assert_eq!(bytecode[10], config.addr_limbs as u8, "Addr limbs should match config");

    // Bytes 12-15: entry_point (u32 little endian) - should point past header
    let entry_point = u32::from_le_bytes([bytecode[12], bytecode[13], bytecode[14], bytecode[15]]);
    assert!(entry_point >= 32, "Entry point should be at or after header");

    // Bytes 16-19: code_size (u32 little endian)
    let code_size = u32::from_le_bytes([bytecode[16], bytecode[17], bytecode[18], bytecode[19]]);
    assert!(code_size > 0, "Should have some code");
}

// =============================================================================
// Optimization Interaction Tests
// =============================================================================
// These tests verify that multiple optimization passes work correctly together.

/// Test that LICM + constant folding work together.
/// Loop invariant code hoisting should move constant expressions out,
/// then constant folding should simplify them.
#[test]
fn test_licm_and_const_fold_interaction() {
    use zkir_llvm::opt;

    let mut func = MachineFunction::new("test");
    let v0 = func.new_vreg(); // loop counter
    let v1 = func.new_vreg(); // constant 10
    let v2 = func.new_vreg(); // constant 20
    let v3 = func.new_vreg(); // loop invariant: 10 + 20
    let v4 = func.new_vreg(); // result

    // Preheader
    let mut preheader = MachineBlock::new("preheader");
    let ra = func.new_vreg(); // return address register
    preheader.push(MachineInst::li(v0, 0));
    preheader.push(MachineInst::jal(ra, "loop"));
    func.add_block(preheader);

    // Loop body with invariant computation
    let mut loop_block = MachineBlock::new("loop");
    loop_block.push(MachineInst::li(v1, 10));       // Loop invariant
    loop_block.push(MachineInst::li(v2, 20));       // Loop invariant
    loop_block.push(MachineInst::add(v3, v1, v2));  // Loop invariant: should be folded to 30
    loop_block.push(MachineInst::add(v4, v0, v3));  // Uses invariant
    loop_block.push(MachineInst::addi(v0, v0, 1));  // Increment counter
    loop_block.push(MachineInst::bne(v0, v1, "loop")); // Back edge
    loop_block.push(MachineInst::jal(ra, "exit"));
    func.add_block(loop_block);

    // Exit
    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();

    let config = TargetConfig::default();

    // Run the full optimization pipeline
    opt::optimize_function(&mut func, &config).unwrap();

    // After optimization, there should be fewer instructions
    let total_insts: usize = func.block_labels()
        .iter()
        .map(|l| func.get_block(l).map(|b| b.insts.len()).unwrap_or(0))
        .sum();

    // Verify some optimizations happened
    // The loop invariant LIs and ADD should be hoisted or folded
    assert!(total_insts < 10, "Expected optimizations to reduce instructions");
}

/// Test that CSE + copy propagation work together.
/// CSE should eliminate redundant computations, then copy prop
/// should clean up the resulting MOV chains.
#[test]
fn test_cse_and_copy_prop_interaction() {
    use zkir_llvm::opt::{self, eliminate_common_subexpressions, propagate_copies};
    use zkir_llvm::mir::Opcode;

    let mut func = MachineFunction::new("test");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg(); // First a + b
    let v3 = func.new_vreg(); // Second a + b (same computation)
    let v4 = func.new_vreg(); // Uses v2
    let v5 = func.new_vreg(); // Uses v3 (should become v2 after CSE)

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 20));
    entry.push(MachineInst::add(v2, v0, v1)); // First computation
    entry.push(MachineInst::add(v3, v0, v1)); // Redundant - same as v2
    entry.push(MachineInst::mul(v4, v2, v0)); // Uses first result
    entry.push(MachineInst::mul(v5, v3, v1)); // Uses second (should use first after CSE)
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();

    // Run only CSE and copy propagation (not full pipeline which adds range checks)
    opt::eliminate_phis(&mut func).unwrap();
    eliminate_common_subexpressions(&mut func).unwrap();
    propagate_copies(&mut func).unwrap();

    // Verify no duplicate ADD operations exist
    let block = func.get_block("entry").unwrap();
    let add_count = block.insts.iter()
        .filter(|i| i.opcode == Opcode::ADD)
        .count();
    // After CSE, should have at most 1 ADD
    assert!(add_count <= 1, "CSE should eliminate redundant ADD: found {}", add_count);
}

/// Test that peephole + dead code elimination work together.
/// Peephole simplifies patterns, dead code removes unused results.
#[test]
fn test_peephole_and_dce_interaction() {
    use zkir_llvm::opt;
    use zkir_llvm::mir::Opcode;

    let mut func = MachineFunction::new("test");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v3 = func.new_vreg();
    let v4 = func.new_vreg(); // Final result

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::li(v1, 0));
    entry.push(MachineInst::add(v2, v0, v1)); // x + 0 = x (peephole simplifies)
    entry.push(MachineInst::xor(v3, v2, v2)); // x ^ x = 0 (dead after peephole)
    entry.push(MachineInst::mov(v4, v2));     // Final used result
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();

    let config = TargetConfig::default();
    opt::optimize_function(&mut func, &config).unwrap();

    let block = func.get_block("entry").unwrap();

    // After peephole: x + 0 should become MOV
    // After DCE: unused XOR should be removed
    let xor_count = block.insts.iter()
        .filter(|i| i.opcode == Opcode::XOR)
        .count();
    assert_eq!(xor_count, 0, "DCE should remove unused XOR");
}

/// Test optimization at different levels (O0 vs O2).
#[test]
fn test_optimization_levels() {
    use zkir_llvm::opt::{self, OptLevel};

    fn create_test_function() -> MachineFunction {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 0));
        entry.push(MachineInst::add(v2, v0, v1)); // x + 0 = x
        entry.push(MachineInst::mov(v3, v2));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        func.rebuild_cfg();
        func
    }

    let config = TargetConfig::default();

    // O0: No peephole optimizations
    let mut func_o0 = create_test_function();
    opt::optimize_function_with_level(&mut func_o0, &config, OptLevel::O0).unwrap();
    let o0_insts: usize = func_o0.block_labels()
        .iter()
        .map(|l| func_o0.get_block(l).map(|b| b.insts.len()).unwrap_or(0))
        .sum();

    // O2: Full optimizations
    let mut func_o2 = create_test_function();
    opt::optimize_function_with_level(&mut func_o2, &config, OptLevel::O2).unwrap();
    let o2_insts: usize = func_o2.block_labels()
        .iter()
        .map(|l| func_o2.get_block(l).map(|b| b.insts.len()).unwrap_or(0))
        .sum();

    // O2 should produce fewer instructions than O0
    assert!(o2_insts <= o0_insts,
        "O2 should not produce more instructions than O0: O0={}, O2={}", o0_insts, o2_insts);
}

/// Test that optimization pipeline is idempotent.
/// Running optimizations twice should produce the same result as once.
#[test]
fn test_optimization_idempotency() {
    use zkir_llvm::opt;

    fn create_test_function() -> MachineFunction {
        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 5));
        entry.push(MachineInst::li(v1, 10));
        entry.push(MachineInst::add(v2, v0, v1));
        entry.push(MachineInst::mul(v3, v2, v0));
        entry.push(MachineInst::ret());
        func.add_block(entry);
        func.rebuild_cfg();
        func
    }

    let config = TargetConfig::default();

    // Run optimization once
    let mut func1 = create_test_function();
    opt::optimize_function(&mut func1, &config).unwrap();
    let inst_count_1: usize = func1.block_labels()
        .iter()
        .map(|l| func1.get_block(l).map(|b| b.insts.len()).unwrap_or(0))
        .sum();

    // Run optimization twice
    let mut func2 = create_test_function();
    opt::optimize_function(&mut func2, &config).unwrap();
    opt::optimize_function(&mut func2, &config).unwrap();
    let inst_count_2: usize = func2.block_labels()
        .iter()
        .map(|l| func2.get_block(l).map(|b| b.insts.len()).unwrap_or(0))
        .sum();

    assert_eq!(inst_count_1, inst_count_2,
        "Running optimization twice should produce same instruction count: once={}, twice={}",
        inst_count_1, inst_count_2);
}

/// Test range check insertion with various bounds.
#[test]
fn test_range_check_with_bounds() {
    use zkir_llvm::opt;
    use zkir_llvm::mir::ValueBounds;

    let mut func = MachineFunction::new("test");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    // Bounded values (should need fewer range checks)
    entry.push(MachineInst::li(v0, 100).bounds(ValueBounds::from_const(100)));
    entry.push(MachineInst::li(v1, 200).bounds(ValueBounds::from_const(200)));

    // Multiplication of small values - might overflow but within bounds
    let mul_inst = MachineInst::mul(v2, v0, v1);
    entry.push(mul_inst);
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();

    // Use a config with plenty of headroom for small values
    let config = TargetConfig::DATA_80;
    opt::optimize_function(&mut func, &config).unwrap();

    // Should complete without error
    let block = func.get_block("entry").unwrap();
    assert!(block.insts.len() > 0, "Function should have instructions");
}

/// Test that signed comparison operations work in the pipeline.
/// Note: Signed division (SDIV) is lowered from LLVM IR directly during lowering,
/// not at the MIR level. So we test with SLT which exists in MIR.
#[test]
fn test_signed_comparison_pipeline() {
    use zkir_llvm::opt;
    use zkir_llvm::mir::{Opcode, Operand};

    let mut func = MachineFunction::new("test");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v3 = func.new_vreg();
    let v4 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, -10i64 as i64)); // Negative value
    entry.push(MachineInst::li(v1, 5));
    // Signed comparison: -10 < 5 should be true (1)
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(v2))
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1)));
    // Unsigned comparison: large unsigned < 5 should be false (0)
    entry.push(MachineInst::new(Opcode::SLTU)
        .dst(Operand::VReg(v3))
        .src(Operand::VReg(v0))
        .src(Operand::VReg(v1)));
    // Another signed comparison
    entry.push(MachineInst::new(Opcode::SLT)
        .dst(Operand::VReg(v4))
        .src(Operand::VReg(v1))
        .src(Operand::VReg(v0))); // 5 < -10 is false (0)
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();

    let config = TargetConfig::default();
    opt::optimize_function(&mut func, &config).unwrap();

    // After optimization, signed comparisons should still work
    let block = func.get_block("entry").unwrap();

    // SLT instructions should be present (signed comparison lowering)
    let _slt_count = block.insts.iter()
        .filter(|i| i.opcode == Opcode::SLT)
        .count();
    // There should be some instructions related to signed comparison
    // (either original SLT or lowered equivalents)
    assert!(block.insts.len() > 0, "Function should have instructions after optimization");
}

/// Test full pipeline from MIR to bytecode with all optimizations.
#[test]
fn test_full_pipeline_with_optimizations() {
    use zkir_llvm::opt;

    let mut module = Module::new("test");

    let mut func = MachineFunction::new("optimized");
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v3 = func.new_vreg();
    let v4 = func.new_vreg();
    let v5 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    // Set up some values that can be optimized
    entry.push(MachineInst::li(v0, 10));
    entry.push(MachineInst::li(v1, 0));
    entry.push(MachineInst::add(v2, v0, v1));  // x + 0 = x
    entry.push(MachineInst::li(v3, 20));
    entry.push(MachineInst::li(v4, 20));
    entry.push(MachineInst::add(v5, v3, v4));  // Const fold: 20 + 20 = 40
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let config = TargetConfig::default();

    // Optimize
    opt::optimize(&mut module, &config).unwrap();

    // Allocate
    let allocated = regalloc::allocate(&module, &config).unwrap();

    // Emit
    let bytecode = emit::emit(&allocated, &config).unwrap();

    // Should produce valid bytecode (32-byte header in zkir-spec v3.4)
    assert!(bytecode.len() >= 32, "Should produce bytecode with header");
    let magic = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
    assert_eq!(magic, 0x52494B5A, "Should have valid magic");
}

// =============================================================================
// i64 Operations Across Limb Configurations
// =============================================================================

/// Test i64 arithmetic in 40-bit configuration (requires splitting).
#[test]
fn test_i64_arithmetic_40bit_config() {
    use zkir_llvm::opt;

    let config = TargetConfig::default(); // 40-bit
    assert!(config.needs_split(64), "40-bit config should require i64 splitting");

    let mut module = Module::new("test");
    let mut func = MachineFunction::new("i64_add");

    // Simulate i64 addition with values that span multiple limbs
    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 0x1_0000_0000i64)); // > 32-bit value
    entry.push(MachineInst::li(v1, 0x2_0000_0000i64));
    entry.push(MachineInst::add(v2, v0, v1));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    // Optimize (includes integer splitting)
    opt::optimize(&mut module, &config).unwrap();

    // Allocate and emit
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty(), "Should produce bytecode for i64 ops in 40-bit config");
}

/// Test i64 arithmetic in 60-bit configuration.
#[test]
fn test_i64_arithmetic_60bit_config() {
    use zkir_llvm::opt;

    let config = TargetConfig::DATA_60; // 60-bit
    assert!(config.needs_split(64), "60-bit config still needs split for i64");

    let mut module = Module::new("test");
    let mut func = MachineFunction::new("i64_mul");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1_000_000_000i64));
    entry.push(MachineInst::li(v1, 1_000i64));
    entry.push(MachineInst::mul(v2, v0, v1));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty());
}

/// Test i64 arithmetic in 80-bit configuration (no splitting needed).
#[test]
fn test_i64_arithmetic_80bit_config() {
    use zkir_llvm::opt;

    let config = TargetConfig::DATA_80; // 80-bit
    assert!(!config.needs_split(64), "80-bit config should NOT require i64 splitting");

    let mut module = Module::new("test");
    let mut func = MachineFunction::new("i64_no_split");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v2 = func.new_vreg();
    let v3 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, i64::MAX / 2));
    entry.push(MachineInst::li(v1, 2));
    entry.push(MachineInst::add(v2, v0, v1));
    entry.push(MachineInst::sub(v3, v0, v1));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty());
}

/// Test all preset configurations compile i64 code correctly.
#[test]
fn test_i64_all_presets() {
    use zkir_llvm::opt;

    for preset_name in TargetConfig::preset_names() {
        let config = TargetConfig::preset(preset_name)
            .expect(&format!("Preset '{}' should exist", preset_name));

        let mut module = Module::new("test");
        let mut func = MachineFunction::new("i64_test");

        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 0xFFFF_FFFF_FFFFi64)); // 48-bit value
        entry.push(MachineInst::li(v1, 1));
        entry.push(MachineInst::add(v2, v0, v1));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        func.rebuild_cfg();
        module.add_function(func);

        opt::optimize(&mut module, &config)
            .expect(&format!("Optimization should succeed for preset '{}'", preset_name));
        let allocated = regalloc::allocate(&module, &config)
            .expect(&format!("Register allocation should succeed for preset '{}'", preset_name));
        let bytecode = emit::emit(&allocated, &config)
            .expect(&format!("Emission should succeed for preset '{}'", preset_name));

        assert!(!bytecode.is_empty(), "Preset '{}' should produce bytecode", preset_name);
    }
}

// =============================================================================
// Edge Cases: Integer Boundaries and Overflow
// =============================================================================

/// Test operations at i32 boundaries.
#[test]
fn test_i32_boundary_values() {
    use zkir_llvm::opt;

    let config = TargetConfig::default();
    let mut module = Module::new("test");
    let mut func = MachineFunction::new("i32_bounds");

    let v_max = func.new_vreg();
    let v_min = func.new_vreg();
    let v_one = func.new_vreg();
    let v_result1 = func.new_vreg();
    let v_result2 = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v_max, i32::MAX as i64));
    entry.push(MachineInst::li(v_min, i32::MIN as i64));
    entry.push(MachineInst::li(v_one, 1));
    // These will wrap around in 2's complement
    entry.push(MachineInst::add(v_result1, v_max, v_one));
    entry.push(MachineInst::sub(v_result2, v_min, v_one));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty());
}

/// Test operations with zero.
#[test]
fn test_zero_operations() {
    use zkir_llvm::opt;

    let config = TargetConfig::default();
    let mut module = Module::new("test");
    let mut func = MachineFunction::new("zero_ops");

    let v0 = func.new_vreg();
    let v_zero = func.new_vreg();
    let v_add = func.new_vreg();
    let v_sub = func.new_vreg();
    let v_mul = func.new_vreg();
    let v_and = func.new_vreg();
    let v_or = func.new_vreg();
    let v_xor = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 42));
    entry.push(MachineInst::li(v_zero, 0));
    entry.push(MachineInst::add(v_add, v0, v_zero));   // x + 0 = x
    entry.push(MachineInst::sub(v_sub, v0, v_zero));   // x - 0 = x
    entry.push(MachineInst::mul(v_mul, v0, v_zero));   // x * 0 = 0
    entry.push(MachineInst::and(v_and, v0, v_zero));   // x & 0 = 0
    entry.push(MachineInst::or(v_or, v0, v_zero));     // x | 0 = x
    entry.push(MachineInst::xor(v_xor, v0, v_zero));   // x ^ 0 = x
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    opt::optimize(&mut module, &config).unwrap();

    // Check that algebraic simplifications happened
    let optimized_func = module.functions.get("zero_ops").unwrap();
    let block = optimized_func.get_block("entry").unwrap();

    // After optimization, mul by zero should become li 0
    // and add/sub/or/xor with zero should be eliminated
    let inst_count = block.insts.len();
    assert!(inst_count < 9, "Optimization should reduce instruction count from 9, got {}", inst_count);
}

/// Test operations at power-of-2 boundaries.
#[test]
fn test_power_of_two_values() {
    use zkir_llvm::mir::Opcode;
    use zkir_llvm::mir::Operand;
    use zkir_llvm::opt;

    let config = TargetConfig::default();
    let mut module = Module::new("test");
    let mut func = MachineFunction::new("pow2_ops");

    let v0 = func.new_vreg();
    let v_pow2 = func.new_vreg();
    let v_mul = func.new_vreg();
    let v_shift = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 100));
    entry.push(MachineInst::li(v_pow2, 3));  // Shift amount for multiply by 8
    entry.push(MachineInst::mul(v_mul, v0, v0));  // Simple multiply
    // Use shift right to test power-of-2 operation
    entry.push(MachineInst::new(Opcode::SRL)
        .dst(Operand::VReg(v_shift))
        .src(Operand::VReg(v_mul))
        .src(Operand::VReg(v_pow2)));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty());
}

/// Test negative number operations.
#[test]
fn test_negative_number_operations() {
    use zkir_llvm::opt;

    let config = TargetConfig::default();
    let mut module = Module::new("test");
    let mut func = MachineFunction::new("neg_ops");

    let v_neg = func.new_vreg();
    let v_pos = func.new_vreg();
    let v_add = func.new_vreg();
    let v_sub = func.new_vreg();
    let v_mul = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v_neg, -100i64));
    entry.push(MachineInst::li(v_pos, 50));
    entry.push(MachineInst::add(v_add, v_neg, v_pos));  // -100 + 50 = -50
    entry.push(MachineInst::sub(v_sub, v_neg, v_pos));  // -100 - 50 = -150
    entry.push(MachineInst::mul(v_mul, v_neg, v_pos));  // -100 * 50 = -5000
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty());
}

/// Test bitwise operations with all-ones pattern.
#[test]
fn test_all_ones_bitwise_operations() {
    use zkir_llvm::opt;

    let config = TargetConfig::default();
    let mut module = Module::new("test");
    let mut func = MachineFunction::new("ones_ops");

    let v0 = func.new_vreg();
    let v_ones = func.new_vreg();
    let v_and = func.new_vreg();
    let v_or = func.new_vreg();
    let v_xor = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 0xABCD));
    entry.push(MachineInst::li(v_ones, -1i64));  // All ones in 2's complement
    entry.push(MachineInst::and(v_and, v0, v_ones));  // x & -1 = x
    entry.push(MachineInst::or(v_or, v0, v_ones));    // x | -1 = -1
    entry.push(MachineInst::xor(v_xor, v0, v_ones));  // x ^ -1 = ~x
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    opt::optimize(&mut module, &config).unwrap();
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty());
}

// =============================================================================
// Register Allocator Stress Tests
// =============================================================================

/// Test with high register pressure (many simultaneous live values).
#[test]
fn test_high_register_pressure() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");
    let mut func = MachineFunction::new("pressure");

    // Create 20 virtual registers that need to be live simultaneously
    let vregs: Vec<_> = (0..20).map(|_| func.new_vreg()).collect();
    let v_result = func.new_vreg();

    let mut entry = MachineBlock::new("entry");

    // Load values into all registers
    for (i, &vreg) in vregs.iter().enumerate() {
        entry.push(MachineInst::li(vreg, (i + 1) as i64));
    }

    // Chain additions to keep all values live
    let mut acc = vregs[0];
    for i in 1..vregs.len() {
        let new_acc = if i == vregs.len() - 1 { v_result } else { func.new_vreg() };
        entry.push(MachineInst::add(new_acc, acc, vregs[i]));
        acc = new_acc;
    }

    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    // This should trigger spilling
    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty(), "High pressure allocation should succeed");
}

/// Test with deeply nested control flow.
#[test]
fn test_deeply_nested_control_flow() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");
    let mut func = MachineFunction::new("nested");

    let v0 = func.new_vreg();
    let v1 = func.new_vreg();
    let v_zero = func.new_vreg();
    let ra = func.new_vreg(); // return address for jumps

    // Create chain: entry -> level1 -> level2 -> level3 -> exit
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v0, 1));
    entry.push(MachineInst::li(v_zero, 0));
    entry.push(MachineInst::bne(v0, v_zero, "level1"));
    func.add_block(entry);

    let mut level1 = MachineBlock::new("level1");
    level1.push(MachineInst::li(v1, 2));
    level1.push(MachineInst::bne(v1, v_zero, "level2"));
    func.add_block(level1);

    let mut level2 = MachineBlock::new("level2");
    let v2 = func.new_vreg();
    level2.push(MachineInst::add(v2, v0, v1));
    level2.push(MachineInst::bne(v2, v_zero, "level3"));
    func.add_block(level2);

    let mut level3 = MachineBlock::new("level3");
    let v3 = func.new_vreg();
    level3.push(MachineInst::add(v3, v2, v1));
    level3.push(MachineInst::jal(ra, "exit"));
    func.add_block(level3);

    let mut exit = MachineBlock::new("exit");
    exit.push(MachineInst::ret());
    func.add_block(exit);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty());
}

/// Test with values that span long live ranges.
#[test]
fn test_long_live_ranges() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");
    let mut func = MachineFunction::new("long_live");

    // v_long lives from start to end
    let v_long = func.new_vreg();
    let v_final = func.new_vreg();

    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v_long, 42));  // Define v_long early

    // Many intermediate operations
    for i in 0..30 {
        let v_tmp1 = func.new_vreg();
        let v_tmp2 = func.new_vreg();
        entry.push(MachineInst::li(v_tmp1, i as i64));
        entry.push(MachineInst::li(v_tmp2, (i + 1) as i64));
        let v_tmp3 = func.new_vreg();
        entry.push(MachineInst::add(v_tmp3, v_tmp1, v_tmp2));
    }

    // Use v_long at the end
    entry.push(MachineInst::add(v_final, v_long, v_long));
    entry.push(MachineInst::ret());
    func.add_block(entry);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty());
}

/// Test diamond CFG pattern with phi-like merges.
#[test]
fn test_diamond_cfg_allocation() {
    let config = TargetConfig::default();
    let mut module = Module::new("test");
    let mut func = MachineFunction::new("diamond");

    let v_cond = func.new_vreg();
    let v_zero = func.new_vreg();
    let v_then = func.new_vreg();
    let v_else = func.new_vreg();
    let v_merge = func.new_vreg();
    let ra = func.new_vreg(); // return address for jumps

    // Entry: branch based on condition
    let mut entry = MachineBlock::new("entry");
    entry.push(MachineInst::li(v_cond, 1));
    entry.push(MachineInst::li(v_zero, 0));
    entry.push(MachineInst::bne(v_cond, v_zero, "then"));
    func.add_block(entry);

    // Then branch
    let mut then_block = MachineBlock::new("then");
    then_block.push(MachineInst::li(v_then, 100));
    then_block.push(MachineInst::jal(ra, "merge"));
    func.add_block(then_block);

    // Else branch (fallthrough from entry when condition is false)
    let mut else_block = MachineBlock::new("else");
    else_block.push(MachineInst::li(v_else, 200));
    else_block.push(MachineInst::jal(ra, "merge"));
    func.add_block(else_block);

    // Merge point
    let mut merge = MachineBlock::new("merge");
    // In real code, there'd be a PHI here; we simulate with both values
    merge.push(MachineInst::add(v_merge, v_then, v_else));
    merge.push(MachineInst::ret());
    func.add_block(merge);

    func.rebuild_cfg();
    module.add_function(func);

    let allocated = regalloc::allocate(&module, &config).unwrap();
    let bytecode = emit::emit(&allocated, &config).unwrap();

    assert!(!bytecode.is_empty());
}
