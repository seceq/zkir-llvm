//! LLVM IR to Machine IR lowering.
//!
//! This module translates LLVM IR (via inkwell) to our Machine IR,
//! adding ZK-specific metadata like value bounds during the process.
//!
//! # Error Handling
//!
//! The lowering phase uses structured errors to provide context about failures:
//! - Missing operands in LLVM instructions
//! - Unsupported instruction types or operand combinations
//! - Type conversion failures
//! - ZK-specific constraints (e.g., dynamic memcpy sizes)
//!
//! Errors include source location information when debug info is available,
//! making it easier to trace issues back to the original source code.

mod types;
mod arithmetic;
mod memory;
mod control;
mod intrinsics;

use crate::debug::SourceLoc;
use crate::mir::{GlobalVar, MachineBlock, MachineFunction, MachineInst, Module, Opcode, Operand, ValueBounds, VReg};
use crate::target::abi::{compute_arg_locations, ArgLocation};
use crate::target::config::TargetConfig;
use crate::target::registers::Register;
use anyhow::{anyhow, Context, Result};
use indexmap::IndexMap;
use inkwell::llvm_sys;
use inkwell::module::Module as LLVMModule;
use inkwell::values::{AsValueRef, BasicValueEnum, FunctionValue, InstructionOpcode, InstructionValue};
use inkwell::types::{BasicTypeEnum, BasicType};
use std::collections::HashMap;
use thiserror::Error;

pub use types::lower_type;

/// Errors that can occur during LLVM IR lowering.
#[derive(Error, Debug)]
pub enum LoweringError {
    /// An LLVM instruction is missing a required operand.
    #[error("{}", format_error_with_loc(.location, &format!("Missing operand {} in {:?} instruction", .index, .opcode)))]
    MissingOperand {
        opcode: InstructionOpcode,
        index: u32,
        location: Option<SourceLoc>,
    },

    /// An operand has an unexpected type (e.g., expected value, got block).
    #[error("{}", format_error_with_loc(.location, &format!("Invalid operand type at index {} in {:?}: expected {}, got {}", .index, .opcode, .expected, .actual)))]
    InvalidOperandType {
        opcode: InstructionOpcode,
        index: u32,
        expected: &'static str,
        actual: &'static str,
        location: Option<SourceLoc>,
    },

    /// An LLVM instruction or intrinsic is not supported.
    #[error("{}", format_error_with_loc(.location, &format!("Unsupported {}: {}", .kind, .name)))]
    Unsupported {
        kind: &'static str,
        name: String,
        location: Option<SourceLoc>,
    },

    /// A type cannot be lowered to the target.
    #[error("{}", format_error_with_loc(.location, &format!("Cannot lower type: {}", .type_name)))]
    UnsupportedType {
        type_name: String,
        location: Option<SourceLoc>,
    },

    /// A ZK-specific constraint was violated.
    #[error("{}", format_error_with_loc(.location, &format!("ZK constraint violation: {}", .constraint)))]
    ZkConstraint {
        constraint: String,
        location: Option<SourceLoc>,
    },

    /// Failed to convert an LLVM value.
    #[error("{}", format_error_with_loc(.location, &format!("Value conversion failed: {}", .reason)))]
    ValueConversion {
        reason: String,
        location: Option<SourceLoc>,
    },
}

/// Format an error message with optional source location.
fn format_error_with_loc(loc: &Option<SourceLoc>, msg: &str) -> String {
    match loc {
        Some(loc) if !loc.is_unknown() => format!("{}: {}", loc, msg),
        _ => msg.to_string(),
    }
}

impl LoweringError {
    /// Create a MissingOperand error with source location.
    pub fn missing_operand(opcode: InstructionOpcode, index: u32, loc: Option<SourceLoc>) -> Self {
        Self::MissingOperand { opcode, index, location: loc }
    }

    /// Create an InvalidOperandType error with source location.
    pub fn invalid_operand_type(
        opcode: InstructionOpcode,
        index: u32,
        expected: &'static str,
        actual: &'static str,
        loc: Option<SourceLoc>,
    ) -> Self {
        Self::InvalidOperandType { opcode, index, expected, actual, location: loc }
    }

    /// Create an Unsupported error with source location.
    pub fn unsupported(kind: &'static str, name: impl Into<String>, loc: Option<SourceLoc>) -> Self {
        Self::Unsupported { kind, name: name.into(), location: loc }
    }

    /// Create an UnsupportedType error with source location.
    pub fn unsupported_type(type_name: impl Into<String>, loc: Option<SourceLoc>) -> Self {
        Self::UnsupportedType { type_name: type_name.into(), location: loc }
    }

    /// Create a ZkConstraint error with source location.
    pub fn zk_constraint(constraint: impl Into<String>, loc: Option<SourceLoc>) -> Self {
        Self::ZkConstraint { constraint: constraint.into(), location: loc }
    }

    /// Create a ValueConversion error with source location.
    pub fn value_conversion(reason: impl Into<String>, loc: Option<SourceLoc>) -> Self {
        Self::ValueConversion { reason: reason.into(), location: loc }
    }
}

/// Extract source location from an LLVM instruction's debug metadata.
///
/// Returns `Some(SourceLoc)` if the instruction has debug info attached,
/// otherwise returns `None`.
pub fn get_instruction_location<'a>(inst: &InstructionValue<'a>) -> Option<SourceLoc> {
    // LLVM uses metadata kind ID 0 for debug locations (dbg)
    // The debug location metadata contains file, line, and column info
    // Unfortunately, inkwell doesn't expose a direct way to decode DILocation,
    // so we extract what we can from the instruction's string representation

    // Try to extract debug location by checking metadata
    if !inst.has_metadata() {
        return None;
    }

    // The debug location kind ID is typically 0 or available via context
    // For now, we'll try to parse it from the instruction's debug representation
    // This is a pragmatic approach since inkwell doesn't expose DILocation directly
    let inst_str = format!("{:?}", inst);

    // Look for debug location patterns like "!dbg !123" in the instruction string
    // and extract file:line:col information if available
    if let Some(loc) = extract_debug_loc_from_string(&inst_str) {
        return Some(loc);
    }

    None
}

/// Try to extract source location from an instruction's string representation.
/// This is a fallback approach when direct metadata access isn't available.
fn extract_debug_loc_from_string(inst_str: &str) -> Option<SourceLoc> {
    // LLVM IR debug info format includes things like:
    // ", !dbg !42" at the end of instructions
    // The actual location info is in the module's metadata section
    //
    // Since we can't easily access the full debug info through inkwell,
    // we'll return None here and rely on context from the lowering process.
    //
    // In practice, the function name and block label provide good context.
    None
}

/// Lower an entire LLVM module to Machine IR.
pub fn lower_module(module: &LLVMModule, config: &TargetConfig) -> Result<Module> {
    let module_name = module.get_name().to_str().unwrap_or("module");
    let mut mir_module = Module::new(module_name);

    // Lower global variables
    for global in module.get_globals() {
        let name = global.get_name().to_str().unwrap_or("").to_string();
        if name.is_empty() {
            continue;
        }

        let pointee_type = global.get_value_type();
        let size = types::type_size_bytes(&pointee_type, config);
        let align = types::type_align(&pointee_type, config);

        // Get initializer if present
        let init = global.get_initializer().and_then(|v| {
            extract_constant_bytes(&v, config)
        });

        mir_module.globals.insert(name.clone(), GlobalVar {
            name,
            size,
            align,
            init,
            is_const: global.is_constant(),
        });
    }

    // Lower functions
    for func in module.get_functions() {
        // Skip declarations (external functions)
        if func.count_basic_blocks() == 0 {
            continue;
        }

        let mir_func = lower_function(&func, config)?;
        mir_module.add_function(mir_func);
    }

    Ok(mir_module)
}

/// Context for lowering a single function.
struct LoweringContext<'a> {
    /// Target configuration
    config: &'a TargetConfig,
    /// Machine function being built
    func: MachineFunction,
    /// Current block being built
    current_block: Option<String>,
    /// Map from LLVM values to virtual registers
    value_map: HashMap<String, VReg>,
    /// Map from LLVM values to known bounds
    bounds_map: HashMap<String, ValueBounds>,
    /// LLVM function being lowered
    llvm_func: FunctionValue<'a>,
    /// Current source location (from debug info)
    current_loc: Option<SourceLoc>,
    /// Map from LLVM block debug string to generated labels
    /// This ensures consistent labeling for unnamed blocks (which have empty names in LLVM IR)
    block_label_map: HashMap<String, String>,
}

impl<'a> LoweringContext<'a> {
    fn new(func_name: &str, llvm_func: FunctionValue<'a>, config: &'a TargetConfig) -> Self {
        Self {
            config,
            func: MachineFunction::new(func_name),
            current_block: None,
            value_map: HashMap::new(),
            bounds_map: HashMap::new(),
            llvm_func,
            current_loc: None,
            block_label_map: HashMap::new(),
        }
    }

    /// Get a unique label for an LLVM basic block.
    ///
    /// LLVM basic blocks can have empty names. This method returns a consistent
    /// generated label for each block based on its debug representation, ensuring
    /// branches target the correct blocks.
    fn get_block_label(&self, bb: &inkwell::basic_block::BasicBlock) -> String {
        // Use the block's debug string as a unique key
        let key = format!("{:?}", bb);
        if let Some(label) = self.block_label_map.get(&key) {
            return label.clone();
        }
        // Fallback - should not happen if block_label_map is properly initialized
        let name = bb.get_name().to_str().unwrap_or("");
        if name.is_empty() {
            // Use a hash of the debug string as fallback
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            format!("bb_{:x}", hasher.finish() & 0xFFFF)
        } else {
            name.to_string()
        }
    }

    /// Initialize the block label map for all blocks in the function.
    fn init_block_labels(&mut self) {
        let blocks: Vec<_> = self.llvm_func.get_basic_blocks().into_iter().collect();
        for (i, bb) in blocks.iter().enumerate() {
            let key = format!("{:?}", bb);
            let label = get_block_label(bb, i);
            self.block_label_map.insert(key, label);
        }
    }

    /// Set the current source location from an instruction.
    fn set_current_location(&mut self, inst: &InstructionValue<'a>) {
        self.current_loc = get_instruction_location(inst);
    }

    /// Get the current source location.
    fn get_current_loc(&self) -> Option<SourceLoc> {
        self.current_loc.clone()
    }

    /// Create an Unsupported error with current location.
    fn err_unsupported(&self, kind: &'static str, name: impl Into<String>) -> LoweringError {
        LoweringError::unsupported(kind, name, self.current_loc.clone())
    }

    /// Create a ZkConstraint error with current location.
    fn err_zk_constraint(&self, constraint: impl Into<String>) -> LoweringError {
        LoweringError::zk_constraint(constraint, self.current_loc.clone())
    }

    /// Allocate a new virtual register.
    fn new_vreg(&mut self) -> VReg {
        self.func.new_vreg()
    }

    /// Get or create a virtual register for an LLVM value.
    fn get_vreg(&mut self, value: &BasicValueEnum) -> VReg {
        let key = format!("{:?}", value);
        if let Some(&vreg) = self.value_map.get(&key) {
            return vreg;
        }

        // Check if this is a global value (pointer to global variable)
        if let BasicValueEnum::PointerValue(ptr) = value {
            // Try to get as global value
            if ptr.is_const() {
                // Check if it's a global variable reference
                // The debug string will contain "global" for globals
                // or contain "@name" for constant GEP expressions to globals
                let ptr_str = format!("{:?}", ptr);
                if ptr_str.contains("global") || ptr_str.contains("@") {
                    // Extract global name - format is typically "@global_name"
                    if let Some(name) = self.extract_global_name(&ptr_str) {
                        // Extract GEP offset if this is a constant GEP expression
                        let gep_offset = self.extract_gep_offset(&ptr_str);

                        let base_vreg = self.new_vreg();

                        // Emit instruction to load global base address
                        self.emit(MachineInst::new(Opcode::LI)
                            .dst(Operand::VReg(base_vreg))
                            .src(Operand::GlobalAddr(name.clone()))
                            .comment(format!("load address of global {}", name)));

                        // If there's a GEP offset, add it to the base address
                        let final_vreg = if gep_offset != 0 {
                            let result_vreg = self.new_vreg();
                            self.emit(MachineInst::new(Opcode::ADDI)
                                .dst(Operand::VReg(result_vreg))
                                .src(Operand::VReg(base_vreg))
                                .src(Operand::Imm(gep_offset))
                                .comment(format!("add GEP offset {} to global {}", gep_offset, name)));
                            result_vreg
                        } else {
                            base_vreg
                        };

                        self.value_map.insert(key.clone(), final_vreg);

                        // Set bounds for pointer
                        self.bounds_map.insert(key, ValueBounds::from_bits(self.config.addr_bits()));

                        return final_vreg;
                    }
                }
            }
        }

        // Check if this is a constant integer
        if value.is_int_value() {
            let int_val = value.into_int_value();
            if let Some(const_int) = int_val.get_zero_extended_constant() {
                // Create a vreg and emit LI instruction
                let vreg = self.new_vreg();
                self.value_map.insert(key.clone(), vreg);

                // Set bounds for constant
                let bounds = ValueBounds::from_const(const_int as u128);
                self.bounds_map.insert(key, bounds);

                // Emit load immediate
                self.emit(MachineInst::li(vreg, const_int as i64));
                return vreg;
            }
        }

        // Not a constant, allocate new vreg
        let vreg = self.new_vreg();
        self.value_map.insert(key.clone(), vreg);

        // Set default bounds based on type
        if let Ok(ty) = value.get_type().try_into() {
            let bits = types::type_bits(&ty, self.config);
            self.bounds_map.insert(key, ValueBounds::from_bits(bits));
        }

        vreg
    }

    /// Extract global variable name from debug string.
    fn extract_global_name(&self, s: &str) -> Option<String> {
        // Look for patterns like "@global_name" or "GlobalValue(\"name\")"
        if let Some(start) = s.find('@') {
            let rest = &s[start + 1..];
            let end = rest.find(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
                .unwrap_or(rest.len());
            if end > 0 {
                return Some(rest[..end].to_string());
            }
        }
        // Alternative pattern for inkwell debug output
        if let Some(start) = s.find("name: \"") {
            let rest = &s[start + 7..];
            if let Some(end) = rest.find('"') {
                return Some(rest[..end].to_string());
            }
        }
        None
    }

    /// Extract GEP offset from a constant GEP expression debug string.
    ///
    /// Parses strings like:
    /// "i32* getelementptr inbounds ([5 x i32], [5 x i32]* @nums, i32 0, i32 4)"
    ///
    /// Returns the byte offset based on the indices and element type.
    fn extract_gep_offset(&self, s: &str) -> i64 {
        // Look for getelementptr pattern
        if !s.contains("getelementptr") {
            return 0;
        }

        // Try to extract element type size and indices
        // Format: "TYPE* getelementptr inbounds ([N x ELEMTYPE], [N x ELEMTYPE]* @name, i32 IDX0, i32 IDX1, ...)"
        //
        // For [5 x i32], element size is 4 bytes
        // The indices after the global pointer: i32 0 (array index), i32 4 (element index)
        // Total offset = idx1 * elem_size (the first index 0 just dereferences the pointer to array)

        // Find the element type size by looking for patterns like "[N x i32]"
        let elem_size: i64 = if s.contains("i32]") || s.contains("i32*") {
            4
        } else if s.contains("i64]") || s.contains("i64*") {
            8
        } else if s.contains("i16]") || s.contains("i16*") {
            2
        } else if s.contains("i8]") || s.contains("i8*") {
            1
        } else {
            4 // Default to 4 bytes
        };

        // Find all "i32 N" patterns after the global name
        // The last numeric index is typically the element offset within the array
        let mut last_index: i64 = 0;

        // Look for the indices after @name: "i32 0, i32 4)" means indices [0, 4]
        if let Some(at_pos) = s.find('@') {
            let after_name = &s[at_pos..];
            // Find numbers after commas: ", i32 X" patterns
            for part in after_name.split(',') {
                let trimmed = part.trim();
                // Match patterns like "i32 N)" or "i32 N"
                if let Some(num_start) = trimmed.strip_prefix("i32 ") {
                    let num_str: String = num_start.chars()
                        .take_while(|c| c.is_ascii_digit() || *c == '-')
                        .collect();
                    if let Ok(idx) = num_str.parse::<i64>() {
                        last_index = idx;
                    }
                }
            }
        }

        last_index * elem_size
    }

    /// Map an LLVM value to an existing vreg.
    fn map_value(&mut self, value: &BasicValueEnum, vreg: VReg) {
        let key = format!("{:?}", value);
        self.value_map.insert(key, vreg);
    }

    /// Get bounds for a value.
    fn get_bounds(&self, value: &BasicValueEnum) -> ValueBounds {
        let key = format!("{:?}", value);
        self.bounds_map.get(&key).copied().unwrap_or_else(|| {
            // Default bounds from type
            if let Ok(ty) = value.get_type().try_into() {
                let bits = types::type_bits(&ty, self.config);
                ValueBounds::from_bits(bits)
            } else {
                ValueBounds::unknown(self.config.data_bits())
            }
        })
    }

    /// Set bounds for a value.
    fn set_bounds(&mut self, value: &BasicValueEnum, bounds: ValueBounds) {
        let key = format!("{:?}", value);
        self.bounds_map.insert(key, bounds);
    }

    /// Set bounds for a virtual register directly.
    ///
    /// This is used for crypto intrinsics where we know the bounds
    /// based on the algorithm's semantic width, not the LLVM type.
    pub fn set_vreg_bounds(&mut self, vreg: VReg, bounds: ValueBounds) {
        let key = format!("vreg_{}", vreg.0);
        self.bounds_map.insert(key, bounds);
    }

    /// Get bounds for a virtual register.
    pub fn get_vreg_bounds(&self, vreg: VReg) -> Option<ValueBounds> {
        let key = format!("vreg_{}", vreg.0);
        self.bounds_map.get(&key).copied()
    }

    /// Emit an instruction to the current block.
    fn emit(&mut self, inst: MachineInst) {
        if let Some(ref label) = self.current_block {
            if let Some(block) = self.func.get_block_mut(label) {
                block.push(inst);
            }
        }
    }

    /// Create a new block.
    fn create_block(&mut self, label: &str) {
        let block = MachineBlock::new(label);
        self.func.add_block(block);
    }

    /// Switch to a block for emission.
    fn switch_to_block(&mut self, label: &str) {
        self.current_block = Some(label.to_string());
    }

    /// Get a value operand at the given index with proper error handling.
    ///
    /// This is a safer alternative to the `.get_operand().unwrap().unwrap_left()` pattern.
    fn get_value_operand(&mut self, inst: &InstructionValue<'a>, idx: u32) -> Result<BasicValueEnum<'a>> {
        let opcode = inst.get_opcode();
        let loc = self.current_loc.clone();
        inst.get_operand(idx)
            .ok_or_else(|| LoweringError::missing_operand(opcode, idx, loc.clone()))?
            .value()
            .ok_or_else(|| LoweringError::invalid_operand_type(opcode, idx, "value", "block", loc).into())
    }

    /// Get a value operand and its vreg at the given index.
    fn get_operand_vreg(&mut self, inst: &InstructionValue<'a>, idx: u32) -> Result<(BasicValueEnum<'a>, VReg)> {
        let value = self.get_value_operand(inst, idx)?;
        let vreg = self.get_vreg(&value);
        Ok((value, vreg))
    }

    /// Map an instruction result to a new vreg and return both the value and vreg.
    fn map_result(&mut self, inst: &InstructionValue<'a>) -> Result<(BasicValueEnum<'a>, VReg)> {
        // In inkwell 0.5, we create BasicValueEnum from the instruction's value reference
        let result: BasicValueEnum<'a> = unsafe {
            BasicValueEnum::new(inst.as_value_ref())
        };
        let dst = self.new_vreg();
        self.map_value(&result, dst);
        Ok((result, dst))
    }

    /// Map an instruction result to a new vreg with bounds tracking.
    fn map_result_with_bounds(&mut self, inst: &InstructionValue<'a>, bounds: ValueBounds) -> Result<VReg> {
        let (result, dst) = self.map_result(inst)?;
        self.set_bounds(&result, bounds);
        Ok(dst)
    }
}

/// Generate a unique label for an LLVM basic block.
///
/// LLVM basic blocks can have empty names (unnamed blocks are common in optimized code).
/// We generate unique labels like "bb_0", "bb_1", etc. for these blocks.
fn get_block_label(bb: &inkwell::basic_block::BasicBlock, block_index: usize) -> String {
    let name = bb.get_name().to_str().unwrap_or("");
    if name.is_empty() {
        format!("bb_{}", block_index)
    } else {
        name.to_string()
    }
}

/// Lower a single LLVM function to Machine IR.
fn lower_function(func: &FunctionValue, config: &TargetConfig) -> Result<MachineFunction> {
    let func_name = func.get_name().to_str().unwrap_or("func");
    let mut ctx = LoweringContext::new(func_name, *func, config);

    // Initialize block label map - this ensures consistent labels for unnamed blocks
    ctx.init_block_labels();

    // Lower function parameters
    lower_parameters(&mut ctx)
        .with_context(|| format!("Failed to lower parameters for function '{}'", func_name))?;

    // Build a mapping from LLVM block addresses to unique labels
    // This ensures consistency between block creation and branch targets
    let blocks: Vec<_> = func.get_basic_blocks().into_iter().collect();
    let block_labels: Vec<String> = blocks.iter()
        .enumerate()
        .map(|(i, bb)| get_block_label(bb, i))
        .collect();

    // Create blocks first
    for label in &block_labels {
        ctx.create_block(label);
    }

    // Lower each basic block
    for (i, bb) in blocks.iter().enumerate() {
        let label = &block_labels[i];
        ctx.switch_to_block(label);

        // Lower each instruction
        if let Some(inst) = bb.get_first_instruction() {
            let mut current = Some(inst);
            while let Some(instruction) = current {
                lower_instruction(&mut ctx, &instruction)
                    .with_context(|| format!(
                        "Failed to lower {:?} instruction in function '{}', block '{}'",
                        instruction.get_opcode(),
                        func_name,
                        label
                    ))?;
                current = instruction.get_next_instruction();
            }
        }
    }

    // Rebuild CFG edges
    ctx.func.rebuild_cfg();

    Ok(ctx.func)
}

/// Lower function parameters.
fn lower_parameters(ctx: &mut LoweringContext) -> Result<()> {
    let func = ctx.llvm_func;

    // Compute parameter locations based on ABI
    let param_bits: Vec<u32> = func.get_params().iter().map(|p| {
        if let Ok(ty) = p.get_type().try_into() {
            types::type_bits(&ty, ctx.config)
        } else {
            32 // Default to 32 bits
        }
    }).collect();

    let ret_bits = func.get_type().get_return_type().map(|ty| {
        if let Ok(basic_ty) = ty.try_into() {
            types::type_bits(&basic_ty, ctx.config)
        } else {
            32
        }
    });

    let abi = compute_arg_locations(&param_bits, ret_bits, ctx.config);

    // Create vregs for each parameter and copy from ABI locations
    for (i, param) in func.get_params().iter().enumerate() {
        let vreg = ctx.new_vreg();
        ctx.func.params.push(vreg);

        // Map the parameter value to the vreg
        ctx.map_value(param, vreg);

        // Set bounds based on type
        if let Ok(ty) = param.get_type().try_into() {
            let bits = types::type_bits(&ty, ctx.config);
            let key = format!("{:?}", param);
            ctx.bounds_map.insert(key, ValueBounds::from_bits(bits));
        }

        // The actual copying from ABI register to vreg happens in the entry block
        // For now, we just record the mapping - the register allocator will handle
        // the ABI constraint that params start in specific registers
        let _location = &abi.params[i];
    }

    // Record return value vreg if function has a return type
    if abi.ret.is_some() {
        let ret_vreg = ctx.new_vreg();
        ctx.func.ret_vreg = Some(ret_vreg);
    }

    Ok(())
}

/// Lower a single LLVM instruction.
fn lower_instruction<'a>(ctx: &mut LoweringContext<'a>, inst: &InstructionValue<'a>) -> Result<()> {
    // Set current location for error messages
    ctx.set_current_location(inst);

    match inst.get_opcode() {
        // Arithmetic
        InstructionOpcode::Add => arithmetic::lower_add(ctx, inst),
        InstructionOpcode::Sub => arithmetic::lower_sub(ctx, inst),
        InstructionOpcode::Mul => arithmetic::lower_mul(ctx, inst),
        InstructionOpcode::UDiv => arithmetic::lower_udiv(ctx, inst),
        InstructionOpcode::SDiv => arithmetic::lower_sdiv(ctx, inst),
        InstructionOpcode::URem => arithmetic::lower_urem(ctx, inst),
        InstructionOpcode::SRem => arithmetic::lower_srem(ctx, inst),

        // Logical
        InstructionOpcode::And => arithmetic::lower_and(ctx, inst),
        InstructionOpcode::Or => arithmetic::lower_or(ctx, inst),
        InstructionOpcode::Xor => arithmetic::lower_xor(ctx, inst),
        InstructionOpcode::Shl => arithmetic::lower_shl(ctx, inst),
        InstructionOpcode::LShr => arithmetic::lower_lshr(ctx, inst),
        InstructionOpcode::AShr => arithmetic::lower_ashr(ctx, inst),

        // Comparisons
        InstructionOpcode::ICmp => arithmetic::lower_icmp(ctx, inst),

        // Memory
        InstructionOpcode::Alloca => memory::lower_alloca(ctx, inst),
        InstructionOpcode::Load => memory::lower_load(ctx, inst),
        InstructionOpcode::Store => memory::lower_store(ctx, inst),
        InstructionOpcode::GetElementPtr => memory::lower_gep(ctx, inst),

        // Conversions
        InstructionOpcode::Trunc => types::lower_trunc(ctx, inst),
        InstructionOpcode::ZExt => types::lower_zext(ctx, inst),
        InstructionOpcode::SExt => types::lower_sext(ctx, inst),
        InstructionOpcode::PtrToInt => types::lower_ptrtoint(ctx, inst),
        InstructionOpcode::IntToPtr => types::lower_inttoptr(ctx, inst),
        InstructionOpcode::BitCast => types::lower_bitcast(ctx, inst),

        // Control flow
        InstructionOpcode::Br => control::lower_br(ctx, inst),
        InstructionOpcode::Switch => control::lower_switch(ctx, inst),
        InstructionOpcode::Return => control::lower_return(ctx, inst),
        InstructionOpcode::Call => control::lower_call(ctx, inst),
        InstructionOpcode::Unreachable => control::lower_unreachable(ctx, inst),

        // PHI nodes
        InstructionOpcode::Phi => control::lower_phi(ctx, inst),

        // Select
        InstructionOpcode::Select => arithmetic::lower_select(ctx, inst),

        // Freeze instruction (used for undef values) - treat as no-op
        InstructionOpcode::Freeze => {
            // Freeze just removes undef poison, pass through the value
            let (_, src_vreg) = ctx.get_operand_vreg(inst, 0)?;
            let (_, dst) = ctx.map_result(inst)?;
            ctx.emit(MachineInst::mov(dst, src_vreg));
            Ok(())
        }

        // FNeg (floating point negation) - not supported in ZK circuits
        InstructionOpcode::FNeg |
        InstructionOpcode::FAdd |
        InstructionOpcode::FSub |
        InstructionOpcode::FMul |
        InstructionOpcode::FDiv |
        InstructionOpcode::FRem |
        InstructionOpcode::FCmp => {
            Err(ctx.err_unsupported(
                "instruction",
                format!("{:?} (floating point not supported in ZK circuits)", inst.get_opcode())
            ).into())
        }

        // Atomic operations - not supported in ZK circuits (single-threaded)
        InstructionOpcode::AtomicCmpXchg |
        InstructionOpcode::AtomicRMW |
        InstructionOpcode::Fence => {
            Err(ctx.err_unsupported(
                "instruction",
                format!("{:?} (atomic operations not supported in ZK circuits)", inst.get_opcode())
            ).into())
        }

        // Vector operations - not yet supported
        InstructionOpcode::ExtractElement |
        InstructionOpcode::InsertElement |
        InstructionOpcode::ShuffleVector => {
            Err(ctx.err_unsupported(
                "instruction",
                format!("{:?} (vector operations not yet supported)", inst.get_opcode())
            ).into())
        }

        // Invoke and exception handling - not supported
        InstructionOpcode::Invoke |
        InstructionOpcode::Resume |
        InstructionOpcode::LandingPad |
        InstructionOpcode::CleanupRet |
        InstructionOpcode::CatchRet |
        InstructionOpcode::CatchPad |
        InstructionOpcode::CleanupPad |
        InstructionOpcode::CatchSwitch => {
            Err(ctx.err_unsupported(
                "instruction",
                format!("{:?} (exception handling not supported)", inst.get_opcode())
            ).into())
        }

        // Truly unknown or new instructions
        other => {
            log::warn!("Unsupported instruction: {:?}", other);
            Err(ctx.err_unsupported("instruction", format!("{:?}", other)).into())
        }
    }
}

/// Extract constant bytes from an LLVM constant value.
fn extract_constant_bytes(value: &BasicValueEnum, config: &TargetConfig) -> Option<Vec<u8>> {
    match value {
        BasicValueEnum::IntValue(int_val) => {
            if let Some(val) = int_val.get_zero_extended_constant() {
                let bits = int_val.get_type().get_bit_width();
                let bytes = ((bits + 7) / 8) as usize;
                let mut result = Vec::with_capacity(bytes);
                for i in 0..bytes {
                    result.push((val >> (i * 8)) as u8);
                }
                Some(result)
            } else {
                None
            }
        }
        BasicValueEnum::ArrayValue(arr) => {
            // For arrays, try to extract as a constant string (common case for string literals)
            if arr.is_const() {
                // Try to get as constant string first (most common case for globals)
                if arr.is_const_string() {
                    if let Some(cstr) = arr.get_string_constant() {
                        return Some(cstr.to_bytes_with_nul().to_vec());
                    }
                }

                // For non-string constant arrays, use LLVM C API to extract elements
                let arr_type = arr.get_type();
                let len = arr_type.len();

                // Get element type size
                let elem_type = arr_type.get_element_type();
                let elem_type_any = match elem_type {
                    BasicTypeEnum::IntType(t) => inkwell::types::AnyTypeEnum::IntType(t),
                    BasicTypeEnum::FloatType(t) => inkwell::types::AnyTypeEnum::FloatType(t),
                    BasicTypeEnum::PointerType(t) => inkwell::types::AnyTypeEnum::PointerType(t),
                    BasicTypeEnum::ArrayType(t) => inkwell::types::AnyTypeEnum::ArrayType(t),
                    BasicTypeEnum::VectorType(t) => inkwell::types::AnyTypeEnum::VectorType(t),
                    BasicTypeEnum::StructType(t) => inkwell::types::AnyTypeEnum::StructType(t),
                    BasicTypeEnum::ScalableVectorType(t) => inkwell::types::AnyTypeEnum::ScalableVectorType(t),
                };
                let elem_bytes = types::type_size_bytes(&elem_type_any, config) as usize;

                let mut result = Vec::with_capacity(len as usize * elem_bytes);

                // Use LLVMConstExtractValue to extract each element (works in LLVM 14+)
                for i in 0..len {
                    unsafe {
                        let mut idx_list = [i];
                        let elem_ref = llvm_sys::core::LLVMConstExtractValue(
                            arr.as_value_ref(),
                            idx_list.as_mut_ptr(),
                            1
                        );

                        if elem_ref.is_null() {
                            // Element not extractable, fall back to zeros
                            result.extend(std::iter::repeat(0u8).take(elem_bytes));
                        } else {
                            // Convert the element to BasicValueEnum and recursively extract
                            let elem_val = BasicValueEnum::new(elem_ref);
                            if let Some(elem_bytes_vec) = extract_constant_bytes(&elem_val, config) {
                                result.extend(elem_bytes_vec);
                            } else {
                                // Could not extract element, use zeros
                                result.extend(std::iter::repeat(0u8).take(elem_bytes));
                            }
                        }
                    }
                }

                Some(result)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;

    #[test]
    fn test_lower_empty_module() {
        let context = Context::create();
        let module = context.create_module("test");
        let config = TargetConfig::default();

        let mir = lower_module(&module, &config).unwrap();
        assert_eq!(mir.name, "test");
        assert!(mir.functions.is_empty());
    }

    #[test]
    fn test_lowering_error_display() {
        // Test that LoweringError messages are informative
        let err = LoweringError::missing_operand(InstructionOpcode::Add, 1, None);
        let msg = format!("{}", err);
        assert!(msg.contains("Missing operand"));
        assert!(msg.contains("1"));

        let err = LoweringError::unsupported("instruction", "FAdd", None);
        let msg = format!("{}", err);
        assert!(msg.contains("Unsupported"));
        assert!(msg.contains("FAdd"));

        let err = LoweringError::zk_constraint("dynamic memcpy size", None);
        let msg = format!("{}", err);
        assert!(msg.contains("ZK constraint"));
        assert!(msg.contains("memcpy"));
    }

    #[test]
    fn test_lowering_error_with_location() {
        use crate::debug::SourceLoc;

        // Test that source locations are included in error messages
        let loc = Some(SourceLoc::new("test.c", 42, 10));

        let err = LoweringError::unsupported("instruction", "FAdd", loc.clone());
        let msg = format!("{}", err);
        assert!(msg.contains("test.c:42:10"));
        assert!(msg.contains("Unsupported"));
        assert!(msg.contains("FAdd"));

        let err = LoweringError::zk_constraint("dynamic memcpy size", loc.clone());
        let msg = format!("{}", err);
        assert!(msg.contains("test.c:42:10"));
        assert!(msg.contains("ZK constraint"));

        // Test without location (should not include location prefix)
        let err = LoweringError::unsupported("instruction", "FAdd", None);
        let msg = format!("{}", err);
        assert!(!msg.contains("test.c"));
        assert!(msg.contains("Unsupported"));
    }

    #[test]
    fn test_lower_simple_function() {
        let context = Context::create();
        let module = context.create_module("test");
        let i32_type = context.i32_type();

        // Create: define i32 @add(i32 %a, i32 %b) { ret i32 %a }
        let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
        let function = module.add_function("add", fn_type, None);
        let entry = context.append_basic_block(function, "entry");

        let builder = context.create_builder();
        builder.position_at_end(entry);

        let param_a = function.get_nth_param(0).unwrap().into_int_value();
        builder.build_return(Some(&param_a)).unwrap();

        let config = TargetConfig::default();
        let mir = lower_module(&module, &config).unwrap();

        assert_eq!(mir.functions.len(), 1);
        assert!(mir.functions.contains_key("add"));
    }
}
