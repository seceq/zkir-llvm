//! Target description for ZKIR.
//!
//! This module defines the ZKIR target architecture including:
//! - Register definitions and ABI names
//! - Target configuration (variable limb sizes)
//! - Calling convention
//! - Type legalization rules

pub mod config;
pub mod registers;
pub mod abi;

pub use config::TargetConfig;
pub use registers::{
    Register, RegisterClass, REGISTERS, ABI_NAMES,
    ARG_REGS, RET_REGS, TEMP_REGS, CALLEE_SAVED, CALLER_SAVED,
    ALLOCATABLE, ALLOCATABLE_REGS,
};
pub use abi::{
    CallingConv, ArgLocation, FunctionABI, StackFrame, compute_arg_locations,
    REGISTER_SIZE_BYTES, PARAM_ALIGNMENT, FRAME_ALIGNMENT,
};
