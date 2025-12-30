//! Machine IR (MIR) for ZKIR.
//!
//! The Machine IR is an intermediate representation between LLVM IR and
//! final ZKIR bytecode. It includes ZK-specific metadata like value bounds
//! and range check flags.
//!
//! # Structure
//!
//! ```text
//! Module
//! └── Functions
//!     └── BasicBlocks
//!         └── Instructions
//! ```

mod instruction;
mod value;
mod function;
mod block;

pub use instruction::{MachineInst, Opcode, Operand};
pub use value::{MachineValue, ValueBounds, VReg};
pub use function::MachineFunction;
pub use block::MachineBlock;

use indexmap::IndexMap;

/// A Machine IR module containing functions and globals.
#[derive(Debug, Clone)]
pub struct Module {
    /// Module name (from LLVM module)
    pub name: String,
    /// Functions in the module
    pub functions: IndexMap<String, MachineFunction>,
    /// Global variables (addresses)
    pub globals: IndexMap<String, GlobalVar>,
}

impl Module {
    /// Create a new empty module.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: IndexMap::new(),
            globals: IndexMap::new(),
        }
    }

    /// Add a function to the module.
    pub fn add_function(&mut self, func: MachineFunction) {
        self.functions.insert(func.name.clone(), func);
    }

    /// Get a function by name.
    pub fn get_function(&self, name: &str) -> Option<&MachineFunction> {
        self.functions.get(name)
    }

    /// Get a mutable function by name.
    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut MachineFunction> {
        self.functions.get_mut(name)
    }

    /// Iterate over all functions.
    pub fn iter_functions(&self) -> impl Iterator<Item = &MachineFunction> {
        self.functions.values()
    }
}

/// A global variable.
#[derive(Debug, Clone)]
pub struct GlobalVar {
    /// Variable name
    pub name: String,
    /// Size in bytes
    pub size: u32,
    /// Alignment in bytes
    pub align: u32,
    /// Initial value (if any)
    pub init: Option<Vec<u8>>,
    /// Is this constant (read-only)?
    pub is_const: bool,
}
