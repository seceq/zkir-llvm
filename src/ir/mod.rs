//! LLVM IR data structures

pub mod module;
pub mod function;
pub mod block;
pub mod instruction;
pub mod types;

pub use module::Module;
pub use function::Function;
pub use block::BasicBlock;
pub use instruction::Instruction;
pub use types::Type;

/// A value in LLVM IR
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Local variable reference (e.g., %x)
    Local(String),

    /// Constant integer
    ConstInt { value: i64, ty: Type },

    /// Constant boolean
    ConstBool(bool),

    /// Null pointer
    Null,

    /// Undefined value
    Undef,
}

impl Value {
    pub fn const_i32(value: i32) -> Self {
        Value::ConstInt {
            value: value as i64,
            ty: Type::Int(32),
        }
    }

    pub fn const_i64(value: i64) -> Self {
        Value::ConstInt {
            value,
            ty: Type::Int(64),
        }
    }

    pub fn local(name: impl Into<String>) -> Self {
        Value::Local(name.into())
    }
}
