//! Type lowering from LLVM to ZK IR

use crate::ir::Type;

/// ZK IR type representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkType {
    /// Single 32-bit register
    I32,

    /// Register pair (64-bit)
    I64,

    /// Register quad (128-bit)
    I128,

    /// Pointer (32-bit)
    Ptr,
}

impl ZkType {
    /// Number of registers needed
    pub fn num_regs(&self) -> usize {
        match self {
            ZkType::I32 | ZkType::Ptr => 1,
            ZkType::I64 => 2,
            ZkType::I128 => 4,
        }
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            ZkType::I32 | ZkType::Ptr => 4,
            ZkType::I64 => 8,
            ZkType::I128 => 16,
        }
    }
}

/// Lower an LLVM type to ZK IR type
pub fn lower_type(ty: &Type) -> Option<ZkType> {
    match ty {
        Type::Void => None,
        Type::Int(bits) => match bits {
            1 | 8 | 16 | 32 => Some(ZkType::I32),
            64 => Some(ZkType::I64),
            128 => Some(ZkType::I128),
            _ => None, // Unsupported bit width
        },
        Type::Ptr => Some(ZkType::Ptr),
        Type::Array(_, _) | Type::Struct(_) => Some(ZkType::Ptr), // Aggregates in memory
        Type::Function { .. } => Some(ZkType::Ptr),                // Function pointers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_type() {
        assert_eq!(lower_type(&Type::Int(1)), Some(ZkType::I32));
        assert_eq!(lower_type(&Type::Int(32)), Some(ZkType::I32));
        assert_eq!(lower_type(&Type::Int(64)), Some(ZkType::I64));
        assert_eq!(lower_type(&Type::Ptr), Some(ZkType::Ptr));
        assert_eq!(lower_type(&Type::Void), None);
    }

    #[test]
    fn test_num_regs() {
        assert_eq!(ZkType::I32.num_regs(), 1);
        assert_eq!(ZkType::I64.num_regs(), 2);
        assert_eq!(ZkType::I128.num_regs(), 4);
    }
}
