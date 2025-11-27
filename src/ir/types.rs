//! LLVM type system

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// Void type
    Void,

    /// Integer type with bit width (e.g., i1, i8, i32, i64)
    Int(u32),

    /// Pointer type (opaque in LLVM 15+)
    Ptr,

    /// Array type [N x T]
    Array(usize, Box<Type>),

    /// Struct type { T1, T2, ... }
    Struct(Vec<Type>),

    /// Function type
    Function {
        ret: Box<Type>,
        params: Vec<Type>,
        varargs: bool,
    },
}

impl Type {
    /// Get the bit width of this type (for scalar types)
    pub fn bit_width(&self) -> u32 {
        match self {
            Type::Int(bits) => *bits,
            Type::Ptr => 32, // 32-bit pointers in ZK IR
            _ => 0,
        }
    }

    /// Check if this is a scalar type
    pub fn is_scalar(&self) -> bool {
        matches!(self, Type::Int(_) | Type::Ptr)
    }

    /// Check if this is an aggregate type
    pub fn is_aggregate(&self) -> bool {
        matches!(self, Type::Array(_, _) | Type::Struct(_))
    }

    /// Get the size in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Type::Void => 0,
            Type::Int(bits) => ((*bits + 7) / 8) as usize,
            Type::Ptr => 4, // 32-bit pointers
            Type::Array(n, elem) => n * elem.size_in_bytes(),
            Type::Struct(fields) => fields.iter().map(|f| f.size_in_bytes()).sum(),
            Type::Function { .. } => std::mem::size_of::<usize>(), // Function pointers
        }
    }

    /// Check if this type is supported by ZK IR
    pub fn is_supported(&self) -> bool {
        match self {
            Type::Void => true,
            Type::Int(bits) => matches!(bits, 1 | 8 | 16 | 32 | 64 | 128),
            Type::Ptr => true,
            Type::Array(_, elem) => elem.is_supported(),
            Type::Struct(fields) => fields.iter().all(|f| f.is_supported()),
            Type::Function { .. } => true, // Function pointers are supported
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_width() {
        assert_eq!(Type::Int(32).bit_width(), 32);
        assert_eq!(Type::Int(64).bit_width(), 64);
        assert_eq!(Type::Ptr.bit_width(), 32);
    }

    #[test]
    fn test_size_in_bytes() {
        assert_eq!(Type::Int(32).size_in_bytes(), 4);
        assert_eq!(Type::Int(64).size_in_bytes(), 8);
        assert_eq!(Type::Ptr.size_in_bytes(), 4);
        assert_eq!(
            Type::Array(10, Box::new(Type::Int(32))).size_in_bytes(),
            40
        );
    }

    #[test]
    fn test_is_supported() {
        assert!(Type::Int(32).is_supported());
        assert!(Type::Int(64).is_supported());
        assert!(Type::Ptr.is_supported());
        assert!(!Type::Int(7).is_supported()); // Odd bit widths not supported
    }
}
