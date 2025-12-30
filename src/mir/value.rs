//! Machine values with ZK metadata.
//!
//! Unlike LLVM's values, our machine values carry bounds information
//! for deferred range checking.

use crate::target::Register;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Virtual register identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VReg(pub u32);

impl VReg {
    /// Create a new virtual register.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the register ID.
    pub fn id(self) -> u32 {
        self.0
    }
}

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Location of a value (virtual register, physical register, or stack).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Location {
    /// Virtual register (before register allocation)
    VReg(VReg),
    /// Physical register (after register allocation)
    Reg(Register),
    /// Stack slot at offset from frame pointer
    Stack(i32),
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Location::VReg(v) => write!(f, "{}", v),
            Location::Reg(r) => write!(f, "{}", r),
            Location::Stack(off) => write!(f, "[fp{:+}]", off),
        }
    }
}

/// A machine value with ZK-specific metadata.
#[derive(Debug, Clone)]
pub struct MachineValue {
    /// Location of the value
    pub location: Location,
    /// Known bounds for this value
    pub bounds: ValueBounds,
    /// Whether this value needs a range check before certain uses
    pub needs_range_check: bool,
    /// Type width in bits
    pub bits: u32,
}

impl MachineValue {
    /// Create a new machine value in a virtual register.
    pub fn vreg(vreg: VReg, bits: u32) -> Self {
        Self {
            location: Location::VReg(vreg),
            bounds: ValueBounds::from_bits(bits),
            needs_range_check: false,
            bits,
        }
    }

    /// Create a new machine value in a physical register.
    pub fn reg(reg: Register, bits: u32) -> Self {
        Self {
            location: Location::Reg(reg),
            bounds: ValueBounds::from_bits(bits),
            needs_range_check: false,
            bits,
        }
    }

    /// Create a value with known bounds.
    pub fn with_bounds(mut self, bounds: ValueBounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Mark this value as needing a range check.
    pub fn with_range_check(mut self) -> Self {
        self.needs_range_check = true;
        self
    }
}

/// Bounds on a value for deferred range checking.
///
/// Tracks the maximum possible value and minimum bits needed,
/// enabling the compiler to skip range checks when the value
/// is known to fit within the target data width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ValueBounds {
    /// Minimum possible value (for bound tracking)
    pub min: u128,
    /// Maximum possible value (for bound tracking)
    pub max: u128,
    /// Minimum number of bits needed to represent this value
    pub bits: u32,
}

impl ValueBounds {
    /// Create bounds for a value that uses the given number of bits.
    pub fn from_bits(bits: u32) -> Self {
        if bits >= 128 {
            Self { min: 0, max: u128::MAX, bits: 128 }
        } else if bits == 0 {
            Self { min: 0, max: 0, bits: 0 }
        } else {
            Self {
                min: 0,
                max: (1u128 << bits) - 1,
                bits,
            }
        }
    }

    /// Create bounds for a known constant value.
    pub fn from_const(value: u128) -> Self {
        let bits = if value == 0 { 1 } else { 128 - value.leading_zeros() };
        Self { min: value, max: value, bits }
    }

    /// Create bounds from a maximum value.
    pub fn from_max(max: u128) -> Self {
        let bits = if max == 0 { 1 } else { 128 - max.leading_zeros() };
        Self { min: 0, max, bits }
    }

    /// Create bounds from min and max values.
    pub fn from_range(min: u128, max: u128) -> Self {
        let bits = if max == 0 { 1 } else { 128 - max.leading_zeros() };
        Self { min, max, bits }
    }

    /// Unknown bounds (full width).
    pub fn unknown(data_bits: u32) -> Self {
        Self::from_bits(data_bits)
    }

    /// Compute bounds after addition: max = a.max + b.max
    pub fn add(a: Self, b: Self) -> Self {
        let min = a.min.saturating_add(b.min);
        let max = a.max.saturating_add(b.max);
        Self::from_range(min, max)
    }

    /// Compute bounds after subtraction (result could be negative, but we track unsigned).
    pub fn sub(a: Self, _b: Self) -> Self {
        // Subtraction can produce any value up to a.max (conservative bound)
        // We can't tighten this because if a < b, the result wraps to a large value
        Self::from_range(0, a.max)
    }

    /// Compute bounds after multiplication: max = a.max * b.max
    pub fn mul(a: Self, b: Self) -> Self {
        let min = a.min.saturating_mul(b.min);
        let max = a.max.saturating_mul(b.max);
        Self::from_range(min, max)
    }

    /// Compute bounds after unsigned division: max = a.max
    pub fn udiv(a: Self, b: Self) -> Self {
        if b.max == 0 {
            // Division by zero - undefined
            Self::from_bits(a.bits)
        } else {
            // Result is at most a.max / 1 = a.max
            // Min is a.min / b.max (integer division)
            let min = if b.max > 0 { a.min / b.max } else { 0 };
            let max = if b.min > 0 { a.max / b.min } else { a.max };
            Self::from_range(min, max)
        }
    }

    /// Compute bounds after AND: max = min(a.max, b.max)
    pub fn and(a: Self, b: Self) -> Self {
        // AND can only reduce bits, so max is min of the two
        Self::from_range(0, a.max.min(b.max))
    }

    /// Compute bounds after OR: max = a.max | b.max (approximately)
    pub fn or(a: Self, b: Self) -> Self {
        // Upper bound: both at max, OR gives at most next power of 2 - 1
        let max_bits = a.bits.max(b.bits);
        let min = a.min.max(b.min); // OR can only increase the value
        Self { min, max: Self::from_bits(max_bits).max, bits: max_bits }
    }

    /// Compute bounds after XOR (same as OR for upper bound).
    pub fn xor(a: Self, b: Self) -> Self {
        // XOR can produce any value within the bit width
        let max_bits = a.bits.max(b.bits);
        Self::from_bits(max_bits)
    }

    /// Compute bounds after left shift: max = a.max << b.max
    pub fn shl(a: Self, b: Self) -> Self {
        if b.max >= 128 {
            Self::from_bits(128)
        } else {
            let min = a.min.saturating_mul(1u128 << b.min as u32);
            let max = a.max.saturating_mul(1u128 << b.max as u32);
            Self::from_range(min, max)
        }
    }

    /// Compute bounds after logical right shift: max = a.max >> b.min (but b.min is usually 0)
    pub fn lshr(a: Self, b: Self) -> Self {
        // Conservative: result is at most a.max
        if b.max >= 128 {
            Self::from_const(0)
        } else {
            let min = if b.max < 128 { a.min >> b.max as u32 } else { 0 };
            let max = if b.min < 128 { a.max >> b.min as u32 } else { 0 };
            Self::from_range(min, max)
        }
    }

    /// Compute bounds after zero extension.
    pub fn zext(self, _to_bits: u32) -> Self {
        // Zero extension doesn't change the value
        self
    }

    /// Compute bounds after truncation.
    pub fn trunc(self, to_bits: u32) -> Self {
        if to_bits >= self.bits {
            self
        } else {
            Self::from_bits(to_bits)
        }
    }

    /// Check if value fits within the given number of bits without range check.
    pub fn fits_in(&self, data_bits: u32) -> bool {
        self.bits <= data_bits
    }

    /// Compute headroom in the given data width.
    pub fn headroom(&self, data_bits: u32) -> u32 {
        data_bits.saturating_sub(self.bits)
    }

    /// Reset bounds after a range check.
    pub fn after_range_check(data_bits: u32) -> Self {
        Self::from_bits(data_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_from_bits() {
        let b = ValueBounds::from_bits(32);
        assert_eq!(b.max, 0xFFFFFFFF);
        assert_eq!(b.bits, 32);
    }

    #[test]
    fn test_bounds_from_const() {
        let b = ValueBounds::from_const(100);
        assert_eq!(b.max, 100);
        assert_eq!(b.bits, 7); // 100 < 128 = 2^7
    }

    #[test]
    fn test_bounds_add() {
        let a = ValueBounds::from_const(100);
        let b = ValueBounds::from_const(50);
        let c = ValueBounds::add(a, b);
        assert_eq!(c.max, 150);
    }

    #[test]
    fn test_bounds_mul() {
        let a = ValueBounds::from_const(100);
        let b = ValueBounds::from_const(50);
        let c = ValueBounds::mul(a, b);
        assert_eq!(c.max, 5000);
    }

    #[test]
    fn test_fits_in() {
        let b = ValueBounds::from_bits(32);
        assert!(b.fits_in(32));
        assert!(b.fits_in(40));
        assert!(!b.fits_in(31));
    }

    #[test]
    fn test_headroom() {
        let b = ValueBounds::from_bits(32);
        assert_eq!(b.headroom(40), 8);
        assert_eq!(b.headroom(32), 0);
        assert_eq!(b.headroom(20), 0);
    }
}
