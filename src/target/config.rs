//! Target configuration for variable limb architecture.
//!
//! ZKIR v3.4 supports variable limb sizes, allowing programs to configure
//! their data width based on application requirements.

use serde::{Deserialize, Serialize};

/// Target configuration for ZKIR code generation.
///
/// ZKIR uses a variable limb architecture where:
/// - `limb_bits`: Size of each limb (16-30 bits, must be even)
/// - `data_limbs`: Number of limbs for data values (1-4)
/// - `addr_limbs`: Number of limbs for addresses (1-2)
///
/// # Common Configurations
///
/// | Config | data_bits | Use Case |
/// |--------|-----------|----------|
/// | 20×2 | 40 | i32 with 8-bit headroom |
/// | 20×3 | 60 | i64 support |
/// | 20×4 | 80 | i64 with large headroom |
/// | 30×2 | 60 | i64 with 28-bit headroom |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetConfig {
    /// Bits per limb (16, 18, 20, 22, 24, 26, 28, or 30)
    pub limb_bits: u8,

    /// Number of limbs for data values (1, 2, 3, or 4)
    pub data_limbs: u8,

    /// Number of limbs for addresses (1 or 2)
    pub addr_limbs: u8,
}

impl Default for TargetConfig {
    /// Default configuration: 20-bit limbs × 2 = 40-bit data
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TargetConfig {
    /// Default configuration: 20-bit limbs × 2 data limbs × 2 address limbs
    pub const DEFAULT: Self = Self {
        limb_bits: 20,
        data_limbs: 2,
        addr_limbs: 2,
    };

    /// Configuration for 60-bit data (good for i64)
    pub const DATA_60: Self = Self {
        limb_bits: 20,
        data_limbs: 3,
        addr_limbs: 2,
    };

    /// Configuration for 80-bit data (i64 with large headroom)
    pub const DATA_80: Self = Self {
        limb_bits: 20,
        data_limbs: 4,
        addr_limbs: 2,
    };

    /// Configuration optimized for i32-only code (minimal limbs)
    pub const I32_COMPACT: Self = Self {
        limb_bits: 18,
        data_limbs: 2,
        addr_limbs: 2,
    };

    /// Configuration with 30-bit limbs for fewer range checks
    pub const LARGE_LIMB: Self = Self {
        limb_bits: 30,
        data_limbs: 2,
        addr_limbs: 2,
    };

    /// Get a preset configuration by name.
    ///
    /// Available presets:
    /// - `"default"` or `"40bit"`: 20×2 = 40-bit (default)
    /// - `"60bit"`: 20×3 = 60-bit (good for i64)
    /// - `"80bit"`: 20×4 = 80-bit (i64 with large headroom)
    /// - `"i32-compact"`: 18×2 = 36-bit (minimal for i32)
    /// - `"large-limb"`: 30×2 = 60-bit (fewer range checks)
    pub fn preset(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "default" | "40bit" => Some(Self::DEFAULT),
            "60bit" => Some(Self::DATA_60),
            "80bit" => Some(Self::DATA_80),
            "i32-compact" | "i32_compact" => Some(Self::I32_COMPACT),
            "large-limb" | "large_limb" => Some(Self::LARGE_LIMB),
            _ => None,
        }
    }

    /// List available preset names.
    pub fn preset_names() -> &'static [&'static str] {
        &["default", "60bit", "80bit", "i32-compact", "large-limb"]
    }

    /// Create a new configuration.
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid.
    pub fn new(limb_bits: u8, data_limbs: u8, addr_limbs: u8) -> Self {
        let config = Self { limb_bits, data_limbs, addr_limbs };
        config.validate().expect("Invalid configuration");
        config
    }

    /// Validate the configuration against zkir-spec constraints.
    ///
    /// This delegates to zkir-spec's Config::new() which validates:
    /// - limb_bits: 16-30, must be even
    /// - data_limbs: 1-4
    /// - addr_limbs: 1-2
    ///
    /// Using zkir-spec validation ensures consistency between zkir-llvm and the specification.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Use zkir-spec's validation for consistency
        zkir_spec::Config::new(self.limb_bits, self.data_limbs, self.addr_limbs)
            .map(|_| ())
            .map_err(|e| match e {
                zkir_spec::ConfigError::InvalidLimbBits => ConfigError::InvalidLimbBits(self.limb_bits),
                zkir_spec::ConfigError::OddLimbBits => ConfigError::OddLimbBits(self.limb_bits),
                zkir_spec::ConfigError::InvalidDataLimbs => ConfigError::InvalidDataLimbs(self.data_limbs),
                zkir_spec::ConfigError::InvalidAddrLimbs => ConfigError::InvalidAddrLimbs(self.addr_limbs),
            })
    }

    /// Total bits for data values.
    ///
    /// This is `limb_bits × data_limbs`.
    #[inline]
    pub fn data_bits(&self) -> u32 {
        self.limb_bits as u32 * self.data_limbs as u32
    }

    /// Total bits for addresses.
    ///
    /// This is `limb_bits × addr_limbs`.
    #[inline]
    pub fn addr_bits(&self) -> u32 {
        self.limb_bits as u32 * self.addr_limbs as u32
    }

    /// Bits per chunk (for range checking).
    ///
    /// This is `limb_bits / 2`.
    #[inline]
    pub fn chunk_bits(&self) -> u32 {
        self.limb_bits as u32 / 2
    }

    /// Size of the lookup table for range checking.
    ///
    /// This is `2^chunk_bits`.
    #[inline]
    pub fn table_size(&self) -> usize {
        1 << self.chunk_bits()
    }

    /// Headroom bits for a value of the given bit width.
    ///
    /// Headroom allows deferred range checking. For example, with 40-bit
    /// data and i32 values, there are 8 bits of headroom, allowing up to
    /// 256 additions before a range check is required.
    #[inline]
    pub fn headroom(&self, value_bits: u32) -> u32 {
        self.data_bits().saturating_sub(value_bits)
    }

    /// Maximum number of deferred additions before range check.
    ///
    /// This is `2^headroom`.
    #[inline]
    pub fn max_deferred_adds(&self, value_bits: u32) -> u64 {
        let headroom = self.headroom(value_bits);
        if headroom >= 64 {
            u64::MAX
        } else {
            1u64 << headroom
        }
    }

    /// Whether a type with the given bit width needs to be split
    /// across multiple registers.
    #[inline]
    pub fn needs_split(&self, type_bits: u32) -> bool {
        type_bits > self.data_bits()
    }

    /// Number of registers needed for a value of the given bit width.
    #[inline]
    pub fn regs_for_bits(&self, type_bits: u32) -> u32 {
        if type_bits == 0 {
            0
        } else {
            type_bits.div_ceil(self.data_bits())
        }
    }

    /// Maximum value that fits in the data width.
    #[inline]
    pub fn max_value(&self) -> u128 {
        if self.data_bits() >= 128 {
            u128::MAX
        } else {
            (1u128 << self.data_bits()) - 1
        }
    }

    /// Maximum limb value.
    #[inline]
    pub fn max_limb(&self) -> u32 {
        (1u32 << self.limb_bits) - 1
    }

    /// Check for common misconfigurations and return warnings.
    ///
    /// This method helps users identify potential issues with their configuration
    /// that won't cause errors but may lead to suboptimal results.
    pub fn check_warnings(&self) -> Vec<ConfigWarning> {
        let mut warnings = Vec::new();

        // Check if i64 operations will require splitting
        if self.data_bits() < 64 {
            warnings.push(ConfigWarning::I64RequiresSplit {
                data_bits: self.data_bits(),
            });
        }

        // Check for insufficient headroom for i32
        let headroom_32 = self.headroom(32);
        if headroom_32 < 4 {
            warnings.push(ConfigWarning::LowHeadroom {
                type_bits: 32,
                headroom: headroom_32,
            });
        }

        // Check for very small table size (increased memory lookups)
        if self.table_size() < 256 {
            warnings.push(ConfigWarning::SmallTableSize {
                table_size: self.table_size(),
            });
        }

        // Check for potentially expensive configurations
        if self.data_limbs > 2 && self.limb_bits > 24 {
            warnings.push(ConfigWarning::ExpensiveConfig {
                total_bits: self.data_bits(),
            });
        }

        warnings
    }

    /// Suggest the best preset for a given maximum integer width.
    ///
    /// # Arguments
    /// * `max_int_bits` - The maximum integer width used in the program (e.g., 32 for i32, 64 for i64)
    ///
    /// # Returns
    /// A suggested configuration that balances performance and capability.
    pub fn suggest_for_int_width(max_int_bits: u32) -> Self {
        match max_int_bits {
            0..=32 => Self::DEFAULT,         // 40-bit is fine for i32
            33..=60 => Self::DATA_60,        // 60-bit for up to i60
            61..=80 => Self::DATA_80,        // 80-bit for i64 with headroom
            _ => Self::DATA_80,              // Best we can do
        }
    }

    /// Convert to zkir-spec Config for binary emission and validation.
    ///
    /// This creates a zkir-spec Config with the same limb configuration,
    /// allowing zkir-llvm to use zkir-spec's validation and encoding functions.
    ///
    /// # Example
    /// ```
    /// use zkir_llvm::target::TargetConfig;
    ///
    /// let target = TargetConfig::DEFAULT;
    /// let spec_config = target.to_spec_config();
    /// assert_eq!(spec_config.limb_bits, 20);
    /// assert_eq!(spec_config.data_limbs, 2);
    /// ```
    pub fn to_spec_config(&self) -> zkir_spec::Config {
        zkir_spec::Config {
            limb_bits: self.limb_bits,
            data_limbs: self.data_limbs,
            addr_limbs: self.addr_limbs,
        }
    }

    /// Create from zkir-spec Config.
    ///
    /// This allows zkir-llvm to accept configurations from zkir-spec,
    /// enabling interoperability between tools using the specification.
    ///
    /// # Example
    /// ```
    /// use zkir_llvm::target::TargetConfig;
    /// use zkir_spec::Config;
    ///
    /// let spec_config = Config::DEFAULT;
    /// let target = TargetConfig::from_spec_config(spec_config);
    /// assert_eq!(target.limb_bits, spec_config.limb_bits);
    /// ```
    pub fn from_spec_config(config: zkir_spec::Config) -> Self {
        Self {
            limb_bits: config.limb_bits,
            data_limbs: config.data_limbs,
            addr_limbs: config.addr_limbs,
        }
    }
}

/// Warnings about configuration that may be suboptimal but not invalid.
#[derive(Debug, Clone)]
pub enum ConfigWarning {
    /// i64 operations will require register splitting
    I64RequiresSplit {
        data_bits: u32,
    },
    /// Low headroom may cause frequent range checks
    LowHeadroom {
        type_bits: u32,
        headroom: u32,
    },
    /// Small table size may increase memory pressure
    SmallTableSize {
        table_size: usize,
    },
    /// Configuration may be expensive (many limbs, large limbs)
    ExpensiveConfig {
        total_bits: u32,
    },
}

impl std::fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::I64RequiresSplit { data_bits } => {
                write!(f, "i64 operations will require register splitting ({}-bit data < 64 bits)", data_bits)
            }
            Self::LowHeadroom { type_bits, headroom } => {
                write!(f, "Low headroom for i{}: only {} bits (may cause frequent range checks)", type_bits, headroom)
            }
            Self::SmallTableSize { table_size } => {
                write!(f, "Small range check table size ({}) may increase memory pressure", table_size)
            }
            Self::ExpensiveConfig { total_bits } => {
                write!(f, "Large configuration ({}-bit) may be expensive in ZK circuits", total_bits)
            }
        }
    }
}

/// Configuration validation errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    #[error("limb_bits must be 16-30, got {0}")]
    InvalidLimbBits(u8),

    #[error("limb_bits must be even, got {0}")]
    OddLimbBits(u8),

    #[error("data_limbs must be 1-4, got {0}")]
    InvalidDataLimbs(u8),

    #[error("addr_limbs must be 1-2, got {0}")]
    InvalidAddrLimbs(u8),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TargetConfig::default();
        assert_eq!(config.limb_bits, 20);
        assert_eq!(config.data_limbs, 2);
        assert_eq!(config.addr_limbs, 2);
        assert_eq!(config.data_bits(), 40);
        assert_eq!(config.addr_bits(), 40);
        assert_eq!(config.chunk_bits(), 10);
        assert_eq!(config.table_size(), 1024);
    }

    #[test]
    fn test_headroom() {
        let config = TargetConfig::default(); // 40-bit
        assert_eq!(config.headroom(32), 8);   // i32: 8 bits headroom
        assert_eq!(config.headroom(40), 0);   // full: no headroom
        assert_eq!(config.headroom(64), 0);   // i64: saturates to 0
    }

    #[test]
    fn test_needs_split() {
        let config_40 = TargetConfig::default(); // 40-bit
        assert!(!config_40.needs_split(32)); // i32 fits
        assert!(!config_40.needs_split(40)); // exact fit
        assert!(config_40.needs_split(64));  // i64 needs split

        let config_80 = TargetConfig::DATA_80; // 80-bit
        assert!(!config_80.needs_split(64)); // i64 fits!
    }

    #[test]
    fn test_regs_for_bits() {
        let config = TargetConfig::default(); // 40-bit
        assert_eq!(config.regs_for_bits(0), 0);
        assert_eq!(config.regs_for_bits(8), 1);
        assert_eq!(config.regs_for_bits(32), 1);
        assert_eq!(config.regs_for_bits(40), 1);
        assert_eq!(config.regs_for_bits(64), 2);
        assert_eq!(config.regs_for_bits(128), 4);
    }

    #[test]
    fn test_validation() {
        // Valid configurations should pass
        assert!(TargetConfig::new(20, 2, 2).validate().is_ok());
        assert!(TargetConfig::DEFAULT.validate().is_ok());
        assert!(TargetConfig::DATA_60.validate().is_ok());
        assert!(TargetConfig::DATA_80.validate().is_ok());
        assert!(TargetConfig::I32_COMPACT.validate().is_ok());
        assert!(TargetConfig::LARGE_LIMB.validate().is_ok());

        // Invalid limb_bits (too small)
        assert!(TargetConfig { limb_bits: 15, ..Default::default() }.validate().is_err());
        assert!(TargetConfig { limb_bits: 14, ..Default::default() }.validate().is_err());

        // Invalid limb_bits (too large)
        assert!(TargetConfig { limb_bits: 31, ..Default::default() }.validate().is_err());
        assert!(TargetConfig { limb_bits: 32, ..Default::default() }.validate().is_err());

        // Invalid limb_bits (odd)
        assert!(TargetConfig { limb_bits: 21, ..Default::default() }.validate().is_err());
        assert!(TargetConfig { limb_bits: 19, ..Default::default() }.validate().is_err());
        assert!(TargetConfig { limb_bits: 23, ..Default::default() }.validate().is_err());

        // Invalid data_limbs (too small)
        assert!(TargetConfig { data_limbs: 0, ..Default::default() }.validate().is_err());

        // Invalid data_limbs (too large)
        assert!(TargetConfig { data_limbs: 5, ..Default::default() }.validate().is_err());
        assert!(TargetConfig { data_limbs: 6, ..Default::default() }.validate().is_err());

        // Invalid addr_limbs (too small)
        assert!(TargetConfig { addr_limbs: 0, ..Default::default() }.validate().is_err());

        // Invalid addr_limbs (too large)
        assert!(TargetConfig { addr_limbs: 3, ..Default::default() }.validate().is_err());

        // Edge cases that should be valid
        assert!(TargetConfig::new(16, 1, 1).validate().is_ok()); // Minimum valid
        assert!(TargetConfig::new(30, 4, 2).validate().is_ok()); // Maximum valid
        assert!(TargetConfig::new(28, 1, 1).validate().is_ok()); // Even limb_bits
    }

    #[test]
    fn test_validation_matches_zkir_spec() {
        // Ensure our validation is consistent with zkir-spec
        let test_cases = vec![
            (16, 1, 1, true),   // Minimum valid
            (30, 4, 2, true),   // Maximum valid
            (20, 2, 2, true),   // Default
            (15, 2, 2, false),  // limb_bits too small
            (31, 2, 2, false),  // limb_bits too large
            (21, 2, 2, false),  // limb_bits odd
            (20, 0, 2, false),  // data_limbs too small
            (20, 5, 2, false),  // data_limbs too large
            (20, 2, 0, false),  // addr_limbs too small
            (20, 2, 3, false),  // addr_limbs too large
        ];

        for (limb_bits, data_limbs, addr_limbs, should_be_valid) in test_cases {
            let target_result = TargetConfig { limb_bits, data_limbs, addr_limbs }.validate();
            let spec_result = zkir_spec::Config::new(limb_bits, data_limbs, addr_limbs);

            assert_eq!(
                target_result.is_ok(),
                spec_result.is_ok(),
                "Validation mismatch for ({}, {}, {}): target={:?}, spec={:?}",
                limb_bits, data_limbs, addr_limbs, target_result, spec_result
            );

            assert_eq!(
                target_result.is_ok(),
                should_be_valid,
                "Expected validity mismatch for ({}, {}, {})",
                limb_bits, data_limbs, addr_limbs
            );
        }
    }

    #[test]
    fn test_presets() {
        assert_eq!(TargetConfig::preset("default"), Some(TargetConfig::DEFAULT));
        assert_eq!(TargetConfig::preset("60bit"), Some(TargetConfig::DATA_60));
        assert_eq!(TargetConfig::preset("80bit"), Some(TargetConfig::DATA_80));
        assert_eq!(TargetConfig::preset("i32-compact"), Some(TargetConfig::I32_COMPACT));
        assert_eq!(TargetConfig::preset("large-limb"), Some(TargetConfig::LARGE_LIMB));
        assert_eq!(TargetConfig::preset("unknown"), None);

        // Case insensitive
        assert_eq!(TargetConfig::preset("Default"), Some(TargetConfig::DEFAULT));
        assert_eq!(TargetConfig::preset("60BIT"), Some(TargetConfig::DATA_60));
    }

    #[test]
    fn test_preset_names() {
        let names = TargetConfig::preset_names();
        assert!(names.contains(&"default"));
        assert!(names.contains(&"60bit"));
        assert!(names.contains(&"80bit"));
    }

    #[test]
    fn test_suggest_for_int_width() {
        // i32 code should get default (40-bit)
        let config = TargetConfig::suggest_for_int_width(32);
        assert_eq!(config.data_bits(), 40);

        // i64 code should get 60-bit or more
        let config = TargetConfig::suggest_for_int_width(64);
        assert!(config.data_bits() >= 64);
    }

    #[test]
    fn test_check_warnings() {
        // Default config (40-bit) should warn about i64 splitting
        let config = TargetConfig::DEFAULT;
        let warnings = config.check_warnings();
        assert!(warnings.iter().any(|w| matches!(w, ConfigWarning::I64RequiresSplit { .. })));

        // 80-bit config should not warn about i64
        let config = TargetConfig::DATA_80;
        let warnings = config.check_warnings();
        assert!(!warnings.iter().any(|w| matches!(w, ConfigWarning::I64RequiresSplit { .. })));

        // Very small config should warn about low headroom
        let config = TargetConfig { limb_bits: 16, data_limbs: 2, addr_limbs: 1 };
        let warnings = config.check_warnings();
        // 32-bit data has 0 headroom for i32
        assert!(warnings.iter().any(|w| matches!(w, ConfigWarning::LowHeadroom { type_bits: 32, .. })));
    }

    #[test]
    fn test_warning_display() {
        let warning = ConfigWarning::I64RequiresSplit { data_bits: 40 };
        let msg = format!("{}", warning);
        assert!(msg.contains("i64"));
        assert!(msg.contains("40"));

        let warning = ConfigWarning::LowHeadroom { type_bits: 32, headroom: 2 };
        let msg = format!("{}", warning);
        assert!(msg.contains("i32"));
        assert!(msg.contains("2 bits"));
    }

    #[test]
    fn test_preset_configs_valid() {
        // All presets should be valid
        for name in TargetConfig::preset_names() {
            let config = TargetConfig::preset(name).expect(&format!("Preset '{}' not found", name));
            assert!(config.validate().is_ok(), "Preset '{}' is invalid", name);
        }
    }

    #[test]
    fn test_to_spec_config() {
        // Test default configuration
        let target = TargetConfig::DEFAULT;
        let spec = target.to_spec_config();
        assert_eq!(spec.limb_bits, 20);
        assert_eq!(spec.data_limbs, 2);
        assert_eq!(spec.addr_limbs, 2);
        assert_eq!(spec.data_bits(), 40);
        assert_eq!(spec.addr_bits(), 40);

        // Test 60-bit configuration
        let target = TargetConfig::DATA_60;
        let spec = target.to_spec_config();
        assert_eq!(spec.limb_bits, 20);
        assert_eq!(spec.data_limbs, 3);
        assert_eq!(spec.addr_limbs, 2);
        assert_eq!(spec.data_bits(), 60);

        // Test 80-bit configuration
        let target = TargetConfig::DATA_80;
        let spec = target.to_spec_config();
        assert_eq!(spec.data_limbs, 4);
        assert_eq!(spec.data_bits(), 80);
    }

    #[test]
    fn test_from_spec_config() {
        // Test default configuration
        let spec = zkir_spec::Config::DEFAULT;
        let target = TargetConfig::from_spec_config(spec);
        assert_eq!(target.limb_bits, 20);
        assert_eq!(target.data_limbs, 2);
        assert_eq!(target.addr_limbs, 2);
        assert_eq!(target.data_bits(), 40);

        // Test custom configuration
        let spec = zkir_spec::Config {
            limb_bits: 30,
            data_limbs: 3,
            addr_limbs: 2,
        };
        let target = TargetConfig::from_spec_config(spec);
        assert_eq!(target.limb_bits, 30);
        assert_eq!(target.data_limbs, 3);
        assert_eq!(target.data_bits(), 90);
    }

    #[test]
    fn test_config_roundtrip() {
        // Test that converting to spec and back preserves values
        let configs = vec![
            TargetConfig::DEFAULT,
            TargetConfig::DATA_60,
            TargetConfig::DATA_80,
            TargetConfig::I32_COMPACT,
            TargetConfig::LARGE_LIMB,
        ];

        for original in configs {
            let spec = original.to_spec_config();
            let roundtrip = TargetConfig::from_spec_config(spec);
            assert_eq!(original, roundtrip, "Roundtrip conversion failed for {:?}", original);
            assert_eq!(original.data_bits(), roundtrip.data_bits());
            assert_eq!(original.addr_bits(), roundtrip.addr_bits());
        }
    }

    #[test]
    fn test_all_presets_convert() {
        // Verify all presets can convert to zkir-spec and back
        for name in TargetConfig::preset_names() {
            let target = TargetConfig::preset(name).unwrap();
            let spec = target.to_spec_config();
            let back = TargetConfig::from_spec_config(spec);
            assert_eq!(target, back, "Preset '{}' failed roundtrip", name);
        }
    }
}
