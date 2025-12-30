//! Optimization passes for Machine IR.
//!
//! This module contains ZK-specific and general optimization passes
//! that run on the Machine IR before register allocation.

mod phi_elimination;
mod range_check;
mod dead_code;
mod int_splitting;
mod signed_ops;
mod const_fold;
mod peephole;
mod loop_unroll;
mod copy_prop;
mod cse;
mod inline;
mod licm;
mod bounds_narrow;
mod mem_coalesce;

use crate::mir::{MachineFunction, Module};
use crate::target::config::TargetConfig;
use anyhow::Result;

/// Optimization level configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations - fastest compilation, largest output.
    O0,
    /// Basic optimizations - good for debugging.
    O1,
    /// Standard optimizations - balance of speed and quality.
    #[default]
    O2,
    /// Aggressive optimizations - smaller output, slower compilation.
    O3,
    /// Size-optimized - minimize constraint count at any cost.
    Os,
}

impl OptLevel {
    /// Parse from a string like "0", "1", "2", "3", or "s".
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "0" => Some(OptLevel::O0),
            "1" => Some(OptLevel::O1),
            "2" => Some(OptLevel::O2),
            "3" => Some(OptLevel::O3),
            "s" | "S" => Some(OptLevel::Os),
            _ => None,
        }
    }

    /// Whether to run inlining.
    pub fn inline(&self) -> bool {
        !matches!(self, OptLevel::O0)
    }

    /// Whether to run loop unrolling.
    pub fn unroll(&self) -> bool {
        matches!(self, OptLevel::O2 | OptLevel::O3 | OptLevel::Os)
    }

    /// Whether to run LICM.
    pub fn licm(&self) -> bool {
        matches!(self, OptLevel::O2 | OptLevel::O3 | OptLevel::Os)
    }

    /// Whether to run CSE.
    pub fn cse(&self) -> bool {
        !matches!(self, OptLevel::O0)
    }

    /// Whether to run global CSE (more expensive).
    pub fn global_cse(&self) -> bool {
        matches!(self, OptLevel::O2 | OptLevel::O3 | OptLevel::Os)
    }

    /// Whether to run copy propagation.
    pub fn copy_prop(&self) -> bool {
        !matches!(self, OptLevel::O0)
    }

    /// Whether to run global copy propagation.
    pub fn global_copy_prop(&self) -> bool {
        matches!(self, OptLevel::O2 | OptLevel::O3 | OptLevel::Os)
    }

    /// Whether to run aggressive peephole optimizations.
    pub fn peephole(&self) -> bool {
        !matches!(self, OptLevel::O0)
    }

    /// Whether to run memory coalescing (combine byte loads/stores into word).
    pub fn mem_coalesce(&self) -> bool {
        matches!(self, OptLevel::O2 | OptLevel::O3 | OptLevel::Os)
    }
}

impl std::fmt::Display for OptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptLevel::O0 => write!(f, "-O0"),
            OptLevel::O1 => write!(f, "-O1"),
            OptLevel::O2 => write!(f, "-O2"),
            OptLevel::O3 => write!(f, "-O3"),
            OptLevel::Os => write!(f, "-Os"),
        }
    }
}

pub use phi_elimination::eliminate_phis;
pub use range_check::insert_range_checks;
pub use dead_code::eliminate_dead_code;
pub use int_splitting::split_large_integers;
pub use signed_ops::lower_signed_ops;
pub use const_fold::{fold_constants, simplify_algebra};
pub use peephole::peephole_optimize;
pub use loop_unroll::unroll_loops;
pub use copy_prop::{propagate_copies, remove_dead_copies, global_copy_propagation};
pub use cse::{eliminate_common_subexpressions, global_cse};
pub use inline::inline_functions;
pub use licm::hoist_loop_invariants;
pub use bounds_narrow::narrow_bounds;
pub use mem_coalesce::coalesce_memory;

/// Run all optimization passes on a module with default optimization level (O2).
pub fn optimize(module: &mut Module, config: &TargetConfig) -> Result<()> {
    optimize_with_level(module, config, OptLevel::default())
}

/// Run optimization passes on a module with a specific optimization level.
pub fn optimize_with_level(module: &mut Module, config: &TargetConfig, opt: OptLevel) -> Result<()> {
    // Module-level optimizations
    if opt.inline() {
        inline_functions(module)?;
    }

    // Function-level optimizations
    for func in module.functions.values_mut() {
        optimize_function_with_level(func, config, opt)?;
    }
    Ok(())
}

/// Run all optimization passes on a single function with default optimization level.
pub fn optimize_function(func: &mut MachineFunction, config: &TargetConfig) -> Result<()> {
    optimize_function_with_level(func, config, OptLevel::default())
}

/// Run optimization passes on a single function with a specific optimization level.
pub fn optimize_function_with_level(
    func: &mut MachineFunction,
    config: &TargetConfig,
    opt: OptLevel,
) -> Result<()> {
    // Phase 1: SSA destruction (PHI elimination) - always needed
    eliminate_phis(func)?;

    // Phase 1.5: Loop optimizations
    if opt.licm() {
        hoist_loop_invariants(func)?;
    }
    if opt.unroll() {
        unroll_loops(func)?;
    }

    // Phase 2: Algebraic simplifications and constant folding (always beneficial)
    simplify_algebra(func)?;
    fold_constants(func)?;

    // Phase 2.5: Peephole optimizations (strength reduction, instruction combining)
    if opt.peephole() {
        peephole_optimize(func)?;
    }

    // Phase 2.55: Memory coalescing (combine byte loads/stores into word)
    if opt.mem_coalesce() {
        coalesce_memory(func)?;
    }

    // Phase 2.6: Common subexpression elimination
    if opt.cse() {
        eliminate_common_subexpressions(func)?;
    }
    if opt.global_cse() {
        global_cse(func)?;
    }

    // Phase 2.7: Copy propagation (eliminate MOV chains)
    if opt.copy_prop() {
        propagate_copies(func)?;
        remove_dead_copies(func)?;
    }
    if opt.global_copy_prop() {
        global_copy_propagation(func)?;
        remove_dead_copies(func)?;
    }

    // Phase 3: Large integer splitting (i64 on 40-bit configs) - always needed
    split_large_integers(func, config)?;

    // Phase 4: Signed operation lowering - always needed
    lower_signed_ops(func, config)?;

    // Phase 4.5: Bounds narrowing from comparisons (reduces range checks)
    if opt.cse() {
        narrow_bounds(func)?;
    }

    // Phase 5: ZK-specific optimizations - always needed
    insert_range_checks(func, config)?;

    // Phase 6: Cleanup - always beneficial
    eliminate_dead_code(func)?;

    // Rebuild CFG after modifications
    func.rebuild_cfg();

    Ok(())
}

/// Pass trait for implementing custom optimization passes.
pub trait Pass {
    /// Name of the pass for debugging.
    fn name(&self) -> &'static str;

    /// Run the pass on a function. Returns true if the function was modified.
    fn run(&mut self, func: &mut MachineFunction, config: &TargetConfig) -> Result<bool>;
}

/// Pass manager for running multiple passes.
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    /// Create a new pass manager.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the manager.
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Run all passes on a function.
    pub fn run(&mut self, func: &mut MachineFunction, config: &TargetConfig) -> Result<()> {
        for pass in &mut self.passes {
            let modified = pass.run(func, config)?;
            if modified {
                log::debug!("Pass '{}' modified function '{}'", pass.name(), func.name);
            }
        }
        Ok(())
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_level_parse() {
        assert_eq!(OptLevel::from_str("0"), Some(OptLevel::O0));
        assert_eq!(OptLevel::from_str("1"), Some(OptLevel::O1));
        assert_eq!(OptLevel::from_str("2"), Some(OptLevel::O2));
        assert_eq!(OptLevel::from_str("3"), Some(OptLevel::O3));
        assert_eq!(OptLevel::from_str("s"), Some(OptLevel::Os));
        assert_eq!(OptLevel::from_str("S"), Some(OptLevel::Os));
        assert_eq!(OptLevel::from_str("x"), None);
        assert_eq!(OptLevel::from_str(""), None);
    }

    #[test]
    fn test_opt_level_flags() {
        // O0 should disable almost everything
        assert!(!OptLevel::O0.inline());
        assert!(!OptLevel::O0.unroll());
        assert!(!OptLevel::O0.licm());
        assert!(!OptLevel::O0.cse());
        assert!(!OptLevel::O0.peephole());

        // O1 should enable basic optimizations
        assert!(OptLevel::O1.inline());
        assert!(!OptLevel::O1.unroll());
        assert!(!OptLevel::O1.licm());
        assert!(OptLevel::O1.cse());
        assert!(!OptLevel::O1.global_cse());

        // O2 should enable most optimizations
        assert!(OptLevel::O2.inline());
        assert!(OptLevel::O2.unroll());
        assert!(OptLevel::O2.licm());
        assert!(OptLevel::O2.cse());
        assert!(OptLevel::O2.global_cse());
        assert!(OptLevel::O2.copy_prop());
        assert!(OptLevel::O2.global_copy_prop());
        assert!(OptLevel::O2.mem_coalesce());

        // O3 should enable all optimizations
        assert!(OptLevel::O3.inline());
        assert!(OptLevel::O3.unroll());
        assert!(OptLevel::O3.licm());

        // Os should match O2 for now
        assert!(OptLevel::Os.unroll());
        assert!(OptLevel::Os.licm());
    }

    #[test]
    fn test_opt_level_display() {
        assert_eq!(format!("{}", OptLevel::O0), "-O0");
        assert_eq!(format!("{}", OptLevel::O1), "-O1");
        assert_eq!(format!("{}", OptLevel::O2), "-O2");
        assert_eq!(format!("{}", OptLevel::O3), "-O3");
        assert_eq!(format!("{}", OptLevel::Os), "-Os");
    }

    #[test]
    fn test_opt_level_default() {
        assert_eq!(OptLevel::default(), OptLevel::O2);
    }
}
