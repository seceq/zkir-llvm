//! Register allocation for ZKIR.
//!
//! Implements a simple linear scan register allocator that maps virtual
//! registers to physical registers, spilling to the stack when necessary.

mod liveness;
mod linear_scan;

use crate::mir::{Module, MachineFunction};
use crate::target::config::TargetConfig;
use anyhow::Result;

pub use liveness::LivenessInfo;
pub use linear_scan::LinearScanAllocator;

/// Allocate registers for all functions in a module.
pub fn allocate(module: &Module, config: &TargetConfig) -> Result<Module> {
    let mut result = Module::new(&module.name);

    // Copy globals
    result.globals = module.globals.clone();

    // Allocate registers for each function
    for func in module.functions.values() {
        let allocated = allocate_function(func, config)?;
        result.add_function(allocated);
    }

    Ok(result)
}

/// Allocate registers for a single function.
fn allocate_function(func: &MachineFunction, config: &TargetConfig) -> Result<MachineFunction> {
    // Compute liveness information
    let liveness = liveness::compute_liveness(func);

    // Run linear scan allocation
    let mut allocator = LinearScanAllocator::new(func, &liveness, config);
    let allocated = allocator.allocate()?;

    Ok(allocated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{MachineBlock, MachineInst};

    #[test]
    fn test_simple_allocation() {
        let config = TargetConfig::default();

        let mut func = MachineFunction::new("test");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 10));
        entry.push(MachineInst::li(v1, 20));
        entry.push(MachineInst::add(v2, v0, v1));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        let mut module = Module::new("test");
        module.add_function(func);

        let result = allocate(&module, &config).unwrap();
        assert!(result.functions.contains_key("test"));
    }
}
