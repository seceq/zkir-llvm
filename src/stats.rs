//! Compilation statistics.
//!
//! Tracks metrics during compilation for verbose output.

use std::time::{Duration, Instant};

/// Compilation statistics.
#[derive(Debug, Default)]
pub struct CompileStats {
    /// Total compilation time
    pub total_time: Duration,
    /// Time spent in lowering
    pub lower_time: Duration,
    /// Time spent in optimization
    pub opt_time: Duration,
    /// Time spent in register allocation
    pub regalloc_time: Duration,
    /// Time spent in emission
    pub emit_time: Duration,

    /// Number of functions
    pub num_functions: usize,
    /// Number of basic blocks
    pub num_blocks: usize,
    /// Number of instructions (before optimization)
    pub num_insts_before: usize,
    /// Number of instructions (after optimization)
    pub num_insts_after: usize,
    /// Number of virtual registers
    pub num_vregs: usize,
    /// Number of range checks inserted
    pub num_range_checks: usize,
    /// Number of spills
    pub num_spills: usize,
    /// Output size in bytes
    pub output_size: usize,
}

impl CompileStats {
    /// Create a new stats tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Display statistics.
    pub fn display(&self) {
        eprintln!("\n=== Compilation Statistics ===");
        eprintln!("Functions:    {}", self.num_functions);
        eprintln!("Blocks:       {}", self.num_blocks);
        eprintln!("Instructions: {} → {} ({:.1}% reduction)",
            self.num_insts_before,
            self.num_insts_after,
            if self.num_insts_before > 0 {
                100.0 * (1.0 - (self.num_insts_after as f64 / self.num_insts_before as f64))
            } else {
                0.0
            }
        );
        eprintln!("VRegs:        {}", self.num_vregs);
        eprintln!("Range checks: {}", self.num_range_checks);
        eprintln!("Spills:       {}", self.num_spills);
        eprintln!("Output size:  {} bytes", self.output_size);
        eprintln!();
        eprintln!("=== Timing ===");
        eprintln!("Lowering:     {:?}", self.lower_time);
        eprintln!("Optimization: {:?}", self.opt_time);
        eprintln!("Regalloc:     {:?}", self.regalloc_time);
        eprintln!("Emission:     {:?}", self.emit_time);
        eprintln!("Total:        {:?}", self.total_time);
    }
}

/// Timer helper for measuring phase durations.
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer.
    pub fn start() -> Self {
        Self { start: Instant::now() }
    }

    /// Stop the timer and return elapsed duration.
    pub fn stop(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Count instructions in a module.
pub fn count_instructions(module: &crate::mir::Module) -> usize {
    module.functions.values()
        .map(|f| f.iter_blocks().map(|b| b.insts.len()).sum::<usize>())
        .sum()
}

/// Count blocks in a module.
pub fn count_blocks(module: &crate::mir::Module) -> usize {
    module.functions.values()
        .map(|f| f.blocks.len())
        .sum()
}

/// Count vregs in a module.
pub fn count_vregs(module: &crate::mir::Module) -> usize {
    module.functions.values()
        .map(|f| f.num_vregs() as usize)
        .sum()
}

/// Count range check instructions.
pub fn count_range_checks(module: &crate::mir::Module) -> usize {
    use crate::mir::Opcode;

    module.functions.values()
        .flat_map(|f| f.iter_blocks())
        .flat_map(|b| b.insts.iter())
        .filter(|i| i.opcode == Opcode::RCHK)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{MachineBlock, MachineFunction, MachineInst, Module, Opcode, Operand};

    #[test]
    fn test_compile_stats_default() {
        let stats = CompileStats::new();
        assert_eq!(stats.num_functions, 0);
        assert_eq!(stats.num_blocks, 0);
        assert_eq!(stats.num_insts_before, 0);
        assert_eq!(stats.num_insts_after, 0);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_count_instructions() {
        let mut module = Module::new("test");

        let mut func = MachineFunction::new("main");
        let v0 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 42));
        entry.push(MachineInst::ret());
        func.add_block(entry);

        module.add_function(func);

        assert_eq!(count_instructions(&module), 2);
    }

    #[test]
    fn test_count_blocks() {
        let mut module = Module::new("test");

        let mut func = MachineFunction::new("main");
        let v0 = func.new_vreg();
        let dummy = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 1));
        entry.push(MachineInst::new(Opcode::JAL)
            .dst(Operand::VReg(dummy))
            .src(Operand::Label("exit".to_string())));
        func.add_block(entry);

        let mut exit = MachineBlock::new("exit");
        exit.push(MachineInst::ret());
        func.add_block(exit);

        module.add_function(func);

        assert_eq!(count_blocks(&module), 2);
    }

    #[test]
    fn test_count_vregs() {
        let mut module = Module::new("test");

        let mut func = MachineFunction::new("main");
        let _v0 = func.new_vreg();
        let _v1 = func.new_vreg();
        let _v2 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::ret());
        func.add_block(entry);

        module.add_function(func);

        assert_eq!(count_vregs(&module), 3);
    }

    #[test]
    fn test_count_range_checks() {
        let mut module = Module::new("test");

        let mut func = MachineFunction::new("main");
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let mut entry = MachineBlock::new("entry");
        entry.push(MachineInst::li(v0, 100));
        entry.push(MachineInst::rchk(v0)); // Range check
        entry.push(MachineInst::li(v1, 200));
        entry.push(MachineInst::rchk(v1)); // Another range check
        entry.push(MachineInst::ret());
        func.add_block(entry);

        module.add_function(func);

        assert_eq!(count_range_checks(&module), 2);
    }

    #[test]
    fn test_stats_reduction_calculation() {
        let mut stats = CompileStats::new();
        stats.num_insts_before = 100;
        stats.num_insts_after = 75;

        // The reduction should be 25%
        let reduction = 100.0 * (1.0 - (stats.num_insts_after as f64 / stats.num_insts_before as f64));
        assert!((reduction - 25.0).abs() < 0.01);
    }
}
