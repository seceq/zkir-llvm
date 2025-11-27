//! Basic block representation

use super::Instruction;

#[derive(Debug, Clone)]
pub struct BasicBlock {
    name: String,
    instructions: Vec<Instruction>,
}

impl BasicBlock {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: Vec::new(),
        }
    }

    pub fn with_instructions(name: impl Into<String>, instructions: Vec<Instruction>) -> Self {
        Self {
            name: name.into(),
            instructions,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    pub fn instructions_mut(&mut self) -> &mut Vec<Instruction> {
        &mut self.instructions
    }

    pub fn add_instruction(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }

    /// Get the terminator instruction (last instruction)
    pub fn terminator(&self) -> Option<&Instruction> {
        self.instructions.last()
    }

    /// Check if this block has a terminator
    pub fn is_terminated(&self) -> bool {
        self.instructions
            .last()
            .map(|i| i.is_terminator())
            .unwrap_or(false)
    }
}
