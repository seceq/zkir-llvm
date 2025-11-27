//! Function representation

use super::{BasicBlock, Type};

#[derive(Debug, Clone)]
pub struct Function {
    name: String,
    ret_ty: Type,
    params: Vec<(String, Type)>,
    blocks: Vec<BasicBlock>,
}

impl Function {
    pub fn new(
        name: impl Into<String>,
        ret_ty: Type,
        params: Vec<(String, Type)>,
        blocks: Vec<BasicBlock>,
    ) -> Self {
        Self {
            name: name.into(),
            ret_ty,
            params,
            blocks,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ret_ty(&self) -> &Type {
        &self.ret_ty
    }

    pub fn params(&self) -> &[(String, Type)] {
        &self.params
    }

    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut Vec<BasicBlock> {
        &mut self.blocks
    }

    pub fn add_block(&mut self, block: BasicBlock) {
        self.blocks.push(block);
    }

    pub fn get_block(&self, name: &str) -> Option<&BasicBlock> {
        self.blocks.iter().find(|b| b.name() == name)
    }

    pub fn get_block_mut(&mut self, name: &str) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|b| b.name() == name)
    }

    /// Get entry block (first block)
    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.blocks.first()
    }

    /// Check if function is a declaration (no body)
    pub fn is_declaration(&self) -> bool {
        self.blocks.is_empty()
    }
}
