//! LLVM Module representation

use super::Function;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Module {
    /// Functions in this module
    functions: Vec<Function>,

    /// Global variables
    globals: HashMap<String, GlobalVariable>,

    /// Module name
    name: String,
}

#[derive(Debug, Clone)]
pub struct GlobalVariable {
    pub name: String,
    pub ty: super::Type,
    pub initializer: Option<super::Value>,
    pub is_constant: bool,
}

impl Module {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            globals: HashMap::new(),
            name: String::new(),
        }
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            functions: Vec::new(),
            globals: HashMap::new(),
            name: name.into(),
        }
    }

    pub fn add_function(&mut self, func: Function) {
        self.functions.push(func);
    }

    pub fn add_global(&mut self, global: GlobalVariable) {
        self.globals.insert(global.name.clone(), global);
    }

    pub fn functions(&self) -> &[Function] {
        &self.functions
    }

    pub fn functions_mut(&mut self) -> &mut Vec<Function> {
        &mut self.functions
    }

    pub fn get_function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name() == name)
    }

    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.functions.iter_mut().find(|f| f.name() == name)
    }

    pub fn globals(&self) -> &HashMap<String, GlobalVariable> {
        &self.globals
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }
}

impl Default for Module {
    fn default() -> Self {
        Self::new()
    }
}
