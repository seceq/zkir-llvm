//! LLVM IR parser

pub mod lexer;
pub mod parser;

pub use parser::parse;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Lexer error: {0}")]
    LexerError(String),

    #[error("Parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("Unexpected token: {0}")]
    UnexpectedToken(String),

    #[error("Unexpected end of input")]
    UnexpectedEof,

    #[error("Invalid type: {0}")]
    InvalidType(String),

    #[error("Invalid instruction: {0}")]
    InvalidInstruction(String),
}

pub type ParseResult<T> = Result<T, ParseError>;
