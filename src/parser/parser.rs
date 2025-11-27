//! LLVM IR parser implementation

use super::{ParseError, ParseResult};
use crate::ir::*;
use super::lexer::{Lexer, Token};

pub struct Parser<'input> {
    lexer: std::iter::Peekable<Lexer<'input>>,
    current_line: usize,
}

impl<'input> Parser<'input> {
    pub fn new(input: &'input str) -> Self {
        Self {
            lexer: Lexer::new(input).peekable(),
            current_line: 1,
        }
    }

    fn peek(&mut self) -> ParseResult<&Token> {
        match self.lexer.peek() {
            Some(Ok((_, tok, _))) => Ok(tok),
            Some(Err(e)) => Err(ParseError::LexerError(e.clone())),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn next(&mut self) -> ParseResult<Token> {
        match self.lexer.next() {
            Some(Ok((_, tok, _))) => Ok(tok),
            Some(Err(e)) => Err(ParseError::LexerError(e)),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn expect(&mut self, expected: Token) -> ParseResult<()> {
        let tok = self.next()?;
        if std::mem::discriminant(&tok) == std::mem::discriminant(&expected) {
            Ok(())
        } else {
            Err(ParseError::ParseError {
                line: self.current_line,
                message: format!("Expected {:?}, got {:?}", expected, tok),
            })
        }
    }

    pub fn parse_module(&mut self) -> ParseResult<Module> {
        let mut module = Module::new();

        while self.lexer.peek().is_some() {
            match self.peek()? {
                Token::Define => {
                    let func = self.parse_function()?;
                    module.add_function(func);
                }
                Token::Declare => {
                    // Skip declarations for now
                    self.skip_declaration()?;
                }
                _ => {
                    // Skip other top-level constructs (metadata, attributes, etc.)
                    self.next()?;
                }
            }
        }

        Ok(module)
    }

    fn parse_function(&mut self) -> ParseResult<Function> {
        self.expect(Token::Define)?;

        // Parse return type
        let ret_ty = self.parse_type()?;

        // Parse function name
        let name = match self.next()? {
            Token::GlobalIdent(s) => s[1..].to_string(), // Remove '@'
            tok => {
                return Err(ParseError::ParseError {
                    line: self.current_line,
                    message: format!("Expected function name, got {:?}", tok),
                })
            }
        };

        // Parse parameters
        self.expect(Token::LParen)?;
        let params = self.parse_parameters()?;
        self.expect(Token::RParen)?;

        // Parse body
        self.expect(Token::LBrace)?;
        let blocks = self.parse_basic_blocks()?;
        self.expect(Token::RBrace)?;

        Ok(Function::new(name, ret_ty, params, blocks))
    }

    fn parse_type(&mut self) -> ParseResult<Type> {
        match self.next()? {
            Token::Void => Ok(Type::Void),
            Token::IntType(bits) => Ok(Type::Int(bits)),
            Token::Ptr => Ok(Type::Ptr),
            Token::LBracket => {
                // Array type: [N x T]
                let size = match self.next()? {
                    Token::Integer(n) => n as usize,
                    tok => {
                        return Err(ParseError::InvalidType(format!(
                            "Expected array size, got {:?}",
                            tok
                        )))
                    }
                };
                self.expect(Token::Star)?; // 'x'
                let elem_ty = self.parse_type()?;
                self.expect(Token::RBracket)?;
                Ok(Type::Array(size, Box::new(elem_ty)))
            }
            Token::LBrace => {
                // Struct type: { T1, T2, ... }
                let mut fields = Vec::new();
                loop {
                    if matches!(self.peek()?, Token::RBrace) {
                        break;
                    }
                    fields.push(self.parse_type()?);
                    if matches!(self.peek()?, Token::Comma) {
                        self.next()?;
                    }
                }
                self.expect(Token::RBrace)?;
                Ok(Type::Struct(fields))
            }
            tok => Err(ParseError::InvalidType(format!("{:?}", tok))),
        }
    }

    fn parse_parameters(&mut self) -> ParseResult<Vec<(String, Type)>> {
        let mut params = Vec::new();

        if matches!(self.peek()?, Token::RParen) {
            return Ok(params);
        }

        loop {
            let ty = self.parse_type()?;
            let name = match self.next()? {
                Token::LocalIdent(s) => s[1..].to_string(), // Remove '%'
                Token::NumericIdent(s) => s[1..].to_string(),
                tok => {
                    return Err(ParseError::ParseError {
                        line: self.current_line,
                        message: format!("Expected parameter name, got {:?}", tok),
                    })
                }
            };

            params.push((name, ty));

            if !matches!(self.peek()?, Token::Comma) {
                break;
            }
            self.next()?; // Consume comma
        }

        Ok(params)
    }

    fn parse_basic_blocks(&mut self) -> ParseResult<Vec<BasicBlock>> {
        let mut blocks = Vec::new();

        // TODO: Implement basic block parsing
        // For now, create a stub that skips until '}'

        while !matches!(self.peek()?, Token::RBrace) {
            self.next()?;
        }

        Ok(blocks)
    }

    fn skip_declaration(&mut self) -> ParseResult<()> {
        // Skip until we find a newline or next define
        while self.lexer.peek().is_some() {
            if matches!(self.peek()?, Token::Define) {
                break;
            }
            self.next()?;
        }
        Ok(())
    }
}

/// Parse LLVM IR source code into a Module
pub fn parse(source: &str) -> ParseResult<Module> {
    let mut parser = Parser::new(source);
    parser.parse_module()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_function() {
        let source = r#"
            define void @main() {
            }
        "#;

        let module = parse(source).unwrap();
        assert_eq!(module.functions().len(), 1);
    }

    #[test]
    fn test_parse_types() {
        let source = "define i32 @test(i32 %x, ptr %p) { }";
        let module = parse(source).unwrap();
        let func = &module.functions()[0];
        assert_eq!(func.name(), "test");
        assert_eq!(func.params().len(), 2);
    }
}
