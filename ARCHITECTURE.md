# Architecture Documentation

## Overview

zkir-llvm is structured as a multi-stage compiler pipeline that transforms LLVM IR into ZK IR bytecode.

## Pipeline Stages

```
┌─────────────┐
│  LLVM IR    │
│  (text)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Lexer     │  Tokenize source
│  (logos)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Parser    │  Build AST
│   (nom)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LLVM IR    │  Internal representation
│    AST      │  (Module → Function → Block → Instruction)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Type Lower  │  LLVM types → ZK types
│             │  (i64 → pair, i128 → quad)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Translate  │  Generate ZK IR instructions
│             │  (virtual registers)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  RegAlloc   │  Allocate physical registers
│             │  (linear scan + spilling)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Encode    │  Encode to RISC-V format
│             │  (32-bit instructions)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Emit      │  Serialize to bytecode
│  (bincode)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ZK IR      │
│  Bytecode   │
└─────────────┘
```

## Module Structure

### Parser Module (`src/parser/`)

**Lexer** (`lexer.rs`):
- Uses `logos` for fast tokenization
- Produces `Token` stream
- Handles comments, whitespace
- Recognizes LLVM IR keywords, types, identifiers

**Parser** (`parser.rs`):
- Uses `nom` parser combinators (planned)
- Recursive descent for structure
- Builds IR AST from tokens
- Error recovery and reporting

### IR Module (`src/ir/`)

**Type System** (`types.rs`):
- LLVM type representation
- Bit width calculations
- Size computations
- Supported type checking

**Module** (`module.rs`):
- Top-level container
- Function collection
- Global variables
- Module metadata

**Function** (`function.rs`):
- Function signature
- Parameter list
- Basic block collection
- Entry block management

**BasicBlock** (`block.rs`):
- Linear sequence of instructions
- Terminator instruction
- Label/name
- Predecessors/successors (planned)

**Instruction** (`instruction.rs`):
- LLVM instruction representation
- Arithmetic, bitwise, memory, control flow
- SSA value naming
- Instruction properties

**Value** (`mod.rs`):
- SSA values
- Constants (i32, i64, bool)
- Local references (%var)
- Undef, Null

### Translation Module (`src/translate/`)

**Context** (`context.rs`):
- Per-function translation state
- Value → Location mapping
- Instruction emission
- Label tracking
- Fixup recording
- Register allocation

**Type Lowering** (`types.rs`):
- LLVM Type → ZK Type
- Scalar types → registers
- Aggregate types → memory
- Size calculations

**Arithmetic** (`arithmetic.rs`):
- 32-bit operations: direct mapping
- 64-bit operations: register pairs
  ```
  add64(a, b):
    lo = a.lo + b.lo
    carry = (lo < a.lo) ? 1 : 0
    hi = a.hi + b.hi + carry
  ```
- Bitwise operations
- Shift operations

**Memory** (`memory.rs`):
- Load: `lw/lh/lhu/lb/lbu`
- Store: `sw/sh/sb`
- Stack allocation: `alloca`
- Address computation

**Control Flow** (`control.rs`):
- Branches: `beq/bne/blt/bge/bltu/bgeu`
- Jumps: `jal/jalr`
- Returns: move to rv + jalr
- Comparisons: `slt/sltu` + logic
- Function calls: arg passing + jal

**Intrinsics** (`intrinsics.rs`):
- LLVM intrinsic handling
- Ignore debug intrinsics
- Ignore lifetime intrinsics
- Error on unsupported

### Register Allocation Module (`src/regalloc/`)

**Linear Scan** (`linear.rs`):
- Sort live intervals by start
- Allocate from free register pool
- Spill when out of registers
- Track active intervals
- Expire old intervals

Algorithm:
```rust
for interval in sorted_intervals {
    expire_old_intervals(interval.start)
    
    if free_regs.is_empty() {
        spill_at_interval(interval)
    } else {
        reg = free_regs.pop()
        assign(interval, reg)
        active.push(interval)
    }
}
```

### Emit Module (`src/emit/`)

**Encoding**:
- ZK IR instructions → u32 words
- RISC-V encoding format
- Opcode, funct3, funct7 fields
- Register indices
- Immediate values

**Serialization**:
- Program → bytecode
- Uses `bincode` for serialization
- Program header + code + data

## Data Flow

### Example: `add i32 %a, %b`

1. **Lexer**:
   ```
   Token::Add, Token::IntType(32), Token::LocalIdent("%a"), 
   Token::Comma, Token::LocalIdent("%b")
   ```

2. **Parser**:
   ```rust
   Instruction::Add {
       result: "%sum",
       ty: Type::Int(32),
       lhs: Value::Local("a"),
       rhs: Value::Local("b"),
   }
   ```

3. **Translation**:
   ```rust
   rs1 = ctx.load_value("%a")  // → r8 (t0)
   rs2 = ctx.load_value("%b")  // → r9 (t1)
   rd = ctx.alloc_temp()       // → r10 (t2)
   ctx.emit(Instruction::Add { rd, rs1, rs2 })
   ctx.bind("%sum", Location::Reg(rd))
   ```

4. **RegAlloc**:
   ```
   Virtual r10 → Physical r10 (t2)
   ```

5. **Encoding**:
   ```
   opcode: 0x33 (R-type)
   rd:     10   (destination)
   funct3: 0x0  (ADD)
   rs1:    8    (source 1)
   rs2:    9    (source 2)
   funct7: 0x00 (ADD)
   
   Encoded: 0x009403B3
   ```

## Key Design Decisions

### 1. SSA Form
- LLVM IR is in SSA (Static Single Assignment)
- Each value defined once
- Phi nodes for control flow merges
- We lower phi nodes to moves before regalloc

### 2. Virtual Registers
- Translation uses unlimited virtual registers
- Register allocator maps to 20 physical registers
- Spills to stack when out of registers

### 3. Two-Stage Translation
- Stage 1: LLVM IR → ZK IR (virtual regs)
- Stage 2: Register allocation → physical regs

### 4. Type Lowering
- i1, i8, i16, i32 → single 32-bit register
- i64 → register pair (lo, hi)
- i128 → register quad (r0, r1, r2, r3)
- Aggregates → memory pointers

### 5. Calling Convention
- First 4 args in a0-a3 (r4-r7)
- Additional args on stack
- Return value in rv (r1)
- 64-bit return in (rv, a0)

### 6. Stack Layout
```
High Address
┌────────────┐
│   Args     │  Arguments beyond a0-a3
├────────────┤
│ Return Addr│  Saved by caller
├────────────┤ ← FP
│  Locals    │  Local variables
├────────────┤
│  Spills    │  Spilled registers
├────────────┤ ← SP
│    ...     │
Low Address
```

## Performance Considerations

### Parser
- Zero-copy tokens where possible
- Minimal allocations during parsing
- Lazy evaluation of unused fields

### Translation
- Single pass through IR
- Direct emission (no IR transformation)
- Minimal intermediate structures

### Register Allocation
- Linear scan for speed
- O(n log n) complexity
- Simple spilling heuristics

## Extension Points

### Adding New Instructions
1. Add token to lexer (`lexer.rs`)
2. Add variant to `Instruction` enum (`instruction.rs`)
3. Add parser case (`parser.rs`)
4. Add translation function (`arithmetic.rs`, etc.)
5. Add tests

### Adding New Types
1. Add variant to `Type` enum (`types.rs`)
2. Implement size/support methods
3. Add to type lowering (`translate/types.rs`)
4. Add load/store support (`memory.rs`)

### Adding Optimizations
1. Create optimization module
2. Add pass infrastructure
3. Integrate with translation pipeline
4. Add opt-level gating

## Testing Strategy

### Unit Tests
- Per-module tests in `src/*/mod.rs`
- Test individual functions
- Mock dependencies

### Integration Tests  
- `tests/*.rs` files
- End-to-end scenarios
- Real LLVM IR inputs

### Property Tests
- (Planned) Use `proptest`
- Random IR generation
- Invariant checking

## Future Directions

### Short Term
- Complete instruction parsing
- Phi node lowering
- Label resolution

### Medium Term
- GEP support
- LLVM intrinsics
- Optimization passes

### Long Term
- Alternative register allocators
- Advanced optimizations
- JIT compilation
