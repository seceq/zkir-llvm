//! Debug information and source mapping.
//!
//! Tracks source locations through compilation for debugging and error reporting.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Source location information.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLoc {
    /// Source file path
    pub file: String,
    /// Line number (1-indexed)
    pub line: u32,
    /// Column number (1-indexed)
    pub column: u32,
}

impl SourceLoc {
    /// Create a new source location.
    pub fn new(file: impl Into<String>, line: u32, column: u32) -> Self {
        Self {
            file: file.into(),
            line,
            column,
        }
    }

    /// Create an unknown location.
    pub fn unknown() -> Self {
        Self {
            file: String::new(),
            line: 0,
            column: 0,
        }
    }

    /// Check if this is an unknown/invalid location.
    pub fn is_unknown(&self) -> bool {
        self.line == 0 && self.column == 0
    }
}

impl std::fmt::Display for SourceLoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_unknown() {
            write!(f, "<unknown>")
        } else if self.file.is_empty() {
            write!(f, "{}:{}", self.line, self.column)
        } else {
            write!(f, "{}:{}:{}", self.file, self.line, self.column)
        }
    }
}

/// Debug information builder.
///
/// Collects source location mappings during compilation.
pub struct DebugInfo {
    /// Map from instruction index to source location
    inst_locations: HashMap<usize, SourceLoc>,
    /// Map from block label to source location
    block_locations: HashMap<String, SourceLoc>,
    /// Map from bytecode offset to source location
    bytecode_map: Vec<(u32, SourceLoc)>,
    /// Current source file being compiled
    current_file: String,
    /// Current line number
    current_line: u32,
}

impl DebugInfo {
    /// Create a new debug info builder.
    pub fn new() -> Self {
        Self {
            inst_locations: HashMap::new(),
            block_locations: HashMap::new(),
            bytecode_map: Vec::new(),
            current_file: String::new(),
            current_line: 0,
        }
    }

    /// Set the current source file.
    pub fn set_file(&mut self, file: impl Into<String>) {
        self.current_file = file.into();
    }

    /// Set the current line number.
    pub fn set_line(&mut self, line: u32) {
        self.current_line = line;
    }

    /// Get the current source location.
    pub fn current_loc(&self) -> SourceLoc {
        if self.current_line == 0 {
            SourceLoc::unknown()
        } else {
            SourceLoc::new(&self.current_file, self.current_line, 0)
        }
    }

    /// Record location for an instruction index.
    pub fn record_inst(&mut self, inst_idx: usize, loc: SourceLoc) {
        if !loc.is_unknown() {
            self.inst_locations.insert(inst_idx, loc);
        }
    }

    /// Record location for a block.
    pub fn record_block(&mut self, label: impl Into<String>, loc: SourceLoc) {
        if !loc.is_unknown() {
            self.block_locations.insert(label.into(), loc);
        }
    }

    /// Record bytecode offset to source mapping.
    pub fn record_bytecode(&mut self, offset: u32, loc: SourceLoc) {
        if !loc.is_unknown() {
            // Avoid duplicate entries for same offset
            if self.bytecode_map.last().map(|(o, _)| *o) != Some(offset) {
                self.bytecode_map.push((offset, loc));
            }
        }
    }

    /// Look up source location for an instruction.
    pub fn get_inst_loc(&self, inst_idx: usize) -> Option<&SourceLoc> {
        self.inst_locations.get(&inst_idx)
    }

    /// Look up source location for a block.
    pub fn get_block_loc(&self, label: &str) -> Option<&SourceLoc> {
        self.block_locations.get(label)
    }

    /// Find source location for a bytecode offset.
    pub fn find_bytecode_loc(&self, offset: u32) -> Option<&SourceLoc> {
        // Binary search for the largest offset <= target
        match self.bytecode_map.binary_search_by_key(&offset, |(o, _)| *o) {
            Ok(idx) => Some(&self.bytecode_map[idx].1),
            Err(idx) if idx > 0 => Some(&self.bytecode_map[idx - 1].1),
            _ => None,
        }
    }

    /// Get all instruction locations.
    pub fn inst_locations(&self) -> &HashMap<usize, SourceLoc> {
        &self.inst_locations
    }

    /// Get all block locations.
    pub fn block_locations(&self) -> &HashMap<String, SourceLoc> {
        &self.block_locations
    }

    /// Get the bytecode map.
    pub fn bytecode_map(&self) -> &[(u32, SourceLoc)] {
        &self.bytecode_map
    }

    /// Merge debug info from another instance.
    pub fn merge(&mut self, other: &DebugInfo, inst_offset: usize, bytecode_offset: u32) {
        for (idx, loc) in &other.inst_locations {
            self.inst_locations.insert(idx + inst_offset, loc.clone());
        }

        for (label, loc) in &other.block_locations {
            self.block_locations.insert(label.clone(), loc.clone());
        }

        for (offset, loc) in &other.bytecode_map {
            self.bytecode_map.push((offset + bytecode_offset, loc.clone()));
        }
    }

    /// Number of recorded locations.
    pub fn num_locations(&self) -> usize {
        self.inst_locations.len() + self.block_locations.len()
    }
}

impl Default for DebugInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Line table entry for bytecode debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineTableEntry {
    /// Bytecode offset
    pub offset: u32,
    /// Source file index
    pub file_idx: u16,
    /// Line number
    pub line: u32,
    /// Column number
    pub column: u16,
}

/// Compact line table for embedding in bytecode.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LineTable {
    /// File paths (indexed by file_idx)
    pub files: Vec<String>,
    /// Line table entries
    pub entries: Vec<LineTableEntry>,
}

impl LineTable {
    /// Create a new line table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from debug info.
    pub fn from_debug_info(debug: &DebugInfo) -> Self {
        let mut table = Self::new();
        let mut file_indices: HashMap<&str, u16> = HashMap::new();

        for (offset, loc) in debug.bytecode_map() {
            let file_idx = if loc.file.is_empty() {
                0
            } else {
                *file_indices.entry(&loc.file).or_insert_with(|| {
                    let idx = table.files.len() as u16;
                    table.files.push(loc.file.clone());
                    idx
                })
            };

            table.entries.push(LineTableEntry {
                offset: *offset,
                file_idx,
                line: loc.line,
                column: loc.column as u16,
            });
        }

        table
    }

    /// Look up source location for a bytecode offset.
    pub fn find_location(&self, offset: u32) -> Option<SourceLoc> {
        // Binary search for the largest offset <= target
        let idx = match self.entries.binary_search_by_key(&offset, |e| e.offset) {
            Ok(idx) => idx,
            Err(idx) if idx > 0 => idx - 1,
            _ => return None,
        };

        let entry = &self.entries[idx];
        let file = if (entry.file_idx as usize) < self.files.len() {
            self.files[entry.file_idx as usize].clone()
        } else {
            String::new()
        };

        Some(SourceLoc {
            file,
            line: entry.line,
            column: entry.column as u32,
        })
    }

    /// Encode line table to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Number of files
        bytes.extend((self.files.len() as u16).to_le_bytes());

        // File paths
        for file in &self.files {
            let file_bytes = file.as_bytes();
            bytes.extend((file_bytes.len() as u16).to_le_bytes());
            bytes.extend(file_bytes);
        }

        // Number of entries
        bytes.extend((self.entries.len() as u32).to_le_bytes());

        // Entries (delta-encoded for compression)
        let mut prev_offset = 0u32;
        let mut prev_line = 0u32;

        for entry in &self.entries {
            let delta_offset = entry.offset.wrapping_sub(prev_offset);
            let delta_line = entry.line.wrapping_sub(prev_line) as i32;

            // Variable-length encoding for deltas
            bytes.extend(Self::encode_varint(delta_offset as i64));
            bytes.extend(Self::encode_varint(delta_line as i64));
            bytes.extend(entry.file_idx.to_le_bytes());
            bytes.extend(entry.column.to_le_bytes());

            prev_offset = entry.offset;
            prev_line = entry.line;
        }

        bytes
    }

    /// Decode line table from bytes.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut pos = 0;

        if bytes.len() < 2 {
            return None;
        }

        // Number of files
        let num_files = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        pos += 2;

        // File paths
        let mut files = Vec::with_capacity(num_files);
        for _ in 0..num_files {
            if pos + 2 > bytes.len() {
                return None;
            }
            let len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;

            if pos + len > bytes.len() {
                return None;
            }
            let file = String::from_utf8_lossy(&bytes[pos..pos + len]).to_string();
            pos += len;
            files.push(file);
        }

        if pos + 4 > bytes.len() {
            return None;
        }

        // Number of entries
        let num_entries = u32::from_le_bytes([
            bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]
        ]) as usize;
        pos += 4;

        // Entries
        let mut entries = Vec::with_capacity(num_entries);
        let mut offset = 0u32;
        let mut line = 0u32;

        for _ in 0..num_entries {
            let (delta_offset, consumed) = Self::decode_varint(&bytes[pos..])?;
            pos += consumed;
            offset = offset.wrapping_add(delta_offset as u32);

            let (delta_line, consumed) = Self::decode_varint(&bytes[pos..])?;
            pos += consumed;
            line = line.wrapping_add(delta_line as u32);

            if pos + 4 > bytes.len() {
                return None;
            }
            let file_idx = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]);
            pos += 2;
            let column = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]);
            pos += 2;

            entries.push(LineTableEntry {
                offset,
                file_idx,
                line,
                column,
            });
        }

        Some(Self { files, entries })
    }

    /// Encode a signed variable-length integer (signed LEB128-like).
    fn encode_varint(value: i64) -> Vec<u8> {
        let mut bytes = Vec::new();
        let negative = value < 0;
        let mut val = if negative { -value } else { value } as u64;

        // First byte: sign bit in bit 0, 6 bits of value in bits 1-6, continuation in bit 7
        let sign_bit = if negative { 1u8 } else { 0u8 };
        let mut byte = ((val & 0x3F) << 1) as u8 | sign_bit;
        val >>= 6;
        if val > 0 {
            byte |= 0x80; // More bytes follow
        }
        bytes.push(byte);

        // Subsequent bytes: 7 bits of value, continuation in bit 7
        while val > 0 {
            let mut b = (val & 0x7F) as u8;
            val >>= 7;
            if val > 0 {
                b |= 0x80;
            }
            bytes.push(b);
        }

        bytes
    }

    /// Decode a signed variable-length integer.
    fn decode_varint(bytes: &[u8]) -> Option<(i64, usize)> {
        if bytes.is_empty() {
            return None;
        }

        let first = bytes[0];
        let negative = (first & 1) != 0;
        let mut value = ((first & 0x7E) >> 1) as u64;
        let mut pos = 1;
        let mut shift = 6;

        if (first & 0x80) != 0 {
            // More bytes follow
            while pos < bytes.len() {
                let byte = bytes[pos];
                value |= ((byte & 0x7F) as u64) << shift;
                shift += 7;
                pos += 1;
                if (byte & 0x80) == 0 {
                    break;
                }
            }
        }

        let result = if negative { -(value as i64) } else { value as i64 };
        Some((result, pos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_loc() {
        let loc = SourceLoc::new("test.c", 10, 5);
        assert_eq!(loc.file, "test.c");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);
        assert!(!loc.is_unknown());
        assert_eq!(format!("{}", loc), "test.c:10:5");

        let unknown = SourceLoc::unknown();
        assert!(unknown.is_unknown());
        assert_eq!(format!("{}", unknown), "<unknown>");
    }

    #[test]
    fn test_debug_info() {
        let mut debug = DebugInfo::new();
        debug.set_file("main.c");
        debug.set_line(42);

        let loc = debug.current_loc();
        assert_eq!(loc.file, "main.c");
        assert_eq!(loc.line, 42);

        debug.record_inst(0, loc.clone());
        debug.record_block("entry", loc.clone());
        debug.record_bytecode(0, loc.clone());

        assert_eq!(debug.get_inst_loc(0), Some(&loc));
        assert_eq!(debug.get_block_loc("entry"), Some(&loc));
        assert_eq!(debug.find_bytecode_loc(0), Some(&loc));
    }

    #[test]
    fn test_bytecode_lookup() {
        let mut debug = DebugInfo::new();

        debug.record_bytecode(0, SourceLoc::new("test.c", 1, 0));
        debug.record_bytecode(10, SourceLoc::new("test.c", 5, 0));
        debug.record_bytecode(20, SourceLoc::new("test.c", 10, 0));

        // Exact matches
        assert_eq!(debug.find_bytecode_loc(0).unwrap().line, 1);
        assert_eq!(debug.find_bytecode_loc(10).unwrap().line, 5);
        assert_eq!(debug.find_bytecode_loc(20).unwrap().line, 10);

        // Between entries - should find previous
        assert_eq!(debug.find_bytecode_loc(5).unwrap().line, 1);
        assert_eq!(debug.find_bytecode_loc(15).unwrap().line, 5);
        assert_eq!(debug.find_bytecode_loc(25).unwrap().line, 10);
    }

    #[test]
    fn test_line_table_encoding() {
        let mut debug = DebugInfo::new();
        debug.record_bytecode(0, SourceLoc::new("test.c", 1, 0));
        debug.record_bytecode(4, SourceLoc::new("test.c", 2, 5));
        debug.record_bytecode(8, SourceLoc::new("other.c", 10, 0));

        let table = LineTable::from_debug_info(&debug);
        assert_eq!(table.files.len(), 2);
        assert_eq!(table.entries.len(), 3);

        // Encode and decode
        let bytes = table.encode();
        let decoded = LineTable::decode(&bytes).unwrap();

        assert_eq!(decoded.files.len(), 2);
        assert_eq!(decoded.entries.len(), 3);

        // Verify lookup
        let loc = decoded.find_location(4).unwrap();
        assert_eq!(loc.file, "test.c");
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 5);
    }

    #[test]
    fn test_varint_encoding() {
        // Test various values
        for value in [0i64, 1, -1, 127, -127, 128, -128, 1000, -1000, 100000] {
            let encoded = LineTable::encode_varint(value);
            let (decoded, _) = LineTable::decode_varint(&encoded).unwrap();
            assert_eq!(decoded, value, "Failed for value {}", value);
        }
    }
}
