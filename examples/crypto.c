// Cryptographic operations example for ZKIR LLVM backend
// Demonstrates patterns commonly used in ZK proofs
//
// Compile with: clang -O2 -emit-llvm -c crypto.c -o crypto.bc
// Then run: zkir-llvm crypto.bc -o crypto.zkir

#include <stdint.h>

// Simple hash computation (not cryptographically secure - just for demo)
// In real ZK applications, use dedicated hash circuits (Poseidon, MiMC, etc.)
uint32_t simple_hash(const uint8_t* data, uint32_t len) {
    uint32_t hash = 0x811c9dc5;  // FNV offset basis
    for (uint32_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 0x01000193;  // FNV prime
    }
    return hash;
}

// Merkle tree verification step
// Combines two child hashes to compute parent
uint32_t merkle_combine(uint32_t left, uint32_t right) {
    uint8_t buf[8];
    buf[0] = left & 0xFF;
    buf[1] = (left >> 8) & 0xFF;
    buf[2] = (left >> 16) & 0xFF;
    buf[3] = (left >> 24) & 0xFF;
    buf[4] = right & 0xFF;
    buf[5] = (right >> 8) & 0xFF;
    buf[6] = (right >> 16) & 0xFF;
    buf[7] = (right >> 24) & 0xFF;
    return simple_hash(buf, 8);
}

// Verify a Merkle proof
// Returns 1 if valid, 0 otherwise
int verify_merkle_proof(
    uint32_t leaf,
    uint32_t* path,
    uint8_t* path_bits,  // 0 = left, 1 = right
    uint32_t depth,
    uint32_t root
) {
    uint32_t current = leaf;
    for (uint32_t i = 0; i < depth; i++) {
        if (path_bits[i]) {
            current = merkle_combine(path[i], current);
        } else {
            current = merkle_combine(current, path[i]);
        }
    }
    return current == root;
}

// Range check: verify value is in [min, max]
// This is a key ZK primitive
int range_check(uint32_t value, uint32_t min, uint32_t max) {
    return value >= min && value <= max;
}

// Bit decomposition check
// Verifies that bits[] correctly represent value
int verify_bits(uint32_t value, const uint8_t* bits, uint32_t num_bits) {
    uint32_t reconstructed = 0;
    for (uint32_t i = 0; i < num_bits; i++) {
        // Each bit must be 0 or 1
        if (bits[i] > 1) {
            return 0;
        }
        reconstructed |= ((uint32_t)bits[i]) << i;
    }
    return reconstructed == value;
}

// Modular exponentiation (square-and-multiply)
// Common in signature verification
uint32_t mod_exp(uint32_t base, uint32_t exp, uint32_t mod) {
    if (mod == 0) return 0;
    if (mod == 1) return 0;

    uint64_t result = 1;
    uint64_t b = base % mod;

    while (exp > 0) {
        if (exp & 1) {
            result = (result * b) % mod;
        }
        exp >>= 1;
        b = (b * b) % mod;
    }

    return (uint32_t)result;
}

// Comparison without branches (constant-time)
// Returns 1 if a == b, 0 otherwise
int constant_time_eq(uint32_t a, uint32_t b) {
    uint32_t diff = a ^ b;
    // If diff is 0, result is 1; otherwise 0
    return (int)(1 - ((diff | (~diff + 1)) >> 31));
}

// Conditional select (constant-time)
// Returns a if cond == 0, b if cond != 0
uint32_t cond_select(uint32_t a, uint32_t b, int cond) {
    uint32_t mask = (uint32_t)(-(int32_t)(cond != 0));
    return (a & ~mask) | (b & mask);
}

// Field element multiplication for ZK (simplified, non-finite field)
// In real ZK: use proper field arithmetic (e.g., BN254 scalar field)
typedef struct {
    uint64_t lo;
    uint64_t hi;
} uint128_t;

uint128_t mul_wide(uint64_t a, uint64_t b) {
    uint128_t result;
    // Split into 32-bit parts for overflow handling
    uint64_t a_lo = a & 0xFFFFFFFF;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFF;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    uint64_t mid = p1 + p2;
    uint64_t carry = (mid < p1) ? 0x100000000ULL : 0;

    result.lo = p0 + (mid << 32);
    result.hi = p3 + (mid >> 32) + carry + (result.lo < p0 ? 1 : 0);

    return result;
}

// Test function
int main() {
    // Test hash
    uint8_t data[] = {1, 2, 3, 4};
    uint32_t h = simple_hash(data, 4);

    // Test range check
    int in_range = range_check(50, 0, 100);

    // Test bit decomposition
    uint8_t bits[8] = {1, 0, 1, 0, 0, 0, 0, 0};  // 5 = 0b101
    int bits_ok = verify_bits(5, bits, 8);

    // Test modular exponentiation
    uint32_t exp_result = mod_exp(2, 10, 1000);  // 2^10 mod 1000 = 24

    // Test constant-time operations
    int eq = constant_time_eq(42, 42);
    uint32_t sel = cond_select(10, 20, 0);  // Should return 10

    // Return combined result
    return (int)(h + in_range + bits_ok + exp_result + eq + sel);
}
