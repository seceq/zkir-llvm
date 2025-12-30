// Privacy-preserving computation patterns for ZKIR
// Demonstrates common ZK circuit patterns for private data handling
//
// Compile with: clang -O2 -emit-llvm -c privacy.c -o privacy.bc
// Then run: zkir-llvm privacy.bc -o privacy.zkir

#include <stdint.h>

// ============================================================
// Private input handling
// In ZK, private inputs are "witness" values that prove a statement
// without revealing the actual values.
// ============================================================

// Verify age >= 18 without revealing exact age
// Returns 1 if age >= 18, 0 otherwise
int verify_adult(uint8_t age) {
    // The verifier only learns that age >= 18, not the actual age
    return age >= 18;
}

// Verify balance is sufficient for transfer
// Returns 1 if balance >= amount, 0 otherwise
int verify_sufficient_balance(uint64_t balance, uint64_t amount) {
    // Prove we have enough without revealing exact balance
    return balance >= amount;
}

// ============================================================
// Commitment schemes
// Commitments hide a value while binding to it
// ============================================================

// Simple Pedersen-like commitment (not cryptographically secure)
// commit(v, r) = v * G + r * H (simplified as linear combination)
uint64_t compute_commitment(uint32_t value, uint32_t randomness, uint64_t g, uint64_t h) {
    return (uint64_t)value * g + (uint64_t)randomness * h;
}

// Verify that a commitment opens to a specific value
int verify_commitment_opening(
    uint64_t commitment,
    uint32_t value,
    uint32_t randomness,
    uint64_t g,
    uint64_t h
) {
    uint64_t expected = compute_commitment(value, randomness, g, h);
    return commitment == expected;
}

// ============================================================
// Nullifier patterns (prevent double-spending)
// ============================================================

// Compute nullifier from secret and commitment
// In real ZK: nullifier = hash(secret || commitment)
uint32_t compute_nullifier(uint32_t secret, uint32_t commitment) {
    // Simple XOR-based mixing (replace with proper hash in production)
    uint32_t result = secret ^ commitment;
    result ^= (result >> 16);
    result *= 0x7feb352d;
    result ^= (result >> 15);
    result *= 0x846ca68b;
    result ^= (result >> 16);
    return result;
}

// Check if nullifier hasn't been used (would query external set in practice)
int check_nullifier_unused(uint32_t nullifier, const uint32_t* used_nullifiers, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        if (used_nullifiers[i] == nullifier) {
            return 0;  // Already used
        }
    }
    return 1;  // Not used yet
}

// ============================================================
// Set membership proofs
// Prove element is in a set without revealing which one
// ============================================================

// Naive O(n) membership check - in practice use accumulators
int prove_membership(uint32_t element, const uint32_t* set, uint32_t set_size) {
    for (uint32_t i = 0; i < set_size; i++) {
        if (set[i] == element) {
            return 1;
        }
    }
    return 0;
}

// Prove element is NOT in a set (non-membership)
int prove_non_membership(uint32_t element, const uint32_t* set, uint32_t set_size) {
    return !prove_membership(element, set, set_size);
}

// ============================================================
// Arithmetic circuits for comparisons
// These compile to efficient constraint systems
// ============================================================

// Less-than comparison using bit decomposition approach
// Returns 1 if a < b
int less_than_bits(uint32_t a, uint32_t b, uint8_t num_bits) {
    uint32_t borrow = 0;

    for (uint8_t i = 0; i < num_bits; i++) {
        uint8_t bit_a = (a >> i) & 1;
        uint8_t bit_b = (b >> i) & 1;

        // Calculate difference with borrow
        int32_t diff = (int32_t)bit_a - (int32_t)bit_b - (int32_t)borrow;

        if (diff < 0) {
            borrow = 1;
        } else {
            borrow = 0;
        }
    }

    return borrow != 0;
}

// Equal-to comparison returning 0 or 1
int equals(uint32_t a, uint32_t b) {
    // XOR gives 0 if equal
    uint32_t diff = a ^ b;
    // Use bit tricks to check if diff is 0
    return (diff | (~diff + 1)) >> 31 ^ 1;
}

// ============================================================
// Private voting example
// ============================================================

typedef struct {
    uint32_t voter_id_hash;   // Hash of voter identity
    uint32_t nullifier;       // Prevents double voting
    uint8_t choice;          // The vote (0 = no, 1 = yes)
    uint32_t signature;       // Simplified signature
} Vote;

// Verify a vote is valid without revealing voter identity
int verify_vote(
    const Vote* vote,
    const uint32_t* registered_voters,  // List of valid voter hashes
    uint32_t num_voters,
    const uint32_t* used_nullifiers,
    uint32_t num_used
) {
    // 1. Check voter is registered
    int is_registered = prove_membership(vote->voter_id_hash, registered_voters, num_voters);
    if (!is_registered) return 0;

    // 2. Check nullifier hasn't been used (no double voting)
    int not_used = check_nullifier_unused(vote->nullifier, used_nullifiers, num_used);
    if (!not_used) return 0;

    // 3. Check vote is valid (0 or 1)
    if (vote->choice > 1) return 0;

    // 4. In practice, would verify signature here

    return 1;
}

// Tally votes (private - result is public, individual votes are hidden)
typedef struct {
    uint32_t yes_count;
    uint32_t no_count;
} VoteResult;

VoteResult tally_votes(const Vote* votes, uint32_t num_votes) {
    VoteResult result = {0, 0};

    for (uint32_t i = 0; i < num_votes; i++) {
        if (votes[i].choice == 1) {
            result.yes_count++;
        } else {
            result.no_count++;
        }
    }

    return result;
}

// Test function
int main() {
    // Test age verification
    int is_adult = verify_adult(25);

    // Test balance verification
    int has_funds = verify_sufficient_balance(1000, 500);

    // Test commitment
    uint64_t commitment = compute_commitment(42, 12345, 7, 11);
    int opens = verify_commitment_opening(commitment, 42, 12345, 7, 11);

    // Test nullifier
    uint32_t nullifier = compute_nullifier(0xdeadbeef, 0xcafebabe);

    // Test comparisons
    int lt_result = less_than_bits(5, 10, 32);
    int eq_result = equals(42, 42);

    return is_adult + has_funds + opens + (int)(nullifier & 0xF) + lt_result + eq_result;
}
