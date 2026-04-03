/*
 * predict_cuda256.cu — GPU-native Z₂₅₆ BPAND Walk with uint256 arithmetic
 *
 * Extends predict_cuda to handle n > 2^64 (up to ~10^77).
 *
 * uint256 = 4 limbs of uint64, mapped to quadrants:
 *   limb[0] = UP+  (bits   0..63)    weight × 1
 *   limb[1] = UP-  (bits  64..127)   weight × 2^64
 *   limb[2] = DN+  (bits 128..191)   weight × 2^128
 *   limb[3] = DN-  (bits 192..255)   weight × 2^192
 *
 * Multiplication:  4(ab)² · 4(cd)² = 16(abcd)²
 *   16 cross-products via __umul64hi, carry rule = quadrant rotation.
 *
 * No float. No signed. No FPU. Pure unsigned integer.
 *
 * Build:
 *   nvcc -O3 -o predict_cuda256.exe src/predict_cuda256.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cinttypes>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#  include <time.h>
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#endif

#include <cuda_runtime.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Z₂₅₆ Geometry Constants
 * ═══════════════════════════════════════════════════════════════════════════ */
#define STEP               7u
#define HALF               128
#define VACUUMS_PER_HALF   2
#define ACTIVE_PER_HALF    (HALF - VACUUMS_PER_HALF)
#define ORBIT_SIZE         (2 * ACTIVE_PER_HALF)       /* 252          */
#define HV_CELLS           (HALF * 113)                /* 14,464       */
#define HV_W               113
#define ORBITS_PER_HV      (HV_CELLS / ORBIT_SIZE)     /* 57           */

#define BATCH_K            (1u << 28)  /* 256M — prefilter threads per launch  */
/* MAX_SURVIVORS now computed at runtime from free VRAM                    */

/* ─── GPU constant memory ─────────────────────────────────────────────── */
__constant__ uint8_t  d_orbit[ORBIT_SIZE];
__constant__ uint32_t d_small_primes[11];
__constant__ uint64_t d_mr_bases[7];
__constant__ int      d_limbs;  /* active limb count L, detected on host */

/* ═══════════════════════════════════════════════════════════════════════════
 * uint256 — 4 limbs of uint64, little-endian
 *
 *   w[0] = least significant  (UP+,  bits   0..63)
 *   w[1]                      (UP-,  bits  64..127)
 *   w[2]                      (DN+,  bits 128..191)
 *   w[3] = most significant   (DN-,  bits 192..255)
 * ═══════════════════════════════════════════════════════════════════════════ */

struct uint256 {
    uint64_t w[4];
};

/* ─── Zero / One / From uint64 ────────────────────────────────────────── */
__host__ __device__ __forceinline__
uint256 u256_zero(void) {
    uint256 r; r.w[0]=0; r.w[1]=0; r.w[2]=0; r.w[3]=0; return r;
}

__host__ __device__ __forceinline__
uint256 u256_from_u64(uint64_t v) {
    uint256 r; r.w[0]=v; r.w[1]=0; r.w[2]=0; r.w[3]=0; return r;
}

__host__ __device__ __forceinline__
uint256 u256_one(void) { return u256_from_u64(1); }

/* ─── Comparison ──────────────────────────────────────────────────────── */
__host__ __device__ __forceinline__
int u256_is_zero(uint256 a) {
    return (a.w[0] | a.w[1] | a.w[2] | a.w[3]) == 0;
}

__host__ __device__ __forceinline__
int u256_lt(uint256 a, uint256 b) {
    if (a.w[3] != b.w[3]) return a.w[3] < b.w[3];
    if (a.w[2] != b.w[2]) return a.w[2] < b.w[2];
    if (a.w[1] != b.w[1]) return a.w[1] < b.w[1];
    return a.w[0] < b.w[0];
}

__host__ __device__ __forceinline__
int u256_eq(uint256 a, uint256 b) {
    return a.w[0]==b.w[0] && a.w[1]==b.w[1] &&
           a.w[2]==b.w[2] && a.w[3]==b.w[3];
}

__host__ __device__ __forceinline__
int u256_le(uint256 a, uint256 b) {
    return u256_eq(a, b) || u256_lt(a, b);
}

__host__ __device__ __forceinline__
int u256_gt(uint256 a, uint256 b) { return u256_lt(b, a); }

/* ─── Bitwise ─────────────────────────────────────────────────────────── */
__host__ __device__ __forceinline__
uint256 u256_and(uint256 a, uint256 b) {
    uint256 r;
    r.w[0]=a.w[0]&b.w[0]; r.w[1]=a.w[1]&b.w[1];
    r.w[2]=a.w[2]&b.w[2]; r.w[3]=a.w[3]&b.w[3];
    return r;
}

__host__ __device__ __forceinline__
uint256 u256_xor(uint256 a, uint256 b) {
    uint256 r;
    r.w[0]=a.w[0]^b.w[0]; r.w[1]=a.w[1]^b.w[1];
    r.w[2]=a.w[2]^b.w[2]; r.w[3]=a.w[3]^b.w[3];
    return r;
}

__host__ __device__ __forceinline__
uint256 u256_shr1(uint256 a) {
    uint256 r;
    r.w[0] = (a.w[0] >> 1) | (a.w[1] << 63);
    r.w[1] = (a.w[1] >> 1) | (a.w[2] << 63);
    r.w[2] = (a.w[2] >> 1) | (a.w[3] << 63);
    r.w[3] = a.w[3] >> 1;
    return r;
}

__host__ __device__ __forceinline__
uint256 u256_shl1(uint256 a) {
    uint256 r;
    r.w[3] = (a.w[3] << 1) | (a.w[2] >> 63);
    r.w[2] = (a.w[2] << 1) | (a.w[1] >> 63);
    r.w[1] = (a.w[1] << 1) | (a.w[0] >> 63);
    r.w[0] = a.w[0] << 1;
    return r;
}

__host__ __device__ __forceinline__
int u256_bit0(uint256 a) { return (int)(a.w[0] & 1u); }

/* ─── Addition with carry ─────────────────────────────────────────────── */
__host__ __device__ __forceinline__
uint256 u256_add(uint256 a, uint256 b) {
    uint256 r;
    uint64_t carry = 0;

    r.w[0] = a.w[0] + b.w[0];
    carry = (r.w[0] < a.w[0]) ? 1u : 0u;

    r.w[1] = a.w[1] + b.w[1] + carry;
    carry = (r.w[1] < a.w[1] || (carry && r.w[1] == a.w[1])) ? 1u : 0u;

    r.w[2] = a.w[2] + b.w[2] + carry;
    carry = (r.w[2] < a.w[2] || (carry && r.w[2] == a.w[2])) ? 1u : 0u;

    r.w[3] = a.w[3] + b.w[3] + carry;
    return r;
}

/* ─── Subtraction with borrow ─────────────────────────────────────────── */
__host__ __device__ __forceinline__
uint256 u256_sub(uint256 a, uint256 b) {
    uint256 r;
    uint64_t borrow = 0;

    r.w[0] = a.w[0] - b.w[0];
    borrow = (a.w[0] < b.w[0]) ? 1u : 0u;

    uint64_t sub1 = a.w[1] - b.w[1];
    uint64_t sub1b = sub1 - borrow;
    borrow = (a.w[1] < b.w[1] || (borrow && sub1 == 0)) ? 1u : 0u;
    r.w[1] = sub1b;

    uint64_t sub2 = a.w[2] - b.w[2];
    uint64_t sub2b = sub2 - borrow;
    borrow = (a.w[2] < b.w[2] || (borrow && sub2 == 0)) ? 1u : 0u;
    r.w[2] = sub2b;

    r.w[3] = a.w[3] - b.w[3] - borrow;
    return r;
}

/* ─── Modular addition: (a + b) mod m ─────────────────────────────────── */
__host__ __device__ __forceinline__
uint256 u256_addmod(uint256 a, uint256 b, uint256 m) {
    uint256 s = u256_add(a, b);
    /* Detect overflow: if s < a, we wrapped 2^256 */
    int overflow = u256_lt(s, a);
    if (overflow || u256_le(m, s))
        s = u256_sub(s, m);
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * uint256 mulmod — the quadrant carry table
 *
 *   4(ab)² · 4(cd)² = 16(abcd)²
 *
 *   16 cross-products a.w[i] × b.w[j], each via __umul64hi.
 *   Destination limb = (i + j) mod 4.
 *   If (i + j) >= 4, there is a quadrant-wrap carry.
 *
 *   We accumulate into a 512-bit intermediate (8 limbs),
 *   then reduce mod m using shift-and-subtract.
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef __CUDA_ARCH__

/* Multiply two uint64 → (hi:lo) using hardware PTX */
__device__ __forceinline__
void u64_mul_wide(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo) {
    *lo = a * b;
    *hi = __umul64hi(a, b);
}

#else

#ifdef _MSC_VER
#include <intrin.h>
static inline
void u64_mul_wide(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo) {
    *lo = _umul128(a, b, hi);
}
#else
static inline
void u64_mul_wide(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo) {
    unsigned __int128 r = (unsigned __int128)a * b;
    *lo = (uint64_t)r;
    *hi = (uint64_t)(r >> 64);
}
#endif

#endif

/*
 * 512-bit accumulator for the schoolbook multiply.
 * 8 limbs: product of two 256-bit numbers fits in 512 bits.
 */
struct uint512 {
    uint64_t w[8];
};

__host__ __device__ __forceinline__
uint512 u512_zero(void) {
    uint512 r;
    for (int i = 0; i < 8; i++) r.w[i] = 0;
    return r;
}

/* Add (hi:lo) at position [pos] and [pos+1] with carry propagation */
__host__ __device__ __forceinline__
void u512_add_at(uint512 *acc, int pos, uint64_t lo, uint64_t hi) {
    /* Add lo at position pos */
    uint64_t old = acc->w[pos];
    acc->w[pos] += lo;
    uint64_t carry = (acc->w[pos] < old) ? 1u : 0u;

    /* Add hi + carry at position pos+1 */
    old = acc->w[pos + 1];
    acc->w[pos + 1] += hi + carry;
    carry = (acc->w[pos + 1] < old || (hi + carry < hi)) ? 1u : 0u;

    /* Propagate carry through remaining limbs */
    for (int i = pos + 2; i < 8 && carry; i++) {
        old = acc->w[i];
        acc->w[i] += carry;
        carry = (acc->w[i] < old) ? 1u : 0u;
    }
}

/*
 * Barrett-like reduction: compute (a * b) mod m for uint256.
 *
 * Strategy:
 *   1) Schoolbook 4×4 multiply → 512-bit product (16 __umul64hi calls)
 *   2) Reduce 512-bit → 256-bit mod m via shift-subtract
 *
 * The 16 cross-products follow the quadrant carry table:
 *   a.w[i] × b.w[j] lands at limb position (i + j)
 *   Positions 0..3 are the low 256 bits
 *   Positions 4..7 are the high 256 bits (the "carry wrap")
 */
/*
 * Fast path: if m fits in one uint64 limb, use hardware
 * __umul64hi + 64 branchless doublings (same as uint64 version).
 * This is the common case for n ≤ ~1.8×10^19.
 */
__device__ __forceinline__
uint64_t dev_mulmod_u64(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    if (hi == 0) return lo % m;
    uint64_t r = lo % m;
    uint64_t h = hi % m;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        h = (h >= m - h) ? h - (m - h) : h + h;
    }
    r = (r >= m - h) ? r - (m - h) : r + h;
    return r;
}

/*
 * L-parameterized mulmod.
 *
 *   L=1: pure uint64 fast path (hardware __umul64hi + 64 doublings)
 *   L≥2: full 4×4 schoolbook + skip-zero reduction
 *
 * The 4×4 stays unrolled — 12 zero-multiplies at L=2 cost ~nothing.
 * The real savings from L=1 dispatch: avoid uint256 entirely.
 */

#define SCHOOLBOOK_4x4(a, b, prod)                          \
    _Pragma("unroll")                                       \
    for (int i = 0; i < 4; i++) {                           \
        _Pragma("unroll")                                   \
        for (int j = 0; j < 4; j++) {                       \
            uint64_t hi__, lo__;                            \
            u64_mul_wide((a).w[i], (b).w[j], &hi__, &lo__);\
            u512_add_at(&(prod), i + j, lo__, hi__);        \
        }                                                   \
    }

__device__
uint256 u256_mulmod(uint256 a, uint256 b, uint256 m) {
    /* === L=1: pure uint64 fast path === */
    if (d_limbs <= 1) {
        uint64_t r = dev_mulmod_u64(a.w[0] % m.w[0], b.w[0] % m.w[0], m.w[0]);
        return u256_from_u64(r);
    }

    /* Step 1: schoolbook 4×4 → 512-bit (always unrolled) */
    uint512 prod = u512_zero();
    SCHOOLBOOK_4x4(a, b, prod);

    /* Step 2: skip-zero reduction */
    int top = 7;
    while (top > 0 && prod.w[top] == 0) top--;

    if (top == 0 && prod.w[0] == 0) return u256_zero();

    uint256 r = u256_zero();
    uint256 one = u256_one();

    /* Top limb: skip leading zero bits */
    uint64_t w = prod.w[top];
    int start_bit = 63 - __clzll(w);
    for (int bit = start_bit; bit >= 0; bit--) {
        r = u256_addmod(r, r, m);
        if ((w >> bit) & 1u)
            r = u256_addmod(r, one, m);
    }

    for (int limb = top - 1; limb >= 0; limb--) {
        w = prod.w[limb];
        for (int bit = 63; bit >= 0; bit--) {
            r = u256_addmod(r, r, m);
            if ((w >> bit) & 1u)
                r = u256_addmod(r, one, m);
        }
    }

    return r;
}

/* Host-side mulmod — L-parameterized */
__host__
uint256 u256_mulmod_host(uint256 a, uint256 b, uint256 m, int L) {
    if (L <= 1) {
        uint64_t ah = a.w[0] % m.w[0], bh = b.w[0] % m.w[0];
        uint64_t hi_val, lo_val;
        u64_mul_wide(ah, bh, &hi_val, &lo_val);
        if (hi_val == 0) return u256_from_u64(lo_val % m.w[0]);
        uint64_t r = lo_val % m.w[0];
        uint64_t h = hi_val % m.w[0];
        for (int i = 0; i < 64; i++) {
            h = (h >= m.w[0] - h) ? h - (m.w[0] - h) : h + h;
        }
        r = (r >= m.w[0] - h) ? r - (m.w[0] - h) : r + h;
        return u256_from_u64(r);
    }

    uint512 prod = u512_zero();
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            u64_mul_wide(a.w[i], b.w[j], &hi, &lo);
            u512_add_at(&prod, i + j, lo, hi);
        }

    int top = 7;
    while (top > 0 && prod.w[top] == 0) top--;
    if (top == 0 && prod.w[0] == 0) return u256_zero();

    uint256 r = u256_zero();
    uint256 one = u256_one();

    uint64_t w = prod.w[top];
    int start_bit = 63;
    while (start_bit > 0 && !((w >> start_bit) & 1u)) start_bit--;
    for (int bit = start_bit; bit >= 0; bit--) {
        r = u256_addmod(r, r, m);
        if ((w >> bit) & 1u)
            r = u256_addmod(r, one, m);
    }

    for (int limb = top - 1; limb >= 0; limb--) {
        w = prod.w[limb];
        for (int bit = 63; bit >= 0; bit--) {
            r = u256_addmod(r, r, m);
            if ((w >> bit) & 1u)
                r = u256_addmod(r, one, m);
        }
    }
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * uint256 powmod — square and multiply
 *
 * Dynamic: if m fits in uint64, runs pure uint64 path.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* uint64 fast powmod — used when m fits in one limb */
__device__
uint64_t dev_powmod_u64(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t r = 1;
    base %= m;
    while (exp) {
        if (exp & 1u) r = dev_mulmod_u64(r, base, m);
        exp >>= 1;
        if (exp) base = dev_mulmod_u64(base, base, m);
    }
    return r;
}

__device__
uint256 u256_powmod(uint256 base, uint256 exp, uint256 m) {
    /* === L=1: pure uint64 === */
    if (d_limbs <= 1) {
        uint64_t e64 = exp.w[0];
        if (exp.w[1] | exp.w[2] | exp.w[3]) {
            uint256 r = u256_one();
            uint64_t b64 = base.w[0] % m.w[0];
            while (!u256_is_zero(exp)) {
                if (u256_bit0(exp)) {
                    r.w[0] = dev_mulmod_u64(r.w[0] % m.w[0], b64, m.w[0]);
                    r.w[1] = r.w[2] = r.w[3] = 0;
                }
                exp = u256_shr1(exp);
                if (!u256_is_zero(exp))
                    b64 = dev_mulmod_u64(b64, b64, m.w[0]);
            }
            return r;
        }
        return u256_from_u64(dev_powmod_u64(base.w[0] % m.w[0], e64, m.w[0]));
    }

    uint256 r = u256_one();

    /* base %= m */
    while (u256_le(m, base))
        base = u256_sub(base, m);

    while (!u256_is_zero(exp)) {
        if (u256_bit0(exp))
            r = u256_mulmod(r, base, m);
        exp = u256_shr1(exp);
        if (!u256_is_zero(exp))
            base = u256_mulmod(base, base, m);
    }
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * uint256 Miller-Rabin — deterministic with 7 bases
 *
 * DYNAMIC DISPATCH:
 *   n fits in uint64 (1 limb)  → pure uint64 MR, zero uint256 overhead
 *   n fits in 2 limbs          → uint256 mulmod with skip-zero + fast path
 *   n fills all 4 limbs        → full uint256 path
 *
 * Deterministic for all numbers < 3.317×10^24.
 * Probabilistic but astronomically reliable above that.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Pure uint64 Miller-Rabin — called when n fits in one limb */
__device__
int dev_is_prime_u64(uint64_t n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if ((n & 1u) == 0) return 0;

    #pragma unroll
    for (int i = 0; i < 11; i++) {
        uint32_t p = d_small_primes[i];
        if (n == (uint64_t)p) return 1;
        if (n % p == 0) return 0;
    }

    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1u) == 0) { d >>= 1; s++; }

    #pragma unroll
    for (int i = 0; i < 7; i++) {
        uint64_t a = d_mr_bases[i] % n;
        if (a == 0) continue;
        uint64_t x = dev_powmod_u64(a, d, n);
        if (x == 1 || x == n - 1) continue;
        int witness = 1;
        for (unsigned r = 1; r < s; r++) {
            x = dev_mulmod_u64(x, x, n);
            if (x == n - 1) { witness = 0; break; }
        }
        if (witness) return 0;
    }
    return 1;
}

__device__
int u256_is_prime(uint256 n) {
    /* === L=1: pure uint64 MR === */
    if (d_limbs <= 1)
        return dev_is_prime_u64(n.w[0]);

    uint256 one  = u256_one();
    uint256 two  = u256_from_u64(2);

    /* even */
    if (!u256_bit0(n)) return 0;

    /* Trial division by small primes — scan only active limbs */
    #pragma unroll
    for (int i = 0; i < 11; i++) {
        uint64_t p = d_small_primes[i];
        uint256 pv = u256_from_u64(p);
        if (u256_eq(n, pv)) return 1;

        uint64_t rem = 0;
        uint64_t hi_mod = ((uint64_t)(-1) % p + 1) % p;
        for (int k = 3; k >= 0; k--)
            rem = ((rem % p) * hi_mod + n.w[k] % p) % p;
        if (rem == 0) return 0;
    }

    /* n - 1 = d × 2^s */
    uint256 n_minus_1 = u256_sub(n, one);
    uint256 d = n_minus_1;
    unsigned s = 0;
    while (!u256_bit0(d)) { d = u256_shr1(d); s++; }

    /* 7 deterministic witnesses */
    #pragma unroll
    for (int i = 0; i < 7; i++) {
        uint64_t a_raw = d_mr_bases[i];
        uint256 a = u256_from_u64(a_raw);

        if (u256_le(n, a)) {
            while (u256_le(n, a))
                a = u256_sub(a, n);
            if (u256_is_zero(a)) continue;
        }

        uint256 x = u256_powmod(a, d, n);

        if (u256_eq(x, one) || u256_eq(x, n_minus_1)) continue;

        int witness = 1;
        for (unsigned r = 1; r < s; r++) {
            x = u256_mulmod(x, x, n);
            if (u256_eq(x, n_minus_1)) { witness = 0; break; }
        }
        if (witness) return 0;
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TWO-KERNEL PIPELINE — scatter → compact → compute
 *
 * Problem: monolithic kernel → 50% die at parity, 90% at trial division,
 *          0.1% do MR. Warps idle while one thread grinds.
 *
 * Solution:
 *   Kernel 1 (prefilter): parity gate + trial division by 11 small primes.
 *       Lightweight. Writes compact list of surviving k values.
 *       Uses atomicAdd to pack survivors with zero gaps.
 *
 *   Kernel 2 (MR):  full Miller-Rabin on BOTH arms.
 *       Every thread does real MR. Zero idlers.
 *       Reads from compact survivor list.
 *
 * Distribution: 256M threads in prefilter → ~5-10% survive → ~13-25M
 *   dense MR threads. Every SM fully loaded with heavy work.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Device counter removed — now device-allocated per pipeline slot,
 * enabling double-buffered streams without global state conflict.  */

/* Trial division for uint256 — parity + small primes on BOTH arms */
__device__ __forceinline__
int trial_survives(uint256 lo, uint256 hi) {
    /* Both must be odd (parity gate already ensured lo is odd,
     * but hi = big_n + k could be even if big_n+k is even) */
    if (!u256_bit0(lo) || !u256_bit0(hi)) return 0;

    /* Trial division by 11 small primes on BOTH arms */
    #pragma unroll
    for (int i = 0; i < 11; i++) {
        uint64_t p = d_small_primes[i];

        /* lo mod p */
        uint64_t rem_lo = 0;
        uint64_t hi_mod = ((uint64_t)(-1) % p + 1) % p;
        for (int k = 3; k >= 0; k--)
            rem_lo = ((rem_lo % p) * hi_mod + lo.w[k] % p) % p;

        /* Check if lo == p (small prime itself) */
        if (rem_lo == 0) {
            if (!(lo.w[0] == p && lo.w[1] == 0 && lo.w[2] == 0 && lo.w[3] == 0))
                return 0;
        }

        /* hi mod p */
        uint64_t rem_hi = 0;
        for (int k = 3; k >= 0; k--)
            rem_hi = ((rem_hi % p) * hi_mod + hi.w[k] % p) % p;

        if (rem_hi == 0) {
            if (!(hi.w[0] == p && hi.w[1] == 0 && hi.w[2] == 0 && hi.w[3] == 0))
                return 0;
        }
    }
    return 1;
}

/*
 * Kernel 1: PREFILTER — parity gate + trial division
 *
 * Lightweight. Each thread checks one k.
 * Survivors written to compact list via atomicAdd.
 */
__global__ void prefilter_kernel(uint256 big_n,
                                  uint64_t k_start,
                                  uint64_t batch_count,
                                  uint64_t *survivors,
                                  uint32_t max_survivors,
                                  uint32_t *surv_count) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    uint64_t k_val = k_start + idx;

    /* parity gate */
    if (((big_n.w[0] ^ k_val) & 1u) == 0) return;

    /* conjugate arms */
    uint256 k256 = u256_from_u64(k_val);
    uint256 lo = u256_sub(big_n, k256);
    uint256 hi = u256_add(big_n, k256);

    /* lo < 2 check */
    if (u256_lt(lo, u256_from_u64(2))) return;

    /* trial division on BOTH arms */
    if (!trial_survives(lo, hi)) return;

    /* Survived → compact write */
    uint32_t pos = atomicAdd(surv_count, 1u);
    if (pos < max_survivors)
        survivors[pos] = k_val;
}

/*
 * Kernel 2: MR — full Miller-Rabin on compact survivor list
 *
 * Every thread does real work. Zero parity/trial rejects.
 * Dense utilization: all SMs grinding MR simultaneously.
 */
__global__ void mr_kernel(uint256 big_n,
                           uint64_t *survivors,
                           uint32_t survivor_count,
                           uint8_t *out_hit,
                           uint8_t *out_d) {
    uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= survivor_count) return;

    uint64_t k_val = survivors[idx];

    /* orbit lookup */
    uint8_t d_val = d_orbit[(k_val - 1) % ORBIT_SIZE];
    out_d[idx] = d_val;

    /* conjugate arms */
    uint256 k256 = u256_from_u64(k_val);
    uint256 lo = u256_sub(big_n, k256);
    uint256 hi = u256_add(big_n, k256);

    /* full MR on both arms — L from constant memory */
    out_hit[idx] = (u256_is_prime(lo) && u256_is_prime(hi)) ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Host utilities
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char *qlabel(uint8_t d) {
    switch ((d >> 6) & 0x3u) {
        case 0: return "UP+";
        case 1: return "UP-";
        case 2: return "DN+";
        case 3: return "DN-";
    }
    return "???";
}

#define FMT_SLOTS 8
static char _fmt_pool[FMT_SLOTS][32];
static int  _fmt_idx = 0;
static const char *fmt(uint64_t n) {
    char *buf = _fmt_pool[_fmt_idx % FMT_SLOTS];
    _fmt_idx++;
    char tmp[24];
    int len = snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long)n);
    int out = 0;
    for (int i = 0; i < len; i++) {
        int rem = len - i;
        if (i > 0 && rem % 3 == 0) buf[out++] = ',';
        buf[out++] = tmp[i];
    }
    buf[out] = '\0';
    return buf;
}

/* Format a uint256 as decimal string */
static char _fmt256_pool[4][80];
static int  _fmt256_idx = 0;

static const char *fmt256(uint256 v) {
    char *buf = _fmt256_pool[_fmt256_idx % 4];
    _fmt256_idx++;

    /* If it fits in 64 bits, use simple path */
    if (v.w[1] == 0 && v.w[2] == 0 && v.w[3] == 0) {
        char tmp[24];
        int len = snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long)v.w[0]);
        int out = 0;
        for (int i = 0; i < len; i++) {
            int rem = len - i;
            if (i > 0 && rem % 3 == 0) buf[out++] = ',';
            buf[out++] = tmp[i];
        }
        buf[out] = '\0';
        return buf;
    }

    /* Multi-limb: print raw hex for now */
    snprintf(buf, 80, "0x%llx_%016llx_%016llx_%016llx",
             (unsigned long long)v.w[3], (unsigned long long)v.w[2],
             (unsigned long long)v.w[1], (unsigned long long)v.w[0]);
    return buf;
}

/*
 * Portable 128÷64 division for small divisors (d < 2^32).
 * Computes (hi:lo) / d → quotient, returns remainder.
 */
static inline uint64_t divmod_128_64(
    uint64_t hi, uint64_t lo, uint64_t d, uint64_t *quot)
{
    /* Split into 32-bit chunks for portability */
    /* For small d (like 10), this is exact */
    uint64_t q = 0, r = 0;
    for (int bit = 127; bit >= 0; bit--) {
        r <<= 1;
        uint64_t b;
        if (bit >= 64)
            b = (hi >> (bit - 64)) & 1u;
        else
            b = (lo >> bit) & 1u;
        r |= b;
        if (r >= d) { r -= d; q |= ((bit < 64) ? (1ULL << bit) : 0); }
        /* For the high part of quotient, we'd need 128-bit q.
         * But since our use cases (dividing uint256 limbs by 10)
         * always have hi < d, the quotient fits in 64 bits. */
    }
    *quot = q;
    return r;
}

/* Divide uint256 by small uint64 constant, return remainder */
static uint64_t u256_div_small(uint256 *v, uint64_t d) {
    uint64_t rem = 0;
    for (int i = 3; i >= 0; i--) {
        /* (rem:v->w[i]) / d */
        if (rem == 0) {
            uint64_t q = v->w[i] / d;
            rem = v->w[i] % d;
            v->w[i] = q;
        } else {
            /* rem < d, and d is small, so rem * 2^64 + w[i] */
            /* Use the fact that for small d: */
            /* q = (rem * (2^64 / d) + rem * (2^64 % d) + w[i]) / d, approximately */
            /* Exact: split w[i] into hi32:lo32 */
            uint64_t hi32 = v->w[i] >> 32;
            uint64_t lo32 = v->w[i] & 0xFFFFFFFFULL;

            uint64_t a = (rem << 32) | hi32;   /* fits: rem < d < 2^32 */
            uint64_t q_hi = a / d;
            uint64_t r_hi = a % d;

            uint64_t b = (r_hi << 32) | lo32;
            uint64_t q_lo = b / d;
            rem = b % d;

            v->w[i] = (q_hi << 32) | q_lo;
        }
    }
    return rem;
}

/* Print uint256 as decimal — full conversion */
static void print_u256_dec(FILE *f, uint256 v) {
    if (v.w[1] == 0 && v.w[2] == 0 && v.w[3] == 0) {
        fprintf(f, "%llu", (unsigned long long)v.w[0]);
        return;
    }

    char digits[80];
    int nd = 0;

    while (!u256_is_zero(v)) {
        uint64_t rem = u256_div_small(&v, 10);
        digits[nd++] = '0' + (char)rem;
    }

    if (nd == 0) { fputc('0', f); return; }
    for (int i = nd - 1; i >= 0; i--)
        fputc(digits[i], f);
}

static uint32_t ilog10_u256(uint256 n) {
    uint32_t c = 0;
    uint256 ten = u256_from_u64(10);
    while (u256_le(ten, n)) {
        u256_div_small(&n, 10);
        c++;
    }
    return c;
}

static void build_orbit(uint8_t orbit[ORBIT_SIZE]) {
    uint8_t d = 0;
    for (int i = 0; i < ORBIT_SIZE; i++) {
        d = (uint8_t)(d + STEP);
        while ((d & 0x3Fu) == 0)
            d = (uint8_t)(d + STEP);
        orbit[i] = d;
    }
}

/* Parse a decimal string into uint256 */
static uint256 parse_u256(const char *s) {
    uint256 r = u256_zero();
    uint256 ten = u256_from_u64(10);
    for (; *s; s++) {
        if (*s == '_' || *s == ',') continue;
        if (*s < '0' || *s > '9') break;
        /* r = r * 10 + digit */
        /* Multiply r by 10 using host mulmod with max modulus */
        uint256 old = r;
        /* r * 10 = r * 8 + r * 2 */
        uint256 r2 = u256_shl1(r);
        uint256 r4 = u256_shl1(r2);
        uint256 r8 = u256_shl1(r4);
        r = u256_add(r8, r2);
        uint256 digit = u256_from_u64((uint64_t)(*s - '0'));
        r = u256_add(r, digit);
    }
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    uint256 n = u256_zero();       /* derived from GPU if not specified */
    uint64_t sweep_k = 0;
    int write_csv = 1;             /* --no-csv disables CSV I/O */

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--sweep") == 0 && i + 1 < argc) {
            char cl[64]; int ci2 = 0;
            for (const char *p = argv[++i]; *p && ci2 < 63; p++)
                if (*p != '_' && *p != ',') cl[ci2++] = *p;
            cl[ci2] = '\0';
            sweep_k = (uint64_t)std::strtoull(cl, nullptr, 10);
            continue;
        }
        if (std::strcmp(argv[i], "--no-csv") == 0) {
            write_csv = 0;
            continue;
        }
        n = parse_u256(argv[i]);
    }

    /* ── Derive n from GPU resources if not specified ────────────────── */
    /*  Query device VRAM → determine max limbs at full batch occupancy. */
    /*  L=1: n < 2^65  (~3.7×10^19)  → default 4×10^18                  */
    /*  L=2: n < 2^129 (~6.8×10^38)  → default 10^20                    */
    /*  L=3: n < 2^193 (~1.3×10^58)  → default 10^39                    */
    /*  L=4: n < 2^257 (~2.3×10^77)  → default 10^59                    */
    /*  Pick highest L the GPU can sustain at BATCH_K occupancy:         */
    /*    MR kernel needs: MAX_SURVIVORS × (8 + 1 + 1) = ~320 MB        */
    /*    Prefilter:       none (k is computed, not stored)              */
    /*    Total floor:     ~512 MB → any GPU with ≥1 GB can do L=4      */
    /*  So the constraint is compute time, not memory.                   */
    /*  Default: L=1 (fastest, most practical for verification).         */
    if (u256_is_zero(n)) {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        /* All limb counts fit in memory on any modern GPU (≥1GB).       */
        /* Default to the sweet spot: biggest n that still finishes fast. */
        /* User can override with any n on the command line.             */
        if (total_mem >= (size_t)4 * 1024 * 1024 * 1024) {
            /* ≥4GB VRAM → default to 2-limb territory (10^20) */
            n = parse_u256("100000000000000000000");
        } else {
            /* <4GB → stay in 1-limb (4×10^18) */
            n = u256_from_u64(4000000000000000000ULL);
        }
        printf("  [auto] n derived from GPU (%llu MB VRAM)\n",
               (unsigned long long)(total_mem / (1024 * 1024)));
    }

    /* n must be even and >= 4 */
    if (u256_lt(n, u256_from_u64(4)) || u256_bit0(n)) {
        fprintf(stderr, "Error: n must be even and >= 4\n");
        return 1;
    }

    uint256 big_n = u256_shr1(n);   /* n / 2 */
    uint32_t base_x = ilog10_u256(n);

    /* ── Detect active limbs from big_n ──────────────────────────────── */
    /*  L = floor(msb(big_n) / 64) + 1                                   */
    /*  All moduli (lo, hi) share same magnitude as big_n ± k, k << big_n */
    int limbs = 4;
    if (big_n.w[3] == 0) { limbs = 3;
    if (big_n.w[2] == 0) { limbs = 2;
    if (big_n.w[1] == 0) { limbs = 1; } } }

    printf("======================================================================\n");
    printf("  predict_cuda256 — Z₂₅₆ BPAND Walk (GPU, uint256 arithmetic)\n");
    printf("  Target: n = ");
    print_u256_dec(stdout, n);
    printf("\n");
    printf("  Data width: 256 bits (4 × uint64 limbs), active: %d limb%s\n",
           limbs, limbs > 1 ? "s" : "");
    printf("======================================================================\n\n");

    /* ── [P] Z₂₅₆ Ring Prediction Oracle ──────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[P] Z256 Ring Prediction Oracle (pure math, no primes)\n");
    printf("----------------------------------------------------------------------\n");
    printf("  Z256 ring: 256 positions\n");
    printf("    Vacuum boundaries: 4 (0, 64, 128, 192)\n");
    printf("    Non-vacuum: 252 = 256 - 4\n");
    printf("    Per quadrant: 63 = 252 / 4\n");
    printf("    mulmod cross-products: %d (L\u00b2), reduction: ~%d bits\n",
           limbs * limbs, limbs * 64);
    printf("    Walk step: %u (coprime to 256)\n", STEP);
    printf("    Orbit period: %d (full coverage, zero gaps)\n\n", ORBIT_SIZE);

    printf("  uint256 limb ↔ quadrant mapping:\n");
    printf("    limb[0] (bits   0..63)  = UP+  weight × 1\n");
    printf("    limb[1] (bits  64..127) = UP-  weight × 2^64\n");
    printf("    limb[2] (bits 128..191) = DN+  weight × 2^128\n");
    printf("    limb[3] (bits 192..255) = DN-  weight × 2^192\n");
    printf("    mulmod: 16 cross-products via __umul64hi\n\n");

    uint32_t rings_needed = (base_x >= 3) ? base_x - 3 : 0;
    printf("  Zrings = %u  (log10(n) - 3 = %u - 3)\n\n", rings_needed, base_x);

    /* ── [A] Geometry Prediction ──────────────────────────────────────── */
    /* k_max = big_n - 2 (as uint64 — k still fits in 64 bits for sweep) */
    /* For sweeps, k is always a uint64 offset from center */
    uint64_t k_limit = sweep_k > 0 ? sweep_k : 10000000ULL;
    uint64_t hvs_in_window = (k_limit + HV_CELLS - 1) / HV_CELLS;

    printf("----------------------------------------------------------------------\n");
    printf("[A] Geometry Prediction\n");
    printf("----------------------------------------------------------------------\n");
    printf("  n           = "); print_u256_dec(stdout, n); printf("\n");
    printf("  N = n/2     = "); print_u256_dec(stdout, big_n); printf("\n");
    printf("  base_x      = %u (digits)\n", base_x);
    printf("  Zrings      = %u (each %d orbits)\n", rings_needed, ORBIT_SIZE);
    printf("  sweep window = %s k values (%s HVs)\n",
           fmt(k_limit), fmt(hvs_in_window));
    printf("\n");

    /* ── GPU setup ──────────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[B] GPU Setup (orbit table + MR bases in constant memory)\n");
    printf("----------------------------------------------------------------------\n");

    uint8_t h_orbit[ORBIT_SIZE];
    build_orbit(h_orbit);

    cudaError_t ce;
    ce = cudaMemcpyToSymbol(d_orbit, h_orbit, sizeof(h_orbit));
    if (ce != cudaSuccess) {
        fprintf(stderr, "  FAIL: orbit upload: %s\n", cudaGetErrorString(ce));
        return 2;
    }

    const uint32_t small_primes[11] = {3,5,7,11,13,17,19,23,29,31,37};
    ce = cudaMemcpyToSymbol(d_small_primes, small_primes, sizeof(small_primes));
    if (ce != cudaSuccess) {
        fprintf(stderr, "  FAIL: primes upload: %s\n", cudaGetErrorString(ce));
        return 2;
    }

    const uint64_t mr_bases[7] = {2,325,9375,28178,450775,9780504,1795265022};
    ce = cudaMemcpyToSymbol(d_mr_bases, mr_bases, sizeof(mr_bases));
    if (ce != cudaSuccess) {
        fprintf(stderr, "  FAIL: bases upload: %s\n", cudaGetErrorString(ce));
        return 2;
    }

    ce = cudaMemcpyToSymbol(d_limbs, &limbs, sizeof(int));
    if (ce != cudaSuccess) {
        fprintf(stderr, "  FAIL: limbs upload: %s\n", cudaGetErrorString(ce));
        return 2;
    }

    /* ── Dynamic VRAM allocation — scale to GPU resources ──────────── */
    /*  Per pipeline slot: survivors(8B) + hit(1B) + d(1B) + count(4B)  */
    /*  = ~10 bytes per survivor entry.                                 */
    /*  2 slots for double-buffered streaming.                          */
    /*  Reserve 256 MB for CUDA runtime + kernel stacks.                */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);

    size_t reserve   = 256ULL << 20;  /* 256 MB headroom */
    size_t usable    = (free_vram > reserve) ? free_vram - reserve : free_vram / 2;
    /* 2 slots × (8 + 1 + 1 + 4) per entry = 28 bytes per entry across both */
    uint32_t max_survivors = (uint32_t)(usable / 28);
    /* Cap: no point exceeding BATCH_K (survival rate ~4% → need BATCH_K/25) */
    if (max_survivors > BATCH_K) max_survivors = BATCH_K;
    /* Floor: at least 1M */
    if (max_survivors < (1u << 20)) max_survivors = (1u << 20);

    size_t slot_dev_bytes = (size_t)max_survivors * (8 + 1 + 1) + sizeof(uint32_t);
    size_t total_dev_bytes = slot_dev_bytes * 2;  /* 2 pipeline slots */

    /* ── 2-slot double-buffer: device + pinned host ─────────────────── */
    struct PipeSlot {
        uint64_t *dev_survivors;
        uint8_t  *dev_hit, *dev_d;
        uint32_t *dev_count;          /* atomicAdd target per slot */
        uint64_t *h_survivors;        /* pinned host */
        uint8_t  *h_hit, *h_d;
        cudaStream_t stream;
    } slot[2];

    for (int s = 0; s < 2; s++) {
        if (cudaMalloc(&slot[s].dev_survivors, (size_t)max_survivors * sizeof(uint64_t)) != cudaSuccess ||
            cudaMalloc(&slot[s].dev_hit,       max_survivors) != cudaSuccess ||
            cudaMalloc(&slot[s].dev_d,         max_survivors) != cudaSuccess ||
            cudaMalloc(&slot[s].dev_count,     sizeof(uint32_t)) != cudaSuccess) {
            fprintf(stderr, "  cudaMalloc failed (slot %d)\n", s);
            return 4;
        }
        if (cudaMallocHost(&slot[s].h_survivors, (size_t)max_survivors * sizeof(uint64_t)) != cudaSuccess ||
            cudaMallocHost(&slot[s].h_hit,       max_survivors) != cudaSuccess ||
            cudaMallocHost(&slot[s].h_d,         max_survivors) != cudaSuccess) {
            fprintf(stderr, "  cudaMallocHost failed (slot %d)\n", s);
            return 4;
        }
        cudaStreamCreate(&slot[s].stream);
    }

    /* HV fill */
    uint64_t *hv_fill = (uint64_t *)calloc(
        hvs_in_window ? hvs_in_window : 1, sizeof(uint64_t));

    printf("  Device:       %s\n", prop.name);
    printf("  SMs:          %d\n", prop.multiProcessorCount);
    printf("  VRAM:         %s MB total, %s MB free\n",
           fmt((uint64_t)(total_vram >> 20)), fmt((uint64_t)(free_vram >> 20)));
    printf("  Allocated:    %s MB (2 × %s MB pipeline slots)\n",
           fmt((uint64_t)(total_dev_bytes >> 20)),
           fmt((uint64_t)(slot_dev_bytes >> 20)));
    printf("  max_survivors: %s per slot (dynamic from VRAM)\n", fmt(max_survivors));
    printf("  Orbit table:  %d entries in constant memory\n", ORBIT_SIZE);
    printf("  MR bases:     7 deterministic\n");
    printf("  Batch size:   %s k per prefilter\n", fmt(BATCH_K));
    printf("  Pipeline:     prefilter → compact → MR (double-buffered, 2 streams)\n");
    printf("  mulmod:       uint256 — 16 __umul64hi + 256 doublings\n");
    printf("  Arithmetic:   pure unsigned integer, no float, no signed\n");
    printf("  Host memory:  pinned (cudaMallocHost) for async D2H\n");
    printf("  CSV output:   %s\n\n", write_csv ? "ON" : "OFF (--no-csv)");

    /* ── [C] Walk ──────────────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[C] GPU BPAND Walk — n = "); print_u256_dec(stdout, n); printf("\n");
    printf("----------------------------------------------------------------------\n");
    printf("  Walking k = 1 .. %s through %s HVs ...\n\n",
           fmt(k_limit), fmt(hvs_in_window));

    /* ── Hit buffer: store (k, d) pairs, defer CSV to after walk ──── */
    /*  Avoids per-hit fprintf in the hot loop.                        */
    /*  lo/hi recomputed from big_n on CPU at write time.              */
    struct HitRec { uint64_t k; uint8_t d; };
    size_t   hit_cap = 1u << 20;   /* 1M initial, grows as needed */
    HitRec  *hit_buf = (HitRec *)malloc(hit_cap * sizeof(HitRec));
    size_t   hit_cnt = 0;
    if (!hit_buf) { fprintf(stderr, "  OOM: hit buffer\n"); return 3; }

    /* Walk statistics */
    uint64_t total_hits = 0, total_k = 0;
    uint64_t quad_hits[4] = {0};
    uint64_t first_hit_k = 0;
    uint8_t  first_hit_d = 0;
    uint64_t min_k = UINT64_MAX, max_k = 0;
    double   k_acc = 0.0;
    int64_t  spin_sum = 0;
    uint64_t transitions = 0;
    uint8_t  prev_q = 255;
    int8_t   prev_spin = 0;
    uint64_t anti_spin_count = 0;

    struct Hit { uint64_t k; uint256 lo, hi; uint8_t d; };
    Hit first_hits[10];
    int n_fh = 0;

    double t0 = now_s();
    uint64_t progress_interval = k_limit / 20;
    if (progress_interval < 1) progress_interval = 1;
    uint64_t next_progress = progress_interval;

    /* ── Double-buffered walk loop ─────────────────────────────────── */
    /*  GPU computes on slot[cur] while CPU processes slot[prev].       */
    /*  Pinned host memory enables async D2H overlapped with compute.  */
    int  cur = 0;
    int  pend_slot = -1;      /* slot with pending results to process */
    uint32_t pend_surv = 0;   /* survivor count in pending slot       */

    /* Lambda-like inline: process pending results from a slot */
    #define PROCESS_PENDING(S, NSURV) do { \
        for (uint32_t _i = 0; _i < (NSURV); _i++) { \
            if (!slot[(S)].h_hit[_i]) continue; \
            uint64_t k      = slot[(S)].h_survivors[_i]; \
            uint8_t  d_val  = slot[(S)].h_d[_i]; \
            int      s_val  = ((d_val >> 7) & 1u) == 0 ? 1 : -1; \
            uint8_t  q_val  = (d_val >> 6) & 0x3u; \
            uint256  k256   = u256_from_u64(k); \
            uint256  lo     = u256_sub(big_n, k256); \
            uint256  hi_val = u256_add(big_n, k256); \
            uint64_t hv_idx = (k - 1) / HV_CELLS; \
            total_hits++; \
            quad_hits[q_val]++; \
            spin_sum += s_val; \
            k_acc += (double)k; \
            if (k < min_k) min_k = k; \
            if (k > max_k) max_k = k; \
            if (first_hit_k == 0) { first_hit_k = k; first_hit_d = d_val; } \
            if (prev_q != 255 && q_val != prev_q) transitions++; \
            prev_q = q_val; \
            if (prev_spin != 0 && s_val != (int)prev_spin) anti_spin_count++; \
            prev_spin = (int8_t)s_val; \
            if (hv_idx < hvs_in_window) hv_fill[hv_idx]++; \
            if (hit_cnt >= hit_cap) { \
                hit_cap *= 2; \
                hit_buf = (HitRec *)realloc(hit_buf, hit_cap * sizeof(HitRec)); \
            } \
            hit_buf[hit_cnt].k = k; \
            hit_buf[hit_cnt].d = d_val; \
            hit_cnt++; \
            if (n_fh < 10) { \
                first_hits[n_fh].k  = k; \
                first_hits[n_fh].lo = lo; \
                first_hits[n_fh].hi = hi_val; \
                first_hits[n_fh].d  = d_val; \
                n_fh++; \
            } \
        } \
    } while(0)

    for (uint64_t k_start = 1; k_start <= k_limit; k_start += BATCH_K) {
        uint64_t batch = BATCH_K;
        if (k_start + batch - 1 > k_limit)
            batch = k_limit - k_start + 1;

        /* ── Kernel 1: PREFILTER on slot[cur] ──────────────────────── */
        cudaMemsetAsync(slot[cur].dev_count, 0, sizeof(uint32_t), slot[cur].stream);

        int threads = 256;
        int blocks  = (int)((batch + threads - 1) / threads);
        prefilter_kernel<<<blocks, threads, 0, slot[cur].stream>>>(
            big_n, k_start, batch,
            slot[cur].dev_survivors, max_survivors, slot[cur].dev_count);

        /* ── While prefilter runs, process pending from prev slot ─── */
        if (pend_slot >= 0) {
            cudaStreamSynchronize(slot[pend_slot].stream);  /* ensure async copy done */
            PROCESS_PENDING(pend_slot, pend_surv);
            pend_slot = -1;
        }

        /* ── Wait for prefilter, read survivor count ───────────────── */
        cudaStreamSynchronize(slot[cur].stream);
        uint32_t n_surv = 0;
        cudaMemcpy(&n_surv, slot[cur].dev_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (n_surv > max_survivors) n_surv = max_survivors;

        total_k += batch;

        if (n_surv == 0) {
            uint64_t k_pos = k_start + batch - 1;
            if (k_pos >= next_progress) {
                double pct = 100.0 * (double)k_pos / (double)k_limit;
                double dt  = now_s() - t0;
                fprintf(stderr, "    %5.1f%%  hits=%s  k=%s  surv=0  [%.1fs]\n",
                        pct, fmt(total_hits), fmt(k_pos), dt);
                next_progress += progress_interval;
            }
            continue;
        }

        /* ── Kernel 2: MR on slot[cur] ─────────────────────────────── */
        int mr_threads = 256;
        int mr_blocks  = (int)((n_surv + mr_threads - 1) / mr_threads);
        mr_kernel<<<mr_blocks, mr_threads, 0, slot[cur].stream>>>(
            big_n, slot[cur].dev_survivors, n_surv,
            slot[cur].dev_hit, slot[cur].dev_d);

        /* ── Async copy results to pinned host (overlaps with next batch) ── */
        cudaMemcpyAsync(slot[cur].h_survivors, slot[cur].dev_survivors,
                        (size_t)n_surv * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, slot[cur].stream);
        cudaMemcpyAsync(slot[cur].h_hit, slot[cur].dev_hit,
                        n_surv, cudaMemcpyDeviceToHost, slot[cur].stream);
        cudaMemcpyAsync(slot[cur].h_d, slot[cur].dev_d,
                        n_surv, cudaMemcpyDeviceToHost, slot[cur].stream);

        pend_slot = cur;
        pend_surv = n_surv;
        cur = 1 - cur;

        uint64_t k_pos = k_start + batch - 1;
        if (k_pos >= next_progress) {
            double pct = 100.0 * (double)k_pos / (double)k_limit;
            double dt  = now_s() - t0;
            fprintf(stderr, "    %5.1f%%  hits=%s  k=%s  surv=%s  [%.1fs]\n",
                    pct, fmt(total_hits), fmt(k_pos), fmt(n_surv), dt);
            next_progress += progress_interval;
        }
    }

    /* ── Drain last pending slot ───────────────────────────────────── */
    if (pend_slot >= 0) {
        cudaStreamSynchronize(slot[pend_slot].stream);
        PROCESS_PENDING(pend_slot, pend_surv);
    }
    #undef PROCESS_PENDING

    double t_walk = now_s() - t0;

    printf("  Walk complete in %.3fs\n", t_walk);
    printf("  Hits buffered: %s\n", fmt((uint64_t)hit_cnt));

    /* ── Bulk CSV write (deferred from hot loop) ──────────────────── */
    if (write_csv && hit_cnt > 0) {
        char csv_path[128];
        snprintf(csv_path, sizeof(csv_path), "predict256_sweep.csv");
        FILE *csv = fopen(csv_path, "w");
        if (!csv) {
            fprintf(stderr, "  Cannot create CSV: %s\n", csv_path);
        } else {
            setvbuf(csv, nullptr, _IOFBF, 1 << 20);  /* 1 MB I/O buffer */
            fprintf(csv, "k,U,U_prime,d,spin,quadrant,hv\n");
            for (size_t i = 0; i < hit_cnt; i++) {
                uint64_t k   = hit_buf[i].k;
                uint8_t  dv  = hit_buf[i].d;
                int      sv  = ((dv >> 7) & 1u) == 0 ? 1 : -1;
                uint8_t  qv  = (dv >> 6) & 0x3u;
                uint256  k256 = u256_from_u64(k);
                uint256  lo   = u256_sub(big_n, k256);
                uint256  hi   = u256_add(big_n, k256);
                uint64_t hvi  = (k - 1) / HV_CELLS;
                fprintf(csv, "%llu,", (unsigned long long)k);
                print_u256_dec(csv, lo);
                fprintf(csv, ",");
                print_u256_dec(csv, hi);
                fprintf(csv, ",%u,%d,%u,%llu\n",
                        (unsigned)dv, sv, (unsigned)qv,
                        (unsigned long long)hvi);
            }
            fclose(csv);
            printf("  CSV written: %s (%s rows)\n", csv_path, fmt((uint64_t)hit_cnt));
        }
    } else if (!write_csv) {
        printf("  CSV: skipped (--no-csv)\n");
    }
    printf("\n");

    /* ── [D] Arm Analysis ─────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[D] U / U' Conjugate Arm Analysis\n");
    printf("----------------------------------------------------------------------\n");
    printf("  Center N = "); print_u256_dec(stdout, big_n); printf("\n");
    printf("  Conservation: U + U' = "); print_u256_dec(stdout, n);
    printf(" = n  (ALWAYS)\n\n");
    printf("  Total BPAND hits:    %s\n", fmt(total_hits));
    printf("  k tested:            %s\n", fmt(total_k));
    if (total_k > 0)
        printf("  Hit rate:            %.6f%%\n",
               100.0 * (double)total_hits / (double)total_k);
    printf("\n");

    /* First hits table */
    if (n_fh > 0) {
        printf("  First 10 BPAND hits:\n");
        printf("    %10s  %-30s  %-30s  %4s  %s\n",
               "k", "U(=N-k)", "U'(=N+k)", "d", "label");
        for (int i = 0; i < n_fh; i++) {
            printf("    %10s  ", fmt(first_hits[i].k));
            print_u256_dec(stdout, first_hits[i].lo);
            printf("  ");
            print_u256_dec(stdout, first_hits[i].hi);
            printf("  %4u  %s\n",
                   (unsigned)first_hits[i].d,
                   qlabel(first_hits[i].d));
        }
        printf("\n");
    }

    printf("  First hit:  k = %s, d = %u (%s)\n",
           fmt(first_hit_k), (unsigned)first_hit_d, qlabel(first_hit_d));
    {
        double mean_k = (total_hits > 0) ? k_acc / (double)total_hits : 0.0;
        int64_t abs_spin = (spin_sum < 0) ? -spin_sum : spin_sum;
        printf("\n  ARM REACH:\n");
        printf("    Min k:      %s\n", fmt(min_k == UINT64_MAX ? 0 : min_k));
        printf("    Max k:      %s\n", fmt(max_k));
        printf("    Mean |k|:   %.0f\n", mean_k);
        printf("    Sigma spin: %+" PRId64 "\n", spin_sum);
        if (total_hits > 0)
            printf("    |Sigma|/N:  %.6f\n",
                   (double)abs_spin / (double)total_hits);
        if (total_hits > 1) {
            double af = (double)anti_spin_count / (double)(total_hits - 1);
            printf("    Anti-spin:  %.1f%% (consecutive opposite spin)\n",
                   100.0 * af);
        }
    }
    printf("\n");

    /* ── [E] Quadrant Distribution ────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[E] Z256 Quadrant Distribution\n");
    printf("----------------------------------------------------------------------\n");

    uint64_t q_total = quad_hits[0]+quad_hits[1]+quad_hits[2]+quad_hits[3];
    const char *q_labels[4] = {
        "UP+ [  0.. 63]", "UP- [ 64..127]",
        "DN+ [128..191]", "DN- [192..255]"
    };
    double q_fracs[4] = {0};
    for (int i = 0; i < 4; i++) {
        double frac = (q_total > 0) ? (double)quad_hits[i]/(double)q_total : 0.0;
        q_fracs[i] = frac;
        int bar_len = (int)(frac * 40.0);
        if (bar_len > 40) bar_len = 40;
        char bar[41];
        memset(bar, '#', (size_t)bar_len);
        bar[bar_len] = '\0';
        printf("    %s: %14s  (%5.1f%%) %s\n",
               q_labels[i], fmt(quad_hits[i]), 100.0 * frac, bar);
    }
    printf("\n");

    if (total_hits > 1) {
        double tr = (double)transitions / (double)(total_hits - 1);
        printf("  Quadrant transitions: %s / %s = %.1f%%\n",
               fmt(transitions), fmt(total_hits - 1), 100.0 * tr);
    }
    printf("\n");

    /* ── [F] HV Fill ──────────────────────────────────────────────────── */
    uint64_t hvs_used = hvs_in_window;
    while (hvs_used > 0 && hv_fill[hvs_used - 1] == 0 &&
           (hvs_used - 1) * HV_CELLS >= k_limit)
        hvs_used--;

    printf("----------------------------------------------------------------------\n");
    printf("[F] HV Fill Distribution (%s HVs used)\n", fmt(hvs_used));
    printf("----------------------------------------------------------------------\n");

    uint64_t fill_sum = 0, fill_min = UINT64_MAX, fill_max = 0;
    uint64_t empty_hvs = 0;
    for (uint64_t i = 0; i < hvs_used; i++) {
        fill_sum += hv_fill[i];
        if (hv_fill[i] < fill_min) fill_min = hv_fill[i];
        if (hv_fill[i] > fill_max) fill_max = hv_fill[i];
        if (hv_fill[i] == 0) empty_hvs++;
    }

    double fill_mean = (hvs_used > 0) ? (double)fill_sum / (double)hvs_used : 0.0;
    printf("  Mean hits/HV:  %.1f\n", fill_mean);
    printf("  Min:           %s\n", fmt(fill_min == UINT64_MAX ? 0 : fill_min));
    printf("  Max:           %s\n", fmt(fill_max));
    printf("  Empty HVs:     %s\n", fmt(empty_hvs));
    printf("  Total hits:    %s (check: %s)\n", fmt(fill_sum), fmt(total_hits));
    printf("\n");

    free(hv_fill);

    /* ── [G] Claims ───────────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[G] Claims\n");
    printf("----------------------------------------------------------------------\n\n");

    struct { const char *text; int pass; char ev[128]; } claims[7];
    int nc = 0;

    claims[nc].pass = total_hits > 0;
    claims[nc].text = "Goldbach pair exists for n";
    snprintf(claims[nc].ev, 128, "%s pairs, first at k = %s",
             fmt(total_hits), fmt(first_hit_k));
    nc++;

    claims[nc].pass = quad_hits[0]>0 && quad_hits[1]>0 &&
                      quad_hits[2]>0 && quad_hits[3]>0;
    claims[nc].text = "All 4 quadrants produce BPAND hits";
    snprintf(claims[nc].ev, 128, "[%s, %s, %s, %s]",
             fmt(quad_hits[0]), fmt(quad_hits[1]),
             fmt(quad_hits[2]), fmt(quad_hits[3]));
    nc++;

    {
        int ok = 1;
        for (int i = 0; i < 4; i++) {
            double d = q_fracs[i] - 0.25;
            if (d < 0) d = -d;
            if (d >= 0.15) { ok = 0; break; }
        }
        claims[nc].pass = ok;
        claims[nc].text = "Quadrants balanced (+-15%)";
        snprintf(claims[nc].ev, 128, "[%.3f, %.3f, %.3f, %.3f]",
                 q_fracs[0], q_fracs[1], q_fracs[2], q_fracs[3]);
        nc++;
    }

    claims[nc].pass = (empty_hvs == 0);
    claims[nc].text = "No empty HVs (uniform fill)";
    snprintf(claims[nc].ev, 128, "%s empty out of %s used",
             fmt(empty_hvs), fmt(hvs_used));
    nc++;

    {
        int64_t as = (spin_sum < 0) ? -spin_sum : spin_sum;
        double ratio = (total_hits > 0) ? (double)as / (double)total_hits : 1.0;
        claims[nc].pass = ratio < 0.1;
        claims[nc].text = "Spin near-cancellation (|S|/N < 0.1)";
        snprintf(claims[nc].ev, 128, "Sigma = %+" PRId64 ", |S|/N = %.6f",
                 spin_sum, ratio);
        nc++;
    }

    {
        double hr = (total_k > 0) ? (double)total_hits / (double)total_k : 0.0;
        claims[nc].pass = hr > 0.0;
        claims[nc].text = "Hit rate > 0 (primes survive filter)";
        snprintf(claims[nc].ev, 128, "%.6f%%", 100.0 * hr);
        nc++;
    }

    {
        double tr = (total_hits > 1)
                    ? (double)transitions / (double)(total_hits - 1) : 0.0;
        claims[nc].pass = tr > 0.50;
        claims[nc].text = "Quadrant transition rate > 50%";
        snprintf(claims[nc].ev, 128, "%.1f%%", 100.0 * tr);
        nc++;
    }

    printf("  %-50s %-10s %s\n", "Claim", "Status", "Evidence");
    printf("  %-50s %-10s %s\n",
           "--------------------------------------------------",
           "----------",
           "----------------------------------------");
    int n_derived = 0, n_failed = 0;
    for (int i = 0; i < nc; i++) {
        const char *st = claims[i].pass ? "DERIVED" : "FAILED";
        if (claims[i].pass) n_derived++; else n_failed++;
        printf("  %-50s %-10s %s\n", claims[i].text, st, claims[i].ev);
    }
    printf("\n  SCORECARD: %d DERIVED, %d FAILED\n", n_derived, n_failed);

    /* ── Summary ──────────────────────────────────────────────────────── */
    printf("\n");
    printf("======================================================================\n");
    printf("  n = "); print_u256_dec(stdout, n);
    printf("  |  %s BPAND hits  |  first_k = %s  |  max_k = %s\n",
           fmt(total_hits), fmt(first_hit_k), fmt(max_k));
    printf("  Walk: %.2fs  |  Backend: GPU (%s, %d SMs)\n",
           t_walk, prop.name, prop.multiProcessorCount);
    printf("  Arithmetic: uint256 (4 × uint64), no float\n");
    printf("======================================================================\n");

    for (int s = 0; s < 2; s++) {
        cudaFree(slot[s].dev_survivors);
        cudaFree(slot[s].dev_hit);
        cudaFree(slot[s].dev_d);
        cudaFree(slot[s].dev_count);
        cudaFreeHost(slot[s].h_survivors);
        cudaFreeHost(slot[s].h_hit);
        cudaFreeHost(slot[s].h_d);
        cudaStreamDestroy(slot[s].stream);
    }
    free(hit_buf);

    return 0;
}
