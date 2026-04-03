/*
 * predict.c  —  Zring Prediction Oracle + Full BPAND Walk
 *
 * C port of predict.rs  (Goldbach / Z₂₅₆ ring analysis).
 * Target: 100 B  (n = 100,000,000,000).
 *
 * For a given even n, walks k = 1..N-2 through the Z₂₅₆ ring,
 * tracking every Goldbach pair hit with:
 *   - HV fill  (hits per 14,464-cell batch)
 *   - Quadrant distribution  (spin × parity)
 *   - Quadrant transition rate between consecutive hits
 *   - Arm reach statistics  (min / max / mean k)
 *   - U/U' conjugate arm analysis
 *
 * Compile:
 *   gcc  -O3 -o predict  src/predict.c
 *   clang -O3 -o predict  src/predict.c
 *   cl /O2 /Fe:predict.exe src/predict.c         (MSVC — no __int128)
 *
 * Usage:  predict <n>
 *   e.g.  predict 100000000000
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <limits.h>

/* ── High-resolution timer ─────────────────────────────────────────────────*/
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

/* ── Integer helpers (no FPU) ──────────────────────────────────────────────*/
static size_t isqrt_u(size_t n) {
    if (n == 0) return 0;
    size_t tmp = n;
    int bits = 0;
    while (tmp) { bits++; tmp >>= 1; }
    size_t x = (size_t)1 << ((bits + 1) / 2);
    for (;;) {
        size_t y = (x + n / x) / 2;
        if (y >= x) return x;
        x = y;
    }
}

static uint32_t ilog10_u(size_t n) {
    if (n == 0) return 0;
    uint32_t count = 0;
    size_t v = n;
    while (v >= 10) { v /= 10; count++; }
    return count;
}

/* ── Z₂₅₆ Geometry Constants ───────────────────────────────────────────────*/
#define STEP             ((uint8_t)7)
#define HALF             128
#define VACUUMS_PER_HALF 2
#define ACTIVE_PER_HALF  (HALF - VACUUMS_PER_HALF)   /* 126          */
#define ORBIT_SIZE       (2 * ACTIVE_PER_HALF)        /* 252          */
#define HV_CELLS         (HALF * 113)                 /* 14,464       */
#define HV_W             113
#define ORBITS_PER_HV    (HV_CELLS / ORBIT_SIZE)      /* 57           */
#define FULL_SIEVE_LIMIT 10000000000ULL
#define SEGMENT_K_BLOCK  200000000ULL
#define FIRST_HIT_BATCH_K 262144ULL

typedef struct {
    size_t   low;
    size_t   high;
    size_t   first_odd;
    size_t   odd_count;
    size_t   bytes;
    uint8_t *bits;
} OddSegment;

typedef struct {
    int             use_full;
    const uint8_t  *full_sieve;
    const uint32_t *base_primes;
    size_t          base_prime_count;
    size_t          block_k;
    size_t          seg_k_start;
    size_t          seg_k_end;
    OddSegment      u_seg;
    OddSegment      v_seg;
    double          segment_build_time;
} PrimeCtx;

/* ── Bitwise Prime Sieve (1 bit per odd) ───────────────────────────────────*/
static uint8_t *make_sieve(size_t limit, size_t *out_bytes) {
    size_t size  = limit / 2 + 1;
    size_t bytes = (size + 7) / 8;
    uint8_t *sieve = (uint8_t *)malloc(bytes);
    if (!sieve) { fprintf(stderr, "OOM: sieve malloc (%zu bytes)\n", bytes); exit(1); }
    memset(sieve, 0xFF, bytes);
    sieve[0] &= ~(uint8_t)1;   /* 1 is not prime */

    size_t sq = isqrt_u(limit);
    for (size_t i = 3; i <= sq; i += 2) {
        size_t idx = i / 2;
        if (sieve[idx >> 3] & (uint8_t)(1u << (idx & 7u))) {
            for (size_t j = i * i; j <= limit; j += 2 * i) {
                size_t jdx = j / 2;
                sieve[jdx >> 3] &= ~(uint8_t)(1u << (jdx & 7u));
            }
        }
    }
    *out_bytes = bytes;
    return sieve;
}

static inline uint8_t is_prime(const uint8_t *sieve, size_t n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if ((n & 1) == 0) return 0;
    size_t idx = n / 2;
    return (uint8_t)((sieve[idx >> 3] >> (idx & 7u)) & 1u);
}

/* Fast path when caller already guarantees n is odd and >= 3. */
static inline uint8_t is_prime_odd_fast(const uint8_t *sieve, size_t odd_n) {
    size_t idx = odd_n >> 1;
    return (uint8_t)((sieve[idx >> 3] >> (idx & 7u)) & 1u);
}

static void odd_segment_free(OddSegment *seg) {
    if (seg->bits) {
        free(seg->bits);
    }
    memset(seg, 0, sizeof(*seg));
}

static uint8_t odd_segment_is_prime(const OddSegment *seg, size_t n) {
    if (n == 2) return 1;
    if (n < 2 || (n & 1u) == 0) return 0;
    if (seg->odd_count == 0 || n < seg->low || n > seg->high || n < seg->first_odd) return 0;
    size_t idx = (n - seg->first_odd) >> 1;
    if (idx >= seg->odd_count) return 0;
    return (uint8_t)((seg->bits[idx >> 3] >> (idx & 7u)) & 1u);
}

static void build_odd_segment(OddSegment *seg,
                              size_t low,
                              size_t high,
                              const uint32_t *base_primes,
                              size_t base_prime_count) {
    odd_segment_free(seg);
    seg->low = low;
    seg->high = high;

    if (high < low || high < 3) {
        return;
    }

    seg->first_odd = low | 1ULL;
    if (seg->first_odd > high) {
        return;
    }

    seg->odd_count = ((high - seg->first_odd) >> 1) + 1;
    seg->bytes = (seg->odd_count + 7) / 8;
    seg->bits = (uint8_t *)malloc(seg->bytes);
    if (!seg->bits) {
        fprintf(stderr, "OOM: segmented sieve (%zu bytes)\n", seg->bytes);
        exit(1);
    }
    memset(seg->bits, 0xFF, seg->bytes);

    if (seg->first_odd == 1) {
        seg->bits[0] &= (uint8_t)~1u;
    }

    for (size_t i = 0; i < base_prime_count; i++) {
        uint32_t p32 = base_primes[i];
        if (p32 == 2) {
            continue;
        }
        size_t p = (size_t)p32;
        size_t p2 = p * p;
        if (p2 > high) {
            break;
        }

        size_t start = ((low + p - 1) / p) * p;
        if (start < p2) start = p2;
        if ((start & 1u) == 0) start += p;

        size_t step = p << 1;
        for (size_t m = start; m <= high; m += step) {
            if (m < seg->first_odd) continue;
            size_t idx = (m - seg->first_odd) >> 1;
            seg->bits[idx >> 3] &= (uint8_t)~(1u << (idx & 7u));
            if (high - m < step) break;
        }
    }
}

static uint32_t *collect_base_primes(size_t limit, size_t *count_out, size_t *mem_bytes_out) {
    size_t sieve_bytes = 0;
    uint8_t *sieve = make_sieve(limit, &sieve_bytes);

    size_t count = (limit >= 2) ? 1 : 0;
    for (size_t p = 3; p <= limit; p += 2) {
        if (is_prime_odd_fast(sieve, p)) count++;
    }

    uint32_t *primes = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!primes) {
        fprintf(stderr, "OOM: base primes list (%zu entries)\n", count);
        exit(1);
    }

    size_t w = 0;
    if (limit >= 2) primes[w++] = 2;
    for (size_t p = 3; p <= limit; p += 2) {
        if (is_prime_odd_fast(sieve, p)) {
            primes[w++] = (uint32_t)p;
        }
    }

    free(sieve);
    *count_out = w;
    *mem_bytes_out = sieve_bytes + w * sizeof(uint32_t);
    return primes;
}

static void prime_ctx_build_block(PrimeCtx *ctx, size_t k, size_t big_n) {
    size_t k_max = big_n - 1;
    size_t k_start = k;
    size_t k_end = k_start + ctx->block_k - 1;
    if (k_end > k_max) k_end = k_max;

    size_t u_low = big_n - k_end;
    size_t u_high = big_n - k_start;
    size_t v_low = big_n + k_start;
    size_t v_high = big_n + k_end;

    double t0 = now_s();
    build_odd_segment(&ctx->u_seg, u_low, u_high, ctx->base_primes, ctx->base_prime_count);
    build_odd_segment(&ctx->v_seg, v_low, v_high, ctx->base_primes, ctx->base_prime_count);
    ctx->segment_build_time += now_s() - t0;

    ctx->seg_k_start = k_start;
    ctx->seg_k_end = k_end;
}

/* ── Z₂₅₆ decode ────────────────────────────────────────────────────────── */
static inline void decode(uint8_t d,
                          int *spin, int *parity,
                          uint8_t *momentum, uint8_t *quadrant) {
    *spin     = ((d >> 7) & 1u) == 0 ?  1 : -1;
    *parity   = ((d >> 6) & 1u) == 0 ?  1 : -1;
    *momentum = d & 0x3Fu;
    *quadrant = (d >> 6) & 0x3u;
}

static const char *qlabel(uint8_t d) {
    switch ((d >> 6) & 0x3u) {
        case 0: return "UP+";
        case 1: return "UP-";
        case 2: return "DN+";
        case 3: return "DN-";
    }
    return "???";
}

/* ── predict_hv — pure arithmetic ──────────────────────────────────────────*/
static void predict_hv(size_t n, size_t *k_max_out,
                        size_t *orbits_out, size_t *hvs_out) {
    size_t big_n  = n / 2;
    size_t k_max  = (big_n >= 2) ? big_n - 2 : 0;
    size_t orbits = (k_max + ORBIT_SIZE - 1) / ORBIT_SIZE;
    size_t hvs    = (k_max + HV_CELLS   - 1) / HV_CELLS;
    *k_max_out  = k_max;
    *orbits_out = orbits;
    *hvs_out    = hvs;
}

static inline uint64_t addmod_u64(uint64_t a, uint64_t b, uint64_t m) {
    if (a >= m - b) return a - (m - b);
    return a + b;
}

/* Portable multiply-mod for full 64-bit range (no __int128 required). */
static uint64_t mulmod_u64(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t r = 0;
    a %= m;
    while (b) {
        if (b & 1u) r = addmod_u64(r, a, m);
        b >>= 1;
        if (b) a = addmod_u64(a, a, m);
    }
    return r;
}

static uint64_t powmod_u64(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1;
    a %= m;
    while (e) {
        if (e & 1u) r = mulmod_u64(r, a, m);
        e >>= 1;
        if (e) a = mulmod_u64(a, a, m);
    }
    return r;
}

/* Deterministic Miller-Rabin for uint64. */
static int is_prime_u64(uint64_t n) {
    static const uint32_t small_primes[] = {
        2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u
    };
    static const uint64_t bases[] = {
        2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL
    };

    if (n < 2) return 0;
    for (size_t i = 0; i < sizeof(small_primes)/sizeof(small_primes[0]); i++) {
        uint32_t p = small_primes[i];
        if (n == p) return 1;
        if (n % p == 0) return 0;
    }

    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1u) == 0) {
        d >>= 1;
        s++;
    }

    for (size_t i = 0; i < sizeof(bases)/sizeof(bases[0]); i++) {
        uint64_t a = bases[i] % n;
        if (a == 0) continue;
        uint64_t x = powmod_u64(a, d, n);
        if (x == 1 || x == n - 1) continue;

        int witness = 1;
        for (unsigned r = 1; r < s; r++) {
            x = mulmod_u64(x, x, n);
            if (x == n - 1) {
                witness = 0;
                break;
            }
        }
        if (witness) return 0;
    }
    return 1;
}

static int passes_small_prime_prefilter_u64(uint64_t n) {
    static const uint32_t wheel[] = {
        3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u
    };
    for (size_t i = 0; i < sizeof(wheel) / sizeof(wheel[0]); i++) {
        uint32_t p = wheel[i];
        if (n == p) return 1;
        if (n % p == 0) return 0;
    }
    return 1;
}

typedef struct {
    uint64_t k;
    uint64_t lo;
    uint64_t hi;
    uint8_t  d;
    uint64_t probe_rank;
    uint64_t filtered_rank;
} FirstHitCandidate;

typedef struct {
    uint64_t probes;
    uint64_t prefiltered;
    uint64_t mr_tests;
} FirstHitStats;

static int first_hit_search_u64(uint64_t n,
                                uint64_t *found_k,
                                uint64_t *found_lo,
                                uint64_t *found_hi,
                                uint8_t *found_d,
                                uint64_t *tested,
                                FirstHitStats *stats) {
    uint64_t big_n = n / 2;
    uint8_t d_ring = 0;
    uint64_t probes = 0;
    uint64_t prefiltered = 0;
    uint64_t mr_tests = 0;

    FirstHitCandidate *batch = (FirstHitCandidate *)malloc(
        FIRST_HIT_BATCH_K * sizeof(FirstHitCandidate)
    );
    if (!batch) {
        fprintf(stderr, "OOM: first-hit batch allocation\n");
        exit(1);
    }

    uint64_t k = 1;
    while (k < big_n) {
        size_t used = 0;
        uint64_t block_end = k + FIRST_HIT_BATCH_K;
        if (block_end > big_n) block_end = big_n;

        while (k < block_end) {
            d_ring = (uint8_t)(d_ring + STEP);
            while ((d_ring & 0x3Fu) == 0) {
                d_ring = (uint8_t)(d_ring + STEP);
            }

            uint64_t lo = big_n - k;
            uint64_t hi = big_n + k;
            if (lo < 2) {
                k = big_n;
                break;
            }

            if (((big_n ^ k) & 1u) == 0 && lo != 2) {
                k++;
                continue;
            }

            probes++;
            if (!passes_small_prime_prefilter_u64(lo) || !passes_small_prime_prefilter_u64(hi)) {
                k++;
                continue;
            }

            if (used < FIRST_HIT_BATCH_K) {
                prefiltered++;
                batch[used].k = k;
                batch[used].lo = lo;
                batch[used].hi = hi;
                batch[used].d = d_ring;
                batch[used].probe_rank = probes;
                batch[used].filtered_rank = prefiltered;
                used++;
            }
            k++;
        }

        for (size_t i = 0; i < used; i++) {
            mr_tests++;
            if (is_prime_u64(batch[i].lo) && is_prime_u64(batch[i].hi)) {
                *found_k = batch[i].k;
                *found_lo = batch[i].lo;
                *found_hi = batch[i].hi;
                *found_d = batch[i].d;
                *tested = batch[i].probe_rank;
                if (stats) {
                    stats->probes = batch[i].probe_rank;
                    stats->prefiltered = batch[i].filtered_rank;
                    stats->mr_tests = mr_tests;
                }
                free(batch);
                return 1;
            }
        }
    }

    *tested = probes;
    if (stats) {
        stats->probes = probes;
        stats->prefiltered = prefiltered;
        stats->mr_tests = mr_tests;
    }
    free(batch);
    return 0;
}

/* ── format_num — comma-separated digit groups (ring buffer, 8 slots) ──────*/
#define FMT_SLOTS 8
static char _fmt_pool[FMT_SLOTS][32];
static int  _fmt_idx = 0;

static const char *fmt(size_t n) {
    char *buf = _fmt_pool[_fmt_idx % FMT_SLOTS];
    _fmt_idx++;
    char tmp[24];
    int len = snprintf(tmp, sizeof(tmp), "%zu", n);
    int out = 0;
    for (int i = 0; i < len; i++) {
        int remaining = len - i;
        if (i > 0 && remaining % 3 == 0) buf[out++] = ',';
        buf[out++] = tmp[i];
    }
    buf[out] = '\0';
    return buf;
}

/* ── main ───────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {

    /* ── parse n ── */
    size_t n = 10000000ULL;
    int mode_first_hit = 0;
    int gpu_hint = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--first-hit") == 0 || strcmp(argv[i], "--mode=first-hit") == 0) {
            mode_first_hit = 1;
            continue;
        }
        if (strcmp(argv[i], "--gpu") == 0) {
            gpu_hint = 1;
            continue;
        }
        char clean[32]; int ci = 0;
        for (const char *p = argv[i]; *p && ci < 31; p++) {
            if (*p != '_' && *p != ',') clean[ci++] = *p;
        }
        clean[ci] = '\0';
        n = (size_t)strtoull(clean, NULL, 10);
    }

    if (n < 4 || (n & 1) != 0) {
        fprintf(stderr, "Error: n must be an even number >= 4\n");
        return 1;
    }

    size_t   big_n  = n / 2;
    uint32_t base_x = ilog10_u(n);

    if (mode_first_hit) {
        printf("======================================================================\n");
        printf("  predict --first-hit  (single Goldbach witness mode)\n");
        printf("  Target: n = %s\n", fmt(n));
        printf("======================================================================\n\n");
        if (gpu_hint) {
            printf("  Backend: CPU deterministic primality (GPU batch path pending wiring)\n");
        }

        double t0 = now_s();
        uint64_t fk = 0, flo = 0, fhi = 0, tested = 0;
        uint8_t fd = 0;
        FirstHitStats st;
        memset(&st, 0, sizeof(st));
        int ok = first_hit_search_u64((uint64_t)n, &fk, &flo, &fhi, &fd, &tested, &st);
        double dt = now_s() - t0;

        if (!ok) {
            printf("  No pair found in scan range.\n");
            printf("  Tested candidates: %s\n", fmt((size_t)tested));
            printf("  Time: %.3fs\n", dt);
            return 2;
        }

        printf("  First hit:\n");
        printf("    k      = %s\n", fmt((size_t)fk));
        printf("    U      = %s\n", fmt((size_t)flo));
        printf("    U'     = %s\n", fmt((size_t)fhi));
        printf("    U+U'   = %s\n", fmt((size_t)(flo + fhi)));
        printf("    d      = %u (%s)\n", (unsigned)fd, qlabel(fd));
        printf("    probes = %s\n", fmt((size_t)tested));
        printf("    filtered candidates = %s\n", fmt((size_t)st.prefiltered));
        printf("    MR tests = %s\n", fmt((size_t)st.mr_tests));
        printf("    time   = %.3fs\n", dt);
        return 0;
    }

    printf("\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\n");
    printf("  predict  \xe2\x80\x94  Zring Prediction Oracle + Full BPAND Walk\n");
    printf("  Target: n = %s\n", fmt(n));
    printf("\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\n\n");

    /* ── [P] Z256 Ring Prediction Oracle ─────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[P] Z256 Ring Prediction Oracle (pure math, no primes)\n");
    printf("----------------------------------------------------------------------\n");
    printf("  Z256 ring: 256 positions\n");
    printf("    Vacuum boundaries: 4 (0, 64, 128, 192)\n");
    printf("    Non-vacuum: 252 = 256 - 4\n");
    printf("    Per quadrant: 63 = 252 / 4\n");
    printf("    63 mod 7 = 0  =>  7 x 9 = 63\n");
    printf("    7-prime walk: 9 steps per cycle, 9 cycles covers 63, 36 cycles covers 252\n");
    printf("    Zero gaps, zero repeats, full coverage\n\n");
    printf("  Four-arm witnessing for k = 1:\n");
    printf("    d1 = %u  d2 = %u  d3 = %u  d4 = %u\n",
           1u, 1u + 64u, 1u + 128u, 1u + 192u);
    printf("    All four point to (N-k, N+k) = (%s, %s)\n",
           fmt(big_n - 1), fmt(big_n + 1));
    printf("    When BPAND = 1, all four ring directions confirm the pair\n\n");
    printf("  Prediction:\n");
    printf("    Each quadrant will be filled in <=63 steps\n");
    printf("    Expected first hit at rotation 9 (one full prime cycle)\n");
    printf("    Sweet spot rule: rings(10^x) = x - 3\n");
    printf("      10^4 -> 1 ring, 10^8 -> 5 rings, 10^12 -> 9 rings, 10^18 -> 15 rings\n\n");

    /* ── [A] Geometry Prediction ─────────────────────────────────────────── */
    size_t   k_max, orbits, hvs;
    predict_hv(n, &k_max, &orbits, &hvs);
    uint32_t rings_needed = (base_x >= 3) ? base_x - 3 : 0;

    printf("----------------------------------------------------------------------\n");
    printf("[A] Geometry Prediction (no sieve needed)\n");
    printf("----------------------------------------------------------------------\n");
    printf("  n           = %s\n", fmt(n));
    printf("  N = n/2     = %s\n", fmt(big_n));
    printf("  k_max       = %s\n", fmt(k_max));
    printf("  orbits      = %s (each %d predictions)\n", fmt(orbits), ORBIT_SIZE);
    printf("  HVs         = %s (each %s cells = %d orbits)\n",
           fmt(hvs), fmt(HV_CELLS), (int)ORBITS_PER_HV);
    printf("  Zrings      = %s (each %d orbits)\n",
           fmt((size_t)rings_needed), ORBIT_SIZE);

    {
        size_t zring_l1 = (hvs > 1) ? (hvs + ORBIT_SIZE - 1) / ORBIT_SIZE : 0;
        size_t zring_l2 = (zring_l1 > 1) ? (zring_l1 + ORBIT_SIZE - 1) / ORBIT_SIZE : 0;
        if (hvs <= 1) {
            printf("  structure   = 1 HV (no chain)\n");
        } else if (zring_l1 <= 1) {
            printf("  structure   = %s HVs -> 1 Zring index\n", fmt(hvs));
        } else if (zring_l2 <= 1) {
            printf("  structure   = %s HVs -> %s Zrings -> 1 root\n",
                   fmt(hvs), fmt(zring_l1));
        } else {
            printf("  structure   = %s HVs -> %s L1 -> %s L2\n",
                   fmt(hvs), fmt(zring_l1), fmt(zring_l2));
        }
    }
    printf("\n");

    /* ── [B] Prime Sieve ─────────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[B] Prime Sieve (bitwise, 1 bit per odd)\n");
    printf("----------------------------------------------------------------------\n");

    PrimeCtx prime_ctx;
    memset(&prime_ctx, 0, sizeof(prime_ctx));

    double   t0 = now_s();
    size_t   sieve_bytes = 0;
    uint8_t *sieve = NULL;
    uint32_t *base_primes = NULL;

    if (n <= FULL_SIEVE_LIMIT) {
        sieve = make_sieve(n, &sieve_bytes);
        prime_ctx.use_full = 1;
        prime_ctx.full_sieve = sieve;
    } else {
        size_t sqrt_n = isqrt_u(n);
        size_t base_prime_count = 0;
        base_primes = collect_base_primes(sqrt_n, &base_prime_count, &sieve_bytes);
        prime_ctx.use_full = 0;
        prime_ctx.base_primes = base_primes;
        prime_ctx.base_prime_count = base_prime_count;
        prime_ctx.block_k = SEGMENT_K_BLOCK;
        prime_ctx.seg_k_start = 1;
        prime_ctx.seg_k_end = 0;
        printf("  Mode: segmented (k-block = %s)\n", fmt(prime_ctx.block_k));
        printf("  Base primes up to sqrt(n) = %s (count = %s)\n", fmt(sqrt_n), fmt(base_prime_count));
    }
    double   t_sieve = now_s() - t0;

    double sieve_mb = (double)sieve_bytes / (1024.0 * 1024.0);
    double sieve_gb = (double)sieve_bytes / (1024.0 * 1024.0 * 1024.0);

    if (sieve_gb >= 1.0)
        printf("  Sieve memory:  %.2f GB (%s bytes)\n", sieve_gb, fmt(sieve_bytes));
    else
        printf("  Sieve memory:  %.1f MB (%s bytes)\n", sieve_mb, fmt(sieve_bytes));
    printf("  Sieve time:    %.3fs\n\n", t_sieve);

    /* ── [C] Full BPAND Walk ─────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[C] Full BPAND Walk  --  n = %s\n", fmt(n));
    printf("----------------------------------------------------------------------\n");
    printf("  Walking k = 1 .. %s through %s HVs ...\n\n", fmt(k_max), fmt(hvs));

    /* Allocate HV fill array */
    uint64_t *hv_fill = (uint64_t *)calloc(hvs ? hvs : 1, sizeof(uint64_t));
    if (!hv_fill) { fprintf(stderr, "OOM: hv_fill\n"); exit(1); }

    /* Walk state */
    uint64_t quad_hits[4]    = {0, 0, 0, 0};
    uint64_t total_hits      = 0;
    uint64_t total_preds     = 0;
    size_t   first_k         = 0;
    uint8_t  first_d         = 0;
    size_t   min_k           = (size_t)-1;
    size_t   max_k           = 0;
    double   k_acc           = 0.0;   /* double avoids __int128 portability */
    int64_t  spin_sum        = 0;
    uint8_t  prev_q          = 255;   /* sentinel */
    uint64_t transitions     = 0;
    int8_t   prev_spin       = 0;
    uint64_t anti_spin_count = 0;

    /* First 10 hits */
    typedef struct { size_t k, lo, hi; uint8_t d; } Hit;
    Hit     first_hits[10];
    int     n_fh = 0;

    /* CSV output */
    char csv_path[64];
    snprintf(csv_path, sizeof(csv_path), "predict_%zu.csv", n);
    FILE *csv = fopen(csv_path, "w");
    if (!csv) { fprintf(stderr, "Cannot create CSV: %s\n", csv_path); exit(1); }
    setvbuf(csv, NULL, _IOFBF, 65536);
    fprintf(csv, "k,U,U_prime,d,spin,quadrant,hv\n");

    t0 = now_s();

    uint8_t d_ring  = 0;
    size_t  k       = 0;
    size_t  hvs_used = 0;
    int     done    = 0;
    size_t  prog_iv = (hvs > 20) ? hvs / 20 : 1;

    for (size_t hv_idx = 0; hv_idx < hvs && !done; hv_idx++) {
        uint64_t hv_total = 0;

        for (int row = 0; row < HALF && !done; row++) {
            for (int col = 0; col < HV_W && !done; col++) {
                k++;
                if (k >= big_n) { done = 1; break; }

                d_ring = (uint8_t)(d_ring + STEP);
                while ((d_ring & 0x3Fu) == 0)
                    d_ring = (uint8_t)(d_ring + STEP);

                total_preds++;

                size_t lo = big_n - k;
                size_t hi = big_n + k;
                if (lo < 2) { done = 1; break; }

                /*
                 * Goldbach hits need both endpoints prime.
                 * For lo=N-k and hi=N+k, parity is equal; if both are even,
                 * the only possible prime case is lo==2.
                 */
                if (((big_n ^ k) & 1u) == 0 && lo != 2) {
                    continue;
                }

                if (!prime_ctx.use_full && (k < prime_ctx.seg_k_start || k > prime_ctx.seg_k_end)) {
                    prime_ctx_build_block(&prime_ctx, k, big_n);
                }

                uint8_t lo_prime;
                uint8_t hi_prime;
                if (prime_ctx.use_full) {
                    lo_prime = (lo == 2) ? 1 : is_prime_odd_fast(prime_ctx.full_sieve, lo);
                    hi_prime = (hi == 2) ? 1 : is_prime_odd_fast(prime_ctx.full_sieve, hi);
                } else {
                    lo_prime = (lo == 2) ? 1 : odd_segment_is_prime(&prime_ctx.u_seg, lo);
                    hi_prime = (hi == 2) ? 1 : odd_segment_is_prime(&prime_ctx.v_seg, hi);
                }

                if (lo_prime && hi_prime) {
                    hv_total++;
                    total_hits++;

                    int     spin, parity;
                    uint8_t momentum, quadrant;
                    decode(d_ring, &spin, &parity, &momentum, &quadrant);

                    quad_hits[quadrant]++;

                    if (first_k == 0) { first_k = k; first_d = d_ring; }
                    if (k < min_k) min_k = k;
                    if (k > max_k) max_k = k;
                    k_acc    += (double)k;
                    spin_sum += spin;

                    if (prev_q != 255 && quadrant != prev_q) transitions++;
                    prev_q = quadrant;

                    if (prev_spin != 0 && spin != (int)prev_spin) anti_spin_count++;
                    prev_spin = (int8_t)spin;

                    fprintf(csv, "%zu,%zu,%zu,%u,%d,%u,%zu\n",
                            k, lo, hi, (unsigned)d_ring,
                            spin, (unsigned)quadrant, hv_idx);

                    if (n_fh < 10) {
                        first_hits[n_fh].k  = k;
                        first_hits[n_fh].lo = lo;
                        first_hits[n_fh].hi = hi;
                        first_hits[n_fh].d  = d_ring;
                        n_fh++;
                    }
                }
            }
        }

        hv_fill[hv_idx] = hv_total;
        hvs_used = hv_idx + 1;

        if (!done && k < big_n && (big_n - k) < 2) done = 1;

        if (hv_idx > 0 && hv_idx % prog_iv == 0) {
            double pct     = 100.0 * (double)hv_idx / (double)hvs;
            double elapsed = now_s() - t0;
            fprintf(stderr, "    %5.1f%%  hits=%s  k=%s  [%.1fs]\n",
                    pct, fmt((size_t)total_hits), fmt(k), elapsed);
        }
    }

    double t_walk = now_s() - t0;
    if (prime_ctx.use_full) {
        free(sieve);
    } else {
        odd_segment_free(&prime_ctx.u_seg);
        odd_segment_free(&prime_ctx.v_seg);
        free(base_primes);
    }
    fclose(csv);

    printf("  Walk complete in %.3fs\n", t_walk);
    if (!prime_ctx.use_full) {
        printf("  Segment build time: %.3fs\n", prime_ctx.segment_build_time);
    }
    printf("  CSV written: %s\n\n", csv_path);

    /* ── [D] U/U' Conjugate Arm Analysis ─────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[D] U / U' Conjugate Arm Analysis\n");
    printf("----------------------------------------------------------------------\n");
    printf("  Center N = %s\n", fmt(big_n));
    printf("  U  = N - k  (arm sweeping LEFT)\n");
    printf("  U' = N + k  (arm sweeping RIGHT)\n");
    printf("  Conservation: U + U' = %s = n  (ALWAYS)\n\n", fmt(n));
    printf("  Total BPAND hits:    %s\n", fmt((size_t)total_hits));
    printf("  Total predictions:   %s\n", fmt((size_t)total_preds));
    if (total_preds > 0)
        printf("  Hit rate:            %.6f%%\n",
               100.0 * (double)total_hits / (double)total_preds);
    printf("\n");

    /* First 10 hits table */
    if (n_fh > 0) {
        printf("  First 10 BPAND hits:\n");
        printf("    %10s  %14s  %14s  %14s  %8s  %8s  %6s  %4s  %6s\n",
               "k", "U(=N-k)", "U'(=N+k)", "U+U'",
               "L_U", "L_U'", "L_tot", "d", "d^128");
        printf("    %s  %s  %s  %s  %s  %s  %s  %s  %s\n",
               "----------", "--------------", "--------------", "--------------",
               "--------", "--------", "------", "----", "------");
        for (int i = 0; i < n_fh; i++) {
            size_t  kk    = first_hits[i].k;
            size_t  lo    = first_hits[i].lo;
            size_t  hi    = first_hits[i].hi;
            uint8_t dd    = first_hits[i].d;
            uint8_t dconj = (uint8_t)(dd + 128u);
            printf("    %10s  %14s  %14s  %14s  %+8" PRId64 "  %+8" PRIu64 "  %6d  %4u  %6u\n",
                   fmt(kk), fmt(lo), fmt(hi), fmt(lo + hi),
                   -(int64_t)kk, (uint64_t)kk, 0, (unsigned)dd, (unsigned)dconj);
        }
        printf("\n");
    }

    printf("  First hit:  k = %s, d = %u (%s)\n",
           fmt(first_k), (unsigned)first_d, qlabel(first_d));

    {
        double mean_k = (total_hits > 0) ? k_acc / (double)total_hits : 0.0;
        int64_t abs_spin = (spin_sum < 0) ? -spin_sum : spin_sum;
        printf("\n  ARM REACH:\n");
        printf("    Min k:    %s\n", fmt(min_k == (size_t)-1 ? 0 : min_k));
        printf("    Max k:    %s\n", fmt(max_k));
        printf("    Mean |k|: %.0f\n", mean_k);
        printf("    Sigma spin:   %+" PRId64 "\n", spin_sum);
        if (total_hits > 0)
            printf("    |Sigma|/N:    %.6f\n",
                   (double)abs_spin / (double)total_hits);
        if (total_hits > 1) {
            double anti_frac = (double)anti_spin_count / (double)(total_hits - 1);
            printf("    Anti-spin: %.1f%% (consecutive hits with opposite spin)\n",
                   100.0 * anti_frac);
        }
    }
    printf("\n");

    /* ── [E] Quadrant Distribution ───────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[E] Z256 Quadrant Distribution\n");
    printf("----------------------------------------------------------------------\n");

    uint64_t q_total = quad_hits[0] + quad_hits[1] + quad_hits[2] + quad_hits[3];
    const char *q_labels[4] = {
        "UP+ [  0.. 63]", "UP- [ 64..127]",
        "DN+ [128..191]", "DN- [192..255]"
    };
    double q_fracs[4] = {0.0, 0.0, 0.0, 0.0};

    for (int i = 0; i < 4; i++) {
        double frac = (q_total > 0) ? (double)quad_hits[i] / (double)q_total : 0.0;
        q_fracs[i] = frac;
        int bar_len = (int)(frac * 40.0);
        if (bar_len > 40) bar_len = 40;
        char bar[41];
        memset(bar, '#', (size_t)bar_len);
        bar[bar_len] = '\0';
        printf("    %s: %14s  (%5.1f%%) %s\n",
               q_labels[i], fmt((size_t)quad_hits[i]), 100.0 * frac, bar);
    }
    printf("\n");

    if (total_hits > 1) {
        double tr = (double)transitions / (double)(total_hits - 1);
        printf("  Quadrant transitions:  %s / %s = %.1f%%\n",
               fmt((size_t)transitions), fmt((size_t)(total_hits - 1)),
               100.0 * tr);
    }
    printf("\n");

    /* ── [F] HV Fill Distribution ────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[F] HV Fill Distribution (%s HVs used / %s predicted)\n",
           fmt(hvs_used), fmt(hvs));
    printf("----------------------------------------------------------------------\n");

    uint64_t fill_sum = 0, fill_min = UINT64_MAX, fill_max = 0;
    size_t   empty_hvs = 0;
    for (size_t i = 0; i < hvs_used; i++) {
        fill_sum += hv_fill[i];
        if (hv_fill[i] < fill_min) fill_min = hv_fill[i];
        if (hv_fill[i] > fill_max) fill_max = hv_fill[i];
        if (hv_fill[i] == 0) empty_hvs++;
    }
    free(hv_fill);

    double fill_mean = (hvs_used > 0) ? (double)fill_sum / (double)hvs_used : 0.0;
    printf("  Mean hits/HV:  %.1f\n", fill_mean);
    printf("  Min:           %" PRIu64 "\n", fill_min == UINT64_MAX ? 0ULL : fill_min);
    printf("  Max:           %" PRIu64 "\n", fill_max);
    printf("  Empty HVs:     %zu\n", empty_hvs);
    printf("  Total hits:    %s (check: %s)\n",
           fmt((size_t)fill_sum), fmt((size_t)total_hits));
    printf("\n");

    {
        size_t n_index_rings = (hvs_used + 251) / 252;
        printf("  STRUCTURE:\n");
        printf("    1 prediction Zring -> %s HVs -> %zu index Zring(s)\n\n",
               fmt(hvs_used), n_index_rings);
    }

    /* ── [G] Claims ──────────────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[G] Claims\n");
    printf("----------------------------------------------------------------------\n\n");

    typedef struct {
        const char *text;
        int         passed;
        char        evidence[128];
    } Claim;

    Claim claims[7];
    int   nc = 0;

    /* 1 — At least one pair found */
    {
        int ok = total_hits > 0;
        claims[nc].text   = "Goldbach pair exists for n";
        claims[nc].passed = ok;
        snprintf(claims[nc].evidence, 128, "%s pairs found, first at k = %s",
                 fmt((size_t)total_hits), fmt(first_k));
        nc++;
    }

    /* 2 — All 4 quadrants active */
    {
        int ok = quad_hits[0]>0 && quad_hits[1]>0 &&
                 quad_hits[2]>0 && quad_hits[3]>0;
        claims[nc].text   = "All 4 quadrants produce BPAND hits";
        claims[nc].passed = ok;
        snprintf(claims[nc].evidence, 128, "[%s, %s, %s, %s]",
                 fmt((size_t)quad_hits[0]), fmt((size_t)quad_hits[1]),
                 fmt((size_t)quad_hits[2]), fmt((size_t)quad_hits[3]));
        nc++;
    }

    /* 3 — Quadrant balance ±15% */
    {
        int ok = 1;
        for (int i = 0; i < 4; i++) {
            double dev = q_fracs[i] - 0.25;
            if (dev < 0.0) dev = -dev;
            if (dev >= 0.15) { ok = 0; break; }
        }
        claims[nc].text   = "Quadrants balanced (+-15%)";
        claims[nc].passed = ok;
        snprintf(claims[nc].evidence, 128, "[%.3f, %.3f, %.3f, %.3f]",
                 q_fracs[0], q_fracs[1], q_fracs[2], q_fracs[3]);
        nc++;
    }

    /* 4 — No empty HVs */
    {
        int ok = empty_hvs == 0;
        claims[nc].text   = "No empty HVs (uniform fill)";
        claims[nc].passed = ok;
        snprintf(claims[nc].evidence, 128, "%zu empty out of %zu used",
                 empty_hvs, hvs_used);
        nc++;
    }

    /* 5 — Spin near-cancellation */
    {
        int64_t abs_spin = (spin_sum < 0) ? -spin_sum : spin_sum;
        double  ratio    = (total_hits > 0)
                           ? (double)abs_spin / (double)total_hits : 1.0;
        int ok = ratio < 0.1;
        claims[nc].text   = "Spin near-cancellation (|S|/N < 0.1)";
        claims[nc].passed = ok;
        snprintf(claims[nc].evidence, 128,
                 "Sigma_spin = %+" PRId64 ", |S|/N = %.6f", spin_sum, ratio);
        nc++;
    }

    /* 6 — Hit rate > 0 */
    {
        double hr = (total_preds > 0)
                    ? (double)total_hits / (double)total_preds : 0.0;
        int ok = hr > 0.0;
        claims[nc].text   = "Hit rate > 0 (primes survive filter)";
        claims[nc].passed = ok;
        snprintf(claims[nc].evidence, 128, "%.6f%%", 100.0 * hr);
        nc++;
    }

    /* 7 — Quadrant transition rate > 50% */
    {
        double tr = (total_hits > 1)
                    ? (double)transitions / (double)(total_hits - 1) : 0.0;
        int ok = tr > 0.50;
        claims[nc].text   = "Quadrant transition rate > 50%";
        claims[nc].passed = ok;
        snprintf(claims[nc].evidence, 128, "%.1f%%", 100.0 * tr);
        nc++;
    }

    /* Print table */
    printf("  %-50s %-10s %s\n", "Claim", "Status", "Evidence");
    printf("  %-50s %-10s %s\n",
           "--------------------------------------------------",
           "----------",
           "----------------------------------------");
    int n_derived = 0, n_failed = 0;
    for (int i = 0; i < nc; i++) {
        const char *status = claims[i].passed ? "DERIVED" : "FAILED";
        if (claims[i].passed) n_derived++; else n_failed++;
        printf("  %-50s %-10s %s\n", claims[i].text, status, claims[i].evidence);
    }
    printf("\n  SCORECARD: %d DERIVED, %d FAILED\n", n_derived, n_failed);

    /* ── Summary ──────────────────────────────────────────────────────────── */
    printf("\n");
    printf("\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\n");
    printf("  n = %s  |  %s BPAND hits  |  first_k = %s  |  max_k = %s\n",
           fmt(n), fmt((size_t)total_hits), fmt(first_k), fmt(max_k));
    if (sieve_gb >= 1.0)
        printf("  Sieve: %.2f GB in %.2fs  |  Walk: %.2fs  |  Total: %.2fs\n",
               sieve_gb, t_sieve, t_walk, t_sieve + t_walk);
    else
        printf("  Sieve: %.1f MB in %.2fs  |  Walk: %.2fs  |  Total: %.2fs\n",
               sieve_mb, t_sieve, t_walk, t_sieve + t_walk);
    printf("\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90"
           "\xe2\x95\x90\n");

    return 0;
}
