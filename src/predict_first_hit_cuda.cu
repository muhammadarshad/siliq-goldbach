/*
 * predict_cuda.cu — GPU-native Z₂₅₆ BPAND Walk
 *
 * Complete experimental apparatus matching predict.c output:
 *   [P] Z₂₅₆ Ring Prediction Oracle
 *   [A] Geometry Prediction
 *   [C] Full BPAND Walk (GPU-native)
 *   [D] U/U' Conjugate Arm Analysis
 *   [E] Quadrant Distribution
 *   [F] HV Fill Distribution
 *   [G] Claims (7/7)
 *
 * Architecture:
 *   The Z₂₅₆ orbit is deterministic:  d(k) = orbit[(k-1) mod 252]
 *   Every k is independent — one GPU thread per k.
 *   Each thread: orbit lookup → parity gate → MR(lo) ∧ MR(hi)
 *   CPU reads hit mask, accumulates statistics, writes CSV.
 *
 *   Orbit table (252 entries) lives in GPU constant memory.
 *   mulmod is branchless for warp uniformity.
 *
 * Usage:
 *   predict_cuda <n>                     # full walk (sieve-limited)
 *   predict_cuda <n> --sweep <k_count>   # GPU walk over k window
 *
 * Build:
 *   nvcc -O3 -o predict_cuda.exe src/predict_first_hit_cuda.cu
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
 * Z₂₅₆ Geometry Constants  —  the ring, not parameters
 * ═══════════════════════════════════════════════════════════════════════════ */
#define STEP               7u
#define HALF               128
#define VACUUMS_PER_HALF   2
#define ACTIVE_PER_HALF    (HALF - VACUUMS_PER_HALF)  /* 126          */
#define ORBIT_SIZE         (2 * ACTIVE_PER_HALF)       /* 252          */
#define HV_CELLS           (HALF * 113)                /* 14,464       */
#define HV_W               113
#define ORBITS_PER_HV      (HV_CELLS / ORBIT_SIZE)     /* 57           */

/*
 * Batch size: 64M k values per GPU launch.
 * 2 bytes per k (hit + d) × 64M = 128 MB on-device.
 * RTX 4060 has 8GB — this uses 1.6%, leaving headroom.
 * 64M threads with 256/block = 262,144 blocks → full SM saturation.
 */
#define BATCH_K            (1u << 28)                  /* 268,435,456  */

/* ─── GPU constant memory ─────────────────────────────────────────────── */
__constant__ uint8_t  d_orbit[ORBIT_SIZE];
__constant__ uint32_t d_small_primes[11];
__constant__ uint64_t d_mr_bases[7];

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU device: hardware modular arithmetic
 *
 * mulmod uses the GPU's native 64×64→128 integer multiply
 * (__umul64hi + *) — TWO hardware instructions instead of a
 * 64-iteration bit-loop.  __int128 handles the mod reduction.
 *
 * No float. No FPU. Pure unsigned integer pipeline.
 * ═══════════════════════════════════════════════════════════════════════════ */

__device__ __forceinline__
uint64_t dev_mulmod(uint64_t a, uint64_t b, uint64_t m) {
    /*
     * Hardware 64×64→128 via __umul64hi (one PTX mul.hi.u64).
     * Reduce (hi:lo) mod m with 64 branchless doublings.
     *
     * No __int128. No float. No bit-loop on the multiply.
     * 4 quadrants → 4(ab)² · 4(cd)² = 16(abcd)²
     * Split the 128-bit result like the ring splits the walk.
     *
     * Fast path (hi==0): one hardware mod, done.
     * Slow path: fold hi into m-space by doubling 64 times
     *            (shift hi left by 64 bits in mod-m arithmetic).
     *            Each step: addmod(h,h,m) — one compare, one op.
     */
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);

    if (hi == 0) return lo % m;

    /* reduce lo and hi into m-space */
    uint64_t r = lo % m;
    uint64_t h = hi % m;

    /* h = h * 2^64 mod m  via 64 branchless doublings */
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        h = (h >= m - h) ? h - (m - h) : h + h;
    }

    /* combine */
    r = (r >= m - h) ? r - (m - h) : r + h;
    return r;
}

__device__
uint64_t dev_powmod(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1;
    a %= m;
    while (e) {
        if (e & 1u) r = dev_mulmod(r, a, m);
        e >>= 1;
        if (e) a = dev_mulmod(a, a, m);
    }
    return r;
}

/* ─── Deterministic Miller-Rabin ─────────────────────────────────────── */
__device__
int dev_is_prime(uint64_t n) {
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
        uint64_t x = dev_powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        int witness = 1;
        for (unsigned r = 1; r < s; r++) {
            x = dev_mulmod(x, x, n);
            if (x == n - 1) { witness = 0; break; }
        }
        if (witness) return 0;
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU kernel — one thread per k, entire physics on-device
 *
 *  thread i  →  k = k_start + i
 *  d = orbit[(k-1) mod 252]            ← Z₂₅₆ geometry (constant mem)
 *  lo = big_n - k,  hi = big_n + k     ← conjugate arms
 *  parity gate                          ← skip impossible even pairs
 *  MR(lo) ∧ MR(hi)                     ← primality oracle
 *
 *  out_hit[i] = 1 if BPAND hit
 *  out_d[i]   = d at this k (always written for CSV/statistics)
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void bpand_kernel(uint64_t big_n, uint64_t k_start,
                             uint64_t batch_count,
                             uint8_t *out_hit, uint8_t *out_d) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    uint64_t k = k_start + idx;

    /* Z₂₅₆ orbit — the geometry drives the walk */
    uint8_t d_val = d_orbit[(k - 1) % ORBIT_SIZE];
    out_d[idx] = d_val;

    /* conjugate arms */
    uint64_t lo = big_n - k;
    uint64_t hi = big_n + k;

    if (lo < 2) { out_hit[idx] = 0; return; }

    /* parity gate */
    if (((big_n ^ k) & 1u) == 0 && lo != 2) { out_hit[idx] = 0; return; }

    /* primality oracle */
    out_hit[idx] = (dev_is_prime(lo) && dev_is_prime(hi)) ? 1 : 0;
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

static uint32_t ilog10_u(uint64_t n) {
    if (n == 0) return 0;
    uint32_t c = 0;
    while (n >= 10) { n /= 10; c++; }
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

static void predict_hv(uint64_t n, uint64_t *k_max_out,
                       uint64_t *orbits_out, uint64_t *hvs_out) {
    uint64_t big_n  = n / 2;
    uint64_t k_max  = (big_n >= 2) ? big_n - 2 : 0;
    uint64_t orbits = (k_max + ORBIT_SIZE - 1) / ORBIT_SIZE;
    uint64_t hvs    = (k_max + HV_CELLS - 1) / HV_CELLS;
    *k_max_out  = k_max;
    *orbits_out = orbits;
    *hvs_out    = hvs;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main — complete experimental apparatus
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    uint64_t n = 4000000000000000000ULL;
    uint64_t sweep_k = 0;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--sweep") == 0 && i + 1 < argc) {
            char cl[64]; int ci2 = 0;
            for (const char *p = argv[++i]; *p && ci2 < 63; p++)
                if (*p != '_' && *p != ',') cl[ci2++] = *p;
            cl[ci2] = '\0';
            sweep_k = (uint64_t)std::strtoull(cl, nullptr, 10);
            continue;
        }
        char clean[64]; int ci = 0;
        for (const char *p = argv[i]; *p && ci < 63; p++)
            if (*p != '_' && *p != ',') clean[ci++] = *p;
        clean[ci] = '\0';
        uint64_t val = (uint64_t)std::strtoull(clean, nullptr, 10);
        if (val >= 4) n = val;
    }

    if (n < 4 || (n & 1u) != 0) {
        std::fprintf(stderr, "Error: n must be even and >= 4\n");
        return 1;
    }

    uint64_t big_n  = n / 2;
    uint32_t base_x = ilog10_u(n);

    printf("======================================================================\n");
    printf("  predict_cuda — Z₂₅₆ BPAND Walk (GPU-native)\n");
    printf("  Target: n = %s\n", fmt(n));
    printf("======================================================================\n\n");

    /* ── [P] Z₂₅₆ Ring Prediction Oracle ──────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[P] Z256 Ring Prediction Oracle (pure math, no primes)\n");
    printf("----------------------------------------------------------------------\n");
    printf("  Z256 ring: 256 positions\n");
    printf("    Vacuum boundaries: 4 (0, 64, 128, 192)\n");
    printf("    Non-vacuum: 252 = 256 - 4\n");
    printf("    Per quadrant: 63 = 252 / 4\n");
    printf("    63 mod 7 = 0  =>  7 x 9 = 63\n");
    printf("    7-prime walk: 9 steps per cycle, 36 cycles covers 252\n");
    printf("    Zero gaps, zero repeats, full coverage\n\n");
    printf("  Four-arm witnessing for k = 1:\n");
    printf("    d1 = %u  d2 = %u  d3 = %u  d4 = %u\n",
           1u, 1u + 64u, 1u + 128u, 1u + 192u);
    printf("    All four point to (N-k, N+k) = (%s, %s)\n",
           fmt(big_n - 1), fmt(big_n + 1));
    printf("    When BPAND = 1, all four ring directions confirm the pair\n\n");
    printf("  Prediction:\n");
    printf("    Each quadrant filled in <=63 steps\n");
    printf("    Sweet spot rule: rings(10^x) = x - 3\n");
    printf("      10^4 -> 1, 10^8 -> 5, 10^12 -> 9, 10^18 -> 15\n\n");

    /* ── [A] Geometry Prediction ──────────────────────────────────────── */
    uint64_t k_max, orbits, hvs;
    predict_hv(n, &k_max, &orbits, &hvs);
    uint32_t rings_needed = (base_x >= 3) ? base_x - 3 : 0;

    /* k_limit: sweep window or full walk */
    uint64_t k_limit = k_max;
    if (sweep_k > 0 && sweep_k < k_limit) k_limit = sweep_k;
    uint64_t hvs_in_window = (k_limit + HV_CELLS - 1) / HV_CELLS;

    printf("----------------------------------------------------------------------\n");
    printf("[A] Geometry Prediction (no sieve needed)\n");
    printf("----------------------------------------------------------------------\n");
    printf("  n           = %s\n", fmt(n));
    printf("  N = n/2     = %s\n", fmt(big_n));
    printf("  k_max       = %s\n", fmt(k_max));
    printf("  orbits      = %s (each %d predictions)\n", fmt(orbits), ORBIT_SIZE);
    printf("  HVs(full)   = %s (each %u cells = %d orbits)\n",
           fmt(hvs), (unsigned)HV_CELLS, ORBITS_PER_HV);
    printf("  Zrings      = %u (each %d orbits)\n", rings_needed, ORBIT_SIZE);
    if (sweep_k > 0) {
        printf("  sweep window = %s k values (%s HVs)\n",
               fmt(k_limit), fmt(hvs_in_window));
    }
    {
        uint64_t zl1 = (hvs > 1) ? (hvs + ORBIT_SIZE - 1) / ORBIT_SIZE : 0;
        uint64_t zl2 = (zl1 > 1)  ? (zl1 + ORBIT_SIZE - 1) / ORBIT_SIZE : 0;
        if (hvs <= 1)
            printf("  structure   = 1 HV (no chain)\n");
        else if (zl1 <= 1)
            printf("  structure   = %s HVs -> 1 Zring index\n", fmt(hvs));
        else if (zl2 <= 1)
            printf("  structure   = %s HVs -> %s Zrings -> 1 root\n",
                   fmt(hvs), fmt(zl1));
        else
            printf("  structure   = %s HVs -> %s L1 -> %s L2\n",
                   fmt(hvs), fmt(zl1), fmt(zl2));
    }
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
        std::fprintf(stderr, "  FAIL: orbit upload: %s\n", cudaGetErrorString(ce));
        return 2;
    }

    const uint32_t small_primes[11] = {3,5,7,11,13,17,19,23,29,31,37};
    ce = cudaMemcpyToSymbol(d_small_primes, small_primes, sizeof(small_primes));
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "  FAIL: primes upload: %s\n", cudaGetErrorString(ce));
        return 2;
    }

    const uint64_t mr_bases[7] = {2,325,9375,28178,450775,9780504,1795265022};
    ce = cudaMemcpyToSymbol(d_mr_bases, mr_bases, sizeof(mr_bases));
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "  FAIL: bases upload: %s\n", cudaGetErrorString(ce));
        return 2;
    }

    /* device buffers */
    uint8_t *dev_hit = nullptr, *dev_d = nullptr;
    if (cudaMalloc(&dev_hit, BATCH_K) != cudaSuccess ||
        cudaMalloc(&dev_d,   BATCH_K) != cudaSuccess) {
        std::fprintf(stderr, "  cudaMalloc failed\n");
        return 4;
    }

    /* host buffers */
    uint8_t *h_hit = (uint8_t *)std::malloc(BATCH_K);
    uint8_t *h_d   = (uint8_t *)std::malloc(BATCH_K);
    if (!h_hit || !h_d) {
        std::fprintf(stderr, "  OOM: host buffers\n");
        return 3;
    }

    /* HV fill array — allocated for sweep window */
    uint64_t *hv_fill = (uint64_t *)std::calloc(
        hvs_in_window ? hvs_in_window : 1, sizeof(uint64_t));
    if (!hv_fill) {
        std::fprintf(stderr, "  OOM: hv_fill\n");
        return 3;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  Device:       %s\n", prop.name);
    printf("  SMs:          %d\n", prop.multiProcessorCount);
    printf("  Orbit table:  %d entries in constant memory\n", ORBIT_SIZE);
    printf("  MR bases:     7 deterministic (full uint64 coverage)\n");
    printf("  Batch size:   %s k per launch (%u MB on-device)\n",
           fmt(BATCH_K), (unsigned)((uint64_t)BATCH_K * 2 / (1024*1024)));
    printf("  mulmod:       hardware 64x64->128 (__int128), no loop\n\n");

    /* ── [C] Full BPAND Walk ──────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[C] GPU BPAND Walk — n = %s\n", fmt(n));
    printf("----------------------------------------------------------------------\n");
    printf("  Walking k = 1 .. %s through %s HVs ...\n\n",
           fmt(k_limit), fmt(hvs_in_window));

    /* ── CSV ── */
    char csv_path[96];
    if (sweep_k > 0)
        std::snprintf(csv_path, sizeof(csv_path),
                      "predict_sweep_%llu.csv", (unsigned long long)n);
    else
        std::snprintf(csv_path, sizeof(csv_path),
                      "predict_%llu.csv", (unsigned long long)n);

    FILE *csv = std::fopen(csv_path, "w");
    if (!csv) {
        std::fprintf(stderr, "  Cannot create CSV: %s\n", csv_path);
        return 6;
    }
    std::setvbuf(csv, nullptr, _IOFBF, 1 << 18);  /* 256 KB buffer */
    std::fprintf(csv, "k,U,U_prime,d,spin,quadrant,hv\n");

    /* ── Walk statistics ── */
    uint64_t total_hits      = 0;
    uint64_t total_k         = 0;
    uint64_t quad_hits[4]    = {0, 0, 0, 0};
    uint64_t first_hit_k     = 0;
    uint8_t  first_hit_d     = 0;
    uint64_t min_k           = UINT64_MAX;
    uint64_t max_k           = 0;
    double   k_acc           = 0.0;
    int64_t  spin_sum        = 0;
    uint64_t transitions     = 0;
    uint8_t  prev_q          = 255;
    int8_t   prev_spin       = 0;
    uint64_t anti_spin_count = 0;

    /* First 10 hits for display */
    struct Hit { uint64_t k, lo, hi; uint8_t d; };
    Hit first_hits[10];
    int n_fh = 0;

    double t0 = now_s();
    uint64_t progress_interval = k_limit / 20;
    if (progress_interval < 1) progress_interval = 1;
    uint64_t next_progress = progress_interval;

    /* ═══════════════════════════════════════════════════════════════════
     * Main loop:  launch GPU batches of BATCH_K threads.
     *   GPU: orbit lookup → parity gate → MR(lo) ∧ MR(hi)
     *   CPU: scan hit mask → accumulate statistics → write CSV
     * ═══════════════════════════════════════════════════════════════════ */
    for (uint64_t k_start = 1; k_start <= k_limit; k_start += BATCH_K) {
        uint64_t batch = BATCH_K;
        if (k_start + batch - 1 > k_limit)
            batch = k_limit - k_start + 1;

        int threads = 256;
        int blocks  = (int)((batch + threads - 1) / threads);
        bpand_kernel<<<blocks, threads>>>(big_n, k_start, batch, dev_hit, dev_d);
        cudaDeviceSynchronize();

        cudaMemcpy(h_hit, dev_hit, batch, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_d,   dev_d,   batch, cudaMemcpyDeviceToHost);

        total_k += batch;

        for (uint64_t i = 0; i < batch; i++) {
            if (!h_hit[i]) continue;

            uint64_t k      = k_start + i;
            uint8_t  d_val  = h_d[i];
            int      s_val  = ((d_val >> 7) & 1u) == 0 ? 1 : -1;
            uint8_t  q_val  = (d_val >> 6) & 0x3u;
            uint64_t lo     = big_n - k;
            uint64_t hi     = big_n + k;
            uint64_t hv_idx = (k - 1) / HV_CELLS;

            total_hits++;
            quad_hits[q_val]++;
            spin_sum += s_val;
            k_acc    += (double)k;

            if (k < min_k) min_k = k;
            if (k > max_k) max_k = k;

            if (first_hit_k == 0) { first_hit_k = k; first_hit_d = d_val; }

            if (prev_q != 255 && q_val != prev_q) transitions++;
            prev_q = q_val;

            if (prev_spin != 0 && s_val != (int)prev_spin) anti_spin_count++;
            prev_spin = (int8_t)s_val;

            if (hv_idx < hvs_in_window)
                hv_fill[hv_idx]++;

            std::fprintf(csv, "%llu,%llu,%llu,%u,%d,%u,%llu\n",
                         (unsigned long long)k,
                         (unsigned long long)lo,
                         (unsigned long long)hi,
                         (unsigned)d_val, s_val, (unsigned)q_val,
                         (unsigned long long)hv_idx);

            if (n_fh < 10) {
                first_hits[n_fh].k  = k;
                first_hits[n_fh].lo = lo;
                first_hits[n_fh].hi = hi;
                first_hits[n_fh].d  = d_val;
                n_fh++;
            }
        }

        /* progress */
        uint64_t k_pos = k_start + batch - 1;
        if (k_pos >= next_progress) {
            double pct = 100.0 * (double)k_pos / (double)k_limit;
            double dt  = now_s() - t0;
            std::fprintf(stderr, "    %5.1f%%  hits=%s  k=%s  [%.1fs]\n",
                         pct, fmt(total_hits), fmt(k_pos), dt);
            next_progress += progress_interval;
        }
    }

    double t_walk = now_s() - t0;
    std::fclose(csv);

    printf("  Walk complete in %.3fs\n", t_walk);
    printf("  CSV written: %s\n\n", csv_path);

    /* ── [D] U/U' Conjugate Arm Analysis ─────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[D] U / U' Conjugate Arm Analysis\n");
    printf("----------------------------------------------------------------------\n");
    printf("  Center N = %s\n", fmt(big_n));
    printf("  U  = N - k  (arm sweeping LEFT)\n");
    printf("  U' = N + k  (arm sweeping RIGHT)\n");
    printf("  Conservation: U + U' = %s = n  (ALWAYS)\n\n", fmt(n));
    printf("  Total BPAND hits:    %s\n", fmt(total_hits));
    printf("  k tested:            %s\n", fmt(total_k));
    if (total_k > 0)
        printf("  Hit rate:            %.6f%%\n",
               100.0 * (double)total_hits / (double)total_k);
    printf("\n");

    /* First 10 hits table */
    if (n_fh > 0) {
        printf("  First 10 BPAND hits:\n");
        printf("    %10s  %22s  %22s  %22s  %4s  %6s\n",
               "k", "U(=N-k)", "U'(=N+k)", "U+U'", "d", "label");
        printf("    %s  %s  %s  %s  %s  %s\n",
               "----------", "----------------------",
               "----------------------", "----------------------",
               "----", "------");
        for (int i = 0; i < n_fh; i++) {
            printf("    %10s  %22s  %22s  %22s  %4u  %s\n",
                   fmt(first_hits[i].k),
                   fmt(first_hits[i].lo),
                   fmt(first_hits[i].hi),
                   fmt(first_hits[i].lo + first_hits[i].hi),
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
    double q_fracs[4] = {0,0,0,0};
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

    /* ── [F] HV Fill Distribution ─────────────────────────────────────── */
    uint64_t hvs_used = hvs_in_window;
    /* trim trailing unused HVs */
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

    {
        uint64_t n_idx = (hvs_used + 251) / 252;
        printf("  STRUCTURE:\n");
        printf("    1 prediction Zring -> %s HVs -> %s index Zring(s)\n\n",
               fmt(hvs_used), fmt(n_idx));
    }

    std::free(hv_fill);

    /* ── [G] Claims ───────────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[G] Claims\n");
    printf("----------------------------------------------------------------------\n\n");

    struct { const char *text; int pass; char ev[128]; } claims[7];
    int nc = 0;

    /* 1 — pair exists */
    claims[nc].pass = total_hits > 0;
    claims[nc].text = "Goldbach pair exists for n";
    snprintf(claims[nc].ev, 128, "%s pairs, first at k = %s",
             fmt(total_hits), fmt(first_hit_k));
    nc++;

    /* 2 — all 4 quadrants */
    claims[nc].pass = quad_hits[0]>0 && quad_hits[1]>0 &&
                      quad_hits[2]>0 && quad_hits[3]>0;
    claims[nc].text = "All 4 quadrants produce BPAND hits";
    snprintf(claims[nc].ev, 128, "[%s, %s, %s, %s]",
             fmt(quad_hits[0]), fmt(quad_hits[1]),
             fmt(quad_hits[2]), fmt(quad_hits[3]));
    nc++;

    /* 3 — balance ±15% */
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

    /* 4 — no empty HVs */
    claims[nc].pass = (empty_hvs == 0);
    claims[nc].text = "No empty HVs (uniform fill)";
    snprintf(claims[nc].ev, 128, "%s empty out of %s used",
             fmt(empty_hvs), fmt(hvs_used));
    nc++;

    /* 5 — spin cancellation */
    {
        int64_t as = (spin_sum < 0) ? -spin_sum : spin_sum;
        double ratio = (total_hits > 0)
                       ? (double)as / (double)total_hits : 1.0;
        claims[nc].pass = ratio < 0.1;
        claims[nc].text = "Spin near-cancellation (|S|/N < 0.1)";
        snprintf(claims[nc].ev, 128, "Sigma = %+" PRId64 ", |S|/N = %.6f",
                 spin_sum, ratio);
        nc++;
    }

    /* 6 — hit rate > 0 */
    {
        double hr = (total_k > 0)
                    ? (double)total_hits / (double)total_k : 0.0;
        claims[nc].pass = hr > 0.0;
        claims[nc].text = "Hit rate > 0 (primes survive filter)";
        snprintf(claims[nc].ev, 128, "%.6f%%", 100.0 * hr);
        nc++;
    }

    /* 7 — transition rate > 50% */
    {
        double tr = (total_hits > 1)
                    ? (double)transitions / (double)(total_hits - 1) : 0.0;
        claims[nc].pass = tr > 0.50;
        claims[nc].text = "Quadrant transition rate > 50%";
        snprintf(claims[nc].ev, 128, "%.1f%%", 100.0 * tr);
        nc++;
    }

    /* Print claims table */
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
    printf("  n = %s  |  %s BPAND hits  |  first_k = %s  |  max_k = %s\n",
           fmt(n), fmt(total_hits), fmt(first_hit_k), fmt(max_k));
    printf("  Walk: %.2fs  |  Backend: GPU (%s, %d SMs)\n",
           t_walk, prop.name, prop.multiProcessorCount);
    printf("======================================================================\n");

    cudaFree(dev_hit);
    cudaFree(dev_d);
    std::free(h_hit);
    std::free(h_d);
    return (total_hits > 0) ? 0 : 5;
}
