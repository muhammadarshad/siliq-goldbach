/*
 * goldbach_enumerate.cu — Minimal Goldbach partition enumeration on GPU
 *
 * Replicates and extends Oliveira e Silva (2014):
 *   For each even n in [n_start, n_start + 2*count),
 *   find the minimal Goldbach partition p(n) — the smallest prime p
 *   such that n - p is also prime.
 *
 * Architecture:
 *   One GPU thread per even n. Each thread scans p = 3, 5, 7, 11, ...
 *   and tests whether (n - p) is prime via deterministic 7-base MR.
 *   Since the minimal p is typically tiny (< 10,000 for all n ≤ 4×10^18),
 *   threads converge fast with minimal warp divergence.
 *
 * Deterministic MR is exact below 3.317×10^24. At n ≤ 4×10^18 (uint64),
 *   zero false positives. No sieve needed.
 *
 * Output:
 *   - Largest minimal p found in the range (the "record")
 *   - Count of verified n values
 *   - Any n where no partition was found (would disprove Goldbach — none expected)
 *
 * Build:
 *   nvcc -O3 -arch=sm_89 -o goldbach_enumerate.exe src/goldbach_enumerate.cu
 *
 * No float. No signed arithmetic. No sieve. Pure MR on GPU.
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
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MAX_SMALL_PRIMES  9600          /* primes up to ~100,000               */
#define BATCH_N           (1u << 22)    /* 4M n values per kernel launch       */
#define P_LIMIT           100000u       /* trial primes up to this             */

/* GPU constant memory: small primes for trial + MR bases */
__constant__ uint32_t d_primes[MAX_SMALL_PRIMES];
__constant__ uint32_t d_num_primes;
__constant__ uint64_t d_mr_bases[7];

/* ═══════════════════════════════════════════════════════════════════════════
 * uint64 modular arithmetic — same as predict_cuda256
 * ═══════════════════════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Deterministic Miller-Rabin (7 bases, exact below 3.317×10^24)
 * ═══════════════════════════════════════════════════════════════════════════ */
__device__
int dev_is_prime_u64(uint64_t n) {
    if (n < 2) return 0;
    if (n == 2 || n == 3) return 1;
    if ((n & 1u) == 0) return 0;
    if (n % 3 == 0) return 0;

    /* Trial division by small primes from constant memory */
    uint32_t np = d_num_primes;
    for (uint32_t i = 0; i < np && i < 30; i++) {
        uint32_t p = d_primes[i];
        if ((uint64_t)p * p > n) return 1;
        if (n == (uint64_t)p) return 1;
        if (n % p == 0) return 0;
    }

    /* Factor out 2s: n-1 = d × 2^s */
    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1u) == 0) { d >>= 1; s++; }

    /* 7 deterministic bases */
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Minimal Goldbach partition kernel
 *
 * Each thread handles one even n = n_start + 2*idx.
 * Scans p = 3, 5, 7, 11, 13, ... (odd primes) and checks: is (n-p) prime?
 * Stops at first hit → that's the minimal partition.
 *
 * Output per n: the minimal prime p (stored in out_p[idx]).
 *               0 means no partition found (should NEVER happen).
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__
void minimal_partition_kernel(
    uint64_t n_start,       /* first even n (must be even, ≥ 4) */
    uint32_t count,         /* number of even values to check   */
    uint32_t *out_p,        /* output: minimal prime p per n    */
    uint32_t p_search_max   /* max p to try before giving up    */
) {
    uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t n = n_start + 2ULL * idx;
    out_p[idx] = 0;

    /* Special case: n = 4 → 2 + 2 */
    if (n == 4) { out_p[idx] = 2; return; }

    /* Scan odd primes from the list */
    uint32_t np = d_num_primes;
    for (uint32_t i = 0; i < np; i++) {
        uint32_t p = d_primes[i];
        if (p >= n) break;
        if (p > p_search_max) break;

        uint64_t q = n - (uint64_t)p;
        if (q < 2) break;

        /* q must be odd to be prime (q > 2) */
        if ((q & 1u) == 0) continue;

        if (dev_is_prime_u64(q)) {
            out_p[idx] = p;
            return;
        }
    }

    /* Exhausted prime list — continue with trial for larger p */
    /* This path is essentially never taken for n ≤ 4×10^18   */
    uint32_t p_cand = d_primes[np > 0 ? np - 1 : 0] + 2;
    while (p_cand <= p_search_max) {
        if (dev_is_prime_u64((uint64_t)p_cand)) {
            uint64_t q = n - (uint64_t)p_cand;
            if (q < 2) break;
            if ((q & 1u) == 0) { p_cand += 2; continue; }
            if (dev_is_prime_u64(q)) {
                out_p[idx] = p_cand;
                return;
            }
        }
        p_cand += 2;
    }
    /* out_p[idx] remains 0: NO PARTITION FOUND — would disprove Goldbach */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Host: number formatter with commas
 * ═══════════════════════════════════════════════════════════════════════════ */
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
        if (i > 0 && (len - i) % 3 == 0) buf[out++] = ',';
        buf[out++] = tmp[i];
    }
    buf[out] = '\0';
    return buf;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Host: CPU sieve to generate small primes
 * ═══════════════════════════════════════════════════════════════════════════ */
static uint32_t *host_sieve_primes(uint32_t limit, uint32_t *count_out) {
    /* Sieve of Eratosthenes for odd primes up to limit */
    uint32_t sz = (limit / 2) + 1;
    uint8_t *sieve = (uint8_t *)calloc(sz, 1);
    if (!sieve) { fprintf(stderr, "OOM: sieve\n"); exit(1); }

    for (uint32_t i = 3; (uint64_t)i * i <= limit; i += 2) {
        if (!sieve[i / 2]) {
            for (uint32_t j = i * i; j <= limit; j += 2 * i)
                sieve[j / 2] = 1;
        }
    }

    /* Count */
    uint32_t cnt = 0;
    for (uint32_t p = 3; p <= limit; p += 2)
        if (!sieve[p / 2]) cnt++;

    uint32_t *primes = (uint32_t *)malloc(cnt * sizeof(uint32_t));
    if (!primes) { fprintf(stderr, "OOM: primes\n"); exit(1); }

    uint32_t w = 0;
    for (uint32_t p = 3; p <= limit; p += 2)
        if (!sieve[p / 2]) primes[w++] = p;

    free(sieve);
    *count_out = w;
    return primes;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    uint64_t n_start  = 4;                 /* first even n to check */
    uint64_t n_count  = 1000000000ULL;     /* number of even values (1B default) */
    int      write_csv = 1;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--start") == 0 && i + 1 < argc) {
            char cl[64]; int ci = 0;
            for (const char *p = argv[++i]; *p && ci < 63; p++)
                if (*p != '_' && *p != ',') cl[ci++] = *p;
            cl[ci] = '\0';
            n_start = (uint64_t)std::strtoull(cl, nullptr, 10);
            continue;
        }
        if (std::strcmp(argv[i], "--count") == 0 && i + 1 < argc) {
            char cl[64]; int ci = 0;
            for (const char *p = argv[++i]; *p && ci < 63; p++)
                if (*p != '_' && *p != ',') cl[ci++] = *p;
            cl[ci] = '\0';
            n_count = (uint64_t)std::strtoull(cl, nullptr, 10);
            continue;
        }
        if (std::strcmp(argv[i], "--no-csv") == 0) {
            write_csv = 0;
            continue;
        }
        /* Positional: n_start */
        char cl[64]; int ci = 0;
        for (const char *p = argv[i]; *p && ci < 63; p++)
            if (*p != '_' && *p != ',') cl[ci++] = *p;
        cl[ci] = '\0';
        n_start = (uint64_t)std::strtoull(cl, nullptr, 10);
    }

    /* Force n_start even and ≥ 4 */
    if (n_start < 4) n_start = 4;
    if (n_start & 1u) n_start++;

    uint64_t n_end = n_start + 2 * (n_count - 1);  /* last even n */

    /* ── Banner ──────────────────────────────────────────────────────── */
    printf("======================================================================\n");
    printf("  goldbach_enumerate — Minimal Goldbach Partition (GPU)\n");
    printf("  Range: n = %s to %s (%s even values)\n",
           fmt(n_start), fmt(n_end), fmt(n_count));
    printf("  Method: deterministic 7-base Miller-Rabin (exact below 3.3×10^24)\n");
    printf("======================================================================\n\n");

    /* ── Generate small primes on CPU ────────────────────────────────── */
    uint32_t num_primes = 0;
    uint32_t *h_primes = host_sieve_primes(P_LIMIT, &num_primes);
    printf("[SETUP] Generated %s odd primes up to %s\n", fmt(num_primes), fmt(P_LIMIT));

    if (num_primes > MAX_SMALL_PRIMES) {
        fprintf(stderr, "  Too many primes for constant memory (%u > %d)\n",
                num_primes, MAX_SMALL_PRIMES);
        free(h_primes);
        return 1;
    }

    /* ── GPU setup ──────────────────────────────────────────────────── */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);

    printf("[SETUP] Device: %s (%d SMs, %s MB VRAM)\n",
           prop.name, prop.multiProcessorCount,
           fmt((uint64_t)(total_vram >> 20)));

    /* Upload primes + MR bases to constant memory */
    cudaMemcpyToSymbol(d_primes, h_primes,
                       num_primes * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_num_primes, &num_primes, sizeof(uint32_t));
    free(h_primes);

    const uint64_t mr_bases[7] = {
        2, 325, 9375, 28178, 450775, 9780504, 1795265022
    };
    cudaMemcpyToSymbol(d_mr_bases, mr_bases, sizeof(mr_bases));

    /* ── Allocate output buffer ──────────────────────────────────────── */
    uint32_t batch_n = BATCH_N;
    if (batch_n > n_count) batch_n = (uint32_t)n_count;

    uint32_t *dev_p = nullptr;
    cudaMalloc(&dev_p, (size_t)batch_n * sizeof(uint32_t));

    uint32_t *h_p = nullptr;
    cudaMallocHost(&h_p, (size_t)batch_n * sizeof(uint32_t));

    printf("[SETUP] Batch size: %s n values per kernel launch\n", fmt(batch_n));
    printf("[SETUP] Prime search limit: %s per n\n", fmt(P_LIMIT));
    printf("[SETUP] CSV output: %s\n\n", write_csv ? "ON" : "OFF");

    /* ── CSV for records ─────────────────────────────────────────────── */
    FILE *csv = nullptr;
    if (write_csv) {
        csv = fopen("goldbach_enumerate.csv", "w");
        if (csv) {
            setvbuf(csv, nullptr, _IOFBF, 1 << 20);
            fprintf(csv, "n,min_p,q\n");
        }
    }

    /* ── Enumeration loop ────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[ENUMERATE] Walking n = %s to %s\n", fmt(n_start), fmt(n_end));
    printf("----------------------------------------------------------------------\n\n");

    double t0 = now_s();

    uint64_t total_verified = 0;
    uint64_t total_failed   = 0;
    uint32_t global_max_p   = 0;
    uint64_t global_max_p_n = 0;        /* n value that needed the largest p */

    /* p distribution: count how often each prime is the minimal p */
    /* Track first 50 primes (3..229) individually */
    uint64_t p_dist[50];
    memset(p_dist, 0, sizeof(p_dist));

    /* Histogram of p buckets for display */
    uint64_t p_bucket[6] = {0};  /* [0]:p=3, [1]:5-10, [2]:11-100, [3]:101-1K, [4]:1K-10K, [5]:>10K */

    uint64_t progress_iv = n_count / 20;
    if (progress_iv < 1) progress_iv = 1;
    uint64_t next_progress = progress_iv;

    for (uint64_t offset = 0; offset < n_count; offset += batch_n) {
        uint32_t this_batch = batch_n;
        if (offset + this_batch > n_count)
            this_batch = (uint32_t)(n_count - offset);

        uint64_t batch_start = n_start + 2 * offset;

        /* Launch kernel */
        int threads = 256;
        int blocks  = (int)((this_batch + threads - 1) / threads);
        minimal_partition_kernel<<<blocks, threads>>>(
            batch_start, this_batch, dev_p, P_LIMIT);

        /* Copy results back */
        cudaDeviceSynchronize();
        cudaMemcpy(h_p, dev_p, (size_t)this_batch * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        /* Process batch on CPU */
        for (uint32_t i = 0; i < this_batch; i++) {
            uint32_t p = h_p[i];
            uint64_t n_val = batch_start + 2ULL * i;

            if (p == 0) {
                total_failed++;
                fprintf(stderr, "  *** GOLDBACH FAILURE at n = %llu ***\n",
                        (unsigned long long)n_val);
            } else {
                total_verified++;

                if (p > global_max_p) {
                    global_max_p = p;
                    global_max_p_n = n_val;
                }

                /* Distribution tracking */
                if (p == 3)         p_bucket[0]++;
                else if (p <= 10)   p_bucket[1]++;
                else if (p <= 100)  p_bucket[2]++;
                else if (p <= 1000) p_bucket[3]++;
                else if (p <= 10000)p_bucket[4]++;
                else                p_bucket[5]++;

                /* CSV: only write records where p ≥ 1000 (interesting cases) */
                if (csv && p >= 1000) {
                    uint64_t q = n_val - (uint64_t)p;
                    fprintf(csv, "%llu,%u,%llu\n",
                            (unsigned long long)n_val, p,
                            (unsigned long long)q);
                }
            }
        }

        uint64_t progress_pos = offset + this_batch;
        if (progress_pos >= next_progress || progress_pos >= n_count) {
            double pct = 100.0 * (double)progress_pos / (double)n_count;
            double dt = now_s() - t0;
            double rate = (double)progress_pos / dt;
            uint64_t n_here = batch_start + 2ULL * (this_batch - 1);
            fprintf(stderr, "    %5.1f%%  verified=%s  max_p=%u  n=%s  [%.1fs, %.0f n/s]\n",
                    pct, fmt(total_verified), global_max_p,
                    fmt(n_here), dt, rate);
            next_progress += progress_iv;
        }
    }

    double t_total = now_s() - t0;

    if (csv) fclose(csv);

    /* ── Results ─────────────────────────────────────────────────────── */
    printf("\n");
    printf("----------------------------------------------------------------------\n");
    printf("[RESULT] Goldbach Enumeration Complete\n");
    printf("----------------------------------------------------------------------\n\n");
    printf("  Range:          n = %s to %s\n", fmt(n_start), fmt(n_end));
    printf("  Verified:       %s even values\n", fmt(total_verified));
    printf("  Failed:         %s\n", fmt(total_failed));
    printf("  Time:           %.2fs\n", t_total);
    printf("  Rate:           %.0f n/s\n", (double)total_verified / t_total);
    printf("\n");

    printf("  Largest minimal p: %u  (at n = %s)\n", global_max_p,
           fmt(global_max_p_n));
    printf("  q = n - p:         %s\n",
           fmt(global_max_p_n - (uint64_t)global_max_p));
    printf("\n");

    /* ── Distribution ────────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[DISTRIBUTION] Minimal prime p frequency\n");
    printf("----------------------------------------------------------------------\n\n");
    const char *bucket_labels[] = {
        "p = 3       ", "p = 5..10   ", "p = 11..100 ",
        "p = 101..1K ", "p = 1K..10K ", "p > 10K     "
    };
    for (int i = 0; i < 6; i++) {
        double frac = total_verified > 0
            ? (double)p_bucket[i] / (double)total_verified : 0.0;
        int bar_len = (int)(frac * 50.0);
        if (bar_len > 50) bar_len = 50;
        char bar[51];
        memset(bar, '#', (size_t)bar_len);
        bar[bar_len] = '\0';
        printf("    %s: %14s  (%5.1f%%) %s\n",
               bucket_labels[i], fmt(p_bucket[i]), 100.0 * frac, bar);
    }
    printf("\n");

    /* ── Verdict ─────────────────────────────────────────────────────── */
    printf("----------------------------------------------------------------------\n");
    printf("[VERDICT]\n");
    printf("----------------------------------------------------------------------\n\n");

    if (total_failed == 0) {
        printf("  GOLDBACH CONJECTURE HOLDS for all even n in [%s, %s]\n",
               fmt(n_start), fmt(n_end));
        printf("  %s values verified, zero failures\n", fmt(total_verified));
        if (n_end > 4000000000000000000ULL) {
            printf("\n  >>> EXTENDS BEYOND Oliveira e Silva (2014) boundary of 4×10^18 <<<\n");
            printf("  >>> New frontier: %s <<<\n", fmt(n_end));
        }
    } else {
        printf("  *** GOLDBACH CONJECTURE VIOLATED — %s failures ***\n",
               fmt(total_failed));
    }
    printf("\n");

    /* ── Summary ─────────────────────────────────────────────────────── */
    printf("======================================================================\n");
    printf("  n = [%s, %s]  |  %s verified  |  max_p = %u\n",
           fmt(n_start), fmt(n_end), fmt(total_verified), global_max_p);
    printf("  Time: %.2fs  |  Rate: %.0f n/s  |  Backend: GPU (%s)\n",
           t_total, (double)total_verified / t_total, prop.name);
    printf("  Primality: 7-base deterministic MR, no float, no sieve\n");
    printf("======================================================================\n");

    cudaFree(dev_p);
    cudaFreeHost(h_p);

    return total_failed > 0 ? 1 : 0;
}
