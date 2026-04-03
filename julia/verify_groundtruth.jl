#!/usr/bin/env julia
"""
    verify_groundtruth.jl — Independent primality ground truth

    Walks k = 1..K_LIMIT for n = 4×10¹⁸, tests both arms (N-k, N+k)
    with Julia's Primes.isprime(), counts hits + quadrant distribution.

    This is the REFEREE. No orbit tables, no sieves, no uint256.
    Just: are both N-k and N+k prime? Count and classify.
"""

using Primes
using Printf

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
const n       = Int128(4_000_000_000_000_000_000)
const big_n   = n ÷ 2   # 2×10¹⁸
const K_LIMIT = 10_000_000

# Z₂₅₆ orbit — exact same logic as both C implementations
const STEP       = UInt8(7)
const ORBIT_SIZE = 252

function build_orbit()::Vector{UInt8}
    orbit = Vector{UInt8}(undef, ORBIT_SIZE)
    d = UInt8(0)
    for i in 1:ORBIT_SIZE
        d += STEP
        while (d & 0x3f) == 0
            d += STEP
        end
        orbit[i] = d
    end
    return orbit
end

function qlabel(d::UInt8)::String
    q = (d >> 6) & 0x03
    return q == 0 ? "UP+" : q == 1 ? "UP-" : q == 2 ? "DN+" : "DN-"
end

# ═══════════════════════════════════════════════════════════════════════════
# Walk
# ═══════════════════════════════════════════════════════════════════════════
function main()
    orbit = build_orbit()

    println("=" ^ 70)
    println("  Julia Ground-Truth Verifier")
    println("  n = $n")
    println("  N = n/2 = $big_n")
    println("  Sweep: k = 1 .. $(K_LIMIT)")
    println("  Primality: Primes.isprime (deterministic)")
    println("=" ^ 70)
    println()

    total_hits   = 0
    quad_hits    = zeros(Int, 4)   # Q0=UP+, Q1=UP-, Q2=DN+, Q3=DN-
    first_k      = 0
    first_d      = UInt8(0)
    min_k        = typemax(Int)
    max_k        = 0
    spin_sum     = 0

    # First 10 hits for display
    first_hits = Vector{NamedTuple{(:k, :lo, :hi, :d), Tuple{Int,Int128,Int128,UInt8}}}()

    t0 = time()
    progress_iv = K_LIMIT ÷ 20
    next_progress = progress_iv

    for k in 1:K_LIMIT
        lo = big_n - k
        hi = big_n + k

        # Parity gate: both arms must be odd to both be prime (except lo=2)
        if xor(big_n, k) % 2 == 0 && lo != 2
            continue
        end

        lo < 2 && break

        # Independent primality test — Julia's Primes.isprime
        if isprime(lo) && isprime(hi)
            total_hits += 1

            d_val = orbit[((k - 1) % ORBIT_SIZE) + 1]
            q = Int((d_val >> 6) & 0x03)
            s = ((d_val >> 7) & 1) == 0 ? 1 : -1

            quad_hits[q + 1] += 1
            spin_sum += s

            if first_k == 0
                first_k = k
                first_d = d_val
            end
            if k < min_k; min_k = k; end
            if k > max_k; max_k = k; end

            if length(first_hits) < 10
                push!(first_hits, (k=k, lo=lo, hi=hi, d=d_val))
            end
        end

        if k >= next_progress
            pct = 100.0 * k / K_LIMIT
            dt = time() - t0
            @printf(stderr, "    %5.1f%%  hits=%d  k=%d  [%.1fs]\n", pct, total_hits, k, dt)
            next_progress += progress_iv
        end
    end

    t_walk = time() - t0

    # ── Results ──────────────────────────────────────────────────────
    println("-" ^ 70)
    println("[RESULT] Julia Ground Truth — n = $n")
    println("-" ^ 70)
    println()
    @printf("  Total BPAND hits:  %d\n", total_hits)
    @printf("  k tested:          %d\n", K_LIMIT)
    @printf("  Hit rate:          %.6f%%\n", 100.0 * total_hits / K_LIMIT)
    @printf("  Walk time:         %.2fs\n\n", t_walk)

    # First hits table
    if !isempty(first_hits)
        println("  First 10 BPAND hits:")
        @printf("    %10s  %-30s  %-30s  %4s  %s\n", "k", "U(=N-k)", "U'(=N+k)", "d", "label")
        for h in first_hits
            @printf("    %10d  %-30d  %-30d  %4d  %s\n", h.k, h.lo, h.hi, h.d, qlabel(h.d))
        end
        println()
    end

    @printf("  First hit:  k = %d, d = %d (%s)\n\n", first_k, first_d, qlabel(first_d))

    # Quadrant distribution
    println("-" ^ 70)
    println("[QUADRANT] Z256 Quadrant Distribution")
    println("-" ^ 70)
    labels = ["UP+ [  0.. 63]", "UP- [ 64..127]", "DN+ [128..191]", "DN- [192..255]"]
    q_total = sum(quad_hits)
    for i in 1:4
        frac = q_total > 0 ? quad_hits[i] / q_total : 0.0
        bar = "#" ^ round(Int, frac * 40)
        @printf("    %s: %10d  (%5.1f%%) %s\n", labels[i], quad_hits[i], 100.0 * frac, bar)
    end
    println()

    @printf("  Spin sum: %+d\n", spin_sum)
    if total_hits > 0
        @printf("  |Spin|/N: %.6f\n", abs(spin_sum) / total_hits)
    end
    println()

    # ── Comparison table ─────────────────────────────────────────────
    println("=" ^ 70)
    println("[COMPARISON] Julia vs predict_cuda256 vs predict.c")
    println("=" ^ 70)
    println()
    @printf("  %-25s  %-15s  %-15s  %-15s\n", "Metric", "Julia", "CUDA256", "predict.c")
    @printf("  %-25s  %-15s  %-15s  %-15s\n", "-"^25, "-"^15, "-"^15, "-"^15)
    @printf("  %-25s  %-15d  %-15d  %-15s\n", "Hits (10M sweep)", total_hits, 9892, "~14,874*")
    @printf("  %-25s  %-15.4f%%  %-15.4f%%  %-15s\n", "Hit rate", 100.0*total_hits/K_LIMIT, 0.0989, "0.1487%*")
    @printf("  %-25s  %d/%d/%d/%d  %s  %s\n", "Quadrant %%",
            round(Int, 100*quad_hits[1]/q_total),
            round(Int, 100*quad_hits[2]/q_total),
            round(Int, 100*quad_hits[3]/q_total),
            round(Int, 100*quad_hits[4]/q_total),
            "33/17/33/17", "25/25/25/25*")
    println()
    println("  * predict.c values scaled from 10B sweep (÷100)")
    println()

    if total_hits == 9892
        println("  ✓ VERDICT: predict_cuda256 MATCHES Julia ground truth ($(total_hits) == 9892)")
    elseif abs(total_hits - 9892) <= 5
        println("  ~ VERDICT: predict_cuda256 NEARLY matches ($(total_hits) vs 9892) — check edge cases")
    else
        println("  ✗ VERDICT: MISMATCH — Julia=$(total_hits), CUDA256=9892 — investigate further")
    end
    println()
    println("=" ^ 70)
end

main()
