//! siliq-goldbach — Biopod Goldbach Engine (Rust)
//!
//! Verifies Goldbach's conjecture for all even n in [4, N_MAX]
//! using the Z₂₅₆ seven-prime-step ring walk.
//!
//! Architecture:
//!   - Bitwise prime sieve (1 bit per odd number → 8× denser than Python uint8)
//!   - siliq engine: exact port of the 12-line Python core
//!   - Parallel scan over even numbers via rayon
//!   - Claims table + scorecard (data-driven, no hardcoded conclusions)

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use std::fs::File;
use std::io::{BufWriter, Write};

// ============================================================================
// Constants
// ============================================================================
const DEFAULT_N_MAX: usize = 2_000_000_000;
const STEPS: [u8; 7] = [2, 3, 5, 7, 11, 13, 17];
const BATCH_OUTER: usize = 128;
const BATCH_INNER: usize = 113;
const BATCH_CELLS: usize = BATCH_OUTER * BATCH_INNER; // = 14,464

// Thresholds — defined BEFORE computation
const FAILURE_THRESHOLD: usize = 0;
const CONVERGENCE_K_THRESH: f64 = 15000.0; // scales with range

// ============================================================================
// Siliq result: distinguish pair-found vs search-space-exhausted vs batch-limit
// ============================================================================
#[derive(Clone, Copy, PartialEq)]
enum SiliqResult {
    PairFound(usize, u8),  // (first_k, d_phase at hit)
    Exhausted,             // k reached N or lo < 2 — search space fully covered
    BatchLimit,            // hit 14,464 cells without finding a pair
}

// ============================================================================
// Bitwise Prime Sieve
// ============================================================================
// Bit i represents odd number 2*i + 1. Even numbers handled separately.
// sieve[i >> 3] & (1 << (i & 7)) != 0  ↔  (2*i+1) is prime

fn make_sieve(limit: usize) -> Vec<u8> {
    let size = limit / 2 + 1;
    let bytes = (size + 7) / 8;
    let mut sieve = vec![0xFFu8; bytes];

    // 1 is not prime: index 0 represents 1
    sieve[0] &= !1u8;

    let sqrt_limit = (limit as f64).sqrt() as usize;
    let mut i = 3usize;
    while i <= sqrt_limit {
        let idx = i / 2;
        if sieve[idx >> 3] & (1 << (idx & 7)) != 0 {
            let mut j = i * i;
            while j <= limit {
                let jdx = j / 2;
                sieve[jdx >> 3] &= !(1 << (jdx & 7));
                j += 2 * i;
            }
        }
        i += 2;
    }
    sieve
}

#[inline(always)]
fn is_prime(sieve: &[u8], n: usize) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n & 1 == 0 { return false; }
    let idx = n / 2;
    sieve[idx >> 3] & (1 << (idx & 7)) != 0
}

// ============================================================================
// Siliq Engine — the 12-line core, zero-allocation
// ============================================================================

#[inline]
fn siliq_has_pair(sieve: &[u8], n: usize, n_max: usize) -> SiliqResult {
    let big_n = n >> 1;
    let mut d: u8 = 1;
    let mut si: usize = 0;
    let mut k: usize = 0;

    for _ in 0..BATCH_OUTER {
        for _ in 0..BATCH_INNER {
            k += 1;
            if k >= big_n { return SiliqResult::Exhausted; }

            d = d.wrapping_add(STEPS[si % 7]);
            si += 1;
            while d & 0x3F == 0 {
                d = d.wrapping_add(STEPS[si % 7]);
                si += 1;
            }

            let lo = big_n - k;
            let hi = big_n + k;
            if lo < 2 { return SiliqResult::Exhausted; }
            if hi <= n_max && is_prime(sieve, lo) && is_prime(sieve, hi) {
                return SiliqResult::PairFound(k, d);
            }
        }
    }
    SiliqResult::BatchLimit
}

// ============================================================================
// Full siliq — returns ALL pairs with (k, d) for CSV output
// ============================================================================
fn siliq_all_pairs(sieve: &[u8], n: usize, n_max: usize) -> Vec<(usize, u8)> {
    let big_n = n >> 1;
    let mut d: u8 = 1;
    let mut si: usize = 0;
    let mut k: usize = 0;
    let mut pairs: Vec<(usize, u8)> = Vec::new();

    for _ in 0..BATCH_OUTER {
        for _ in 0..BATCH_INNER {
            k += 1;
            if k >= big_n { return pairs; }

            d = d.wrapping_add(STEPS[si % 7]);
            si += 1;
            while d & 0x3F == 0 {
                d = d.wrapping_add(STEPS[si % 7]);
                si += 1;
            }

            let lo = big_n - k;
            let hi = big_n + k;
            if lo < 2 { return pairs; }
            if hi <= n_max && is_prime(sieve, lo) && is_prime(sieve, hi) {
                pairs.push((k, d));
            }
        }
    }
    pairs
}

// Wrapper: counts + quadrant totals from all_pairs (used by section [C])
fn siliq_full(sieve: &[u8], n: usize, n_max: usize) -> (usize, [u64; 4]) {
    let pairs = siliq_all_pairs(sieve, n, n_max);
    let mut quads = [0u64; 4];
    for &(_k, d) in &pairs {
        quads[(d >> 6) as usize] += 1;
    }
    (pairs.len(), quads)
}

// ============================================================================
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_max: usize = if args.len() > 1 {
        args[1].replace("_", "").replace(",", "").parse().unwrap_or(DEFAULT_N_MAX)
    } else {
        DEFAULT_N_MAX
    };

    println!("════════════════════════════════════════════════════════════════════════");
    println!("  siliq-goldbach — Biopod Goldbach Engine (Rust)");
    println!("  Z₂₅₆ Seven-Prime-Step Pair Scanner");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  N_MAX              = {:>15}", format_num(n_max));
    println!("  Batch geometry     = {} × {} = {} cells", BATCH_OUTER, BATCH_INNER,
             BATCH_OUTER * BATCH_INNER);
    println!("  7-prime steps      = {:?}", STEPS);
    println!("  Failure threshold  = {}", FAILURE_THRESHOLD);
    println!();

    // ── [A] Prime Sieve ─────────────────────────────────────────────────────
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[A] Prime Sieve (bitwise, 1 bit per odd)");
    println!("──────────────────────────────────────────────────────────────────────");

    let t0 = Instant::now();
    let sieve = make_sieve(n_max);
    let t_sieve = t0.elapsed();

    let sieve_bytes = sieve.len();
    let sieve_mb = sieve_bytes as f64 / (1024.0 * 1024.0);

    // Count primes (parallel)
    let n_primes: usize = (2..=n_max)
        .into_par_iter()
        .filter(|&n| is_prime(&sieve, n))
        .count();

    println!("  Primes <= {}:  {}", format_num(n_max), format_num(n_primes));
    println!("  Sieve memory:        {:.1} MB ({} bytes)", sieve_mb, format_num(sieve_bytes));
    println!("  Sieve time:          {:.3}s", t_sieve.as_secs_f64());
    println!();

    // ── [B] Siliq Engine — Parallel Goldbach Verification ───────────────────
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[B] Siliq Engine — Parallel Goldbach Verification");
    println!("──────────────────────────────────────────────────────────────────────");

    let t0 = Instant::now();

    // ── CSV block writer: 8 MB buffer, every even n gets a row ──────────
    // pair_count = total Goldbach pairs found by the biopod walk (for comet)
    let csv_path = format!("siliq_{}.csv", n_max);
    let csv_file = File::create(&csv_path).expect("cannot create CSV");
    let csv_writer = Mutex::new(BufWriter::with_capacity(8 * 1024 * 1024, csv_file));
    {
        let mut w = csv_writer.lock().unwrap();
        writeln!(w, "n,N,result,first_k,lo,hi,d,quadrant,pair_count").unwrap();
    }

    let n_tested = AtomicUsize::new(0);
    let n_failures = AtomicUsize::new(0);
    let n_identity = AtomicUsize::new(0);
    let n_batch_overflow = AtomicUsize::new(0);  // batch limit hit, no pair found
    let first_k_sum = AtomicU64::new(0);
    let max_first_k = AtomicUsize::new(0);
    let progress_count = AtomicUsize::new(0);
    let progress_interval = std::cmp::max(1, n_max / 40);

    // Parallel scan: each even n from 4 to n_max
    // Uses siliq_all_pairs to get EVERY pair, records first-hit + total count
    (2..=(n_max / 2)).into_par_iter().for_each(|half| {
        let n = half * 2;
        let big_n = n >> 1;
        n_tested.fetch_add(1, Ordering::Relaxed);

        let pairs = siliq_all_pairs(&sieve, n, n_max);
        let pair_count = pairs.len();

        if pair_count > 0 {
            // First hit details
            let (fk, d) = pairs[0];
            let lo = big_n - fk;
            let hi = big_n + fk;
            let q = d >> 6;
            {
                let mut w = csv_writer.lock().unwrap();
                let _ = writeln!(w, "{},{},PairFound,{},{},{},{},{},{}",
                                 n, big_n, fk, lo, hi, d, q, pair_count);
            }
            first_k_sum.fetch_add(fk as u64, Ordering::Relaxed);
            let mut cur = max_first_k.load(Ordering::Relaxed);
            while fk > cur {
                match max_first_k.compare_exchange_weak(
                    cur, fk, Ordering::Relaxed, Ordering::Relaxed
                ) {
                    Ok(_) => break,
                    Err(actual) => cur = actual,
                }
            }
        } else {
            // No pairs found — check if identity or true failure
            if is_prime(&sieve, big_n) {
                {
                    let mut w = csv_writer.lock().unwrap();
                    let _ = writeln!(w, "{},{},Identity,0,{},{},0,0,1",
                                     n, big_n, big_n, big_n);
                }
                n_identity.fetch_add(1, Ordering::Relaxed);
            } else if big_n <= BATCH_CELLS {
                // Search space fully covered (n small enough)
                {
                    let mut w = csv_writer.lock().unwrap();
                    let _ = writeln!(w, "{},{},Exhausted,0,0,0,0,0,0", n, big_n);
                }
                n_failures.fetch_add(1, Ordering::Relaxed);
            } else {
                // Batch limit hit
                {
                    let mut w = csv_writer.lock().unwrap();
                    let _ = writeln!(w, "{},{},BatchLimit,0,0,0,0,0,0", n, big_n);
                }
                n_batch_overflow.fetch_add(1, Ordering::Relaxed);
                n_failures.fetch_add(1, Ordering::Relaxed);
            }
        }

        let p = progress_count.fetch_add(1, Ordering::Relaxed);
        if progress_interval > 0 && p % progress_interval == 0 && p > 0 {
            let pct = 100.0 * p as f64 / (n_max as f64 / 2.0);
            let elapsed = t0.elapsed().as_secs_f64();
            let fails = n_failures.load(Ordering::Relaxed);
            let ident = n_identity.load(Ordering::Relaxed);
            let overflow = n_batch_overflow.load(Ordering::Relaxed);
            eprintln!("    {:5.1}%  fails={}  identity={}  overflow={}  [{:.1}s]",
                      pct, fails, ident, overflow, elapsed);
        }
    });

    // Flush CSV block buffer
    {
        let mut w = csv_writer.lock().unwrap();
        w.flush().unwrap();
    }

    let t_engine = t0.elapsed();
    println!("  CSV written: {}", csv_path);
    let total_tested = n_tested.load(Ordering::Relaxed);
    let total_failures = n_failures.load(Ordering::Relaxed);
    let total_identity = n_identity.load(Ordering::Relaxed);
    let total_overflow = n_batch_overflow.load(Ordering::Relaxed);
    let total_fk_sum = first_k_sum.load(Ordering::Relaxed);
    let total_max_fk = max_first_k.load(Ordering::Relaxed);
    let n_with_pairs = total_tested - total_failures - total_identity;
    let avg_first_k = if n_with_pairs > 0 {
        total_fk_sum as f64 / n_with_pairs as f64
    } else {
        0.0
    };

    println!();
    println!("  Even numbers tested:    {}", format_num(total_tested));
    println!("  Engine time:            {:.2}s", t_engine.as_secs_f64());
    println!("  Throughput:             {}/s",
             format_num((total_tested as f64 / t_engine.as_secs_f64()) as usize));
    println!("  Pairs found (k >= 1):  {}", format_num(n_with_pairs));
    println!("  Identity-only (k = 0): {}", total_identity);
    println!("  Batch overflow (k > {}): {}", format_num(BATCH_CELLS), total_overflow);
    println!("  True failures:          {} (exhausted={}, overflow={})",
             total_failures, total_failures - total_overflow, total_overflow);
    println!("  Avg first-hit k:        {:.2}", avg_first_k);
    println!("  Max first-hit k:        {}", format_num(total_max_fk));

    let goldbach_pass = total_failures == 0;
    println!();
    if goldbach_pass {
        println!("  Goldbach [4..{}]: PASS ({} true failures, {} identity-only)",
                 format_num(n_max), total_failures, total_identity);
    } else {
        println!("  Goldbach [4..{}]: FAIL ({} true failures, {} identity-only)",
                 format_num(n_max), total_failures, total_identity);
    }

    // ── [C] Z₂₅₆ Ring Analysis (sampled) ───────────────────────────────────
    println!();
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[C] Z₂₅₆ Ring Analysis (10,000 samples)");
    println!("──────────────────────────────────────────────────────────────────────");

    let sample_size = 10_000usize;
    let step = std::cmp::max(1, n_max / sample_size);
    let t0s = Instant::now();

    let sample_results: Vec<(usize, [u64; 4])> = (0..sample_size)
        .into_par_iter()
        .map(|i| {
            let n = ((i * step) + 4) & !1;
            if n > n_max || n < 4 { return (0, [0u64; 4]); }
            siliq_full(&sieve, n, n_max)
        })
        .collect();

    let t_sample = t0s.elapsed();
    let mut total_pairs: u64 = 0;
    let mut quad_totals = [0u64; 4];
    let mut pair_counts: Vec<usize> = Vec::with_capacity(sample_size);

    for (count, quads) in &sample_results {
        total_pairs += *count as u64;
        pair_counts.push(*count);
        for i in 0..4 {
            quad_totals[i] += quads[i];
        }
    }

    let total_q: u64 = quad_totals.iter().sum();
    let q_labels = ["UP+ [  0.. 63]", "UP- [ 64..127]",
                    "DN+ [128..191]", "DN- [192..255]"];

    println!("  Sample size:         {}", format_num(sample_size));
    println!("  Sample time:         {:.2}s", t_sample.as_secs_f64());
    println!("  Total BPAND hits:    {}", format_num(total_pairs as usize));
    if !pair_counts.is_empty() {
        let avg_pairs = total_pairs as f64 / pair_counts.len() as f64;
        let min_p = pair_counts.iter().min().unwrap();
        let max_p = pair_counts.iter().max().unwrap();
        println!("  Avg pairs / n:       {:.1}", avg_pairs);
        println!("  Min / Max pairs:     {} / {}", min_p, max_p);
    }

    println!();
    println!("  Z₂₅₆ Quadrant Distribution:");
    let mut q_fracs = [0.0f64; 4];
    for i in 0..4 {
        let frac = if total_q > 0 { quad_totals[i] as f64 / total_q as f64 } else { 0.0 };
        q_fracs[i] = frac;
        let bar_len = (frac * 40.0) as usize;
        let bar: String = std::iter::repeat('█').take(bar_len).collect();
        println!("    {}: {:>12}  ({:5.1}%) {}", q_labels[i],
                 format_num(quad_totals[i] as usize), 100.0 * frac, bar);
    }

    // ── [D] 7-Prime Step Ring Walk ──────────────────────────────────────────
    println!();
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[D] 7-Prime Step Ring Walk");
    println!("──────────────────────────────────────────────────────────────────────");

    let mut d: u8 = 1;
    let mut si: usize = 0;
    let mut visited = [false; 256];
    let mut vacuum_skips: usize = 0;

    for _ in 0..BATCH_CELLS {
        d = d.wrapping_add(STEPS[si % 7]);
        si += 1;
        while d & 0x3F == 0 {
            vacuum_skips += 1;
            d = d.wrapping_add(STEPS[si % 7]);
            si += 1;
        }
        visited[d as usize] = true;
    }

    let non_vacuum: usize = (0..256u16).filter(|&v| v & 0x3F != 0).count();
    let coverage: usize = (0..256u16)
        .filter(|&v| v & 0x3F != 0 && visited[v as usize])
        .count();
    let cov_frac = coverage as f64 / non_vacuum as f64;

    println!("  Ring positions visited: {} / {} ({:.1}%)",
             coverage, non_vacuum, 100.0 * cov_frac);
    println!("  Vacuum skips:           {}", vacuum_skips);
    println!("  Total steps (w/ skip):  {}", si);

    // ── [E] Claims & Scorecard ──────────────────────────────────────────────
    println!();
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[E] Claims");
    println!("──────────────────────────────────────────────────────────────────────");
    println!();

    let mut claims: Vec<(String, &str, String)> = Vec::new();

    claims.push((
        format!("Goldbach pairs for all even n in [4, {}]", format_num(n_max)),
        if goldbach_pass { "DERIVED" } else { "FAILED" },
        format!("{} true failures ({} exhausted, {} overflow), {} identity-only",
                total_failures, total_failures - total_overflow, total_overflow, total_identity),
    ));

    let bpand_conv = avg_first_k < CONVERGENCE_K_THRESH;
    claims.push((
        format!("BPAND converges (avg k < {})", CONVERGENCE_K_THRESH as u64),
        if bpand_conv { "DERIVED" } else { "FAILED" },
        format!("avg k = {:.1}, max k = {}", avg_first_k, format_num(total_max_fk)),
    ));

    let batch_sufficient = total_overflow == 0;
    claims.push((
        format!("Batch D={}x{} sufficient", BATCH_OUTER, BATCH_INNER),
        if batch_sufficient { "DERIVED" } else { "FAILED" },
        format!("{} cells, {} batch overflows", BATCH_CELLS, total_overflow),
    ));

    let q_balanced = q_fracs.iter().all(|&f| (f - 0.25).abs() < 0.15);
    claims.push((
        "Z256 quadrants balanced (+-15%)".to_string(),
        if q_balanced { "DERIVED" } else { "FAILED" },
        format!("[{:.3}, {:.3}, {:.3}, {:.3}]",
                q_fracs[0], q_fracs[1], q_fracs[2], q_fracs[3]),
    ));

    claims.push((
        "7-prime walk covers >90% of ring".to_string(),
        if cov_frac > 0.90 { "DERIVED" } else { "FAILED" },
        format!("{}/{} = {:.1}%", coverage, non_vacuum, 100.0 * cov_frac),
    ));

    claims.push((
        "Vacuum boundaries (d mod 64=0) skipped".to_string(),
        if vacuum_skips > 0 { "DERIVED" } else { "FAILED" },
        format!("{} skips in {} steps", vacuum_skips, si),
    ));

    let throughput = total_tested as f64 / t_engine.as_secs_f64();
    claims.push((
        "Throughput > 1M even/sec".to_string(),
        if throughput > 1_000_000.0 { "DERIVED" } else { "FAILED" },
        format!("{}/s", format_num(throughput as usize)),
    ));

    println!("  {:<50} {:<10} {}", "Claim", "Status", "Evidence");
    println!("  {:<50} {:<10} {}",
             "─".repeat(50), "─".repeat(10), "─".repeat(40));
    for (text, status, evidence) in &claims {
        println!("  {:<50} {:<10} {}", text, status, evidence);
    }

    let n_derived = claims.iter().filter(|c| c.1 == "DERIVED").count();
    let n_failed = claims.iter().filter(|c| c.1 == "FAILED").count();

    println!();
    println!("  SCORECARD: {} DERIVED, {} FAILED", n_derived, n_failed);

    // ── Summary ─────────────────────────────────────────────────────────────
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    if goldbach_pass {
        println!("  GOLDBACH VERIFIED: {} even numbers checked, {} failures",
                 format_num(total_tested), total_failures);
    } else {
        println!("  GOLDBACH NOT VERIFIED: {} even numbers checked, {} failures",
                 format_num(total_tested), total_failures);
    }
    println!("  Total wall time:   {:.2}s (sieve) + {:.2}s (engine) = {:.2}s",
             t_sieve.as_secs_f64(), t_engine.as_secs_f64(),
             t_sieve.as_secs_f64() + t_engine.as_secs_f64());
    println!("  Sieve memory:      {:.1} MB", sieve_mb);
    println!("════════════════════════════════════════════════════════════════════════");
}

fn format_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(','); }
        result.push(c);
    }
    result.chars().rev().collect()
}
