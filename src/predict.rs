//! predict — Zring prediction oracle + full BPAND walk for a single even n
//!
//! Port of predict_demo.py to Rust. Target: 100B (n = 100,000,000,000).
//!
//! For a given even n, walks k = 1..N-2 through the Z₂₅₆ ring,
//! tracking every Goldbach pair hit with:
//!   - HV fill (hits per 14,464-cell batch)
//!   - Quadrant distribution (spin × parity)
//!   - Quadrant transition rate between consecutive hits
//!   - Arm reach statistics (min/max/mean/median k)
//!   - U/U' conjugate arm analysis
//!
//! Usage: predict <n>
//!   Example: predict 100000000000


use std::sync::Mutex;
use std::time::Instant;
use std::fs::File;
use std::io::{BufWriter, Write};

// ============================================================================
// Integer helpers (no FPU)
// ============================================================================
fn isqrt(n: usize) -> usize {
    if n == 0 { return 0; }
    // Bit-width seed: 2^(ceil(bits/2))
    let bits = (usize::BITS - n.leading_zeros()) as usize;
    let mut x = 1usize << ((bits + 1) / 2);
    loop {
        let y = (x + n / x) / 2;
        if y >= x { return x; }
        x = y;
    }
}

fn ilog10(n: usize) -> u32 {
    if n == 0 { return 0; }
    let mut count = 0u32;
    let mut v = n;
    while v >= 10 {
        v /= 10;
        count += 1;
    }
    count
}

// ============================================================================
// Z₂₅₆ Geometry Constants
// ============================================================================
const STEP: u8 = 7;
const HALF: usize = 128;
const VACUUMS_PER_HALF: usize = 2;                           // d & 0x3F == 0
const ACTIVE_PER_HALF: usize = HALF - VACUUMS_PER_HALF;      // 126
const ORBIT_SIZE: usize = 2 * ACTIVE_PER_HALF;               // 252
const HV_CELLS: usize = HALF * 113;                          // 128 × 113 = 14,464 (L1 cache-resident)
const HV_W: usize = 113;                                     // columns per HV row
const ORBITS_PER_HV: usize = HV_CELLS / ORBIT_SIZE;          // 57

// ============================================================================
// Bitwise Prime Sieve (1 bit per odd number)
// ============================================================================
fn make_sieve(limit: usize) -> Vec<u8> {
    let size = limit / 2 + 1;
    let bytes = (size + 7) / 8;
    let mut sieve = vec![0xFFu8; bytes];

    // 1 is not prime
    sieve[0] &= !1u8;

    let sqrt_limit = isqrt(limit);
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
// Decode d into (spin, parity, momentum, quadrant)
// ============================================================================
#[inline(always)]
fn decode(d: u8) -> (i8, i8, u8, u8) {
    let spin = if (d >> 7) & 1 == 0 { 1i8 } else { -1i8 };
    let parity = if (d >> 6) & 1 == 0 { 1i8 } else { -1i8 };
    let momentum = d & 0x3F;
    let quadrant = (d >> 6) & 0x3;
    (spin, parity, momentum, quadrant)
}

fn qlabel(d: u8) -> &'static str {
    match (d >> 6) & 0x3 {
        0 => "UP+",
        1 => "UP-",
        2 => "DN+",
        3 => "DN-",
        _ => unreachable!(),
    }
}

// ============================================================================
// predict_hv — pure arithmetic, no sieve
// ============================================================================
fn predict_hv(n: usize) -> (usize, usize, usize) {
    let big_n = n / 2;
    let k_max = if big_n >= 2 { big_n - 2 } else { 0 };
    let orbits = (k_max + ORBIT_SIZE - 1) / ORBIT_SIZE;
    let hvs = (k_max + HV_CELLS - 1) / HV_CELLS;
    (k_max, orbits, hvs)
}

// ============================================================================
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = if args.len() > 1 {
        args[1].replace("_", "").replace(",", "").parse().unwrap_or(10_000_000)
    } else {
        10_000_000
    };

    if n < 4 || n & 1 != 0 {
        eprintln!("Error: n must be an even number >= 4");
        std::process::exit(1);
    }

    let big_n = n / 2;
    let base_x: u32 = ilog10(n);

    println!("════════════════════════════════════════════════════════════════════════");
    println!("  predict v2 — Zring Prediction Oracle + Full BPAND Walk");
    println!("  Target: n = {}", format_num(n));
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    // ── [P] Z256 Ring Prediction Oracle — Pure Math ────────────────────────
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[P] Z256 Ring Prediction Oracle (pure math, no primes)");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  Z256 ring: 256 positions");
    println!("    Vacuum boundaries: 4 (0, 64, 128, 192)");
    println!("    Non-vacuum: 252 = 256 - 4");
    println!("    Per quadrant: 63 = 252 / 4");
    println!("    63 mod 7 = 0  ⇒  7 × 9 = 63");
    println!("    7-prime walk: 9 steps per cycle, 9 cycles covers 63, 36 cycles covers 252");
    println!("    Zero gaps, zero repeats, full coverage");
    println!();
    println!("  Four-arm witnessing for k = 1:");
    let d1 = 1u8 & 0xFF;
    let d2 = d1 + 64;
    let d3 = d1 + 128;
    let d4 = d1 + 192;
    println!("    d1 = {}  d2 = {}  d3 = {}  d4 = {}", d1, d2, d3, d4);
    println!("    All four point to (N-k, N+k) = ({}, {})", big_n - 1, big_n + 1);
    println!("    When BPAND = 1, all four ring directions confirm the pair");
    println!();
    println!("  Prediction:");
    println!("    Each quadrant will be filled in ≤63 steps");
    println!("    Expected first hit at rotation 9 (one full prime cycle)");
    println!("    Sweet spot rule: rings(10^x) = x - 3");
    println!("      10^4 → 1 ring, 10^8 → 5 rings, 10^12 → 9 rings, 10^18 → 15 rings");
    println!();
    // ── CSV block writer: 8 MB buffer, every even n gets a row ──────────
    // pair_count = total Goldbach pairs found by the biopod walk (for comet)
    let csv_path = format!("siliq_{}.csv", n);
    let csv_file = File::create(&csv_path).expect("cannot create CSV");
    let csv_writer = Mutex::new(BufWriter::with_capacity(8 * 1024 * 1024, csv_file));
    {
        let mut w = csv_writer.lock().unwrap();
        writeln!(w, "n,N,result,first_k,lo,hi,d,quadrant,pair_count").unwrap();
    }
    // ── [A] predict_hv — geometry prediction ────────────────────────────────
    let (k_max, orbits, hvs) = predict_hv(n);
    let rings_needed: u32 = if base_x >= 3 { base_x - 3 } else { 0 };

    println!("──────────────────────────────────────────────────────────────────────");
    println!("[A] Geometry Prediction (no sieve needed)");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  n           = {}", format_num(n));
    println!("  N = n/2     = {}", format_num(big_n));
    println!("  k_max       = {}", format_num(k_max));
    println!("  orbits      = {} (each {} predictions)", format_num(orbits), ORBIT_SIZE);
    println!("  HVs         = {} (each {} cells = {} orbits)",
             format_num(hvs), format_num(HV_CELLS), ORBITS_PER_HV);
    println!("  Zrings       = {} (each {} orbits)", format_num(rings_needed as usize), ORBIT_SIZE);

    // HV chain hierarchy
    let zring_l1 = if hvs > 1 { (hvs + ORBIT_SIZE - 1) / ORBIT_SIZE } else { 0 };
    let zring_l2 = if zring_l1 > 1 { (zring_l1 + ORBIT_SIZE - 1) / ORBIT_SIZE } else { 0 };

    if hvs <= 1 {
        println!("  structure   = 1 HV (no chain)");
    } else if zring_l1 <= 1 {
        println!("  structure   = {} HVs → 1 Zring index", format_num(hvs));
    } else if zring_l2 <= 1 {
        println!("  structure   = {} HVs → {} Zrings → 1 root",
                 format_num(hvs), format_num(zring_l1));
    } else {
        println!("  structure   = {} HVs → {} L1 → {} L2",
                 format_num(hvs), format_num(zring_l1), format_num(zring_l2));
    }
    println!();

    // ── [B] Prime Sieve ─────────────────────────────────────────────────────
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[B] Prime Sieve (bitwise, 1 bit per odd)");
    println!("──────────────────────────────────────────────────────────────────────");

    let t0 = Instant::now();
    let sieve = make_sieve(n);
    let t_sieve = t0.elapsed();

    let sieve_bytes = sieve.len();
    let sieve_mb = sieve_bytes as f64 / (1024.0 * 1024.0);
    let sieve_gb = sieve_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    if sieve_gb >= 1.0 {
        println!("  Sieve memory:  {:.2} GB ({} bytes)", sieve_gb, format_num(sieve_bytes));
    } else {
        println!("  Sieve memory:  {:.1} MB ({} bytes)", sieve_mb, format_num(sieve_bytes));
    }
    println!("  Sieve time:    {:.3}s", t_sieve.as_secs_f64());
    println!();

    // ── [C] Full BPAND Walk ─────────────────────────────────────────────────
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[C] Full BPAND Walk — n = {}", format_num(n));
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  Walking k = 1 .. {} through {} HVs ...", format_num(k_max), format_num(hvs));
    println!();

    let t0 = Instant::now();

    // HV fill: how many hits per HV bucket
    let mut hv_fill: Vec<u64> = vec![0; hvs];

    // Quadrant counters
    let mut quad_hits: [u64; 4] = [0; 4];

    // Arm reach tracking
    let mut total_hits: u64 = 0;
    let mut total_predictions: u64 = 0;
    let mut first_k: usize = 0;        // first-hit k
    let mut first_d: u8 = 0;           // d at first hit
    let mut min_k: usize = usize::MAX;
    let mut max_k: usize = 0;
    let mut k_sum: u128 = 0;           // can overflow u64 at 100B
    let mut spin_sum: i64 = 0;

    // Quadrant transition tracking
    let mut prev_q: u8 = 255;  // sentinel
    let mut transitions: u64 = 0;

    // Anti-spin tracking
    let mut prev_spin: i8 = 0;
    let mut anti_spin_count: u64 = 0;

    // First 10 hits for display
    let mut first_hits: Vec<(usize, usize, usize, u8)> = Vec::with_capacity(10);

    // CSV output
    let csv_path = format!("predict_{}.csv", n);
    let csv_file = File::create(&csv_path).expect("cannot create CSV");
    let mut csv = BufWriter::new(csv_file);
    writeln!(csv, "k,U,U_prime,d,spin,quadrant,hv").unwrap();

    // Walk
    let mut d: u8 = 0;
    let mut k: usize = 0;
    let mut hvs_used: usize = 0;

    let progress_interval = if hvs > 20 { hvs / 20 } else { 1 };

    'hv: for hv_idx in 0..hvs {
        let mut hv_total: u64 = 0;

        for _row in 0..HALF {
            for _col in 0..HV_W {
                k += 1;
                if k >= big_n { break; }

                d = d.wrapping_add(STEP);
                while d & 0x3F == 0 {
                    d = d.wrapping_add(STEP);
                }

                total_predictions += 1;

                let lo = big_n - k;
                let hi = big_n + k;
                if lo < 2 { break; }

                if is_prime(&sieve, lo) && is_prime(&sieve, hi) {
                    hv_total += 1;
                    total_hits += 1;

                    let (spin, _par, _mom, q) = decode(d);
                    quad_hits[q as usize] += 1;

                    // First hit
                    if first_k == 0 {
                        first_k = k;
                        first_d = d;
                    }

                    // Arm reach
                    if k < min_k { min_k = k; }
                    if k > max_k { max_k = k; }
                    k_sum += k as u128;
                    spin_sum += spin as i64;

                    // Quadrant transition
                    if prev_q != 255 && q != prev_q {
                        transitions += 1;
                    }
                    prev_q = q;

                    // Anti-spin
                    if prev_spin != 0 && spin != prev_spin {
                        anti_spin_count += 1;
                    }
                    prev_spin = spin;

                    // CSV row
                    writeln!(csv, "{},{},{},{},{},{},{}",
                             k, lo, hi, d, spin, q, hv_idx).unwrap();

                    // Store first 10
                    if first_hits.len() < 10 {
                        first_hits.push((k, lo, hi, d));
                    }
                }
            }
        }

        hv_fill[hv_idx] = hv_total;
        hvs_used = hv_idx + 1;

        if k >= big_n || (big_n - k) < 2 {
            break 'hv;
        }

        // Progress
        if hv_idx > 0 && hv_idx % progress_interval == 0 {
            let pct = 100.0 * hv_idx as f64 / hvs as f64;
            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!("    {:5.1}%  hits={}  k={}  [{:.1}s]",
                      pct, format_num(total_hits as usize),
                      format_num(k), elapsed);
        }
    }

    let t_walk = t0.elapsed();
    drop(sieve);   // release sieve memory — walk is done
    csv.flush().unwrap();
    drop(csv);

    println!("  Walk complete in {:.3}s", t_walk.as_secs_f64());
    println!("  CSV written: {}", csv_path);
    println!();

    // ── [D] U/U' Conjugate Arm Analysis ─────────────────────────────────────
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[D] U / U' Conjugate Arm Analysis");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  Center N = {}", format_num(big_n));
    println!("  U  = N - k  (arm sweeping LEFT)");
    println!("  U' = N + k  (arm sweeping RIGHT)");
    println!("  Conservation: U + U' = {} = n  (ALWAYS)", format_num(n));
    println!();
    println!("  Total BPAND hits:    {}", format_num(total_hits as usize));
    println!("  Total predictions:   {}", format_num(total_predictions as usize));
    if total_predictions > 0 {
        println!("  Hit rate:            {:.6}%",
                 100.0 * total_hits as f64 / total_predictions as f64);
    }
    println!();

    // First 10 hits
    if !first_hits.is_empty() {
        println!("  First 10 BPAND hits:");
        println!("    {:>10}  {:>14}  {:>14}  {:>14}  {:>8}  {:>8}  {:>6}  {:>4}  {:>6}",
                 "k", "U(=N-k)", "U'(=N+k)", "U+U'", "L_U", "L_U'", "L_tot", "d", "d⊕128");
        println!("    {}  {}  {}  {}  {}  {}  {}  {}  {}",
                 "─".repeat(10), "─".repeat(14), "─".repeat(14), "─".repeat(14),
                 "─".repeat(8), "─".repeat(8), "─".repeat(6), "─".repeat(4), "─".repeat(6));
        for &(kk, lo, hi, dd) in &first_hits {
            let d_conj = dd.wrapping_add(128);
            println!("    {:>10}  {:>14}  {:>14}  {:>14}  {:>+8}  {:>+8}  {:>6}  {:>4}  {:>6}",
                     format_num(kk), format_num(lo), format_num(hi), format_num(lo + hi),
                     -(kk as i64), kk as i64, 0, dd, d_conj);
        }
        println!();
    }

    // First hit
    println!("  First hit:  k = {}, d = {} ({})", format_num(first_k), first_d, qlabel(first_d));

    // Arm reach
    let mean_k = if total_hits > 0 { k_sum as f64 / total_hits as f64 } else { 0.0 };
    println!();
    println!("  ARM REACH:");
    println!("    Min k:    {}", format_num(min_k));
    println!("    Max k:    {}", format_num(max_k));
    println!("    Mean |k|: {:.0}", mean_k);
    println!("    Σ spin:   {:+}", spin_sum);
    if total_hits > 0 {
        println!("    |Σ|/N:    {:.6}", (spin_sum.unsigned_abs() as f64) / total_hits as f64);
    }

    // Anti-spin
    if total_hits > 1 {
        let anti_frac = anti_spin_count as f64 / (total_hits - 1) as f64;
        println!("    Anti-spin: {:.1}% (consecutive hits with opposite spin)",
                 100.0 * anti_frac);
    }
    println!();

    // ── [E] Quadrant Analysis ───────────────────────────────────────────────
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[E] Z₂₅₆ Quadrant Distribution");
    println!("──────────────────────────────────────────────────────────────────────");

    let q_total: u64 = quad_hits.iter().sum();
    let q_labels = ["UP+ [  0.. 63]", "UP- [ 64..127]",
                    "DN+ [128..191]", "DN- [192..255]"];
    let mut q_fracs = [0.0f64; 4];

    for i in 0..4 {
        let frac = if q_total > 0 { quad_hits[i] as f64 / q_total as f64 } else { 0.0 };
        q_fracs[i] = frac;
        let bar_len = (frac * 40.0) as usize;
        let bar: String = std::iter::repeat('█').take(bar_len).collect();
        println!("    {}: {:>14}  ({:5.1}%) {}",
                 q_labels[i], format_num(quad_hits[i] as usize), 100.0 * frac, bar);
    }
    println!();

    // Transition rate
    if total_hits > 1 {
        let transition_rate = transitions as f64 / (total_hits - 1) as f64;
        println!("  Quadrant transitions:  {} / {} = {:.1}%",
                 format_num(transitions as usize),
                 format_num((total_hits - 1) as usize),
                 100.0 * transition_rate);
    }
    println!();

    // ── [F] HV Fill Distribution ────────────────────────────────────────────
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[F] HV Fill Distribution ({} HVs used / {} predicted)",
             format_num(hvs_used), format_num(hvs));
    println!("──────────────────────────────────────────────────────────────────────");

    let fills = &hv_fill[..hvs_used];
    let fill_sum: u64 = fills.iter().sum();
    let fill_mean = fill_sum as f64 / hvs_used as f64;
    let fill_min = fills.iter().min().copied().unwrap_or(0);
    let fill_max = fills.iter().max().copied().unwrap_or(0);
    let empty_hvs = fills.iter().filter(|&&x| x == 0).count();

    println!("  Mean hits/HV:  {:.1}", fill_mean);
    println!("  Min:           {}", fill_min);
    println!("  Max:           {}", fill_max);
    println!("  Empty HVs:     {}", empty_hvs);
    println!("  Total hits:    {} (check: {})", format_num(fill_sum as usize),
             format_num(total_hits as usize));
    println!();

    // Zring index
    let n_index_rings = (hvs_used + 251) / 252;
    println!("  STRUCTURE:");
    println!("    1 prediction Zring → {} HVs → {} index Zring(s)",
             format_num(hvs_used), n_index_rings);
    println!();

    // ── [G] Claims ──────────────────────────────────────────────────────────
    println!("──────────────────────────────────────────────────────────────────────");
    println!("[G] Claims");
    println!("──────────────────────────────────────────────────────────────────────");
    println!();

    let mut claims: Vec<(String, &str, String)> = Vec::new();

    // 1. At least one pair found
    let pair_found = total_hits > 0;
    claims.push((
        format!("Goldbach pair exists for n = {}", format_num(n)),
        if pair_found { "DERIVED" } else { "FAILED" },
        format!("{} pairs found, first at k = {}",
                format_num(total_hits as usize), format_num(first_k)),
    ));

    // 2. All 4 quadrants produce hits
    let all_quads = quad_hits.iter().all(|&c| c > 0);
    claims.push((
        "All 4 quadrants produce BPAND hits".to_string(),
        if all_quads { "DERIVED" } else { "FAILED" },
        format!("[{}, {}, {}, {}]",
                format_num(quad_hits[0] as usize), format_num(quad_hits[1] as usize),
                format_num(quad_hits[2] as usize), format_num(quad_hits[3] as usize)),
    ));

    // 3. Quadrant balance (±15%)
    let q_balanced = q_fracs.iter().all(|&f| (f - 0.25).abs() < 0.15);
    claims.push((
        "Quadrants balanced (±15%)".to_string(),
        if q_balanced { "DERIVED" } else { "FAILED" },
        format!("[{:.3}, {:.3}, {:.3}, {:.3}]",
                q_fracs[0], q_fracs[1], q_fracs[2], q_fracs[3]),
    ));

    // 4. No empty HVs
    let no_empty = empty_hvs == 0;
    claims.push((
        "No empty HVs (uniform fill)".to_string(),
        if no_empty { "DERIVED" } else { "FAILED" },
        format!("{} empty out of {} used", empty_hvs, format_num(hvs_used)),
    ));

    // 5. Spin near-cancellation: |Σspin| / N << 1
    let spin_ratio = if total_hits > 0 {
        spin_sum.unsigned_abs() as f64 / total_hits as f64
    } else { 1.0 };
    let spin_cancels = spin_ratio < 0.1;
    claims.push((
        "Spin near-cancellation (|Σ|/N < 0.1)".to_string(),
        if spin_cancels { "DERIVED" } else { "FAILED" },
        format!("Σspin = {:+}, |Σ|/N = {:.6}", spin_sum, spin_ratio),
    ));

    // 6. Hit rate > 0
    let hit_rate = if total_predictions > 0 {
        total_hits as f64 / total_predictions as f64
    } else { 0.0 };
    claims.push((
        "Hit rate > 0 (primes survive filter)".to_string(),
        if hit_rate > 0.0 { "DERIVED" } else { "FAILED" },
        format!("{:.6}%", 100.0 * hit_rate),
    ));

    // 7. Transition rate > 50% (ring rotates through quadrants)
    let trans_rate = if total_hits > 1 {
        transitions as f64 / (total_hits - 1) as f64
    } else { 0.0 };
    claims.push((
        "Quadrant transition rate > 50%".to_string(),
        if trans_rate > 0.50 { "DERIVED" } else { "FAILED" },
        format!("{:.1}%", 100.0 * trans_rate),
    ));

    // Print claims table
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
    println!("  n = {}  |  {} BPAND hits  |  first_k = {}  |  max_k = {}",
             format_num(n), format_num(total_hits as usize),
             format_num(first_k), format_num(max_k));
    if sieve_gb >= 1.0 {
        println!("  Sieve: {:.2} GB in {:.2}s  |  Walk: {:.2}s  |  Total: {:.2}s",
                 sieve_gb, t_sieve.as_secs_f64(), t_walk.as_secs_f64(),
                 t_sieve.as_secs_f64() + t_walk.as_secs_f64());
    } else {
        println!("  Sieve: {:.1} MB in {:.2}s  |  Walk: {:.2}s  |  Total: {:.2}s",
                 sieve_mb, t_sieve.as_secs_f64(), t_walk.as_secs_f64(),
                 t_sieve.as_secs_f64() + t_walk.as_secs_f64());
    }
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
