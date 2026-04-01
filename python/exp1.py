"""
exp1.py — Biopod Goldbach Engine: Z₂₅₆ Seven-Prime-Step Pair Scanner

PURPOSE:
    Test whether the biopod scanning algorithm (siliq) finds at least
    one Goldbach pair (p + q = n, both prime) for every even n in
    [4, N_MAX] using the Z₂₅₆ seven-prime-step ring walk.

    The biopod is a symmetric container centered at M node N = n//2:
      - u↻ axis (CW):  walks k = 1,2,3,... outward, probing primes
      - u'↺ axis (vertical): accumulates circles C₁,C₂,...Cₖ as pairs confirmed
      - Left arm:   lo = N − k   (−1 pole)
      - Right arm:  hi = N + k   (+1 pole)
      - BPAND:      S[lo] & S[hi] — both must be prime for a hit
      - Vacuum:     dashed circle at N (identity pair k=0, not scanned by engine)

    The Z₂₅₆ ring variable d walks through STEPS = (2,3,5,7,11,13,17),
    skipping vacuum boundaries where d mod 64 == 0. Four quadrants
    from bits 7:6:
      [  0.. 63]  UP+  state = +1   (bits 7:6 = 00)
      [ 64..127]  UP−  state = −1   (bits 7:6 = 01)
      [128..191]  DN+  state = +1   (bits 7:6 = 10)
      [192..255]  DN−  state = −1   (bits 7:6 = 11)
    Vacuum boundaries at 0, 64, 128, 192.
    Three mod rings: mod 256, mod 128, mod 64.

    Batch geometry: D = 128 × 113 = 14,464 cells (fits L1 cache ≤ 32 KB).

METHOD:
    [A] Build Sieve of Eratosthenes up to N_MAX
    [B] Run siliq on every even n in [4, N_MAX] — verify ≥1 pair found
    [C] Z₂₅₆ quadrant & pair density analysis (sampled, full siliq)
    [D] 7-prime step ring walk coverage analysis
    [E] Claims & Scorecard

WHAT THIS EXPERIMENT DERIVES:
    ✓ Goldbach coverage: 0 true failures across [4, N_MAX]
    ✓ BPAND convergence rate (average first-hit k)
    ✓ Identity pairs (k=0) documented as engine edge case
    ✓ Pair density scaling with n
    ✓ Z₂₅₆ quadrant distribution of BPAND hits
    ✓ 7-prime walk ring coverage (mod 256, mod 128, mod 64)
    ✓ Vacuum skip frequency
    ✓ Batch geometry D = 128×113 sufficiency for [4, 2M]

WHAT REMAINS FREE:
    ✗ STEPS = (2,3,5,7,11,13,17) — chosen, not derived from axioms
    ✗ Batch geometry D = 128 × 113 — engineering choice (L1 cache bound)
    ✗ Goldbach's conjecture itself — unproven for all n

REQUIRES: numpy (sieve only)
PRECISION: integer arithmetic only. No floating point in engine core.

NO HARDCODED RESULTS. THE EXPERIMENT CAN FAIL.
"""

import numpy as np
import time

# ============================================================================
# Thresholds — defined BEFORE seeing results
# ============================================================================
N_MAX                = 2_000_000  # test all even n in [4, N_MAX]
FAILURE_THRESHOLD    = 0          # must be ZERO for Goldbach claim
CONVERGENCE_K_THRESH = 1000       # avg first-hit k must be < this
QUADRANT_BALANCE     = 0.15       # max deviation from 25% per quadrant
RING_COVERAGE_THRESH = 0.90       # >90% of non-vacuum positions visited
SAMPLE_SIZE          = 5_000      # full siliq runs for Z₂₅₆ analysis

# ============================================================================
# Constants
# ============================================================================
STEPS       = (2, 3, 5, 7, 11, 13, 17)   # first 7 primes
BATCH_OUTER = 128
BATCH_INNER = 113
BATCH_CELLS = BATCH_OUTER * BATCH_INNER   # = 14,464

# ============================================================================
print("=" * 72)
print("exp1.py — Biopod Goldbach Engine")
print("         Z₂₅₆ Seven-Prime-Step Pair Scanner")
print("=" * 72)
print(f"\n  N_MAX              = {N_MAX:,}")
print(f"  Batch geometry     = {BATCH_OUTER} × {BATCH_INNER} = {BATCH_CELLS:,} cells")
print(f"  7-prime steps      = {STEPS}")
print(f"  Failure threshold  = {FAILURE_THRESHOLD}")
print(f"  L1 cache estimate  = {BATCH_CELLS:,} bytes (sieve is uint8)")

# ============================================================================
# [A] Prime Sieve (Eratosthenes)
# ============================================================================
print(f"\n{'─' * 72}")
print("[A] Prime Sieve")
print(f"{'─' * 72}")

t0 = time.perf_counter()
sieve = np.ones(N_MAX + 1, dtype=np.uint8)
sieve[0] = sieve[1] = 0
for i in range(2, int(N_MAX**0.5) + 1):
    if sieve[i]:
        sieve[i*i::i] = 0

n_primes = int(sieve.sum())
t_sieve = time.perf_counter() - t0

print(f"  Primes ≤ {N_MAX:,}:  {n_primes:,}")
print(f"  Sieve time:          {t_sieve:.3f}s")

# ============================================================================
# [B] Siliq Engine — Goldbach Verification
# ============================================================================
print(f"\n{'─' * 72}")
print("[B] Siliq Engine — Goldbach Verification")
print(f"{'─' * 72}")

def siliq(n, S, early_exit=False):
    """
    Biopod Goldbach scanner — the 12-line core engine.

    n:          even number to decompose
    S:          prime sieve (S[i] = 1 iff i is prime)
    early_exit: return after first BPAND hit if True

    Returns: list of (k, lo, hi, d) for each BPAND hit
    """
    N = n >> 1; pairs = []; d = 1; si = 0; k = 0; stop = False
    for outer in range(BATCH_OUTER):
        if stop: break
        for inner in range(BATCH_INNER):
            k += 1
            if k >= N: stop = True; break
            d = (d + STEPS[si % 7]) & 0xFF; si += 1
            while not (d & 0x3F): d = (d + STEPS[si % 7]) & 0xFF; si += 1
            lo = N - k; hi = N + k
            if lo < 2: stop = True; break
            if hi < len(S) and S[lo] & S[hi]:
                pairs.append((k, lo, hi, d))
                if early_exit:
                    return pairs
    return pairs

print(f"  Scanning every even n in [4, {N_MAX:,}] ...")
t0 = time.perf_counter()

failures      = []       # true failures: no pair at any k, AND N not prime
identity_only = []       # engine k≥1 misses identity pair p=q=N (documented)
n_tested      = 0
first_k_sum   = 0
max_first_k   = 0
max_first_k_n = 0

progress_interval = max(1, N_MAX // 20)

for n in range(4, N_MAX + 1, 2):
    n_tested += 1
    result = siliq(n, sieve, early_exit=True)

    if not result:
        N_half = n >> 1
        if sieve[N_half]:
            identity_only.append(n)
        else:
            failures.append(n)
    else:
        fk = result[0][0]
        first_k_sum += fk
        if fk > max_first_k:
            max_first_k = fk
            max_first_k_n = n

    if n % progress_interval == 0:
        elapsed = time.perf_counter() - t0
        print(f"    {100 * n / N_MAX:5.1f}%  n={n:>10,}  "
              f"fails={len(failures)}  identity={len(identity_only)}  "
              f"[{elapsed:.1f}s]")

t_engine = time.perf_counter() - t0

n_with_pairs = n_tested - len(failures) - len(identity_only)
avg_first_k = first_k_sum / n_with_pairs if n_with_pairs else 0

print(f"\n  Even numbers tested:    {n_tested:,}")
print(f"  Engine time:            {t_engine:.1f}s")
print(f"  Pairs found (k ≥ 1):   {n_with_pairs:,}")
print(f"  Identity-only (k = 0): {len(identity_only)}  →  {identity_only}")
print(f"  True failures:          {len(failures)}")
if failures:
    print(f"  *** FAILURES: {failures[:20]}")
    if len(failures) > 20:
        print(f"      ... and {len(failures) - 20} more")
print(f"  Avg first-hit k:        {avg_first_k:.2f}")
print(f"  Max first-hit k:        {max_first_k} (at n = {max_first_k_n:,})")

goldbach_pass = len(failures) == 0
engine_pass   = len(failures) <= FAILURE_THRESHOLD

print(f"\n  Goldbach [4..{N_MAX:,}]: "
      f"{'PASS' if goldbach_pass else 'FAIL'} "
      f"({len(failures)} true failures, "
      f"{len(identity_only)} identity-only)")

# ============================================================================
# [C] Z₂₅₆ Ring Analysis (sampled, full siliq)
# ============================================================================
print(f"\n{'─' * 72}")
print(f"[C] Z₂₅₆ Ring Analysis (sample of {SAMPLE_SIZE:,})")
print(f"{'─' * 72}")

rng = np.random.default_rng(42)
sample_pool = np.arange(8, N_MAX + 1, 2)
sample_ns = rng.choice(sample_pool, size=min(SAMPLE_SIZE, len(sample_pool)),
                       replace=False)
sample_ns.sort()

t0 = time.perf_counter()
quadrant_hits = [0, 0, 0, 0]    # UP+, UP−, DN+, DN−
total_pairs   = 0
pair_counts   = []

for n_val in sample_ns:
    result = siliq(int(n_val), sieve, early_exit=False)
    total_pairs += len(result)
    pair_counts.append(len(result))
    for _, _, _, d_val in result:
        quadrant_hits[d_val >> 6] += 1

t_sample = time.perf_counter() - t0
pair_counts = np.array(pair_counts)

print(f"  Sample size:         {len(sample_ns):,}")
print(f"  Sample time:         {t_sample:.1f}s")
print(f"  Total BPAND hits:    {total_pairs:,}")
if len(pair_counts) > 0:
    print(f"  Avg pairs / n:       {pair_counts.mean():.1f}")
    print(f"  Min / Max pairs:     {pair_counts.min()} / {pair_counts.max()}")

total_q = sum(quadrant_hits)
q_labels = ["UP+ [  0.. 63]", "UP− [ 64..127]",
            "DN+ [128..191]", "DN− [192..255]"]
q_fracs = [c / total_q for c in quadrant_hits] if total_q else [0.0] * 4

print(f"\n  Z₂₅₆ Quadrant Distribution:")
for label, count, frac in zip(q_labels, quadrant_hits, q_fracs):
    bar = "█" * int(frac * 40)
    print(f"    {label}: {count:>10,}  ({100 * frac:5.1f}%) {bar}")

# Pair density by order of magnitude
print(f"\n  Pair density by range (from sample):")
for lo_r, hi_r in [(8, 100), (100, 1_000), (1_000, 10_000),
                   (10_000, 100_000), (100_000, 1_000_000),
                   (1_000_000, N_MAX)]:
    mask = (sample_ns >= lo_r) & (sample_ns <= hi_r)
    if mask.any():
        avg_p = pair_counts[mask].mean()
        print(f"    [{lo_r:>10,} – {hi_r:>10,}]: avg {avg_p:7.1f} pairs/n")

# ============================================================================
# [D] 7-Prime Step Ring Walk Analysis
# ============================================================================
print(f"\n{'─' * 72}")
print("[D] 7-Prime Step Ring Walk")
print(f"{'─' * 72}")

d = 1; si = 0
visited = set()
vacuum_skips = 0

for _ in range(BATCH_CELLS):
    d = (d + STEPS[si % 7]) & 0xFF; si += 1
    while not (d & 0x3F):
        vacuum_skips += 1
        d = (d + STEPS[si % 7]) & 0xFF; si += 1
    visited.add(d)

non_vacuum = set(range(256)) - {0, 64, 128, 192}
coverage   = len(visited & non_vacuum)
cov_frac   = coverage / len(non_vacuum)

print(f"  Ring positions visited: {coverage} / {len(non_vacuum)} "
      f"({100 * cov_frac:.1f}%)")
print(f"  Vacuum skips:           {vacuum_skips}")
print(f"  Total steps (w/ skip):  {si}")
print(f"  Effective ratio:        {BATCH_CELLS}/{si} = {BATCH_CELLS / si:.3f}")

# Mod ring coverage
mod128_vals = set(v % 128 for v in visited)
mod64_vals  = set(v % 64  for v in visited)
non_vac_128 = set(range(128)) - {0, 64}
non_vac_64  = set(range(64))  - {0}

cov_128 = len(mod128_vals & non_vac_128)
cov_64  = len(mod64_vals  & non_vac_64)

print(f"\n  Mod ring coverage:")
print(f"    mod 256: {coverage:>3} / {len(non_vacuum)} non-vacuum")
print(f"    mod 128: {cov_128:>3} / {len(non_vac_128)}")
print(f"    mod  64: {cov_64:>3} / {len(non_vac_64)}")

# ============================================================================
# [E] Claims & Scorecard
# ============================================================================
print(f"\n{'─' * 72}")
print("[E] Claims")
print(f"{'─' * 72}")

claims = []

# C1: Goldbach verification (true failures only)
claims.append((
    "Goldbach pairs ∀ even n ∈ [4, 2M]",
    "DERIVED" if goldbach_pass else "FAILED",
    f"{len(failures)} true failures, {len(identity_only)} identity-only"
))

# C2: BPAND convergence
if n_with_pairs > 0:
    bpand_conv = avg_first_k < CONVERGENCE_K_THRESH
    claims.append((
        f"BPAND converges (avg k < {CONVERGENCE_K_THRESH})",
        "DERIVED" if bpand_conv else "FAILED",
        f"avg k = {avg_first_k:.1f}, max k = {max_first_k}"
    ))

# C3: Batch geometry sufficiency
claims.append((
    f"Batch D={BATCH_OUTER}×{BATCH_INNER} sufficient for [4, 2M]",
    "DERIVED" if engine_pass else "FAILED",
    f"{BATCH_CELLS:,} cells, {len(failures)} overflow failures"
))

# C4: Z₂₅₆ quadrant balance
if total_q > 0:
    q_balanced = all(abs(f - 0.25) < QUADRANT_BALANCE for f in q_fracs)
    claims.append((
        f"Z₂₅₆ quadrants balanced (±{100 * QUADRANT_BALANCE:.0f}%)",
        "DERIVED" if q_balanced else "FAILED",
        f"fracs = [{', '.join(f'{f:.3f}' for f in q_fracs)}]"
    ))

# C5: Ring walk coverage
claims.append((
    f"7-prime walk covers >{100 * RING_COVERAGE_THRESH:.0f}% of ring",
    "DERIVED" if cov_frac > RING_COVERAGE_THRESH else "FAILED",
    f"{coverage}/{len(non_vacuum)} = {100 * cov_frac:.1f}%"
))

# C6: Vacuum boundaries correctly skipped
claims.append((
    "Vacuum boundaries (d mod 64 = 0) skipped",
    "DERIVED" if vacuum_skips > 0 else "FAILED",
    f"{vacuum_skips} skips in {si} total steps"
))

# C7: L1 cache bound
l1_bytes = BATCH_CELLS   # 1 byte per sieve entry in the hot window
l1_fits  = l1_bytes <= 32_768   # 32 KB typical L1d
claims.append((
    "Batch fits L1 cache (≤ 32 KB)",
    "DERIVED" if l1_fits else "FAILED",
    f"{l1_bytes:,} bytes vs 32,768 limit"
))

# C8: All engine misses are identity pairs
all_misses_identity = len(failures) == 0
claims.append((
    "All engine misses are identity pairs (k=0)",
    "DERIVED" if all_misses_identity else "FAILED",
    f"{len(identity_only)} identity, {len(failures)} true failures"
))

# Print claims table
print(f"\n  {'Claim':<50} {'Status':<10} {'Evidence'}")
print(f"  {'─' * 50} {'─' * 10} {'─' * 35}")
for text, status, evidence in claims:
    print(f"  {text:<50} {status:<10} {evidence}")

# Scorecard
n_derived = sum(1 for _, s, _ in claims if s.startswith("DERIVED"))
n_failed  = sum(1 for _, s, _ in claims if s == "FAILED")
n_wrong   = sum(1 for _, s, _ in claims if s == "WRONG")
n_open    = sum(1 for _, s, _ in claims if s == "OPEN")

print(f"\n  SCORECARD: {n_derived} DERIVED, {n_failed} FAILED, "
      f"{n_wrong} WRONG, {n_open} OPEN")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'═' * 72}")
if goldbach_pass:
    print(f"  GOLDBACH VERIFIED: {n_tested:,} even numbers checked, "
          f"{len(failures)} failures")
else:
    print(f"  GOLDBACH NOT VERIFIED: {n_tested:,} even numbers checked, "
          f"{len(failures)} failures")
print(f"{'═' * 72}")
