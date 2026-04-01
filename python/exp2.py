"""
exp2.py — Z₂₅₆ Spin-Parity Physics: Angular Momentum Conservation in Goldbach Pairs

PURPOSE:
    Test three falsifiable physical claims derived from the handwritten
    schematic analysis of the biopod Z₂₅₆ structure:

    CLAIM 1 — Quadrant 4-Cycle Structure:
        The coprime walk (step=7, gcd(7,64)=1) treats vacuum boundaries
        (d mod 64 = 0) as quadrant transition operators. Since 7 is coprime
        to the quadrant size 64, the walk visits all residues mod 64
        deterministically, crossing each vacuum boundary at regular
        intervals. The 4-cycle UP+→UP-→DN+→DN- should be deterministic.
        If the walk crosses boundaries randomly, the quadrant
        transitions have no structure — the claim FAILS.

    CLAIM 2 — Biopod Conjugate Mirror Symmetry:
        For a BPAND hit at (lo, hi, d), the d-value encodes the spin-parity
        state of the matching pair. If the biopod geometry enforces U/U'
        mirror symmetry, then for each hit in the UP quadrants (d < 128),
        there should be a statistically equal density of hits in the DOWN
        quadrants (d ≥ 128). The UP-DOWN ratio should be 1.00 ± tolerance.

    CLAIM 3 — Angular Momentum Cancellation:
        If {U, U'} ∝ angular momentum, then BPAND pairs should preferentially
        lock into spin-cancelling configurations. Define:
          - Same-spin pair:  both d values in same spin sector (UP-UP or DN-DN)
          - Anti-spin pair:  d values in opposite spin sectors (UP-DN)
        If angular momentum conservation drives pair selection, anti-spin
        configurations should dominate. If not, spin should be uniform.

        Additionally: within each BPAND hit, the lo-arm and hi-arm carry
        opposite displacement from center (−k and +k). Test whether the
        spin state encoded in d at the first hit vs last hit shows net
        cancellation across the full pair set for a given n.

METHOD:
    [A] Enumerate full siliq results for a large sample of even n
    [B] Track every vacuum boundary crossing in the 7-prime walk:
        record (d_before, d_after) and classify the bit-flip
    [C] Measure UP/DOWN balance across all BPAND hits (Claim 2)
    [D] Classify each BPAND hit's d-value by spin-parity quadrant;
        measure same-spin vs anti-spin pairing across consecutive hits (Claim 3)
    [E] Compute angular momentum proxy: L_net = Σ spin(d_i) across all
        hits for each n, test whether |L_net| / n_hits → 0 (cancellation)

WHAT THIS EXPERIMENT DERIVES:
    ✓ Vacuum crossing bit-flip structure (ordered or random)
    ✓ UP/DOWN mirror ratio across BPAND hits
    ✓ Same-spin vs anti-spin dominance in pair selection
    ✓ Net angular momentum cancellation metric
    ✓ Quadrant transition matrix (which quadrant flows to which)

WHAT REMAINS FREE:
    ✗ The identification d ↔ quantum state is POSTULATED, not derived
    ✗ "Angular momentum" is a label — the conserved quantity is spin(d)
    ✗ The connection to thermodynamic stability is CLAIMED, not proven
    ✗ No multi-node dynamics — this is single-ring analysis

REQUIRES: numpy
PRECISION: integer arithmetic for engine, float64 for statistics

NO HARDCODED RESULTS. THE EXPERIMENT CAN FAIL.
"""

import numpy as np
import time

# ============================================================================
# Thresholds — defined BEFORE seeing results
# ============================================================================
N_MAX                = 2_000_000   # scan range
SAMPLE_SIZE          = 20_000      # full siliq runs for physics analysis
UD_MIRROR_TOLERANCE  = 0.05        # UP/DOWN ratio must be 1.00 ± 5%
ANTI_SPIN_THRESHOLD  = 0.50        # anti-spin fraction must exceed this
                                   # (0.50 = no preference; > 0.50 = dominance)
L_CANCEL_THRESHOLD   = 0.30        # |L_net|/n_hits must be below this
                                   # (0 = perfect cancellation, 1 = all same spin)
FLIP_STRUCTURE_THRESH = 0.90       # >90% of vacuum crossings must follow
                                   # the deterministic 4-cycle (UP+→UP-→DN+→DN-)

# ============================================================================
# Constants
# ============================================================================
STEP        = 7              # gcd(7, 64) = 1 — coprime to quadrant size
                             # guarantees full Z₂₅₆ coverage (visits all 256
                             # residues) and uniform vacuum boundary crossings
BATCH_OUTER = 128
BATCH_INNER = 113
BATCH_CELLS = BATCH_OUTER * BATCH_INNER

# ============================================================================
print("=" * 72)
print("exp2.py — Z₂₅₆ Spin-Parity Physics")
print("         Angular Momentum Conservation in Goldbach Pairs")
print("=" * 72)
print(f"\n  N_MAX              = {N_MAX:,}")
print(f"  Sample size        = {SAMPLE_SIZE:,}")
print(f"  UD mirror tol      = ±{100*UD_MIRROR_TOLERANCE:.0f}%")
print(f"  Anti-spin thresh   = {ANTI_SPIN_THRESHOLD}")
print(f"  L cancel thresh    = {L_CANCEL_THRESHOLD}")
print(f"  Flip structure     = >{100*FLIP_STRUCTURE_THRESH:.0f}%")

# ============================================================================
# Prime Sieve
# ============================================================================
sieve = np.ones(N_MAX + 1, dtype=np.uint8)
sieve[0] = sieve[1] = 0
for i in range(2, int(N_MAX**0.5) + 1):
    if sieve[i]:
        sieve[i*i::i] = 0

# ============================================================================
# Spin-Parity Helpers
# ============================================================================
def spin(d):
    """Spin quantum number from d: UP (+1) if bit7=0, DOWN (-1) if bit7=1."""
    return +1 if (d >> 7) & 1 == 0 else -1

def parity(d):
    """Parity quantum number from d: +1 if bit6=0, -1 if bit6=1."""
    return +1 if (d >> 6) & 1 == 0 else -1

def quadrant(d):
    """Quadrant index from bits 7:6. Returns 0-3."""
    return (d >> 6) & 0x3

def quadrant_label(q):
    return ["UP+", "UP-", "DN+", "DN-"][q]

# ============================================================================
# Full Siliq with d-tracking
# ============================================================================
def siliq_full(n, S):
    """Returns list of (k, lo, hi, d) for ALL BPAND hits."""
    N = n >> 1; pairs = []; d = 1; k = 0; stop = False
    for outer in range(BATCH_OUTER):
        if stop: break
        for inner in range(BATCH_INNER):
            k += 1
            if k >= N: stop = True; break
            d = (d + STEP) & 0xFF
            while not (d & 0x3F): d = (d + STEP) & 0xFF
            lo = N - k; hi = N + k
            if lo < 2: stop = True; break
            if hi < len(S) and S[lo] & S[hi]:
                pairs.append((k, lo, hi, d))
    return pairs

# ============================================================================
# [A] Vacuum × Step Analysis: coprime walk gcd(7,64)=1
# ============================================================================
print(f"\n{'─' * 72}")
print(f"[A] Vacuum Structure: coprime step={STEP}, gcd({STEP},64)=1")
print(f"{'─' * 72}")

VACUUM_LABELS = {0: "V₀", 64: "V₆₄", 128: "V₁₂₈", 192: "V₁₉₂"}
VACUUMS = [0, 64, 128, 192]

from math import gcd
assert gcd(STEP, 64) == 1, f"STEP={STEP} not coprime to 64!"
print(f"\n  gcd({STEP}, 64) = {gcd(STEP, 64)} — walk visits all residues mod 64")
print(f"  gcd({STEP}, 256) = {gcd(STEP, 256)} — walk visits all 256 values in Z₂₅₆")

# Walk the FULL ring and log every vacuum crossing
d = 1
crossings = []

for _ in range(BATCH_CELLS):
    d_before = d
    d = (d + STEP) & 0xFF
    while not (d & 0x3F):
        # We hit a vacuum boundary
        d_at_vacuum = d       # which vacuum: 0, 64, 128, or 192
        q_before = quadrant(d_before)

        d = (d + STEP) & 0xFF
        q_after = quadrant(d)

        # Which bits flipped between entry and exit?
        bit7_flip = ((d_before >> 7) & 1) != ((d >> 7) & 1)
        bit6_flip = ((d_before >> 6) & 1) != ((d >> 6) & 1)
        flip_count = int(bit7_flip) + int(bit6_flip)

        crossings.append({
            'd_pre': d_before, 'd_post': d,
            'vacuum': d_at_vacuum,
            'q_pre': q_before, 'q_post': q_after,
            'bit7_flip': bit7_flip, 'bit6_flip': bit6_flip,
            'flip_count': flip_count
        })
        d_before = d  # for chained skips

n_crossings = len(crossings)

# ── Per-vacuum crossing counts ──
print(f"\n  Total vacuum crossings: {n_crossings}")
print(f"\n  Per-Vacuum Crossing Counts:")
for v in VACUUMS:
    cnt = sum(1 for c in crossings if c['vacuum'] == v)
    print(f"    {VACUUM_LABELS[v]}: {cnt:>5}")

# Per-vacuum quadrant transition
print(f"\n  Per-Vacuum Quadrant Transitions:")
for v in VACUUMS:
    v_crossings = [c for c in crossings if c['vacuum'] == v]
    if not v_crossings:
        continue
    trans_v = {}
    for c in v_crossings:
        key = (quadrant_label(c['q_pre']), quadrant_label(c['q_post']))
        trans_v[key] = trans_v.get(key, 0) + 1
    print(f"    {VACUUM_LABELS[v]}:  ", end="")
    for (qp, qa), cnt in sorted(trans_v.items()):
        print(f"{qp}→{qa}:{cnt}  ", end="")
    print()

# Bit-flip breakdown
flip_0 = sum(1 for c in crossings if c['flip_count'] == 0)
flip_1 = sum(1 for c in crossings if c['flip_count'] == 1)
flip_2 = sum(1 for c in crossings if c['flip_count'] == 2)
spin_only = sum(1 for c in crossings if c['bit7_flip'] and not c['bit6_flip'])
parity_only = sum(1 for c in crossings if c['bit6_flip'] and not c['bit7_flip'])

print(f"\n  Bit-flip breakdown:")
print(f"    0-bit (same Q):     {flip_0:>5} ({100*flip_0/n_crossings:.1f}%)")
print(f"    1-bit (parity):     {parity_only:>5} ({100*parity_only/n_crossings:.1f}%)")
print(f"    1-bit (spin):       {spin_only:>5} ({100*spin_only/n_crossings:.1f}%)")
print(f"    2-bit (both):       {flip_2:>5} ({100*flip_2/n_crossings:.1f}%)")

# The KEY test: is the quadrant cycle deterministic?
# Expected: UP+→UP-→DN+→DN-→UP+ always (perfect 4-cycle)
expected_next = {0: 1, 1: 2, 2: 3, 3: 0}  # q → q+1 mod 4
cycle_hits = sum(1 for c in crossings if c['q_post'] == expected_next[c['q_pre']])
cycle_frac = cycle_hits / n_crossings if n_crossings else 0

print(f"\n  Deterministic 4-cycle test:")
print(f"    Crossings following UP+→UP-→DN+→DN-→UP+: "
      f"{cycle_hits}/{n_crossings} ({100*cycle_frac:.1f}%)")

# Full transition matrix
trans = np.zeros((4, 4), dtype=int)
for c in crossings:
    trans[c['q_pre'], c['q_post']] += 1

print(f"\n  Quadrant Transition Matrix (all crossings):")
print(f"         {'  '.join(quadrant_label(j) for j in range(4))}")
for i in range(4):
    row = '  '.join(f'{trans[i,j]:>4}' for j in range(4))
    print(f"    {quadrant_label(i)}: {row}")

# ============================================================================
# [B] Full Sample: Collect BPAND data
# ============================================================================
print(f"\n{'─' * 72}")
print(f"[B] Sampling {SAMPLE_SIZE:,} even numbers for spin-parity analysis")
print(f"{'─' * 72}")

rng = np.random.default_rng(42)
sample_pool = np.arange(8, N_MAX + 1, 2)
sample_ns = rng.choice(sample_pool, size=min(SAMPLE_SIZE, len(sample_pool)),
                       replace=False)
sample_ns.sort()

t0 = time.perf_counter()

# Accumulators
all_d_values        = []    # every d from every BPAND hit
all_spins           = []    # spin(d) for every hit
all_parities        = []    # parity(d) for every hit
per_n_L_ratios      = []    # |L_net|/n_hits per n
per_n_anti_fracs    = []    # anti-spin fraction per n (consecutive pair analysis)
up_count = 0
dn_count = 0
quad_counts = [0, 0, 0, 0]

for n_val in sample_ns:
    pairs = siliq_full(int(n_val), sieve)
    if len(pairs) < 2:
        continue

    d_vals = [p[3] for p in pairs]
    spins  = [spin(d) for d in d_vals]

    for d in d_vals:
        all_d_values.append(d)
        s = spin(d)
        all_spins.append(s)
        all_parities.append(parity(d))
        quad_counts[quadrant(d)] += 1
        if s > 0:
            up_count += 1
        else:
            dn_count += 1

    # Net angular momentum for this n
    L_net = abs(sum(spins))
    L_ratio = L_net / len(spins)
    per_n_L_ratios.append(L_ratio)

    # Same-spin vs anti-spin (consecutive hit pairs)
    anti = 0
    same = 0
    for i in range(len(spins) - 1):
        if spins[i] * spins[i+1] < 0:
            anti += 1
        else:
            same += 1
    if anti + same > 0:
        per_n_anti_fracs.append(anti / (anti + same))

t_sample = time.perf_counter() - t0
total_hits = len(all_d_values)

print(f"  Sample time:           {t_sample:.1f}s")
print(f"  Total BPAND hits:      {total_hits:,}")
print(f"  Even numbers with ≥2 hits: {len(per_n_L_ratios):,}")

# ============================================================================
# [C] Claim 2: UP/DOWN Mirror Symmetry
# ============================================================================
print(f"\n{'─' * 72}")
print("[C] Claim 2: Biopod UP/DOWN Mirror Symmetry")
print(f"{'─' * 72}")

total_ud = up_count + dn_count
up_frac = up_count / total_ud if total_ud else 0
dn_frac = dn_count / total_ud if total_ud else 0
ud_ratio = up_count / dn_count if dn_count else float('inf')
ud_dev = abs(ud_ratio - 1.0)

print(f"  UP  hits (d < 128):  {up_count:>10,} ({100*up_frac:.2f}%)")
print(f"  DN  hits (d ≥ 128):  {dn_count:>10,} ({100*dn_frac:.2f}%)")
print(f"  UP/DN ratio:         {ud_ratio:.4f}")
print(f"  Deviation from 1.0:  {ud_dev:.4f}")

print(f"\n  Quadrant breakdown:")
for q in range(4):
    frac = quad_counts[q] / total_hits if total_hits else 0
    bar = "█" * int(frac * 40)
    print(f"    {quadrant_label(q)}: {quad_counts[q]:>10,} ({100*frac:5.1f}%) {bar}")

mirror_pass = ud_dev < UD_MIRROR_TOLERANCE

# ============================================================================
# [D] Claim 3: Angular Momentum — Anti-Spin Dominance
# ============================================================================
print(f"\n{'─' * 72}")
print("[D] Claim 3: Angular Momentum — Spin Cancellation")
print(f"{'─' * 72}")

# D.1: Anti-spin fraction across consecutive BPAND hits
anti_arr = np.array(per_n_anti_fracs)
mean_anti = anti_arr.mean() if len(anti_arr) else 0
std_anti  = anti_arr.std()  if len(anti_arr) else 0

print(f"\n  [D.1] Consecutive-hit anti-spin fraction:")
print(f"    Mean anti-spin:  {mean_anti:.4f}")
print(f"    Std:             {std_anti:.4f}")
print(f"    Threshold:       > {ANTI_SPIN_THRESHOLD}")
anti_pass = mean_anti > ANTI_SPIN_THRESHOLD

if anti_pass:
    print(f"    Result: PASS — anti-spin dominates ({mean_anti:.4f} > {ANTI_SPIN_THRESHOLD})")
else:
    print(f"    Result: FAIL — anti-spin does NOT dominate ({mean_anti:.4f} ≤ {ANTI_SPIN_THRESHOLD})")

# D.2: Net angular momentum cancellation
L_arr = np.array(per_n_L_ratios)
mean_L = L_arr.mean() if len(L_arr) else 1.0
std_L  = L_arr.std()  if len(L_arr) else 0

print(f"\n  [D.2] Net angular momentum |L|/N_hits per even n:")
print(f"    Mean |L|/N:   {mean_L:.4f}")
print(f"    Std:          {std_L:.4f}")
print(f"    Threshold:    < {L_CANCEL_THRESHOLD}")
cancel_pass = mean_L < L_CANCEL_THRESHOLD

if cancel_pass:
    print(f"    Result: PASS — net L cancels ({mean_L:.4f} < {L_CANCEL_THRESHOLD})")
else:
    print(f"    Result: FAIL — net L does NOT cancel ({mean_L:.4f} ≥ {L_CANCEL_THRESHOLD})")

# D.3: Global spin balance
global_L = sum(all_spins)
global_L_ratio = abs(global_L) / total_hits if total_hits else 1.0
print(f"\n  [D.3] Global spin sum:")
print(f"    Σspin:         {global_L:+,}")
print(f"    |Σspin|/N:     {global_L_ratio:.6f}")
print(f"    (0 = perfect cancellation)")

# ============================================================================
# [E] Claims & Scorecard
# ============================================================================
print(f"\n{'─' * 72}")
print("[E] Claims")
print(f"{'─' * 72}")

claims = []
# C1: Deterministic 4-cycle at vacuum crossings
claims.append((
    "Vacuum crossings follow 4-cycle (>90%)",
    "DERIVED" if cycle_frac > FLIP_STRUCTURE_THRESH else "FAILED",
    f"{100*cycle_frac:.1f}% follow UP+→UP-→DN+→DN- (thresh >{100*FLIP_STRUCTURE_THRESH:.0f}%)"
))

# C2: UP/DOWN mirror symmetry
claims.append((
    "UP/DOWN mirror ratio = 1.00 ± 5%",
    "DERIVED" if mirror_pass else "FAILED",
    f"ratio = {ud_ratio:.4f}, dev = {ud_dev:.4f}"
))

# C3a: Anti-spin dominance
claims.append((
    "Anti-spin pairs dominate (>50%)",
    "DERIVED" if anti_pass else "FAILED",
    f"mean = {mean_anti:.4f} ± {std_anti:.4f}"
))

# C3b: Net L cancellation
claims.append((
    "Net |L|/N → 0 (angular momentum cancels)",
    "DERIVED" if cancel_pass else "FAILED",
    f"mean = {mean_L:.4f} ± {std_L:.4f}"
))

# C4: Global spin neutrality
global_neutral = global_L_ratio < 0.01  # within 1% of zero
claims.append((
    "Global spin sum neutral (|Σ|/N < 1%)",
    "DERIVED" if global_neutral else "FAILED",
    f"|Σspin|/N = {global_L_ratio:.6f}"
))

# C5: Quadrant balance (all 4 within 15% of 25%)
q_fracs = [c / total_hits for c in quad_counts] if total_hits else [0]*4
q_balanced = all(abs(f - 0.25) < 0.15 for f in q_fracs)
claims.append((
    "All 4 quadrants balanced (±15%)",
    "DERIVED" if q_balanced else "FAILED",
    f"[{', '.join(f'{f:.3f}' for f in q_fracs)}]"
))

# Print claims table
print(f"\n  {'Claim':<50} {'Status':<10} {'Evidence'}")
print(f"  {'─' * 50} {'─' * 10} {'─' * 40}")
for text, status, evidence in claims:
    print(f"  {text:<50} {status:<10} {evidence}")

# Scorecard
n_derived = sum(1 for _, s, _ in claims if s.startswith("DERIVED"))
n_failed  = sum(1 for _, s, _ in claims if s == "FAILED")
n_wrong   = sum(1 for _, s, _ in claims if s == "WRONG")

print(f"\n  SCORECARD: {n_derived} DERIVED, {n_failed} FAILED, {n_wrong} WRONG")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'═' * 72}")
main_pass = mirror_pass and cancel_pass
if main_pass:
    print(f"  SPIN-PARITY PHYSICS: CONFIRMED")
    print(f"    Mirror symmetry holds, angular momentum cancels.")
else:
    reasons = []
    if not mirror_pass:  reasons.append(f"mirror dev={ud_dev:.4f}")
    if not cancel_pass:  reasons.append(f"|L|/N={mean_L:.4f}")
    if not anti_pass:    reasons.append(f"anti-spin={mean_anti:.4f}")
    print(f"  SPIN-PARITY PHYSICS: NOT CONFIRMED")
    print(f"    Failed: {', '.join(reasons)}")
print(f"{'═' * 72}")
