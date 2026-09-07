# Calibration thresholds are not robustness of observational equivalence

Status: **internal finite-model diagnostic** following `replication_switch_audit.md`.
No change to the active MEE manuscript, frozen G2 results/figures, submission
manifests, public exports or production information objectives. The implementation
reuses the existing noisy-likelihood scorer and replication-information profile.
This is a scoped sensitivity calculation, not a new robust-design optimizer.

## 1. Keep the uncertainty family explicit

The first audit varies only the alternative observation's independent symmetric
binary error e in [0,1/2]. Its ideal per-world bit X, the declared question target T,
current data D, current weights and old repeat protocol are fixed. X need not be a
function of T; within-target variation in X is permitted. The whole full-world
ensemble and one fixed target are shared across every candidate and error value.

The resulting observation is Q_e = X XOR E_e, where E_e is an independent
Bernoulli(e) bit. This is a declared prediction family, not fitted calibration.
In particular, false-positive and false-negative rates are assumed equal, and
persistent measurement error or history dependence is not inferred or removed.
A calibration reference string is provenance, not verification of these conditions.

If changing a calibration parameter also changes the interpretation of past D,
recompute current weights under that parameter. Holding the posterior fixed in
that case would answer a different question. The second example below does
recondition the past contact-positive result under each changed old law.

## 2. Endpoint bounds over an entire interval, not a sampled grid

For 0 <= e1 <= e2 <= 1/2 and e1 < 1/2, set

```text
r = (e2-e1)/(1-2*e1).
```

Then 0 <= r <= 1/2 and a BSC(e1) followed by independent BSC(r) has the BSC(e2)
law. Consequently T -> Q_e1 -> Q_e2 can be coupled as a Markov chain, and

```text
I(T;Q_e2 | D) <= I(T;Q_e1 | D).
```

The e1=e2=1/2 case is trivial. Hence the information at the upper error endpoint
is the worst case over the closed interval, and the lower endpoint is the best.
This holds for the declared binary-error family, not arbitrary calibration paths.

For each fixed benchmark -- one repeat or the entire repeat-only ceiling -- the
code reports worst/best information margins and numerical equality brackets.
Positive margin at the worst endpoint implies superiority throughout the interval
under this monotonicity theorem. The numerical implementation uses ordinary
floating-point arithmetic, an explicit bit tolerance, and bisection width. It is
not rigorous interval arithmetic or a confidence interval estimated from data.
A near-tie or unbracketed numerical improvement is never reported as exact repair.
Missing old-protocol predictions leave both comparisons non-estimable, not zero.

## 3. Two different breakpoints in the existing three-program example

After one contact-positive observation, the synthetic program weights are
(9/19,1/19,9/19), for pollination-only, abiotic-only and combined. Old contact
probabilities are (0.9,0.1,0.9), while the alternative's ideal bits are (0,1,1).
Only the alternative's independent error varies in this first calculation.

The next contact reading supplies 0.120730268962302 bits, and the entire remaining
contact-repeat ceiling is 0.297472248919290 bits. Write h2 for binary entropy.
Because the alternative ideal bit is a function of the three-program target,

```text
I(T;Q_e | D) = h2(e + (1-2e)*10/19) - h2(e).
```

The two equality points, independently checked with 60-digit Decimal arithmetic,
are approximately

```text
e_ceiling = 0.190138065920987
e_next    = 0.298090224610244.
```

Below e_ceiling the alternative beats every finite repeat-only transcript in
expected additional information. Between these thresholds it beats the next
repeat without beating the unlimited-repeat ceiling. Above e_next the next
repeat is preferable in this two-candidate comparison; equality gives a tie.
These are absolute error probabilities, not percentage drift of a fitted estimate.

For example, throughout e in [0.10,0.18] both comparisons favor switching. Over
[0.10,0.20], only next-step superiority is uniform. Over [0.20,0.30], even the
next-step comparison crosses. No global-best, cost, intervention, adaptation or
realised-outcome guarantee follows from these expected-information comparisons.

## 4. A different issue: the repeat ceiling relies on an exact equivalence

The old-law class in #135/#136 groups pollination-only and combined because they
have exactly the same contact probability, 0.9. That equality must come from a
justified model restriction, not merely two fitted values that look close.

Change ONLY the combined contact probability to 0.9+delta, where 0<delta<0.1,
and condition the same observed contact-positive result again. The current
weights become

```text
(0.9, 0.1, 0.9+delta)/(1.9+delta).
```

For every strictly positive delta in this range, all three per-world contact
laws differ. The asymptotic repeat information then equals H_delta(T|D): the
old residual floor is zero. The noisy alternative with fixed error 0.1 has
strictly positive likelihood for both outcomes in every program, so its
information is strictly below H_delta(T|D). It no longer beats unlimited repeats.
No finite entropy rounding is used to determine this law-class structure.

Yet at delta=1e-6 its advantage over the next repeat is about 0.408994676 bits,
and over 50 repeats about 0.232253027 bits. A failed unlimited-repeat comparison
therefore does not imply that repeating is good use of a finite budget.

For fixed finite n the information depends continuously on delta, whereas the
infinite-repeat limit jumps at the exact law collision. Here the limits do not
commute: taking infinite repeats before delta->0 gives H_0(T|D), while taking
delta->0 before infinite repeats gives I_0(T;C|D). Thus the unrestricted
law-splitting perturbation family has no positive robustness radius for the
strong unlimited-repeat dominance claim. A restricted family that preserves
the equality is a different sensitivity question and can retain that claim.

Each sensitivity scenario treats its law as fixed and known for the calculation.
We do NOT marginalize an unknown delta or establish identifiability in a larger
unknown-calibration model. Near-equal rates are never merged by a tolerance.

## 5. Recover a uniform finite-budget guarantee without exact equality

This is an analytic perturbation bound, not interpolation of sampled deltas.
For distributions P,Q on a common target alphabet of size k and any observation,
let eta be their joint total-variation distance. The common-part mixture
P=(1-eta)R+eta P', Q=(1-eta)R+eta Q' gives

```text
|H_P(T|Y)-H_Q(T|Y)| <= eta*log2(k)+h2(eta).
```

Proof: introduce the mixture indicator J. Conditional entropy lies between
H(T|Y,J), the weighted component entropy, and that quantity plus H(J)<=h2(eta).
The component entropies lie in [0,log2(k)]. Apply the same argument to H(T).
Thus, for an upper bound eta<=1/2,

```text
|I_P(T;Y)-I_Q(T;Y)| <= f_k(eta)
f_k(eta) = min(log2(k), 2*[eta*log2(k)+h2(eta)]).
```

The right-hand side is increasing on [0,1/2]. Larger TV bounds use the trivial
log2(k) information-difference bound. This is deliberately conservative.

For delta in [0,d], the posterior weights above differ from their delta=0 values
in TV by delta/[1.9*(1.9+delta)] <= d/1.9. A common-uniform coupling bounds the
probability that n Bernoulli repeats differ by n*d. Therefore

```text
TV(alternative joint laws) <= d/1.9,
TV(n-repeat joint laws) <= min(1, d/1.9+n*d).
```

Using the original remaining-repeat ceiling B_0 to bound ALL baseline budgets,
for every delta in [0,d] and every integer 0<=n<=m,

```text
I_delta(T;Q|D)-I_delta(T;Y_1:n|D)
 >= [I_0(T;Q|D)-B_0] - f_3(d/1.9) - f_3(min(1,d/1.9+m*d)).
```

For d=1e-6 and m=50, the right-hand side is 0.230479560341586 bits. Thus a
strictly positive finite-budget switching advantage survives across this whole
specified interval, even though any delta>0 removes unlimited-repeat dominance.
The number is an ordinary floating evaluation of a proven analytic lower bound,
not a formally outward-rounded machine certificate. A nonpositive lower bound
would mean this conservative argument is inconclusive, not that repeating wins.

## 6. Reproduction, scope and prior art

```bash
python -m examples.calibration_switch_report > calibration_switch_report.json
python -m pytest -q tests/test_calibration_switch_audit.py
```

The independent tests include 1,024 finite binary-map information comparisons,
60-digit root checks, law splitting with past-data reconditioning, and 200
random joint-distribution continuity checks. The local isolated source harness
uses byte-identical existing dependency files; it is not a full repository clone.
Full repository CI remains a separate integration gate.

Robust information design and model-misspecification-aware acquisition already
exist. For example, Go & Isaac (2022), *Robust expected information gain for optimal
Bayesian experimental design using ambiguity sets*, PMLR 180:728-737,
https://proceedings.mlr.press/v180/go22a.html, addresses prior ambiguity. Tang,
Sloman & Kaski (2026), *Representative, Informative, and De-Amplifying: Requirements
for Robust Bayesian Active Learning under Model Misspecification*, PMLR 300:3016-3024,
https://proceedings.mlr.press/v300/tang26d.html, addresses generalization under
misspecification. Neither is claimed to supply this exact witness or this bound.

No invention of channel degradation, continuity inequalities or robust design is
claimed here. The contribution to this repository is narrower: separate a
calibration-sensitive next-step recommendation from a structurally fragile
infinite-repeat claim, then retain a checked finite-budget reporting alternative.
Nothing here demonstrates empirical channel calibration, causality, selection,
fitness effects, natural-system exhaustiveness, or a universally optimal policy.
