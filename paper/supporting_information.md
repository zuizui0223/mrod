# Supporting Information

## Mechanism-Resolving Observation Design: information-theoretic selection of observations under ecological mechanism ambiguity

This Supporting Information accompanies the anonymised Research Article. It expands the admissible-region quantities, observation-information calculation, sequential design, frozen controlled benchmarks and reproducibility map. The separate mechanistic-evidence / identification-boundary Perspective is owned by the `boundary` repository and is not part of this submission.

---

## S1. Admissible mechanism regions and evidence roles

### S1.1 Admissible region

For switches `S in {0,1}^K`, parameters `theta`, pre-data constraint grammar `G`, simulator `f`, pattern maps `P_sim,P_obs`, discrepancy `d`, tolerance `epsilon`, observed targets `y_obs` and fixed context `x_obs`,

```text
A_epsilon
= {(theta,s): G(theta)=1 and
   d(P_sim(f(x_obs;theta,s)),P_obs(y_obs))<=epsilon}.
```

The finite implementation approximates this region by prior sampling and rejection. The retained region, not its modal switch row, is the inferential object.

### S1.2 Evidence-role contract

Every quantity is assigned one role before inference:

| Role | Use |
|---|---|
| `observed_target` | may enter the acceptance discrepancy |
| `input_context` | conditions the simulator but is not an independent target |
| `diagnostic_only` | evaluates inference or software behaviour after fitting |
| `future_observation` | withheld as a candidate measurement |

The same datum may not be silently used as context, acceptance evidence and independent validation.

### S1.3 Mechanism quantities

For switch `j`,

```text
CA_j=P(s_j=1|A_epsilon).
```

Joint mechanism entropy and resolvability are

```text
D=H(S|A_epsilon),
R=1-D/K.
```

A `K`-bit vector has entropy at most `K`, so `0<=D<=K` and `0<=R<=1`. Mechanism-equivalence and replaceability summaries are calculated from the same accepted switch rows and are not substitutes for the joint entropy.

---

## S2. Observation information value and sequential design

### S2.1 Predictive-partition requirement

For a finite candidate observation `Q`, its outcome maps must be mutually exclusive and exhaustive over the current accepted region. The predictive distribution is then the pushforward

```text
Pr(Q=q|A_epsilon).
```

If outcomes overlap, are incomplete or require unavailable simulator columns, validated stored-region observation information value is non-estimable. An external outcome prior is not silently substituted and relabelled as validated information value.

### S2.2 Information identity

Define expected resolvability gain

```text
V(Q)=E_Q[R(A_epsilon|Q)-R(A_epsilon)].
```

Then

```text
V(Q)
={H(S|A_epsilon)-H(S|A_epsilon,Q)}/K
=I(S;Q|A_epsilon)/K.
```

Therefore

```text
0<=V(Q)<=1-R(A_epsilon)<=1.
```

`V(Q)=0` exactly when the candidate measurement carries no information about residual mechanism identity under the current accepted region.

### S2.3 Sequential recomputation

At step `t`, the information-guided policy scores every verified remaining candidate by

```text
V_t(Q)=I(S;Q|A_t)/K.
```

It selects the largest positive current value, obtains the realised outcome only after selection, conditions `A_t` on that outcome, and recomputes all remaining predictive probabilities and information values. The sequence stops when the budget is exhausted, the declared confounding structure is resolved, or all available validated values are zero.

An explicitly named normalized edge-cut fallback is retained only for candidates whose predictive partition is unavailable. Every step records its score source; fallback scores are not reported as validated observation information values.

### S2.4 When recomputation has strict value

Recomputation is not assumed to improve every candidate family. For a two-step finite design, let `X` be the first observation and let `U_q(x)` denote the information value of remaining candidate `q` after branch `X=x`. The adaptive and strongest precommitted-static second-step values are

```text
V_adapt  = E[max_q U_q(X)],
V_static = max_q E[U_q(X)].
```

Then `V_adapt>=V_static`. Equality holds if and only if at least one candidate is branchwise optimal on every positive-probability first-outcome branch. Strict adaptive advantage occurs exactly when the intersection of those branchwise argmax sets is empty. This is a two-step finite-design result; it does not establish global optimality of a full multi-step greedy policy.

---

## S3. Frozen G2 observation-selection benchmark

### S3.1 Protocol and historical labels

The frozen machine-readable protocol retains the historical identifier `rach-g2-truth-peek-free-v2` and stored policy key `rach_seq` to preserve exact provenance. These strings are legacy record identifiers, not the active method name. In this manuscript and current software documentation the policy is called **information-guided sequential design**.

The protocol uses five predeclared seeds, 200 systems per seed, 1,500 prior draws per system, `K in {4,5,6}`, one or two disjoint confounds, random pre-data driver coefficients, two mechanism-independent binary nuisance candidates and budgets 0–4.

The information-guided and `random_order` policies receive identical systems, hidden truths, candidates and budgets. Hidden truth is used only after candidate selection to materialise the chosen outcome. The policy comparison is descriptive and has no favourable-result acceptance threshold.

### S3.2 Policy-specific means

**Table S1. Frozen policy means across five seeds.**

| Policy | Budget | Converged | Initial edges resolved | Mean observations | Mean nuisance selections | False exclusion |
|---|---:|---:|---:|---:|---:|---:|
| information-guided | 0 | 0.000 | 0.0000 | 0.000 | 0.000 | 0.000 |
| information-guided | 1 | 0.495 | 0.7480 | 1.000 | 0.000 | 0.000 |
| information-guided | 2 | 0.990 | 1.0000 | 1.505 | 0.001 | 0.000 |
| information-guided | 3 | 0.997 | 1.0000 | 1.515 | 0.011 | 0.000 |
| information-guided | 4 | 0.999 | 1.0000 | 1.518 | 0.014 | 0.000 |
| random_order | 0 | 0.000 | 0.0000 | 0.000 | 0.000 | 0.000 |
| random_order | 1 | 0.179 | 0.2995 | 1.000 | 0.580 | 0.000 |
| random_order | 2 | 0.435 | 0.6045 | 1.821 | 0.974 | 0.000 |
| random_order | 3 | 0.689 | 0.8650 | 2.386 | 1.152 | 0.000 |
| random_order | 4 | 0.940 | 1.0000 | 2.673 | 1.169 | 0.000 |

At budget two, across-seed sample SDs for the information-guided policy were 0.0079 for convergence, 0 for edge resolution, 0.0302 for observations and 0.00224 for nuisance selections. Random-order SDs were 0.0355, 0.0231, 0.0243 and 0.0277 respectively.

At budget four, the nuisance-selection ratio was

```text
1.169/0.014=83.5,
```

and the relative reduction was

```text
1-0.014/1.169=0.9880.
```

The manuscript reports the absolute counts with the ratio because a fold change is unstable when the denominator approaches zero. All 10,000 system–policy–budget records retained the hidden true explanation.

### S3.3 Post-frozen static initial-information diagnostic

The preregistered G2 comparison above was not changed. To determine whether its guided-versus-random advantage also demonstrated a practical benefit of adaptive recomputation, we subsequently ran a matched diagnostic with a stronger nonadaptive policy. `static_initial_information` ranks candidates once using their information values in the initial admissible region, discards candidates with non-positive initial value, and follows that fixed order without recomputation.

The diagnostic reused the same generator settings, five seeds, 200 systems per seed, hidden truths, candidate vocabularies, nuisance measurements and budgets. It is explicitly **post-frozen and non-preregistered**.

**Table S2. Claim-ceiling diagnostic comparing adaptive and static information ordering.**

| Budget | Policy | Converged | Initial edges resolved | Mean observations | Mean nuisance selections | False exclusion |
|---:|---|---:|---:|---:|---:|---:|
| 2 | information-guided adaptive | 0.990 | 1.0000 | 1.505 | 0.001 | 0.000 |
| 2 | static initial information | 0.990 | 1.0000 | 1.505 | 0.001 | 0.000 |
| 2 | random order | 0.435 | 0.6045 | 1.821 | 0.974 | 0.000 |
| 4 | information-guided adaptive | 0.999 | 1.0000 | 1.518 | 0.014 | 0.000 |
| 4 | static initial information | 0.998 | 1.0000 | 1.518 | 0.014 | 0.000 |
| 4 | random order | 0.940 | 1.0000 | 2.673 | 1.169 | 0.000 |

The two information-based policies are essentially indistinguishable on this family. The frozen G2 evidence therefore supports **information-guided candidate screening** much more strongly than an empirical performance gain from adaptive recomputation. The separate theorem in S2.4 specifies when branch-dependent changes in the best remaining measurement make recomputation strictly valuable.

---

## S4. Auxiliary controlled checks

### S4.1 Known-truth self-consistency

Under unchanged defaults, mean switch-state accuracy in the zero pattern-noise stratum was 0.6562 and recall of applicable true-ON switches was 1.000. Recall remained 1.000 in the 0.1 and 0.2 pattern-noise strata. Additional confounded explanations were allowed to survive, so exact-state accuracy was not expected to equal one.

**Table S3. Known-truth aggregate results.**

| Pattern noise | Cases | Accuracy | Precision | Recall | F1 | Mean admissibility error | R | D |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 8 | 0.6562 | 0.4792 | 1.0000 | 0.6042 | 0.3388 | 0.3699 | 2.5203 |
| 0.1 | 8 | 0.6875 | 0.5729 | 1.0000 | 0.6667 | 0.3359 | 0.4719 | 2.1123 |
| 0.2 | 8 | 0.7083 | 0.5417 | 1.0000 | 0.6429 | 0.2868 | 0.4238 | 2.3050 |

### S4.2 Stored-region conditioning

For six quantitative observations, gains obtained by filtering the stored deterministic accepted region equalled gains from fresh re-inference; the maximum absolute difference was zero.

| Candidate observation | Filter gain | Fresh gain |
|---|---:|---:|
| quantitative candidate 1 | 0.2684 | 0.2684 |
| quantitative candidate 2 | 0.1043 | 0.1043 |
| quantitative candidate 3 | 0.2581 | 0.2581 |
| quantitative candidate 4 | 0.0215 | 0.0215 |
| quantitative candidate 5 | 0.0672 | 0.0672 |
| quantitative candidate 6 | 0.2304 | 0.2304 |

Across eight candidate observations and four controlled truths per candidate, predicted information value correlated with mean realised gain at `r=0.7664`; mean absolute predictive-minus-realised difference was 0.0739.

---

## S5. Reproducibility and reviewer bundle

### S5.1 Frozen evidence

The anonymised reviewer bundle contains the manuscript and Supporting Information, frozen G2 protocol and result summary, frozen auxiliary-validation summary, figure inventory and generated figures, publication-facing observation-design implementation modules, benchmark generators, tests and a per-file SHA-256 manifest.

The static initial-information comparison is a post-frozen claim-ceiling diagnostic rather than preregistered G2 evidence. Its purpose is to constrain interpretation: the frozen G2 random-order contrast establishes information-guided screening, while the adaptive-recomputation theorem states the conditions under which recomputation itself has strict expected value.

### S5.2 Explicit exclusions

The methods submission excludes the separate mechanistic-evidence / identification-boundary Perspective, prospective natural-system mechanism claims, provisional ecological-rule panels, causal-structure discovery, externally owned eco-genetic work, optional incubator backends and UI material.

### S5.3 Software validation

The release-candidate distribution is `mechanism-resolution-design` version 0.1.0. Clean validation rebuilds Figures 1–3 and Figure S1, reproduces frozen values, builds and installs the wheel outside the repository and checks the public API on Python 3.10–3.12.

## Figure S1 caption

**Figure S1. Known-truth self-consistency.** Synthetic switch-state recovery under predeclared pattern-noise strata. The benchmark checks that generating switches remain admissible; confounded alternatives are not required to disappear from a deliberately non-identifying target pattern.
