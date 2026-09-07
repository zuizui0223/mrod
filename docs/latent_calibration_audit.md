# An unknown calibration state is not a known scenario: when to measure a standard

Status: **internal finite-model audit**, extending the calibration and repeat/switch
notes without replacing any public objective. Active MEE manuscript, submission
manifests, frozen G2 results/figures and public exports are unchanged. The code
reuses `score_likelihood_candidates`, `condition_on_selected`, and the existing
replication-information profile. All examples below are synthetic, not fitted
field relationships, selection evidence, or evidence of adaptation.

## 1. Three distinct uncertainty tasks

A sensitivity analysis evaluates each fixed calibration separately. Robust
scenario comparison asks which action is defensible across a declared set. An
unknown-calibration Bayesian calculation instead needs a justified joint current
distribution over target T and calibration L (and any other full-world variables).
It must not invent a probability distribution over a mere sensitivity list.

This audit handles only the third task, given an explicit finite joint ensemble
with positive weights and future likelihoods valid conditional on current D.
Both T and L are updated on the same past evidence. A shared detector/background
state is persistent, not silently redrawn at every reading. Reference strings
record those assumptions; they do not establish calibration or transportability.

## 2. Do not average known-calibration utilities and call that unknown-calibration MI

At fixed current D, the actual target information is I(T;Q|D), with L marginalized
in the joint predictive distribution. The different quantity I(T;Q|L,D) averages
target MI conditional on L as though calibration were known. They obey

    I(T;Q|L,D) - I(T;Q|D)
      = I(T;L|Q,D) - I(T;L|D).

Proof: expand I(T;Q,L|D) in its two chain-rule orders and rearrange. The audit
computes the four terms using the existing scoring/conditioning API and checks
the identity. The conditional quantities average over possible outcomes or L.
They are not a realised-outcome guarantee.

Before data, independent T and L make the known-calibration average no smaller
than actual MI. After D induces dependence, the gap can have either sign. In
particular, a reference R whose likelihood depends only on L has

    I(T;R|L,D) = 0,
    I(T;R|D) = I(T;L|D) - I(T;L|R,D) <= I(T;L|D).

It can therefore resolve target ambiguity by reducing target-calibration
confounding, even though it does not directly measure the target. At a current
state where T and L are independent its immediate target information is zero;
that is not a statement about its joint or future value.

`calibration_only_likelihood_in_pool` checks equality of the normalized supplied
likelihood rows within each declared L group. It is only a finite-table property,
not proof that a real standard is independent of biology or shares calibration
with the original record. No estimated approximate equality is promoted to a
structural law. The oracle average is diagnostic, never the recommended utility.

## 3. Concrete cross-calibration aliasing without reversing a sensor

Let T,L each be binary and initially independent/uniform. The response assay has

    Pr(Y=present|T,L) = 0.1 + 0.4*T + 0.4*L.

T labels two process states. L labels a persistent low/high measurement-background
state. A known-standard reading R has positive probability 0.1 when L=0 and 0.9
when L=1, independently of the response history given (T,L). This independence
and shared-state assumption are part of this synthetic experiment.

| T | L | Response-positive probability | Standard-high probability |
|---:|---:|---:|---:|
| 0 | 0 | 0.1 | 0.1 |
| 0 | 1 | 0.5 | 0.9 |
| 1 | 0 | 0.5 | 0.1 |
| 1 | 1 | 0.9 | 0.9 |

At each fixed L, the two target laws differ, so ideal unlimited repetition could
learn T. In the unknown persistent-L model, (T=0,L=1) and (T=1,L=0) have exactly
the same response law. Response-only data cannot distinguish their target states.
The prior expected residual floor is H(T|C)=0.5 bit, with C the response-law class.
Thus zero floors in every known-calibration scenario do NOT imply a zero floor
in their joint unknown-calibration model.

Marginalizing L before forming each repeated likelihood creates a different
experiment. Persistent L requires

    P(Y_1:n|T) = sum_l P(l|T) product_i P(Y_i|T,l),

not product_i sum_l P(l|T) P(Y_i|T,l). The latter redraws calibration each time.
Here that alternative experiment has Bernoulli rates 0.3 and 0.7 and a zero
asymptotic floor. Both protocols are explicitly reported, not conflated.

## 4. A reference becomes the right next observation after current data

The response and reference are scored without seeing future outcomes. History
probabilities in the JSON refer to the specified ORDERED histories, not count
events; under the stated conditional-iid model their counts are sufficient for
posterior weights.

| Current response data | Next response MI | Shared-standard MI | All further response-only ceiling |
|---|---:|---:|---:|
| None | 0.118709101 | 0 | 0.500000000 |
| One present | 0.068845785 | 0.016300979 | 0.381290899 |
| One present, one absent | 0.032594535 | 0.104799983 | 0.264705882 |
| Five present, five absent | 0.000016677 | 0.515945428 | 0.006010276 |

Units are raw expected target-information bits. For the last history, the current
world weights in table order are approximately

    (0.003005138, 0.496994862, 0.496994862, 0.003005138).

The reference exceeds even the entire remaining response-only ceiling by about
0.509935153 bit, under this exact model. Yet averaging the known-L target scores
would give the reference zero and the response about 0.003174806 bit: a different,
inappropriate task for unknown calibration would reverse the next-action ranking.

A standard from an independent, unlinked calibration state is represented by a
constant 0.5 likelihood here and has zero target information. Likewise, raising
the shared standard's error to 0.5 removes its benefit. Merely naming something
a calibration observation does not make it informative.

One noisy shared-standard result is not exact determination. For example, a high
standard result after the balanced ten-response history reduces target entropy
to about 0.484054572 bit but retains both target states with positive probability.
The full joint law (response probability, standard probability) separates all
four supplied worlds at standard error below 0.5; that asymptotic law statement
must not be relabelled as finite-data complete resolution.

## 5. Reporting contract and limits

`audit_latent_calibration` requires current joint weights, target and calibration
columns, support provenance and conditional-likelihood provenance. It reports
actual marginal MI, known-calibration average MI, their signed gap, current and
expected post-observation target-calibration dependence, and the chain-rule
error. Positive next-action rankings use only actual marginalized target MI.

Missing predictions remain non-estimable; incomplete vocabularies yield only a
provisional ranking. No positive singleton is not a sequence-impossibility claim.
Exact zero-likelihood exclusions explicitly restrict posterior support before
rescoring; numerical underflow raises rather than excluding an admissible world.
Small entropy or numerical tie tolerance does not establish point identification.
No future realised data enter ranking. Prospective outcome integration is not
truth-peeking. The input joint weighting remains model-dependent, not a structural
exhaustiveness certificate: `feasible_domain_exhaustiveness=not_certified`.

The equal response laws and the linked calibration kernel are explicit model
restrictions. Unknown continuous calibration, model misspecification, dependence
between readings, and unequal costs require further models. This audit does not
fit those models, estimate a sensor error, or provide a universal calibration-first
policy. It does not add a new weighted nuisance objective or optimal sequence.

## 6. Reproduce and verify

    python -m examples.latent_calibration_report > latent_calibration_report.json
    python -m pytest -q tests/test_latent_calibration_audit.py

An independent probability-ratio oracle checks every binary T,L,Q mapping on four
worlds (4,096 cases), plus 200 seeded weighted/noisy joint distributions. Other
tests cover signed gaps, exact posterior restrictions, persistent/redrawn laws,
missing predictions/targets, incomplete rankings, ties, no mutation and underflow.
Local isolated tests use byte-identical copies of the two existing dependencies;
full repository CI is a separate merge gate.

## 7. Prior art and claim ceiling

Nuisance-aware Bayesian active learning and using auxiliary information to improve
a target are established, not new inventions here. Sloman, Bharti, Martinelli &
Kaski (2024), *Bayesian Active Learning in the Presence of Nuisance Parameters*,
PMLR 244:3245-3263, directly studies the target-versus-nuisance acquisition problem:
https://proceedings.mlr.press/v244/sloman24a.html . Its arXiv version also explicitly
marginalizes nuisance conditional on the target and updates on the full history:
https://arxiv.org/html/2310.14968v2 . The present table and chain-rule audit are
supplied independently; they are not claimed to be that paper's examples.

A closely related new preprint was found during the 2026-09-07 search: Fotias,
*Resolution-Aware Experimental Design under Partial Identifiability*, arXiv:2609.03686
(submitted 2026-09-03). The indexed author abstract describes structural candidate
sets under false-exclusion control and cross-nuisance aliasing. The primary full
text could not be retrieved in this audit, so this is an **abstract-level prior-art
flag, not a completed theorem or benchmark comparison**. Do not claim that
partial-identification-aware design or cross-nuisance aliasing is unique to MROD.
https://arxiv.org/abs/2609.03686

Boundary continues to own structural target/law distinctions. MROD here implements
an internal handoff: the same current-data confounding can turn a target-free
calibration measurement into useful target information. Existing publication
claims, objective and empirical evidence level are unchanged.
