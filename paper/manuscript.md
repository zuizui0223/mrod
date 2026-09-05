# Mechanism-Resolving Observation Design: information-theoretic selection of observations under ecological mechanism ambiguity

> **Submission-track draft for Methods in Ecology and Evolution.** This manuscript reports a mechanism-resolving observation-design method and its frozen controlled validation. No new empirical mechanism claim is made.

---

## Abstract

1. Ecological mechanism inference often ends with several explanations that reproduce the same observed pattern. Selecting the highest-ranked explanation can conceal this ambiguity, whereas collecting every conceivable measurement is rarely feasible. We develop **Mechanism-Resolving Observation Design**, a workflow that retains all mechanism programs compatible with a predeclared model family, biological constraint grammar and observation set, then treats the remaining ambiguity as an experimental-design problem rather than forcing a winner.

2. The method represents the compatible parameter–mechanism combinations as an admissible region `A_ε`, quantifies residual mechanism entropy `D=H(S|A_ε)` and normalized resolvability `R=1-D/K`, and evaluates a candidate observation `Q` by its **observation information value** `V(Q)=I(S;Q|A_ε)/K` whenever the candidate outcomes form a verified predictive partition of the current region. Sequential design selects the candidate with maximum current value, conditions the admissible region on the realised outcome, and recomputes every remaining candidate value.

3. We validated selection in a frozen truth-peek-free synthetic benchmark containing random confounded systems, informative measurements and two mechanism-independent nuisance measurements. At budget two, the information-guided policy resolved all initial confounding edges on average and converged in 99.0% of systems, versus 60.45% edge resolution and 43.5% convergence under random order. At budget four, random order selected 1.169 nuisance measurements per system versus 0.014 under information-guided design, an 83.5-fold difference, while using 2.673 versus 1.518 observations. Hidden-truth false exclusion was zero throughout.

4. Independent checks recovered the mutual-information identity, exact stored-region conditioning in the deterministic validation model and positive calibration against realised resolvability gains. The contribution is therefore a validated observation-selection method, not an empirical mechanism claim: given a declared candidate family, it reports what remains unresolved, whether the available measurements contain information about that ambiguity, and which measurement should be taken next under a limited budget.

**Data/Code for peer review:** An anonymised reviewer bundle containing executable Python code, frozen protocol and result summaries, tests, and figure commands will accompany the submission. No new empirical data are reported.

**Keywords:** approximate Bayesian computation; experimental design; mechanism inference; mechanistic ambiguity; mutual information; sequential design; value of information.

---

## 1. Introduction

Ecological studies frequently seek mechanisms from patterns. A trait shift may be compatible with altered mutualistic service, a correlated life-history pathway, shared environmental forcing, or several combinations of these processes. A simulation or statistical model can make each explanation explicit, but explicitness does not guarantee distinguishability. Different mechanism programs may occupy overlapping regions of observation space and remain compatible with the evidence already collected.

A common response is model selection: define candidate models and rank them by posterior probability, likelihood or an information criterion. This is useful when the observations actually separate the candidates. Under strong mechanism ambiguity, however, a modal explanation can appear more decisive than the information in the data warrants. Approximate Bayesian computation model choice is one prominent setting in which reliability concerns have been demonstrated (Robert et al. 2011). The broader issue is not specific to ABC: if several mechanism programs make effectively indistinguishable predictions for the observed targets, ranking alone does not remove the underlying ambiguity.

A second response is to collect more data. Yet `more data` is not a design. Field observations, experiments, assays and genetic measurements differ greatly in cost and in the particular mechanism distinctions they can resolve. A measurement can be precise and biologically important while carrying almost no information about the ambiguity that remains. When observation budgets are limited, the scientific task is therefore not simply to reduce variance but to select measurements that discriminate among the currently admissible mechanisms.

Mechanism-Resolving Observation Design changes the inferential target. Rather than asking which single mechanism currently ranks first, it asks four sequential questions:

1. which parameter–mechanism combinations remain compatible with the declared evidence;
2. how much uncertainty remains about mechanism identity;
3. which candidate observation is predicted to reduce that uncertainty most;
4. when the available candidate vocabulary contains no further resolving information.

The method combines prior restriction, explicit biological constraints, entropy and value-of-information logic. Its novelty lies in their joint use for a specific ecological target: preservation and sequential reduction of mechanism ambiguity. The compatible set and its uncertainty are scientific outputs rather than intermediate objects discarded after a best-model decision. When the predictive outcomes of a candidate measurement are identified by the current admissible region, candidate value is exactly the normalized mutual information between that measurement and the remaining mechanism vector.

This paper makes four contributions. First, it defines a reproducible admissible mechanism region and separates observed targets from context, diagnostics and future measurements. Second, it derives a normalized observation information value with a direct mutual-information interpretation. Third, it closes the loop through sequential recomputation after each realised observation and gives an exact two-step condition for when recomputation has strict expected value over the best precommitted next measurement. Fourth, it tests observation selection itself in a controlled truth-peek-free benchmark where informative candidates compete with valid but mechanism-independent nuisance measurements.

The validation claim is intentionally algorithmic and conditional. We do not use a natural system to claim discovery of a true ecological mechanism. Instead, we test whether, in a declared family of confounded systems with known hidden truth, information-guided design chooses informative measurements without seeing their outcomes in advance, reduces ambiguity under a limited budget, and avoids excluding the generating explanation. This makes the synthetic benchmark—not an illustrative field narrative—the principal evidence for the observation-selection method.

## 2. Materials and Methods

### 2.1 Admissible mechanism region

Let `S in {0,1}^K` be a binary mechanism vector and `theta in Theta` continuous or discrete parameters. Complex pathways are represented by several switches being active together rather than by assigning a single mutually exclusive model label. Let `G(theta)` be a pre-data biological constraint grammar, `x_obs` fixed context, `y_obs` independent observed targets, `f` a simulator or predictive model, `P_sim` and `P_obs` maps into a shared pattern space, `d` a predeclared discrepancy, and `epsilon` an acceptance tolerance.

Define the admissible mechanism region

```text
A_epsilon(y_obs,x_obs)
= {(theta,s) in Theta x S:
   G(theta)=1 and
   d(P_sim(f(x_obs;theta,s)),P_obs(y_obs)) <= epsilon}.
```

The implementation approximates this region by prior sampling and rejection. This resembles ABC restriction, but the inferential output differs from ABC model choice: the full joint parameter–mechanism region is retained because its multiplicity is the object needed for ambiguity diagnostics and observation design.

#### 2.1.1 Evidence roles and circularity control

Every empirical or synthetic quantity receives one role before inference:

```text
observed_target     may enter the acceptance discrepancy
input_context       conditions the simulator but is not an independent target
diagnostic_only     evaluates behaviour after inference
future_observation  is withheld and evaluated as a candidate next measurement
```

This taxonomy prevents the same evidence from defining simulator context, entering the acceptance distance and then being presented again as independent validation. The constraint grammar is also applied before observed targets are evaluated, so biological feasibility is not tuned to favour the realised pattern.

An ecological example is a signed functional starting position such as `plant_trait - pollinator_functional_center`. When used, it is fixed before outcome inspection and assigned to `input_context`: it may condition `f`, but it must not re-enter `d` as an independent `observed_target`. This prevents a hypothesis-derived coordinate from being used both to define the mechanism's starting state and to validate the same mechanism.

#### 2.1.2 Mechanism entropy and resolvability

For switch `j`, marginal admissibility is

```text
CA_j = P(s_j=1 | A_epsilon).
```

Let `H(S|A_epsilon)` be the base-2 entropy of the joint switch vector. Define residual mechanism entropy and normalized resolvability as

```text
D = H(S|A_epsilon),
R = 1 - D/K.
```

Because a `K`-bit vector has at most `K` bits of entropy,

```text
0 <= D <= K,
0 <= R <= 1.
```

The denominator is maximum switch entropy, not realised prior entropy. This preserves a fixed interpretation across priors: `R=1` means the switch vector is completely resolved inside the accepted region, while lower values retain joint ambiguity. Pairwise or higher-order mechanism-equivalence summaries can be constructed from the same accepted switch rows. Replaceability measures whether one mechanism's accepted contribution can be substituted by alternative programs rather than merely whether its marginal admissibility is high.

### 2.2 Observation information value

Let `Q` be a candidate future measurement with finite outcomes `q`. For validated stored-region calculation, the outcome maps must form a mutually exclusive and exhaustive partition of current `A_epsilon`. The predictive probability is then the pushforward of the restricted current region:

```text
Pr(Q=q | A_epsilon).
```

Define observation information value as expected gain in resolvability:

```text
V(Q)
= E_Q[R(A_epsilon | Q)-R(A_epsilon)].
```

Using the entropy definition of resolvability,

```text
V(Q)
= {H(S|A_epsilon)-H(S|A_epsilon,Q)}/K
= I(S;Q|A_epsilon)/K.
```

Therefore

```text
0 <= V(Q) <= 1-R(A_epsilon) <= 1.
```

`V(Q)=0` exactly when `Q` is conditionally independent of residual mechanism identity under the current accepted region. The upper bound is attained when the observation removes all remaining switch entropy. An individual realised outcome may increase conditional entropy, but expected gain under the coherent predictive distribution cannot be negative.

A candidate is reported as non-estimable when its outcomes overlap, fail to cover the current region or depend on simulator outputs absent from stored rows. A declared external outcome prior is not silently substituted and labelled as validated information value. A structural edge-cut score remains available only as an explicitly labelled fallback when the predictive partition cannot be computed; every selection step records which score source was used.

### 2.3 Sequential observation design

The design is adaptive because candidate value depends on the current admissible region:

```text
A_0 = current admissible region
for t = 0,1,... until stopping:
    score each verified remaining Q by I(S;Q|A_t)/K
    select the maximum positive current score
    obtain the realised outcome only after selection
    condition A_t on that outcome to form A_{t+1}
    recompute all predictive probabilities and scores
```

Recomputation is not assumed to be uniformly beneficial. For a two-step finite design, let `X` be the first observation and let `U_q(x)=V(Q_q | X=x)` denote the normalized information value of a remaining candidate `q` on branch `x`. The adaptive and strongest precommitted-static second-step values are

```text
V_adapt  = E[max_q U_q(X)],
V_static = max_q E[U_q(X)].
```

Therefore `V_adapt>=V_static`. Equality holds if and only if at least one candidate is branchwise optimal on every positive-probability first-outcome branch; strict adaptive advantage occurs exactly when the intersection of those branchwise argmax sets is empty. This result characterizes the value of recomputation in the declared finite two-step setting. It does not establish global optimality of the full multi-step greedy policy.

The procedure stops when the observation budget is exhausted, the declared confounding structure is resolved, or every available verified candidate has zero current information value. The last condition is substantive: unresolved mechanisms may remain, but the declared measurement vocabulary contains no additional information about them.

### 2.4 AI-assisted development disclosure

OpenAI ChatGPT was used interactively to assist with code review, draft editing and repository/documentation maintenance. The author reviewed and takes responsibility for all generated or edited text and code. AI outputs were not treated as empirical observations or independent scientific evidence. Frozen benchmark configurations and reported numerical results were executed and checked through the reproducible workflows described below.

### 2.5 Controlled validation design

We used four complementary controlled checks for the primary method validation, plus one post-frozen claim-ceiling diagnostic. None is presented as natural-system causal validation.

#### 2.5.1 Confounding demonstration

A compact synthetic example was constructed in which multiple switch programs reproduce the same ordinal target pattern. The demonstration contrasts a single low-mass MAP switch combination with the retained admissible region, reports residual ambiguity, and shows how a quantitative candidate observation separates previously equivalent programs.

#### 2.5.2 Known-truth self-consistency

Synthetic observations were generated under declared switch states and passed through the same inference model. The purpose was to check that generating switches remain admissible under pattern-noise strata. Because the target pattern is deliberately non-identifying, additional confounded switches are not required to disappear and exact switch-state accuracy is not expected to equal one.

#### 2.5.3 Frozen G2 truth-peek-free selection benchmark

The primary selection validation uses a frozen protocol with five predeclared seeds and 200 systems per seed. Every system has `K in {4,5,6}`, one or two disjoint two-driver confounds, random pre-data driver coefficients, 1,500 prior draws and an explicit resolving quantitative observation for each confound. Two additional binary nuisance measurements are generated independently of the mechanism vector. They are valid mutually exclusive and exhaustive candidate observations but have no designed mechanism information.

The same seed-defined systems, hidden truths, candidate sets and budgets 0–4 are supplied to two policies:

```text
information-guided  choose the remaining candidate with maximum current V(Q)
random_order        choose uniformly among remaining candidates
```

Neither policy observes a hidden outcome before candidate selection. Hidden truth is used only after selection to materialise the chosen candidate's realised benchmark outcome. The accepted region is then conditioned and the information-guided policy recomputes all current candidate values. Random order is an uninformed selection baseline, not a competing mechanism-inference method.

Primary outcomes are the fraction of initial confounding edges resolved, convergence to an empty confounding graph, number of observations used, number of nuisance measurements selected and false exclusion of the hidden true explanation. Policy contrasts were designated descriptive. The protocol contains no favourable-result threshold requiring the information-guided policy to outperform random selection, and scientific parameters cannot be overridden at execution.

Historical protocol identifiers and stored policy keys are preserved unchanged in the frozen machine-readable files for provenance. They are not used as the active method name.

#### 2.5.4 Information identity and calibration

One implementation independently computes expected resolvability gain and empirical mutual information from the joint `(S,Q)` table. A second check compares stored-region conditioning with fresh deterministic re-inference for quantitative observations. Finally, predicted observation information value is compared with realised resolvability gains across controlled hidden truths. These checks distinguish an algebraic identity, a computational shortcut and empirical calibration.

#### 2.5.5 Post-frozen static-information diagnostic

Because the frozen G2 contrast uses uniform random ordering as its baseline, we subsequently ran a matched claim-ceiling diagnostic with a stronger nonadaptive comparator. `static_initial_information` ranks all candidates once by their information values in the initial admissible region, discards candidates with non-positive initial value, and follows the resulting fixed order without recomputing after realised outcomes. The diagnostic reused the same generator settings, five seeds, 200 systems per seed, hidden truths, candidate vocabularies, nuisance measurements and headline budgets as G2. It was conducted after the G2 freeze, was not preregistered, and did not modify the frozen protocol or its reported results.

## 3. Results

### 3.1 The admissible region preserves confounding instead of manufacturing a winner

In the compact confounding example, conventional ranking returned a single MAP switch combination with low posterior mass. The accepted sample nevertheless contained multiple coupled mechanism programs. The set-valued analysis exposed that multiplicity through marginal admissibility, joint entropy and mechanism-equivalence structure. A candidate quantitative observation that separated the coupled switches had positive information value, whereas mechanism-independent candidates had zero or negligible information under the current region.

This example demonstrates the reporting difference between model ranking and admissible-set inference. The result is not that the synthetic generating mechanism was ecologically true, but that the method did not hide observational equivalence behind a modal label.

### 3.2 Known-truth checks retain generating mechanisms

Under unchanged known-truth defaults, the zero pattern-noise stratum had mean switch-state accuracy 0.6562 and recall of applicable true-ON switches 1.000. Recall remained 1.000 in the 0.1 and 0.2 noise strata. Lower exact-state accuracy reflected retention of additional confounded explanations, which is the expected signature when the observed pattern does not uniquely identify the switch vector.

The benchmark therefore supports self-consistency in a limited sense: generating switches were not discarded merely because equivalent alternatives survived. It does not show universal recovery under simulator misspecification or establish that any retained program is correct in nature.

### 3.3 G2 validates information-guided observation selection under limited budget

The frozen G2 benchmark contained 1,000 generated systems per policy. At budget two, information-guided design resolved `1.000 ± 0.000` of initial confounding edges and converged in `0.990 ± 0.0079` of systems across the five predeclared seeds. It used `1.505 ± 0.030` observations and selected `0.001 ± 0.0022` nuisance measurements per system. The matched random-order policy resolved `0.6045 ± 0.0231` of initial edges, converged in `0.435 ± 0.0355` of systems, used `1.821 ± 0.024` observations and selected `0.974 ± 0.0277` nuisance measurements.

The within-seed information-guided minus random-order contrast was therefore `+0.3955 ± 0.0231` for edge resolution and `+0.555 ± 0.0417` for convergence, while information-guided design used `0.316 ± 0.020` fewer observations. At budget one, convergence was 0.495 under information-guided design and 0.179 under random order.

Budget four isolates measurement efficiency after both policies had resolved all initial confounding edges on average. Information-guided design converged in 0.999 of systems and used 1.518 observations, whereas random order converged in 0.940 and used 2.673. Most visibly, random order selected 1.169 mechanism-independent nuisance measurements per system versus 0.014 under information-guided design. The absolute difference was 1.155 nuisance measurements; the ratio was `1.169/0.014=83.5`, equivalent to an approximately 98.8% reduction relative to random order.

The fold ratio is descriptive and is reported with its absolute values because ratios become unstable when the selected count approaches zero. At budget two the aggregate ratio is much larger because the information-guided mean is 0.001, but the budget-four comparison provides the more conservative headline after both policies have enough budget to resolve all edges on average.

Hidden-truth false exclusion was zero in every policy-by-budget cell. All 10,000 system–policy–budget records retained the hidden generating explanation. Thus the selection advantage was not obtained by narrowing the accepted set so aggressively that the truth was discarded.

### 3.4 A stronger static-information baseline limits the adaptive claim

The post-frozen matched diagnostic showed essentially no practical difference between adaptive recomputation and a static initial-information ordering on this benchmark family. At budget two, both information-based policies converged in 0.990 of systems, resolved 1.000 of initial edges on average, used 1.505 observations and selected 0.001 nuisance measurements. At budget four, adaptive and static policies converged in 0.999 and 0.998 of systems respectively; both resolved 1.000 of initial edges, used 1.518 observations and selected 0.014 nuisance measurements. False exclusion remained zero.

Thus the frozen G2 family provides strong evidence for information-guided candidate screening but little empirical evidence for an incremental performance gain from recomputation itself. This negative diagnostic does not contradict the adaptive theorem: strict adaptive value is expected only when outcome branches disagree on the best remaining candidate in the sense formalized in Section 2.3. The diagnostic was not part of preregistered G2 and is used to narrow, not expand, the validation claim.

### 3.5 Information-value implementation and calibration checks

Expected resolvability gain and independently computed `I(S;Q|A_epsilon)/K` agreed to the implementation's display tolerance. For six directly checked quantitative observations, conditioning the stored deterministic admissible region and performing fresh re-inference produced identical resolvability gains; the maximum absolute difference was zero.

Across eight candidate observations and four controlled truths per observation, predicted information value correlated positively with mean realised resolvability gain (`r=0.7664`). The mean absolute difference between prediction and mean realised gain was 0.0739. Individual outcomes remained variable, as expected for preposterior quantities. These results support the intended average information interpretation rather than a claim that the value predicts every realised gain exactly.

## 4. Software and reproducibility

The public Python surface exposes descriptive functions for admissible mechanism regions, entropy, resolvability, replaceability, mechanism equivalence, observation information value and sequential observation selection. Historical implementation labels remain only as compatibility backends or frozen-provenance identifiers and are excluded from the advertised publication API.

The final G2 result is tied to the frozen protocol and stored result summary. Every output row records the protocol SHA-256 and clean execution provenance. Earlier pilot values are excluded from the active manuscript. A clean reproducibility workflow rebuilds Figures 1–3 and Figure S1, reproduces frozen validation summaries, builds and installs the release-candidate wheel outside the repository and checks its public API across Python 3.10–3.12.

The reviewer bundle excludes author metadata and public repository locators while retaining executable source, tests, frozen protocol/result summaries, figure commands and a per-file SHA-256 manifest. No new empirical data are reported, and no ecological mechanism conclusion is derived from the controlled examples.

## 5. Discussion

Mechanism-Resolving Observation Design treats unresolved mechanism multiplicity as a result rather than an inconvenience to be hidden. This matters because a low-mass modal mechanism can look decisive in a table while the accepted region remains broadly ambiguous. Reporting the admissible set, its entropy and its replaceability structure makes the remaining uncertainty inspectable and reproducible.

The identity

```text
V(Q)=I(S;Q|A_epsilon)/K
```

provides a direct interpretation for observation value. A measurement is useful exactly to the extent that it carries information about the mechanism distinctions still unresolved inside the current admissible region. This differs from ranking candidates by general precision, sample size or ecological prominence. A measurement can be scientifically interesting and still have zero information value for the ambiguity at hand.

Sequential recomputation is conditionally valuable rather than uniformly necessary. After one observation, the admissible region changes and the value of every remaining candidate can change, but a static information ordering can be equally effective when one candidate remains branchwise optimal across all positive-probability outcomes. The two-step theorem makes this distinction exact: adaptive expected value is never smaller than the best precommitted second measurement, and it is strictly larger exactly when the branchwise argmax sets have no common candidate. The post-frozen static-information diagnostic is consistent with the equality side of this result for the present G2 family. Sequential observation design therefore recomputes by default because the relevant branch structure is generally unknown in advance, not because every problem is asserted to gain from adaptation.

The G2 benchmark was designed to test selection rather than observation sufficiency. A candidate set containing only direct resolvers would show that informative measurements can solve confounds, but not that the method distinguishes them from wasted measurements. Adding valid mechanism-independent nuisance candidates created a controlled competition for budget. The resulting approximately 84-fold difference at budget four measures how often the uninformed policy spent scarce observations on candidates that had no designed mechanism information after both policies had enough budget to resolve the edge structure on average.

The benchmark nevertheless defines a narrow claim. Information-guided design outperformed uniform random order over one frozen family of random confounded systems, while the stronger post-frozen static-information comparator essentially matched the adaptive policy. We therefore interpret the benchmark as validation of information-guided candidate screening, not as empirical proof that recomputation adds value in every system. The theorem supplies the conditional adaptive claim; neither result proves global optimality, superiority to every Bayesian design method, or performance under every stochastic ecological simulator. Candidate vocabularies were finite and explicitly represented. The nuisance measurements were independent of mechanisms rather than subtly correlated proxies. Broader misspecification challenges remain future work.

Admissibility is always relative to a declared mechanism vocabulary, parameter prior, constraint grammar, observation map, discrepancy and tolerance. An omitted mechanism cannot be recovered by retaining the accepted set. A predictive partition must also be identified before stored-region information value can be computed. When outcomes overlap, are incomplete or require an unmodelled process, the honest result is non-estimability until an additional predictive model is supplied.

Synthetic validation is appropriate to the present claim because the hidden mechanism, candidate information structure and outcome timing must be controlled to test truth leakage and selection behaviour. A natural-system application could demonstrate usability but could not reveal whether the selected measurement was optimal relative to an unknown causal truth. The absence of new empirical data is therefore a boundary, not a missing validation layer: this paper validates an observation-selection algorithm under known controlled conditions and does not claim empirical discovery.

The practical output for ecologists is a disciplined sequence:

```text
declare mechanisms and constraints
→ retain compatible explanations
→ quantify residual mechanism uncertainty
→ verify candidate predictive outcomes
→ select the maximum-current-information measurement
→ condition and repeat
→ stop when resolved, budget-limited or information-limited.
```

This reframes mechanistic ambiguity from a reason to force a winner or postpone inference into a quantitative experimental-design problem.

## Figure captions

**Figure 1. Admissible-set reporting under controlled confounding.** A compact synthetic example contrasts a single low-mass MAP switch combination with the full accepted region. Panels report model-ranking mass, mechanism admissibility and entropy, mechanism-equivalence structure, observation information value for candidate measurements and the change after a confound-breaking observation. The figure diagnoses inferential behaviour and is not a natural-system mechanism claim.

**Figure 2. Truth-peek-free sequential observation selection.** Frozen G2 results compare information-guided design, which selects the remaining candidate with maximum current observation information value, with a matched uniform random-order policy. Panels show convergence, fraction of initial confounding edges resolved, observations used and mechanism-independent nuisance measurements selected across budgets 0–4. Error bars are sample standard deviations across five predeclared seeds. Hidden-truth false exclusion was zero in every policy-by-budget cell. The budget-four nuisance-selection panel highlights 1.169 selections under random order versus 0.014 under information-guided design, an 83.5-fold difference.

**Figure 3. Observation-information identity and calibration.** Left, resolvability gains obtained by filtering the current deterministic admissible region are compared with fresh re-inference for six quantitative observations. Right, predicted observation information value is compared with realised resolvability gain across controlled hidden truths; individual outcomes and candidate-wise mean realised gains are distinguished.

**Figure S1. Known-truth self-consistency.** Synthetic switch-state recovery under predeclared pattern-noise strata. The figure tests whether generating switches remain admissible. Confounded alternatives are not required to disappear from a deliberately non-identifying pattern.

## References

- Beaumont, M.A., Zhang, W. & Balding, D.J. 2002. Approximate Bayesian computation in population genetics. *Genetics* 162: 2025–2035.
- Beaumont, M.A. 2010. Approximate Bayesian computation in evolution and ecology. *Annual Review of Ecology, Evolution, and Systematics* 41: 379–406.
- Canessa, S., Guillera-Arroita, G., Lahoz-Monfort, J.J., Southwell, D.M., Armstrong, D.P., Chadès, I., Lacy, R.C. & Converse, S.J. 2015. When do we need more data? A primer on calculating the value of information for applied ecologists. *Methods in Ecology and Evolution* 6: 1219–1228.
- Chaloner, K. & Verdinelli, I. 1995. Bayesian experimental design: a review. *Statistical Science* 10: 273–304.
- Csilléry, K., Blum, M.G.B., Gaggiotti, O.E. & François, O. 2010. Approximate Bayesian computation in practice. *Trends in Ecology & Evolution* 25: 410–418.
- Grimm, V., Revilla, E., Berger, U., Jeltsch, F., Mooij, W.M., Railsback, S.F., Thulke, H.-H., Weiner, J., Wiegand, T. & DeAngelis, D.L. 2005. Pattern-oriented modelling of agent-based complex systems: lessons from ecology. *Science* 310: 987–991.
- Hartig, F., Calabrese, J.M., Reineking, B., Wiegand, T. & Huth, A. 2011. Statistical inference for stochastic simulation models: theory and application. *Ecology Letters* 14: 816–827.
- Raiffa, H. & Schlaifer, H. 1961. *Applied Statistical Decision Theory.* Harvard University Press, Boston.
- Robert, C.P., Cornuet, J.-M., Marin, J.-M. & Pillai, N.S. 2011. Lack of confidence in approximate Bayesian computation model choice. *Proceedings of the National Academy of Sciences* 108: 15112–15117.
