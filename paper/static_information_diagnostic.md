# Static initial-information comparator diagnostic

Status: **post-frozen claim-ceiling diagnostic; not part of preregistered G2**.

## Question

The frozen G2 benchmark compares an information-guided adaptive policy with uniform random ordering. That comparison establishes whether the method can avoid mechanism-independent measurements under limited budget, but it does not by itself isolate the value of recomputing candidate scores after each realised outcome.

We therefore ran a matched post-frozen diagnostic with a stronger nonadaptive comparator, `static_initial_information`. It ranks every candidate once using its information value in the initial admissible mechanism region, discards candidates with non-positive initial value, and then follows that fixed order without recomputing after outcomes.

The diagnostic reused the same generator settings as the frozen G2 family: five seeds, 200 systems per seed, the same hidden truths, candidate vocabularies, nuisance measurements and budgets. It did **not** alter the frozen G2 protocol, stored policy keys, scientific parameters or headline results.

## Aggregate result

| Budget | Policy | Convergence | Fraction resolved | Mean observations | Mean nuisance selections | False exclusion |
|---:|---|---:|---:|---:|---:|---:|
| 2 | information-guided adaptive | 0.990 | 1.0000 | 1.505 | 0.001 | 0.000 |
| 2 | static initial information | 0.990 | 1.0000 | 1.505 | 0.001 | 0.000 |
| 2 | random order | 0.435 | 0.6045 | 1.821 | 0.974 | 0.000 |
| 4 | information-guided adaptive | 0.999 | 1.0000 | 1.518 | 0.014 | 0.000 |
| 4 | static initial information | 0.998 | 1.0000 | 1.518 | 0.014 | 0.000 |
| 4 | random order | 0.940 | 1.0000 | 2.673 | 1.169 | 0.000 |

Values are means across the five matched seeds (1,000 generated systems per policy-budget cell in aggregate).

## Interpretation

The current G2 system family strongly supports **information-guided candidate screening**: both information-based policies avoid nuisance measurements and resolve ambiguity much more efficiently than uninformed ordering. It does **not** provide meaningful empirical evidence that adaptive recomputation improves performance over a strong static ordering based on initial information value; the two information-based policies are essentially identical on the headline cells.

This negative result is compatible with the separate adaptive-recomputation theorem. For a two-step finite design, adaptive recomputation has strict expected value over the best precommitted second measurement exactly when the positive-probability outcome branches have no common information-maximizing candidate. The current G2 family evidently does not often create such branchwise rank conflicts.

Accordingly, the manuscript treats G2 as validation of information-guided observation selection and uses the theorem—not the G2 random-order contrast—to state when recomputation itself has strict value. No claim of universal adaptive superiority is made.
