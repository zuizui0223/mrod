# Expected-cost tie-break claim guard

This repository's adaptive planner optimizes target-information regret under a hard pathwise budget. It does **not** optimize expected acquisition cost.

A controlled three-target witness shows why the distinction matters. Two adaptive trees can have identical target information and identical worst-path cost while differing in expected cost because different first measurements isolate branches with different probabilities. Reversing candidate order can therefore change which equal-information tree is retained without changing the information objective.

This is a claim ceiling, not a planner modification. The expected-cost audit reports the operational consequence of the selected tree; it does not license the statement that the selected tree is expected-cost optimal. Cost-aware optimization is a separate established problem.
