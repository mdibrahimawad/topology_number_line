# Digit rules and written numbers

## Question
Does changing the required output order change the representation at the query equals sign? Does writing numbers as English words change numerical geometry?

## Fixed design
Three pinned models: Crystal, StarCoderBase-3B and OpenLLaMA-3B, matching the earlier experiments. All native returned levels are saved; the final level includes each model's native final normalization. Inference retains Crystal bfloat16 and the other models float32. Cross-model precision differences remain a limitation.

| Condition | Example | Targets per context |
|---|---|---:|
| Four-digit copy | 1234=1234 | 1,000 matched four-digit numbers |
| Reverse | 1234=4321 | Same 1,000 |
| Swap first two | 1234=2134 | Same 1,000 |
| Swap last two | 1234=1243 | Same 1,000 |
| Numeric copy | 234=234 | Every integer 1–1,000 |
| Written-number copy | two hundred and thirty-four=two hundred and thirty-four | Every integer 1–1,000 |

Each condition uses two fixed demonstration contexts, analyzed separately. Four-digit demonstrations and targets are identical across permutation conditions; only example outputs change. Eight examples define each rule. After a bounded pilot comparison, the four-digit conditions also have explicit instructions; numeric and written-number copying retain the original example-only format. Instruction text and its token length vary across digit rules and remain a confound for purely observational task comparisons. Four-digit outputs retain leading zeroes. Four-digit targets exclude demonstration values. Copying 1–1,000 retains and flags demonstration collisions because complete range coverage is required.

English uses lowercase British spelling, 'and', and hyphenated compound tens. The numeric-copy control uses the same numerical demonstration values and target coverage as written-number copying.

## Pilot and validity
A deterministic 32-target sample from each task/context is greedily generated: 384 prompts per model. Exact-answer accuracy is measured. Query vectors and probabilities are checked against unpadded native full-model forwards. A separate, bounded explicit-instruction pilot tests whether poor rule performance can be rescued; both pilots are preserved. The final full-run policy is explicitly recorded above.

Successful extraction does not establish successful task execution. Conditions below 80% pilot exact accuracy are flagged as attempted/failed tasks; this is a practical reporting threshold, not a statistical theorem. Target/source labels and desired-output labels are predetermined metadata, not discovered clusters or proof of behavior.

## Saved analysis
- All original hidden vectors, input token IDs, expected strings, native next-token predictions and checkpoint receipts.
- Original-space exact Euclidean four-nearest-neighbor graph; PCA is used only for display.
- Separate PCA2/PCA3 projections, plus shared PCA bases across matched tasks within each model/layer/context.
- Original-space label associations for input digits, desired first output, and token count.
- H0/H1 persistent homology on each nondegenerate 1,000-point cloud, normalized by that cloud's median pair distance; persistence diagrams and Betti curves. Degenerate or undefined normalized cases are marked explicitly.

Independent PCA axes can rotate or reflect; use shared-coordinate panels for changes across conditions. Contexts are replications, not independent extra numbers. This is a descriptive experiment, not a causal intervention and not a claim of a unique topological shape.

## Resources and recovery
One native model per GPU worker, at most ten L40S workers total. CPU analysis uses independent groups, up to 24 workers with two cores each. Shared-coordinate comparisons use up to six two-core workers. Stages have barriers; retries are bounded by explicit reruns. Checkpoints use atomic files, hashes and durable volume commits. Completed chunks and layers are verified and skipped on resume.

The compute ledger carries earlier failed reservations and reserves conservatively before launching work. The allowance is $14.50 inside the user's reported $17 balance. It is an estimate, not a billing API or account-level spending cap; no billing settings are changed.
