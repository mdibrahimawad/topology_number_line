from pathlib import Path
import json,html
import numpy as np
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/pca3d_analysis'
R=json.loads((O/'metrics.json').read_text());H=json.loads((O/'hidden_digit_contributions.json').read_text());D=json.loads((O/'digit_contributions.json').read_text());C=json.loads((O/'shape_comparisons.json').read_text())
def metric(m,l):return next(r for r in R if r['model']==m and r['layer']==l)
def hidden(m,l):return next(r for r in H if r['model']==m and r['layer']==l)
models=[('crystal','Crystal',1,31,32),('starcoder','StarCoderBase-3B',5,35,36),('openllama','OpenLLaMA-3B',2,25,26)]
shift='| Model | Early example: last-digit share | Second-last: first-digit share | Final: first-digit share |\n|---|---:|---:|---:|\n'
for m,name,e,p,f in models:
 shift+=f'| {name} | L{e}: {hidden(m,e)["digit_variance_fractions"]["units"]:.1%} | L{p}: {hidden(m,p)["digit_variance_fractions"]["thousands"]:.1%} | L{f}: {hidden(m,f)["digit_variance_fractions"]["thousands"]:.1%} |\n'
fits='| Model / second-last layer | First-digit categories | Linear + repeated waves | Waves plus a broad bend | Four digit categories |\n|---|---:|---:|---:|---:|\n'
for m,name,e,p,f in models:
 q=metric(m,p)['fits'];fits+=f'| {name} L{p} | {q["leading_digit"]["block100"]:.1%} | {q["generalized_decimal"]["block100"]:.1%} | {q["generalized_plus_global_bend"]["block100"]:.1%} | {q["decimal_digits"]["block100"]:.1%} |\n'
comparison='| Models | Closest layers by averaged shape | Similarity of 90 bin means | Similarity of all 9,000 matched points |\n|---|---|---:|---:|\n'
for r in C:
 if r['kind']=='best_bin_means':comparison+=f'| {r["model_a"]} / {r["model_b"]} | {r["layer_a"]} / {r["layer_b"]} | {r["bin_mean_similarity"]:.3f} | {r["pointwise_similarity"]:.3f} |\n'
text=f'''# What the 3D number shapes tell us

**There is meaningful structure here. The strongest measured pattern is a change from last-digit structure early in the network to first-digit structure late in the network. The visible bends and repeating patterns are worth investigating, but these results do not establish a DNA-like double helix.**

I inspected all **97 saved views** and measured all **94 contextual layers**: Crystal 1–32, StarCoder 1–36, and OpenLLaMA 1–26. All 10,000 targets were used for the overall geometry and neighborhood checks. To separate digit effects from number length, the main comparison uses **all 9,000 four-digit numbers, 1000–9999**. I also re-read the original hidden vectors for every contextual layer. No model inference or new cloud jobs were needed.

[Open the original interactive 3D gallery](../pca3d/index.html) · [Existing full-space PH analysis](../ph_analysis/index.html) · [All layer measurements](metrics.csv)

## 1. The strongest finding: the dominant digit changes

**What we checked:** how much of the variation can be accounted for by the thousands, hundreds, tens, and units digit separately. For four-digit numbers, thousands is the first digit and units is the last digit. The four places vary independently over this balanced sample.

**What we found in the original hidden vectors, before PCA:**

{shift}
These are fractions of variation associated with differences between digit-group means, not prediction accuracies. Early examples are each model’s strongest units-digit layer. The full curves below show every layer, including the less orderly middle regions.

![Digit effects across all contextual layers](digit_shift.png)

**Why this matters:** the shared process is clearer than a shared named shape. All three models gradually change which part of a number dominates their representation. OpenLLaMA follows a less smooth route through its middle layers.

**Working hypothesis:** our prompt ends with a numeral and then `=`. An early representation can carry strong information about the numeral’s last digit; to copy the answer, the next expected token is its first digit. A shift toward preparing that first output digit is consistent with the measured pattern. This is a mechanism hypothesis, not a demonstrated attention circuit or a test of actual answer accuracy.

## 2. There is measurable decimal repetition, but it is not enough to identify a helix

**What we checked:** fits of the form `position(n) = offset + linear trend + cosine wave + sine wave`. We tested periods 2, 5, 10, 100, 1000, 3000, and 10000; combined decimal frequencies; and straight, cubic, and digit-category alternatives. Coefficients were fitted on training targets and evaluated on held-out targets. A second split withheld entire 100-number blocks.

**What we found:** adding a period-10 wave to a baseline with a separate local straight trend for each leading digit improves held-out 3D variance explained by:

- **41.0 percentage points at Crystal L3**: 3.1% → 44.1%.
- **48.2 points at StarCoder L7**: 5.5% → 53.6%.
- **23.8 points at OpenLLaMA L2**: 0.9% → 24.7%.

These early examples were selected from the all-layer scan for the strongest period-10 improvement in each model. The repetition follows decimal structure and is measurable beyond a favorable viewing angle.

However, the ten units-digit means do not sit on perfect ellipses. A single period-10 ellipse explains only **52%, 60%, and 38%** of their centroid spread in those three examples. Some additional structure follows other decimal harmonics and irregular digit-specific offsets.

![Observed digit means and fitted ellipses](periodic_digit_shapes.png)

At late layers, separate digit categories describe the visible geometry much better than the tested repeated-wave descriptions:

{fits}
All values in this table are held-out **3D** variance explained on four-digit targets. Repeated waves use periods 2, 5, 10, 100, and 1000. The broad-bend version adds period 10000; over 1000–9999, that is less than one full turn, so a good fit would not demonstrate repeated winding. Digit categories have more coefficients, so their advantage supports a useful description rather than an equal-complexity model-selection theorem.

**Interpretation:** there is evidence for periodic digit features, and for curved numerical layouts. There is not yet evidence that all the points form one regular helix, let alone two coherent strands. These tests do not rule out a useful helix in a different subspace or with a different parameterization. They test the particular numerical-periodic descriptions above in the saved top-three-PC views.

## 3. What the shapes look like across the layers

| Model | Early / middle observations | Late observations |
|---|---|---|
| Crystal | L3–4 show separated bodies; L5–9 show repeated curved groups with substantial digit-length separation. The middle layers become more diffuse. | L26 onward develops strong leading-digit patches. L27–31 have clearly separated groups; L32 becomes more elongated and mixed in the displayed subspace. |
| StarCoder | Early clouds develop digit structure; L10–20 often show bent or fan-like arrangements. Around L20–24, leading-digit groups occupy a curved, partly ring-like arrangement. | L25–35 forms separated patches whose relative layout changes. L36 rearranges and broadens them; some camera angles produce an apparent loop. |
| OpenLLaMA | Early/middle views show changing lobes and extensions, with less stable smooth ordering than the clearest Crystal/StarCoder examples. | From roughly L16 onward, leading-digit patches become visible. L19–26 is an irregular multi-lobed arrangement; the last transition is less dramatic. |

These are visual descriptions of the saved projections, not automatically identified manifold classes. Colors were predefined by digit; no clustering algorithm created these labels.

**Crystal L7 is particularly instructive.** Changing the camera and coloring by digit length separates the apparent strands into number-length groups. Digit length accounts for **59.1% of the overall 3D variation** there. Within the four-digit group, first-digit means account for **76.3%** of the remaining 3D variation. This is a substantial explanation of its shape without needing a double helix.

![Crystal layer 7 from two viewpoints and with digit-length colors](crystal_07_views.png)

The black path joins means of successive 100-number intervals among 1000–9999. Those drawn lines are a numerical-order guide, not observed connections or PH cycles.

## 4. Some shapes match across models at different depths

**What we checked:** compare corresponding numbers after allowing translation, rotation/reflection, and uniform scale. To separate the broad trajectory from point-level variation, compare both all four-digit points and 90 means of successive 100-number intervals. Similarity is between 0 and 1; 1 means identical normalized coordinates after alignment.

{comparison}
**The clearest match is Crystal L25 / StarCoder L20.** Its bin-mean similarity is **0.978**, with point-level similarity **0.896**. Splitting each bin into two random halves gives **0.978 and 0.977** for the two averaged comparisons. This supports a stable broad resemblance rather than a single lucky camera angle.

![Crystal 25 and StarCoder 20 after alignment](aligned_shapes.png)

The OpenLLaMA pairings show why averaging and point-level comparisons should both be reported: broad means can agree substantially while individual clouds differ. These are the best pairs selected after searching all cross-layer pairs, not independently confirmed matches or p-values. The models also share the same target-specific demonstration prompts.

This answers the earlier question directly: **yes, there are similar numerical layouts across models without matching their layer numbers.** It does not imply equal high-dimensional topology.

## 5. The last-layer change is measurable in both spaces

The first-digit share of full hidden-space variation drops from second-last to final by **14.5 percentage points in Crystal, 8.2 in StarCoder, and 3.3 in OpenLLaMA**. In the displayed 3D space, the corresponding changes are: **97.6% → 69.3%, 97.7% → 82.7%, and 87.2% → 86.2%**.

That agrees with the observation that Crystal and StarCoder become more mixed visually at the end. The last saved transition includes **both the final transformer block and the model’s final normalization**. These data alone cannot assign the change specifically to normalization. The decisive follow-up would capture the same final-block output immediately before and after that normalization.

## 6. What the existing PH adds

The previous full-space PH analysis supports substantial late-layer leading-digit organization. It also found that Crystal L32 / StarCoder L36 have the closest normalized H1 Betti-curve profiles among all 1,152 Crystal/StarCoder layer pairings. Yet a different PH comparison, bottleneck distance, makes final Crystal/OpenLLaMA closer. Broad loop-count profiles and the hardest-to-match persistent features are different questions.

**A helix is primarily a geometric claim.** An ideal open, non-self-intersecting helix is topologically an interval: it has no intrinsic closed one-dimensional hole. A closed ring does have a loop. Thickening a sampled helix or building its Rips complexes can create loops at some distances, but their presence does not uniquely identify a helix. Our saved H0/H1 diagrams do not supply strand assignments or verify the spatial winding of the points. H2 was not measured.

Thus, a visible spiral and a PH loop are not interchangeable findings. PH studies connectivity and holes across a range of distances; it does not label these data “DNA.” See the foundational [Ghrist survey on persistent topology](https://graphics.stanford.edu/courses/cs164-09-spring/Handouts/paper_barcodes.pdf).

## 7. How much trust to put in the 3D manifold picture

Across the 94 contextual layers, the saved three PCs retain roughly **18.0%–64.5%** of total hidden-space variance. Final-layer totals are **43.2% Crystal, 25.9% StarCoder, and 25.8% OpenLLaMA**.

Only a median **2.81% of the original four-nearest-neighbor IDs** remain among the nearest four in the 3D projection. This is a strict point-identity measure, and interchangeable near-neighbors can make it harsh; it does not mean the broad groups are meaningless. It does mean that apparent touching, tiny bridges, and small holes in the 3D view are unreliable substitutes for original-space topology.

![Fit quality and neighborhood retention over depth](fits_and_projection.png)

Independent PCA fits also choose different axes. A left-handed versus right-handed-looking twist, or a sudden change in camera-facing orientation, is not intrinsically meaningful. The three displayed axes alone cannot establish the intrinsic dimension of the full cloud.

## 8. The most useful mathematical description to test next

A strong candidate is an **additive digit representation**:

`hidden vector ≈ overall mean + first-digit contribution + hundreds contribution + tens contribution + units contribution + remainder`

Each digit contributes a learned set of positions. Combining these sets can create repeated bends, bands, offsets, and curved patches. Some digit contributions may contain circular Fourier structure. Their relative strength changes with layer depth.

This is more specific than saying the model makes an interesting shape, while accommodating the strong variation between layers. It does not make the cloud a torus: digit sets are discrete, and we have not established independent continuous circular coordinates.

The general idea is related to existing research. [Levy and Geva’s digit-representation paper](https://arxiv.org/abs/2410.11781) studies circular per-digit representations with causal interventions. [Kantamneni and Tegmark’s helix paper](https://arxiv.org/abs/2502.00873) fits linear and periodic number features and tests their causal role in addition. Those studies use different models, prompts, and sampled token positions. **A helix-like number code itself would not be a new literature claim.** The potential contribution here is the measured cross-model depth progression and how it interacts with copying and final-layer geometry.

## 9. Hypotheses ranked by the evidence here

| Hypothesis | Current assessment | Most informative next test |
|---|---|---|
| Representations shift from the last input digit toward the first expected output digit. | **Strong descriptive support in all three models, including original hidden space.** Causal interpretation remains open. | Repeat selected layers under copying, comparison, and arithmetic prompts; inspect attention/causal interventions at the same sampled position. |
| The spiral-like appearance partly combines decimal repetition with number-length and leading-digit offsets. | **Supported as an explanation of substantial measured structure.** A regular double helix is unestablished. | Plot fixed-length, fixed-leading-digit subsets and test phase/pitch consistency in more PCs and original space. |
| Different models pass through similar numerical layouts at different depths. | **Strong for the selected Crystal L25 / StarCoder L20 pair in 3D; weaker or more averaged for OpenLLaMA matches.** | Repeat with independent prompt seeds; compare matched-number distance matrices in full space. |
| Final normalization is a major cause of the late visual rearrangement. | **Plausible, not isolated by the saved layers.** The final transformer block also changes. | Compare pre-/post-normalization vectors from the same final-block computation. |
| These views reveal one universal helix or a common named topological manifold. | **Not established.** Candidate fits and PH summaries do not support that level of identification. | Fit competing manifolds with held-out checks, test more dimensions, and locate persistent cycles rather than relying on bar counts alone. |

**Recommended priority:** start with the last-digit-to-first-digit shift. It is already measurable across every model in the original space, and it gives a concrete mechanism to test. Use the apparent spirals to locate interesting examples, rather than making a named shape the required conclusion.

## Methods and limits

- **Coverage:** 97 views inspected; 94 contextual layers measured. Crystal/OpenLLaMA embedding vectors coincide. StarCoder’s embedding view depends on position and is constant within the four-digit stratum, so it is excluded from the numeric-shape regressions.
- **Input:** saved unwhitened, centered PCA scores; PC1/PC2 unchanged; previously validated PC3. Original hidden vectors are also used for the digit decomposition. Existing PH results are reused, not recomputed on PCA coordinates.
- **Digit decomposition:** one-way group-mean variance for each decimal place. In the balanced four-digit sample these effects are orthogonal and add to the additive model’s explained fraction. The original-space decomposition is descriptive/in-sample; held-out digit fits are separately reported for the 3D coordinates. Remaining variance includes digit interactions and context-dependent variation.
- **Regression:** ordinary least squares; five fixed folds; random-target and 100-number-block splits. Each leading-digit interval contributes two held-out blocks per fold. Tables use block splits. PCA itself was fitted on the full target collection, so these are held-out coefficient-fit checks in a fixed descriptive coordinate system, not a fully independent end-to-end representation benchmark.
- **Period search:** the stated seven periods and two combined bases only. Period 2 has a cosine feature only because its sine vanishes on integer targets. No exhaustive optimization over arbitrary pitches, axes, or nonlinear parameterizations was attempted.
- **Shape alignment:** Procrustes similarity, the sum of singular values of the cross-product of centered, unit-Frobenius-norm clouds. Allows reflection. It is a descriptive alignment score, not an explained-variance percentage. Pair selection is exploratory.
- **Dependence:** one prompt context per target, shared across models. Held-out targets and split-bin checks are not independent prompt-seed replication. These observations do not identify whether numerical semantics, tokenization, copying behavior, or their combination causes each pattern.
- **Preservation:** analysis files are separate; existing hidden states, 2D/3D galleries, and PH outputs are unchanged. No GPUs or Modal jobs were used.

[Detailed 3D metrics and all candidate fits](metrics.json) · [All original-space digit effects](hidden_digit_contributions.json) · [Digit centroids and circular fits](digit_contributions.json) · [Cross-model shape comparisons](shape_comparisons.json) · [Checks](validation.json)
'''
(O/'FINDINGS.md').write_text(text)
md=MarkdownIt('commonmark',{'html':True}).enable('table')
body=md.render(text)
atlas='<h2 id="atlas">Visual atlas: all 97 saved levels</h2><p>All points are shown. Colors identify leading digits 1–9; each layer has its own scale and PCA frame. Click an image to enlarge it.</p>'
for m,name,*_ in models:
 for p in sorted((O/'atlas').glob(m+'_*.png')):
  atlas+=f'<details><summary>{name} · sheet {p.stem.rsplit("_",1)[1]}</summary><a href="atlas/{p.name}"><img loading="lazy" src="atlas/{p.name}" alt="{name} layer atlas"></a></details>'
rows=''
for r in R:
 label='Embedding' if not r['layer'] else f'Layer {r["layer"]}'
 url=f'../pca3d/index.html#model={r["model"]}&layer={r["layer"]}'
 if not r['layer']:summary='Embedding-level control; excluded from four-digit shape fits.'
 else:
  d=next(x for x in D if x['model']==r['model'] and x['layer']==r['layer']);h=hidden(r['model'],r['layer']);dominant=max(h['digit_variance_fractions'],key=h['digit_variance_fractions'].get)
  summary=f'Largest digit effect in full space: {dominant}; first digit {h["digit_variance_fractions"]["thousands"]:.1%}, last digit {h["digit_variance_fractions"]["units"]:.1%}.'
 rows+=f'<tr><td>{r["name"]}</td><td><a href="{url}">{label}</a></td><td>{r["variance3"]:.1%}</td><td>{summary}</td></tr>'
alltable='<details id="layers"><summary>Every saved layer: measurements and direct 3D links</summary><div class="tablewrap"><table><thead><tr><th>Model</th><th>View</th><th>3D variance</th><th>Four-digit observation</th></tr></thead><tbody>'+rows+'</tbody></table></div></details>'
css='''body{font-family:system-ui,-apple-system,sans-serif;color:#1e293b;background:#f7f8fa;margin:0;line-height:1.65}main{max-width:1100px;margin:auto;background:white;padding:42px 48px 70px}h1{font-size:38px;line-height:1.15;letter-spacing:-.8px}h2{font-size:25px;line-height:1.3;margin-top:48px;padding-top:18px;border-top:1px solid #dce2ea}p,li{max-width:1000px}a{color:#1763a4}img{display:block;max-width:100%;height:auto;margin:22px auto}table{width:100%;border-collapse:collapse;margin:22px 0;font-size:14px}th,td{padding:11px 12px;border:1px solid #dde3eb;text-align:left;vertical-align:top}th{background:#edf3f8}tr:nth-child(even){background:#f9fafb}code{background:#eef2f6;padding:2px 5px;border-radius:4px;white-space:normal}details{border:1px solid #d4dce6;border-radius:8px;margin:16px 0;padding:15px}summary{cursor:pointer;font-weight:650}nav{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}.tablewrap{overflow:auto}footer{margin-top:40px;color:#627083;font-size:13px}@media(max-width:700px){main{padding:22px 18px}h1{font-size:30px}table{font-size:12px}td,th{padding:8px}}'''
(O/'index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>3D number shapes — observations and hypotheses</title><style>'+css+'</style></head><body><main><nav><a href="../pca3d/index.html">Rotate the 3D graphs</a><a href="#atlas">All-layer atlas</a><a href="#layers">Every layer</a><a href="FINDINGS.md">Download report</a></nav>'+body+atlas+alltable+'<footer>Analysis completed 20 September 2026. Saved data only; no cloud inference.</footer></main></body></html>')
print('Report and reading page written.')
