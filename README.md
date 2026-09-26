# AdvLS-Reimplementation

Digital-only reimplementation of **AdvLS** (Hu, Wang, Tiliwalidi & Li, *"Adversarial Laser Spot: Robust and Covert Physical-World Attack to DNNs,"* ACML 2022) — a method that fools image classifiers by projecting small colored laser spots onto objects, optimized via a genetic algorithm.

Rather than using physical laser pointers as the original paper does, this reimplementation renders simulated laser spots directly onto digital photographs, keeping the entire pipeline — including the "physical-world" component — inside a software simulation.

## Setup

| Component | Choice |
|---|---|
| Target classifier | ResNet50, pretrained on ImageNet (`torchvision`, `IMAGENET1K_V2` weights) |
| Dataset | ImageNet-mini (Kaggle), `val` split |
| Correctly-classified image pool | 3,122 images (out of 3,923 checked — ~79.6% baseline accuracy) |
| Hardware | CPU only (no GPU) |
| Language / libraries | Python, PyTorch, torchvision, Pillow, NumPy |

## Pipeline (8 steps)

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_setup_v2.py`, `step1_setup_v3.py` | Load pretrained ResNet50, filter ImageNet-mini down to correctly-classified images |
| 2 | `step2_spots.py` | Parameterize a laser spot θ = (m, n, r, g, b) and spot groups Gθ; flatten/rebuild to/from GA chromosomes |
| 3 | `step3_fusion.py` | Fusion function S(X, Gθ): render Gaussian-blurred, alpha-blended glowing spots onto a clean image |
| 4 | `step4_ga.py` (uses `ga_core.py`) | Genetic algorithm search: selection (worst 10% replaced by best 10%), crossover (P=0.7), mutation (P=0.1), plus elitism (bug fix — see below) |
| 5 | `step5_batch_eval.py` | Batch attack evaluation across color modes (random/red/green/blue), tracking ASR and queries-to-success — mirrors the paper's Table 1 |
| 6 | `step6_spots_ablation.py` | Ablation over number of spots (5/20/50) — mirrors Figure 8 |
| 7 | `step7_transferability.py` | Black-box transfer of successful attacks to VGG19, MobileNet, DenseNet — mirrors the paper's transfer tables |
| 8 | `step8_misclassification.py` | Semantic analysis of which wrong labels attacks land on — mirrors Figures 5/9 |

`check_asr.py`, `check_transfer.py`, and `check_all_results.py` are helper scripts for summarizing the JSON result files.

### Bug found and fixed

The original GA loop discarded the entire population every generation via crossover/mutation with no elitism, so a strong candidate could be mutated away and lost. Adding elitism (carrying the single best-ever solution unchanged into the next generation) raised ASR on a fixed 5-image test set from 40% to 60% under identical settings.

## Results

### Color ablation — 30 images × 4 colors, 120 runs (35 spots, population 30, 100 generations)

| Color | ASR | Avg. queries (successes only) | Paper's ASR |
|---|---|---|---|
| Random | 46.7% | 603.6 | 75.8% |
| Red | 53.3% | 275.6 | 82.6% |
| Green | **56.7%** | 385.3 | 87.7% |
| Blue | 53.3% | 246.5 | 78.7% |

### Number-of-spots ablation — 15 images × 3 counts, 45 runs (green)

| Spots | ASR | Avg. queries (successes only) |
|---|---|---|
| 5 | 33.3% | 334.0 |
| 20 | 40.0% | 147.2 |
| 50 | **53.3%** | 206.0 |

Paper's finding (Figure 8): ASR rises steadily with more spots, from ~40% at 5 spots to ~85–90% approaching 50–100 spots. Trend direction matches.

### Attack transferability — 15 images (green, 35 spots)

Source (ResNet50) ASR: 53.3% (8/15). Of those 8 successes:

| Target model | Transfer rate |
|---|---|
| VGG19 | 75.0% |
| MobileNet | 62.5% |
| DenseNet | 50.0% |

Paper's reported VGG19 transfer: 83.2%.

### Misclassification semantics — 40 images (green, 35 spots)

ASR: 47.5% (19/40). Most frequent wrong label: **"croquet ball"** (3/19, 15.8%) — small, round, and brightly colored, consistent with the paper's finding that classifiers confuse laser-spotted images for objects like "Bubble," "Envelope," and "Petri dish."

## What matched vs. what didn't

**Matched qualitatively, in every experiment:**
- Fixed spot colors outperform random color
- More spots → higher attack success rate
- Successful attacks transfer to unrelated architectures at meaningfully high rates
- Misclassifications cluster around small, round, "spot-like" object categories

**Did not match quantitatively:**
- Absolute ASR was consistently lower than the paper's (~40–57% here vs. 75–95% reported)
- Query efficiency was worse (100s–600s avg. vs. the paper's ~140–240)

**Likely cause:** compute budget. This reimplementation ran on CPU with population 20–30, up to 100 generations (≤3,000 queries/image), and sample sizes of 15–40 images per experiment. Reproducing the paper's exact scale (up to 1,000 images, spot sweeps to 100) was estimated at 5+ days of continuous CPU runtime, suggesting the original used GPU acceleration and/or a much larger search budget. A ~200-images-per-experiment run was scoped but not completed (est. ~1–1.5 days/experiment on this hardware).

## Repository contents

- `advls_reimplementation_report.docx` — full write-up of the project
- `step1_setup_v2.py`, `step1_setup_v3.py` — data setup
- `step2_spots.py` — laser spot parameterization
- `step3_fusion.py` — fusion/rendering function
- `ga_core.py` — genetic algorithm operators (crossover/mutation/selection)
- `step4_ga.py` — full GA attack loop
- `step5_batch_eval.py` — color ablation
- `step6_spots_ablation.py` — spot-count ablation
- `step7_transferability.py` — cross-model transfer
- `step8_misclassification.py` — semantic analysis
- `check_asr.py`, `check_transfer.py`, `check_all_results.py` — result-summary helpers
- `*.json` — result files (ASR comparisons, transferability, misclassification tallies, etc.)
- `adversarial_*.png` — sample adversarial images produced by the attack

## Usage

```bash
pip install torch torchvision pillow numpy

# 1. Build the pool of correctly-classified images
python step1_setup_v3.py --image_dir <imagenet-mini/val> --class_index <imagenet_class_index.json> --output correctly_classified.json

# 2. Preview a single adversarial example
python step3_fusion.py --image <path/to/image.jpg> --num_spots 20 --output adversarial_sample.png

# 3. Run an experiment, e.g. the color ablation
python step5_batch_eval.py --manifest correctly_classified.json
```

See each script's docstring for full CLI options.

## Citation

Hu, W., Wang, X., Tiliwalidi, K., & Li, H. (2022). *Adversarial Laser Spot: Robust and Covert Physical-World Attack to DNNs.* ACML 2022.

This is an independent, unofficial reimplementation for research/learning purposes and is not affiliated with the original authors.
