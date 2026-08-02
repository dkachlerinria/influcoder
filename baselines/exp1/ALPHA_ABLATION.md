# InfluCoder Alpha Ablation (Pearson vs KL Divergence)

This experiment sweeps the `alpha` hyperparameter in `pearson_kl_loss` for the `jhu-clsp/ettin-encoder-68m` model on the InfluCoder task. The loss is defined as a weighted combination of Pearson correlation and KL divergence. `alpha = 0.0` is pure KL Divergence, and `alpha = 1.0` is pure Pearson Correlation.

## Methodology
- Model: `jhu-clsp/ettin-encoder-68m`
- Seeds: `[0, 1, 2]`
- Epochs: `4` (early stopping evaluation)
- Output Location: `baselines/out/fig1_dolci/default/alpha_ablation.json`

## Results
Averages and sample standard deviations (±) across the 3 seeds, evaluated at Epoch 4 (Final) and at the Best Checkpoint (Peak).

| Alpha | Avg Final (Epoch 4) | Std Dev (±) | Avg Peak Checkpoint | Checkpoint Epoch Trend |
|:---|:---|:---|:---|:---|
| **0.0** (Pure KL) | `+0.7653` | `0.0217` | `+0.7858` | Peaked around epoch 2-3 |
| **0.25** | `+0.7637` | `0.0172` | `+0.7775` | Peaked around epoch 3 |
| **0.5** (Baseline) | `+0.7699` | `0.0042` | `+0.7737` | Highly variable (ep 1, 3, or 4) |
| **0.75** | `+0.7658` | `0.0145` | `+0.7754` | Peaked around epoch 2-3 |
| **1.0** (Pure Pearson)* | **`+0.7882`** | **`0.0007`** | **`+0.7882`** | **Peaked consistently at epoch 4** |

*\*Note: `alpha=1.0` is based on 2 seeds (walltime cut off the 3rd seed).*

## Key Findings
1. **`alpha = 1.0` wins outright:** Pure Pearson correlation (with no KL divergence penalty) achieved the highest peak score (`+0.7882`), the highest final score, and drastically lowered variance between seeds (`±0.0007`).
2. **Early stopping vs. Final:** Lowering the alpha forces the model to overfit faster on this dataset, causing the validation correlation to peak at epoch 2 or 3 before degrading by epoch 4. Pure Pearson (`alpha=1.0`), however, steadily improved through all 4 epochs without degrading.
