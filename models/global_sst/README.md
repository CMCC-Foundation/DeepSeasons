# Global SST case: trained model and verification data

Files of the global SST configuration of the paper (case `Y620d`, Sections 3 and 5, Figs 2–5, 12–13, Tables 1–2).

| File | Content |
|---|---|
| `DeepSeasons_GlobalSST_weights.pt` | PyTorch `state_dict` of the trained network (8.8 MB, 2.27 M parameters) |
| `DeepSeasons_GlobalSST_verification.pkl` | verification of the 25 test forecasts (December 2019 – December 2021): spatial ACC and MSE per initial date and lead time for DeepSeasons (`corr`, `rms`), persistence (`pers`, `rmsper`) and the CMCC operational system (`GCMcorr`, `GCMrms`), plus the full configuration (`params`) |
| `DeepSeasons_GlobalSST_benchmark_AR4.pkl` | same verification for the AR(4) linear benchmark (Section 2b) |
| `DeepSeasons_GlobalSST_benchmark_ridge.pkl` | same for the ridge benchmark on the full input window |
| `DeepSeasons_GlobalSST_ACC_mean.csv`, `..._MSE_mean.csv` | mean scores per lead (DeepSeasons, persistence, GCM) |
| `DeepSeasons_GlobalSST_IG_attributions.npz` | Integrated Gradients attributions (Section 5): array `attr` of shape (lead, output feature, case, input month, input feature) for leads 1, 3, 6, 12, with the feature layout and the deterministic forecast |

## Model configuration

Input SST (15 EOFs, 60S–60N) and U850 (25 EOFs), 40 features, input window 48 months (context 44 + 4 lags), lags 1–4, `d_model = d_ff = 256`, 8 heads, 2 encoder and 2 decoder layers, Student‑t output, discount γ = 0.9, MaxMin scaling fitted on 1940–2005, EOFs and climatology on 1940–2015. Trained on 1940–2005, validated on 2006–2015, tested on 2016–2022 (see Table S1 of the Supplementary Material).

## Loading the model

The weights match the `InformerConfig` built in `notebooks/SST-Y620d.ipynb`. To reuse them without retraining, run the notebook with `params['RESTART'] = True` and copy this file to `$DEEPSEASONS_WORKDIR/MOD_informer_Y620d_anom_2_256_48_False.pt` (the name the notebook expects), or load it directly:

```python
model = zcf.ClimFormer(configuration)                 # configuration as in the notebook
model.load_state_dict(torch.load('models/global_sst/DeepSeasons_GlobalSST_weights.pt', map_location='cpu'))
```

The EOF basis and the MaxMin scaling are recomputed by the notebook from the ERA5 data, so the input data are still needed to produce forecasts.

## Reading the verification data

```python
import pickle, numpy as np
d = pickle.load(open('models/global_sst/DeepSeasons_GlobalSST_verification.pkl', 'rb'))
acc = d['corr']          # (13 leads incl. lead 0, 25 initial dates)
print(acc[1:].mean(axis=1))   # mean ACC at leads 1..12 (Table 2 of the paper)
```

Reading the pickles requires the bundled `lib/AIModels` on `sys.path` (they contain `AIModels.AIClasses.Field` objects) and `torch`. The IG file is a plain `numpy.savez` archive and can be read with `numpy.load(..., allow_pickle=True)`; see `lib/ig_attribution.py` (`load`, `aggregate`) for the aggregations used in the figures.
