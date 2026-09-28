# Europe T2M: trained model and verification data

Files of the Europe T2M configuration of the paper (case `Y621d`, Sections 4a and 5, Figs 6–8, 12–13, Table 2).

| File | Content |
|---|---|
| `DeepSeasons_EuropeT2M_weights.pt` | PyTorch `state_dict` of the trained network (4.5 MB, 1.10 M parameters) |
| `DeepSeasons_EuropeT2M_verification.pkl` | verification of the 55 test forecasts (initial dates in the test period 2016–2022): spatial ACC and MSE per initial date and lead time for DeepSeasons (`corr`, `rms`), persistence (`pers`, `rmsper`) and the CMCC operational system (`GCMcorr`, `GCMrms`), plus the full configuration (`params`) |
| `DeepSeasons_EuropeT2M_benchmark_AR4.pkl` | same verification for the AR4 linear benchmark (Section 2b) |
| `DeepSeasons_EuropeT2M_benchmark_ridge.pkl` | same for the ridge benchmark on the full input window |
| `DeepSeasons_EuropeT2M_ACC_mean.csv`, `..._MSE_mean.csv` | mean scores per lead (DeepSeasons, persistence, GCM) |
| `DeepSeasons_EuropeT2M_IG_attributions.npz` | Integrated Gradients attributions (Section 5): array `attr` of shape (lead, output feature, case, input month, input feature) for leads 1, 3, 6, 12, with the feature layout and the deterministic forecast |

## Model configuration

Input T850 (5 EOFs) and T2M (5 EOFs), 75 % of the variance each, over Europe, 10 features, input window 18 months (context 14 + 4 lags), lags 1–4, `d_model = d_ff = 256`, 8 heads, 1 encoder and 1 decoder layer, Student‑t output, discount γ = 0.9, MaxMin scaling fitted on 1940–2005, EOFs and climatology on 1940–2015. Trained on 1940–2005, validated on 2006–2015, tested on 2016–2022 (see Table S1 of the Supplementary Material).

## Loading the model

The weights match the `InformerConfig` built in `notebooks/SST-Y621d-MONT.ipynb`. To reuse them without retraining, run the notebook with `params['RESTART'] = True` and copy this file to `$DEEPSEASONS_WORKDIR/MOD_informer_Y621d_anom_1_256_18_False.pt` (the name the notebook expects), or load it directly:

```python
model = zcf.ClimFormer(configuration)                 # configuration as in the notebook
model.load_state_dict(torch.load('models/europe_t2m/DeepSeasons_EuropeT2M_weights.pt', map_location='cpu'))
```

The EOF basis and the MaxMin scaling are recomputed by the notebook from the ERA5 data, so the input data are still needed to produce forecasts.

## Reading the verification data

```python
import pickle, numpy as np
d = pickle.load(open('models/europe_t2m/DeepSeasons_EuropeT2M_verification.pkl', 'rb'))
mse = d['rms']           # (13 leads incl. lead 0, 55 initial dates), area-mean squared error in K^2
print(mse[1:].mean(axis=1))   # mean MSE at leads 1..12 (Table 2 of the paper)
```

Reading the pickles requires the bundled `lib/AIModels` on `sys.path` (they contain `AIModels.AIClasses.Field` objects) and `torch`. See `models/global_sst/README.md` for the layout of the IG attribution file.
