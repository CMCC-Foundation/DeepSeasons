# North America T2M: trained model and verification data

Files of the North America T2M configuration of the paper (case `Y622d`, Section 4b, Figs 9–11, Table 2).

| File | Content |
|---|---|
| `DeepSeasons_NorthAmericaT2M_weights.pt` | PyTorch `state_dict` of the trained network (34.6 MB, 8.61 M parameters) |
| `DeepSeasons_NorthAmericaT2M_verification.pkl` | verification of the 55 test forecasts (initial dates in the test period 2016–2022): spatial ACC and MSE per initial date and lead time for DeepSeasons (`corr`, `rms`), persistence (`pers`, `rmsper`) and the CMCC operational system (`GCMcorr`, `GCMrms`), plus the full configuration (`params`) |
| `DeepSeasons_NorthAmericaT2M_benchmark_AR2.pkl` | same verification for the AR2 linear benchmark (Section 2b) |
| `DeepSeasons_NorthAmericaT2M_benchmark_ridge.pkl` | same for the ridge benchmark on the full input window |
| `DeepSeasons_NorthAmericaT2M_ACC_mean.csv`, `..._MSE_mean.csv` | mean scores per lead (DeepSeasons, persistence, GCM) |

## Model configuration

Input tropical SST (29 EOFs, 75 % of the variance) and T2M over North America (5 EOFs), 34 features, input window 18 months (context 16 + 2 lags), lags 1–2, `d_model = d_ff = 512`, 8 heads, 2 encoder and 2 decoder layers, Student‑t output, discount γ = 0.9, MaxMin scaling fitted on 1940–2005, EOFs and climatology on 1940–2015. Trained on 1940–2005, validated on 2006–2015, tested on 2016–2022 (see Table S1 of the Supplementary Material).

## Loading the model

The weights match the `InformerConfig` built in `notebooks/SST-Y622d-MONT.ipynb`. To reuse them without retraining, run the notebook with `params['RESTART'] = True` and copy this file to `$DEEPSEASONS_WORKDIR/MOD_informer_Y622d_anom_2_512_18_False.pt` (the name the notebook expects), or load it directly:

```python
model = zcf.ClimFormer(configuration)                 # configuration as in the notebook
model.load_state_dict(torch.load('models/north_america_t2m/DeepSeasons_NorthAmericaT2M_weights.pt', map_location='cpu'))
```

The EOF basis and the MaxMin scaling are recomputed by the notebook from the ERA5 data, so the input data are still needed to produce forecasts.

## Reading the verification data

```python
import pickle, numpy as np
d = pickle.load(open('models/north_america_t2m/DeepSeasons_NorthAmericaT2M_verification.pkl', 'rb'))
mse = d['rms']           # (13 leads incl. lead 0, 55 initial dates), area-mean squared error in K^2
print(mse[1:].mean(axis=1))   # mean MSE at leads 1..12 (Table 2 of the paper)
```

Reading the pickles requires the bundled `lib/AIModels` on `sys.path` (they contain `AIModels.AIClasses.Field` objects) and `torch`. See `models/global_sst/README.md` for the layout of the IG attribution file.
