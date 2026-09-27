#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
linear_baselines.py
===================

Linear benchmarks for DeepSeasons (revision REV4, Editor point 6 and autocorrelation point 2).

Two baselines are fitted on exactly the same EOF coefficient sequences, the same
train / validation / test split and the same MaxMin scaling used by DeepSeasons:

  RIDGE  -- the "linear DeepSeasons": a direct multi-output linear map from the whole
            input sequence (TIN months x all EOF features, plus the 3 time features)
            to all target PCs at all Tpredict leads.  Ridge penalty chosen on the
            validation period (2006-2015).  Replaces the ill-posed OLS used in the
            SST-W*.ipynb notebooks (P > N, minimum-norm solution).

  AR     -- a lagged multivariate autoregression (direct, one map per lead):
            y(t+h) = A_h [x(t), x(t-1), ..., x(t-p+1)],  p = max(LAGS) of the DS case.
            No time features.  This is the "linear autocorrelation" benchmark:
            what can be predicted from the recent memory of the PCs alone.
            Ridge-regularised too (tiny penalty grid), alpha chosen on validation.

The data pipeline (cells 2-30 of SST-W620d.ipynb) and the verification pipeline
(cells 48-83) are reproduced verbatim, so the results are directly comparable with the
DeepSeasons pickles  Y620d_V100_1_corr_data  etc.

Output (in --wkdir):
    {case}_RIDGE_V{subcase_var}_{subcase}_corr_data     pickle, same keys as DS pickles
    {case}_AR_V{subcase_var}_{subcase}_corr_data         pickle
    Baselines_{case}_scores.csv                          mean/median ACC and MSE per lead
    Baselines_{case}_alpha.csv                           validation curve used to pick alpha

Usage (on the Mac, in the environment used for the notebooks):
    python linear_baselines.py --case Y620d
    python linear_baselines.py --case Y621d
    python linear_baselines.py --case Y621s
    python linear_baselines.py --case Y622d
    python linear_baselines.py --case all

Paths are taken from lib/dspaths.py (env vars DEEPSEASONS_DATA / DEEPSEASONS_WORKDIR).
"""

import argparse
import os
import sys
import pickle
import datetime

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Paths (identical to the notebooks)
# ----------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'lib'))
import dspaths                        # noqa: E402  (repository path configuration)
dspaths.setup(chdir=False, verbose=False)

import AIModels.AIutil as zai          # noqa: E402
import AIModels.AIClasses as zaic      # noqa: E402
import AIModels.ClimFormer as zcf      # noqa: E402

today_date = datetime.date.today().strftime("%Y-%m-%d")

# ----------------------------------------------------------------------------
# Final DeepSeasons configurations (taken from the Y* notebooks / Y*_Score CSVs,
# i.e. the runs actually shown in the paper).  Only what defines the predictor
# matters here: fields, EOF truncation, TIN, LAGS, SMOOTH.
# ----------------------------------------------------------------------------
def field_defs():
    return {
        # Global SST case (Fig. 1)
        'Y620d': dict(
            SMOOTH=False, TIN=48, LAGS=[1, 2, 3, 4], Score_Field='SSTSST',
            subcase=1, subcase_var=100,
            InputVars=lambda: [zaic.Field('SST', ['SST'], 'WORLD', 15, dropX=False),
                               zaic.Field('U', ['850'], 'WORLD', 25, dropX=False)],
        ),
        # Europe T2M monthly (Fig. 5)
        'Y621d': dict(
            SMOOTH=False, TIN=18, LAGS=[1, 2, 3, 4], Score_Field='T2MT2M',
            subcase=1, subcase_var=100,
            InputVars=lambda: [zaic.Field('T', ['850'], 'EUROPE', 0.75, dropX=True),
                               zaic.Field('t2m', ['t2m'], 'EUROPE', 0.75)],
        ),
        # Europe T2M seasonal (three-month running mean) (Fig. 11)
        'Y621s': dict(
            SMOOTH=True, TIN=18, LAGS=[1, 2, 3, 4], Score_Field='T2MT2M',
            subcase=1, subcase_var=299,
            InputVars=lambda: [zaic.Field('T', ['850'], 'EUROPE', 0.95, dropX=True),
                               zaic.Field('t2m', ['t2m'], 'EUROPE', 0.95)],
        ),
        # North America T2M monthly (Fig. 8)
        'Y622d': dict(
            SMOOTH=False, TIN=18, LAGS=[1, 2], Score_Field='T2MT2M',
            subcase=1, subcase_var=100,
            InputVars=lambda: [zaic.Field('SST', ['SST'], 'TROPIC', 0.75, dropX=False),
                               zaic.Field('t2m', ['t2m'], 'NORTH_AMERICA', 5)],
        ),
    }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def dataloader_to_xy(dataloader, mode='full', p=None):
    """
    Convert a TimeSeriesDataset loader into (X, y).

    mode='full' : X = whole input sequence (TIN x MIN) flattened + past time features
                  flattened  -> RIDGE predictor
    mode='lags' : X = last p months of the input sequence flattened, no time features
                  -> AR(p) predictor
    y           : (N, Tpredict, K) targets in scaled space (all leads kept)
    """
    X_list, y_list = [], []
    for batch in dataloader:
        src, tgt, pasft = batch[0], batch[1], batch[2]
        B = src.shape[0]
        if mode == 'full':
            X = torch.cat([src.reshape(B, -1), pasft.reshape(B, -1)], dim=1)
        elif mode == 'lags':
            X = src[:, -p:, :].reshape(B, -1)
        else:
            raise ValueError(mode)
        X_list.append(X.detach().cpu().numpy())
        y_list.append(tgt.detach().cpu().numpy())
    return np.vstack(X_list), np.vstack(y_list)


def fit_ridge_select_alpha(Xtr, ytr, Xva, yva, alphas, label=''):
    """
    Standardise predictors on the training set, fit Ridge for each alpha on the
    training set only, pick the alpha with the lowest validation MSE (all leads,
    all features, scaled space).  Returns fitted (scaler, model, table).
    """
    N, T, K = ytr.shape
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    ytr2 = ytr.reshape(N, T * K)
    rows = []
    best = (np.inf, None)
    for a in alphas:
        m = Ridge(alpha=a, fit_intercept=True).fit(Xtr_s, ytr2)
        pva = m.predict(Xva_s).reshape(-1, T, K)
        ptr = m.predict(Xtr_s).reshape(-1, T, K)
        mse_va = float(np.mean((pva - yva) ** 2))
        mse_tr = float(np.mean((ptr - ytr) ** 2))
        # validation MSE per lead, useful diagnostics
        mse_va_lead = np.mean((pva - yva) ** 2, axis=(0, 2))
        rows.append(dict(model=label, alpha=a, mse_train=mse_tr, mse_val=mse_va,
                         **{f'mse_val_M{h+1}': v for h, v in enumerate(mse_va_lead)}))
        if mse_va < best[0]:
            best = (mse_va, a)
    a_best = best[1]
    model = Ridge(alpha=a_best, fit_intercept=True).fit(Xtr_s, ytr2)
    print(f'[{label}] N_train={N}, P={Xtr.shape[1]}, chosen alpha={a_best:g}, '
          f'val MSE={best[0]:.4f}')
    return sc, model, pd.DataFrame(rows), a_best


def predict(sc, model, X, T, K):
    return model.predict(sc.transform(X)).reshape(-1, T, K)


# ----------------------------------------------------------------------------
# Main per-case routine
# ----------------------------------------------------------------------------
def run_case(case, wkdir, ridge_alphas, ar_alphas, seasonal_offset=True):
    cfg = field_defs()[case]
    os.chdir(wkdir)
    print(f'\n{"="*78}\nCASE {case}   working directory {os.getcwd()}\n{"="*78}')

    SMOOTH = cfg['SMOOTH']
    normalization = 'anom'
    only_valtrain_eof = True
    clim_train = True
    eof_interval = ['1940-03-01', '2015-12-01'] if SMOOTH else ['1940-01-01', '2015-12-01']
    clim_int = eof_interval if clim_train else None
    InputVars = cfg['InputVars']()
    Score_Field = cfg['Score_Field']
    LAGS = cfg['LAGS']
    TIN = cfg['TIN']
    subcase, subcase_var = cfg['subcase'], cfg['subcase_var']
    device = torch.device('cpu')
    t_type = torch.float32
    torch.set_default_dtype(t_type)

    # ---- Data base: EOFs on training+validation period, anomalies wrt training clim
    invar_dict = {'period': 'ANN', 'version': 'V5', 'SMOOTH': SMOOTH, 'normalization': normalization,
                  'detrend': False, 'eof_interval': eof_interval, 'clim_eof_interval': clim_int,
                  'shift': 'ERA5', 'case': case, 'datatype': 'Source_data', 'location': dspaths.DATA_DIR}
    INX = zai.make_data_base(InputVars, **invar_dict)
    PDX = INX  # predict the same fields that are used as input

    fd = INX[Score_Field]['X'].A.time.data[0]
    fl = INX[Score_Field]['X'].A.time.data[-1]
    data_time = pd.date_range(start=fd, end=fl, freq='1MS')

    num_features_src, _ = zai.make_features(INX)
    num_features_tgt, _ = zai.make_features(PDX)

    # ---- params (only the entries the pipeline needs)
    params = {'TIN': TIN, 'MIN': num_features_src, 'T': 12, 'K': num_features_tgt, 'Tpredict': 12,
              'scaling': 'MaxMin', 'lags': LAGS, 'device': device, 't_type': t_type,
              'FFN_DIM': 0, 'ENC_Heads': 0, 'enc_dec_layers': 0, 'D_DIM': 0}
    Tpredict, T, K, MIN = params['Tpredict'], params['T'], params['K'], params['MIN']

    start_date = data_time.strftime('%Y-%m-%d')[0]
    end_training, start_val, end_val = '2005-12-01', '2006-01-01', '2015-12-01'
    test_start, test_end = '2016-01-01', '2022-12-01'
    idx = {k: data_time.get_loc(v) for k, v in
           dict(train_period_start=start_date, train_period_end=end_training,
                val_period_start=start_val, val_period_end=end_val,
                test_period_start=test_start, test_period_end=test_end).items()}
    params |= idx
    params |= {'start_date': start_date, 'end_training': end_training, 'start_val': start_val,
               'end_val': end_val, 'test_start': test_start, 'test_end': test_end}
    params['train_first_fcs'] = TIN
    params['train_last_fcs'] = params['train_period_end'] - Tpredict + 1
    params['val_first_fcs'] = params['val_period_start'] + TIN
    params['val_last_fcs'] = params['val_period_end'] - Tpredict + 1
    params['test_first_fcs'] = params['test_period_start'] + TIN
    params['test_last_fcs'] = params['test_period_end'] - Tpredict + 1

    train_size = params['train_period_end'] - params['train_period_start'] + 1
    val_size = params['val_period_end'] - params['val_period_start'] + 1
    tot_size = params['test_period_end'] - params['train_period_start'] + 1

    # ---- Scaled data, datasets, loaders (as in the notebooks)
    train_data, val_data, test_data, _ = zai.make_data(INX, params)
    tgt_train_data, tgt_val_data, tgt_test_data, _ = zai.make_data(PDX, params)

    tim_train = zai.create_time_features(data_time[:train_size + 1], 0, device)
    tim_val = zai.create_time_features(data_time[train_size:train_size + val_size + 1], 0, device)
    tim_test = zai.create_time_features(data_time[train_size + val_size:tot_size + 1], 0, device)

    dl = {}
    for name, (src, tgt, tf) in dict(train=(train_data, tgt_train_data, tim_train),
                                     val=(val_data, tgt_val_data, tim_val),
                                     test=(test_data, tgt_test_data, tim_test)).items():
        ds = zcf.TimeSeriesDataset(src, tgt, TIN, MIN, T, K, time_features=tf)
        dl[name] = DataLoader(ds, batch_size=64, shuffle=False)

    # ---- Build predictor matrices
    p_ar = max(LAGS)
    XY = {}
    for mode, p in (('full', None), ('lags', p_ar)):
        XY[mode] = {s: dataloader_to_xy(dl[s], mode=mode, p=p) for s in ('train', 'val', 'test')}
    ytr, yva, yte = (XY['full'][s][1] for s in ('train', 'val', 'test'))
    print(f'Samples: train {ytr.shape[0]}, val {yva.shape[0]}, test {yte.shape[0]};  '
          f'RIDGE predictors P={XY["full"]["train"][0].shape[1]},  AR({p_ar}) predictors '
          f'P={XY["lags"]["train"][0].shape[1]}, targets T x K = {T} x {K}')

    # ---- Fit both baselines
    results = {}
    alpha_tables = []
    for label, mode, alphas in (('RIDGE', 'full', ridge_alphas), ('AR', 'lags', ar_alphas)):
        Xtr, Xva, Xte = (XY[mode][s][0] for s in ('train', 'val', 'test'))
        sc, model, tab, a_best = fit_ridge_select_alpha(Xtr, ytr, Xva, yva, alphas, label=label)
        alpha_tables.append(tab)
        out_train = predict(sc, model, Xtr, T, K)
        out_val = predict(sc, model, Xva, T, K)
        out_test = predict(sc, model, Xte, T, K)
        results[label] = dict(out_train=out_train, out_val=out_val, out_test=out_test,
                              alpha=a_best, P=Xtr.shape[1])
    pd.concat(alpha_tables).to_csv(f'Baselines_{case}_alpha.csv', index=False)

    # ---- Verification on the grid, exactly as in the notebooks (cells 48-59)
    plfield = Score_Field
    indt2m = PDX[plfield]['index']
    startdate = data_time[params['test_first_fcs'] - 1]
    enddate = data_time[params['test_last_fcs'] - 1]
    truncated_verification = INX[plfield]['mr']

    score_rows = []
    for label, r in results.items():
        out_train_tot, out_val_tot, out_test_tot, true_tot = zai.rescale(
            params, PDX, r['out_train'], r['out_val'], r['out_test'], verbose=False)
        out_test_p = out_test_tot[..., indt2m]
        F, P, Obs = zai.eof_to_grid(Score_Field, out_test_p, startdate, enddate, INX=PDX,
                                    params=params, truncation=truncated_verification)
        Ft = zai.make_fcst_array(startdate, enddate, Tpredict + 1, F)
        Pt = zai.make_fcst_array(startdate, enddate, Tpredict + 1, P)
        Np, Nl, NumPoints = Ft.shape
        corrresult = np.zeros((Tpredict + 1, Np))
        persistence = np.zeros((Tpredict + 1, Np))
        rmsres = np.zeros((Tpredict + 1, Np))
        rmsper = np.zeros((Tpredict + 1, Np))
        rmsclim = np.zeros((Tpredict + 1, Np))   # climatology forecast = zero anomaly
        for i, ic in enumerate(Ft.time):
            ftmp = zai.select_fcst(ic, Ft)
            ptmp = zai.select_fcst(ic, Pt)
            otmp = Obs.sel(time=ftmp.time)
            corrresult[:, i] = xr.corr(otmp, ftmp, dim='z').data
            persistence[:, i] = xr.corr(otmp, ptmp, dim='z').data
            # NB: as in the notebooks this is the mean *squared* error (no sqrt)
            rmsres[:, i] = np.sum(abs(ftmp - otmp) ** 2, axis=1).data / NumPoints
            rmsper[:, i] = np.sum(abs(otmp - ptmp) ** 2, axis=1).data / NumPoints
            # MSE of the climatological (zero anomaly) forecast, i.e. the mean square
            # of the verified anomaly itself (bias^2 + variance over the test period)
            rmsclim[:, i] = np.sum(abs(otmp) ** 2, axis=1).data / NumPoints

        # ---- pickle with the same keys as the DS pickles (GCM keys left None)
        figinfield = list(INX.keys())
        idfig = (f"{label}: FIELDS={figinfield}, MIN={MIN}, MOUT={K}, in={TIN}, "
                 f"lags={LAGS}, alpha={r['alpha']:g}, P={r['P']}")
        data_dict = {"description": case, "subcase_var": subcase_var, "subcase": subcase,
                     "dataversion": 'V5', "invar_dict": invar_dict, "pred_dict": invar_dict,
                     "author": "AN", "date": today_date, "name": idfig, 'config': {'baseline': label},
                     'area': INX[plfield]['area'], "size_model": 0, "best_val_loss": 0.0,
                     "model_config": {'baseline': label, 'alpha': r['alpha'], 'P': r['P'],
                                      'p_ar': p_ar if label == 'AR' else None},
                     "params": params, "verification_field": plfield,
                     "input_fields": figinfield, "output_fields": list(PDX.keys()),
                     'eof': str([i.mr for i in InputVars]),
                     'GCMcorr': None, "corr": corrresult, "pers": persistence,
                     'GCMrms': None, "rms": rmsres, "rmsper": rmsper, "rmsclim": rmsclim}
        fileout = f'{case}_{label}_V{subcase_var}_{subcase}_corr_data'
        with open(fileout, "wb") as f:
            pickle.dump(data_dict, f)
        print(f'Written {fileout}')

        for stat, fn in (('mean', np.mean), ('median', np.median)):
            score_rows.append(dict(case=case, model=label, stat=stat, score='ACC', alpha=r['alpha'],
                                   **{f'M{h}': v for h, v in enumerate(fn(corrresult, axis=1))}))
            score_rows.append(dict(case=case, model=label, stat=stat, score='MSE', alpha=r['alpha'],
                                   **{f'M{h}': v for h, v in enumerate(fn(rmsres, axis=1))}))
        score_rows.append(dict(case=case, model='PERSISTENCE', stat='mean', score='ACC', alpha=np.nan,
                               **{f'M{h}': v for h, v in enumerate(np.mean(persistence, axis=1))}))
        score_rows.append(dict(case=case, model='PERSISTENCE', stat='mean', score='MSE', alpha=np.nan,
                               **{f'M{h}': v for h, v in enumerate(np.mean(rmsper, axis=1))}))
        score_rows.append(dict(case=case, model='CLIMATOLOGY', stat='mean', score='MSE', alpha=np.nan,
                               **{f'M{h}': v for h, v in enumerate(np.mean(rmsclim, axis=1))}))
        score_rows.append(dict(case=case, model='CLIMATOLOGY', stat='median', score='MSE', alpha=np.nan,
                               **{f'M{h}': v for h, v in enumerate(np.median(rmsclim, axis=1))}))

    df = pd.DataFrame(score_rows).drop_duplicates()
    df.to_csv(f'Baselines_{case}_scores.csv', index=False, float_format='%.4f')
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(df[df.stat == 'mean'].round(3).to_string(index=False))
    return df


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', default='all', help='Y620d | Y621d | Y621s | Y622d | all')
    ap.add_argument('--wkdir', default=dspaths.WORK_DIR,
                    help='directory containing the Y*_corr_data pickles (default: DEEPSEASONS_WORKDIR)')
    ap.add_argument('--ridge-alphas', default='1,3,10,30,100,300,1000,3000,10000,30000,100000',
                    help='comma separated ridge penalties for the full-sequence model')
    ap.add_argument('--ar-alphas', default='0.1,1,10,30,100,300,1000,3000,10000,30000,100000',
                    help='comma separated ridge penalties for the AR(p) model')
    args = ap.parse_args()

    ridge_alphas = [float(a) for a in args.ridge_alphas.split(',')]
    ar_alphas = [float(a) for a in args.ar_alphas.split(',')]
    cases = list(field_defs().keys()) if args.case == 'all' else [args.case]
    allres = []
    for c in cases:
        allres.append(run_case(c, args.wkdir, ridge_alphas, ar_alphas))
    if len(allres) > 1:
        pd.concat(allres).to_csv(os.path.join(args.wkdir, 'Baselines_all_scores.csv'),
                                 index=False, float_format='%.4f')
