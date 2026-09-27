#!/usr/bin/env python
"""
make_baselines_table.py
Build REV4/Baselines_table.tex (Table `tab:baselines`) from
  Baselines_all_scores.csv        (RIDGE, AR, PERSISTENCE, CLIMATOLOGY; linear_baselines.py)
  ACC_Score_Y620d_100_1_mean.csv, RMS_Score_Y62*_mean.csv   (DeepSeasons, GCM)
Usage:  python make_baselines_table.py [--out PATH]
"""
import argparse, os, sys
import numpy as np, pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'lib'))
import dspaths                        # noqa: E402  (repository path configuration)
dspaths.setup(chdir=False, verbose=False)
default_out = os.path.join(dspaths.WORK_DIR, 'Baselines_table.tex')
ap = argparse.ArgumentParser(); ap.add_argument('--out', default=default_out); ap.add_argument('--wkdir', default=dspaths.WORK_DIR)
args = ap.parse_args(); os.chdir(args.wkdir)

leads = [1, 2, 3, 4, 6, 9, 12]
cases = [('Y620d', 'ACC_Score_Y620d_100_1_mean.csv', 'ACC', 'Global SST (ACC)', 4),
         ('Y621d', 'RMS_Score_Y621d_100_1_mean.csv', 'MSE', 'Europe T2M, monthly (MSE, K$^2$)', 4),
         ('Y622d', 'RMS_Score_Y622d_100_1_mean.csv', 'MSE', 'North America T2M, monthly (MSE, K$^2$)', 2),
         ('Y621s', 'RMS_Score_Y621s_299_1_mean.csv', 'MSE', 'Europe T2M, seasonal mean (MSE, K$^2$)', 4)]
b = pd.read_csv('Baselines_all_scores.csv'); b = b[b.stat == 'mean']

def fmt(v):
    try:
        v = float(v)
    except Exception:
        return '--'
    return '--' if np.isnan(v) else f'{v:.2f}'

def row(name, src):
    return f'{name} & ' + ' & '.join(fmt(src.get(f'M{l}', np.nan)) for l in leads) + r' \\'

out = [r'\begin{table*}[t]', r'\centering',
       r'\caption{Mean skill over the initial dates of the test period as a function of lead time (months) for \DS, the operational GCM (available to month six), persistence, the climatological forecast (zero anomaly with respect to the 1940--2015 climatology; MSE only) and the two linear benchmarks of Section~2\ref{sec:autocorr}: the autoregression on the last $p$ months (AR($p$), $p=4$ for SST and Europe, $p=2$ for North America) and the ridge regression on the full input window. Global SST is scored with the spatial anomaly correlation, the regional T2M cases with the area mean squared error. All systems are verified on the same truncated EOF basis. The ridge penalty selected on the validation period is indicated.}',
       r'\label{tab:baselines}', r'\begin{tabular}{l' + 'r' * len(leads) + '}', r'\toprule',
       'System & ' + ' & '.join(f'M{l}' for l in leads) + r' \\']
for case, f, score, title, p in cases:
    y = pd.read_csv(f)
    pick = lambda pre: y[y.Exp.str.startswith(pre)].iloc[0]
    bb = b[(b.case == case) & (b.score == score)]
    get = lambda m: bb[bb.model == m].iloc[0] if (bb.model == m).any() else {}
    ar, rg, cl = get('AR'), get('RIDGE'), get('CLIMATOLOGY')
    out += [r'\midrule', r'\multicolumn{%d}{l}{\textbf{%s}} \\' % (len(leads) + 1, title),
            row(r'\DS', pick('AC-')), row('GCM', pick('DY-')), row('Persistence', pick('PE-'))]
    if score == 'MSE' and len(cl):
        out.append(row('Climatology', cl))
    out.append(row(f"AR({p}) [$\\alpha$={ar['alpha']:g}]", ar))
    out.append(row(f"Ridge [$\\alpha$={rg['alpha']:g}]", rg))
out += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
open(args.out, 'w').write('\n'.join(out) + '\n')
print('\n'.join(out)); print('\nwritten', args.out)
