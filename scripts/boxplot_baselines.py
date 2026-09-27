#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
boxplot_baselines.py
====================

Redraw the skill box plots of the DeepSeasons paper (Figs. 1, 5, 8, 11) with five
boxes per lead time:

    Persistence (green) | GCM (pink) | DeepSeasons (blue) | Ridge (yellow) | AR(p) (violet)

DeepSeasons, Persistence and GCM are read from the original pickles
(Y620d_V100_1_corr_data, ...).  Ridge and AR are read from the pickles produced by
linear_baselines.py ({case}_RIDGE_V..._corr_data, {case}_AR_V..._corr_data).

Usage:
    python boxplot_baselines.py                 # all four cases -> BOXPLOT5_{case}.pdf
    python boxplot_baselines.py --case Y621d
    python boxplot_baselines.py --sqrt          # show RMSE instead of MSE for the T2M cases
"""

import argparse
import os
import sys
import pickle

import numpy as np

# The pickles contain AIModels.AIClasses.Field objects and torch devices:
# make ZapataLibrary importable so they can be unpickled.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'lib'))
import dspaths                        # noqa: E402  (repository path configuration)
dspaths.setup(chdir=False, verbose=False)
import matplotlib
import matplotlib.pyplot as plt

CASES = {
    #  case     DS pickle                 score  title left                 title centre      ylim
    'Y620d': ('Y620d_V100_1_corr_data', 'ACC', 'SST - Monthly - ACC', 'GLOBAL SST', None),
    'Y621d': ('Y621d_V100_1_corr_data', 'RMS', 'T2m - Monthly', 'EUROPE', None),
    'Y622d': ('Y622d_V100_1_corr_data', 'RMS', 'T2m - Monthly', 'NORTH AMERICA', (0, 6)),
    'Y621s': ('Y621s_V299_1_corr_data', 'RMS', 'T2m - Seasonal Mean', 'EUROPE', None),
}

STYLE = {
    # label         face          edge/median   offset
    'Persistence': ('lightgreen', 'green', -0.32),
    'GCM':         ('lightpink', 'red', -0.16),
    'DeepSeasons': ('lightblue', 'blue', 0.00),
    'Ridge':       ('lightyellow', 'darkorange', 0.16),
    'AR':          ('thistle', 'darkviolet', 0.32),
}
WIDTH = 0.14


def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def one_box(ax, data, positions, face, edge):
    return ax.boxplot(data.T, patch_artist=True, widths=WIDTH, positions=positions,
                      boxprops=dict(facecolor=face, color=edge),
                      medianprops=dict(color=edge, linewidth=2),
                      whiskerprops=dict(color=edge), capprops=dict(color=edge),
                      flierprops=dict(markeredgecolor=edge, markersize=3))


def make_figure(case, wkdir, sqrt=False, istart=1, ax=None, fontname='Futura'):
    dsfile, pltype, tleft, tcenter, ylim = CASES[case]
    ds = load(os.path.join(wkdir, dsfile))
    sv, sc = ds['subcase_var'], ds['subcase']
    ridge = load(os.path.join(wkdir, f'{case}_RIDGE_V{sv}_{sc}_corr_data'))
    ar = load(os.path.join(wkdir, f'{case}_AR_V{sv}_{sc}_corr_data'))

    key, keyp, keyg = ('corr', 'pers', 'GCMcorr') if pltype == 'ACC' else ('rms', 'rmsper', 'GCMrms')
    series = {
        'Persistence': ds[keyp],
        'GCM': ds[keyg],
        'DeepSeasons': ds[key],
        'Ridge': ridge[key],
        'AR': ar[key],
    }
    ylabel = pltype
    if pltype == 'RMS':
        # the pickles store the mean squared error; optionally show its square root
        if sqrt:
            series = {k: np.sqrt(v) for k, v in series.items()}
            ylabel = 'RMSE'
        else:
            ylabel = 'MSE'

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))
    else:
        fig = ax.figure

    handles, labels = [], []
    ntim = ds[key].shape[0]
    for label, (face, edge, off) in STYLE.items():
        d = series[label]
        if d is None:
            continue
        pos = np.arange(istart, d.shape[0]) + off
        b = one_box(ax, d[istart:, :], pos, face, edge)
        handles.append(b['boxes'][0])
        labels.append(label if label != 'AR' else f"AR({ar['model_config']['p_ar']})")

    ax.set_xticks(np.arange(istart, ntim))
    ax.set_xticklabels([str(i) for i in range(istart, ntim)], fontname=fontname, fontsize=10)
    ax.set_xlabel('Forecast Month', fontname=fontname)
    ax.set_ylabel(ylabel, fontname=fontname)
    ax.set_title(tleft, loc='left', fontname=fontname)
    ax.set_title(tcenter, loc='center', fontsize=18, fontname=fontname)
    ax.set_title(f"ridge alpha={ridge['model_config']['alpha']:g}, AR alpha={ar['model_config']['alpha']:g}",
                 loc='right', fontsize=9, fontname=fontname)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(handles, labels, loc='best', prop={'family': fontname})
    return fig, ax


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='all')
    ap.add_argument('--wkdir', default=dspaths.WORK_DIR)
    ap.add_argument('--sqrt', action='store_true', help='plot RMSE (sqrt of stored MSE) for T2M cases')
    ap.add_argument('--font', default='Futura')
    args = ap.parse_args()

    # fall back to a default font when Futura is not installed
    from matplotlib.font_manager import findfont, FontProperties
    try:
        if 'Futura' not in findfont(FontProperties(family=args.font)):
            args.font = 'DejaVu Sans'
    except Exception:
        args.font = 'DejaVu Sans'

    cases = list(CASES) if args.case == 'all' else [args.case]
    for c in cases:
        fig, ax = make_figure(c, args.wkdir, sqrt=args.sqrt, fontname=args.font)
        out = os.path.join(args.wkdir, f'BOXPLOT5_{c}{"_rmse" if args.sqrt else ""}.pdf')
        fig.tight_layout()
        fig.savefig(out)
        print('written', out)
        plt.close(fig)
