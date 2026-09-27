#!/usr/bin/env python
"""
plot_ig_combined.py -- two-case Integrated Gradients figures for the paper (REV4)

  FIG_IG_EOF.pdf     sensitivity of the forecast to the input EOFs, per lead:
                     top row Global SST case (SST, U850), bottom row Europe case (T850, T2M)
  FIG_IG_FIELDS.pdf  input field -> output field shares vs lead, left Global SST, right Europe

Reads IG_Y620d_attr.npz and IG_Y621d_attr.npz written by ig_attribution.compute().
Needs only numpy + matplotlib (no torch): the module is imported with a stub if torch is absent.
"""
import os, sys, types, importlib.util
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB = os.path.join(_REPO, 'lib')
here = os.environ.get('DEEPSEASONS_WORKDIR', os.path.join(_REPO, 'work')); os.chdir(here)
try:
    import torch  # noqa
except ImportError:
    t = types.ModuleType('torch'); t.device = lambda *a, **k: None; t.Tensor = type('Tensor', (), {}); sys.modules['torch'] = t
spec = importlib.util.spec_from_file_location('iga', os.path.join(_LIB, 'ig_attribution.py'))
iga = importlib.util.module_from_spec(spec); spec.loader.exec_module(iga)
LAB = iga.FIELD_LABEL

CASES = [('Y620d', 'Global SST'), ('Y621d', 'Europe T2M')]
# field whose forecast is verified in the paper (the network predicts all input fields)
TARGET = {'Y620d': 'SSTSST', 'Y621d': 'T2MT2M'}


def target_shares(agg, case):
    """Attribution of the forecast of the verified field only: (H, J) shares in % summing to 100 per lead."""
    b = agg['fields_out'].index(TARGET[case])
    S = agg['S_eof_out'][:, b, :]
    return 100 * S / S.sum(axis=1, keepdims=True)
COL = {1: '#1f77b4', 3: '#ff7f0e', 6: '#2ca02c', 12: '#d62728'}
OUT = sys.argv[1] if len(sys.argv) > 1 else here


def fig_eof(aggs, fname):
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.5))
    for r, ((case, title), agg) in enumerate(zip(CASES, aggs)):
        leads = agg['leads']; fin = agg['feat_field']; P = target_shares(agg, case)
        for c, f in enumerate(agg['fields_in']):
            ax = axs[r, c]; sel = np.where(fin == f)[0]; nE = len(sel); w = 0.8 / len(leads)
            for ih, h in enumerate(leads):
                ax.bar(np.arange(1, nE + 1) + (ih - (len(leads) - 1) / 2) * w, P[ih, sel],
                       width=w, color=COL.get(int(h)), label=f'lead {h}')
            ax.axhline(100 / len(fin), color='k', lw=0.8, ls='--')      # uniform share
            ax.set_xticks(np.arange(1, nE + 1)); ax.set_xlabel(f'{LAB.get(f, f)} EOF number')
            ax.set_ylabel(f'share of {LAB[TARGET[case]]} forecast attribution (%)')
            ax.set_title(f'{title} forecast: input {LAB.get(f, f)}', loc='left')
            ax.set_title(f'({chr(97 + 2*r + c)})', loc='right')
            ax.grid(axis='y', alpha=0.3)
            if r == 0 and c == 0: ax.legend(fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig); print('written', fname)


def fig_fields(aggs, fname):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for c, ((case, title), agg) in enumerate(zip(CASES, aggs)):
        ax = axs[c]; leads = agg['leads']; fi = agg['fields_in']; fin = agg['feat_field']
        P = target_shares(agg, case); tgt = LAB[TARGET[case]]
        for a, fa in enumerate(fi):
            ax.plot(leads, P[:, fin == fa].sum(axis=1), marker='os'[a % 2], lw=1.8, label=f'from {LAB.get(fa, fa)}')
        ax.set_xticks(list(leads)); ax.set_xlabel('lead month'); ax.set_ylabel(f'share of {tgt} forecast attribution (%)')
        ax.set_ylim(0, 100); ax.set_title(f'{title} forecast', loc='left'); ax.set_title(f'({chr(97 + c)})', loc='right'); ax.grid(alpha=0.3)
        ax.legend(fontsize=9, title='input field', title_fontsize=9)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight'); plt.close(fig); print('written', fname)


if __name__ == '__main__':
    aggs = [iga.aggregate(iga.load(f'IG_{c}_attr.npz')) for c, _ in CASES]
    fig_eof(aggs, os.path.join(OUT, 'FIG_IG_EOF.pdf'))
    fig_fields(aggs, os.path.join(OUT, 'FIG_IG_FIELDS.pdf'))
    # numbers for the text
    for (c, title), agg in zip(CASES, aggs):
        print(f'\n{title}: share of total attribution per input->output block (%), leads {list(agg["leads"])}')
        for a, fa in enumerate(agg['fields_in']):
            for b, fb in enumerate(agg['fields_out']):
                print(f'  {LAB.get(fa,fa):>5s} -> {LAB.get(fb,fb):<5s}: ' + ' '.join(f'{v:5.1f}' for v in agg['S_field_pct'][:, a, b]))
        for a, fa in enumerate(agg['fields_in']):
            print(f'  total from {LAB.get(fa,fa):<5s}: ' + ' '.join(f'{v:5.1f}' for v in agg['S_field_pct'][:, a, :].sum(axis=1)))
        for b, fb in enumerate(agg['fields_out']):
            print(f'  total to   {LAB.get(fb,fb):<5s}: ' + ' '.join(f'{v:5.1f}' for v in agg['S_field_pct'][:, :, b].sum(axis=1)))
        fin = agg['feat_field']
        for f in agg['fields_in']:
            sel = np.where(fin == f)[0]; S = agg['S_eof_pct'][:, sel]
            print(f'  {LAB.get(f,f)}: max/min share over EOFs per lead: ' + ' '.join(f'{S[i].max():.1f}/{S[i].min():.1f}' for i in range(S.shape[0])) +
                  f'  (uniform = {100/len(fin):.1f});  top EOF per lead: {[int(agg["eof_index"][sel][np.argmax(S[i])]) for i in range(S.shape[0])]}')
