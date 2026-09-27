# -*- coding: utf-8 -*-
"""
ig_attribution.py
=================
Integrated Gradients (IG) attribution of DeepSeasons (Informer) forecasts,
resolved by lead time, for the REV4 revision.

Usage from a notebook (after the model has been trained/loaded and
`test_dataloader_fut` built with zcf.TimeSeriesFuture):

    import ig_attribution as iga
    res = iga.compute(model, test_dataloader_fut, INX, leads=[1,3,6,12], n_steps=32,
                      baseline='persistence', device='cpu', save=f'IG_{case}_attr.npz')
    agg = iga.aggregate(res)
    iga.print_summary(agg)
    iga.plot_leads(agg, title=case, savefile=f'FIG_IG_{case}_leads.pdf')
    iga.plot_fields(agg, savefile=f'FIG_IG_{case}_fields.pdf')

    # later, without recomputing:
    res = iga.load(f'IG_{case}_attr.npz'); agg = iga.aggregate(res)

Design
------
* The attributed quantity is the deterministic (mean) forecast obtained with the
  same autoregressive decoding as model.generate(), feeding back distr.mean instead
  of distr.sample(); it is differentiable and corresponds to the ensemble mean that
  is verified in the paper.  (The previous implementation attributed
  encoder_last_hidden_state, because model.forward() with future_values=None does
  not return a forecast.)
* Attributions are computed for every forecast component (lead h, output feature k)
  with respect to every input (month t, input feature j), for all test cases at once.
* Baseline: 'persistence' (last input month repeated over the window) or 'zeros'.
* |attribution| is averaged (never the signed value, which cancels).
"""
import time
import numpy as np
import torch

FIELD_LABEL = {'SSTSST': 'SST', 'U850': 'U850', 'T850': 'T850', 'T2MT2M': 'T2M',
               'SP': 'SLP', 'Z500': 'Z500', 'MTNLWRF': 'OLR'}


# ----------------------------------------------------------------------------
def make_mean_forecast(model):
    """Return f(past_values, past_time_features, future_time_features) -> (B, Tpredict, K),
    the deterministic forecast of `model` (mean of the predicted distribution at each step)."""
    TIN = model.config.context_length + max(model.config.lags_sequence)
    MIN = model.config.input_size
    decoder = model.model.get_decoder()

    def mean_forecast(past_values, past_time_features, future_time_features):
        B = past_values.shape[0]
        pasobs = torch.ones([B, TIN, MIN], dtype=past_values.dtype, device=past_values.device)
        out = model(past_values=past_values, past_time_features=past_time_features,
                    past_observed_mask=pasobs, future_time_features=future_time_features,
                    future_values=None, return_dict=True, use_cache=False)
        enc_last, loc, scale, static_feat = out.encoder_last_hidden_state, out.loc, out.scale, out.static_features
        past_scaled = (past_values - loc) / scale
        feats = torch.cat((static_feat.unsqueeze(1).expand(-1, future_time_features.shape[1], -1),
                           future_time_features), dim=-1)
        preds = []
        for k in range(model.config.prediction_length):
            lagged = model.model.get_lagged_subsequences(sequence=past_scaled, subsequences_length=1 + k, shift=1)
            lagged = lagged.reshape(lagged.shape[0], lagged.shape[1], -1)
            dec_in = torch.cat((lagged, feats[:, :k + 1]), dim=-1)
            dec_out = decoder(inputs_embeds=dec_in, encoder_hidden_states=enc_last)
            prm = model.parameter_projection(dec_out.last_hidden_state[:, -1:])
            nxt = model.output_distribution(prm, loc=loc, scale=scale).mean
            past_scaled = torch.cat((past_scaled, (nxt - loc) / scale), dim=1)
            preds.append(nxt)
        return torch.cat(preds, dim=1)

    return mean_forecast


def _feature_layout(INX):
    feat_field, eof_index, var_frac = [], [], []
    for key in INX.keys():
        mr = INX[key]['mr']
        s2 = INX[key]['sdat'] ** 2
        for j in range(mr):
            feat_field.append(key); eof_index.append(j + 1); var_frac.append(float(s2[j] / s2.sum()))
    return np.array(feat_field), np.array(eof_index), np.array(var_frac)


# ----------------------------------------------------------------------------
def compute(model, dataloader, INX, leads=(1, 3, 6, 12), n_steps=32, baseline='persistence',
            device='cpu', save=None, verbose=True, PDX=None):
    """
    IG attribution of the mean forecast for all cases in `dataloader`
    (a TimeSeriesFuture loader yielding src, tgt, pasft, futft).

    Returns dict with
      attr      (H, K, N, T, J) signed attributions  [lead, output feature, case, input month, input feature]
      delta     (H, K, N)       completeness error of the path integral
      forecast  (N, Tpredict, K) deterministic forecast (scaled space)
      leads, feat_field, eof_index, var_frac, out_field, n_steps, baseline
    """
    from captum.attr import IntegratedGradients
    dev = torch.device(device)
    orig_device = next(model.parameters()).device
    orig_grad = [p.requires_grad for p in model.parameters()]
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fwd = make_mean_forecast(model)

    SRC, PAS, FUT = [], [], []
    for src, tgt, pasft, futft in dataloader:
        SRC.append(src); PAS.append(pasft); FUT.append(futft)
    SRC = torch.cat(SRC).to(dev).float(); PAS = torch.cat(PAS).to(dev).float(); FUT = torch.cat(FUT).to(dev).float()
    N, T, J = SRC.shape
    with torch.no_grad():
        FCST = fwd(SRC, PAS, FUT)
    K = FCST.shape[-1]
    if verbose:
        print(f'IG: {N} cases, input {T} months x {J} features, output {FCST.shape[1]} leads x {K} features, '
              f'baseline={baseline}, n_steps={n_steps}, device={device}')

    if baseline == 'persistence':
        BASE = SRC[:, -1:, :].repeat(1, T, 1)
    elif baseline == 'zeros':
        BASE = torch.zeros_like(SRC)
    else:
        raise ValueError(baseline)

    leads = list(leads)
    ATTR = np.zeros((len(leads), K, N, T, J), dtype=np.float32)
    DELTA = np.zeros((len(leads), K, N), dtype=np.float32)
    t0 = time.time()
    for ih, h in enumerate(leads):
        for k in range(K):
            def f(x, pas, fut, h=h, k=k):
                return fwd(x, pas, fut)[:, h - 1, k]
            ig = IntegratedGradients(f, multiply_by_inputs=True)
            a, d = ig.attribute(SRC, baselines=BASE, additional_forward_args=(PAS, FUT), n_steps=n_steps,
                                method='riemann_trapezoid', internal_batch_size=N, return_convergence_delta=True)
            ATTR[ih, k] = a.detach().cpu().numpy(); DELTA[ih, k] = d.detach().cpu().numpy()
        if verbose:
            scale_ref = float(np.abs(FCST[:, h - 1, :].cpu().numpy()).mean())
            print(f'  lead {h:2d} done ({time.time()-t0:5.0f} s); mean |delta| = {np.abs(DELTA[ih]).mean():.2e} '
                  f'(mean |forecast| = {scale_ref:.2e})')

    # restore the model as it was (device and requires_grad) so the notebook state is untouched
    for p, g in zip(model.parameters(), orig_grad):
        p.requires_grad_(g)
    model.to(orig_device)

    feat_field, eof_index, var_frac = _feature_layout(INX)
    out_field, _, _ = _feature_layout(PDX if PDX is not None else INX)
    res = dict(attr=ATTR, delta=DELTA, forecast=FCST.cpu().numpy(), leads=np.array(leads),
               feat_field=feat_field, eof_index=eof_index, var_frac=var_frac, out_field=out_field,
               n_steps=n_steps, baseline=baseline)
    if save:
        np.savez_compressed(save, **res)
        if verbose:
            print('saved', save)
    return res


def load(fname):
    z = np.load(fname, allow_pickle=True)
    res = {k: z[k] for k in z.files}
    res['n_steps'] = int(res['n_steps']); res['baseline'] = str(res['baseline'])
    return res


# ----------------------------------------------------------------------------
def connectivity_check(model, dataloader, device='cpu', eps=0.5):
    """
    Diagnostic: is every input month connected to the forecast?
    Prints, for each input month t, the mean |d forecast / d x_t| (all leads, all outputs)
    and the change of the forecast when the month is perturbed by `eps` (scaled units).
    A month with exactly zero in both columns is not used by the network.
    """
    dev = torch.device(device)
    orig_device = next(model.parameters()).device
    model = model.to(dev).eval()
    fwd = make_mean_forecast(model)
    src, tgt, pasft, futft = next(iter(dataloader))
    src, pasft, futft = src.to(dev).float(), pasft.to(dev).float(), futft.to(dev).float()
    x = src.clone().requires_grad_(True)
    y = fwd(x, pasft, futft)
    g, = torch.autograd.grad(y.sum(), x)
    g = g.abs().mean(dim=(0, 2)).detach().cpu().numpy()           # per input month
    with torch.no_grad():
        y0 = fwd(src, pasft, futft)
        dy = []
        for t in range(src.shape[1]):
            xp = src.clone(); xp[:, t, :] += eps
            dy.append(float((fwd(xp, pasft, futft) - y0).abs().mean()))
    T = src.shape[1]
    print(f"context_length={model.config.context_length}, lags={list(model.config.lags_sequence)}, "
          f"distil={getattr(model.config, 'distil', None)}, distribution={model.config.distribution_output}, "
          f"scaling={model.config.scaling}")
    print(' month   mean|grad|   |dF| for +eps')
    for t in range(T):
        flag = '   <-- disconnected' if g[t] == 0 and dy[t] == 0 else ''
        print(f'  {t-T:4d}   {g[t]:10.3e}   {dy[t]:10.3e}{flag}')
    model.to(orig_device)
    return g, np.array(dy)


def encoder_input_check(model, dataloader, device='cpu'):
    """
    Diagnostic: gradient of the forecast with respect to the encoder input tensor
    (context steps x [lag blocks | time features | static features]).
    Prints mean |grad| per encoder step and per lag block, to see whether the first
    steps or some lag blocks are ignored by the network.
    """
    dev = torch.device(device)
    orig_device = next(model.parameters()).device
    model = model.to(dev).eval()
    fwd = make_mean_forecast(model)
    src, tgt, pasft, futft = next(iter(dataloader))
    src, pasft, futft = src.to(dev).float(), pasft.to(dev).float(), futft.to(dev).float()
    src = src.clone().requires_grad_(True)       # so that the encoder input is part of the graph
    captured = {}

    def hook(module, args, kwargs):
        x = kwargs.get('inputs_embeds', args[0] if args else None)
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        else:
            x.retain_grad()
        captured['x'] = x
        if 'inputs_embeds' in kwargs:
            kwargs = dict(kwargs); kwargs['inputs_embeds'] = x
            return args, kwargs
        return (x,) + tuple(args[1:]), kwargs
    h = model.model.encoder.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        y = fwd(src, pasft, futft)
        y.sum().backward()
    finally:
        h.remove()
    g = captured['x'].grad.abs()                 # (B, context, feature_size)
    nlag = len(model.config.lags_sequence); nin = model.config.input_size
    print(f'encoder input: {tuple(captured["x"].shape)}  = (batch, context steps, {nlag} lag blocks x {nin} + '
          f'{model.config.num_time_features} time feats + static)')
    per_step = g.mean(dim=(0, 2)).cpu().numpy()
    print(' step  mean|grad|'); [print(f'  {s:3d}  {v:10.3e}') for s, v in enumerate(per_step)]
    for i, lag in enumerate(model.config.lags_sequence):
        print(f' lag {lag} block: mean|grad| = {g[:, :, i*nin:(i+1)*nin].mean().item():.3e}')
    print(f' time features   : {g[:, :, nlag*nin:nlag*nin+model.config.num_time_features].mean().item():.3e}')
    print(f' static features : {g[:, :, nlag*nin+model.config.num_time_features:].mean().item():.3e}')
    model.to(orig_device)
    return per_step, g.cpu().numpy()


def encoder_ablation(model, dataloader, device='cpu'):
    """
    Diagnostic: how much does the forecast depend on the encoder?
    Recomputes the deterministic forecast with (a) the encoder output set to zero,
    (b) the encoder outputs shuffled between cases, (c) the scaler statistics (loc, scale)
    shuffled between cases, and reports the RMS change relative to the RMS forecast
    anomaly and to the RMS (forecast - persistence) signal.  All test cases are used.
    """
    dev = torch.device(device)
    orig_device = next(model.parameters()).device
    model = model.to(dev).eval()
    decoder = model.model.get_decoder()
    TIN = model.config.context_length + max(model.config.lags_sequence); MIN = model.config.input_size
    SRC, PAS, FUT = [], [], []
    for src, tgt, pasft, futft in dataloader:
        SRC.append(src); PAS.append(pasft); FUT.append(futft)
    SRC = torch.cat(SRC).to(dev).float(); PAS = torch.cat(PAS).to(dev).float(); FUT = torch.cat(FUT).to(dev).float()
    B = SRC.shape[0]

    def decode(enc_last, loc, scale, static_feat):
        past_scaled = (SRC - loc) / scale
        feats = torch.cat((static_feat.unsqueeze(1).expand(-1, FUT.shape[1], -1), FUT), dim=-1)
        preds = []
        for k in range(model.config.prediction_length):
            lagged = model.model.get_lagged_subsequences(sequence=past_scaled, subsequences_length=1 + k, shift=1)
            lagged = lagged.reshape(lagged.shape[0], lagged.shape[1], -1)
            dec_out = decoder(inputs_embeds=torch.cat((lagged, feats[:, :k + 1]), dim=-1), encoder_hidden_states=enc_last)
            nxt = model.output_distribution(model.parameter_projection(dec_out.last_hidden_state[:, -1:]), loc=loc, scale=scale).mean
            past_scaled = torch.cat((past_scaled, (nxt - loc) / scale), dim=1); preds.append(nxt)
        return torch.cat(preds, dim=1)

    with torch.no_grad():
        pasobs = torch.ones([B, TIN, MIN], device=dev)
        out = model(past_values=SRC, past_time_features=PAS, past_observed_mask=pasobs,
                    future_time_features=FUT, future_values=None, return_dict=True, use_cache=False)
        enc, loc, scale, sf = out.encoder_last_hidden_state, out.loc, out.scale, out.static_features
        F0 = decode(enc, loc, scale, sf)
        perm = torch.randperm(B, device=dev)
        Fz = decode(torch.zeros_like(enc), loc, scale, sf)
        Fs = decode(enc[perm], loc, scale, sf)
        Fl = decode(enc, loc[perm], scale[perm], sf[perm])
        pers = SRC[:, -1:, :].expand_as(F0)
    rms = lambda x: float(torch.sqrt((x ** 2).mean()))
    a0, sig = rms(F0), rms(F0 - pers)
    print(f'RMS forecast anomaly {a0:.4f};  RMS (forecast - persistence) {sig:.4f}   [scaled units]')
    for name, F in (('encoder output = 0', Fz), ('encoder output shuffled', Fs), ('scaler loc/scale shuffled', Fl)):
        d = rms(F - F0)
        print(f'  {name:28s}: RMS change {d:.4f}  = {100*d/a0:5.1f}% of forecast, {100*d/sig:6.1f}% of forecast-persistence signal')
    # per-lead view of the encoder ablation
    dl = torch.sqrt(((Fz - F0) ** 2).mean(dim=(0, 2))).cpu().numpy(); sl = torch.sqrt(((F0 - pers) ** 2).mean(dim=(0, 2))).cpu().numpy()
    print('  lead: ' + ' '.join(f'{h+1:5d}' for h in range(len(dl))))
    print('  enc=0 change / signal (%): ' + ' '.join(f'{100*d/s:5.1f}' for d, s in zip(dl, sl)))
    model.to(orig_device)
    return F0.cpu().numpy(), Fz.cpu().numpy(), Fs.cpu().numpy(), Fl.cpu().numpy()


def aggregate(res):
    """
    Sensitivities from |attr| (H, K, N, T, J):
      S_eof   (H, J)        whole forecast at lead h vs input feature j   (mean over k, n, t)
      S_time  (H, T)        whole forecast at lead h vs input month t     (mean over k, n; sum over j)
      S_field (H, Fin, Fout) input field -> output field blocks
      S_eof_out (H, Fout, J) forecast of each output field vs input feature j
    All are also returned normalised to 100 % per lead (suffix _pct).
    """
    A = np.abs(res['attr'])
    fin, fout = res['feat_field'], res['out_field']
    fields_in = list(dict.fromkeys(fin)); fields_out = list(dict.fromkeys(fout))
    H, K, N, T, J = A.shape
    S_eof = A.mean(axis=(1, 2, 3))
    S_time = A.mean(axis=(1, 2)).sum(axis=-1)
    S_field = np.zeros((H, len(fields_in), len(fields_out)))
    S_eof_out = np.zeros((H, len(fields_out), J))
    for b, fb in enumerate(fields_out):
        S_eof_out[:, b, :] = A[:, fout == fb].mean(axis=(1, 2, 3))
        for a, fa in enumerate(fields_in):
            S_field[:, a, b] = A[:, fout == fb][:, :, :, :, fin == fa].mean(axis=(1, 2, 3, 4))
    pct = lambda S: 100 * S / S.reshape(H, -1).sum(axis=1)[(slice(None),) + (None,) * (S.ndim - 1)]
    return dict(S_eof=S_eof, S_time=S_time, S_field=S_field, S_eof_out=S_eof_out,
                S_eof_pct=pct(S_eof), S_time_pct=pct(S_time), S_field_pct=pct(S_field),
                leads=res['leads'], feat_field=fin, eof_index=res['eof_index'], out_field=fout,
                fields_in=fields_in, fields_out=fields_out, T=T, baseline=res['baseline'],
                N=N, delta=res['delta'])


def print_summary(agg, ntop=5):
    names = [f"{FIELD_LABEL.get(f, f)}-{e}" for f, e in zip(agg['feat_field'], agg['eof_index'])]
    for ih, h in enumerate(agg['leads']):
        top = np.argsort(agg['S_eof'][ih])[::-1][:ntop]
        print(f"lead {h:2d}: " + ', '.join(f"{names[j]} {agg['S_eof_pct'][ih, j]:4.1f}%" for j in top))
        M = agg['S_field_pct'][ih]
        for a, fa in enumerate(agg['fields_in']):
            print('          ' + ' '.join(f"{FIELD_LABEL.get(fa, fa):>5s}->{FIELD_LABEL.get(fb, fb):<5s} {M[a, b]:5.1f}%"
                                          for b, fb in enumerate(agg['fields_out'])))
    # sensitivity-weighted mean look-back (months) per lead
    t = np.arange(-agg['T'], 0)
    for ih, h in enumerate(agg['leads']):
        w = agg['S_time'][ih] / agg['S_time'][ih].sum()
        print(f"lead {h:2d}: mean look-back {-(w * t).sum():5.1f} months; share of last 4 months {100*w[-4:].sum():4.1f}%")


# ----------------------------------------------------------------------------
def plot_leads(agg, title='', savefile=None, fontname=None):
    import matplotlib.pyplot as plt
    fk = {'fontname': fontname} if fontname else {}
    leads = agg['leads']; fields = agg['fields_in']; fin = agg['feat_field']
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, len(fields), height_ratios=[1.3, 1], hspace=0.35, wspace=0.25)
    for c, f in enumerate(fields):
        ax = fig.add_subplot(gs[0, c]); sel = np.where(fin == f)[0]; nE = len(sel); w = 0.8 / len(leads)
        for ih, h in enumerate(leads):
            ax.bar(np.arange(1, nE + 1) + (ih - (len(leads) - 1) / 2) * w, agg['S_eof_pct'][ih, sel], width=w, label=f'lead {h}')
        ax.set_xticks(np.arange(1, nE + 1)); ax.set_xlabel(f'{FIELD_LABEL.get(f, f)} EOF number', **fk)
        ax.set_ylabel('share of total attribution (%)', **fk); ax.set_title(FIELD_LABEL.get(f, f), loc='right', **fk)
        if c == 0: ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, :]); t = np.arange(-agg['T'], 0)
    for ih, h in enumerate(leads):
        ax.plot(t, agg['S_time_pct'][ih], marker='o', ms=3, label=f'lead {h}')
    ax.set_xlabel('input month relative to the initial date', **fk); ax.set_ylabel('share of total attribution (%)', **fk)
    ax.set_title('Sensitivity to the input months', loc='left', **fk); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle(f"Integrated Gradients {title}: sensitivity of the {'+'.join(FIELD_LABEL.get(f, f) for f in agg['fields_out'])} "
                 f"forecast (baseline {agg['baseline']}, {agg['N']} cases)", **fk)
    if savefile: plt.savefig(savefile, bbox_inches='tight')
    plt.show(); return fig


def plot_fields(agg, savefile=None, fontname=None):
    import matplotlib.pyplot as plt
    fk = {'fontname': fontname} if fontname else {}
    leads = agg['leads']; fi, fo = agg['fields_in'], agg['fields_out']
    fig, axs = plt.subplots(1, len(fo), figsize=(4 * len(fo), 3.5), sharey=True); axs = np.atleast_1d(axs)
    for b, fb in enumerate(fo):
        for a, fa in enumerate(fi):
            axs[b].plot(leads, agg['S_field_pct'][:, a, b], marker='o', label=f'from {FIELD_LABEL.get(fa, fa)}')
        axs[b].set_title(f'forecast of {FIELD_LABEL.get(fb, fb)}', loc='right', **fk)
        axs[b].set_xlabel('lead month', **fk); axs[b].set_xticks(list(leads)); axs[b].grid(alpha=0.3)
    axs[0].set_ylabel('share of total attribution (%)', **fk); axs[0].legend(fontsize=8)
    if savefile: plt.savefig(savefile, bbox_inches='tight')
    plt.show(); return fig
