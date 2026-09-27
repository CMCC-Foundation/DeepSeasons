"""
Validation module for ensemble forecasts and lead time construction.

This module combines:
  1. Lead time construction from observations (construct_obs_leads)
  2. Tercile-based verification metrics: RPSS and Brier Skill Score (BSS)

For ensemble forecast verification, this module computes:
  - Rank Probability Score (RPS) and Rank Probability Skill Score (RPSS) for tercile categories.
  - Brier Score (BS) and Brier Skill Score (BSS) for each tercile category (lower/middle/upper).

For observation lead construction:
  - Takes a timeseries of observations and creates lead time dimension
  - Maintains spatial coordinates and adds lead date tracking

Supported input shapes (flattened spatial dimension G):
  * Deterministic forecasts:
      f: (N, G) or (N, L, G)
  * Ensemble forecasts:
      f: (M, N, G) or (M, N, L, G)
  * Observations / reference (persistence):
      o: (N, G) or (N, L, G)
      p: (N, G) or (N, L, G)

Where:
  - N = number of verifying cases / initial conditions
  - L = number of lead times
  - M = ensemble members
  - G = number of grid points (flattened)

Device Support:
  All heavy computations are done in PyTorch on the requested device (cpu/cuda/mps).
  Automatic float64 -> float32 conversion for MPS (which doesn't support float64).
"""

from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from typing import Optional, Union, Dict, Any, Tuple


# ============================================================================
# LEAD TIME CONSTRUCTION
# ============================================================================

def construct_obs_leads(obs: xr.DataArray, n_leads: int, ic0: np.datetime64 = None, 
                        n_init_conditions: int = None) -> xr.DataArray:
    """
    Construct lead times from observation timeseries.
    
    Takes a 1D or 2D observation array (N, G) or (time, space) and creates 
    a 3D array (N, L, G) where each initial condition has L lead times.
    
    Parameters
    ----------
    obs : xr.DataArray
        Observations with shape (N, G) or (time, space). 
        Must have a time dimension as the first axis.
    n_leads : int
        Number of lead times to extract from each initial condition.
    ic0 : np.datetime64, optional
        Starting date for initial conditions. If None, starts from the first time step.
    n_init_conditions : int, optional
        Number of initial conditions to extract. If None, uses all available from ic0 onward.
        
    Returns
    -------
    obs_leads : xr.DataArray
        Shape (N, L, G) where:
          - N: number of initial conditions
          - L: number of lead times
          - G: spatial dimension
        Includes coordinates:
          - 'time': initial condition times
          - 'lead': lead time indices (0, 1, ..., L-1)
          - 'lead_date': calendar dates for each lead
          
    Example
    -------
    >>> obs = xr.DataArray(
    ...     np.random.randn(100, 1000),  # 100 time steps, 1000 grid points
    ...     coords={
    ...         'time': pd.date_range('2000-01-01', periods=100, freq='D'),
    ...         'space': np.arange(1000)
    ...     },
    ...     dims=['time', 'space']
    ... )
    >>> # Extract 10 leads starting from 2000-02-01, with 20 initial conditions
    >>> obs_leads = construct_obs_leads(obs, n_leads=10, 
    ...                                  ic0=np.datetime64('2000-02-01'),
    ...                                  n_init_conditions=20)
    >>> obs_leads.shape
    (20, 10, 1000)
    """
    
    # Extract time and spatial coordinates
    time_coord = obs.coords[obs.dims[0]]
    time_values = time_coord.values
    n_time = len(time_coord)
    
    # Get spatial dimensions (all dims except first)
    spatial_dims = obs.dims[1:]
    
    # Extract spatial coordinates from input
    spatial_coords = {}
    for dim in spatial_dims:
        if dim in obs.coords:
            spatial_coords[dim] = obs.coords[dim]
    
    # Find starting index for initial conditions
    if ic0 is None:
        start_idx = 0
    else:
        # Find the index of ic0 in the time coordinate
        ic0 = np.datetime64(ic0)
        idx_match = np.where(time_values == ic0)[0]
        if len(idx_match) == 0:
            raise ValueError(
                f"Initial condition date {ic0} not found in observations. "
                f"Available dates range from {time_values[0]} to {time_values[-1]}"
            )
        start_idx = idx_match[0]
    
    # Determine number of initial conditions
    if n_init_conditions is None:
        # Use as many as possible from start_idx onward
        n_init = n_time - start_idx - n_leads + 1
    else:
        n_init = n_init_conditions
    
    # Validate we have enough data
    required_time_steps = start_idx + n_init + n_leads - 1
    if required_time_steps > n_time:
        raise ValueError(
            f"Not enough time steps. Starting at index {start_idx} with {n_init} initial conditions "
            f"and {n_leads} leads requires {required_time_steps} time steps, but only {n_time} available."
        )
    
    # Create the output array
    obs_data = obs.values
    spatial_shape = obs_data.shape[1:]
    
    obs_leads_data = np.zeros((n_init, n_leads) + spatial_shape, dtype=obs_data.dtype)
    
    # Fill the leads array: each row is one initial condition with its leads
    for i in range(n_init):
        obs_leads_data[i] = obs_data[start_idx + i:start_idx + i + n_leads]
    
    # Create time coordinates for initial conditions
    init_times = time_values[start_idx:start_idx + n_init]
    
    # Create lead indices
    lead_indices = np.arange(n_leads)
    
    # Create lead dates for each initial time and lead
    lead_dates = np.zeros((n_init, n_leads), dtype='datetime64[D]')
    
    # Determine the frequency (assume daily if using datetime64)
    try:
        # Try to infer frequency from first two time steps
        if n_time >= 2:
            time_diff = time_values[1] - time_values[0]
            # time_diff is in nanoseconds for datetime64[ns]
            if hasattr(time_diff, 'astype'):
                days_per_step = time_diff.astype('timedelta64[D]').astype(float)
            else:
                days_per_step = time_diff / np.timedelta64(1, 'D')
        else:
            days_per_step = 1  # default to 1 day
    except:
        days_per_step = 1
    
    for i in range(n_init):
        for j in range(n_leads):
            offset_days = int(j * days_per_step)
            lead_dates[i, j] = init_times[i] + np.timedelta64(offset_days, 'D')
    
    # Create the output DataArray
    coords = {
        'time': ('time', init_times),
        'lead': ('lead', lead_indices),
        'lead_date': (('time', 'lead'), lead_dates),
    }
    # Add spatial coordinates
    coords.update(spatial_coords)
    
    obs_leads = xr.DataArray(
        obs_leads_data,
        coords=coords,
        dims=['time', 'lead'] + list(spatial_dims),
        name=obs.name if obs.name else 'obs_leads',
        attrs=obs.attrs
    )
    
    return obs_leads


# ============================================================================
# RPSS AND BSS COMPUTATION
# ============================================================================

def _to_torch(x: Any, device: Union[str, torch.device]) -> torch.Tensor:
    """Convert various input types to torch tensor on specified device."""
    # Check if device is MPS
    is_mps = (isinstance(device, str) and "mps" in device.lower()) or \
             (isinstance(device, torch.device) and device.type == "mps")
    
    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, np.ndarray):
        # Convert numpy to float32 if targeting MPS (which doesn't support float64)
        if is_mps and x.dtype == np.float64:
            x = x.astype(np.float32)
        tensor = torch.from_numpy(x)
    elif hasattr(x, "values"):  # xarray.DataArray or xarray.Variable
        arr = np.asarray(x.values)
        # Convert numpy to float32 if targeting MPS
        if is_mps and arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        tensor = torch.from_numpy(arr)
    else:
        raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor, np.ndarray, or xarray.DataArray.")
    
    # Convert float64 to float32 if on MPS device
    if is_mps and tensor.dtype == torch.float64:
        tensor = tensor.to(torch.float32)
    
    return tensor.to(device)



def _compute_loo_terciles(o_ng: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute leave-one-out tercile thresholds for each case.

    Parameters
    ----------
    o_ng : torch.Tensor, shape (N, G)
        Verifying observations.

    Returns
    -------
    q1, q2 : torch.Tensor, shape (N, G)
        Case-dependent tercile thresholds excluding the verifying case.
    """
    N, G = o_ng.shape
    if N < 3:
        raise ValueError("leave_one_out=True requires N>=3 to define terciles.")

    # Sort along N once (nan-aware by pushing NaNs to the end)
    o = o_ng
    nan_mask = torch.isnan(o)
    o_fill = o.clone()
    o_fill[nan_mask] = torch.tensor(float("inf"), device=o.device, dtype=o.dtype)
    o_sorted, sort_idx = torch.sort(o_fill, dim=0)
    # Count finite
    finite_sorted = torch.isfinite(o_sorted)
    n_finite = finite_sorted.long().sum(dim=0)  # (G,)

    # Inverse ranks: inv_rank[i,g] = position of case i in sorted order for grid g
    inv_rank = torch.empty_like(sort_idx)
    inv_rank.scatter_(0, sort_idx, torch.arange(N, device=o.device)[:, None].expand(N, G))

    def loo_quantile(q: float) -> torch.Tensor:
        # For each grid point g, LOO sample size = n_finite[g]-1 if the excluded element is finite, else n_finite[g]
        # Compute LOO quantile position in [0, n_loo-1], then pick from sorted excluding element.
        # Use linear interpolation between adjacent order stats (like nanquantile).
        q = float(q)
        # whether excluded is finite for each (i,g)
        excluded_finite = torch.isfinite(o_fill)  # (N,G), inf marks NaN
        n0 = n_finite[None, :].expand(N, G)  # (N,G)
        n_loo = n0 - excluded_finite.long()
        n_loo = torch.clamp(n_loo, min=1)

        pos = (n_loo.to(o.dtype) - 1) * q
        lo = torch.floor(pos).long()
        hi = torch.ceil(pos).long()
        lo = torch.minimum(lo, n_loo - 1)
        hi = torch.minimum(hi, n_loo - 1)
        w = (pos - lo.to(o.dtype)).clamp(0, 1)

        r = inv_rank  # (N,G)
        # Map LOO order-stat index k to full-sample index, skipping r if excluded is finite.
        # If excluded is NaN (inf), we don't skip anything in finite part.
        def map_k(k: torch.Tensor) -> torch.Tensor:
            skip = excluded_finite
            # if skip and k >= r -> k+1 else k
            return torch.where(skip & (k >= r), k + 1, k)

        lo_full = map_k(lo)
        hi_full = map_k(hi)

        v_lo = o_sorted.gather(0, lo_full)
        v_hi = o_sorted.gather(0, hi_full)
        out = (1 - w) * v_lo + w * v_hi

        # Any gridpoints with <2 finite values after exclusion are unreliable -> NaN
        out = torch.where(n_loo >= 2, out, torch.full_like(out, torch.nan))
        # If out is inf (all NaN), make NaN
        out[~torch.isfinite(out)] = torch.nan
        return out

    q1 = loo_quantile(1/3)
    q2 = loo_quantile(2/3)
    return q1, q2


def _cat_tercile(z: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Categorize z into {1,2,3} using tercile thresholds.
    z, q1, q2 are broadcastable to the same shape.
    """
    c = torch.full_like(z, 3, dtype=torch.int64)
    c = torch.where(z <= q2, torch.full_like(c, 2), c)
    c = torch.where(z <= q1, torch.full_like(c, 1), c)
    # keep NaNs as 0 category marker
    c = torch.where(torch.isnan(z), torch.zeros_like(c), c)
    return c


def rpss_tercile_ensemble(
    f: Union[np.ndarray, torch.Tensor],
    o: Union[np.ndarray, torch.Tensor],
    p: Union[np.ndarray, torch.Tensor],
    tercile_thresholds: Optional[np.ndarray] = None,
    leave_one_out: bool = False,
    lat_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Union[str, torch.device] = "cpu",
    eps: float = 1e-12,
    chunk_size: int = 32,
) -> Dict[str, Any]:
    """
    Compute tercile-based RPSS and BSS for ensemble forecasts vs observations with a persistence reference.

    Parameters
    ----------
    f : array-like
        Forecasts.
        Supported shapes:
          - deterministic: (N, G) or (N, L, G)
          - ensemble:      (M, N, G) or (M, N, L, G)
    o : array-like
        Verifying observations, shape (N, G) or (N, L, G).
    p : array-like
        Persistence/reference forecasts, shape (N, G) or (N, L, G).
    tercile_thresholds : np.ndarray, optional
        Precomputed tercile thresholds with shape (G, 2) giving (q1,q2) per grid point.
        If provided, the same thresholds are used for all cases and leads.
        If None, thresholds are estimated from `o`:
          - leave_one_out=False: q1,q2 are computed from all N cases (per lead if L present).
          - leave_one_out=True : q1,q2 are computed by excluding each verifying case (per lead if L present).
    leave_one_out : bool
        Whether to use LOO tercile thresholds when tercile_thresholds is None.
    lat_weights : array-like, optional
        Area weights for aggregation, shape (G,). Typically cos(lat).
        If None, uses uniform weights.
    device : str or torch.device
        Compute device ("cpu", "cuda", "mps", ...).
    eps : float
        Small stabilizer to avoid division by ~0 in skill scores.
    chunk_size : int
        Number of verifying cases (N) processed per chunk to reduce peak memory.

    Returns
    -------
    out : dict
        Contains (lead dimension present only if L present in inputs):

        Gridpoint means (numpy):
          - 'rps_grid'      : (G,) or (L,G) mean RPS of ensemble forecast
          - 'rps_ref_grid'  : (G,) or (L,G) mean RPS of persistence reference
          - 'rpss_grid'     : (G,) or (L,G) RPSS = 1 - rps_grid / rps_ref_grid

          - 'bs_grid'       : dict with keys 'p1','p2','p3', each (G,) or (L,G) mean BS
          - 'bs_ref_grid'   : dict with keys 'p1','p2','p3', each (G,) or (L,G) mean BS of reference
          - 'bss_grid'      : dict with keys 'p1','p2','p3', each (G,) or (L,G) BSS = 1 - bs/bs_ref

        Area-weighted (ratio-of-sums) scalars (numpy):
          - 'rps_area'      : scalar or (L,) area-weighted mean RPS
          - 'rps_ref_area'  : scalar or (L,) area-weighted mean RPS of reference
          - 'rpss_area'     : scalar or (L,) area-weighted RPSS = 1 - sum(w*rps)/sum(w*rps_ref)

          - 'bs_area'       : dict 'p1','p2','p3' scalar or (L,) area-weighted mean BS
          - 'bs_ref_area'   : dict 'p1','p2','p3' scalar or (L,) area-weighted mean BS of reference
          - 'bss_area'      : dict 'p1','p2','p3' scalar or (L,) area-weighted BSS ratio-of-sums

        Misc:
          - 'weights'       : (G,) numpy weights used for area aggregation (normalized to mean 1)
    """
    device = torch.device(device)
    f_t = _to_torch(f, device)
    o_t = _to_torch(o, device)
    p_t = _to_torch(p, device)

    # Normalize shapes to:
    #   Deterministic: f: (N, L, G), o,p: (N, L, G)
    #   Ensemble: f: (M, N, L, G), o,p: (N, L, G)
    # First, ensure o,p are (N,L,G) by adding L dimension if needed
    if o_t.ndim == 2:
        o_t = o_t[:, None, :]  # (N,G) -> (N,1,G)
    if p_t.ndim == 2:
        p_t = p_t[:, None, :]  # (N,G) -> (N,1,G)
    
    # Now determine if f is deterministic (3D) or ensemble (4D)
    if f_t.ndim == 2:
        # (N,G) deterministic -> (N,1,G)
        f_t = f_t[:, None, :]
        M = 1
    elif f_t.ndim == 3:
        # (N,L,G) deterministic
        M = 1
    elif f_t.ndim == 4:
        # (M,N,L,G) ensemble
        M = f_t.shape[0]
    else:
        raise ValueError(f"Unsupported f.ndim={f_t.ndim}; expected 2, 3, or 4 dimensions")

    # Extract (N,L,G) from forecast
    if f_t.ndim == 3:
        N, L, G = f_t.shape
        f_t = f_t[None, :, :, :]  # (N,L,G) -> (1,N,L,G)
    else:
        M, N, L, G = f_t.shape

    if o_t.shape != (N, L, G):
        raise ValueError(f"o must have shape (N,L,G)={(N,L,G)} after normalization; got {tuple(o_t.shape)}")
    if p_t.shape != (N, L, G):
        raise ValueError(f"p must have shape (N,L,G)={(N,L,G)} after normalization; got {tuple(p_t.shape)}")

    # weights
    if lat_weights is None:
        w = torch.ones(G, device=device, dtype=torch.float32)
    else:
        w = _to_torch(lat_weights, device).to(torch.float32)
        if w.ndim != 1 or w.shape[0] != G:
            raise ValueError(f"lat_weights must have shape (G,)={(G,)}; got {tuple(w.shape)}")
        w = torch.clamp(w, min=0.0)
    # normalize to mean 1 for numerical stability
    w = w / (w.mean() + eps)

    # handle provided thresholds
    if tercile_thresholds is not None:
        tt = np.asarray(tercile_thresholds)
        if tt.shape != (G, 2):
            raise ValueError(f"tercile_thresholds must have shape (G,2)={(G,2)}; got {tt.shape}")
        tt_t = torch.from_numpy(tt).to(device=device, dtype=o_t.dtype)
        q1_fixed = tt_t[:, 0]
        q2_fixed = tt_t[:, 1]
    else:
        q1_fixed = None
        q2_fixed = None

    # accumulators per lead
    sum_rps = torch.zeros((L, G), device=device, dtype=torch.float32)
    sum_rps_ref = torch.zeros((L, G), device=device, dtype=torch.float32)
    cnt_rps = torch.zeros((L, G), device=device, dtype=torch.float32)

    sum_bs = {k: torch.zeros((L, G), device=device, dtype=torch.float32) for k in ("p1","p2","p3")}
    sum_bs_ref = {k: torch.zeros((L, G), device=device, dtype=torch.float32) for k in ("p1","p2","p3")}
    cnt_bs = {k: torch.zeros((L, G), device=device, dtype=torch.float32) for k in ("p1","p2","p3")}

    # loop over leads to keep memory bounded and keep LOO logic simple
    with torch.no_grad():
        for l in range(L):
            o_ng = o_t[:, l, :]  # (N,G)
            p_ng = p_t[:, l, :]
            f_mng = f_t[:, :, l, :]  # (M,N,G)

            # thresholds
            if q1_fixed is not None:
                q1_ng = q1_fixed[None, :].expand(N, G)
                q2_ng = q2_fixed[None, :].expand(N, G)
            else:
                if leave_one_out:
                    q1_ng, q2_ng = _compute_loo_terciles(o_ng)
                else:
                    q1 = torch.nanquantile(o_ng, 1/3, dim=0)  # (G,)
                    q2 = torch.nanquantile(o_ng, 2/3, dim=0)
                    q1_ng = q1[None, :].expand(N, G)
                    q2_ng = q2[None, :].expand(N, G)

            # obs category and cumulative indicators
            c_obs = _cat_tercile(o_ng, q1_ng, q2_ng)  # (N,G) with 0 for NaN
            valid = c_obs > 0  # (N,G)

            O1 = (c_obs == 1).to(torch.float32)
            O2 = ((c_obs == 1) | (c_obs == 2)).to(torch.float32)

            # stream over cases
            for i0 in range(0, N, chunk_size):
                i1 = min(N, i0 + chunk_size)
                sl = slice(i0, i1)

                q1_c = q1_ng[sl, :]
                q2_c = q2_ng[sl, :]
                o1 = O1[sl, :]
                o2 = O2[sl, :]
                v = valid[sl, :]

                # ensemble probs for chunk
                f_chunk = f_mng[:, sl, :]  # (M,chunk,G)
                p1 = (f_chunk <= q1_c[None, :, :]).to(torch.float32).mean(dim=0)              # (chunk,G)
                p2 = ((f_chunk > q1_c[None, :, :]) & (f_chunk <= q2_c[None, :, :])).to(torch.float32).mean(dim=0)
                p3 = (f_chunk > q2_c[None, :, :]).to(torch.float32).mean(dim=0)

                F1 = p1
                F2 = p1 + p2

                rps = (F1 - o1) ** 2 + (F2 - o2) ** 2  # (chunk,G)

                # reference probs (deterministic persistence)
                c_ref = _cat_tercile(p_ng[sl, :], q1_c, q2_c)
                # If ref is NaN, we treat as invalid for ref too (will be masked by v_ref)
                v_ref = (c_ref > 0) & v
                R1 = (c_ref == 1).to(torch.float32)
                R2 = ((c_ref == 1) | (c_ref == 2)).to(torch.float32)
                rps_ref = (R1 - o1) ** 2 + (R2 - o2) ** 2
                
                # accumulate RPS
                rps = torch.where(v, rps, torch.zeros_like(rps))
                rps_ref = torch.where(v_ref, rps_ref, torch.zeros_like(rps_ref))

                sum_rps[l, :] += rps.sum(dim=0)
                sum_rps_ref[l, :] += rps_ref.sum(dim=0)
                cnt_rps[l, :] += v.to(torch.float32).sum(dim=0)
               
                # BS for each category (event = cat==k), forecast prob = pk, ref prob = 1/0 from c_ref
                y1 = (c_obs[sl, :] == 1).to(torch.float32)
                y2 = (c_obs[sl, :] == 2).to(torch.float32)
                y3 = (c_obs[sl, :] == 3).to(torch.float32)

                pr1 = (c_ref == 1).to(torch.float32)
                pr2 = (c_ref == 2).to(torch.float32)
                pr3 = (c_ref == 3).to(torch.float32)

                for key, pf, yr, pr in (
                    ("p1", p1, y1, pr1),
                    ("p2", p2, y2, pr2),
                    ("p3", p3, y3, pr3),
                ):
                    bs = (pf - yr) ** 2
                    bs_ref = (pr - yr) ** 2
                    bs = torch.where(v, bs, torch.zeros_like(bs))
                    bs_ref = torch.where(v_ref, bs_ref, torch.zeros_like(bs_ref))

                    sum_bs[key][l, :] += bs.sum(dim=0)
                    sum_bs_ref[key][l, :] += bs_ref.sum(dim=0)
                    cnt_bs[key][l, :] += v.to(torch.float32).sum(dim=0)

    # finalize means
    rps_grid = sum_rps / (cnt_rps + eps)
    rps_ref_grid = sum_rps_ref / (cnt_rps + eps)  # same denom for mean; skill uses ratio with eps below
    # Avoid huge negative RPSS when rps_ref is near zero: add eps to both numerator and denominator
    rpss_grid = torch.where(
        rps_ref_grid + eps > 0,
        1.0 - (rps_grid / (rps_ref_grid + eps)),
        torch.zeros_like(rps_grid)
    )

    bs_grid = {k: (sum_bs[k] / (cnt_bs[k] + eps)) for k in ("p1","p2","p3")}
    bs_ref_grid = {k: (sum_bs_ref[k] / (cnt_bs[k] + eps)) for k in ("p1","p2","p3")}
    bss_grid = {k: (1.0 - (bs_grid[k] / (bs_ref_grid[k] + eps))) for k in ("p1","p2","p3")}

    # area-weighted ratio-of-sums: sum(w * sum_rps) / sum(w * sum_rps_ref)
    wG = w[None, :]  # (1,G)
    rps_area = (wG * sum_rps).sum(dim=1) / ((wG * cnt_rps).sum(dim=1) + eps)
    rps_ref_area = (wG * sum_rps_ref).sum(dim=1) / ((wG * cnt_rps).sum(dim=1) + eps)
    # Avoid huge negative RPSS: add eps to denominator and check for zero reference
    rps_ref_denom = (wG * sum_rps_ref).sum(dim=1)
    rpss_area = torch.where(
        rps_ref_denom + eps > 0,
        1.0 - ((wG * sum_rps).sum(dim=1) / (rps_ref_denom + eps)),
        torch.zeros_like(rps_ref_denom)
    )

    bs_area = {}
    bs_ref_area = {}
    bss_area = {}
    for k in ("p1","p2","p3"):
        bs_area[k] = (wG * sum_bs[k]).sum(dim=1) / ((wG * cnt_bs[k]).sum(dim=1) + eps)
        bs_ref_area[k] = (wG * sum_bs_ref[k]).sum(dim=1) / ((wG * cnt_bs[k]).sum(dim=1) + eps)
        bs_ref_denom = (wG * sum_bs_ref[k]).sum(dim=1)
        bss_area[k] = torch.where(
            bs_ref_denom + eps > 0,
            1.0 - ((wG * sum_bs[k]).sum(dim=1) / (bs_ref_denom + eps)),
            torch.zeros_like(bs_ref_denom)
        )
    
    # if L==1, squeeze lead dimension for backwards compatibility
    def maybe_squeeze(x: torch.Tensor):
        return x.squeeze(0) if L == 1 else x

    out = {
        "rps_grid": maybe_squeeze(rps_grid).detach().cpu().numpy(),
        "rps_ref_grid": maybe_squeeze(rps_ref_grid).detach().cpu().numpy(),
        "rpss_grid": maybe_squeeze(rpss_grid).detach().cpu().numpy(),
        "rps_area": maybe_squeeze(rps_area).detach().cpu().numpy(),
        "rps_ref_area": maybe_squeeze(rps_ref_area).detach().cpu().numpy(),
        "rpss_area": maybe_squeeze(rpss_area).detach().cpu().numpy(),
        "bs_grid": {k: maybe_squeeze(v).detach().cpu().numpy() for k, v in bs_grid.items()},
        "bs_ref_grid": {k: maybe_squeeze(v).detach().cpu().numpy() for k, v in bs_ref_grid.items()},
        "bss_grid": {k: maybe_squeeze(v).detach().cpu().numpy() for k, v in bss_grid.items()},
        "bs_area": {k: maybe_squeeze(v).detach().cpu().numpy() for k, v in bs_area.items()},
        "bs_ref_area": {k: maybe_squeeze(v).detach().cpu().numpy() for k, v in bs_ref_area.items()},
        "bss_area": {k: maybe_squeeze(v).detach().cpu().numpy() for k, v in bss_area.items()},
        "weights": w.detach().cpu().numpy(),
    }
    return out



def compute_rpss_by_lead(
    f: Union[np.ndarray, torch.Tensor],
    o: Union[np.ndarray, torch.Tensor],
    p: Union[np.ndarray, torch.Tensor],
    *,
    tercile_thresholds: Optional[np.ndarray] = None,
    leave_one_out: bool = False,
    lat_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Union[str, torch.device] = "cpu",
    eps: float = 1e-12,
    chunk_size: int = 32,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper that computes RPSS/BSS across lead times.

    This function is meant for the common case where inputs are provided with an explicit
    lead dimension L (your convention: (N, L, G) for deterministic forecasts / observations).

    It forwards all arguments to :func:`rpss_tercile_ensemble`, which natively supports
    both lead-free and lead-aware shapes.

    Parameters
    ----------
    f, o, p : array-like
        Forecasts, observations, and reference (e.g., persistence). See :func:`rpss_tercile_ensemble`
        for supported shapes. In particular:
          - deterministic: f (N, G) or (N, L, G)
          - ensemble:      f (M, N, G) or (M, N, L, G)
          - obs/ref:       o, p (N, G) or (N, L, G)
    tercile_thresholds : np.ndarray, optional
        Fixed tercile thresholds. If provided, must be either:
          - (2, G) and applied to all leads, or
          - (2, L, G) lead-dependent thresholds.
        If None, thresholds are estimated from `o` (or via LOO if `leave_one_out=True`).
    leave_one_out : bool
        If True and `tercile_thresholds` is None, compute leave-one-out tercile thresholds.
    lat_weights : array-like, optional
        Spatial weights of shape (G,). Used for area-averaged summaries.
    device : str or torch.device
        Device to run on (e.g., 'cpu', 'cuda', 'mps').
    eps : float
        Small constant to stabilize divisions in skill scores.
    chunk_size : int
        Number of verifying cases processed per chunk (reduces peak memory).
    verbose : bool
        If True, print area-mean RPSS and BSS (p1/p2/p3) per lead (or once if no L).

    Returns
    -------
    out : dict
        The full output dictionary from :func:`rpss_tercile_ensemble`, including:
          - rpss_grid : (G,) or (L, G)
          - rpss_area : () or (L,)
          - rps_grid, rps_ref_grid, rps_area, rps_ref_area
          - bss_grid, bs_grid, bs_ref_grid
          - bss_area, bs_area, bs_ref_area
          - weights
    """
    out = rpss_tercile_ensemble(
        f=f,
        o=o,
        p=p,
        tercile_thresholds=tercile_thresholds,
        leave_one_out=leave_one_out,
        lat_weights=lat_weights,
        device=device,
        eps=eps,
        chunk_size=chunk_size,
    )

    if verbose:
        rpss_area = out.get("rpss_area", None)
        bss_area = out.get("bss_area", {})
        # rpss_area is either scalar () or (L,)
        if isinstance(rpss_area, np.ndarray) and rpss_area.ndim == 1:
            L = rpss_area.shape[0]
            for l in range(L):
                p1 = bss_area.get("p1", np.full((L,), np.nan))[l]
                p2 = bss_area.get("p2", np.full((L,), np.nan))[l]
                p3 = bss_area.get("p3", np.full((L,), np.nan))[l]
                print(
                    f"Lead {l:02d}: RPSS(area)={rpss_area[l]: .4f} | "
                    f"BSS(area) p1={p1: .4f}, p2={p2: .4f}, p3={p3: .4f}",
                    flush=True,
                )
        else:
            # scalar
            p1 = bss_area.get("p1", np.nan)
            p2 = bss_area.get("p2", np.nan)
            p3 = bss_area.get("p3", np.nan)
            if isinstance(p1, np.ndarray): p1 = float(np.squeeze(p1))
            if isinstance(p2, np.ndarray): p2 = float(np.squeeze(p2))
            if isinstance(p3, np.ndarray): p3 = float(np.squeeze(p3))
            if isinstance(rpss_area, np.ndarray): rpss_area = float(np.squeeze(rpss_area))
            print(
                f"RPSS(area)={rpss_area: .4f} | "
                f"BSS(area) p1={p1: .4f}, p2={p2: .4f}, p3={p3: .4f}",
                flush=True,
            )

    return out


def compute_rpss_dyn(
    f: Union[np.ndarray, torch.Tensor],
    o: Union[np.ndarray, torch.Tensor],
    p: Union[np.ndarray, torch.Tensor],
    *,
    tercile_thresholds: Optional[np.ndarray] = None,
    leave_one_out: bool = False,
    lat_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Union[str, torch.device] = "cpu",
    eps: float = 1e-12,
    chunk_size: int = 32,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Dynamic-forecast wrapper (lead-aware) for RPSS/BSS.

    In earlier iterations of this project, a ``compute_rpss_dyn`` helper was used when
    forecasts/observations needed time alignment outside this module. In this refactored
    implementation, alignment is assumed to have already been applied so that `f`, `o`, `p`
    share the same verifying cases and lead dimension(s).

    This function currently delegates to :func:`compute_rpss_by_lead` and exists for API
    compatibility with pipelines that expect a ``compute_rpss_dyn`` entry point.

    Parameters
    ----------
    f, o, p : array-like
        See :func:`compute_rpss_by_lead`.
    kwargs : Any
        Ignored extras for backward compatibility.

    Returns
    -------
    out : dict
        Same as :func:`compute_rpss_by_lead`.
    """
    if kwargs:
        print(f"compute_rpss_dyn: ignoring extra kwargs: {sorted(kwargs.keys())}", flush=True)

    return compute_rpss_by_lead(
        f=f,
        o=o,
        p=p,
        tercile_thresholds=tercile_thresholds,
        leave_one_out=leave_one_out,
        lat_weights=lat_weights,
        device=device,
        eps=eps,
        chunk_size=chunk_size,
        verbose=verbose,
    )


def save_rpss_to_csv(rpss_output: Dict[str, Any], filepath: str) -> None:
    """
    Save RPSS and BSS metrics to a CSV file with leads as rows and metrics as columns.
    
    Parameters
    ----------
    rpss_output : dict
        Dictionary output from rpss_tercile_ensemble, compute_rpss_by_lead, or compute_rpss_dyn.
        Must contain keys: 'rpss_area' and 'bss_area'.
    filepath : str
        Path to save the CSV file.
        
    Returns
    -------
    None
        
    Example
    -------
    >>> out = compute_rpss_by_lead(f, o, p)
    >>> save_rpss_to_csv(out, 'rpss_results.csv')
    """
    import csv
    
    rpss_area = rpss_output.get("rpss_area")
    bss_area = rpss_output.get("bss_area", {})
    
    if rpss_area is None:
        raise ValueError("rpss_output must contain 'rpss_area' key")
    
    # Convert to numpy if needed
    if isinstance(rpss_area, np.ndarray):
        rpss_area = np.atleast_1d(rpss_area)
    else:
        rpss_area = np.array([rpss_area])
    
    # Extract BSS for each category
    bss_p1 = bss_area.get("p1", np.full(rpss_area.shape, np.nan))
    bss_p2 = bss_area.get("p2", np.full(rpss_area.shape, np.nan))
    bss_p3 = bss_area.get("p3", np.full(rpss_area.shape, np.nan))
    
    if isinstance(bss_p1, np.ndarray):
        bss_p1 = np.atleast_1d(bss_p1)
    else:
        bss_p1 = np.array([bss_p1])
    
    if isinstance(bss_p2, np.ndarray):
        bss_p2 = np.atleast_1d(bss_p2)
    else:
        bss_p2 = np.array([bss_p2])
    
    if isinstance(bss_p3, np.ndarray):
        bss_p3 = np.atleast_1d(bss_p3)
    else:
        bss_p3 = np.array([bss_p3])
    
    # Determine number of leads
    n_leads = len(rpss_area)
    
    # Write CSV
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header row
        writer.writerow(['Lead', 'RPSS(area)', 'BSS(area) p1', 'BSS(area) p2', 'BSS(area) p3'])
        
        # Data rows
        for lead_idx in range(n_leads):
            writer.writerow([
                lead_idx,
                f"{rpss_area[lead_idx]:.2f}",
                f"{bss_p1[lead_idx]:.2f}",
                f"{bss_p2[lead_idx]:.2f}",
                f"{bss_p3[lead_idx]:.2f}",
            ])


def comparison_scores_to_latex(testval_file, dynval_file, title="", region="", output_file=None, decimal_places=2):
    """
    Generate a LaTeX table comparing testval (DeepSeason) and dynval (GCM) validation scores.
    
    Creates a professional LaTeX table with dual columns for each metric (RPSS, BSS p1, p2, p3),
    comparing GCM (from dynval) and DeepSeason (from testval) results.
    
    Parameters
    ==========
    testval_file : str
        Path to the testval CSV file (DeepSeason results)
        Expected columns: Lead, RPSS(area), BSS(area) p1, BSS(area) p2, BSS(area) p3
        
    dynval_file : str
        Path to the dynval CSV file (GCM results)
        Expected columns: Lead, RPSS(area), BSS(area) p1, BSS(area) p2, BSS(area) p3
        
    title : str
        Title for the table caption (e.g., "Global SST")
        
    region : str
        Region description for the caption (e.g., "Area-aggregated")
        
    output_file : str or None
        Path to write the LaTeX table. If None, generates a default filename
        
    decimal_places : int
        Number of decimal places for rounding (default: 2)
        
    Returns
    =======
    latex_table : str
        LaTeX formatted table
        
    Examples
    ========
    >>> table = comparison_scores_to_latex(
    ...     'testval_Y621s.csv',
    ...     'dynval_Y621s.csv',
    ...     title='Global SST',
    ...     region='Area-aggregated',
    ...     output_file='comparison_table.tex'
    ... )
    """
    
    import os
    import pandas as pd
    
    # Read both CSV files
    test_df = pd.read_csv(testval_file)
    dyn_df = pd.read_csv(dynval_file)
    
    # Merge on Lead column
    merged_df = pd.merge(test_df, dyn_df, on='Lead', suffixes=('_test', '_dyn'), how='outer')
    merged_df = merged_df.sort_values('Lead')
    
    # Extract columns
    lead_col = merged_df['Lead']
    
    # RPSS columns
    rpss_dyn = merged_df['RPSS(area)_dyn']
    rpss_test = merged_df['RPSS(area)_test']
    
    # BSS columns
    bss_p1_dyn = merged_df['BSS(area) p1_dyn']
    bss_p1_test = merged_df['BSS(area) p1_test']
    
    bss_p2_dyn = merged_df['BSS(area) p2_dyn']
    bss_p2_test = merged_df['BSS(area) p2_test']
    
    bss_p3_dyn = merged_df['BSS(area) p3_dyn']
    bss_p3_test = merged_df['BSS(area) p3_test']
    
    # Build LaTeX table
    latex_lines = []
    
    latex_lines.append('\\begin{table}[ht]')
    latex_lines.append('\\centering')
    
    # Caption
    caption = f'\\textbf{{{title}}}. {region} RPSS and Brier Skill Scores (BSS) for tercile events '
    caption += '($p_1$, $p_2$, $p_3$) as a function of lead time. Each metric is shown for the '
    caption += 'operational GCM and DeepSeason.'
    latex_lines.append(f'\\caption{{{caption}}}')
    latex_lines.append(f'\\label{{tab:{title.lower().replace(" ", "_")}}}')
    
    # Table opening with column specification
    latex_lines.append('\\begin{tabular}{r*{8}{r}}')
    latex_lines.append('\\toprule')
    
    # Header row 1: Metric names with multicolumn
    header1 = ' & \\multicolumn{2}{c}{RPSS} & \\multicolumn{2}{c}{BSS($p_1$)} & \\multicolumn{2}{c}{BSS($p_2$)} & \\multicolumn{2}{c}{BSS($p_3$)} \\\\'
    latex_lines.append(header1)
    
    # Partial rules after metric names
    latex_lines.append('\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}')
    
    # Header row 2: Column names
    header2 = 'Lead & GCM & DeepSeason & GCM & DeepSeason & GCM & DeepSeason & GCM & DeepSeason \\\\'
    latex_lines.append(header2)
    latex_lines.append('\\midrule')
    
    # Data rows
    for idx, row in merged_df.iterrows():
        lead = int(row['Lead'])
        
        # Build row string
        row_str = str(lead)
        
        # RPSS columns
        rpss_dyn_val = row['RPSS(area)_dyn']
        rpss_test_val = row['RPSS(area)_test']
        row_str += f' & {rpss_dyn_val:.{decimal_places}f}' if pd.notna(rpss_dyn_val) else ' & '
        row_str += f' & {rpss_test_val:.{decimal_places}f}' if pd.notna(rpss_test_val) else ' & '
        
        # BSS p1 columns
        bss_p1_dyn_val = row['BSS(area) p1_dyn']
        bss_p1_test_val = row['BSS(area) p1_test']
        row_str += f' & {bss_p1_dyn_val:.{decimal_places}f}' if pd.notna(bss_p1_dyn_val) else ' & '
        row_str += f' & {bss_p1_test_val:.{decimal_places}f}' if pd.notna(bss_p1_test_val) else ' & '
        
        # BSS p2 columns
        bss_p2_dyn_val = row['BSS(area) p2_dyn']
        bss_p2_test_val = row['BSS(area) p2_test']
        row_str += f' & {bss_p2_dyn_val:.{decimal_places}f}' if pd.notna(bss_p2_dyn_val) else ' & '
        row_str += f' & {bss_p2_test_val:.{decimal_places}f}' if pd.notna(bss_p2_test_val) else ' & '
        
        # BSS p3 columns
        bss_p3_dyn_val = row['BSS(area) p3_dyn']
        bss_p3_test_val = row['BSS(area) p3_test']
        row_str += f' & {bss_p3_dyn_val:.{decimal_places}f}' if pd.notna(bss_p3_dyn_val) else ' & '
        row_str += f' & {bss_p3_test_val:.{decimal_places}f}' if pd.notna(bss_p3_test_val) else ' & '
        
        row_str += ' \\\\'
        latex_lines.append(row_str)
    
    # Table closing
    latex_lines.append('\\bottomrule')
    latex_lines.append('\\end{tabular}')
    latex_lines.append('\\end{table}')
    
    latex_table = '\n'.join(latex_lines)
    
    # Generate default output filename if not specified
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(testval_file))[0]
        output_file = f"{base_name}_comparison.tex"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table written to {output_file}")
    
    return latex_table
