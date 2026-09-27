#!/usr/bin/env python
"""
download_era5.py -- fetch the ERA5 monthly means used by the DeepSeasons notebooks
from the Copernicus Climate Data Store (CDS) and store them in the layout expected
by zapata.data.read_era5:

    $DEEPSEASONS_DATA/SST/SST_SST_V5.nc      sea surface temperature     (variable `sst`)
    $DEEPSEASONS_DATA/T2M/t2m_t2m_V5.nc      2 m temperature             (variable `t2m`)
    $DEEPSEASONS_DATA/U/U_850_V5.nc          u wind at 850 hPa           (variable `u`)
    $DEEPSEASONS_DATA/T/T_850_V5.nc          temperature at 850 hPa      (variable `t`)

Monthly averaged reanalysis, 0.25 deg global grid, January 1940 - December 2022
(996 months, ~2 GB per file).

Prerequisites
-------------
1. A free CDS account: https://cds.climate.copernicus.eu
2. Accept the ERA5 licence on the dataset pages (see README).
3. `pip install "cdsapi>=0.7.2"` and a ~/.cdsapirc file with your API key:
       url: https://cds.climate.copernicus.eu/api
       key: <your-personal-access-token>

Usage
-----
    python download_era5.py                       # all four fields
    python download_era5.py --fields SST T2M      # a subset
    python download_era5.py --out /path/to/ERA5_DATA
"""
import argparse
import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'lib'))
import dspaths  # noqa: E402

YEARS = [str(y) for y in range(1940, 2023)]
MONTHS = [f'{m:02d}' for m in range(1, 13)]

# field key -> (CDS dataset, CDS variable, pressure level, sub-folder, file name, netCDF variable name)
FIELDS = {
    'SST': ('reanalysis-era5-single-levels-monthly-means', 'sea_surface_temperature', None, 'SST', 'SST_SST_V5.nc', 'sst'),
    'T2M': ('reanalysis-era5-single-levels-monthly-means', '2m_temperature', None, 'T2M', 't2m_t2m_V5.nc', 't2m'),
    'U850': ('reanalysis-era5-pressure-levels-monthly-means', 'u_component_of_wind', '850', 'U', 'U_850_V5.nc', 'u'),
    'T850': ('reanalysis-era5-pressure-levels-monthly-means', 'temperature', '850', 'T', 'T_850_V5.nc', 't'),
}


def normalise(src, dst, varname):
    """Make files from the new CDS back-end look like the originals (time/latitude/longitude dims)."""
    import xarray as xr
    ds = xr.open_dataset(src)
    if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    for extra in ('number', 'expver', 'pressure_level'):
        if extra in ds.dims:
            ds = ds.squeeze(extra, drop=True)
        elif extra in ds.coords:
            ds = ds.drop_vars(extra)
    ds = ds[[varname]].transpose('time', 'latitude', 'longitude').sortby('time')
    ds.to_netcdf(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fields', nargs='+', default=list(FIELDS), choices=list(FIELDS))
    ap.add_argument('--out', default=dspaths.DATA_DIR, help='root of the ERA5 archive (default: DEEPSEASONS_DATA)')
    ap.add_argument('--force', action='store_true', help='re-download existing files')
    args = ap.parse_args()

    import cdsapi
    client = cdsapi.Client()

    for key in args.fields:
        dataset, variable, level, folder, fname, varname = FIELDS[key]
        outdir = os.path.join(args.out, folder)
        os.makedirs(outdir, exist_ok=True)
        target = os.path.join(outdir, fname)
        if os.path.exists(target) and not args.force:
            print(f'[skip] {target} exists')
            continue
        request = {
            'product_type': ['monthly_averaged_reanalysis'],
            'variable': [variable],
            'year': YEARS,
            'month': MONTHS,
            'time': ['00:00'],
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        }
        if level is not None:
            request['pressure_level'] = [level]
        print(f'[get ] {key}: {dataset} / {variable}' + (f' {level} hPa' if level else ''))
        with tempfile.TemporaryDirectory(dir=outdir) as tmp:
            raw = os.path.join(tmp, 'raw.nc')
            client.retrieve(dataset, request, raw)
            normalise(raw, target, varname)
        print(f'[done] {target}')


if __name__ == '__main__':
    main()
