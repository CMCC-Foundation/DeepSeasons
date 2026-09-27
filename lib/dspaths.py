"""
dspaths -- central path configuration for the DeepSeasons repository.

Every notebook and script imports its locations from here, so the code runs
unchanged on any machine. Override the defaults with environment variables
(recommended) or by editing the values below.

    DEEPSEASONS_DATA      root of the monthly ERA5 archive       (default: <repo>/data/ERA5_DATA)
    DEEPSEASONS_SEASONAL  root of the C3S seasonal hindcasts     (default: <repo>/data/SEASONAL)
    DEEPSEASONS_WORKDIR   where models, pickles, CSVs, figures go (default: <repo>/work)
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = REPO_ROOT / 'lib'

DATA_DIR = os.environ.get('DEEPSEASONS_DATA', str(REPO_ROOT / 'data' / 'ERA5_DATA'))
SEASONAL_DIR = os.environ.get('DEEPSEASONS_SEASONAL', str(REPO_ROOT / 'data' / 'SEASONAL'))
WORK_DIR = os.environ.get('DEEPSEASONS_WORKDIR', str(REPO_ROOT / 'work'))


def setup(chdir=True, verbose=True):
    """Put the bundled libraries on sys.path, create WORK_DIR and (optionally) cd into it."""
    for p in (str(LIB_DIR / 'Zapata'), str(LIB_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.makedirs(WORK_DIR, exist_ok=True)
    os.environ.setdefault('DEEPSEASONS_WORKDIR', WORK_DIR)
    if chdir:
        os.chdir(WORK_DIR)
    if verbose:
        print(f'DeepSeasons repo : {REPO_ROOT}')
        print(f'ERA5 data        : {DATA_DIR}')
        print(f'Seasonal data    : {SEASONAL_DIR}')
        print(f'Working directory: {os.getcwd() if chdir else WORK_DIR}')
    return REPO_ROOT


def seasonal_dir(field):
    """Directory with the seasonal hindcasts for `field` (e.g. 'SST', 'T2M')."""
    return os.path.join(SEASONAL_DIR, f'SEASONAL_{field}')
