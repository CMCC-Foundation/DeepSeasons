# AIModels

PyTorch-based machine learning models for climate and weather prediction, including transformer architectures and ensemble verification metrics.

## Contents

### Core Models
- **TimeSeriesTransformer.py** - Transformer architecture for time series forecasting
- **ClimFormer.py** - Climate-specific transformer model
- **ClimFormerAttn.py**, **ClimFormerAttn2.py** - Attention mechanism variants
- **ClimLSTM.py** - LSTM-based climate model
- **LocalInformer.py** - Local attention Informer architecture
- **AutoEncoder.py** - Autoencoder for dimensionality reduction

### Utilities and Validation
- **AIClasses.py** - Core classes and data structures
- **AIutil.py** - Utility functions for data preparation and processing
- **Validation.py** - RPSS/BSS verification metrics and lead time construction
- **RPSS.py** - Tercile-based RPSS and Brier Skill Score computation
- **UtilPlot.py** - Plotting utilities
- **ModelTraining.py** - Training loop and utilities

### Testing and Examples
- **test_transformer.py** - Unit tests for transformer models
- **transformer_example.py** - Example usage of transformer models

## Features

- **Ensemble Verification**: Tercile-based RPSS and BSS computation with leave-one-out support
- **Lead Time Construction**: Convert observation timeseries into lead time dimension with climatology tracking
- **Device Support**: CPU, CUDA, and MPS (Apple Silicon) support with automatic dtype conversion
- **Flexible Data Handling**: Support for numpy arrays, xarray, and PyTorch tensors

## Installation

```bash
git clone https://github.com/CMCC-Foundation/AIModels.git
cd AIModels
pip install -r requirements.txt
```

## Usage

See example notebooks and `Notebook` folder for examples of usage. The notebook will require ERA5 monthly mean data in netcdf format as input data. The input functions in AIModels will require a naming convention for the ERA5 monthly mean files such that a U850 field, for instance has to be named as U_850_V5.nc and surface fields will repeat as SST_SST_V5.nc.

## License

See LICENSE file in the main Zapata repository.
