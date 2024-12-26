# Raman Analyzer

![Fitting Results](./fit.png)

**Raman Analyzer** is an open-source Python package that streamlines the preprocessing and analysis of Raman spectroscopy data. This repository provides functions to:
1. **Parse raw CSV files** into DataFrames.
2. **Despike** (remove cosmic rays).
3. **Average** multiple spectra to improve SNR.
4. **Perform baseline correction** (using ALS or IARPLS).
5. **Smooth** noisy spectra (Savitzky–Golay, optional).
6. **Convert** wavelength to wavenumber.
7. **Fit Voigt peaks** iteratively, adding additional peaks from residuals.

## Features

- **Cosmic Ray Removal**  
  Uses a modified Z-score to detect and remove high-intensity spikes caused by cosmic rays.
- **Averaging**  
  Interpolates multiple DataFrames onto a common wavenumber grid and averages to improve signal-to-noise ratio (SNR).
- **Baseline Correction**  
  - Asymmetric Least Squares (ALS) method  
  - Improved Asymmetric Reweighted Penalized Least Squares (IARPLS) method
- **Despiking**  
  Easily remove outliers while preserving spectral features.
- **Convert Wavelength to Wavenumber**  
  Convert an axis in nanometers (nm) to wavenumbers (cm⁻¹) using the excitation wavelength.
- **Voigt Peak Fitting**  
  Automatically detect peaks in the spectrum and fit them to Voigt profiles. An iterative approach updates the fit by adding peaks found in the residual.

## Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/mkizilov/RamanAnalyzer.git

2. **Install the required packages**  
    Manually install the libraries in your Python environment:
    ```bash
    pip install numpy scipy pandas matplotlib lmfit
    ```

3. **Import and use**  
    In your Python scripts or Jupyter notebook, import the module:
    ```python
    import RamanAnalyzer as ra
    ```
    Then call the functions on your data.

## Usage Examples

Below is a short demonstration of how to use RamanAnalyzer in a Jupyter notebook. For a complete example, see `example.ipynb`.

```python
import RamanAnalyzer as ra

# 1. Parse CSV (returns a list of DataFrames)
dataframes = ra.raman_parser('data/sample_raman.csv')

# 2. Despike
df_despiked = ra.despike_raman(dataframes[0], moving_average=2, threshold=7, plot=True)

# 3. Baseline correction
df_baselined = ra.estimate_baseline(df_despiked, lam=1e5, p=0.001, niter=10, plot=True)

# 4. Convert wavelength to wavenumber (if needed)
df_wavenumber = ra.convert_wavelength_to_wavenumber(df_baselined, 532.0)

# 5. Fit Voigt peaks
fit_result = ra.fit_voigts(df_wavenumber, 
                                  threshold=10,
                                  min_dist=5, 
                                  min_height=5, 
                                  plot=True)
```

## Structure

```
RamanAnalyzer/
├── RamanAnalyzer.py        # Python module containing main data-processing functions
├── example.ipynb           # Jupyter Notebook demonstrating module usage
├── README.md              # This file
```

## Contributing

We welcome pull requests that fix bugs or add new features.

## License

This project is licensed under the MIT License. Feel free to use and modify the code within the terms of the license.