import numpy as np
from scipy.special import wofz
import pandas as pd
from lmfit import Parameters
from lmfit.models import VoigtModel
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse.linalg import spsolve


def raman_parser(file_path):
    """
    Reads a CSV file and splits it into a list of DataFrames.
    Each DataFrame has two columns: Wavenumber and Intensity.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        list of pd.DataFrame: List of DataFrames with Wavenumber and Intensity columns.
    """
    # Read the CSV file
    df = pd.read_csv(file_path, skiprows=1)  # Skip the first row with units
    
    # Rename the first column to 'Wavenumber' for clarity
    df.rename(columns={df.columns[0]: "Wavenumber"}, inplace=True)
    
    # List to store the DataFrames
    dataframes = []
    
    # Iterate over intensity columns (skip the first "Wavenumber" column)
    for col in df.columns[1:]:
        temp_df = pd.DataFrame({
            "Wavenumber": df["Wavenumber"],
            "Intensity": df[col]
        })
        dataframes.append(temp_df)
    
    return dataframes

def calculate_wavenumber_from_wavelength(excitation_wavelength_nm, raman_wavelength_nm):
    """
    Calculate the Raman shift (in cm^-1) from the excitation and Raman-scattered wavelengths.
    """
    # Convert both excitation and Raman wavelengths to wavenumbers (cm^-1)
    excitation_wavenumber = 1e7 / excitation_wavelength_nm
    raman_wavenumber = 1e7 / raman_wavelength_nm

    # Calculate the Raman shift
    raman_shift = excitation_wavenumber - raman_wavenumber

    return raman_shift
    # Create a function that converts wavelength to wavenumber for the Raman spectra
def convert_wavelength_to_wavenumber(df, excitation_wavelength_nm):
    """
    Convert wavelength data in a DataFrame to Raman shift wavenumbers.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing wavelength data
        excitation_wavelength_nm (float): Excitation wavelength in nm
    
    Returns:
        pd.DataFrame: New DataFrame with converted wavenumbers
    """
    df_copy = df.copy()
    
    # Convert wavelength column to wavenumber
    df_copy['Wavenumber'] = df_copy['Wavenumber'].apply(
        lambda x: calculate_wavenumber_from_wavelength(excitation_wavelength_nm, x)
    )
    
    return df_copy

def plot_raman(df, title_prefix="Intensity Plot"):
    """
    Plots DataFrame with Wavenumber vs Intensity with a single panel.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Wavenumber' and 'Intensity' columns.
        title_prefix (str): Title for the plot.
    """
    # Create figure with single plot
    plt.figure(figsize=(12, 6))
    
    # Plot data
    plt.plot(df['Wavenumber'], df['Intensity'], lw=1, color='black')
    plt.ylabel("Intensity/Arbitr. Units")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.title(title_prefix)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def average_raman(dataframes, resolution = None, plot=False):
    """Average a list of DataFrames, ensuring they have the same length and Wavenumber values."""
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    min_wavenumber = max(df['Wavenumber'].min() for df in dataframes)
    max_wavenumber = min(df['Wavenumber'].max() for df in dataframes)
    
    if resolution is None:
        resolution = max(df.shape[0] for df in dataframes)
    
    # Define the common wavenumber range for interpolation
    common_wavenumbers = np.linspace(min_wavenumber, max_wavenumber, resolution)
        
    # Interpolate each dataframe
    interpolated_dfs = []
    for df in dataframes:
        interp_func = interp1d(df['Wavenumber'], df['Intensity'], kind='linear', fill_value='extrapolate')
        interpolated_intensity = interp_func(common_wavenumbers)
        interpolated_dfs.append(pd.DataFrame({'Wavenumber': common_wavenumbers, 'Intensity': interpolated_intensity}))
    
    # Average the interpolated dataframes
    averaged_df = pd.concat(interpolated_dfs).groupby('Wavenumber').mean().reset_index()
    
    if plot:
        plt.figure(figsize=(12, 6))
        for df in interpolated_dfs:
            plt.plot(df['Wavenumber'], df['Intensity'], lw=1, color='gray', alpha=0.5)
        plt.plot(averaged_df['Wavenumber'], averaged_df['Intensity'], color='black', linewidth=2, label='Averaged Raman Spectrum')
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity/Arbitr. Units')
        plt.title('Averaged Raman Spectra')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return averaged_df

def cut_spectra(df, start=None, end=None):
    """Cut the DataFrame to the specified Raman Shift range."""
    if start is None:
        start = df['Wavenumber'].min()
    if end is None:
        end = df['Wavenumber'].max()
    mask = (df['Wavenumber'] >= start) & (df['Wavenumber'] <= end)
    return df.loc[mask].reset_index(drop=True)

def estimate_baseline(df, lam=10000000, p=0.05, niter=3, plot=False):
    """Estimate and subtract the baseline from the Raman spectra."""
    def baseline_als(y, lam, p, niter):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    counts = df['Intensity']
    baseline = baseline_als(counts, lam, p, niter)
    baselined_counts = counts - baseline

    df_baselined = df.copy()
    df_baselined['Intensity'] = baselined_counts

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Wavenumber'], counts, label='Spectra', color='black')
        plt.plot(df['Wavenumber'], baseline, label='Estimated Baseline', color='red', ls='--')
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity/Arbitr. Units')
        plt.title('Baseline Estimation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(df['Wavenumber'], baselined_counts, label='Baselined Spectra', color='black')
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity/Arbitr. Units')
        plt.title('Baselined Spectra')
        plt.legend()
        plt.grid(True)
        plt.tight_layout() 
        plt.show()

    return df_baselined

def estimate_baseline_iarpls(df, lam=1e7, niter=50, epsilon=1e-6, plot=False):
    """
    Estimate and subtract the baseline from Raman spectra using IarPLS algorithm.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'Wavenumber' and 'Intensity' columns.
        lam (float): Smoothness parameter for penalization.
        niter (int): Maximum number of iterations.
        epsilon (float): Convergence threshold.
        plot (bool): Whether to plot the results.

    Returns:
        pd.DataFrame: DataFrame with baseline-corrected 'Intensity'.
    """
    def isru_function(d, t, sigma_d):
        """ISRU weight function."""
        scaled_diff = (d - 2 * sigma_d) / sigma_d
        return 0.5 * (1 - np.exp(t * scaled_diff) / np.sqrt(1 + np.exp(2 * t * scaled_diff)))

    def iarpls(y, lam, niter, epsilon):
        """Improved Asymmetrically Reweighted Penalized Least Squares."""
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        z = np.zeros(L)

        for t in range(1, niter + 1):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + D
            z_prev = z.copy()
            z = spsolve(Z, w * y)

            diff = y - z
            neg_diff = diff[diff < 0]
            sigma_d = np.std(neg_diff) if len(neg_diff) > 0 else 1e-8

            w = np.ones(L)
            mask = diff > 0
            w[mask] = isru_function(diff[mask], t, sigma_d)

            if np.linalg.norm(w - w) / np.linalg.norm(w) < epsilon:
                break

        return z

    counts = df['Intensity'].values
    baseline = iarpls(counts, lam, niter, epsilon)
    baselined_counts = counts - baseline

    df_baselined = df.copy()
    df_baselined['Intensity'] = baselined_counts

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Wavenumber'], counts, label='Spectra', color='black')
        plt.plot(df['Wavenumber'], baseline, label='Estimated Baseline', color='red', ls='--')
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity/Arbitr. Units')
        plt.title('Baseline Estimation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(df['Wavenumber'], baselined_counts, label='Baselined Spectra', color='black')
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity/Arbitr. Units') 
        plt.title('Baselined Spectra')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df_baselined


def despike_raman(df, moving_average, threshold=7, plot=False):
    """Remove spikes from the Raman spectra using modified Z-score."""
    def modified_z_score(y_values):
        y_diff = np.diff(y_values)
        median_y = np.median(y_diff)
        mad_y = np.median([np.abs(y - median_y) for y in y_diff])
        return [0.6745 * (y - median_y) / mad_y for y in y_diff]

    def fix_spikes(y_values, ma, threshold):
        spikes = abs(np.array(modified_z_score(y_values))) > threshold
        y_out = y_values.copy()
        for i in range(len(spikes)):
            if spikes[i]:
                w = np.arange(max(0, i - ma), min(len(y_values) - 1, i + ma))
                we = w[spikes[w] == 0]
                if len(we) > 0:
                    y_out[i] = np.median(y_values[we])
                else:
                    y_out[i] = y_values[i]
        return y_out

    def plot_despiking_results(wavelength, original, despiked):
        plt.figure(figsize=(12, 6))
        plt.plot(wavelength, original, label='Original Spectra', color='black')
        plt.plot(wavelength, despiked, label='Despiked Spectra', color='red')
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity/Arbitr. Units')
        plt.title('Despiked Raman Spectrum')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    counts = df['Intensity']
    despiked_counts = fix_spikes(counts, moving_average, threshold)
    
    df_despiked = df.copy()
    df_despiked['Intensity'] = despiked_counts
    
    if plot:
        plot_despiking_results(df['Wavenumber'], counts, despiked_counts)
        plt.figure(figsize=(12, 6))
        plt.plot(df['Wavenumber'][1:], modified_z_score(counts), label='Modified Z-score', color='blue')
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Z-score')
        plt.title('Modified Z-score for Despiking')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df_despiked

def voigt_amplitude_from_height(height, sigma, gamma):
    """Calculate the amplitude of a Voigt profile given peak height, sigma, and gamma."""
    return height * sigma * np.sqrt(2 * np.pi) / \
        np.real(wofz((1j * gamma) / (sigma * np.sqrt(2))))

def voigt_height(amplitude, sigma, gamma):
    """Calculate the peak height of a Voigt profile given amplitude, sigma, and gamma."""
    return (amplitude / (sigma * np.sqrt(2 * np.pi))) * \
           np.real(wofz((1j * gamma) / (sigma * np.sqrt(2))))

def create_residual_models(
    indices,
    xs,
    residuals,
    iteration,
    max_amplitude,
    max_sigma,
    max_gamma,
    center_vary,
    min_height
):
    """
    Create Voigt models for residual peaks.

    Parameters:
    - indices: Indices of peaks in the residuals.
    - xs: Full x data.
    - residuals: Residuals data.
    - iteration: Iteration number.
    - max_amplitude, max_sigma, max_gamma: Bounds for parameters.
    - center_vary: How much to vary peak centers during fitting.
    - min_height: Minimum height for peaks.
    
    Returns:
    - model: Composite model for the residual peaks.
    - params: Parameters for the composite model.
    """
    model = None
    params = Parameters()

    min_sigma = 1e-6
    min_gamma = 1e-6
    # Estimate widths of residual peaks
    widths, width_heights, left_ips, right_ips = peak_widths(residuals, indices, rel_height=0.5)
    dx = np.diff(xs).mean()  # Average spacing
    fwhms = widths * dx  # Convert width in samples to width in x units

    for idx, fwhm in zip(indices, fwhms):
        pos = xs[idx]
        height = residuals[idx]

        sigma_guess = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        gamma_guess = fwhm / 2  # Lorentzian HWHM

        # Ensure sigma and gamma are within bounds
        sigma_guess = np.clip(sigma_guess, min_sigma, max_sigma)
        gamma_guess = np.clip(gamma_guess, min_gamma, max_gamma)

        # Estimate amplitude using the Voigt amplitude formula
        amplitude_guess = voigt_amplitude_from_height(height, sigma_guess, gamma_guess)
        amplitude_guess = np.clip(amplitude_guess, None, max_amplitude)

        # Compute min_amplitude for this peak
        min_amplitude = voigt_amplitude_from_height(min_height, sigma_guess, gamma_guess)

        prefix = f'iter{iteration}_p{idx}_'

        voigt = VoigtModel(prefix=prefix)
        pars = voigt.make_params()

        pars[prefix + 'amplitude'].set(value=amplitude_guess, min=min_amplitude, max=max_amplitude, vary=True, expr=None)
        pars[prefix + 'center'].set(value=pos, min=pos - center_vary, max=pos + center_vary, vary=True, expr=None)
        pars[prefix + 'sigma'].set(value=sigma_guess, min=min_sigma, max=max_sigma, vary=True, expr=None)
        pars[prefix + 'gamma'].set(value=gamma_guess, min=min_gamma, max=max_gamma, vary=True, expr=None)

        if model is None:
            model = voigt
        else:
            model += voigt
        params.update(pars)

    return model, params
def fit_voigts(
    df,
    title='',
    threshold=0.25,
    min_dist=2,
    min_height=2,
    prominence=None,
    reduce_data_points_factor=None,
    plot=False,
    max_iterations=5,
    center_vary=5,
    fit_method='leastsq',
    verbose=True
):
    if verbose:
        print("Extracting data...")

    # Extract data
    xs_full = df['Wavenumber'].values
    ys_full = df['Intensity'].values

    # Optionally reduce data points to speed up fitting
    if reduce_data_points_factor:
        xs = xs_full[::reduce_data_points_factor]
        ys = ys_full[::reduce_data_points_factor]
    else:
        xs = xs_full
        ys = ys_full

    # Peak detection on the full data
    if verbose:
        print("Detecting peaks in the data...")

    peak_indices, properties = find_peaks(
        ys, height=threshold, distance=min_dist, prominence=prominence)
    num_peaks = len(peak_indices)

    if verbose:
        print(f"Found {num_peaks} peaks in the data.")

    if num_peaks == 0:
        if verbose:
            print("No peaks detected in the data, exiting.")
        return None

    peak_positions = xs[peak_indices]
    peak_heights = ys[peak_indices]

    # Estimate widths using peak_widths
    results_half = peak_widths(ys, peak_indices, rel_height=0.5)
    width_in_samples = results_half[0]
    dx = np.diff(xs).mean()  # Average spacing
    width_in_x = width_in_samples * dx
    fwhms = width_in_x  # Full width at half maximum

    # Estimate sigma and gamma
    sigmas = fwhms / (2 * np.sqrt(2 * np.log(2)))  # Gaussian sigma
    gammas = fwhms / 2  # Lorentzian HWHM

    # Create composite model with initial peaks
    model = None
    params = Parameters()

    min_sigma = 1e-6   # Avoid division by zero
    min_gamma = 1e-6

    max_amplitude = ys.max() * np.pi * (xs.max() - xs.min()) / 2
    max_sigma = (xs.max() - xs.min()) / 10
    max_gamma = (xs.max() - xs.min()) / 10

    for i in range(num_peaks):
        prefix = f'p{i}_'
        voigt = VoigtModel(prefix=prefix)
        pars = voigt.make_params()

        pos = peak_positions[i]
        height = peak_heights[i]
        sigma_guess = sigmas[i] if len(sigmas) > i else (xs.max() - xs.min()) / 50
        gamma_guess = gammas[i] if len(gammas) > i else (xs.max() - xs.min()) / 50

        # Use the Voigt amplitude formula to estimate amplitude
        amplitude_guess = voigt_amplitude_from_height(height, sigma_guess, gamma_guess)
        amplitude_guess = np.clip(amplitude_guess, None, max_amplitude)

        # Compute min_amplitude for this peak
        min_amplitude = voigt_amplitude_from_height(min_height, sigma_guess, gamma_guess)

        pars[prefix + 'amplitude'].set(value=amplitude_guess, min=min_amplitude, max=max_amplitude, vary=True, expr=None)
        pars[prefix + 'center'].set(value=pos, min=pos - center_vary, max=pos + center_vary, vary=True, expr=None)
        pars[prefix + 'sigma'].set(value=sigma_guess, min=min_sigma, max=max_sigma, vary=True, expr=None)
        pars[prefix + 'gamma'].set(value=gamma_guess, min=min_gamma, max=max_gamma, vary=True, expr=None)

        # Build composite model
        if model is None:
            model = voigt
        else:
            model += voigt
        params.update(pars)

    if verbose:
        print(f"Starting initial fitting with method '{fit_method}'...")

    # Initial fitting with specified method
    final_result = model.fit(ys, params, x=xs, method=fit_method)
    if verbose:
        print(final_result.fit_report())

    # Update params with optimized values
    params.update(final_result.params)

    # Remove negligible peaks after initial fitting
    negligible_peaks = []
    for param_name in final_result.params:
        if 'amplitude' in param_name:
            prefix = param_name.split('amplitude')[0]
            amplitude = final_result.params[prefix + 'amplitude'].value
            sigma = final_result.params[prefix + 'sigma'].value
            gamma = final_result.params[prefix + 'gamma'].value
            # Compute peak height
            height = voigt_height(amplitude, sigma, gamma)
            if height < min_height * 1.1:
                negligible_peaks.append(prefix)

    # Remove negligible peaks
    if negligible_peaks:
        if verbose:
            print(f"Removing negligible peaks ({len(negligible_peaks)} peaks) and refitting the model...")

        # Remove negligible peaks from parameters
        for prefix in negligible_peaks:
            for par in list(params.keys()):
                if par.startswith(prefix):
                    del params[par]

        # Rebuild the model excluding negligible peaks
        model = None
        prefixes_added = set()
        for param_name in params:
            prefix = '_'.join(param_name.split('_')[:-1]) + '_'
            if prefix not in prefixes_added:
                voigt = VoigtModel(prefix=prefix)
                if model is None:
                    model = voigt
                else:
                    model += voigt
                prefixes_added.add(prefix)

        # Refit the model after removing negligible peaks
        if verbose:
            print("Refitting the model after removing negligible peaks...")
        final_result = model.fit(ys, params, x=xs, method=fit_method)
        if verbose:
            print(final_result.fit_report())

        # Update params with optimized values
        params = final_result.params.copy()

    # Plotting after initial fitting
    if plot:
        if verbose:
            print("Plotting the results after initial fitting...")
        plot_fit_results(xs, ys, final_result, title, iteration=0, plot_residuals=True)

    iteration = 0
    while iteration < max_iterations:
        # Compute residuals
        residuals = ys - final_result.best_fit

        if verbose:
            print(f"Iteration {iteration+1}: Detecting peaks in residuals...")

        # Find peaks in residuals
        residual_peak_indices, properties = find_peaks(
            residuals, height=threshold, distance=min_dist, prominence=prominence)
        num_residual_peaks = len(residual_peak_indices)

        if verbose:
            print(f"Iteration {iteration+1}: Found {num_residual_peaks} peaks in residuals.")

        if num_residual_peaks == 0:
            if verbose:
                print(f"Iteration {iteration+1}: No significant peaks in residuals, stopping iterations.")
            break

        # Create models for residual peaks
        model_new, params_new = create_residual_models(
            residual_peak_indices,
            xs,
            residuals,
            iteration=iteration+1,
            max_amplitude=max_amplitude,
            max_sigma=max_sigma,
            max_gamma=max_gamma,
            center_vary=center_vary,
            min_height=min_height
        )

        # Add new models to the combined model
        model += model_new
        params.update(params_new)
        # Use the optimized parameters from the previous fit as initial parameters
        params.update(final_result.params)
        # Refit the combined model
        if verbose:
            print(f"Iteration {iteration+1}: Starting fitting with method '{fit_method}'...")
        final_result = model.fit(ys, params, x=xs, method=fit_method)
        if verbose:
            print(final_result.fit_report())

        # Update params with optimized values
        params = final_result.params.copy()

        # Remove negligible peaks after each iteration
        negligible_peaks = []
        for param_name in final_result.params:
            if 'amplitude' in param_name:
                prefix = param_name.split('amplitude')[0]
                amplitude = final_result.params[prefix + 'amplitude'].value
                sigma = final_result.params[prefix + 'sigma'].value
                gamma = final_result.params[prefix + 'gamma'].value
                # Compute peak height
                height = voigt_height(amplitude, sigma, gamma)
                if height < min_height * 1.1:
                    negligible_peaks.append(prefix)

        # Remove negligible peaks
        if negligible_peaks:
            if verbose:
                print(f"Iteration {iteration+1}: Removing negligible peaks ({len(negligible_peaks)} peaks) and refitting the model...")

            # Remove negligible peaks from parameters
            for prefix in negligible_peaks:
                for par in list(params.keys()):
                    if par.startswith(prefix):
                        del params[par]

            # Rebuild the model excluding negligible peaks
            model = None
            prefixes_added = set()
            for param_name in params:
                prefix = '_'.join(param_name.split('_')[:-1]) + '_'
                if prefix not in prefixes_added:
                    voigt = VoigtModel(prefix=prefix)
                    if model is None:
                        model = voigt
                    else:
                        model += voigt
                    prefixes_added.add(prefix)

            # Use the optimized parameters from the previous fit as initial parameters
            params.update(final_result.params)

            # Refit the model after removing negligible peaks
            if verbose:
                print(f"Iteration {iteration+1}: Refitting after removing negligible peaks...")
            final_result = model.fit(ys, params, x=xs, method=fit_method)
            if verbose:
                print(final_result.fit_report())

            # Update params with optimized values
            params.update(final_result.params)

        # Plotting after iteration
        if plot:
            if verbose:
                print(f"Iteration {iteration+1}: Plotting the results...")
            plot_fit_results(xs, ys, final_result, title, iteration=iteration+1,
                            residuals=residuals, residual_peak_indices=residual_peak_indices, plot_residuals=True)

        iteration += 1

    return final_result

def plot_fit_results(xs, ys, result, title, iteration, residuals=None, residual_peak_indices=None, plot_residuals=True):
    """
    Plot the fit results and residuals.

    Parameters:
    - xs: X data.
    - ys: Y data.
    - result: lmfit ModelResult object.
    - title: Title for the plot.
    - iteration: Current iteration number.
    - residuals: Residuals data (optional).
    - residual_peak_indices: Indices of peaks in the residuals (optional).
    - plot_residuals: Boolean indicating whether to plot residuals.
    """
    print(f"Plot for iteration: {iteration}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Plot data and fit
    ax1.plot(xs, ys, lw=1, label='Data', color='black')
    ax1.plot(xs, result.best_fit, label='Fitted Spectrum', color='red', lw=2, ls='--')
    comps = result.eval_components()
    for key in comps:
        ax1.plot(xs, comps[key], ls='-', alpha=0.6)
    ax1.set_ylabel("Intensity/Arbitr. Units")
    ax1.set_title(title)
    ax1.legend()

    # Plot residuals
    if plot_residuals:
        residuals_norm = ys - result.best_fit
        ax2.plot(xs, residuals_norm, label='Residuals', color='blue')
        ax2.axhline(0, color='black', linestyle='--')
        ax2.set_xlabel("Wavenumber (cm⁻¹)")
        ax2.set_ylabel("Residuals")

        # # Highlight peaks in residuals if provided
        # if residual_peak_indices is not None:
        #     ax2.plot(xs[residual_peak_indices], residuals_norm[residual_peak_indices], 'ro', label='Detected Peaks')
        #     ax2.legend()

    plt.tight_layout()
    plt.show()
def voigt_amplitude_from_height(height, sigma, gamma):
    """Calculate the amplitude of a Voigt profile given peak height, sigma, and gamma."""
    return height * sigma * np.sqrt(2 * np.pi) / \
        np.real(wofz((1j * gamma) / (sigma * np.sqrt(2))))
        
from scipy.special import wofz
def voigt_height(amplitude, sigma, gamma):
    """Calculate the peak height of a Voigt profile given amplitude, sigma, and gamma."""
    return (amplitude / (sigma * np.sqrt(2 * np.pi))) * \
           np.real(wofz((1j * gamma) / (sigma * np.sqrt(2))))

def create_residual_models(
    indices,
    xs,
    residuals,
    iteration,
    max_amplitude,
    max_sigma,
    max_gamma,
    center_vary,
    min_height
):
    """
    Create Voigt models for residual peaks.

    Parameters:
    - indices: Indices of peaks in the residuals.
    - xs: Full x data.
    - residuals: Residuals data.
    - iteration: Iteration number.
    - max_amplitude, max_sigma, max_gamma: Bounds for parameters.
    - center_vary: How much to vary peak centers during fitting.
    - min_height: Minimum height for peaks.
    
    Returns:
    - model: Composite model for the residual peaks.
    - params: Parameters for the composite model.
    """
    model = None
    params = Parameters()

    min_sigma = 1e-6
    min_gamma = 1e-6
    # Estimate widths of residual peaks
    widths, width_heights, left_ips, right_ips = peak_widths(residuals, indices, rel_height=0.5)
    dx = np.diff(xs).mean()  # Average spacing
    fwhms = widths * dx  # Convert width in samples to width in x units

    for idx, fwhm in zip(indices, fwhms):
        pos = xs[idx]
        height = residuals[idx]

        sigma_guess = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        gamma_guess = fwhm / 2  # Lorentzian HWHM

        # Ensure sigma and gamma are within bounds
        sigma_guess = np.clip(sigma_guess, min_sigma, max_sigma)
        gamma_guess = np.clip(gamma_guess, min_gamma, max_gamma)

        # Estimate amplitude using the Voigt amplitude formula
        amplitude_guess = voigt_amplitude_from_height(height, sigma_guess, gamma_guess)
        amplitude_guess = np.clip(amplitude_guess, None, max_amplitude)

        # Compute min_amplitude for this peak
        min_amplitude = voigt_amplitude_from_height(min_height, sigma_guess, gamma_guess)

        prefix = f'iter{iteration}_p{idx}_'

        voigt = VoigtModel(prefix=prefix)
        pars = voigt.make_params()

        pars[prefix + 'amplitude'].set(value=amplitude_guess, min=min_amplitude, max=max_amplitude, vary=True, expr=None)
        pars[prefix + 'center'].set(value=pos, min=pos - center_vary, max=pos + center_vary, vary=True, expr=None)
        pars[prefix + 'sigma'].set(value=sigma_guess, min=min_sigma, max=max_sigma, vary=True, expr=None)
        pars[prefix + 'gamma'].set(value=gamma_guess, min=min_gamma, max=max_gamma, vary=True, expr=None)

        if model is None:
            model = voigt
        else:
            model += voigt
        params.update(pars)

    return model, params