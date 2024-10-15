from scipy.optimize import curve_fit
import numpy as np

class FitManager:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    def fit_model(self, model, fmin=None, fmax=None, bounds=(0, np.inf)):
        """
        Fit a model to a filtered range of frequencies and impedance data.
        - model: the instance of the model to be fitted
        - fmin, fmax: frequency range to filter
        - bounds: optional bounds for curve fitting
        """
        # Filter the data based on the given frequency range
        filtered_df = self.data_handler.filter_frequencies(fmin, fmax)

        # Get the omega (frequencies) and Z (impedance) data
        omega_data = 2 * np.pi *filtered_df['freq/Hz'].values

        real = filtered_df['Re(Z)/Ohm'].values
        imaginary = -filtered_df['-Im(Z)/Ohm'].values
        Z_data = np.concatenate((real, imaginary))

        # Define the model function for curve fitting
        #def model_wrapper(omega, *params):
        #    return model.func(omega, *params)

        # Perform curve fitting using scipy's curve_fit
        popt, pcov = curve_fit(
            model.func,
            omega_data,
            Z_data,
            p0=model.params,  # Initial guess
            bounds=bounds  # Bounds if provided
        )

        # Update model parameters with optimized values
        model.params = popt

        # Return the fitted model and covariance
        return model, pcov
