import numpy as np
from scipy.optimize import curve_fit

# Base model class with flexible parameters
class BaseModel:
    def __init__(self, initial_guess=None, bounds=None):
        self.params = initial_guess
        self.bounds = bounds
    
    # This method must be implemented in each derived model
    def impedance(self, omega, *params):
        raise NotImplementedError("Each model must implement the impedance function!")
    
    # Method to return initial guess dynamically
    def get_initial_guess(self):
        return self.params
    
    # Method to return bounds dynamically
    def get_bounds(self):
        return self.bounds

# RRCRC Model class
class RRCRCModel(BaseModel):
    def impedance(self, omega, R0, R1, C1, R2, C2):
        jomega = 1j * omega
        Z_C1 = 1 / (jomega * C1)
        Z_R1C1 = R1 + Z_C1
        Z_C2 = 1 / (jomega * C2)
        Z_R2C2 = R2 + Z_C2
        Z_total = R0 + 1 / (1 / Z_R1C1 + 1 / Z_R2C2)
        return np.concatenate((np.real(Z_total), np.imag(Z_total)))
 
class RRCRCCPEModel(Model):
    def impedance(self,omega, R0, R1, fs1, n1, R2, fs2, n2):
        """
        R-RC-RC model with CPE equation.
        """
        Z_R1CPE1 = R1 / (1 + (1j * omega / fs1)**n1)
        Z_R2CPE2 = R2 / (1 + (1j * omega / fs2)**n2)
        Z = R0 + Z_R1CPE1 + Z_R2CPE2
        return np.concatenate((np.real(Z), np.imag(Z)))
    
class RCRCModel(BaseModel):
    def impedance(self, omega, R0, R1, C1, R2):
        jomega = 1j * omega
        Z_C1 = 1 / (jomega * C1)
        Z_R1C1 = R1 + Z_C1
        Z_R2 = R2
        Z_total = R0 + 1 / (1 / Z_R1C1 + 1 / Z_R2)
        return np.concatenate((np.real(Z_total), np.imag(Z_total)))

# FitManager class to handle the fitting process
class FitManager:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    # Generic method to fit a model
    def fit_model(self, model, fmin=None, fmax=None, initial_guess=None, bounds=None):
        # Filter the data within the specified frequency range
        filtered_df = self.data_handler.filter_frequencies(fmin, fmax)
        omega, Z_data = self.data_handler.prepare_data(filtered_df)

        # Wrapper to use in curve_fit for handling varying number of parameters
        def model_wrapper(omega, *params):
            return model.impedance(omega, *params)

        # Use initial guess and bounds from the model if not provided
        if initial_guess is None:
            initial_guess = model.get_initial_guess()
        if bounds is None:
            bounds = model.get_bounds()

        # Perform curve fitting
        popt, pcov = curve_fit(
            model_wrapper,
            omega,
            Z_data,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )

        # Update the model with the optimized parameters
        model.params = popt
        return model, pcov

# DataHandler class to manage data import, filtering, and transformation
class DataHandler:
    def __init__(self, filepath):
        # Load data from file
        mpr_file = BioLogic.MPRfile(filepath)
        self.df = pd.DataFrame(mpr_file.data)

    # Method to filter frequencies based on a range
    def filter_frequencies(self, fmin=None, fmax=None):
        filtered_df = self.df.copy()
        if fmin is not None:
            filtered_df = filtered_df[filtered_df['freq/Hz'] >= fmin]
        if fmax is not None:
            filtered_df = filtered_df[filtered_df['freq/Hz'] <= fmax]
        return filtered_df

    # Prepare the data for fitting (frequencies and impedance)
    def prepare_data(self, filtered_df):
        frequencies = filtered_df['freq/Hz'].values
        omega = 2 * np.pi * frequencies
        real = filtered_df['Re(Z)/Ohm'].values
        imaginary = -filtered_df['-Im(Z)/Ohm'].values  # Correct sign
        Z_data = np.concatenate((real, imaginary))
        return omega, Z_data
