import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from galvani import BioLogic
import os
import csv


def generate_random_initial_guess(initial_guess, perturbation=0.1):
    """
    Generate random initial guesses based on the provided initial guess.
    
    Parameters:
    - initial_guess: A list or array of initial guesses.
    - perturbation: Fractional perturbation applied to the initial guess values.
    
    Returns:
    - new_initial_guess: A list of new initial guesses.
    """
    new_initial_guess = []

    for guess in initial_guess:
        # Calculate the perturbation range
        perturb_range = perturbation * abs(guess) if guess != 0 else perturbation

        # Generate a new guess within the perturbation range
        new_guess = guess + np.random.uniform(-perturb_range, perturb_range)

        # Ensure the new guess is non-negative
        if new_guess < 0:
            new_guess = 0.1  # Set a minimum value for non-negativity

        new_initial_guess.append(float(new_guess))  # Convert to standard float

    return new_initial_guess

def generate_random_bounds(bounds, perturbation=0.1, min_distance=0.1):
    """
    Generate random bounds for the parameters within the specified perturbation range.

    Parameters:
    - bounds: A tuple containing the lower and upper bounds as lists.
    - perturbation: A fraction of the range to perturb the bounds by.
    - min_distance: Minimum distance between new_lb and new_ub.

    Returns:
    - new_bounds: A tuple of the new lower and upper bounds as lists.
    """
    lower_bounds, upper_bounds = bounds
    new_lower_bounds = []
    new_upper_bounds = []

    for lb, ub in zip(lower_bounds, upper_bounds):
        valid_bounds = False
        
        while not valid_bounds:
            # If the parameter is in logarithmic scale, perturb accordingly
            if lb > 0 and ub > 0:
                log_lb = np.log10(lb)
                log_ub = np.log10(ub)
                log_range = log_ub - log_lb

                # Calculate maximum perturbation size in logarithmic scale
                max_perturbation = perturbation * log_range

                # Generate new lower and upper bounds in logarithmic scale
                new_log_lb = log_lb + np.random.uniform(-max_perturbation, max_perturbation)
                new_log_ub = log_ub + np.random.uniform(-max_perturbation, max_perturbation)

                # Convert back to linear scale
                new_lb = 10 ** new_log_lb
                new_ub = 10 ** new_log_ub
            else:
                # Non-logarithmic parameters
                range_value = ub - lb
                max_perturbation = perturbation * range_value
                new_lb = lb + np.random.uniform(-max_perturbation, max_perturbation)
                new_ub = ub + np.random.uniform(-max_perturbation, max_perturbation)

            # Ensure new bounds maintain a minimum distance
            if new_ub <= new_lb + min_distance:
                continue

            # Ensure new lower bounds are greater than a minimum threshold (e.g., 0.1)
            if new_lb < 0.1:
                new_lb = 0.1

            # Ensure new bounds are non-negative and valid
            if new_lb < new_ub:
                valid_bounds = True  # Valid bounds found

        # Append valid bounds as standard Python floats
        new_lower_bounds.append(float(new_lb))
        new_upper_bounds.append(float(new_ub))

    return (new_lower_bounds, new_upper_bounds)


def write_fit_results_to_file(model, root_folder, Ewe, fmin, fmax, initial_guess, params, bounds, residual):
    """Writes the fitting results to a CSV file."""
    log_filename = f"{os.path.basename(root_folder)}_{model.__class__.__name__}_fit_log.csv"
    filepath = os.path.join(os.getcwd(), log_filename)

    # Check if file exists; if not, write the header
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file doesn't exist
        if not file_exists:
            # Headers for initial guesses
            header = ['Ewe', 'fmin', 'fmax'] + [f"Initial_{param}" for param in model.param_names]
            
            # Headers for bounds (min and max for each parameter)
            for param in model.param_names:
                header.append(f"{param}_bound_min")
            for param in model.param_names:
                header.append(f"{param}_bound_max")

            # Headers for fitted parameters and residual
            header += [f"Fit_{param}" for param in model.param_names] + ['Residual']

            # Write the header row
            writer.writerow(header)

        # Write the row of fitting results
        row = [Ewe, fmin, fmax] + list(initial_guess)  # Start with Ewe, fmin, fmax, and initial guess
        row += bounds[0] + bounds[1]  # Add bounds min and max
        row += list(params)  + [residual]  # Add fitted parameters and residual

        # Write the data row
        writer.writerow(row)

def find_files(root_folder, extension):
    file_list = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

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
 
class RRCRCCPEModel(BaseModel):
    def __init__(self, initial_guess=None, bounds=None):
        super().__init__(initial_guess, bounds)
        # Define parameter names specific to this model
        self.param_names = ['R0', 'R1', 'fs1', "n1", 'R2', 'fs2', "n2"]

    def impedance(self,omega, *params):#R0, R1, fs1, n1, R2, fs2, n2):
        """
        R-RC-RC model with CPE equation.
        """
        R0, R1, fs1, n1, R2, fs2, n2 = params
        Z_R1CPE1 = R1 / (1 + (1j * omega / fs1)**n1)
        Z_R2CPE2 = R2 / (1 + (1j * omega / fs2)**n2)
        Z = R0 + Z_R1CPE1 + Z_R2CPE2
        return np.concatenate((np.real(Z), np.imag(Z)))
            
    #def impedance(self,omega, R0, R1, fs1, n1, R2, fs2, n2):
    #    """
    #    R-RC-RC model with CPE equation.
    #    """
    #    Z_R1CPE1 = R1 / (1 + (1j * omega / fs1)**n1)
    #    Z_R2CPE2 = R2 / (1 + (1j * omega / fs2)**n2)
    #    Z = R0 + Z_R1CPE1 + Z_R2CPE2
    #    return np.concatenate((np.real(Z), np.imag(Z)))
    
class RCRCModel(BaseModel):
    def impedance(self, omega, R0, R1, C1, R2):
        jomega = 1j * omega
        Z_C1 = 1 / (jomega * C1)
        Z_R1C1 = R1 + Z_C1
        Z_R2 = R2
        Z_total = R0 + 1 / (1 / Z_R1C1 + 1 / Z_R2)
        return np.concatenate((np.real(Z_total), np.imag(Z_total)))

import numpy as np
from scipy.optimize import curve_fit

# BaseModel, FitManager, DataHandler remain the same

# Adding a class to compute fit quality
class FitQuality:
    
    @staticmethod
    def compute_residuals(actual, predicted):
        """Compute the residuals between actual data and model predictions."""
        return actual - predicted
    
    @staticmethod
    def mean_squared_error(actual, predicted):
        """Calculate Mean Squared Error (MSE)."""
        residuals = FitQuality.compute_residuals(actual, predicted)
        mse = np.mean(residuals ** 2)
        return mse
    
    @staticmethod
    def root_mean_squared_error(actual, predicted):
        """Calculate Root Mean Squared Error (RMSE)."""
        mse = FitQuality.mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        return rmse
    
    @staticmethod
    def r_squared(actual, predicted):
        """Calculate the R-squared value."""
        residuals = FitQuality.compute_residuals(actual, predicted)
        ss_res = np.sum(residuals ** 2)  # Residual sum of squares
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot)
        return r2

    @staticmethod
    def adjusted_r_squared(actual, predicted):
        """
        Calculate the Adjusted R-squared value.

        :param actual: The actual data points (observations).
        :param predicted: The predicted data points from the model.
        :param n_params: The number of fitted parameters in the model.
        :return: Adjusted R-squared value.
        """
        n = len(actual)  # Number of data points
        residuals = FitQuality.compute_residuals(actual, predicted)
        ss_res = np.sum(residuals ** 2)  # Residual sum of squares
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
        
        r2 = 1 - (ss_res / ss_tot)  # Standard R-squared
        
        # Adjusted R-squared
        r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - 1)
        
        return r2_adj
    
    @staticmethod
    def evaluate_fit(actual, predicted):
        """Return a dictionary of fit quality metrics."""
        return {
            "MSE": FitQuality.mean_squared_error(actual, predicted),
            "RMSE": FitQuality.root_mean_squared_error(actual, predicted),
            "R-squared": FitQuality.adjusted_r_squared(actual, predicted),
        }
    

    @staticmethod
    def check_boundaries_hit(params, bounds, tolerance=0.01):
        """
        Check if any parameters are near the boundaries.
        - `params`: The fitted parameters.
        - `bounds`: A tuple of (lower_bounds, upper_bounds).
        - `tolerance`: Percentage (e.g., 0.01 = 1%) within which a parameter is considered close to the boundary.
        Returns a list of booleans indicating whether each parameter is near the boundary.
        """
        lower_bounds, upper_bounds = bounds
        hit_status = []
        for param, lower, upper in zip(params, lower_bounds, upper_bounds):
            if abs(param - lower) / (upper - lower) < tolerance:
                hit_status.append(True)
            elif abs(param - upper) / (upper - lower) < tolerance:
                hit_status.append(True)
            else:
                hit_status.append(False)
        return hit_status


# FitManager class to handle the fitting process
class FitManager:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.previous_fitted_params = None  # Store the fitted parameters from the previous dataset

        self.Ewe = self.data_handler.Ewe
        self.root_folder=self.data_handler.root_folder

    def fit_model(self, model, fmin=None, fmax=None, initial_guess=None, bounds=None):
        """Fit the model to a data set."""
        filtered_df = self.data_handler.filter_frequencies(fmin, fmax)
        omega, Z_data = self.data_handler.prepare_data(filtered_df)

        def model_wrapper(omega, *params):
            return model.impedance(omega, *params)

        if initial_guess is None:
            if self.previous_fitted_params is not None:
                initial_guess = self.previous_fitted_params  # Use previous fit parameters
            else:
                initial_guess = model.get_initial_guess()

        if bounds is None:
            bounds = model.get_bounds()

        popt, pcov = curve_fit(model_wrapper, 
                               omega, 
                               Z_data,
                               p0=initial_guess, 
                               bounds=bounds, 
                               maxfev=1e6)
        model.params = popt
        self.previous_fitted_params = popt  # Store for next iteration

        # Calculate fit quality using R-squared
        fitted_Z_data = model_wrapper(omega, *popt)
        fit_quality = FitQuality.adjusted_r_squared(Z_data, fitted_Z_data)
        
        # Call the function to write the results to the file
        write_fit_results_to_file(
            model=model,  # Use the model name dynamically
            root_folder=self.root_folder,
            Ewe=self.Ewe,
            fmin=fmin,
            fmax=fmax,
            initial_guess=initial_guess,
            params=model.params,
            bounds=bounds,
            residual=fit_quality
        )
        #hit_boundaries = FitQuality.check_boundaries_hit(popt, bounds)
        #if any(hit_boundaries):
        #    print("Warning: Some parameters are near the bounds!")
        #    print("Parameters hitting bounds:", np.array(popt)[hit_boundaries])

        return model, pcov, fit_quality
    
    def fit_multiple_files(self, model, data_handlers, fmin, fmax):
        """Fit multiple .mpr files, using the previous file's fit parameters for the next one."""
        for i, data_handler in enumerate(data_handlers):
            self.data_handler = data_handler
            print(f"Fitting file {i+1}/{len(data_handlers)}...")
            model2, pcov, fit_quality = self.fit_model(model, fmin, fmax)
            print(f"Fit quality (R²) for file {i+1}: {fit_quality['R_squared']}")
        return model2
    
    def reset_previous_parameters(self):
        """Reset the stored fitted parameters (useful if you want to start fresh)."""
        self.previous_fitted_params = None
        #print("Previous fitted parameters have been reset.")

# DataHandler class to manage data import, filtering, and transformation
class DataHandler:
    def __init__(self, filepath):
        # Load data from file
        self.filepath = filepath
        self.root_folder = self.extract_root_folder()
        mpr_file = BioLogic.MPRfile(self.filepath)
        self.df = pd.DataFrame(mpr_file.data)
        self.Ewe = self.df["<Ewe>/V"].mean()

    def extract_root_folder(self):
        # Get the directory of the file
        directory = os.path.dirname(self.filepath)
        # Extract the first subfolder from the full path
        root_folder = os.path.split(directory)[0]  # This extracts the top-level directory
        return root_folder

    # Method to filter frequencies based on a range
    def filter_frequencies(self, fmin=4, fmax=1e6):
        """Filter data based on a frequency range."""
        self.filtered_df = self.df.copy()
        self.filtered_df = self.filtered_df[(self.filtered_df['freq/Hz'] >= fmin) & (self.filtered_df['freq/Hz'] <= fmax)& (self.filtered_df['-Im(Z)/Ohm'] > 0)]
        return self.filtered_df

    # Prepare the data for fitting (frequencies and impedance)
    def prepare_data(self, filtered_df):
        frequencies = filtered_df['freq/Hz'].values
        omega = 2 * np.pi * frequencies
        real = filtered_df['Re(Z)/Ohm'].values
        imaginary = -filtered_df['-Im(Z)/Ohm'].values  # Correct sign
        Z_data = np.concatenate((real, imaginary))
        return omega, Z_data
