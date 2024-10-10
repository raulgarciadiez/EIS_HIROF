from scipy.optimize import curve_fit
import numpy as np

class Model:
    def __init__(self, initial_guess):
        self.params = initial_guess

    def fit(self, x_data, y_data):
        """Fit the model to data. To be implemented in subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, x_data):
        """Predict values based on the fitted parameters."""
        raise NotImplementedError("Subclasses should implement this method.")

class RRCRCModel(Model):
    def rrcrc_func(self, omega, R0, R1, C1, R2, C2):
        """
        R-RC-RC model equation.
        """
        Z_R1C1 = R1 / (1 + 1j * omega * R1 * C1)
        Z_R2C2 = R2 / (1 + 1j * omega * R2 * C2)
        Z = R0 + Z_R1C1 + Z_R2C2
        return np.concatenate((np.real(Z), np.imag(Z)))

    def fit(self, omega_data, Z_data):
        """Fit the RRCRC model to the data."""
        # Initial guess for the parameters [R0, R1, C1, R2, C2]
        def rrcrc_wrapper(omega, R0, R1, C1, R2, C2):
            return self.rrcrc_func(omega, R0, R1, C1, R2, C2)

        # Fit using curve_fit
        self.params, _ = curve_fit(rrcrc_wrapper, omega_data, Z_data, p0=self.params)
    
    def predict(self, omega_data):
        """Predict the impedance based on fitted RRCRC model."""
        R0, R1, C1, R2, C2 = self.params
        return self.rrcrc_func(omega_data, R0, R1, C1, R2, C2)
    


class RRCRCCPEModel(Model):
    def fit(self, x_data, y_data):
        def rrcrc_cpe_func(x, *params):
            return params[0] * x + params[1]  # Replace with actual equation

        self.params, _ = curve_fit(rrcrc_cpe_func, x_data, y_data, p0=self.params)

    def predict(self, x_data):
        return self.params[0] * x_data + self.params[1]
