from scipy.optimize import curve_fit
import numpy as np

class Model:
    def __init__(self, initial_guess):
        self.params = initial_guess


class RRCRCModel(Model):
    def func(self, omega, R0, R1, C1, R2, C2):
        """
        R-RC-RC model equation.
        """
        Z_R1C1 = R1 / (1 + 1j * omega * R1 * C1)
        Z_R2C2 = R2 / (1 + 1j * omega * R2 * C2)
        Z = R0 + Z_R1C1 + Z_R2C2
        return np.concatenate((np.real(Z), np.imag(Z)))

    


class RRCRCCPEModel(Model):
    def fit(self, x_data, y_data):
        def rrcrc_cpe_func(x, *params):
            return params[0] * x + params[1]  # Replace with actual equation

        self.params, _ = curve_fit(rrcrc_cpe_func, x_data, y_data, p0=self.params)

    def predict(self, x_data):
        return self.params[0] * x_data + self.params[1]
