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
    def func(self,omega, R0, R1, fs1, n1, R2, fs2, n2):
        """
        R-RC-RC model with CPE equation.
        """
        Z_R1CPE1 = R1 / (1 + (1j * omega / fs1)**n1)
        Z_R2CPE2 = R2 / (1 + (1j * omega / fs2)**n2)
        Z = R0 + Z_R1CPE1 + Z_R2CPE2
        return np.concatenate((np.real(Z), np.imag(Z)))
