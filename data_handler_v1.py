import pandas as pd
from galvani import BioLogic
import numpy as np

class DataHandler:
    def __init__(self, file_path):
        self.mpr_file = BioLogic.MPRfile(file_path)
        self.df = pd.DataFrame(self.mpr_file.data)

    def compute_ewe(self):
        """Compute the Ewe parameter."""
        self.Ewe = self.df["<Ewe>/V"].mean()
        return self.Ewe

    def filter_frequencies(self, fmin=4, fmax=1e6):
        """Filter data based on a frequency range."""
        self.filtered_df = self.df.copy()
        self.filtered_df = self.filtered_df[(self.filtered_df['freq/Hz'] >= fmin) & (self.filtered_df['freq/Hz'] <= fmax)& (self.filtered_df['-Im(Z)/Ohm'] > 0)]
        return self.filtered_df