import pandas as pd
import BioLogic

class DataHandler:
    def __init__(self, file_path):
        self.mpr_file = BioLogic.MPRfile(file_path)
        self.df = pd.DataFrame(self.mpr_file.data)

    def compute_ewe(self):
        """Compute the Ewe parameter."""
        self.ewe = self.df["<Ewe>/V"].mean()
        return self.ewe

    def filter_frequencies(self, fmin=None, fmax=None):
        """Filter data based on a frequency range."""
        self.filtered_df = self.df.copy()
        if fmin:
            self.filtered_df = self.filtered_df[self.filtered_df['frequencies'] >= fmin]
        if fmax:
            self.filtered_df = self.filtered_df[self.filtered_df['frequencies'] <= fmax]
        return self.filtered_df
