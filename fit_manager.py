class FitManager:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    def fit_model(self, model, fmin=None, fmax=None):
        """Fit a model to a filtered range of frequencies."""
        filtered_df = self.data_handler.filter_frequencies(fmin, fmax)

        x_data = filtered_df['frequencies']
        y_data = filtered_df['<Ewe>/V']  # Replace with correct column

        model.fit(x_data, y_data)
        return model

    def fit_with_initial_params(self, model, initial_guess, fmin=None, fmax=None):
        model.params = initial_guess
        return self.fit_model(model, fmin, fmax)
