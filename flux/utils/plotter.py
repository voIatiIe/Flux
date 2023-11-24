import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Stats:
    def __init__(self, history: pd.DataFrame, target: float) -> None:
        self.history = history
        self.target = target

        self.n_points = history["n_points"].iloc[0]
    
    def average_list(
        self,
        integrals: np.ndarray[float],
        uncertainties: np.ndarray[float],
    ) -> (float, float):
        integral = integrals.mean()
        integral_unc = ((uncertainties ** 2).sum() / uncertainties.shape[0] ** 2) ** 0.5

        return integral, integral_unc

    def get_integral_and_uncertainty(self) -> (float, float):
        history = self.history

        data = history.loc[history["phase"] == "refine"]
        integral = data["integral"].mean()
        uncertainty = ((data["uncertainty"] ** 2).sum() / data['uncertainty'].shape[0] ** 2) ** 0.5

        return integral, uncertainty

    def get_convergence_arrays(self) -> (np.ndarray[float], np.ndarray[float]):
        history = self.history

        data = history.loc[history["phase"] == "refine"]
        data = data.sort_values('rank').sort_index().groupby('rank')

        integrals = data['integral'].agg(list)
        uncertainties = data['uncertainty'].agg(list)

        integrals_, uncertainties_ = [], []
        for int_, unc_ in zip(list(zip(*integrals)), list(zip(*uncertainties))):
            tot_int, tot_unc = self.average_list(np.array(int_), np.array(unc_))

            integrals_.append(tot_int)
            uncertainties_.append(tot_unc)

        return np.array(integrals_), np.array(uncertainties_)


    def plot_convergence(self) -> (np.ndarray[float], np.ndarray[float], np.ndarray[float]):
        integrals_, uncertainties_ = self.get_convergence_arrays()

        plot_integrals, plot_uncertainties = [], []
        for i in range(1, len(integrals_)+1):
            integral, uncertainty = self.average_list(integrals_[:i], uncertainties_[:i])

            plot_integrals.append(integral)
            plot_uncertainties.append(uncertainty)

        x_ = self.n_points * (np.arange(len(plot_integrals)) + 1)
        divergence = np.abs(np.array(plot_integrals) - self.target)
        plt.plot(x_, divergence)
        plt.grid()
        plt.show()

        plt.plot(x_, plot_uncertainties)
        plt.grid()
        plt.show()

        return divergence, plot_uncertainties, x_

    def plot_deviation_uncertainty_scatter(self) -> None:
        integrals_, uncertainties_ = self.get_convergence_arrays()

        plt.scatter(np.abs(integrals_ - self.target), uncertainties_, s=0.5, c='red')
        plt.grid()
        plt.show()

    @staticmethod
    def show_rows(df, nrows=10000):
        with pd.option_context("display.max_rows", nrows):
            print(df)
