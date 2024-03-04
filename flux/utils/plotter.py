import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from flux.models.flows import BaseFlow
from flux.models.integrators import BaseIntegrator
from flux.models.integrators.integrators import DefaultIntegrator


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


    def plot_convergence(self, plot: bool=True) -> (np.ndarray[float], np.ndarray[float], np.ndarray[float]):
        integrals_, uncertainties_ = self.get_convergence_arrays()

        plot_integrals, plot_uncertainties = [], []
        for i in range(1, len(integrals_)+1):
            integral, uncertainty = self.average_list(integrals_[:i], uncertainties_[:i])

            plot_integrals.append(integral)
            plot_uncertainties.append(uncertainty)

        x_ = self.n_points * (np.arange(len(plot_integrals)) + 1)
        divergence = np.abs(np.array(plot_integrals) - self.target)

        if plot:
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

    @classmethod
    def plot_learned_distribution(
        cls,
        integrator: DefaultIntegrator,
        n_points: int = 100000,
        bins: int = 50,
    ):
        assert integrator.trainer.prior.dim == 2

        sample = integrator.trainer.sample(n_points)

        x_dim = np.linspace(0, 1 - 1.0 / bins, bins) + 1.0 / bins
        y_dim = np.linspace(0, 1 - 1.0 / bins, bins) + 1.0 / bins

        X, Y = np.meshgrid(x_dim, y_dim)

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        tensor = torch.stack([X, Y], dim=1)
        res = integrator.integrand.callable(tensor) / integrator.integrand.target
        res /= bins*bins

        # fig, axs = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={'width_ratios': [1, 1.25,]})
        fig, axs = plt.subplots(2, 1, figsize=(5,9))

        hist, _, _ = np.histogram2d(sample[:, 0], sample[:, 1], bins=bins, range=[[0, 1], [0, 1]], density=True)
        hist /= bins*bins

        cmax = max(hist.max(), res.max())
        cmin = min(hist.min(), res.min())
        clim = (cmin, cmax)

        axs[0].imshow(hist, extent=[0, 1, 0, 1], origin='lower', clim=clim, cmap='viridis')
        axs[0].set_title('Learned distribution')

        axs[1].imshow(res, extent=[0, 1, 0, 1], origin='lower', clim=clim, cmap='viridis')
        axs[1].set_title('Target distribution')

        fig.colorbar(axs[1].imshow(res, extent=[0, 1, 0, 1], origin='lower', cmap='viridis'), ax=axs)

        plt.show()
