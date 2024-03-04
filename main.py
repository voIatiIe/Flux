import torch

from datetime import datetime
import matplotlib.pyplot as plt

from flux.models.integrators.multprocessing import integrate as integrate_mp
from flux.utils.integrands import GaussIntegrand, LongIntegrand, ShortIntegrand, CamelIntegrand, SinusoideIntegrand, HalfSinusoideIntegrand, CircteIntegrand
from flux.utils.plotter import Stats

from dataclasses import dataclass
from flux.utils.constants import CellType, MaskingType
from flux.models.integrands.base import BaseIntegrand
from flux.models.integrators import BaseIntegrator, UniformSurveyIntegrator, SobolSurveyIntegrator, VegasIntegrator
from flux.models.integrators.base import UniformIntegrator


@dataclass
class IntegrationConfig:
    integrand: BaseIntegrand
    n_points: int
    n_survey_steps: int
    n_refine_steps: int
    average_grads: bool = False
    masking_type: MaskingType = MaskingType.CHECKERBOARD
    cell_type: CellType = CellType.PWLINEAR
    n_cells: int | None = None
    integrator_class: BaseIntegrator = UniformSurveyIntegrator


def main_sequent(n_refine_steps, n_survey_steps, dim=8, sobol_sample_prior = False):

    from flux.models.flows import RepeatedCouplingCellFlow
    from flux.models.samplers.samplers import UniformSampler, SobolSampler
    from flux.models.trainers import VarianceTrainer

    flow = RepeatedCouplingCellFlow(
        dim=dim,
        cell=CellType.PWQUADRATIC,
        masking=MaskingType.CHECKERBOARD,
        cell_parameters=dict(
            n_bins=16,
            n_hidden=3,
            dim_hidden=32,
        )
    )

    trainer = VarianceTrainer(
        flow=flow,
        prior=UniformSampler(dim=dim),
        sample_prior=SobolSampler(dim=dim) if sobol_sample_prior else None,
    )

    integrand = CamelIntegrand(dim=dim)

    integrator = UniformSurveyIntegrator(
        integrand=integrand,
        trainer=trainer,
        n_points=20000,
    )

    result = integrator.integrate(n_refine_steps=n_refine_steps, n_survey_steps=n_survey_steps)

    history = result.history
    history = history.loc[history["phase"] == "refine"]

    print('FLUX, unweighted')
    print(f'n_survey_steps: {n_survey_steps}')

    integrals, uncertainties, deviations = [], [], []
    for i in range(len(history)):
        integral = history['integral'][:i+1].mean()
        integral_unc = ((history["uncertainty"][:i+1] ** 2).sum() / history["uncertainty"][:i+1].count() ** 2) ** 0.5

        integrals.append(round(integral, 8))
        uncertainties.append(round(integral_unc, 8))
        deviations.append(round(abs(integral - integrand.target), 8))

        # print(f'n_refine_steps: {i+1}')
        # print(f'Result: {integral:.7e} +- {integral_unc:.7e}')
        # print(f'Target: {integrand.target:.7e}, deviation: {abs(integral - integrand.target):.7e}')
        # print('-'*20)

    print(f'Integrals: {integrals}')
    print(f'Uncertainties: {uncertainties}')
    print(f'Deviations: {deviations}')

    print(f'Target: {integrand.target:.7e}')
    print(f'Survey time: {result.survey_time}, refine time: {result.refine_time}')
    print('='*20)

    # Stats.plot_learned_distribution(integrator, n_points=300000, bins=30)


def test_0(config: IntegrationConfig, compare_uniform: bool = True, compare_vegas: bool = False):

    divergences, uncertainties = {}, {}
    for n_survey_steps in (5, 10, 20, 40):
        config.integrand.reset()

        t0 = datetime.now()
        result, timings = integrate_mp(
            config.integrand,
            n_points=config.n_points,
            average_grads=config.average_grads,
            n_survey_steps=n_survey_steps,
            n_refine_steps=config.n_refine_steps,
            masking_type=config.masking_type,
            cell_type=config.cell_type,
            n_cells=config.n_cells,
            integrator_class=config.integrator_class,
        )
        print(f'nSurveySteps: {n_survey_steps} - time: {datetime.now() - t0}')
        print(f'Survey time: {timings[0]}, refine time: {timings[1]}')

        stats = Stats(result, target=config.integrand.target)
        integral, integral_unc = stats.get_integral_and_uncertainty()

        print(f"Final result: {integral:.5e} +/- {integral_unc:.5e}")

        divergences[n_survey_steps], uncertainties[n_survey_steps], x_ = stats.plot_convergence(plot=False)

    if compare_uniform:
        config.integrand.reset()

        t0 = datetime.now()
        result, timings = integrate_mp(
            config.integrand,
            n_points=config.n_points,
            average_grads=config.average_grads,
            n_survey_steps=n_survey_steps,
            n_refine_steps=config.n_refine_steps,
            masking_type=config.masking_type,
            cell_type=config.cell_type,
            n_cells=config.n_cells,
            integrator_class=UniformIntegrator,
        )

        print(f'Uniform integrator - time: {datetime.now() - t0}')
        print(f'Survey time: {timings[0]}, refine time: {timings[1]}')

        stats = Stats(result, target=config.integrand.target)
        integral, integral_unc = stats.get_integral_and_uncertainty()

        print(f"Final result: {integral:.5e} +/- {integral_unc:.5e}")

        divergences_, uncertainties_, x_ = stats.plot_convergence(plot=False)

    if compare_uniform: plt.plot(x_, divergences_, label=f'Uniform')

    for n_survey_steps, data in divergences.items():
        plt.plot(x_, data, label=f'{n_survey_steps} steps')

    plt.legend()
    plt.title('Divergence')
    plt.grid()
    plt.show()

    if compare_uniform: plt.plot(x_, uncertainties_, label=f'Uniform')

    for n_survey_steps, data in uncertainties.items():
        plt.plot(x_, data, label=f'{n_survey_steps} steps')

    plt.legend()
    plt.title('Uncertainty')
    plt.grid()
    plt.show()


def test1(config: IntegrationConfig):
    t0 = datetime.now()
    result, timings = integrate_mp(
        config.integrand,
        n_points=config.n_points,
        average_grads=config.average_grads,
        n_survey_steps=config.n_survey_steps,
        n_refine_steps=config.n_refine_steps,
        masking_type=config.masking_type,
        cell_type=config.cell_type,
        n_cells=config.n_cells,
        integrator_class=config.integrator_class,
    )
    print(f'nSurveySteps: {config.n_survey_steps} - time: {datetime.now() - t0}')
    print(f'Survey time: {timings[0]}, refine time: {timings[1]}')

    stats = Stats(result, target=config.integrand.target)
    integral, integral_unc = stats.get_integral_and_uncertainty()

    print(f"Final result: {integral:.5e} +/- {integral_unc:.5e}")

    divergences, uncertainties, x_ = stats.plot_convergence(plot=False)

    plt.plot(x_, divergences)

    plt.legend()
    plt.title('Divergence')
    plt.grid()
    plt.show()

    plt.plot(x_, uncertainties)

    plt.legend()
    plt.title('Uncertainty')
    plt.grid()
    plt.show()


def vegas_test(n_refine_steps, n_survey_steps, dim=8):
    integrand = CamelIntegrand(dim=dim)
    integrator = VegasIntegrator(dim)

    t0 = datetime.now()
    integrator.train(
        integrand,
        n_train_steps=n_survey_steps,
        n_train_samples=20000,
    )
    survey_time = datetime.now() - t0

    t0 = datetime.now()
    history = integrator.refine(
        integrand,
        n_refine_steps=n_refine_steps,
        n_refine_samples=20000,
    )
    refine_time = datetime.now() - t0
    # history['rank'] = 0

    # stats = Stats(history, target=integrand.target)
    # integral, integral_unc = stats.get_integral_and_uncertainty()

    # print(f"Final result: {integral:.5e} +/- {integral_unc:.5e}")
    # print(f'Target: {integrand.target:.5e}')

    # divergences, uncertainties, x_ = stats.plot_convergence(plot=True)

    history = history.loc[history["phase"] == "refine"]
    print('VEGAS, unweighted')
    print(f'n_survey_steps: {n_survey_steps}')

    integrals, uncertainties, deviations = [], [], []
    for i in range(len(history)):
        integral = history['integral'][:i+1].mean()
        integral_unc = ((history["uncertainty"][:i+1] ** 2).sum() / history["uncertainty"][:i+1].count() ** 2) ** 0.5

        integrals.append(round(integral, 8))
        uncertainties.append(round(integral_unc, 8))
        deviations.append(round(abs(integral - integrand.target), 8))

        # print(f'n_refine_steps: {i+1}')
        # print(f'Result: {integral:.7e} +- {integral_unc:.7e}')
        # print(f'Target: {integrand.target:.7e}, deviation: {abs(integral - integrand.target):.7e}')
        # print('-'*20)

    print(f'Integrals: {integrals}')
    print(f'Uncertainties: {uncertainties}')
    print(f'Deviations: {deviations}')

    print(f'Target: {integrand.target:.7e}')
    print(f'Survey time: {survey_time}, refine time: {refine_time}')
    print('='*20)







    # integral = history["integral"].mean()
    # integral_unc = ((history["uncertainty"] ** 2).sum() / history["uncertainty"].count() ** 2) ** 0.5

    # print('VEGAS, unweighted')
    # print(f'n_refine_steps: {n_refine_steps}, n_survey_steps: {n_survey_steps}')
    # print(f'Result: {integral:.7e} +- {integral_unc:.7e}')
    # print(f'Target: {integrand.target:.7e}, deviation: {abs(integral - integrand.target):.7e}')
    # print(f'Survey time: {survey_time}, refine time: {refine_time}')
    # print('-'*20)



def integrate_vegas(
    integrand: BaseIntegrand,
    n_train_steps: int,
    n_train_samples: int,
    n_refine_steps: int,
    n_refine_samples: int,
):
    integrator = VegasIntegrator(integrand.dim)

    t0 = datetime.now()
    integrator.train(
        integrand,
        n_train_steps=n_train_steps,
        n_train_samples=n_train_samples,
    )
    survey_time = datetime.now() - t0


    t0 = datetime.now()
    history = integrator.refine(
        integrand,
        n_refine_steps=n_refine_steps,
        n_refine_samples=n_refine_samples,
    )
    refine_time = datetime.now() - t0

    history['rank'] = 0

    return history, (survey_time, refine_time)


def test_2(config: IntegrationConfig, compare_uniform: bool = True, compare_vegas: bool = True):

    divergences, uncertainties = {}, {}
    for n_survey_steps in (5, 10, 20):
        print(f'FLUX nSurveySteps: {n_survey_steps}')
    # for n_survey_steps in (5,):
        config.integrand.reset()

        t0 = datetime.now()
        result, timings = integrate_mp(
            config.integrand,
            n_points=config.n_points,
            average_grads=config.average_grads,
            n_survey_steps=n_survey_steps,
            n_refine_steps=config.n_refine_steps,
            masking_type=config.masking_type,
            cell_type=config.cell_type,
            n_cells=config.n_cells,
            integrator_class=config.integrator_class,
            world_size=1,
        )
        print(f'nSurveySteps: {n_survey_steps} - time: {datetime.now() - t0}')
        print(f'Survey time: {timings[0]}, refine time: {timings[1]}')

        stats = Stats(result, target=config.integrand.target)
        integral, integral_unc = stats.get_integral_and_uncertainty()

        print(f"Final result: {integral:.5e} +/- {integral_unc:.5e}")

        divergences[n_survey_steps], uncertainties[n_survey_steps], x_ = stats.plot_convergence(plot=False)

    if compare_vegas:
        divergences_vegas, uncertainties_vegas = {}, {}
        for n_survey_steps in (5, 10, 20):
            print(f'VEGAS nSurveySteps: {n_survey_steps}')
        # for n_survey_steps in (5,):
            config.integrand.reset()
            t0 = datetime.now()

            result, timings = integrate_vegas(
                config.integrand,
                n_train_steps=n_survey_steps,
                n_train_samples=config.n_points,
                n_refine_steps=config.n_refine_steps,
                n_refine_samples=config.n_points,
            )
            print(f'nSurveySteps: {n_survey_steps} - time: {datetime.now() - t0}')
            print(f'Survey time: {timings[0]}, refine time: {timings[1]}')

            stats = Stats(result, target=config.integrand.target)
            integral, integral_unc = stats.get_integral_and_uncertainty()

            print(f"Final result: {integral:.5e} +/- {integral_unc:.5e}")

            divergences_vegas[n_survey_steps], uncertainties_vegas[n_survey_steps], x_ = stats.plot_convergence(plot=False)

    if compare_uniform:
        config.integrand.reset()

        t0 = datetime.now()
        result, timings = integrate_mp(
            config.integrand,
            n_points=config.n_points,
            average_grads=config.average_grads,
            n_survey_steps=n_survey_steps,
            n_refine_steps=config.n_refine_steps,
            masking_type=config.masking_type,
            cell_type=config.cell_type,
            n_cells=config.n_cells,
            integrator_class=UniformIntegrator,
        )

        print(f'Uniform integrator - time: {datetime.now() - t0}')
        print(f'Survey time: {timings[0]}, refine time: {timings[1]}')

        stats = Stats(result, target=config.integrand.target)
        integral, integral_unc = stats.get_integral_and_uncertainty()

        print(f"Final result: {integral:.5e} +/- {integral_unc:.5e}")

        divergences_, uncertainties_, x_ = stats.plot_convergence(plot=False)

    if compare_uniform: plt.plot(x_, divergences_, label=f'Uniform')

    if compare_vegas:
        for n_survey_steps, data in divergences_vegas.items():
            plt.plot(x_, divergences_vegas[n_survey_steps], label=f'VEGAS {n_survey_steps} steps')

    for n_survey_steps, data in divergences.items():
        plt.plot(x_, data, label=f'FLUX {n_survey_steps} steps')

    plt.legend()
    plt.title('Divergence')
    plt.grid()
    plt.yscale('log')
    plt.show()

    if compare_uniform: plt.plot(x_, uncertainties_, label=f'Uniform')

    if compare_vegas:
        for n_survey_steps, data in uncertainties.items():
            plt.plot(x_, uncertainties_vegas[n_survey_steps], label=f'VEGAS {n_survey_steps} steps')

    for n_survey_steps, data in uncertainties.items():
        plt.plot(x_, data, label=f'FLUX {n_survey_steps} steps')

    plt.legend()
    plt.title('Uncertainty')
    plt.grid()
    plt.yscale('log')
    plt.show()





def flux_sequent_train(dim=8, sobol_sample_prior=False):
    from flux.models.flows import RepeatedCouplingCellFlow
    from flux.models.samplers.samplers import UniformSampler, SobolSampler
    from flux.models.trainers import VarianceTrainer

    flow = RepeatedCouplingCellFlow(
        dim=dim,
        cell=CellType.PWLINEAR,
        masking=MaskingType.CHECKERBOARD,
        cell_parameters=dict(
            n_bins=16,
            n_hidden=3,
            dim_hidden=32,
        )
    )

    trainer = VarianceTrainer(
        flow=flow,
        prior=UniformSampler(dim=dim),
        sample_prior=SobolSampler(dim=dim) if sobol_sample_prior else None,
    )

    integrand = GaussIntegrand(dim=dim)

    integrator = UniformSurveyIntegrator(
        integrand=integrand,
        trainer=trainer,
        n_points=2000,
    )

    n_survey_acts = 40
    n_survey_steps_per_act = 2

    print('FLUX, unweighted')
    print(f'sobol_sample_prior: {sobol_sample_prior}')
    print(f'Target: {integrand.target:.7e}')

    for i in range(n_survey_acts):
        t0 = datetime.now()
        integrator.survey(n_steps=n_survey_steps_per_act)
        survey_time = datetime.now() - t0

        integrator.history = integrator.get_empty_history()

        t1 = datetime.now()
        integrator.refine(n_steps=25)
        refine_time = datetime.now() - t1

        history = integrator.history

        print(f'n_survey_steps: {(i + 1) * n_survey_steps_per_act}')

        integrals, uncertainties, deviations = [], [], []
        for i in range(len(history)):
            integral = history['integral'][:i+1].mean()
            integral_unc = ((history["uncertainty"][:i+1] ** 2).sum() / history["uncertainty"][:i+1].count() ** 2) ** 0.5

            integrals.append(round(integral, 8))
            uncertainties.append(round(integral_unc, 8))
            deviations.append(round(abs(integral - integrand.target), 8))

        print(f'Integrals: {integrals}')
        print(f'Uncertainties: {uncertainties}')
        print(f'Deviations: {deviations}')
        print(f'Survey time: {survey_time}, refine time: {refine_time}')
        print('='*20)



def vegas_sequent_train(dim=8):
    import pandas

    integrand = CamelIntegrand(dim=dim)
    integrator = VegasIntegrator(dim)

    n_survey_acts = 40
    n_survey_steps_per_act = 2

    print('VEGAS, unweighted')
    print(f'Target: {integrand.target:.7e}')

    for i in range(n_survey_acts):
        t0 = datetime.now()
        integrator.train(
            integrand,
            n_train_steps=n_survey_steps_per_act,
            n_train_samples=20000,
        )
        survey_time = datetime.now() - t0
        integrator.history = pandas.DataFrame({
            "integral": pandas.Series([], dtype="float"),
            "uncertainty": pandas.Series([], dtype="float"),
            "n_points": pandas.Series([], dtype="int"),
            "phase": pandas.Series([], dtype="str"),
        })
        t1 = datetime.now()
        integrator.refine(
            integrand,
            n_refine_steps=25,
            n_refine_samples=20000,
        )
        refine_time = datetime.now() - t1

        history = integrator.history

        print(f'n_survey_steps: {(i + 1) * n_survey_steps_per_act}')

        integrals, uncertainties, deviations = [], [], []
        for i in range(len(history)):
            integral = history['integral'][:i+1].mean()
            integral_unc = ((history["uncertainty"][:i+1] ** 2).sum() / history["uncertainty"][:i+1].count() ** 2) ** 0.5

            integrals.append(round(integral, 8))
            uncertainties.append(round(integral_unc, 8))
            deviations.append(round(abs(integral - integrand.target), 8))

        print(f'Integrals: {integrals}')
        print(f'Uncertainties: {uncertainties}')
        print(f'Deviations: {deviations}')
        print(f'Survey time: {survey_time}, refine time: {refine_time}')
        print('='*20)



if __name__ == '__main__':
    # test_0(
    #     IntegrationConfig(
    #         integrand=GaussIntegrand(dim=5),
    #         n_points=1000,
    #         n_survey_steps=0,
    #         n_refine_steps=150*5,
    #         # average_grads=False,
    #         # masking_type=MaskingType.CHECKERBOARD,
    #         # cell_type=CellType.PWLINEAR,
    #         # n_cells=None,
    #     )
    # )
    # test1(
    #     IntegrationConfig(
    #         integrand=LongIntegrand(),
    #         n_points=1000,
    #         n_survey_steps=1000,
    #         n_refine_steps=100,
    #         # average_grads=False,
    #         masking_type=MaskingType.TEST,
    #         cell_type=CellType.PWQUADRATIC,
    #         # n_cells=None,
    #     )
    # )

    # main_sequent()
    # vegas_test()
    # test_2(
    #      IntegrationConfig(
    #         integrand=CamelIntegrand(dim=8),
    #         n_points=10000,
    #         n_survey_steps=0,
    #         n_refine_steps=40,
    #     )
    # )

    # for n_refine_steps in (10, 40, 100, 160):
    #     for n_survey_steps in (5, 10, 20, 40, 70, 100):
    #         try:
    #             main_sequent(n_refine_steps, n_survey_steps)
    #         except Exception as e:
    #             print(f'ERROR: {e}')
    #             print(n_refine_steps, n_survey_steps)

    # print('+'*20)

    # for n_refine_steps in (10, 40, 100, 160):
    #     for n_survey_steps in (5, 10, 20, 40, 70, 100):
    #         try:
    #             main_sequent(n_refine_steps, n_survey_steps, sobol_sample_prior=True)
    #         except Exception as e:
    #             print(f'ERROR: {e}')
    #             print(n_refine_steps, n_survey_steps, True)

    # print('+'*20)

    # for n_refine_steps in (10, 40, 100, 160):
    #     for n_survey_steps in (5, 10, 20, 40, 70, 100):
    #         try:
    #             vegas_test(n_refine_steps, n_survey_steps)
    #         except Exception as e:
    #             print(f'ERROR: {e}')
    #             print(n_refine_steps, n_survey_steps, True)


    # print('#'*10, 'DIM = 8', '#'*10)
    # for n_survey_steps in (5, 10, 20, 40, 70, 100):
    # # for n_survey_steps in (5, 10):
    #     try:
    #         main_sequent(80, n_survey_steps, 8, False)
    #     except Exception as e:
    #         print(f'ERROR: {e}')
    #         print(n_survey_steps, False)
    #         raise

    # for n_survey_steps in (5, 10, 20, 40, 70, 100):
    # # for n_survey_steps in (5, 10):
    #     try:
    #         main_sequent(80, n_survey_steps, 8, True)
    #     except Exception as e:
    #         print(f'ERROR: {e}')
    #         print(n_survey_steps, True)
    
    # for n_survey_steps in (5, 10, 20, 40, 70, 100):
    # # for n_survey_steps in (5, 10):
    #     try:
    #         vegas_test(20, n_survey_steps, 8)
    #     except Exception as e:
    #         print(f'ERROR: {e}')
    #         print(n_survey_steps)


    # print()
    # print('#'*10, 'DIM = 16', '#'*10)
    # for n_survey_steps in (5, 10, 20, 40, 70, 100):
    # # for n_survey_steps in (5, 10):
    #     try:
    #         main_sequent(80, n_survey_steps, 16, False)
    #     except Exception as e:
    #         print(f'ERROR: {e}')
    #         print(n_survey_steps, False)
    #         raise

    # for n_survey_steps in (5, 10, 20, 40, 70, 100):
    # # for n_survey_steps in (5, 10):
    #     try:
    #         main_sequent(80, n_survey_steps, 16, True)
    #     except Exception as e:
    #         print(f'ERROR: {e}')
    #         print(n_survey_steps, True)
    
    # for n_survey_steps in (5, 10, 20, 40, 70, 100):
    # # for n_survey_steps in (5, 10):
    #     try:
    #         vegas_test(20, n_survey_steps, 16)
    #     except Exception as e:
    #         print(f'ERROR: {e}')
    #         print(n_survey_steps)

    # print('#'*10, 'DIM = 8', '#'*10)
    # flux_sequent_train(8, False)
    # # flux_sequent_train(8, True)
    # vegas_sequent_train(8)
    # print()
    # print('#'*10, 'DIM = 16', '#'*10)
    # flux_sequent_train(16, False)
    # # flux_sequent_train(16, True)
    # vegas_sequent_train(16)

    # flux_sequent_train(4, False)

    def trial(dim):
        from flux.models.flows import RepeatedCouplingCellFlow
        from flux.models.samplers.samplers import UniformSampler
        from flux.models.trainers import VarianceTrainer

        flow = RepeatedCouplingCellFlow(
            dim=dim,
            cell=CellType.PWLINEAR,
            masking=MaskingType.CHECKERBOARD,
            cell_parameters=dict(
                n_bins=16,
                n_hidden=3,
                dim_hidden=32,
            )
        )

        trainer = VarianceTrainer(
            flow=flow,
            prior=UniformSampler(dim=dim),
        )

        integrand = GaussIntegrand(dim=dim)

        integrator = UniformSurveyIntegrator(
            integrand=integrand,
            trainer=trainer,
            n_points=20000,
        )

        integrator.survey(n_steps=10)
        integrator.refine(n_steps=10)

        data = integrator.history.loc[integrator.history["phase"] == "refine"]

        integral = data["integral"].mean()
        integral_unc = ((data["uncertainty"] ** 2).sum() / data["uncertainty"].count() ** 2) ** 0.5

        print(f'{integral} +- {integral_unc}')

    trial(4)
