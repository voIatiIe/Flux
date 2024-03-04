import os
import pandas as pd
import torch
import torch.multiprocessing as mp
import typing as t

from datetime import timedelta
from multiprocessing.connection import Connection

from flux.models.integrators import UniformSurveyIntegrator, BaseIntegrator
from flux.models.integrands.base import BaseIntegrand
from flux.models.trainers import BaseTrainer
from flux.utils.constants import IntegrationResult
from flux.utils.pgm import ProcessGroupManager
from flux.models.trainers import VarianceTrainer
from flux.models.flows import BaseFlow, RepeatedCouplingCellFlow
from flux.models.samplers import BaseSampler, UniformSampler
from flux.utils.constants import CellType, MaskingType, Backend


def _integrate(
    rank: int,
    world_size: int,
    integrand: BaseIntegrand,
    flow_class: t.Type[BaseFlow] | None,
    trainer_class: t.Type[BaseTrainer] | None,
    integrator_class: t.Type[BaseIntegrator],
    trainer_prior_class: t.Type[BaseSampler] | None,
    cell_type: CellType,
    masking_type: MaskingType,
    n_cells: int,
    n_points: int,
    pipe: Connection,
    threads_per_process: int,
    average_grads: bool,
    device: torch.device,
    backend: Backend,
    n_survey_steps: int,
    n_refine_steps: int,
) -> None:
    torch.set_num_threads(threads_per_process)
    dim = integrand.dim

    with ProcessGroupManager(rank, world_size, backend.value):
        if flow_class and trainer_class and trainer_prior_class:
            flow = flow_class(
                dim=dim,
                cell=cell_type,
                masking=masking_type,
                n_cells=n_cells,
            )
            flow = flow.to(device)

            trainer = trainer_class(
                flow=flow,
                prior=trainer_prior_class(dim=dim, device=device),
                average_grads=average_grads,
            )

            integrator = integrator_class(
                integrand=integrand,
                trainer=trainer,
                n_points=n_points,
                device=device,
            )
        else:
            integrator = integrator_class(
                integrand=integrand,
                n_points=n_points,
                device=device,
            )

        result = integrator.integrate(n_survey_steps=n_survey_steps, n_refine_steps=n_refine_steps)
        result.history['rank'] = rank
        result.history.to_csv(f'history_{rank}.csv')

        pipe.send((result.survey_time, result.refine_time))


def integrate(
    integrand: BaseIntegrand,
    *,
    flow_class: t.Type[BaseFlow] = RepeatedCouplingCellFlow,
    trainer_class: t.Type[BaseTrainer] = VarianceTrainer,
    integrator_class: t.Type[BaseIntegrator] = UniformSurveyIntegrator,
    trainer_prior_class: t.Type[BaseSampler] = UniformSampler,
    cell_type: CellType = CellType.PWLINEAR,
    masking_type: MaskingType = MaskingType.CHECKERBOARD,
    n_cells: int | None = None,
    n_points: int = 10000,
    world_size: int = 4,
    threads_per_process: int = 2,
    average_grads: bool = False,
    device: torch.device = torch.device("cpu"),
    backend: Backend = Backend.gloo,
    n_survey_steps: int = 10,
    n_refine_steps: int = 10,
) -> (pd.DataFrame, (timedelta, timedelta)):
    parent_pipe, child_pipe = mp.Pipe()

    args = (
        world_size,
        integrand,
        flow_class,
        trainer_class,
        integrator_class,
        trainer_prior_class,
        cell_type,
        masking_type,
        n_cells,
        n_points,
        child_pipe,
        threads_per_process,
        average_grads,
        device,
        backend,
        n_survey_steps,
        n_refine_steps,
    )
    mp.spawn(
        _integrate,
        args=args,
        nprocs=world_size,
        join=True,
    )

    results: t.List[pd.DataFrame] = [pd.read_csv(f'history_{rank}.csv') for rank in range(world_size)]
    for rank in range(world_size):
        os.remove(f'history_{rank}.csv')

    history = pd.concat(results, ignore_index=False)

    survey_time, refine_time = timedelta(), timedelta()

    for rank in range(world_size):
        s, r = parent_pipe.recv()

        survey_time += s / world_size
        refine_time += r / world_size

    return history, (survey_time, refine_time)
