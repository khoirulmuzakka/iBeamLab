"""Native simulation backends and their input/result types.

``SimnraSimulator`` owns persistent SIMNRA COM workers. Pass an instance to
``ibeamlab.generation.DataGenerator`` to generate a SIMNRA-backed dataset.
"""

from ._ibeamlab_cpp.simulator import (
    DummySimulator,
    ISimulator,
    SimnraMethod,
    SimnraSimulator,
    SimnraSimulatorConfig,
    SimulationFailure,
    SimulationInput,
    SimulationOptions,
    SimulationResult,
    Spectrum,
)

__all__ = [
    "DummySimulator", "ISimulator", "SimnraMethod", "SimnraSimulator",
    "SimnraSimulatorConfig", "SimulationFailure", "SimulationInput",
    "SimulationOptions", "SimulationResult", "Spectrum",
]
