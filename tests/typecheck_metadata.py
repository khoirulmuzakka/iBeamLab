"""Static API contract: run mypy with --warn-unused-ignores.

Expected errors below must remain errors; unused ignores detect weakened types.
Functions are never executed and need no compiled extension.
"""
import ibeamlab as ibl
from ibeamlab import native


def valid_metadata(setup: ibl.ExperimentalSetup) -> ibl.ModelMetadata:
    operations: ibl.TensorPipeline = [ibl.MultiplyScale([1e-4])]
    inputs = ibl.LayerInputSchema(["Si", "O"], [
        ibl.ElementComposition("Si", (ibl.Isotope(28, 0.9), ibl.Isotope(29, 0.1)))])
    outputs = ibl.SpectraSchema([ibl.SpectrumSpec("RBS", 5000)])
    forward = ibl.ForwardModelMetadata(setup, inputs, outputs,
        input_transform=ibl.TransformPipeline(operations),
        output_transform=[ibl.SpectrumTransform("RBS", ibl.TransformPipeline([ibl.MultiplyScale([.01])],
            direction=ibl.TransformDirection.INVERSE))],
        output_preparation=[ibl.SpectrumProcessing("RBS", [ibl.Rebin()])])
    return ibl.ModelMetadata(forward)


def native_interoperability(setup: native.ExperimentalSetup, sample: native.Sample) -> ibl.ModelMetadata:
    parameters = [ibl.ParameterSpec("thickness", ibl.LayerThickness(0))]
    inputs = ibl.ParameterInputSchema(sample, parameters)
    return ibl.ModelMetadata(ibl.ForwardModelMetadata(setup, inputs,
        native.SpectraSchema([native.SpectrumSpec("RBS", 5000)])))


def invalid_metadata(setup: ibl.ExperimentalSetup) -> None:
    ibl.ParameterSpec("bad", "thickness")  # type: ignore[arg-type]
    ibl.LayerInputSchema([123])  # type: ignore[list-item]
    ibl.Isotope("Si")  # type: ignore[arg-type]
    ibl.LayerInputSchema(["Si"], ["natural"])  # type: ignore[list-item]
    ibl.EdpSchema(["Si"], "8")  # type: ignore[arg-type]
    ibl.ForwardModelMetadata(setup, 123, ibl.SpectraSchema())  # type: ignore[arg-type]
    ibl.ForwardModelMetadata(setup, ibl.LayerInputSchema(["Si"]), ibl.SpectraSchema(),
        output_preparation=[ibl.MultiplyScale([1])])  # type: ignore[list-item]


def inference_options() -> ibl.InferenceOptions:
    ibl.InferenceOptions(execution_provider="invalid")  # type: ignore[arg-type]
    return ibl.InferenceOptions(execution_provider="cuda", device_id=0)
