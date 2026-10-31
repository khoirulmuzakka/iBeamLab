"""Export custom PyTorch models to iBeamLab's deployment contract."""
from pathlib import Path
import tempfile
import inspect


def validate_onnx_contract(graph, metadata):
    """Check a standalone ONNX graph against its scientific tensor contract.

    Parameters
    ----------
    graph : onnx.ModelProto
        Graph loaded without resolving external tensor files.
    metadata : ModelMetadata or native ModelMetadata
        Directional scientific contract, including tensor identifiers and widths.

    Raises
    ------
    ValueError
        Metadata or graph names, types, ranks, dimensions or external-data use
        violate the contract.

    Notes
    -----
    Requires ONNX only when called. Checks exactly one float32 graph input and
    one output, fixed feature widths, and dynamic batch/layer axes for layer graphs.
    Runs the ONNX structural checker, including nested graph validation for external
    initializers. Does not execute the graph or prove provider/operator support.
    Errors from the ONNX checker are propagated.
    """
    import onnx
    from ._engine import to_core
    metadata = to_core(metadata)
    metadata.validate()
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError("graph must have exactly one input and one output")
    for tensor, name, rank, width, sequence in (
        (graph.graph.input[0], metadata.input_name, metadata.input_rank(), metadata.input_width(), metadata.layer_input()),
        (graph.graph.output[0], metadata.output_name, 2, metadata.output_width(), False),
    ):
        spec = tensor.type.tensor_type
        dims = spec.shape.dim
        if tensor.name != name or spec.elem_type != onnx.TensorProto.FLOAT or len(dims) != rank:
            raise ValueError("graph tensor name, dtype or rank differs from metadata")
        if not dims[-1].HasField("dim_value") or dims[-1].dim_value != width:
            raise ValueError("graph feature width differs from metadata")
        if sequence and any(dims[i].HasField("dim_value") for i in (0, 1)):
            raise ValueError("layer graph requires dynamic batch and layer dimensions")
        if metadata.layer_input() and dims[0].HasField("dim_value"):
            raise ValueError("layer graph output requires a dynamic batch dimension")
    def standalone(body):
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in body.initializer):
            raise ValueError("External ONNX tensors are not supported")
        for node in body.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    standalone(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for nested in attr.graphs:
                        standalone(nested)
    standalone(graph.graph)
    onnx.checker.check_model(graph, full_check=True)


def export_package(model, example, metadata, destination, *, opset=17, verify=True,
                   graph_exporter=None):
    """Export a CPU PyTorch model and its scientific contract to a new .ibm file.

    Parameters
    ----------
    model : torch.nn.Module
        CPU float32 model with one tensor input and one packed two-dimensional output.
    example : torch.Tensor
        Float32 CPU example shaped (batch, features) or (batch, layers, elements).
        It must be at the declared ONNX input boundary, after runtime input transforms.
    metadata : ModelMetadata or native ModelMetadata
        Fitted scientific contract. Numerical transforms must use the coefficients
        used for training; physical preparation is declared for runtime execution.
    destination : str or pathlib.Path
        New .ibm file. The parent directory must exist; existing files are rejected.
    opset : int, default=17
        ONNX operator-set version passed to the graph exporter.
    verify : bool, default=True
        Compare PyTorch outputs with native ONNX execution for the example batch
        and a second batch size. Requires an ONNX-enabled native build.
    graph_exporter : callable(model, example, path, options), optional
        Architecture-specific exporter that writes the ONNX graph. Required for
        layer-sequence models so the layer computation is truly dynamic.

    Returns
    -------
    Package or native Package
        Written package. Python metadata inputs return the Python Package facade;
        native metadata preserves the native package return type.

    Raises
    ------
    FileExistsError
        destination already exists.
    ValueError
        Example shape/device/dtype or the scientific graph contract is invalid.
    RuntimeError
        Native graph execution or package writing fails.

    Notes
    -----
    PyTorch and ONNX are imported on use. Structural graph validation always
    runs; verify controls numerical parity only. Export errors from those
    dependencies are propagated. The original model train/eval mode is restored
    on completion or failure.

    Transforms marked GRAPH must already be embedded and must not be applied
    twice. Output-transform placement declares whether graph values are numerical
    or restored physical reference units. Physical spectrum preparation remains
    at runtime. Inverse uncertainty includes means, probabilities and standard
    deviations, with affine-only runtime propagation of deviations.

    External tensor files are unsupported. Dynamic shape declarations and numerical
    parity checks do not establish scientific accuracy at untested layer counts.
    """
    import torch
    import numpy as np
    from ._engine import engine, to_core
    from .metadata import ModelMetadata
    from .package import Package as PythonPackage
    wrapped = isinstance(metadata, ModelMetadata)
    metadata = to_core(metadata)
    Package, run_graph = engine().Package, engine().run_graph
    metadata.validate()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if example.ndim != metadata.input_rank() or example.shape[-1] != metadata.input_width():
        raise ValueError("Example rank/feature width must match the input schema")
    if metadata.layer_input() and graph_exporter is None:
        raise ValueError("Layer-sequence models require a dynamic graph_exporter callback")
    if example.dtype != torch.float32 or example.device.type != 'cpu':
        raise ValueError("Export requires a float32 CPU example and CPU model")
    was_training = model.training
    model.eval()
    try:
        with tempfile.TemporaryDirectory(prefix='ibeamlab-export-') as directory:
            path = Path(directory) / 'model.onnx'
            options = dict(input_names=[metadata.input_name],
                           output_names=[metadata.output_name],
                           dynamic_axes={metadata.input_name: {0: 'batch'},
                                         metadata.output_name: {0: 'batch'}},
                           opset_version=opset)
            if metadata.layer_input():
                options["dynamic_axes"][metadata.input_name][1] = "layers"
            if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
                options['dynamo'] = False
            with torch.no_grad():
                expected = model(example).detach().cpu().numpy()
                if expected.shape != (len(example), metadata.output_width()):
                    raise ValueError("Model output differs from metadata schema")
                if graph_exporter is None:
                    torch.onnx.export(model, example, path, **options)
                else:
                    graph_exporter(model, example, path, options)
            import onnx
            graph = onnx.load(path, load_external_data=False)
            validate_onnx_contract(graph, metadata)
            package = Package.from_onnx(path, metadata)
            if verify:
                np.testing.assert_allclose(run_graph(package, np.ascontiguousarray(example.detach().numpy())),
                                           expected, rtol=1e-4, atol=1e-5)
                # Check a second batch size to detect fixed-batch tracing.
                second = example[:1].repeat((3,) + (1,) * (example.ndim - 1))
                with torch.no_grad():
                    expected_second = model(second).detach().numpy()
                np.testing.assert_allclose(run_graph(package, second.numpy()), expected_second,
                                           rtol=1e-4, atol=1e-5)
            package.write(destination)
            return PythonPackage(package) if wrapped else package
    finally:
        model.train(was_training)
