"""Export custom PyTorch models to iBeamLab's deployment contract."""
from pathlib import Path
import tempfile
import inspect


def validate_onnx_contract(graph, metadata):
    """Validate exported tensor names/types/shapes without constructing a runtime.

    ONNX is needed only for export. Native package I/O remains runtime independent;
    the C++ inference session repeats shape checks when loading an executable graph.
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
    """Export one input tensor and a packed [batch,width] output; verify raw parity.

    `example` is a float32 CPU tensor at the declared ONNX input boundary.
    Runtime input transforms have already been applied; Graph input transforms
    remain embedded in the supplied model and must not be applied a second time.
    Output transforms declare whether graph outputs are numerical or already
    restored to physical reference units. Physical preparation is always runtime.
    Inverse uncertainty packs [means, probabilities, posterior std]. Only affine
    output transforms can propagate its standard deviations outside the graph.
    External-data graphs are deliberately rejected: packages embed one graph.
    Layer models require graph_exporter(model, example, path, options), an
    architecture-specific callback that writes an ONNX graph with a runtime loop
    or other truly dynamic layer computation. Package writing stays generic.
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
