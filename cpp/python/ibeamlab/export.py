"""Export custom PyTorch models to iBeamLab's deployment contract."""
from pathlib import Path
import tempfile
import inspect


def export_package(model, example, metadata, destination, *, opset=17, verify=True):
    """Export one tensor [batch,width] in/out; verify raw graph parity.

    Metadata preprocessing runs outside the graph. `example` is therefore an
    already-preprocessed float32 CPU tensor. Inverse uncertainty output packs
    [mean cells, probabilities, posterior std cells] before EDP postprocessing.
    External-data graphs are deliberately rejected: packages embed one graph.
    """
    import torch
    import numpy as np
    try:
        from ._ibeamlab import Package, run_graph
    except ModuleNotFoundError as error:
        if error.name != __package__ + "._ibeamlab":
            raise
        from _ibeamlab import Package, run_graph
    metadata.validate()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if example.ndim != 2 or example.shape[1] != metadata.input_width():
        raise ValueError("Example must have shape [batch,input_width]")
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
            if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
                options['dynamo'] = False
            with torch.no_grad():
                expected = model(example).detach().cpu().numpy()
                if expected.shape != (len(example), metadata.output_width()):
                    raise ValueError("Model output differs from metadata schema")
                torch.onnx.export(model, example, path, **options)
            import onnx
            graph = onnx.load(path, load_external_data=False)
            if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
                raise ValueError("External ONNX tensors are not supported")
            onnx.checker.check_model(graph)
            package = Package.from_onnx(path, metadata)
            if verify:
                np.testing.assert_allclose(run_graph(package, np.ascontiguousarray(example.detach().numpy())),
                                           expected, rtol=1e-4, atol=1e-5)
                # Check a second batch size to detect fixed-batch tracing.
                second = example[:1].repeat(3, 1)
                with torch.no_grad():
                    expected_second = model(second).detach().numpy()
                np.testing.assert_allclose(run_graph(package, second.numpy()), expected_second,
                                           rtol=1e-4, atol=1e-5)
            package.write(destination)
            return package
    finally:
        model.train(was_training)
