"""Generate small standalone ONNX graphs, independent of the legacy runtime."""
from pathlib import Path
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper
import numpy as np
root = Path(__file__).parent / 'fixtures'
root.mkdir(exist_ok=True)
for name, width, weights, bias in [
    ('forward', 1, [[1, 2, 3, 4]], [0, 0, 0, 0]),
    ('forward_5000', 1, [list(range(1, 5001))], [0] * 5000),
    ('forward_parameters', 10, [[1, 2, 3, 4]] * 10, [0] * 4),
    ('inverse', 4, [[1, 2, 0, 0, 0, 0]] * 4, [0, 0, .8, .5, .2, .3]),
    ('forward_multi', 1, [[1, 2, 3, 4, 5, 6, 7]], [0] * 7),
    ('inverse_multi', 7, [[1, 2, 0, 0, 0, 0]] * 4 + [[10, 20, 0, 0, 0, 0]] * 3, [0, 0, .8, .5, .2, .3]),
]:
    w = np.asarray(weights, dtype=np.float32)
    graph = h.make_graph([h.make_node('MatMul', ['inputs', 'weights'], ['product']),
                          h.make_node('Add', ['product', 'bias'], ['outputs'])], name,
                         [h.make_tensor_value_info('inputs', T.FLOAT, ['batch', width])],
                         [h.make_tensor_value_info('outputs', T.FLOAT, ['batch', w.shape[1]])],
                         [numpy_helper.from_array(w, 'weights'),
                          numpy_helper.from_array(np.asarray(bias, dtype=np.float32), 'bias')])
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, root / f'{name}.onnx')

# Architecture-independent layer-sequence fixture: sum layers, then decode 5000 channels.
weights = np.stack((np.arange(1, 5001), 2 * np.arange(1, 5001))).astype(np.float32)
graph = h.make_graph([
    h.make_node('ReduceSum', ['inputs', 'layer_axis'], ['summed'], keepdims=0),
    h.make_node('MatMul', ['summed', 'weights'], ['outputs'])], 'forward_layers',
    [h.make_tensor_value_info('inputs', T.FLOAT, ['batch', 'layers', 2])],
    [h.make_tensor_value_info('outputs', T.FLOAT, ['batch', 5000])],
    [numpy_helper.from_array(np.array([1], dtype=np.int64), 'layer_axis'),
     numpy_helper.from_array(weights, 'weights')])
model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=9)
onnx.checker.check_model(model)
onnx.save(model, root / 'forward_layers.onnx')
