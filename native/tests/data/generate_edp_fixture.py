"""Regenerate edp.onnx with the optional Python onnx package.

The dynamic-batch model maps [a, b, c, d] to [a, b, c, d, 0, 0].
Native tests interpret the six outputs as three layers of two elements.
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

weights = helper.make_tensor(
    "weights", TensorProto.FLOAT, [4, 6],
    [float(row == column) for row in range(4) for column in range(6)],
)
graph = helper.make_graph(
    [helper.make_node("MatMul", ["inputs", "weights"], ["outputs"])],
    "edp-test",
    [helper.make_tensor_value_info("inputs", TensorProto.FLOAT, ["batch", 4])],
    [helper.make_tensor_value_info("outputs", TensorProto.FLOAT, ["batch", 6])],
    [weights],
)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8,
)
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name("edp.onnx"))
