"""Build a runtime ONNX Loop, rather than tracing a fixed layer count."""
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F


class _Step(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.block = model.layer_block
        self.register_buffer("factor", model.ad_input_factor.detach().clone())

    def forward(self, layer, contributions, hidden):
        layer = layer * self.factor
        mask = (~(layer == 0).all(dim=1)).to(layer.dtype).unsqueeze(1)
        contribution, next_hidden = self.block(layer, layer[:, :0], hidden)
        return contributions + contribution * mask, hidden * (1 - mask) + next_hidden * mask


class _Decode(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.context = model.setup_context_encoder
        self.decoder = model.spectrum_decoder
        self.refiner = model.spectrum_refiner
        self.register_buffer("scale", model.output_scale.detach().clone())

    def forward(self, contributions, hidden):
        setup = hidden[:, :0]
        context = self.context(hidden.new_ones((hidden.shape[0], 1)))
        coarse = F.softplus(self.decoder(torch.cat((contributions, hidden, context), 1)))
        return F.softplus(self.refiner(coarse, setup)) * F.softplus(self.scale)


def _graph(model, directory, opset):
    import inspect
    import onnx
    from onnx import helper as h, TensorProto as T, compose

    c, s, e = model.contribution_size, model.hidden_size, len(model.elements)
    kwargs = dict(opset_version=opset, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    for name, module, args, inputs, outputs in (
        ("step", _Step(model), (torch.zeros(2, e), torch.zeros(2, c), torch.zeros(2, s)),
         ["layer", "contributions", "hidden"], ["next_contributions", "next_hidden"]),
        ("decode", _Decode(model), (torch.zeros(2, c), torch.zeros(2, s)),
         ["contributions", "hidden"], ["spectra"]),
    ):
        torch.onnx.export(module.eval(), args, str(directory / f"{name}.onnx"),
                          input_names=inputs, output_names=outputs,
                          dynamic_axes={key: {0: "batch"} for key in inputs + outputs}, **kwargs)
    step = compose.add_prefix(onnx.load(directory / "step.onnx"), "step/")
    decode = compose.add_prefix(onnx.load(directory / "decode.onnx"), "decode/")
    vi = lambda name, shape, dtype=T.FLOAT: h.make_tensor_value_info(name, dtype, shape)
    const = lambda name, values, dtype=T.INT64: h.make_node(
        "Constant", [], [name], value=h.make_tensor(name, dtype, [len(values)], values))
    body = h.make_graph(
        [h.make_node("Gather", ["inputs", "iteration"], ["step/layer"], axis=1),
         *step.graph.node, h.make_node("Identity", ["condition"], ["continue"])],
        "layer_step", [vi("iteration", [], T.INT64), vi("condition", [], T.BOOL),
                       vi("step/contributions", ["batch", c]), vi("step/hidden", ["batch", s])],
        [vi("continue", [], T.BOOL), vi("step/next_contributions", ["batch", c]),
         vi("step/next_hidden", ["batch", s])], initializer=step.graph.initializer)
    nodes = [h.make_node("Shape", ["inputs"], ["shape"]), const("batch_axis", [0]),
             const("layer_axis", [1]), const("c_width", [c]), const("h_width", [s]),
             h.make_node("Gather", ["shape", "batch_axis"], ["batch_shape"], axis=0),
             h.make_node("Gather", ["shape", "layer_axis"], ["layers_vector"], axis=0),
             h.make_node("Squeeze", ["layers_vector", "batch_axis"], ["layers"]),
             const("start", [True], T.BOOL),
             h.make_node("Squeeze", ["start", "batch_axis"], ["start_scalar"])]
    # Determine the last nonempty layer across the entire batch once. Reducing
    # the Loop trip count avoids executing the recurrent block on trailing
    # padding. Interior empty layers remain masked no-ops, never stop markers.
    reduce_axes = lambda source, output, axes, axes_input: h.make_node(
        "ReduceMax", [source, axes_input] if opset >= 18 else [source], [output],
        **({"keepdims": 0} if opset >= 18 else {"axes": axes, "keepdims": 0}))
    nodes += [const("zero_int_vector", [0]), const("one_int_vector", [1]),
              *([const("activity_axes", [0, 2])] if opset >= 18 else []),
              const("zero_float", [0.], T.FLOAT),
              h.make_node("Squeeze", ["zero_int_vector", "batch_axis"], ["zero_int"]),
              h.make_node("Squeeze", ["one_int_vector", "batch_axis"], ["one_int"]),
              h.make_node("Abs", ["inputs"], ["absolute_ad"]),
              reduce_axes("absolute_ad", "layer_magnitude", [0, 2], "activity_axes"),
              h.make_node("Greater", ["layer_magnitude", "zero_float"], ["active_layers"]),
              h.make_node("Range", ["zero_int", "layers", "one_int"], ["layer_indices"]),
              h.make_node("Add", ["layer_indices", "one_int"], ["layer_positions"]),
              h.make_node("Where", ["active_layers", "layer_positions", "zero_int"], ["active_positions"]),
              reduce_axes("active_positions", "effective_layers", [0], "batch_axis")]
    for label, width in (("c", "c_width"), ("h", "h_width")):
        nodes += [h.make_node("Concat", ["batch_shape", width], [label + "_shape"], axis=0),
                  h.make_node("ConstantOfShape", [label + "_shape"], [label + "_zero"])]
    nodes += [h.make_node("Loop", ["effective_layers", "start_scalar", "c_zero", "h_zero"],
                          ["decode/contributions", "decode/hidden"], body=body),
              *decode.graph.node, h.make_node("Identity", ["decode/spectra"], ["outputs"])]
    graph = h.make_graph(nodes, "LRN_AD_dynamic", [vi("inputs", ["batch", "layers", e])],
                         [vi("outputs", ["batch", model.output_size])],
                         initializer=decode.graph.initializer)
    result = h.make_model(graph, opset_imports=list(step.opset_import), ir_version=step.ir_version)
    onnx.checker.check_model(result, full_check=True)
    target = directory / "dynamic.onnx"
    onnx.save(result, target)
    return target


def export_dynamic(model, path, **kwargs):
    device = next(model.parameters()).device
    training = model.training
    try:
        model.cpu().eval()
        return _export_dynamic_cpu(model, path, **kwargs)
    finally:
        model.to(device).train(training)


def _export_dynamic_cpu(model, path, **kwargs):
    from ._native import native
    from .lrn_ad import _PhysicalInputWrapper
    with TemporaryDirectory(prefix="ibeamlab-dynamic-") as temporary:
        directory = Path(temporary)
        # Reuse the established setup/output metadata and validation path.
        fixed = _PhysicalInputWrapper(model).export(directory / "fixed.zip", **kwargs)
        metadata = native.model.ModelPackage.open(fixed).metadata
        metadata.format_version = 5
        metadata.input_dimension = 0
        transform = native.model.TransformSpec()
        transform.type = "identity"
        transform.input_dimension = 0
        metadata.input_transform = transform
        forward = metadata.forward
        forward.input_parameters = []
        spec = native.model.LayerInputSpec()
        spec.dynamic_layers = True
        spec.elements = list(model.elements)
        spec.trained_max_layers = model.layer_count
        spec.ad_input_factor = float(model.ad_input_factor)
        forward.layer_input = spec
        metadata.forward = forward
        model.eval()
        graph = _graph(model, directory, kwargs.get("opset_version", 17))
        native.model.ModelPackage.from_onnx(graph, metadata).write(Path(path))
    return Path(path)
