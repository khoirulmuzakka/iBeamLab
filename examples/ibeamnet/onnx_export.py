"""Architecture-specific RecurrentSurrogate runtime loop; package serialization stays generic."""
from pathlib import Path
import tempfile
import torch
from torch import nn


class _Step(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.layer_encoder = network.layer_encoder
        self.state_projection = network.state_projection
        self.fusion = network.fusion
        self.gru = network.gru
        self.contribution = network.contribution
        # Fixed setup conditioning is constant at deployment: calculate once,
        # rather than evaluating its MLP for every layer at inference.
        with torch.no_grad():
            setup = network.setup_encoder(torch.ones(1, 1))
            gamma, beta = network.setup_to_film(setup).chunk(2, 1)
        self.register_buffer("setup", setup)
        self.register_buffer("gamma", torch.tanh(gamma) + 1)
        self.register_buffer("beta", beta)

    def forward(self, layer, contributions, hidden):
        mask = layer.ne(0).any(dim=1, keepdim=True).to(layer.dtype)
        encoded = self.layer_encoder(layer) * self.gamma + self.beta
        fused = self.fusion(torch.cat((encoded, self.setup.expand(layer.shape[0], -1),
                                      self.state_projection(hidden)), 1))
        next_hidden = self.gru(fused, hidden)
        contribution = self.contribution(torch.cat((fused, next_hidden), 1))
        return contributions + contribution * mask, hidden * (1 - mask) + next_hidden * mask


class _Decode(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.heads = network.heads
        with torch.no_grad():
            self.register_buffer("context", network.setup_context(torch.ones(1, 1)))

    def forward(self, contributions, hidden):
        latent = torch.cat((contributions, hidden, self.context.expand(hidden.shape[0], -1)), 1)
        return torch.cat([head(latent) for head in self.heads], 1)


def export_graph(network, example, path, options):
    """export_package callback. Inputs are already-scaled nonnegative elemental AD.

    Both batch and layer axes are dynamic. A real ONNX Loop carries only the
    hidden state and accumulated contributions. Its trip count is the last
    nonzero layer across the batch; interior empty layers remain masked no-ops.
    """
    import inspect
    import onnx
    from onnx import helper as h, TensorProto as T, compose
    if next(network.parameters()).device.type != "cpu":
        raise ValueError("RecurrentSurrogate export requires a CPU network")
    opset = options["opset_version"]
    c, s, e = network.contribution_size, network.hidden_size, len(network.elements)
    kwargs = dict(opset_version=opset, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with tempfile.TemporaryDirectory(prefix="recurrent-surrogate-graph-") as temporary:
        directory = Path(temporary)
        for name, module, args, inputs, outputs in (
            ("step", _Step(network), (torch.zeros(2, e), torch.zeros(2, c), torch.zeros(2, s)),
             ["layer", "contributions", "hidden"], ["next_contributions", "next_hidden"]),
            ("decode", _Decode(network), (torch.zeros(2, c), torch.zeros(2, s)),
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
    input_name, output_name = options["input_names"][0], options["output_names"][0]
    body = h.make_graph(
        [h.make_node("Gather", [input_name, "iteration"], ["step/layer"], axis=1),
         *step.graph.node, h.make_node("Identity", ["condition"], ["continue"])],
        "layer_step", [vi("iteration", [], T.INT64), vi("condition", [], T.BOOL),
                       vi("step/contributions", ["batch", c]), vi("step/hidden", ["batch", s])],
        [vi("continue", [], T.BOOL), vi("step/next_contributions", ["batch", c]),
         vi("step/next_hidden", ["batch", s])], initializer=step.graph.initializer)
    nodes = [h.make_node("Shape", [input_name], ["shape"]), const("batch_axis", [0]),
        const("layer_axis", [1]), const("c_width", [c]), const("h_width", [s]),
        h.make_node("Gather", ["shape", "batch_axis"], ["batch_shape"], axis=0),
        h.make_node("Gather", ["shape", "layer_axis"], ["layers_vector"], axis=0),
        h.make_node("Squeeze", ["layers_vector", "batch_axis"], ["layers"]),
        const("start", [True], T.BOOL), h.make_node("Squeeze", ["start", "batch_axis"], ["start_scalar"])]
    reduce_max = lambda source, output, axes, axes_input: h.make_node(
        "ReduceMax", [source, axes_input] if opset >= 18 else [source], [output],
        **({"keepdims": 0} if opset >= 18 else {"axes": axes, "keepdims": 0}))
    nodes += [const("zero_int_vector", [0]), const("one_int_vector", [1]),
        *([const("activity_axes", [0, 2])] if opset >= 18 else []), const("zero_float", [0.], T.FLOAT),
        h.make_node("Squeeze", ["zero_int_vector", "batch_axis"], ["zero_int"]),
        h.make_node("Squeeze", ["one_int_vector", "batch_axis"], ["one_int"]),
        reduce_max(input_name, "layer_magnitude", [0, 2], "activity_axes"),
        h.make_node("Greater", ["layer_magnitude", "zero_float"], ["active_layers"]),
        h.make_node("Range", ["zero_int", "layers", "one_int"], ["layer_indices"]),
        h.make_node("Add", ["layer_indices", "one_int"], ["layer_positions"]),
        h.make_node("Where", ["active_layers", "layer_positions", "zero_int"], ["active_positions"]),
        reduce_max("active_positions", "effective_layers", [0], "batch_axis")]
    for label, width in (("c", "c_width"), ("h", "h_width")):
        nodes += [h.make_node("Concat", ["batch_shape", width], [label + "_shape"], axis=0),
                  h.make_node("ConstantOfShape", [label + "_shape"], [label + "_zero"])]
    nodes += [h.make_node("Loop", ["effective_layers", "start_scalar", "c_zero", "h_zero"],
        ["decode/contributions", "decode/hidden"], body=body),
        *decode.graph.node, h.make_node("Identity", ["decode/spectra"], [output_name])]
    graph = h.make_graph(nodes, "recurrent_surrogate", [vi(input_name, ["batch", "layers", e])],
        [vi(output_name, ["batch", sum(network.spectra_channels)])], initializer=decode.graph.initializer)
    result = h.make_model(graph, opset_imports=list(step.opset_import), ir_version=step.ir_version)
    onnx.checker.check_model(result, full_check=True)
    onnx.save(result, path)
