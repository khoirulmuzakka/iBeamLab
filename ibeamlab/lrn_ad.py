"""LRN with scaled elemental areal-density layer inputs."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from .lrn import LRNModel


class LRN_AD(LRNModel):
    """Accept flattened layer-major AD values multiplied by ``ad_input_factor``.

    Elemental densities use physical units of 1e15 atoms/cm2. The core forward
    method accepts already scaled densities, just like iBeamNet posterior means.
    Fixed export retains the study's thickness/concentration interface. Dynamic
    export accepts physical AD [batch, layers, elements] and embeds AD scaling
    in a runtime ONNX loop. Currently supports fixed experimental setups.
    """

    def __init__(self, study, output_spectra_lengths, *, elements,
                 ad_input_factor=1e-5, **architecture):
        elements = tuple(elements)
        if not elements or len(set(elements)) != len(elements):
            raise ValueError("elements must be nonempty and unique")
        if not math.isfinite(ad_input_factor) or ad_input_factor <= 0:
            raise ValueError("ad_input_factor must be finite and positive")
        layer_count = len(study.experiment.sample.layers)
        thickness_indices, concentration_indices = [], []
        for layer in range(layer_count):
            thickness = [i for i, p in enumerate(study.parameters)
                         if p.layer == layer and p.kind == "layer_thickness"]
            concentrations = {p.element: i for i, p in enumerate(study.parameters)
                              if p.layer == layer and p.kind == "concentration"}
            if len(thickness) != 1 or set(concentrations) != set(elements):
                raise ValueError("Each layer needs thickness and every elemental fraction")
            thickness_indices.append(thickness[0])
            concentration_indices.append([concentrations[e] for e in elements])
        if len(study.parameters) != layer_count * (len(elements) + 1):
            raise ValueError("LRN_AD currently requires a fixed setup and only thickness/composition parameters")
        # Reuse the generic recurrent architecture with an internal AD layout.
        parameters = [SimpleNamespace(kind="areal_density", element=e,
                                      detector=None, layer=l, name=f"layer_{l}_{e}_ad")
                      for l in range(layer_count) for e in elements]
        layout_study = SimpleNamespace(experiment=study.experiment, parameters=parameters)
        super().__init__(layout_study, output_spectra_lengths, **architecture)
        self.study = study
        self.elements = elements
        self.architecture = dict(architecture)
        self.register_buffer("ad_input_factor", torch.tensor(float(ad_input_factor)))
        self.register_buffer("physical_thickness_indices", torch.tensor(thickness_indices))
        self.register_buffer("physical_concentration_indices", torch.tensor(concentration_indices))

    def scale_areal_density(self, densities):
        """Scale physical AD once; accepts flat or [batch, layers, elements]."""
        return densities.flatten(1) * self.ad_input_factor

    def physical_to_scaled_ad(self, parameters):
        """Convert physical thickness and valid atomic fractions to scaled AD."""
        thickness = parameters.index_select(1, self.physical_thickness_indices)
        fractions = parameters.index_select(1, self.physical_concentration_indices.flatten())
        fractions = fractions.reshape(parameters.shape[0], self.layer_count, len(self.elements))
        return self.scale_areal_density(thickness.unsqueeze(-1) * fractions)

    def predict_ad(self, densities):
        """Predict from physical AD [batch, layers, elements], in surface order."""
        if densities.ndim != 3 or densities.shape[1] < 1 or densities.shape[2] != len(self.elements):
            raise ValueError("densities must have shape [batch, layers>=1, elements]")
        if not torch.isfinite(densities).all() or (densities < 0).any():
            raise ValueError("densities must be finite and nonnegative")
        return self.predict(self.scale_areal_density(densities))

    def export(self, path, *, layer_layout="fixed", **kwargs):
        if any(kwargs.get(key) is not None for key in
               ("input_transform", "input_minimum", "input_scale")):
            raise ValueError("LRN_AD export embeds AD conversion/scaling; supply physical atomic fractions")
        if layer_layout == "dynamic":
            if type(self).encode_latent is not LRNModel.encode_latent:
                raise ValueError("Dynamic recurrent export requires the LRN_AD recurrent encoder")
            from ._lrn_ad_export import export_dynamic
            return export_dynamic(self, path, **kwargs)
        if layer_layout != "fixed":
            raise ValueError("layer_layout must be 'fixed' or 'dynamic'")
        return _PhysicalInputWrapper(self).export(path, **kwargs)


class _PhysicalInputWrapper(LRNModel):
    """Reuse package export while tracing the physical-to-AD conversion."""

    export_class_name = "LRN_AD"

    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model
        self.study = model.study
        self.input_dimension = len(model.study.parameters)
        self.output_size = model.output_size
        self.spectrum_labels = model.spectrum_labels
        self.output_spectra_lengths = model.output_spectra_lengths

    def forward(self, inputs):
        return self.model(self.model.physical_to_scaled_ad(inputs))


__all__ = ["LRN_AD"]
