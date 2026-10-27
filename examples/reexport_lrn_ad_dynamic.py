"""Reexport trained LRN_AD weights with a true runtime layer loop.

Run with the Python environment containing the newly compiled native extension.
The fixed-layout package supplies the original setup and parameter metadata.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ibeamlab._native import native
from ibeamlab.lrn_ad import LRN_AD


def load_model(checkpoint_path, reference_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    package = native.model.ModelPackage.open(Path(reference_path))
    metadata = package.metadata
    forward = metadata.forward
    parameters = []
    for p in forward.input_parameters:
        target = p.target
        thickness = isinstance(target, native.generation.LayerThickness)
        parameters.append(SimpleNamespace(
            name=p.name, kind="layer_thickness" if thickness else "concentration",
            layer=target.layer, element=None if thickness else target.element, detector=None))
    config = SimpleNamespace(sample=forward.sample_template, setup=forward.setup_template,
                             parameters=forward.input_parameters)
    study = SimpleNamespace(
        parameters=parameters,
        experiment=SimpleNamespace(sample=forward.sample_template,
                                   detectors=forward.setup_template.detectors),
        _config=lambda: config)
    if [p.name for p in parameters] != list(checkpoint["input_parameter_names"]):
        raise ValueError("Reference parameter order differs from checkpoint")
    model = LRN_AD(study, {s.label: s.length for s in forward.output_spectra},
                   elements=checkpoint["elements"], ad_input_factor=checkpoint["ad_input_factor"],
                   **checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent / "artifacts/lrn_ad_rbs_8layers_13el_try2_noLiWeight"
    parser.add_argument("--checkpoint", type=Path, default=base / "lrn_ad_state.pt")
    parser.add_argument("--reference", type=Path, default=base / "lrn_ad_rbs_8layers.zip")
    parser.add_argument("--output", type=Path, default=base / "lrn_ad_rbs_dynamic_layers.zip")
    args = parser.parse_args()
    model, metadata = load_model(args.checkpoint, args.reference)
    output = metadata.output_transform
    if output.type not in ("identity", "constant_factor"):
        raise ValueError("Unsupported reference output transform")
    model.export(args.output, layer_layout="dynamic",
                 output_inverse_factor=output.factor if output.type == "constant_factor" else 1.,
                 bare_spectrum_corrections=metadata.forward.bare_spectrum_corrections,
                 Apply_pileup_on_inference=metadata.forward.Apply_pileup_on_inference,
                 pileup_fudge_factor_seconds=metadata.forward.pileup_fudge_factor_seconds)
    print(args.output)


if __name__ == "__main__":
    main()
