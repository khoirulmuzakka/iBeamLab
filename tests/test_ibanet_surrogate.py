"""Exercise the fine-tuning notebook with a small synthetic collection."""
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import ibeamlab as ibl


def test_surrogate_notebook_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    root = Path(__file__).parents[1]
    config_path = root / "examples/multilayer-generation.toml"
    configuration = ibl.load_generation_configuration(config_path)
    datasets = []
    for layers in (1, 2):
        path = tmp_path / "datasets" / f"layers_{layers:02}"
        study = configuration.study(layers)
        study.generate(path, simulator=ibl.DummySimulator(channels=32), samples=20, seed=layers)
        datasets.append(str(path.resolve()))
    study = configuration.study(2)
    architecture = dict(cnn_channels=(4, 8), compression_channels=4,
        head_hidden_sizes=(12,), decoder_hidden_size=16, decoder_layers=1,
        layer_embedding_size=4, dropout=0.0, kernel_size=3, minimum_std=1e-5)
    inverse = ibl.IBAnet(study, {"RBS":32}, elements=configuration.elements, **architecture)
    signature = dict(max_layers=2, elements=list(configuration.elements), labels=["RBS"],
        spectrum_lengths={"RBS":32}, target_scale=1e-5, architecture=architecture,
        input_noise=dict(poisson=True, validation_seed=8, test_seed=9),
        config_text=config_path.read_text(encoding="utf-8"), datasets=datasets)
    indices = np.random.default_rng(7).permutation(40)
    inverse_path = tmp_path / "inverse.pt"
    torch.save(dict(model_state_dict=inverse.state_dict(), signature=signature,
        train_indices=torch.tensor(indices[:32]), val_indices=torch.tensor(indices[32:36]),
        test_indices=torch.tensor(indices[36:])), inverse_path)
    lrn_architecture = dict(hidden_size=8, contribution_size=8, setup_embedding_dim=4,
        layer_embedding_dim=8, block_hidden_sizes=(12,), decoder_hidden_sizes=(12,),
        refiner_hidden_channels=4, refiner_kernel_size=3)
    surrogate = ibl.LRNModel(study, {"RBS":32}, **lrn_architecture)
    from ibeamlab import transforms
    scaler = transforms.build_lrn_input_transform(study).transforms[-1]
    lrn_path = tmp_path / "lrn.pt"
    torch.save(dict(model_state_dict=surrogate.state_dict(), max_layers=2, method="RBS",
        output_scale=0.01, input_parameter_names=study.parameter_names,
        input_minimum=np.asarray(scaler.minimum, dtype=np.float32),
        input_scale=np.asarray(scaler.scale, dtype=np.float32),
        layer_representation="thickness_and_concentrations"), lrn_path)
    notebook = json.loads((root / "examples/finetune_ibanet_surrogate.ipynb").read_text(encoding="utf-8"))
    context = {"__name__":"__main__"}
    monkeypatch.chdir(root)
    saved_lrn = {k:v.clone() for k,v in surrogate.state_dict().items()}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code": continue
        source = "".join(cell["source"])
        if "IBANET_CHECKPOINT =" in source:
            # Keep the user's configuration cell intact, then override paths/settings.
            marker = "for path in (IBANET_CHECKPOINT, LRN_CHECKPOINT):"
            prefix, suffix = source.split(marker, 1)
            source = prefix + f"\nDATASET_ROOT = Path({str(tmp_path / 'datasets')!r})\nIBANET_CHECKPOINT = Path({str(inverse_path)!r})\nLRN_CHECKPOINT = Path({str(lrn_path)!r})\nOUTPUT_DIR = Path({str(tmp_path / 'out')!r})\nEPOCHS = 2\nRAMP_EPOCHS = 1\nBATCH_SIZE = 16\nLRN_ARCHITECTURE = {lrn_architecture!r}\n" + marker + suffix
        if "weights_path = OUTPUT_DIR" in source:
            source = source.replace("EPOCHS = 20", "EPOCHS = 2").replace("RAMP_EPOCHS = 5", "RAMP_EPOCHS = 1").replace("BATCH_SIZE = 256", "BATCH_SIZE = 16")
            source = source.replace("OUTPUT_DIR = EXAMPLES / 'artifacts/ibanetv2_surrogate'", f"OUTPUT_DIR = Path({str(tmp_path / 'out')!r})")
        exec(compile(source, "finetune_ibanet_surrogate.ipynb", "exec"), context)
    assert len(context["history"]) == 2
    assert context["history"][0]["weight"] == 0
    assert context["history"][1]["weight"] > 0
    assert all(np.isfinite(r["score"]) for r in context["history"])
    assert context["weights_path"].exists() and context["package_path"].exists()
    assert all(p.grad is None for p in context["lrn"].parameters())
    for name, value in context["lrn"].state_dict().items():
        torch.testing.assert_close(value.cpu(), saved_lrn[name], rtol=0, atol=0)
    # Rerunning the training cell must safely reuse the output folder/checkpoint.
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "weights_path = OUTPUT_DIR" in source:
            source = source.replace("EPOCHS = 20", "EPOCHS = 2").replace("RAMP_EPOCHS = 5", "RAMP_EPOCHS = 1").replace("BATCH_SIZE = 256", "BATCH_SIZE = 16")
            source = source.replace("OUTPUT_DIR = EXAMPLES / 'artifacts/ibanetv2_surrogate'", f"OUTPUT_DIR = Path({str(tmp_path / 'out')!r})")
            exec(compile(source, "finetune rerun", "exec"), context)
    assert len(context["history"]) == 4
