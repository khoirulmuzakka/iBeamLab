"""Small end-to-end reconstruction-only encoder run and protected resume."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
import ibeamlab as ibl
from ibeamlab.spectrum_encoder import SpectrumEncoder


def make_sources(tmp_path):
    root = Path(__file__).parents[1]
    config = root/'examples/multilayer-generation.toml'
    configuration = ibl.load_generation_configuration(config)
    datasets = []
    for layers in (1, 2):
        path = tmp_path/'datasets'/f'layers_{layers:02}'
        configuration.study(layers).generate(path, simulator=ibl.DummySimulator(channels=32), samples=20, seed=layers)
        datasets.append(str(path.resolve()))
    architecture = dict(hidden_size=8, contribution_size=8, setup_embedding_dim=4,
        layer_embedding_dim=8, block_hidden_sizes=(12,), decoder_hidden_sizes=(12,),
        refiner_hidden_channels=4, refiner_kernel_size=3)
    study = configuration.study(2)
    surrogate = ibl.LRN_AD(study, {'RBS':32}, elements=configuration.elements, ad_input_factor=1e-5, **architecture)
    decoder_path = tmp_path/'decoder.pt'
    torch.save(dict(model_state_dict=surrogate.state_dict(), max_layers=2, method='RBS', output_scale=0.01,
        input_parameter_names=study.parameter_names, architecture=architecture,
        elements=list(configuration.elements), ad_input_factor=1e-5, layer_representation='areal_density'), decoder_path)
    return tmp_path/'datasets', decoder_path, surrogate


def run_notebook(dataset_root, decoder_path, output_dir, *, context=None, training_only=False):
    root = Path(__file__).parents[1]
    notebook = json.loads((root/'examples/train_encoder.ipynb').read_text(encoding='utf8'))
    context = {'__name__':'__main__'} if context is None else context
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code': continue
        source = ''.join(cell['source'])
        if training_only and 'weights_path = OUTPUT_DIR' not in source: continue
        if 'LRN_AD_CHECKPOINT =' in source:
            marker = 'if not LRN_AD_CHECKPOINT.is_file():'
            prefix, suffix = source.split(marker, 1)
            source = prefix + f"\nDATASET_ROOT=Path({str(dataset_root)!r})\nLRN_AD_CHECKPOINT=Path({str(decoder_path)!r})\n" + marker + suffix
            source += f"\nOUTPUT_DIR=Path({str(output_dir)!r})\nENCODER_ARCHITECTURE=dict(cnn_channels=(4,8),max_layers=MAX_LAYERS,elements_per_layer=len(lrn_checkpoint['elements']),attention_size=16,attention_heads=4,attention_blocks=2,kernel_size=3,initial_density=0.01)\nTRAINING_SCHEDULE=[dict(learning_rate=1e-4,epochs=1),dict(learning_rate=5e-5,epochs=1)]\nBATCH_SIZE=16\nVALIDATION_BATCH_SIZE=4\nEVALUATION_SAMPLES=20\n"
        if 'weights_path = OUTPUT_DIR' in source and 'TRAINING_SCHEDULE = [' in source:
            start = source.index('TRAINING_SCHEDULE = [')
            end = source.index('if not TRAINING_SCHEDULE', start)
            source = source[:start] + "TRAINING_SCHEDULE=[dict(learning_rate=1e-4,epochs=1),dict(learning_rate=5e-5,epochs=1)]\n" + source[end:]
        exec(compile(source, 'train_encoder.ipynb', 'exec'), context)
    return context


def test_encoder_shapes_loss_gradients_and_normalization():
    model = SpectrumEncoder(33, 100, cnn_channels=(4,8), compression_channels=4, hidden_sizes=(12,), kernel_size=3)
    output = model(torch.ones(3,33))
    assert output.shape == (3,100) and (output > 0).all()
    output.sum().backward()
    assert model.cnn[0].weight.grad.abs().sum() > 0
    assert model.output.weight.grad.abs().sum() > 0
    restored = SpectrumEncoder(33,100,**model.architecture)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(torch.ones(3,33)), output.detach())


def test_notebook_full_run_and_resume(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: plt.close('all'))
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    split, decoder, surrogate = make_sources(tmp_path)
    originals = decoder.read_bytes()
    context = run_notebook(split, decoder, tmp_path/'reference')
    assert context['completed_epochs'] == 2
    assert len(context['history']) == 2
    assert context['model'].output_size == 20
    assert 'SPLIT_CHECKPOINT' not in context
    np.testing.assert_allclose(context['input_mean'], np.log1p(context['x_train']).mean(axis=0), rtol=1e-5, atol=1e-6)
    checkpoint = torch.load(context['resume_path'], weights_only=True)
    combined = torch.cat([checkpoint[key] for key in ('train_indices','val_indices','test_indices')])
    assert torch.equal(combined.sort().values, torch.arange(40))
    assert all(p.grad is None for p in context['lrn'].parameters())
    for name, value in surrogate.state_dict().items():
        torch.testing.assert_close(context['lrn'].state_dict()[name], value, rtol=0, atol=0)
    expected = torch.load(context['resume_path'], weights_only=True)
    history = list(context['history'])
    run_notebook(split, decoder, tmp_path/'reference', context=context, training_only=True)
    assert context['history'] == history
    assert context['protected']['best_validation_reconstruction'] <= min(row['val_reconstruction'] for row in history)
    assert originals == decoder.read_bytes()

    # Interrupt a partial second epoch; exact stochastic replay must reproduce
    # the uninterrupted model, AdamW and random-generator states.
    original_step = torch.optim.AdamW.step
    calls = 0
    def interrupt(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        original_step(optimizer, *args, **kwargs)
        if calls == 3: raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, 'step', interrupt)
        interrupted = run_notebook(split, decoder, tmp_path/'interrupted')
    partial = torch.load(interrupted['resume_path'], weights_only=True)
    assert partial['completed_epochs'] == 1
    run_notebook(split, decoder, tmp_path/'interrupted', context=interrupted, training_only=True)
    actual = torch.load(interrupted['resume_path'], weights_only=True)
    assert actual['completed_epochs'] == 2
    for name, value in expected['model_state_dict'].items():
        torch.testing.assert_close(actual['model_state_dict'][name], value, rtol=0, atol=0)
    for name in ('loader_rng_state','noise_rng_state','torch_rng_state'):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    for key, state in expected['optimizer_state_dict']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(actual['optimizer_state_dict']['state'][key][name], value, rtol=0, atol=0)


@pytest.mark.parametrize('failure', [KeyboardInterrupt, RuntimeError])
def test_interruption_or_error_preserves_baseline(tmp_path, monkeypatch, failure):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: plt.close('all'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.set_num_threads(1)
    split, decoder, _ = make_sources(tmp_path)
    original_step = torch.optim.AdamW.step
    def fail_after_update(optimizer, *args, **kwargs):
        original_step(optimizer, *args, **kwargs)
        raise failure('simulated training failure')
    context = {'__name__':'__main__'}
    monkeypatch.setattr(torch.optim.AdamW, 'step', fail_after_update)
    if failure is RuntimeError:
        with pytest.raises(RuntimeError, match='simulated'):
            run_notebook(split, decoder, tmp_path/'failed', context=context)
    else:
        run_notebook(split, decoder, tmp_path/'failed', context=context)
    best = torch.load(context['weights_path'], weights_only=True)
    latest = torch.load(context['resume_path'], weights_only=True)
    assert best['best_epoch'] == 0 and latest['completed_epochs'] == 0
    for name, value in best['model_state_dict'].items():
        torch.testing.assert_close(context['model'].state_dict()[name], value, rtol=0, atol=0)


def test_gru_encoder_layer_context_gradients_and_roundtrip():
    from ibeamlab.spectrum_encoder import SpectrumGRUEncoder
    model = SpectrumGRUEncoder(33, 20, max_layers=2, elements_per_layer=10,
        cnn_channels=(4,8), compression_channels=2, gru_hidden_size=16, layer_embedding_size=4, kernel_size=3)
    counts = torch.ones(3,33)
    output = model(counts)
    assert output.shape == (3,20) and (output > 0).all()
    output.sum().backward()
    for parameter in (model.gru.weight_hh_l0, model.layer_embedding.weight, model.cnn[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    restored = SpectrumGRUEncoder(33,20,**model.architecture)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(counts), output.detach())
    before = output.detach().reshape(3,2,10)
    with torch.no_grad(): model.layer_embedding.weight[1].add_(10)
    after = model(counts).reshape(3,2,10)
    assert model.cnn(torch.ones(3,1,33)).shape == (3,2,9)
    assert any(isinstance(block, torch.nn.GroupNorm) for block in model.cnn)
    assert isinstance(model.slot_norm, torch.nn.LayerNorm)
    assert model.dropout.p == 0.0
    assert model.gru.bidirectional
    assert model.output.in_features == 32
    assert model.gru.weight_hh_l0_reverse.grad.abs().sum() > 0
    assert not torch.equal(before[:,0], after[:,0])
    assert not torch.equal(before[:,1], after[:,1])


def test_attention_query_and_spectral_gradients_roundtrip():
    from ibeamlab.spectrum_encoder import SpectrumAttentionEncoder
    model = SpectrumAttentionEncoder(33,20,max_layers=2,elements_per_layer=10,
        cnn_channels=(4,8),attention_size=16,attention_heads=4,attention_blocks=2,kernel_size=3)
    counts = torch.ones(3,33)
    result = model(counts)
    assert result.shape == (3,20) and (result > 0).all()
    result.sum().backward()
    for parameter in (model.layer_queries,model.energy_positions,model.cnn[0].weight,
                      model.blocks[0].cross_attention.key.weight,model.blocks[0].self_attention.value.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    restored = SpectrumAttentionEncoder(33,20,**model.architecture)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(counts),result.detach())
