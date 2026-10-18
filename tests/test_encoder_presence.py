"""Presence gates and protected fine-tuning notebook integration."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ibeamlab.spectrum_encoder import SpectrumAttentionEncoder, SpectrumPresenceEncoder
from test_encoder_notebook import make_sources, run_notebook


def test_hard_sample_gates_and_straight_through_gradients():
    torch.set_num_threads(1)
    architecture = dict(max_layers=2, elements_per_layer=3, cnn_channels=(4,8),
        attention_size=8, attention_heads=2, attention_blocks=1, kernel_size=3)
    original = SpectrumAttentionEncoder(32,6,**architecture).eval()
    model = SpectrumPresenceEncoder(32,6,**architecture).eval()
    model.load_state_dict(original.state_dict(),strict=False)
    counts = torch.ones(4,32)
    torch.testing.assert_close(model(counts), original(counts), rtol=0, atol=0)
    with torch.no_grad():
        model.presence_head[-1].bias.copy_(torch.tensor([2.,-2.,2.]))
    model.train()
    ad, logits, probabilities = model.forward_components(counts)
    assert (ad.reshape(4,2,3)[:,:,1] == 0).all()
    assert (ad.reshape(4,2,3)[:,:,[0,2]] > 0).all()
    ad.sum().backward()
    assert model.presence_head[-1].bias.grad[1].abs() > 0
    assert model.output.weight.grad.abs().sum() > 0
    model.eval()
    torch.testing.assert_close(model(counts),ad.detach(),rtol=0,atol=0)


def run_finetune(dataset_root, decoder_path, base_path, output_dir, context=None, training_only=False, collapse_epoch=False):
    root = Path(__file__).parents[1]
    notebook = json.loads((root/'examples/train_encoder_finetune.ipynb').read_text(encoding='utf8'))
    context = {'__name__':'__main__'} if context is None else context
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code': continue
        source = ''.join(cell['source'])
        if training_only and 'weights_path = OUTPUT_DIR' not in source: continue
        if 'LRN_AD_CHECKPOINT =' in source:
            source = source.replace("LRN_AD_CHECKPOINT = EXAMPLES / 'artifacts/lrn_ad_rbs_10layers/lrn_ad_state.pt'",f"LRN_AD_CHECKPOINT = Path({str(decoder_path)!r})")
            source = source.replace("BASE_ENCODER_CHECKPOINT = EXAMPLES/'artifacts/encoder_attention_lrn_ad/encoder_best.pt'",f"BASE_ENCODER_CHECKPOINT = Path({str(base_path)!r})")
            source += f"\nDATASET_ROOT=Path({str(dataset_root)!r})\nOUTPUT_DIR=Path({str(output_dir)!r})\nBATCH_SIZE=16\nVALIDATION_BATCH_SIZE=4\nEVALUATION_SAMPLES=20\n"
        if 'TRAINING_SCHEDULE = [' in source:
            a=source.index('TRAINING_SCHEDULE = ['); b=source.index('if not math.isfinite(PRESENCE_WEIGHT)',a)
            source=source[:a]+"TRAINING_SCHEDULE=[dict(learning_rate=1e-3,epochs=1,head_only=True),dict(learning_rate=1e-5,epochs=1,head_only=False)]\n"+source[b:]
        if collapse_epoch and 'weights_path = OUTPUT_DIR' in source:
            source=source.replace('validation_score = evaluate_validation()', 'validation_score = evaluate_validation()*1e9')
        exec(compile(source,'train_encoder_finetune.ipynb','exec'),context)
    return context


def test_presence_notebook_resume_export_and_source_protection(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt,'show',lambda:plt.close('all'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    torch.set_num_threads(1)
    dataset,decoder,surrogate=make_sources(tmp_path)
    base=run_notebook(dataset,decoder,tmp_path/'base')
    path=base['weights_path']; original=path.read_bytes(); original_decoder=decoder.read_bytes()
    result=run_finetune(dataset,decoder,path,tmp_path/'finetune')
    assert result['completed_epochs']==2
    assert result['history'][0]['head_only'] and not result['history'][1]['head_only']
    assert all(p.grad is None for p in result['lrn'].parameters())
    assert result['protected']['best_validation_reconstruction'] <= base['protected']['best_validation_reconstruction']+1e-8
    np.testing.assert_array_equal(result['train_indices'],base['train_indices'])
    assert result['y_train'].shape==(len(result['x_train']),10)
    latest=torch.load(result['resume_path'],weights_only=True)
    run_finetune(dataset,decoder,path,tmp_path/'finetune',context=result,training_only=True)
    assert result['completed_epochs']==2 and len(result['history'])==2
    assert path.read_bytes()==original and decoder.read_bytes()==original_decoder
    restored=torch.load(result['resume_path'],weights_only=True)
    for name,value in latest['model_state_dict'].items():
        torch.testing.assert_close(restored['model_state_dict'][name],value,rtol=0,atol=0)
    assert result['package_path'].is_file()

    # Interrupt the first joint update and replay from the completed warm-up epoch.
    original_step = torch.optim.AdamW.step
    calls = 0
    def interrupt(optimizer,*args,**kwargs):
        nonlocal calls
        calls += 1
        original_step(optimizer,*args,**kwargs)
        if calls == 3: raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW,'step',interrupt)
        interrupted=run_finetune(dataset,decoder,path,tmp_path/'interrupted')
    assert torch.load(interrupted['resume_path'],weights_only=True)['completed_epochs']==1
    run_finetune(dataset,decoder,path,tmp_path/'interrupted',context=interrupted,training_only=True)
    replay=torch.load(interrupted['resume_path'],weights_only=True)
    for name,value in latest['model_state_dict'].items():
        torch.testing.assert_close(replay['model_state_dict'][name],value,rtol=0,atol=0)
    for name in ('loader_rng_state','noise_rng_state','torch_rng_state'):
        torch.testing.assert_close(replay[name],latest[name],rtol=0,atol=0)

    # A collapsing first epoch must not replace either protected baseline or latest.
    failed={"__name__":"__main__"}
    with pytest.raises(RuntimeError,match='Hard-gate reconstruction collapsed'):
        run_finetune(dataset,decoder,path,tmp_path/'collapsed',context=failed,collapse_epoch=True)
    safe_best=torch.load(failed['weights_path'],weights_only=True)
    safe_resume=torch.load(failed['resume_path'],weights_only=True)
    assert safe_best['best_epoch']==0 and safe_resume['completed_epochs']==0
    assert safe_best['best_validation_reconstruction'] <= base['protected']['best_validation_reconstruction']+1e-8
    for name,value in safe_best['model_state_dict'].items():
        torch.testing.assert_close(failed['model'].state_dict()[name].cpu(),value,rtol=0,atol=0)
        torch.testing.assert_close(safe_resume['model_state_dict'][name],value,rtol=0,atol=0)
