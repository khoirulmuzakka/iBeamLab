import json
from pathlib import Path
import torch
import pytest
from test_encoder_notebook import make_sources


def run_notebook(dataset,output,context=None,training_only=False,version=3,decoder_path=None):
    root=Path(__file__).parents[1]
    notebook=json.loads((root/'examples/train_ibeamnet.ipynb').read_text(encoding='utf8'))
    context={'__name__':'__main__'} if context is None else context
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        s=''.join(cell['source'])
        if training_only and 'weights_path, resume_path' not in s:continue
        if 'DATASET_ROOT =' in s:
            s=s.replace('configuration = ibl.load_generation_configuration',f'DATASET_ROOT=Path({str(dataset)!r})\nNmaxlayers=2\nMAX_LAYERS=2\nconfiguration = ibl.load_generation_configuration',1)
        if 'prepared_max_layers=MAX_LAYERS' in s:
            s=s.replace('EVALUATION_SAMPLES = 20_000','EVALUATION_SAMPLES = 20')
        if 'weights_path, resume_path' in s:
            a=s.index('TRAINING_SCHEDULE = [');b=s.index('# Version 1 options',a)
            s=s[:a]+"TRAINING_SCHEDULE=[dict(learning_rate=1e-4,epochs=1),dict(learning_rate=5e-5,epochs=1)]\n"+s[b:]
            marker='import importlib'
            options=(dict(cnn_channels=(4,8),kernel_size=3,attention_size=8,attention_heads=2,attention_blocks=1,dropout=0.,minimum_std=1e-5) if version in (3,4) else dict(cnn_channels=(4,8),kernel_size=3,compression_channels=4,head_hidden_sizes=(8,),decoder_hidden_size=8,decoder_layers=1,layer_embedding_size=4,dropout=0.,minimum_std=1e-5))
            if version==4:
                options['initial_density']=.01;options['presence_mapping']='sigmoid'
            overrides=f"VERSION={version}\nOUTPUT_DIR=Path({str(output)!r})\nMODEL_OPTIONS={options!r}\nBATCH_SIZE=16\nVALIDATION_BATCH_SIZE=4\nEVALUATION_SAMPLES=20\nCPU_THREADS=1\n"
            overrides += "use_reconstruction_loss=False\nRECONSTRUCTION_WEIGHT=0.0\n" if decoder_path is None else f"use_reconstruction_loss=True\nRECONSTRUCTION_WEIGHT=1.0\nLRN_AD_CHECKPOINT=Path({str(decoder_path)!r})\n"
            s=s.replace(marker,overrides+marker,1)
        exec(compile(s,'train_ibeamnet.ipynb','exec'),context)
    return context


def test_fresh_resume_and_interrupted_replay(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt,'show',lambda:plt.close('all'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    dataset,_,_=make_sources(tmp_path)
    result=run_notebook(dataset,tmp_path/'full')
    assert result['completed_epochs']==2 and result['RESUME'] is True
    assert len(result['epoch_diagnostics'])==2
    assert all(r['training_seconds']>=r['input_preparation_seconds']>=0 and r['evaluation_seconds']>=0 for r in result['epoch_diagnostics'])
    assert result['model'].version==3 and result['package_path'].exists()
    expected=torch.load(result['resume_path'],weights_only=True)
    prepared={key:result[key] for key in ('x_train','x_val','a_train','input_mean','input_std','train_eval_indices','x_train_eval')}
    run_notebook(dataset,tmp_path/'full',context=result,training_only=True)
    assert len(result['history'])==2
    assert all(result[key] is value for key,value in prepared.items())
    assert result['protected']['best_validation_nll']<=min(r[1] for r in result['history'])
    original=torch.optim.AdamW.step;calls=0
    def interrupt(optimizer,*args,**kwargs):
        nonlocal calls
        calls+=1;original(optimizer,*args,**kwargs)
        if calls==3:raise KeyboardInterrupt
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW,'step',interrupt)
        partial=run_notebook(dataset,tmp_path/'partial')
    assert torch.load(partial['resume_path'],weights_only=True)['completed_epochs']==1
    run_notebook(dataset,tmp_path/'partial',context=partial,training_only=True)
    actual=torch.load(partial['resume_path'],weights_only=True)
    for k,v in expected['model_state_dict'].items():
        torch.testing.assert_close(actual['model_state_dict'][k],v,rtol=0,atol=0)
    for k in ('loader_rng_state','noise_rng_state','torch_rng_state'):
        torch.testing.assert_close(actual[k],expected[k],rtol=0,atol=0)
    for key,state in expected['optimizer_state_dict']['state'].items():
        for name,value in state.items():
            torch.testing.assert_close(actual['optimizer_state_dict']['state'][key][name],value,rtol=0,atol=0)


def test_optional_reconstruction_and_objective_change_resume(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt,'show',lambda:plt.close('all'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    dataset,decoder,_=make_sources(tmp_path)
    context=run_notebook(dataset,tmp_path/'rec')
    previous=context['weights_path'].read_bytes()
    # Enable the new loss on a completed run: re-evaluate best, preserve source, retain latest optimizer.
    run_notebook(dataset,tmp_path/'rec',context=context,training_only=True,decoder_path=decoder)
    assert context['use_reconstruction_loss'] and context['RECONSTRUCTION_WEIGHT']==1
    assert context['completed_epochs']==2
    assert all(p.grad is None and not p.requires_grad for p in context['lrn'].parameters())
    assert context['weights_path'].with_name('ibeamnet_best_before_objective_change.pt').read_bytes()==previous
    total,nll,rec,*_=context['loss_components'](context['model'](torch.from_numpy(context['x_train'][:2])),
        torch.from_numpy(context['a_train'][:2]),torch.from_numpy(context['x_train'][:2]))
    torch.testing.assert_close(total,nll+rec)
    context['model'].zero_grad(set_to_none=True)
    rec.backward()
    assert context['model'].mean_head.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in context['lrn'].parameters())
    # Fresh end-to-end reconstruction training and native export.
    trained=run_notebook(dataset,tmp_path/'rec_fresh',decoder_path=decoder)
    assert trained['completed_epochs']==2 and trained['package_path'].is_file()


def test_v4_nll_without_decoder_and_posterior_export(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt,'show',lambda:plt.close('all'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    dataset,decoder,_=make_sources(tmp_path)
    c=run_notebook(dataset,tmp_path/'posterior',version=4)
    assert c['model'].version==4 and hasattr(c['model'],'std_head')
    assert not c['use_reconstruction_loss'] and c['RECONSTRUCTION_WEIGHT']==0 and c['lrn'] is None
    x=torch.from_numpy(c['x_train'][:2]);target=torch.from_numpy(c['a_train'][:2])
    prediction=c['model'](x)
    total,nll,rec=c['loss_components'](prediction,target,x)
    torch.testing.assert_close(total,nll);assert rec==0
    torch.testing.assert_close(prediction.mean,prediction.P*prediction.Y)
    assert c['loss_function'].include_constant
    import ibeamlab as ibl
    native=ibl.InverseModel(c['package_path'])
    assert native.metadata.inverse.uncertainty_predicted


def test_smaller_inverse_zero_pads_for_larger_frozen_decoder(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    import ibeamlab as ibl
    monkeypatch.setattr(plt,'show',lambda:plt.close('all'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    dataset,decoder,_=make_sources(tmp_path)
    saved=torch.load(decoder,weights_only=False)
    root=Path(__file__).parents[1]
    config=ibl.load_generation_configuration(root/'examples/multilayer-generation.toml')
    larger=ibl.LRN_AD(config.study(3),{'RBS':32},elements=config.elements,
        ad_input_factor=1e-5,**saved['architecture'])
    saved.update(max_layers=3,input_parameter_names=config.study(3).parameter_names,model_state_dict=larger.state_dict())
    torch.save(saved,decoder)
    c=run_notebook(dataset,tmp_path/'smaller',version=4,decoder_path=decoder)
    assert c['MAX_LAYERS']==2 and c['DECODER_MAX_LAYERS']==3
    assert c['completed_epochs']==2
    x=torch.from_numpy(c['x_train'][:2])
    prediction=c['model'](x)
    padded=torch.nn.functional.pad(prediction.mean,(0,0,0,1))
    assert padded.shape==(2,3,10) and (padded[:,2]==0).all()
    components=c['loss_components'](prediction,torch.from_numpy(c['a_train'][:2]),x)
    assert torch.isfinite(components[2])
    components[2].backward()
    assert c['model'].mean_head.weight.grad is not None
    for name,value in larger.state_dict().items():
        torch.testing.assert_close(c['lrn'].state_dict()[name],value,rtol=0,atol=0)
