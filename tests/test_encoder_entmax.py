"""Binary entmax gradients, structural zeros, and notebook integration."""
from pathlib import Path
import numpy as np
import pytest
import torch
from ibeamlab.spectrum_encoder import binary_entmax15, binary_entmax15_loss, SpectrumEntmaxEncoder, distinct_element_penalty, sample_element_presence
from test_encoder_notebook import make_sources
import json


def test_binary_entmax_and_matching_loss():
    logits=torch.tensor([[0.,-3.],[0.,0.],[0.,.7],[0.,3.]],dtype=torch.float64,requires_grad=True)
    p=binary_entmax15(logits)
    torch.testing.assert_close(p.sum(-1),torch.ones(4,dtype=torch.float64))
    assert p[0,1]==0 and p[3,1]==1 and p[1,1]==.5
    # Interior KKT condition for 1.5-entmax.
    torch.testing.assert_close(2*(p[2,1].sqrt()-p[2,0].sqrt()), logits[2,1]-logits[2,0])
    labels=torch.tensor([1.,0.,1.,0.],dtype=torch.float64)
    loss=binary_entmax15_loss(logits,labels)
    loss.backward()
    targets=torch.stack((1-labels,labels),-1)
    torch.testing.assert_close(logits.grad,(p.detach()-targets)/4,rtol=1e-10,atol=1e-10)
    assert logits.grad[0,1]!=0 and logits.grad[3,1]!=0
    assert loss>=0 and torch.isfinite(logits.grad).all()
    assert torch.autograd.gradcheck(binary_entmax15,(torch.tensor([[.2,.7]],dtype=torch.float64,requires_grad=True),))


def test_structural_zeros_and_model_gradients():
    torch.set_num_threads(1)
    model=SpectrumEntmaxEncoder(32,6,max_layers=2,elements_per_layer=3,cnn_channels=(4,8),
        attention_size=8,attention_heads=2,attention_blocks=1,kernel_size=3)
    counts=torch.ones(4,32)
    ad,logits,p=model.forward_components(counts)
    (ad.sum()+binary_entmax15_loss(logits,torch.ones_like(p))).backward()
    assert model.presence_head.weight.grad.abs().sum()>0
    assert model.output.weight.grad.abs().sum()>0
    assert model.cnn[0].weight.grad.abs().sum()>0
    with torch.no_grad():
        model.presence_head.weight.zero_()
        model.presence_head.bias.copy_(torch.tensor([0.,-3.,0.,3.,0.,0.]))
    ad,_,p=model.forward_components(counts)
    assert (ad.reshape(4,2,3)[:,:,0]==0).all()
    assert (p[:,:,0]==0).all() and (p[:,:,1]==1).all()


def test_entmax_notebook_resume_and_native_export(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt,'show',lambda:plt.close('all'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    torch.set_num_threads(1)
    dataset,decoder,_=make_sources(tmp_path)
    original=decoder.read_bytes()
    context=run_notebook(dataset,decoder,tmp_path/'entmax')
    assert context['completed_epochs']==2 and context['y_train'].shape==(32,2,10)
    assert context['signature']['model_type']=='SpectrumEntmaxEncoder'
    assert context['package_path'].is_file()
    state=torch.load(context['resume_path'],weights_only=True)
    context['test_resume']=True
    run_notebook(dataset,decoder,tmp_path/'entmax',context=context,training_only=True)
    restored=torch.load(context['resume_path'],weights_only=True)
    for key,value in state['model_state_dict'].items():
        torch.testing.assert_close(restored['model_state_dict'][key],value,rtol=0,atol=0)
    assert decoder.read_bytes()==original and all(p.grad is None for p in context['lrn'].parameters())

    source_path=context['weights_path'];source_bytes=source_path.read_bytes()
    context['OUTPUT_DIR']=tmp_path/'missing_resume'
    with pytest.raises(FileNotFoundError,match='resume=True requires'):
        run_notebook(dataset,decoder,tmp_path/'missing_resume',context=context,training_only=True)
    context['OUTPUT_DIR']=source_path.parent
    context['test_resume']=False
    with pytest.raises(ValueError,match='choose a fresh OUTPUT_DIR'):
        run_notebook(dataset,decoder,source_path.parent,context=context,training_only=True)
    assert source_path.read_bytes()==source_bytes


def test_distinct_element_penalty_counts_elements_not_cells():
    p=torch.zeros(2,10,3,dtype=torch.float64)
    p[0,0,0]=1
    p[1,:,0]=1
    assert distinct_element_penalty(p)==1
    p[0,0,1]=1
    assert distinct_element_penalty(p)==1.5
    assert (sample_element_presence(p)[:,2]==0).all()
    uncertain=torch.full((2,10,3),.1,dtype=torch.float64,requires_grad=True)
    cost=distinct_element_penalty(uncertain)
    torch.testing.assert_close(cost,torch.tensor(3*(1-.9**10),dtype=torch.float64))
    cost.backward()
    assert torch.isfinite(uncertain.grad).all() and (uncertain.grad>0).all()


def run_notebook(dataset, decoder, output, *, context=None, training_only=False):
    root=Path(__file__).parents[1]
    notebook=json.loads((root/'examples/train_encoder_entmax.ipynb').read_text(encoding='utf8'))
    context={'__name__':'__main__','test_resume':False} if context is None else context
    for cell in notebook['cells']:
        if cell['cell_type']!='code': continue
        source=''.join(cell['source'])
        if training_only and 'weights_path = OUTPUT_DIR' not in source: continue
        if 'LRN_AD_CHECKPOINT =' in source:
            marker='if not LRN_AD_CHECKPOINT.is_file():'
            source=source.replace(marker,f"DATASET_ROOT=Path({str(dataset)!r})\nLRN_AD_CHECKPOINT=Path({str(decoder)!r})\n"+marker)
        if 'weights_path = OUTPUT_DIR' in source:
            a=source.index('TRAINING_SCHEDULE = [');b=source.index('# Build labels',a)
            source=source[:a]+"TRAINING_SCHEDULE=[dict(learning_rate=1e-4,epochs=1),dict(learning_rate=5e-5,epochs=1)]\n"+source[b:]
            marker='# Build labels'
            override=f"OUTPUT_DIR=Path({str(output)!r})\nBATCH_SIZE=16\nVALIDATION_BATCH_SIZE=4\nEVALUATION_SAMPLES=20\nresume=test_resume\nENCODER_ARCHITECTURE=dict(cnn_channels=(4,8),max_layers=MAX_LAYERS,elements_per_layer=len(lrn_checkpoint['elements']),attention_size=16,attention_heads=4,attention_blocks=2,kernel_size=3,initial_density=0.01)\n"
            source=source.replace(marker,override+marker,1)
        exec(compile(source,'train_encoder_entmax.ipynb','exec'),context)
    return context
