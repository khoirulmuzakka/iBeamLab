"""Version selection, posterior heads, persistence, and native inference."""
import json,zipfile,tomllib
import numpy as np
import pytest
import torch
import ibeamlab as ibl
from ibeamlab import iBeamNet, iBeamNetLoss


def study():
    return ibl.GenerationStudy(ibl.Experiment(ibl.Sample([ibl.Layer(10,{'Si':.5,'O':.5})]*3),
        [ibl.Detector('RBS',ibl.Beam('He',2000),particles_sr=1e8)]),[])


def make_model(version):
    common=dict(cnn_channels=(4,8),kernel_size=3,dropout=0.)
    if version==1:common.update(compression_channels=4,decoder_hidden_size=8,decoder_layers=1,layer_embedding_size=4,head_hidden_sizes=(8,))
    else:common.update(attention_size=8,attention_heads=2,attention_blocks=1)
    return iBeamNet(study(),{'RBS':32},version=version,model_options=common)


@pytest.mark.parametrize('version',[1,3,4])
def test_checkpoint_prediction_and_export(tmp_path,version):
    torch.set_num_threads(1)
    model=make_model(version)
    x=torch.ones(2,32)
    prediction=model(x)
    assert prediction.Y.shape==prediction.E.shape==prediction.P.shape==(2,3,2)
    assert (prediction.Y>0).all() and (prediction.E>=model.minimum_std).all()
    target=torch.ones_like(prediction.Y);target[:,:,1]=0
    iBeamNetLoss()(prediction,target).backward()
    assert all(h.weight.grad is not None and torch.isfinite(h.weight.grad).all() for h in (model.mean_head,model.std_head,model.presence_head))
    path=model.save(tmp_path/'model.pt')
    restored=iBeamNet.load(path,study=model.study).eval()
    model.eval()
    torch.testing.assert_close(restored(x).mean,model(x).mean,rtol=0,atol=0)
    assert restored.version==version and restored.model_options==model.model_options
    package=model.export(tmp_path/'model.zip',output_inverse_factor=1e-5)
    with zipfile.ZipFile(package) as archive:
        metadata=tomllib.loads(archive.read('package.toml').decode())
        assert metadata['ibeamnet']['version']==version
        assert json.loads(metadata['ibeamnet']['model_options_json'])['cnn_channels']==[4,8]
    native=ibl.InverseModel(package)
    assert native.metadata.class_name=='iBeamNet'
    results=native.predict_prepared([{'RBS':ibl.Spectrum('RBS',row)} for row in x.numpy()])
    np.testing.assert_allclose(np.stack([r.values for r in results]),model(x).mean.detach().numpy()/1e-5,rtol=2e-4)
    np.testing.assert_allclose(np.stack([r.presence_probability for r in results]),model(x).P.detach().numpy(),rtol=2e-4)
    np.testing.assert_allclose(np.stack([r.posterior_std for r in results]),model(x).variance.sqrt().detach().numpy()/1e-5,rtol=2e-4)


def test_options_are_version_specific():
    for version in (0,2,True,'3'):
        with pytest.raises(ValueError,match='Unsupported'):iBeamNet(study(),{'RBS':32},version=version)
    with pytest.raises(ValueError,match='Unknown options'):iBeamNet(study(),{'RBS':32},version=3,model_options={'decoder_layers':2})
    with pytest.raises(ValueError,match='Reserved'):iBeamNet(study(),{'RBS':32},model_options={'elements':['Si']})


def test_v3_has_no_threshold_option():
    model=make_model(3)
    result=model(torch.ones(2,32))
    torch.testing.assert_close(result.mean,result.P*result.Y)
    assert 'discrete_presence' not in model.model_options
    with pytest.raises(ValueError,match='Unknown options'):
        iBeamNet(study(),{'RBS':32},version=3,model_options={'discrete_presence':True})


def test_v4_posterior_heads_and_exact_nll():
    model=make_model(4)
    prediction=model(torch.ones(2,32))
    assert prediction.Y.shape==prediction.R.shape==prediction.P.shape==(2,3,2)
    torch.testing.assert_close(prediction.E,prediction.R)
    torch.testing.assert_close(prediction.mean,prediction.Y*prediction.P)
    target=torch.ones_like(prediction.Y);target[:,:,1]=0
    mask=target>0
    manual=(torch.nn.functional.binary_cross_entropy_with_logits(prediction.logits,mask.float(),reduction='none')
        +torch.where(mask,prediction.R.log()+.5*((target-prediction.Y)/prediction.R).square()+.5*np.log(2*np.pi),0.)).flatten(1).sum(1).mean()
    exact=iBeamNetLoss(include_constant=True)(prediction,target)
    torch.testing.assert_close(exact,manual)
    exact.backward()
    assert all(h.weight.grad.abs().sum()>0 for h in (model.mean_head,model.std_head,model.presence_head))


def test_depth_weighted_ad_loss_normalization_padding_and_gradients():
    from ibeamlab.ibeamnet import depth_weighted_ad_mse
    target=torch.tensor([[[1.,0.],[2.,1.],[0.,0.]]],dtype=torch.float64)
    predicted=target.clone().requires_grad_()
    with torch.no_grad():predicted[0,:,0]+=1
    thickness=target.sum(-1)
    depth=thickness.cumsum(1)-.5*thickness
    weights=torch.exp(-2*depth/thickness.sum(1,keepdim=True));weights/=weights.sum(1,keepdim=True)
    expected=(weights*(predicted-target).square().mean(-1)).sum(1).mean()
    loss=depth_weighted_ad_mse(predicted,target)
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert predicted.grad[0,0,0]>predicted.grad[0,1,0]>predicted.grad[0,2,0]>0
    # Padded layers remain penalized, but deeper than every occupied midpoint.
    torch.testing.assert_close(depth_weighted_ad_mse(predicted,target,alpha=0),torch.nn.functional.mse_loss(predicted,target))
    torch.testing.assert_close(depth_weighted_ad_mse(predicted/1e5,target/1e5),loss.detach()/1e10)
    empty=torch.zeros_like(target)
    torch.testing.assert_close(depth_weighted_ad_mse(torch.ones_like(target),empty),torch.tensor(1.,dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda x:depth_weighted_ad_mse(x,target),(predicted.detach().requires_grad_(),))


def test_gaussian_depth_weighting_preserves_bce_and_scale_invariance():
    from ibeamlab.ibeamnet import iBeamNetPrediction
    target=torch.tensor([[[1.,0.],[2.,1.],[0.,0.]],[[0.,0.],[0.,0.],[0.,0.]]])
    y=torch.full_like(target,0.7,requires_grad=True)
    r=torch.full_like(target,0.8,requires_grad=True)
    logits=torch.zeros_like(target,requires_grad=True)
    prediction=iBeamNetPrediction(y,r,logits)
    mask=target>0
    thickness=target.sum(-1)
    relative=(thickness.cumsum(1)-.5*thickness)/thickness.sum(1,keepdim=True).clamp_min(1e-30)
    weights=3*torch.softmax(-2*relative,dim=1)
    torch.testing.assert_close(weights.mean(1),torch.ones(2))
    assert weights[0,0]>weights[0,1]>weights[0,2]
    gaussian=torch.where(mask,r.log()+.5*((target-y)/r).square()+.5*np.log(2*np.pi),0.)
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,mask.float(),reduction='none')
    expected=(bce+weights.unsqueeze(-1)*gaussian).flatten(1).sum(1).mean()
    loss=iBeamNetLoss(include_constant=True,gaussian_depth_alpha=2)(prediction,target)
    torch.testing.assert_close(loss,expected)
    loss.backward()
    torch.testing.assert_close(logits.grad,(.5-mask.float())/2)
    assert (y.grad[~mask]==0).all() and (r.grad[~mask]==0).all()
    assert torch.isfinite(y.grad).all() and torch.isfinite(r.grad).all()
    # Rescaling all AD leaves relative weights intact; Gaussian density changes
    # only by the analytically expected log-Jacobian on present cells.
    scale=1e4
    scaled=iBeamNetPrediction(y.detach()*scale,r.detach()*scale,logits.detach())
    actual=iBeamNetLoss(include_constant=True,gaussian_depth_alpha=2)(scaled,target*scale)
    offset=(weights.unsqueeze(-1)*mask*np.log(scale)).flatten(1).sum(1).mean()
    torch.testing.assert_close(actual,loss.detach()+offset)
    torch.testing.assert_close(iBeamNetLoss(gaussian_depth_alpha=0)(prediction,target),iBeamNetLoss()(prediction,target))
    for alpha in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):iBeamNetLoss(gaussian_depth_alpha=alpha)
