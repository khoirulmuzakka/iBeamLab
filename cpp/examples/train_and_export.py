"""Small reproducible forward training example using the iBeamLab C++ library.
Run in an environment with the new package, PyTorch, ONNX and ONNX-enabled C++.
Output is a new directory; an existing directory is rejected.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

parser=argparse.ArgumentParser()
parser.add_argument('output', type=Path)
parser.add_argument('--extension-dir', type=Path)
args=parser.parse_args()
if args.extension_dir:
    sys.path.insert(0,str(args.extension_dir.resolve()))
    sys.path.insert(0,str(Path(__file__).parents[2]))
from ibeamlab import native as ibl
args.output.mkdir()  # refuse overwrites
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)
x=ibl.Experiment()
species=ibl.Species();species.element='Si';species.concentration=1
layer=ibl.Layer();layer.thickness=100;layer.species=[species]
sample=ibl.Sample();sample.layers=[layer];x.sample=sample
beam=ibl.Beam();beam.particle='H';beam.energy=2000
measurement=ibl.Measurement();measurement.label='RBS';measurement.beam=beam
setup=ibl.ExperimentalSetup();setup.measurements=[measurement];x.setup=setup
plan=ibl.GenerationPlan();plan.reference=x
plan.parameters=[ibl.Parameter('thickness',ibl.LayerThickness(0),'1e15 atoms/cm2')]
rows=np.random.default_rng(42).uniform(100,1000,(200,1))
ibl.generate(plan,rows,ibl.DummySimulator(16),args.output/'dataset',provenance='numpy-uniform;seed=42')
records=list(ibl.DatasetReader(args.output/'dataset'))
inputs=np.asarray([r.parameters for r in records],dtype=np.float32)
targets=np.asarray([r.result.spectra[0].counts for r in records],dtype=np.float32)
order=np.random.default_rng(123).permutation(len(records))
train,validation,test=order[:160],order[160:180],order[180:]
# Fit preprocessing using the training split only.
scale=float(inputs[train].std())
inputs=torch.from_numpy(inputs/scale)
targets=torch.from_numpy(targets/1000.)
class Forward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=torch.nn.Parameter(torch.zeros(1,16))
    def forward(self, values):
        return values @ torch.nn.functional.softplus(self.weights)
model=Forward()
optimizer=torch.optim.Adam(model.parameters(),lr=.05)
best=float('inf');best_state=None
for epoch in range(200):
    model.train();optimizer.zero_grad()
    loss=(model(inputs[train])-targets[train]).square().mean()
    loss.backward();optimizer.step()
    model.eval()
    with torch.no_grad(): score=float((model(inputs[validation])-targets[validation]).square().mean())
    if score<best:
        best=score;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
model.load_state_dict(best_state)
torch.save({'model_state_dict':best_state,'input_scale':scale,'seed':42},args.output/'checkpoint.pt')
forward=ibl.ForwardModelMetadata();forward.reference_setup=x.setup
schema=ibl.ParameterInputSchema();schema.reference_sample=x.sample;schema.parameters=plan.parameters;forward.input_preparation=schema
forward.output_spectra=ibl.SpectraSchema([ibl.SpectrumSpec('RBS',16)])
forward.input_transform=ibl.TransformPipeline([ibl.StandardScale([0.],[scale])])
forward.output_transform=[ibl.SpectrumTransform('RBS',ibl.TransformPipeline([ibl.MultiplyScale([1000.])]))]
metadata=ibl.ModelMetadata(forward)
metadata.provenance='example-forward;seed=42;split_seed=123;best-validation-checkpoint'
ibl.export_package(model,inputs[train[:2]],metadata,args.output/'forward.ibm')
loaded=ibl.ForwardModel(ibl.Package.open(args.output/'forward.ibm'))
with torch.no_grad(): expected=model(torch.tensor([[100./scale]],dtype=torch.float32)).numpy()[0]*1000
np.testing.assert_allclose(loaded.predict(x).spectra[0].counts,expected,rtol=1e-4,atol=1e-4)
with torch.no_grad(): test_loss=float((model(inputs[test])-targets[test]).square().mean())
print(f'Validation MSE={best:.6g}; held-out test MSE={test_loss:.6g}; exported {args.output / "forward.ibm"}')
