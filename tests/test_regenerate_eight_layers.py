"""The regeneration must retain chemistry while changing only discretization."""
import importlib.util
import json
from pathlib import Path

import numpy as np


def test_grid_and_profiles_match_existing_generators():
    root = Path(__file__).resolve().parents[1]
    path = root/"examples/datasets/surr_batt/script/regenerate_eight_layers.py"
    spec = importlib.util.spec_from_file_location("regenerate_eight_layers", path)
    regen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regen)
    import generate_spatial_scan_test as heldout
    edges = regen.log_grid()
    assert len(edges) == 9 and edges[0] == 0 and edges[-1] == 100000
    assert np.all(np.diff(edges) > 0)
    z = (edges[:-1]+edges[1:])/2
    source = root/"examples/datasets/surr_batt"
    for folder in sorted(source.iterdir()):
        meta = folder/"scan_metadata.json"
        if not meta.is_file(): continue
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        if "chemistry" not in metadata: continue
        # Disk centres, radial offsets and an Al holder point.
        xy = np.array([(c["center_x_mm"]+dx,c["center_y_mm"]) for c in metadata["cells"]
                       for dx in (0., .5)] + [(-regen.scan.HOLDER_WIDTH_MM/2,metadata["cells"][0]["center_y_mm"])])
        elements, truth = regen.truth_profiles(metadata, xy, edges)
        chem = metadata["chemistry"]
        if folder.name.startswith("Test_"):
            original = heldout.array_composition(xy[:,0,None], xy[:,1,None], z[None,:],
                                                 heldout.TEST_CHEMISTRIES[chem["name"]])
            labels = heldout.ELEMENTS
        else:
            original = regen.scan.array_composition(xy[:,0,None], xy[:,1,None], z[None,:])
            labels = regen.scan.ARRAY_ELEMENTS
        for j,e in enumerate(elements):
            np.testing.assert_allclose(truth[:,:,j], original[:,:,labels.index(e)], atol=2e-15)
        np.testing.assert_allclose(truth.sum(2), 1., atol=1e-14)
