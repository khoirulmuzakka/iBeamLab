"""Independent reproduction of SIMNRA's composition round-trip sensitivity.
Run separately from other SIMNRA automation jobs; no iBeamLab extension is used.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    import pythoncom
    import win32com.client
    root = ET.parse(args.reference).getroot()
    ns = {'i': 'http://idf.schemas.itn.pt'}
    layers = root.findall('./i:sample/i:structure/i:layeredstructure/i:layers/i:layer', ns)
    if len(layers) != 1:
        raise ValueError('This focused diagnostic expects a single-layer reference')
    xml_values = [float(n.text) for n in layers[0].findall('i:layerelements/i:layerelement/i:concentration', ns)]
    pythoncom.CoInitialize()
    objects = []
    temporary = tempfile.TemporaryDirectory(prefix='ibeamlab-precision-', ignore_cleanup_errors=True)
    try:
        copied = Path(temporary.name)/args.reference.name
        shutil.copy2(args.reference, copied)
        for name in ['App', 'Target', 'Spectrum']:
            objects.append(win32com.client.Dispatch('SIMNRA.'+name))
        app, target, spectrum = objects
        def get(name, *values):
            return target._oleobj_.Invoke(target._oleobj_.GetIDsOfNames(name), 0,
                                          pythoncom.DISPATCH_PROPERTYGET, True, *values)
        def calculate():
            if not app.CalculateSpectrum():
                raise RuntimeError(app.LastMessage)
            return np.asarray(spectrum.GetDataArray(2), dtype=np.float64)
        if not app.Open(str(copied), -1):
            raise RuntimeError(app.LastMessage)
        effective = [get('ElementConcentration', 1, i) for i in range(1, int(get('NumberOfElements', 1))+1)]
        if len(effective) != len(xml_values):
            raise ValueError('This diagnostic expects natural-element XML entries')
        baseline = calculate()
        results = {}
        for label, values in [('effective_COM_values', effective), ('rounded_XML_values', xml_values)]:
            if not app.Open(str(copied), -1):
                raise RuntimeError(app.LastMessage)
            for i, value in enumerate(values, 1):
                target._oleobj_.Invoke(target._oleobj_.GetIDsOfNames('ElementConcentration'), 0,
                                      pythoncom.DISPATCH_PROPERTYPUT, False, 1, i, value)
            counts = calculate()
            width = max(len(counts), len(baseline))
            difference = np.pad(counts, (0,width-len(counts)))-np.pad(baseline, (0,width-len(baseline)))
            results[label] = float(np.max(np.abs(difference)))
        report = dict(reference=args.reference.name, simnra_version=app.Version,
                      max_concentration_roundtrip_error=max(abs(a-b) for a,b in zip(effective,xml_values)),
                      max_spectrum_error_counts=results)
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
        if results['effective_COM_values'] > 1e-10:
            raise AssertionError('Replaying the effective values changed the reference')
    finally:
        app = target = spectrum = None
        objects.clear()
        pythoncom.CoUninitialize()
        temporary.cleanup()


if __name__ == '__main__':
    main()
