"""Compare the iBeamLab simulator directly against setup and spectra saved in XNRA.
No new expected dataset is required. Reference projects are never modified.
"""
import argparse
import json
import sys
import time
import tempfile
import shutil
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def load_reference(path, ibl, isotope_masses=None, effective_fractions=None):
    root = ET.parse(path).getroot()
    for node in root.iter():
        node.tag = node.tag.rsplit('}', 1)[-1]
    def text(node, location, default=None):
        result = node.findtext(location)
        if result is None:
            if default is not None:
                return default
            raise ValueError(f'Missing {location} in {path}')
        return result.strip()
    saved_pileup = root.find('.//pileupcalculationdefault/pileup')
    if saved_pileup is None:
        raise ValueError('Missing pileup calculation flag')
    if 'nopu' in path.stem.lower() and saved_pileup.text.strip() != 'false':
        raise ValueError('nopu reference unexpectedly enables pileup')
    x = ibl.Experiment()
    sample = ibl.Sample()
    layers = []
    for layer_index, item in enumerate(root.findall('./sample/structure/layeredstructure/layers/layer'), 1):
        layer = ibl.Layer()
        layer.thickness = float(text(item, 'layerthickness'))
        if text(item, 'hasroughness', 'false') != 'false':
            raise ValueError('Reference loader needs an explicit roughness mapping')
        if text(item, 'porosity/hasporosity', 'false') != 'false':
            raise ValueError('Reference loader needs an explicit porosity mapping')
        species = []
        grouped = {}
        for element in item.findall('layerelements/layerelement'):
            name = text(element, 'name')
            fraction = float(text(element, 'concentration'))
            match = re.fullmatch(r'(\d+)([A-Za-z]+)', name)
            symbol = match.group(2) if match else name
            entry = grouped.setdefault(symbol, {'fraction': 0., 'isotopes': []})
            entry['fraction'] += fraction
            if match:
                entry['isotopes'].append((int(match.group(1)), fraction))
        for symbol, entry in grouped.items():
            value = ibl.Species()
            value.element = symbol
            value.concentration = entry['fraction']
            if effective_fractions is not None:
                effective = effective_fractions[layer_index, symbol]
                # SIMNRA normalizes decimal XNRA fractions while loading. Use
                # its effective binary values for a like-for-like comparison,
                # but reject any difference beyond serialization roundoff.
                roundoff = 16*np.finfo(np.float64).eps*max(abs(effective),abs(value.concentration))
                if abs(effective-value.concentration)>roundoff:
                    raise ValueError(f'Effective composition differs from XNRA: layer {layer_index}, {symbol}; XML={value.concentration!r}, COM={effective!r}')
                value.concentration = effective
            if entry['isotopes']:
                isotopes = []
                for mass, fraction in entry['isotopes']:
                    isotope = ibl.Isotope()
                    isotope.massNumber = mass
                    if isotope_masses is None or (symbol, mass) not in isotope_masses:
                        raise ValueError(f'Exact isotope mass missing for {mass}{symbol}')
                    isotope.fraction = fraction/entry['fraction'] if entry['fraction'] else float(not isotopes)
                    isotopes.append(isotope)
                value.isotopes = isotopes
            species.append(value)
        layer.species = species
        layers.append(layer)
    sample.layers = layers
    x.sample = sample
    source = root.find('./sample/spectra/spectrum')
    if source is None:
        raise ValueError('Missing reference spectrum')
    measurement = ibl.Measurement()
    measurement.label = path.stem
    beam = ibl.Beam()
    beam.particle = text(source, 'beam/beamparticle')
    beam.energy = float(text(source, 'beam/beamenergy'))
    beam.spread = float(text(source, 'beam/beamenergyspread'))
    measurement.beam = beam
    calibration = source.findall('calibrations/energycalibrations/energycalibration/calibrationparameters/calibrationparameter')
    if len(calibration) != 3:
        raise ValueError('Expected quadratic energy calibration')
    detector = ibl.Detector()
    detector.calibrationOffset, detector.calibrationLinear, detector.calibrationQuadratic = map(lambda e: float(e.text), calibration)
    detector.resolution = float(text(source, 'calibrations/detectorresolutions/detectorresolution/resolutionparameters/resolutionparameter'))
    geometry = ibl.Geometry()
    geometry.incidenceAngle = float(text(source, 'geometry/incidenceangle'))
    geometry.exitAngle = float(text(source, 'geometry/exitangle'))
    geometry.scatteringAngle = float(text(source, 'geometry/scatteringangle'))
    detector.geometry = geometry
    measurement.detector = detector
    acquisition = ibl.Acquisition()
    solid_angle = source.find('detection/detector/solidangle')
    if solid_angle.attrib.get('units') != 'msr':
        raise ValueError('Expected solid angle in msr')
    acquisition.particlesSr = float(text(source, 'beam/beamfluence')) * float(solid_angle.text) / 1000
    measurement.acquisition = acquisition  # Zero times retain reference acquisition times.
    setup = ibl.ExperimentalSetup()
    setup.measurements = [measurement]
    x.setup = setup
    x.validate()
    simulations = source.findall('.//simulation')
    total = next(node for node in simulations if text(node, 'simulationtype') == 'total')
    expected = np.fromstring(text(total, 'simpledata/y'), sep=' ')
    channels = np.fromstring(text(total, 'simpledata/x'), sep=' ')
    if not np.array_equal(channels, np.arange(len(expected))) or not len(expected):
        raise ValueError('Stored total spectrum must contain contiguous zero-based channels')
    return x, measurement, expected



def recalculate_reference(path, first_layer_scale=1.0, fast=False, geometry=None):
    """Independent COM calculation with the saved target/setup left untouched."""
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    objects = []
    # COM may retain the working directory briefly after its final Release.
    temporary = tempfile.TemporaryDirectory(prefix='ibeamlab-xnra-baseline-', ignore_cleanup_errors=True)
    try:
        copied = Path(temporary.name)/path.name
        shutil.copy2(path, copied)
        for _ in range(1024):
            app = win32com.client.DispatchEx('SIMNRA.App')
            if not app.OLEUser:
                app.OLEUser = 'iBeamLab-reference-test'
                break
            app = None
        else:
            raise RuntimeError('Cannot acquire an unclaimed SIMNRA instance')
        objects.append(app)
        for name in ['Setup', 'Target', 'Spectrum']:
            objects.append(win32com.client.DispatchEx('SIMNRA.'+name))
        app, setup, target, spectrum = objects
        if not app.Open(str(copied), -1):
            raise RuntimeError('SIMNRA could not open baseline reference')
        version = app.Version
        enabled = bool(setup.PUCalculation)
        if 'nopu' in path.stem.lower() and enabled:
            raise AssertionError('SIMNRA enabled pileup for a nopu reference')
        def get(obj, name, *arguments):
            return obj._oleobj_.Invoke(obj._oleobj_.GetIDsOfNames(name), 0,
                                      pythoncom.DISPATCH_PROPERTYGET, True, *arguments)
        isotope_masses = {}
        effective_fractions = {}
        for li in range(1, int(get(target, 'NumberOfLayers'))+1):
            for ei in range(1, int(get(target, 'NumberOfElements', li))+1):
                symbol = get(target, 'ElementName', li, ei).strip()
                effective_fractions[li, symbol] = get(target, 'ElementConcentration', li, ei)
                for ii in range(1, int(get(target, 'NumberOfIsotopes', li, ei))+1):
                    mass = get(target, 'IsotopeMass', li, ei, ii)
                    isotope_masses[symbol, round(mass)] = mass
        if first_layer_scale != 1.0:
            thickness = get(target, 'LayerThickness', 1)*first_layer_scale
            target._oleobj_.Invoke(target._oleobj_.GetIDsOfNames('LayerThickness'), 0,
                                  pythoncom.DISPATCH_PROPERTYPUT, False, 1, thickness)
        if fast:
            calculation = win32com.client.DispatchEx('SIMNRA.Calc')
            calculation.ElementSpectra = False
            objects.append(calculation)
            calculation = None
        if geometry is not None:
            setup.Alpha, setup.Theta, setup.Beta = geometry[0], geometry[2], geometry[1]
        calculate = app.CalculateSpectrumFast if fast else app.CalculateSpectrum
        if not calculate():
            raise RuntimeError('SIMNRA baseline calculation failed')
        counts = np.asarray(spectrum.GetDataArray(2), dtype=np.float64)
        return counts, version, isotope_masses, enabled, effective_fractions
    finally:
        # Explicitly drop local aliases before uninitializing their apartment.
        calculate = None
        if objects:
            try: objects[0].OLEUser = ''
            except Exception: pass
        app = setup = target = spectrum = None
        objects.clear()
        pythoncom.CoUninitialize()
        temporary.cleanup()


def padded_pair(actual, expected):
    width = max(len(actual), len(expected))
    return np.pad(actual, (0, width-len(actual))), np.pad(expected, (0, width-len(expected)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('extension_dir', type=Path)
    parser.add_argument('references', type=Path, nargs='+')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--skip-unavailable', action='store_true')
    parser.add_argument('--check-saved-spectra', action='store_true',
                        help='Fail on outdated saved curves as well as backend discrepancies')
    args = parser.parse_args()
    if args.skip_unavailable:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r'SIMNRA.App\CLSID'):
                pass
        except (ImportError, FileNotFoundError):
            print('SIMNRA COM is not registered; reference comparison skipped')
            return 77
    sys.path.insert(0, str(args.extension_dir.resolve()))
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from ibeamlab import native as ibl
    report = []
    for path in args.references:
        started = time.monotonic()
        entry = {'reference': str(path.resolve())}
        print(f'Testing {path.name}', flush=True)
        try:
            baseline, version, isotope_masses, pileup_enabled, effective_fractions = recalculate_reference(path)
            entry['pileup_enabled'] = pileup_enabled
            entry['reference_rtol']=1e-5
            entry['reference_atol']=1e-5
            experiment, measurement, expected = load_reference(path, ibl, isotope_masses, effective_fractions)
            raw_experiment, _, _ = load_reference(path, ibl, isotope_masses)
            entry['max_composition_roundtrip_error'] = max(
                abs(raw.concentration-effective.concentration)
                for raw_layer,effective_layer in zip(raw_experiment.sample.layers,experiment.sample.layers)
                for raw,effective in zip(raw_layer.species,effective_layer.species))
            method = ibl.SimnraMethod()
            method.measurement = measurement.label
            method.reference = path.resolve()
            method.particle = measurement.beam.particle
            saved_actual, saved_expected = padded_pair(baseline, expected)
            saved_matches = bool(np.allclose(saved_actual, saved_expected, rtol=1e-5, atol=1e-5))
            entry.update(simnra_version=version, saved_matches_strict=saved_matches,
                         saved_max_absolute_error=float(np.max(np.abs(saved_actual-saved_expected))),
                         saved_relative_l1_error=float(np.sum(np.abs(saved_actual-saved_expected))/max(1.,np.sum(saved_expected))))
            config = ibl.SimnraConfig()
            config.methods = [method]
            config.workers = 2
            backend = ibl.SimnraSimulator(config)
            try:
                results = backend.simulate_batch([experiment, experiment])
                if len(results) != 2:
                    raise AssertionError('Batch did not return two results')
                arrays = []
                for result in results:
                    if result.failure:
                        raise RuntimeError(result.failure.message)
                    counts = np.asarray(result.spectra[0].counts)
                    # SIMNRA may retain extra zero channels beyond stored support.
                    actual, reference = padded_pair(counts, baseline)
                    entry.update(stored_channels=len(expected), returned_channels=len(counts),
                                 max_absolute_error=float(np.max(np.abs(actual-reference))),
                                 relative_l1_error=float(np.sum(np.abs(actual-reference))/max(1.,np.sum(reference))))
                    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
                    arrays.append(counts)
                np.testing.assert_allclose(arrays[0], arrays[1], rtol=1e-10, atol=1e-10)
                # Reuse a worker session and exercise its cached target topology.
                repeated = backend.simulate(experiment)
                if repeated.failure:
                    raise RuntimeError(repeated.failure.message)
                np.testing.assert_allclose(repeated.spectra[0].counts, arrays[0], rtol=1e-10, atol=1e-10)
                # Force replacement with the exact same physical target and
                # species ordering, independent of the normal topology cache.
                config.reuse_target_topology = False
                rebuilding = ibl.SimnraSimulator(config)
                try:
                    rebuilt_results = rebuilding.simulate_batch([experiment, experiment])
                    for rebuilt in rebuilt_results:
                        if rebuilt.failure:
                            raise RuntimeError(rebuilt.failure.message)
                        a, b = padded_pair(np.asarray(rebuilt.spectra[0].counts), baseline)
                        entry['rebuild_max_absolute_error'] = float(np.max(np.abs(a-b)))
                        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
                    again = rebuilding.simulate(experiment)
                    if again.failure:
                        raise RuntimeError(again.failure.message)
                    a, b = padded_pair(np.asarray(again.spectra[0].counts), baseline)
                    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
                    entry['append_before_delete_rebuild_passed'] = True
                finally:
                    rebuilding.close()
                # Check a real parameter update, not just reference replay.
                changed, _, _ = load_reference(path, ibl, isotope_masses, effective_fractions)
                changed_sample = changed.sample
                changed_layers = changed_sample.layers
                changed_layers[0].thickness *= .9
                changed_sample.layers = changed_layers
                changed.sample = changed_sample
                independent_changed, _, _, _, _ = recalculate_reference(path, first_layer_scale=.9)
                varied = backend.simulate(changed)
                if varied.failure:
                    raise RuntimeError(varied.failure.message)
                a, b = padded_pair(np.asarray(varied.spectra[0].counts), independent_changed)
                entry['varied_thickness_max_absolute_error'] = float(np.max(np.abs(a-b)))
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
                entry['varied_thickness_passed'] = True
                entry['backend_matches_fresh_reference'] = True
                # Fast is a distinct algorithm: compare it with independent Fast
                # calculations, never weaken the standard-mode tolerance.
                independent_fast, _, _, _, _ = recalculate_reference(path, fast=True)
                config.fast_calculation = True
                fast_backend = ibl.SimnraSimulator(config)
                try:
                    for fast_result in fast_backend.simulate_batch([experiment, experiment]):
                        if fast_result.failure:
                            raise RuntimeError(fast_result.failure.message)
                        a, b = padded_pair(np.asarray(fast_result.spectra[0].counts), independent_fast)
                        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
                        entry['fast_max_absolute_error'] = float(np.max(np.abs(a-b)))
                    independent_fast_changed, _, _, _, _ = recalculate_reference(path, first_layer_scale=.9, fast=True)
                    fast_varied = fast_backend.simulate(changed)
                    if fast_varied.failure:
                        raise RuntimeError(fast_varied.failure.message)
                    a, b = padded_pair(np.asarray(fast_varied.spectra[0].counts), independent_fast_changed)
                    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
                    entry['fast_matches_independent_fast'] = True
                    a, b = padded_pair(independent_fast, baseline)
                    entry['fast_vs_standard_relative_l1'] = float(np.sum(np.abs(a-b))/max(1., np.sum(b)))
                finally:
                    fast_backend.close()
                entry['passed'] = saved_matches if args.check_saved_spectra else True
                if not saved_matches:
                    message = 'Saved spectrum differs from an untouched SIMNRA recalculation at rtol=1e-5, atol=1e-5; backend matches the fresh reference.'
                    entry['error' if args.check_saved_spectra else 'saved_snapshot_note'] = message
            finally:
                backend.close()
                backend.close()
        except Exception as error:
            entry.update(passed=False, error=str(error))
        entry['seconds'] = time.monotonic()-started
        report.append(entry)
        print(json.dumps(entry), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    return 0 if all(entry['passed'] for entry in report) else 1


if __name__ == '__main__':
    raise SystemExit(main())
