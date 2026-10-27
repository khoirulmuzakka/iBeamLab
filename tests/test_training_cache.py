import pickle

import numpy as np
import pytest

from ibeamlab._training_cache import IndexedRows, cached_arrays, log_normalization, poisson_batches


def test_cache_reuse_and_incomplete_build(tmp_path):
    calls = []

    def build(arrays):
        calls.append(1)
        arrays['counts'][:] = np.arange(24).reshape(6, 4)

    arrays = cached_arrays(tmp_path, {'version': 1}, {'counts': (6, 4)}, build, description='test')
    again = cached_arrays(tmp_path, {'version': 1}, {'counts': (6, 4)}, build, description='test')
    assert len(calls) == 1
    assert isinstance(arrays['counts'], np.memmap)
    assert arrays['counts'].flags.writeable is False
    np.testing.assert_array_equal(again['counts'], arrays['counts'])

    def fail(arrays):
        arrays['counts'][:2] = 1
        raise RuntimeError('interrupted')

    with pytest.raises(RuntimeError, match='interrupted'):
        cached_arrays(tmp_path, {'version': 2}, {'counts': (6, 4)}, fail, description='test')
    assert len(list(tmp_path.iterdir())) == 1  # No partial cache is published or left behind.
    changed = cached_arrays(tmp_path, {'version': 2}, {'counts': (6, 4)}, build, description='test')
    assert len(calls) == 2
    np.testing.assert_array_equal(changed['counts'], arrays['counts'])


def test_normalization_and_noise_never_request_full_input():
    pytest.importorskip('torch')
    values = np.random.default_rng(7).uniform(0, 100, (37, 8)).astype(np.float32)
    values[:, -1] = 0

    class BoundedRows:
        shape = values.shape

        def __len__(self):
            return len(values)

        def __getitem__(self, selection):
            batch = values[selection]
            assert len(batch) <= 7, 'Full dataset materialization'
            return batch

    mean, std = log_normalization(BoundedRows(), batch_size=7)
    np.testing.assert_allclose(mean, np.log1p(values).mean(0), rtol=1e-6)
    np.testing.assert_allclose(std, np.maximum(np.log1p(values).std(0), 1e-3), rtol=1e-6)
    first = poisson_batches(BoundedRows(), 5, np.empty_like(values), batch_size=7)
    second = poisson_batches(BoundedRows(), 5, np.empty_like(values), batch_size=7)
    np.testing.assert_array_equal(first, second)


def test_indexed_rows_pickle_reopens_disk_instead_of_copying(tmp_path):
    path = tmp_path / 'counts.npy'
    data = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(10000, 16))
    data[:] = np.arange(16)
    data.flush()
    view = IndexedRows(data, np.array([9999, 0, 42]), scale=.1)
    payload = pickle.dumps(view)
    assert len(payload) < 4096
    restored = pickle.loads(payload)
    assert isinstance(restored.array, np.memmap)
    assert restored.array.flags.writeable is False
    np.testing.assert_array_equal(restored[:], view[:])
    np.testing.assert_array_equal(restored.subset([2, 0])[:], view[[2, 0]])


def test_streaming_dataset_matches_eager_order_across_shards(tmp_path):
    import ibeamlab as ibl
    study = ibl.GenerationStudy(ibl.Experiment(
        ibl.Sample([ibl.Layer(10, {'Si': 1})]), [ibl.Detector('RBS', ibl.Beam('H', 1000))]), [])
    study.generate(tmp_path / 'data', simulator=ibl.DummySimulator(32), samples=11, shards=3)
    dataset = ibl.open_dataset(tmp_path / 'data')
    batches = list(dataset.iter_batches(2))
    assert [len(batch) for batch in batches] == [2, 2, 2, 2, 2, 1]
    assert dataset._records is None
    streamed = [record for batch in batches for record in batch]
    assert [r.sample_index for r in streamed] == [r.sample_index for r in dataset.records]
    np.testing.assert_array_equal(np.stack([r.result['RBS'].counts for r in streamed]),
                                  dataset.spectra('RBS'))
    # Each iterator owns its own cursor, and restart/remainder behavior is stable.
    assert next(dataset.iter_batches(3))[0].sample_index == streamed[0].sample_index
    with pytest.raises(ValueError, match='positive integer'):
        list(dataset.iter_batches(0))


@pytest.mark.parametrize('threads', [1, 2, 128])
def test_native_batch_pileup_matches_scalar(threads):
    from ibeamlab.spectrum import pileup, pileup_batch
    inputs = np.random.default_rng(4).uniform(0, 100, (17, 32)).astype(np.float32)
    # Noncontiguous input exercises the NumPy boundary conversion too.
    actual = pileup_batch(inputs[:, ::2], 10., 8., .003, threads=threads)
    expected = np.array([pileup(row[::2].tolist(), 10., 8., .003) for row in inputs], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert pileup_batch(inputs[:0], 10., 8., .003, threads=threads).shape == (0, 63)
    with pytest.raises(ValueError, match='positive'):
        pileup_batch(inputs, 10., 8., .003, threads=0)
    with pytest.raises(ValueError, match='nonnegative'):
        pileup_batch(-inputs, 10., 8., .003, threads=threads)
