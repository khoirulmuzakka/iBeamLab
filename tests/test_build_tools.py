"""Fast dependency cache and runtime packaging checks; no network or training."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import prepare_local_dependencies as dependencies
from package_gpu_runtime import package


class BuildToolsTests(unittest.TestCase):
    def sdk(self, directory, gpu=False):
        directory = Path(directory)
        (directory/"include").mkdir(parents=True)
        (directory/"include/onnxruntime_cxx_api.h").touch()
        (directory/"lib").mkdir()
        for name in ("onnxruntime.lib", "onnxruntime.dll"):
            (directory/"lib"/name).touch()
        if gpu:
            for name in ("onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll"):
                (directory/"lib"/name).touch()
        return directory

    def test_explicit_sdk_never_downloads(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(dependencies.urllib.request, "urlretrieve") as fetch:
            sdk = self.sdk(Path(temporary)/"existing")
            self.assertEqual(dependencies.prepare_sdk("CPU", sdk), sdk.resolve())
            fetch.assert_not_called()

    def test_invalid_override_fails_without_download(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(dependencies.urllib.request, "urlretrieve") as fetch:
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                dependencies.prepare_sdk("CPU", Path(temporary)/"missing")
            fetch.assert_not_called()

    def test_cpu_sdk_rejected_for_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = self.sdk(Path(temporary)/"cpu")
            with self.assertRaisesRegex(ValueError, "GPU ONNX Runtime"):
                dependencies.validate_sdk(sdk, "CUDA")

    def test_cuda_sdk_supports_both_providers(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = self.sdk(Path(temporary)/"cuda", gpu=True)
            for provider in ("CPU", "CUDA"):
                self.assertEqual(dependencies.validate_sdk(sdk, provider), sdk.resolve())

    def test_cached_sdk_never_downloads(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(dependencies, "CACHE", Path(temporary)), \
             patch.object(dependencies.urllib.request, "urlretrieve") as fetch:
            sdk = self.sdk(Path(temporary)/"onnxruntime"/dependencies.ORT_VERSION/"cpu")
            self.assertEqual(dependencies.prepare_sdk("CPU"), sdk.resolve())
            fetch.assert_not_called()

    def test_gpu_bundle_rejects_cpu_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/"share/ibeamlab/runtime-manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"provider": "CPU"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CUDA build"):
                package(root, root/"gpu.zip", [])
            self.assertFalse((root/"gpu.zip").exists())


    def installation(self, root):
        native = root/"bin"
        native.mkdir()
        for name in ("ibeamlab.dll", "onnxruntime.dll", "onnxruntime_providers_cuda.dll",
                     "onnxruntime_providers_shared.dll", "cudnn64_9.dll", "ibeamlab_runtime_probe.exe"):
            (native/name).write_bytes(b"fixture")
        share = root/"share/ibeamlab"
        share.mkdir(parents=True)
        (share/"probe.onnx").write_bytes(b"probe")
        (share/"runtime-manifest.json").write_text(json.dumps({"provider": "CUDA"}), encoding="utf-8")

    def test_failed_probe_never_publishes_bundle(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.installation(root)
            failure = SimpleNamespace(returncode=1, stdout="", stderr="missing CUDA DLL")
            with patch("package_gpu_runtime.subprocess.run", return_value=failure):
                with self.assertRaisesRegex(RuntimeError, "Isolated CPU probe failed"):
                    package(root, root/"bundle.zip", [])
            self.assertFalse((root/"bundle.zip").exists())

    def test_bundle_contains_verified_file_hashes(self):
        from types import SimpleNamespace
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.installation(root)
            success = SimpleNamespace(returncode=0, stdout=json.dumps({
                "onnxruntime_version": "test", "output_count": 1, "output_sum": 2}), stderr="")
            with patch("package_gpu_runtime.subprocess.run", return_value=success):
                output = package(root, root/"bundle.zip", [])
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                entry = manifest["files"]["ibeamlab.dll"]
                self.assertEqual(entry["sha256"], dependencies.checksum(root/"bin/ibeamlab.dll"))
                self.assertEqual(archive.read("runtime/ibeamlab.dll"), b"fixture")
                self.assertEqual(len(manifest["checks"]), 2)


if __name__ == "__main__":
    unittest.main()
