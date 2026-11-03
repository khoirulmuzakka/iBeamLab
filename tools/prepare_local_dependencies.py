"""Pinned dependency preparation shared by the local build and packaging tools.

CMake only locates installed packages. Explicit SDK/package paths take precedence;
otherwise this module prepares reusable installations outside build directories.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ORT_VERSION = "1.25.1"
MINION_VERSION = "2.0.1"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".dependencies"


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sdk(root, provider):
    root = Path(root).resolve()
    headers = [root/p/"onnxruntime_cxx_api.h" for p in
               ("include", "include/onnxruntime/core/session", "build/native/include",
                "buildTransitive/native/include")]
    native = next((root/p for p in ("lib", "lib64", "runtimes/win-x64/native")
                   if (root/p/"onnxruntime.lib").is_file()), None)
    if not any(p.is_file() for p in headers) or native is None:
        raise ValueError(f"Incomplete Windows x64 ONNX Runtime SDK: {root}")
    runtime = next((root/p for p in ("bin", "lib", "runtimes/win-x64/native")
                    if (root/p/"onnxruntime.dll").is_file()), None)
    if runtime is None:
        raise ValueError(f"Missing onnxruntime.dll in {root}")
    if provider == "CUDA" and not all((runtime/name).is_file() for name in
            ("onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll")):
        raise ValueError(f"CUDA requires a GPU ONNX Runtime SDK: {root}")
    return root


def prepare_sdk(provider, explicit=None):
    if explicit:
        return validate_sdk(explicit, provider)
    root = CACHE / "onnxruntime" / ORT_VERSION / provider.lower()
    if root.exists():
        return validate_sdk(root, provider)
    package = "microsoft.ml.onnxruntime" + (".gpu.windows" if provider == "CUDA" else "")
    url = f"https://api.nuget.org/v3-flatcontainer/{package}/{ORT_VERSION}/{package}.{ORT_VERSION}.nupkg"
    print(f"Preparing ONNX Runtime {ORT_VERSION} ({provider}) in {root}", flush=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ort-", dir=root.parent) as temporary:
        archive = Path(temporary)/"sdk.zip"
        urllib.request.urlretrieve(url, archive)
        staged = Path(temporary)/"sdk"
        with zipfile.ZipFile(archive) as package_file:
            package_file.extractall(staged)
        validate_sdk(staged, provider)
        (staged/"archive.sha256").write_text(checksum(archive) + "\n", encoding="ascii")
        staged.rename(root)
    return root


def prepare_minion(config, environment):
    # CMake resolves user-provided installations normally.
    if environment.get("Minion_DIR") or environment.get("CMAKE_PREFIX_PATH"):
        return
    prefix = CACHE / "minion" / MINION_VERSION / config
    if not (prefix/"lib/cmake/Minion/MinionConfig.cmake").is_file():
        subprocess.run([sys.executable, str(ROOT/"tools/build_minion.py"),
                        "--prefix", str(prefix), "--work-dir", str(CACHE/"sources/minion"),
                        "--config", config, "--parallel", str(min(8, os.cpu_count() or 2))], check=True)
    environment["CMAKE_PREFIX_PATH"] = str(prefix)
