"""Bundle a CUDA build and CUDA/cuDNN DLLs, verifying isolated CPU/GPU inference."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from prepare_local_dependencies import ROOT, checksum

GPU_PATTERNS = ("cudart*.dll", "cublas*.dll", "cudnn*.dll", "cufft*.dll", "nvrtc*.dll",
                "nvJitLink*.dll", "curand*.dll", "cusolver*.dll", "cusparse*.dll")


def package(installation, output, cuda_dirs, cpu_manifest=None):
    installation = Path(installation).resolve()
    manifest = json.loads((installation/"share/ibeamlab/runtime-manifest.json").read_text(encoding="utf-8"))
    if manifest["provider"] != "CUDA":
        raise ValueError("GPU packaging requires a CUDA build: compile.bat --release --gpu")
    if cpu_manifest and Path(cpu_manifest).is_file():
        cpu = json.loads(Path(cpu_manifest).read_text(encoding="utf-8"))
        for key in ("abi", "configuration", "architecture_bits", "onnxruntime_api"):
            if cpu[key] != manifest[key]:
                raise ValueError(f"CPU/GPU compatibility mismatch: {key}")
    sources = {}

    def add(path):
        key = path.name.lower()
        if key in sources and checksum(sources[key]) != checksum(path):
            raise ValueError(f"Conflicting DLLs named {path.name}; specify one compatible set of --cuda-dir paths")
        sources[key] = path

    for directory in (installation/"bin", installation/"ibeamlab"):
        for pattern in ("*.dll", "*.pyd"):
            for path in directory.glob(pattern):
                add(path)
    for path in (installation/"bin/ibeamlab_runtime_probe.exe", installation/"share/ibeamlab/probe.onnx"):
        if not path.is_file():
            raise FileNotFoundError(path)
        add(path)
    for directory in cuda_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for pattern in GPU_PATTERNS + ("vcruntime*.dll", "msvcp*.dll"):
            for path in directory.glob(pattern):
                add(path)
    # Include the compiler's redistributable CRT when installed with Visual Studio.
    visual_studio = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))/"Microsoft Visual Studio"
    crt_directories = sorted(visual_studio.glob("*/*/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT"), reverse=True)
    if crt_directories:
        for path in crt_directories[0].glob("*.dll"):
            if path.name.lower() not in sources:
                add(path)
    for path in Path(sys.base_prefix).glob("vcruntime*.dll"):
        if path.name.lower() not in sources:
            add(path)
    for name in ("ibeamlab.dll", "onnxruntime.dll", "onnxruntime_providers_cuda.dll",
                 "onnxruntime_providers_shared.dll", "cudnn64_9.dll"):
        if name not in sources:
            raise FileNotFoundError(f"Missing {name}; provide compatible CUDA/cuDNN DLL directories with --cuda-dir")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gpu-package-", dir=output.parent) as temporary:
        runtime = Path(temporary)/"runtime"
        runtime.mkdir()
        for name, path in sources.items():
            shutil.copy2(path, runtime/name)
        # No CUDA installation or developer PATH may supply missing runtime DLLs.
        environment = {k: v for k, v in os.environ.items()
                       if not k.upper().startswith(("CUDA", "CUDNN"))}
        windows = Path(environment.get("SystemRoot", "C:/Windows"))
        environment["PATH"] = os.pathsep.join(map(str, (runtime, windows/"System32", windows)))
        checks = []
        for provider in ("CPU", "CUDA"):
            result = subprocess.run([str(runtime/"ibeamlab_runtime_probe.exe"), str(runtime/"probe.onnx"), provider],
                cwd=runtime, env=environment, capture_output=True, text=True, timeout=90,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                raise RuntimeError(f"Isolated {provider} probe failed: {result.stdout}\n{result.stderr}")
            checks.append(json.loads(result.stdout))
        if checks[0]["output_count"] != checks[1]["output_count"] or not math.isclose(
                checks[0]["output_sum"], checks[1]["output_sum"], rel_tol=1e-4, abs_tol=1e-5):
            raise RuntimeError("CPU/CUDA probe results differ")
        manifest.update(onnxruntime_version=checks[0]["onnxruntime_version"], checks=checks,
            files={name: {"sha256": checksum(runtime/name), "size": (runtime/name).stat().st_size}
                   for name in sorted(sources)})
        archive = Path(temporary)/"bundle.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
            for path in sorted(runtime.iterdir()):
                bundle.write(path, "runtime/"+path.name)
            for path in (installation/"share/licenses").rglob("*"):
                if path.is_file():
                    bundle.write(path, "licenses/"+path.relative_to(installation/"share/licenses").as_posix())
            for path in (installation/"ibeamlab/licenses").rglob("*"):
                if path.is_file():
                    bundle.write(path, "licenses/"+path.relative_to(installation/"ibeamlab/licenses").as_posix())
        archive.replace(output)
    print(f"GPU bundle: {output}\nSHA-256: {checksum(output)}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", type=Path, default=ROOT/"dist/cuda/Release")
    parser.add_argument("--output", type=Path, default=ROOT/"dist/gpu/ibeamlab-gpu-win-x64.zip")
    parser.add_argument("--cpu-manifest", type=Path,
                        default=ROOT/"dist/cpu/Release/share/ibeamlab/runtime-manifest.json")
    parser.add_argument("--cuda-dir", type=Path, action="append", help="Repeat for CUDA and cuDNN DLL locations")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This packaging tool creates Windows x64 bundles")
    directories = args.cuda_dir
    if directories is None:
        site = Path(sysconfig.get_paths()["purelib"])
        directories = [p for p in [site/"torch/lib", *site.glob("nvidia/*/bin")] if p.is_dir()]
        for key in ("CUDA_PATH", "CUDNN_PATH"):
            if os.environ.get(key) and (Path(os.environ[key])/"bin").is_dir():
                directories.append(Path(os.environ[key])/"bin")
    package(args.installation, args.output, directories, args.cpu_manifest)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
