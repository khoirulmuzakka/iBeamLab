"""Configure, compile, test and install a CPU or CUDA Windows build."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
from prepare_local_dependencies import ROOT, prepare_sdk, prepare_minion


def run(command, environment):
    subprocess.run(list(map(str, command)), cwd=ROOT, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    configs = parser.add_mutually_exclusive_group()
    for name in ("release", "debug", "relwithdebinfo", "minsizerel"):
        configs.add_argument("--" + name, action="store_const", const=name, dest="config")
    parser.set_defaults(config="release")
    parser.add_argument("--gpu", action="store_true", help="Build CUDA-capable inference")
    parser.add_argument("--onnx-root", type=Path, help="Use an existing SDK without downloading")
    parser.add_argument("--parallel", type=int, default=min(8, os.cpu_count() or 2))
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Use CMake directly on non-Windows platforms")
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    for module in ("numpy", "pybind11", "pythoncom", "tqdm", "torch", "onnx", "onnxruntime"):
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise RuntimeError(f"Missing {module} in {sys.executable}; activate your build environment") from error
    config = dict(release="Release", debug="Debug", relwithdebinfo="RelWithDebInfo",
                  minsizerel="MinSizeRel")[args.config]
    provider = "CUDA" if args.gpu else "CPU"
    build = ROOT/"build"/provider.lower()/config
    environment = os.environ.copy()
    explicit = args.onnx_root or environment.get("ONNXRUNTIME_ROOT")
    # Preserve a previously selected SDK in this build, without searching other repositories.
    cache = build/"CMakeCache.txt"
    if not explicit and cache.is_file():
        explicit = next((line.split("=", 1)[1] for line in cache.read_text(encoding="utf-8").splitlines()
                         if line.startswith("ONNXRUNTIME_ROOT:PATH=") and line.split("=", 1)[1]), None)
    sdk = prepare_sdk(provider, explicit)
    prepare_minion("Debug" if config == "Debug" else "Release", environment)
    print(f"Python: {sys.executable}\nProvider: {provider}\nSDK: {sdk}\nBuild: {build}", flush=True)
    run(["cmake", "-S", ROOT, "-B", build, f"-DCMAKE_BUILD_TYPE={config}",
         f"-DPython_EXECUTABLE={sys.executable}", "-DBUILD_SHARED_LIBS=ON",
         "-DIBEAMLAB_BUILD_TESTS=ON", "-DIBEAMLAB_BUILD_PYTHON=ON",
         "-DIBEAMLAB_BUILD_EXAMPLES=ON", "-DIBEAMLAB_BUILD_BENCHMARKS=ON",
         "-DIBEAMLAB_ENABLE_SIMNRA=ON", "-DIBEAMLAB_REQUIRE_SIMNRA_TEST_RUNTIME=ON",
         f"-DIBEAMLAB_ONNX_PROVIDER={provider}", f"-DONNXRUNTIME_ROOT={sdk}"], environment)
    run(["cmake", "--build", build, "--config", config, "--parallel", args.parallel], environment)
    run(["ctest", "--test-dir", build, "-C", config, "--no-tests=error", "--output-on-failure"], environment)
    destination = ROOT/"dist"/provider.lower()/config
    run(["cmake", "--install", build, "--config", config, "--prefix", destination], environment)
    print(f"Build and tests passed. Python extension: {build / 'bin'}\nInstalled distribution: {destination}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError, FileExistsError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
