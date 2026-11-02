"""Install pinned Minion and Eigen packages for development/CI.

The main iBeamLab build uses find_package; this optional helper prepares ordinary
upstream installations without embedding dependency sources in iBeamLab.
"""
import argparse
from pathlib import Path
import subprocess

MINION_REVISION = "3fd46f1afc99bdd1744c3a330539783b9959af7c"
EIGEN_REVISION = "3147391d946bb4b6c68edd901f2add6ac1f31f8c"  # 3.4.0


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def source(directory, repository, revision):
    if not directory.exists():
        directory.mkdir(parents=True)
        run("git", "init", directory)
        run("git", "-C", directory, "remote", "add", "origin", repository)
        run("git", "-C", directory, "fetch", "--depth", "1", "origin", revision)
        run("git", "-C", directory, "checkout", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    expected = subprocess.check_output(["git", "-C", str(directory), "rev-parse", revision], text=True).strip()
    if actual != expected:
        raise ValueError(f"dependency checkout differs from {revision}: choose a fresh work directory")
    if not (directory / "CMakeLists.txt").is_file():
        raise ValueError(f"missing dependency sources: {directory}")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("build/minion-dependency"))
    parser.add_argument("--minion-source", type=Path)
    parser.add_argument("--eigen-source", type=Path)
    parser.add_argument("--config", default="Release", choices=("Release", "Debug"))
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    prefix, work = args.prefix.resolve(), args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    eigen = args.eigen_source.resolve() if args.eigen_source else source(
        work/"eigen-src", "https://gitlab.com/libeigen/eigen.git", EIGEN_REVISION)
    minion = args.minion_source.resolve() if args.minion_source else source(
        work/"minion-src", "https://github.com/khoirulmuzakka/Minion.git", MINION_REVISION)
    for directory in (eigen, minion):
        if not (directory/"CMakeLists.txt").is_file():
            raise ValueError(f"missing dependency sources: {directory}")
    run("cmake", "-S", eigen, "-B", work/"eigen-build",
        f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DBUILD_TESTING=OFF", "-DEIGEN_BUILD_DOC=OFF",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", f"-DCMAKE_BUILD_TYPE={args.config}")
    run("cmake", "--install", work/"eigen-build", "--config", args.config)
    run("cmake", "-S", minion, "-B", work/"minion-build",
        f"-DCMAKE_INSTALL_PREFIX={prefix}", f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DCMAKE_BUILD_TYPE={args.config}", "-DMINION_BUILD_BENCHMARK=OFF",
        "-DMINION_BUILD_PYTHON=OFF", "-DMINION_BUILD_EXAMPLES=OFF", "-DMINION_BUILD_TESTS=OFF",
        "-DMINION_FETCH_EIGEN=OFF", "-DCMAKE_DEBUG_POSTFIX=d")
    run("cmake", "--build", work/"minion-build", "--config", args.config, "--parallel", args.parallel)
    run("cmake", "--install", work/"minion-build", "--config", args.config)
    print(f"Minion installed. Configure iBeamLab with -DCMAKE_PREFIX_PATH={prefix}")


if __name__ == "__main__":
    main()
