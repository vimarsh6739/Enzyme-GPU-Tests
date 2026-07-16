#!/usr/bin/env python3
"""Build Reactant's C++ frontend and run RSBench through its Makefile.

The default directory layout is:

  <parent>/Reactant
  <parent>/Enzyme-GPU-Tests

Reactant's Bazel workspace owns the LLVM/Clang and pinned Enzyme-JAX dependency
used by the frontend, so one Bazel build produces a mutually compatible
compiler, plugin, and libRaise shared library.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> str:
    print(f"[{cwd}] $ {shlex.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if completed.returncode != 0:
        if capture:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    return completed.stdout if capture else ""


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required program is not on PATH: {name}")


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"Missing {description}: {path}")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {description}: {path}")


def cquery_output(
    enzyme_dir: Path,
    target: str,
    bazel_options: list[str],
    *,
    preferred_suffix: str,
) -> Path:
    output = run(
        [
            "bazel",
            "cquery",
            *bazel_options,
            "--ui_event_filters=-INFO",
            "--noshow_progress",
            "--output=files",
            target,
        ],
        cwd=enzyme_dir,
        capture=True,
    )
    files = [line.strip() for line in output.splitlines() if line.strip()]
    for filename in files:
        if filename.endswith(preferred_suffix):
            path = Path(filename)
            return (path if path.is_absolute() else enzyme_dir / path).resolve()
    raise SystemExit(
        f"Bazel target {target} did not produce a file ending in "
        f"{preferred_suffix!r}. Outputs: {files}"
    )


def default_cuda_path() -> Path:
    if value := os.environ.get("CUDA_PATH"):
        return Path(value)
    versioned = Path("/usr/local/cuda-12.9")
    return versioned if versioned.is_dir() else Path("/usr/local/cuda")


def parse_args() -> argparse.Namespace:
    gpu_tests = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build the Reactant C++ frontend, run LLVM-to-MLIR/GPU raising, "
            "and build or run RSBench"
        )
    )
    parser.add_argument(
        "--reactant",
        type=Path,
        default=gpu_tests.parent / "Reactant",
        help="Reactant checkout (default: sibling of Enzyme-GPU-Tests)",
    )
    parser.add_argument(
        "--gpu-tests",
        type=Path,
        default=gpu_tests,
        help="Enzyme-GPU-Tests checkout (default: directory containing this script)",
    )
    parser.add_argument(
        "--cuda-path",
        type=Path,
        default=default_cuda_path(),
        help="CUDA toolkit used to parse the original CUDA sources",
    )
    parser.add_argument(
        "--backend",
        choices=("cuda", "rocm"),
        default="cuda",
        help="Reactant raising backend (default: cuda)",
    )
    parser.add_argument(
        "--mode",
        choices=("plugin", "embedded"),
        default="plugin",
        help="Use the Clang plugin or the custom reactant-clang binary",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel make jobs",
    )
    parser.add_argument(
        "--sm-version",
        type=str,
        default="120",
        help="CUDA architecture used while parsing the input program (default: 120)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Compile RSBench at -O3 (the crash reproducer defaults to -O0)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean RSBench before building",
    )
    parser.add_argument(
        "--no-build-tools",
        action="store_true",
        help="Reuse existing Bazel artifacts instead of building the frontend",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Build RSBench but do not execute it",
    )
    parser.add_argument(
        "--debug-reactant",
        action="store_true",
        help="Enable verbose imported/final MLIR diagnostics",
    )
    parser.add_argument(
        "--bazel-arg",
        action="append",
        default=[],
        help="Additional option passed to Bazel build/cquery (repeatable)",
    )
    parser.add_argument(
        "--make-arg",
        action="append",
        default=[],
        help="Additional make variable or option (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")

    reactant = args.reactant.resolve()
    enzyme_dir = reactant / "enzyme"
    gpu_tests = args.gpu_tests.resolve()
    rsbench_dir = gpu_tests / "RSBench"
    cuda_path = args.cuda_path.resolve()

    require_program("bazel")
    require_program("make")
    require_directory(enzyme_dir, "Reactant C++ frontend workspace")
    require_directory(rsbench_dir, "RSBench source directory")
    require_file(rsbench_dir / "Makefile", "RSBench Makefile")
    require_file(cuda_path / "include" / "cuda.h", "CUDA header")
    require_file(
        cuda_path / "nvvm" / "libdevice" / "libdevice.10.bc",
        "CUDA libdevice",
    )

    bazel_options = [
        "--experimental_repo_remote_exec",
        "-c",
        "opt",
        *args.bazel_arg,
    ]

    if args.mode == "plugin":
        build_targets = [
            "//:ClangReactantPlugin",
            "//:reactant-clang-resource",
            "@llvm-project//clang:clang",
            "@llvm-project//clang:clang-linker-wrapper",
            "@enzyme_ad//:libRaise.so",
        ]
    else:
        build_targets = [
            "//:reactant-clang",
            "//:reactant-clang-resource",
            "@llvm-project//clang:clang-linker-wrapper",
            "@enzyme_ad//:libRaise.so",
        ]

    if not args.no_build_tools:
        run(
            ["bazel", "build", *bazel_options, *build_targets],
            cwd=enzyme_dir,
        )

    lib_raise = cquery_output(
        enzyme_dir,
        "@enzyme_ad//:libRaise.so",
        bazel_options,
        preferred_suffix="/libRaise.so",
    )
    resource_header = cquery_output(
        enzyme_dir,
        "//:reactant-clang-resource",
        bazel_options,
        preferred_suffix="/include/__clang_cuda_runtime_wrapper.h",
    )
    resource_dir = resource_header.parent.parent

    if args.mode == "plugin":
        clang = cquery_output(
            enzyme_dir,
            "@llvm-project//clang:clang",
            bazel_options,
            preferred_suffix="/clang",
        )
        plugin = cquery_output(
            enzyme_dir,
            "//:ClangReactantPlugin",
            bazel_options,
            preferred_suffix=".so",
        )
        embedded_clang = "no"
    else:
        clang = cquery_output(
            enzyme_dir,
            "//:reactant-clang",
            bazel_options,
            preferred_suffix="/reactant-clang",
        )
        plugin = None
        embedded_clang = "yes"

    require_file(clang, "Reactant Clang driver")
    require_file(lib_raise, "libRaise shared library")
    require_directory(resource_dir, "Reactant Clang resource directory")
    if plugin is not None:
        require_file(plugin, "Reactant Clang plugin")

    print("\nResolved frontend artifacts:")
    print(f"  clang:        {clang}")
    print(f"  resource dir: {resource_dir}")
    print(f"  plugin:       {plugin if plugin is not None else '(embedded)'}")
    print(f"  libRaise:     {lib_raise}")
    print(f"  backend:      {args.backend}\n")

    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_path)
    env["CUDA_PATH"] = str(cuda_path)
    env["PATH"] = os.pathsep.join(
        [str(clang.parent), str(cuda_path / "bin"), env.get("PATH", "")]
    )
    env["REACTANT_PASS_TIMING"] = "1"
    if args.debug_reactant:
        env["DEBUG_REACTANT"] = "1"
    if not args.no_clean:
        run(["make", "clean"], cwd=rsbench_dir, env=env)

    clang_command = f"{clang} --driver-mode=g++ -resource-dir={resource_dir}"
    make_command = [
        "make",
        "-j",
        str(args.jobs),
        f"CUDA_PATH={cuda_path}",
        f"CLANG_PATH={clang_command}",
        f"LIB_RAISE_PATH={lib_raise}",
        f"REACTANT_BACKEND={args.backend}",
        f"EMBEDDED_CLANG={embedded_clang}",
        f"OPTIMIZE={'yes' if args.optimize else 'no'}",
        f"SM_VERSION={args.sm_version}",
    ]
    if plugin is not None:
        make_command.append(f"ENZYME_PATH={plugin}")
    make_command.extend(args.make_arg)
    run(make_command, cwd=rsbench_dir, env=env)

    if not args.no_run:
        run(["make", "run"], cwd=rsbench_dir, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
