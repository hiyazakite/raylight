"""Native extension build logic for raylight.

All package metadata lives in pyproject.toml. This file ONLY handles
compilation of the GGUF fused CUDA kernels (_C_gguf extension).

The standalone allocator (raylight_alloc.so) is built via ``make build-alloc``
since it uses gcc directly and is not a Python extension.

Usage:
    make build                            # build all native extensions
    python setup.py build_ext --inplace   # build CUDA extension only
    pip install .                         # install package + attempt build
    pip install -e ".[dev]"               # editable install for development
"""

import glob
import os
import shutil

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC_DIR = os.path.join(HERE, "csrc", "quantization", "gguf")
SRC_DIR = os.path.join(HERE, "src")


# ---------------------------------------------------------------------------
# CUDA extension discovery
# ---------------------------------------------------------------------------

def _get_cuda_ext_modules():
    """Return CUDAExtension list if torch + nvcc are available, else empty."""
    # Skip compilation if a prebuilt .so already exists in the source tree
    prebuilt = glob.glob(
        os.path.join(SRC_DIR, "raylight", "expansion", "comfyui_gguf", "_C_gguf*.so")
    )
    if prebuilt:
        print(f"  Found prebuilt GGUF extension: {os.path.basename(prebuilt[0])}")
        return []

    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError:
        print(
            "NOTE: torch not found — GGUF CUDA extension will not be built.\n"
            "      Install torch first, then: make build-cuda"
        )
        return []

    return [
        CUDAExtension(
            name="raylight.expansion.comfyui_gguf._C_gguf",
            sources=[os.path.join(CSRC_DIR, "gguf_kernel.cu")],
            include_dirs=[CSRC_DIR],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--diag-suppress=20236",
                    "-Xcudafe",
                    "--diag_suppress=20236",
                ],
            },
        )
    ]


# ---------------------------------------------------------------------------
# Build command: graceful failure + post-build copy to src/
# ---------------------------------------------------------------------------

def _make_safe_build_ext():
    """Create SafeBuildExt using torch's BuildExtension as base if available."""
    try:
        from torch.utils.cpp_extension import BuildExtension
        base = BuildExtension
    except ImportError:
        base = _build_ext

    class SafeBuildExt(base):
        """Extension builder that:
        1. Logs failures instead of aborting ``pip install``
        2. Copies built .so into src/ for ``PYTHONPATH=src`` dev workflow
        """

        def run(self):
            try:
                super().run()
            except Exception as exc:
                self._log_failure(exc)
                return
            self._copy_to_src()

        def build_extension(self, ext):
            try:
                return super().build_extension(ext)
            except Exception:
                raise

        def _copy_to_src(self):
            """Copy built .so from build/lib* into src/ for dev imports."""
            for build_dir in glob.glob(os.path.join(HERE, "build", "lib*")):
                if not os.path.isdir(build_dir):
                    continue
                for so_file in glob.glob(
                    os.path.join(
                        build_dir, "raylight", "expansion", "comfyui_gguf", "_C_gguf*.so"
                    )
                ):
                    dest = os.path.join(
                        SRC_DIR, "raylight", "expansion", "comfyui_gguf"
                    )
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy2(so_file, dest)
                    print(f"  Copied {os.path.basename(so_file)} -> {dest}")

        @staticmethod
        def _log_failure(exc):
            import traceback

            log_path = os.path.join(HERE, "build_error.log")
            try:
                with open(log_path, "w") as f:
                    f.write("CUDA extension build failed:\n")
                    f.write(traceback.format_exc())
            except OSError:
                pass
            print(
                f"WARNING: CUDA build failed — falling back to pure-Python path.\n"
                f"         Details: {log_path}"
            )

    return SafeBuildExt


# ---------------------------------------------------------------------------
# Assemble and invoke setup()
# ---------------------------------------------------------------------------

ext_modules = _get_cuda_ext_modules()
cmdclass = {"build_ext": _make_safe_build_ext()} if ext_modules else {}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
