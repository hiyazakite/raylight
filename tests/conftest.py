import os
import sys

# =============================================================================
# Path setup for tests.
#
# Ensure src/ is on sys.path so `from raylight.xxx import ...` resolves to
# the library code in src/raylight/ (not the root __init__.py which is the
# ComfyUI entry point).
# =============================================================================
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _src not in sys.path:
    sys.path.insert(0, _src)

# Add ComfyUI root so tests that need `import comfy` can find it.
_comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if os.path.isdir(os.path.join(_comfy_root, 'comfy')) and _comfy_root not in sys.path:
    sys.path.insert(0, _comfy_root)
    # Eagerly import comfy so it's registered as the real package in
    # sys.modules BEFORE any test file can install a mock.
    try:
        import comfy  # noqa: F401
    except Exception:
        pass

# Force raylight to resolve to src/raylight/ rather than the project root
# (which has __init__.py for ComfyUI). If pytest has already cached the root
# package, replace it.
import importlib
import raylight as _rl
_src_init = os.path.join(_src, 'raylight', '__init__.py')
if os.path.abspath(_rl.__file__ or '') != os.path.abspath(_src_init):
    sys.modules.pop('raylight', None)
    # Also remove any submodules cached from the wrong package
    for _k in list(sys.modules):
        if _k.startswith('raylight.'):
            sys.modules.pop(_k, None)
    import raylight  # noqa: F811 — reimport from src/
