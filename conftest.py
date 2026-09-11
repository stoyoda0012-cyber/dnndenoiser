"""Make the src/ package importable during local/CI test runs without install.

With a src/ layout, `dnndenoiser` lives under ./src. When the package is not
pip-installed (e.g. plain `pytest` from a fresh checkout), add src/ to the path
so `import dnndenoiser` resolves. An editable/real install makes this a no-op.
"""
import pathlib
import sys

_src = str(pathlib.Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
