"""Merged IntraTalker package.

The package keeps transcription factor analysis and perturbation analysis in
separate importable modules:

    import intratalkerpy.tf
    import intratalkerpy.perturbation
"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("intratalkerpy")
except PackageNotFoundError:  # pragma: no cover - local source tree import
    __version__ = "0.3.0"

__all__ = ["tf", "perturbation", "__version__"]


def __getattr__(name):
    if name in {"tf", "perturbation"}:
        from importlib import import_module

        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
