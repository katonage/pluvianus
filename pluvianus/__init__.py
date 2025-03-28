# pluvianus/__init__.py

try:
    from importlib.metadata import version
except ImportError:
    # For Python <3.8 fallback
    from importlib_metadata import version

__version__ = version("pluvianus")
