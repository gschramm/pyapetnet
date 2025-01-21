from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pymirc")
except PackageNotFoundError:
    __version__ = "unknown"
