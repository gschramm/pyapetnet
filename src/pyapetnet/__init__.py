from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyapetnet")
except PackageNotFoundError:
    __version__ = "unknown"
