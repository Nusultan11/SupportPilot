"""SupportPilot: multilingual grounded customer-support assistant."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("supportpilot")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0"
