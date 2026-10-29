"""Direct access to the C++ bindings for reusable objects and lower-level workflows.

No extension is loaded until an attribute is requested. CMake build-tree use is
supported by adding its bin directory to PYTHONPATH.
"""
from ._engine import engine


def __getattr__(name):
    if name == "export_package":
        from .export import export_package
        return export_package
    return getattr(engine(), name)
