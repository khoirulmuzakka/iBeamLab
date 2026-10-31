"""Advanced access to the compiled C++ execution engine.

Attributes are resolved lazily from the extension. Ordinary Python workflows use
the immutable descriptions and wrappers exported by ibeamlab; this namespace
provides mutable native values, reusable processing workspaces and lower-level
buffer operations. Native fields follow their C++ names, including camelCase.

Units are keV, degrees, seconds and 1e15 atoms/cm2 unless a descriptor states
otherwise. NumPy views retain native owners, while convenience vector properties
may allocate Python lists. Do not resize native storage while exported array
views are in use; keeping an owner alive does not prevent buffer relocation.
One inference model or scratch workspace must not be
called concurrently. Type and class documentation is also supplied in native.pyi.
CMake build-tree use requires its bin directory on the Python import path.
"""
from ._engine import engine


def __getattr__(name):
    if name == "export_package":
        from .export import export_package
        return export_package
    return getattr(engine(), name)
