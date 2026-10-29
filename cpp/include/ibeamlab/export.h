#pragma once
// STL value types cross the C++ boundary; consumers must use a compatible
// compiler and runtime. C4251 does not describe a defect under that contract.
#ifdef _MSC_VER
#pragma warning(disable : 4251)
#endif
#if defined(_WIN32) && defined(IBEAMLAB_SHARED)
#if defined(IBEAMLAB_BUILDING_LIBRARY)
#define IBEAMLAB_API __declspec(dllexport)
#else
#define IBEAMLAB_API __declspec(dllimport)
#endif
#else
#define IBEAMLAB_API
#endif
