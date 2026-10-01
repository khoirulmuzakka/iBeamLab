#pragma once

/** @file export.h @brief Shared-library visibility for the public C++ API. */

// The public API intentionally uses standard-library value types. On Windows,
// consumers must use a binary-compatible MSVC toolset and runtime. C4251 warns
// about that documented constraint rather than a defect in an individual type.
#if defined(_MSC_VER)
#pragma warning(disable : 4251)
#endif

#if defined(IBEAMLAB_STATIC_DEFINE)
#define IBEAMLAB_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#if defined(IBEAMLAB_BUILDING_LIBRARY)
#define IBEAMLAB_API __declspec(dllexport)
#else
#define IBEAMLAB_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define IBEAMLAB_API __attribute__((visibility("default")))
#else
#define IBEAMLAB_API
#endif
