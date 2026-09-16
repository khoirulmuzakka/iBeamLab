#ifndef COM_H
#define COM_H

#include <windows.h>
#include <comdef.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

DISPID ResolveDispID(IDispatch* disp, const wchar_t* name);

struct VariantWrapper {
    VARIANT v;
    VariantWrapper() { VariantInit(&v); }
    VariantWrapper(const VariantWrapper&) = delete;
    VariantWrapper& operator=(const VariantWrapper&) = delete;
    ~VariantWrapper() { VariantClear(&v); }
    operator VARIANT&() { return v; }
    VARIANT* operator&() { return &v; }
};

std::string w2s(const std::wstring& wstr);
void check_hresult(HRESULT hr, const std::string& msg);
VARIANT GetPropertyValue(IDispatch* disp, const wchar_t* propertyName, const std::vector<VARIANT>& args);
void SetPropertyValue(IDispatch* disp, const wchar_t* propertyName, const std::vector<VARIANT>& args);
IDispatch* CreateDispatch(const wchar_t* progID);

template <typename T>
VARIANT CreateVariantFrom(const T& val) {
    VARIANT v;
    VariantInit(&v);

    if constexpr (std::is_same_v<T, std::wstring>) {
        v.vt = VT_BSTR;
        v.bstrVal = SysAllocString(val.c_str());
    }
    else if constexpr (std::is_same_v<T, bool>) {
        v.vt = VT_BOOL;
        v.boolVal = val ? VARIANT_TRUE : VARIANT_FALSE;
    }
    else if constexpr (std::is_same_v<T, int>) {
        v.vt = VT_I4;
        v.intVal = val;
    }
    else if constexpr (std::is_same_v<T, double>) {
        v.vt = VT_R8;
        v.dblVal = val;
    }
    else {
        // unsupported type -> empty VARIANT
        v.vt = VT_EMPTY;
    }

    return v;
};

template<typename T>
T variantTo(const VARIANT& var) {
    if constexpr (std::is_same_v<T, VARIANT>) {
        return var; // just return as-is
    } else if constexpr (std::is_same_v<T, int>) {
        if (var.vt == VT_I4) return var.lVal;
        if (var.vt == VT_I2) return static_cast<int>(var.iVal);
        throw std::runtime_error("Variant is not convertible to int");
    } else if constexpr (std::is_same_v<T, bool>) {
        if (var.vt == VT_BOOL) return var.boolVal == VARIANT_TRUE;
        throw std::runtime_error("Variant is not convertible to bool");
    } else if constexpr (std::is_same_v<T, double>) {
        if (var.vt == VT_R8) return var.dblVal;
        if (var.vt == VT_R4) return static_cast<double>(var.fltVal);
        if (var.vt == VT_I4) return static_cast<double>(var.lVal);
        throw std::runtime_error("Variant is not convertible to double");
    } else if constexpr (std::is_same_v<T, std::wstring>) {
        if (var.vt == VT_BSTR) return std::wstring(var.bstrVal, SysStringLen(var.bstrVal));
        throw std::runtime_error("Variant is not convertible to wstring");
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (var.vt == VT_BSTR) return std::string((const char*)_bstr_t(var.bstrVal));
        throw std::runtime_error("Variant is not convertible to string");
    } else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type in variantTo<T>()");
    }
}


template<typename T>
T get(IDispatch* disp, const wchar_t* prop, const std::vector<VARIANT>& args = {}) {
    if (!disp) {
        throw std::runtime_error("COM dispatch pointer is null while getting property: " + w2s(prop));
    }
    try {
        VARIANT result = GetPropertyValue(disp, prop, args);
        return variantTo<T>(result);
    }
    catch (const std::exception& ex) {
        throw std::runtime_error("Failed to get property '" + w2s(prop) + "': " + ex.what());
    }
}

template<typename T>
void set(IDispatch* disp, const wchar_t* prop, const T& val, const std::vector<VARIANT>& extraArgs = {}) {
    if (!disp) {
        throw std::runtime_error("COM dispatch pointer is null while setting property: " + w2s(prop));
    }
    try {
        std::vector<VARIANT> args(extraArgs);
        args.push_back(CreateVariantFrom<T>(val));
        SetPropertyValue(disp, prop, args);
    }
    catch (const std::exception& ex) {
        throw std::runtime_error("Failed to set property '" + w2s(prop) + "': " + ex.what());
    }
}

template<typename T>
T invokeMethod(IDispatch* disp, const wchar_t* method, const std::vector<VARIANT>& args = {}) {
    if (!disp) throw std::runtime_error("Null IDispatch in invokeMethod");

    DISPID dispid = ResolveDispID(disp, method);

    // Copy and reverse args (COM expects right-to-left order)
    std::vector<VARIANT> revArgs(args.rbegin(), args.rend());

    DISPPARAMS dp{};
    if (!revArgs.empty()) {
        dp.cArgs = static_cast<UINT>(revArgs.size());
        dp.rgvarg = revArgs.data();
    }

    HRESULT hr;
    if constexpr (std::is_void_v<T>) {
        hr = disp->Invoke(dispid, IID_NULL, LOCALE_USER_DEFAULT, DISPATCH_METHOD,
                          &dp, nullptr, nullptr, nullptr);
        if (FAILED(hr)) {
            throw std::runtime_error("invokeMethod<void>: Invoke failed for method " + w2s(method));
        }
    } else {
        VariantWrapper result;
        hr = disp->Invoke(dispid, IID_NULL, LOCALE_USER_DEFAULT, DISPATCH_METHOD,
                          &dp, &result.v, nullptr, nullptr);
        if (FAILED(hr)) {
            throw std::runtime_error("invokeMethod: Invoke failed for method " + w2s(method));
        }
        return variantTo<T>(result.v);
    }
}




#endif
