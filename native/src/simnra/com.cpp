#include "com.h"

#include <string>
#include <stdexcept>
#include <iostream> 
#include <sstream>

#include <unordered_map>

// DISPID cache: IDispatch*  -> (name -> DISPID)
// DISPID cache: per-thread IDispatch* -> (name -> DISPID)
using NameToID = std::unordered_map<std::wstring, DISPID>;
// thread-local cache: each COM apartment/thread has its own cache
static thread_local std::unordered_map<IDispatch*, NameToID> g_dispid_cache;

std::string w2s(const std::wstring& wstr) {
    if (wstr.empty()) return {};
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wstr.data(),
        static_cast<int>(wstr.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0) throw std::runtime_error("failed to encode UTF-16 COM text as UTF-8");
    std::string output(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wstr.data(),
        static_cast<int>(wstr.size()), output.data(), size, nullptr, nullptr);
    return output;
}


DISPID ResolveDispID(IDispatch* disp, const wchar_t* name) {
    if (!disp) {
        std::string msg = "ResolveDispID: IDispatch is null";
        fprintf(stderr, "%s\n", msg.c_str());
        throw std::runtime_error(msg);
    }
    if (!name) {
        std::string msg = "ResolveDispID: name is null";
        fprintf(stderr, "%s\n", msg.c_str());
        throw std::runtime_error(msg);
    }
    // Fast path: try to find in cache under lock
    // Fast path: try to find in thread-local cache
    auto itObj = g_dispid_cache.find(disp);
    if (itObj != g_dispid_cache.end()) {
        auto itName = itObj->second.find(name);
        if (itName != itObj->second.end()) {
            return itName->second;
        }
    }
    // Not cached: resolve via COM
    LPOLESTR namePtr = const_cast<LPOLESTR>(name);
    DISPID dispid = DISPID_UNKNOWN;
    HRESULT hr = disp->GetIDsOfNames(IID_NULL, &namePtr, 1, LOCALE_USER_DEFAULT, &dispid);
    if (FAILED(hr)) {
        std::wstring wname(name);
        std::string msg = "ResolveDispID: GetIDsOfNames failed for " + w2s(wname)+ ")";
        fprintf(stderr, "%s\n", msg.c_str());
        throw std::runtime_error(msg);
    }
    // Store in cache
    g_dispid_cache[disp].emplace(name, dispid);
    return dispid;
}


void check_hresult(HRESULT hr, const std::string& msg) {
    if (FAILED(hr)) {
        std::ostringstream detail;
        detail << msg << " (HRESULT 0x" << std::hex
               << static_cast<unsigned long>(hr) << ')';
        throw std::runtime_error(detail.str());
    }
}

VARIANT GetPropertyValue(IDispatch* disp, const wchar_t* propertyName, const std::vector<VARIANT>& args) {
    if (!disp) throw std::runtime_error("GetPropertyValue: IDispatch is null");

    DISPID dispid = ResolveDispID(disp, propertyName);

    std::vector<VARIANT> args_reverse(args.rbegin(), args.rend());
    DISPPARAMS dp{ const_cast<VARIANT*>(args_reverse.data()), nullptr,
                   static_cast<UINT>(args_reverse.size()), 0 };

    VARIANT result;
    VariantInit(&result);

    HRESULT hr = disp->Invoke(dispid, IID_NULL, LOCALE_USER_DEFAULT, DISPATCH_PROPERTYGET, &dp, &result, nullptr, nullptr);
    if (FAILED(hr)) {
        throw std::runtime_error("GetPropertyValue: Invoke PROPERTYGET failed for property " +
                                 w2s(propertyName));
    }

    return result;
}


void SetPropertyValue(IDispatch* disp, const wchar_t* propertyName, const std::vector<VARIANT>& args) {
    if (!disp) throw std::runtime_error("SetPropertyValue: IDispatch is null");
    if (args.empty()) throw std::runtime_error("SetPropertyValue: Setter requires at least one argument");

    DISPID dispid = ResolveDispID(disp, propertyName);

    DISPID namedArg = DISPID_PROPERTYPUT;
    std::vector<VARIANT> args_reverse(args.rbegin(), args.rend());
    DISPPARAMS dp{ const_cast<VARIANT*>(args_reverse.data()), &namedArg,
                   static_cast<UINT>(args_reverse.size()), 1 };

    HRESULT hr = disp->Invoke(dispid, IID_NULL, LOCALE_USER_DEFAULT, DISPATCH_PROPERTYPUT, &dp, nullptr, nullptr, nullptr);
    if (FAILED(hr)) {
        throw std::runtime_error("SetPropertyValue: Invoke PROPERTYPUT failed for property " +
                                 w2s(propertyName));
    }
}


IDispatch* CreateDispatch(const wchar_t* progID) {
    CLSID clsid;
    HRESULT hr = CLSIDFromProgID(progID, &clsid);
    if (FAILED(hr)) throw std::runtime_error("CLSIDFromProgID failed");

    IDispatch* disp = nullptr;
    hr = CoCreateInstance(clsid, NULL, CLSCTX_LOCAL_SERVER, IID_IDispatch, (void**)&disp);
    if (FAILED(hr) || !disp) throw std::runtime_error("CoCreateInstance failed");
    return disp;
}
