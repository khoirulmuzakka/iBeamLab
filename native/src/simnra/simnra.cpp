#include "simnra.h"
#include <limits>
SIMNRA::SIMNRA(bool mta, int threadPriorityIndex) {
    HRESULT hr;

    // Initialize COM with requested threading model
    if (mta) {
        hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    } else {
        hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    }

    if (hr == RPC_E_CHANGED_MODE) {
        std::wcout << "Warning : COM already initialized with a different threading model\n";
    } else if (FAILED(hr)) {
        throw std::runtime_error("CoInitializeEx failed");
    } else {
        m_comInitialized = true;
    }
    
    // Map threadPriorityIndex to Windows priorities
    int winPriority = THREAD_PRIORITY_NORMAL;
    switch(threadPriorityIndex) {
        case -3: winPriority = THREAD_PRIORITY_IDLE; break;
        case -2: winPriority = THREAD_PRIORITY_LOWEST; break;
        case -1: winPriority = THREAD_PRIORITY_BELOW_NORMAL; break;
        case 0:  winPriority = THREAD_PRIORITY_NORMAL; break;
        case 1:  winPriority = THREAD_PRIORITY_ABOVE_NORMAL; break;
        case 2:  winPriority = THREAD_PRIORITY_HIGHEST; break;
        case 3:  winPriority = THREAD_PRIORITY_TIME_CRITICAL; break;
        default: winPriority = THREAD_PRIORITY_NORMAL; break;
    }

    if (!SetThreadPriority(GetCurrentThread(), winPriority)) {
        std::wcout << "[WARNING] Failed to set thread priority. Error code: " << GetLastError() << std::endl;
    }

    try {
        // Create main App instance
        m_App = CreateDispatch(L"SIMNRA.App");

        // Create sub-interfaces
        m_Setup      = CreateDispatch(L"SIMNRA.Setup");
        m_Target     = CreateDispatch(L"SIMNRA.Target");
        m_Calc       = CreateDispatch(L"SIMNRA.Calc");
        m_Fit        = CreateDispatch(L"SIMNRA.Fit");
        m_Projectile = CreateDispatch(L"SIMNRA.Projectile");
        m_Spectrum   = CreateDispatch(L"SIMNRA.Spectrum");
        m_Stopping   = CreateDispatch(L"SIMNRA.Stopping");
        m_PIGE       = CreateDispatch(L"SIMNRA.PIGE");
        m_CrossSec   = CreateDispatch(L"SIMNRA.CrossSec");

        // Verify all pointers
        if (!m_App || !m_Setup || !m_Target || !m_Calc || !m_Fit || !m_Projectile ||
            !m_Spectrum || !m_Stopping || !m_PIGE || !m_CrossSec)
        {
            throw std::runtime_error("Failed to create one or more sub-interfaces");
        }
    } catch (...) {
        // Release already created interfaces
        if (m_App) m_App->Release();
        if (m_Setup) m_Setup->Release();
        if (m_Target) m_Target->Release();
        if (m_Calc) m_Calc->Release();
        if (m_Fit) m_Fit->Release();
        if (m_Projectile) m_Projectile->Release();
        if (m_Spectrum) m_Spectrum->Release();
        if (m_Stopping) m_Stopping->Release();
        if (m_PIGE) m_PIGE->Release();
        if (m_CrossSec) m_CrossSec->Release();

        if (m_comInitialized) {
            CoUninitialize();
            m_comInitialized = false;
        }
        throw;
    }
}

SIMNRA::~SIMNRA() {
    // Release all interfaces
    if (m_App) m_App->Release();
    if (m_Setup) m_Setup->Release();
    if (m_Target) m_Target->Release();
    if (m_Calc) m_Calc->Release();
    if (m_Fit) m_Fit->Release();
    if (m_Projectile) m_Projectile->Release();
    if (m_Spectrum) m_Spectrum->Release();
    if (m_Stopping) m_Stopping->Release();
    if (m_PIGE) m_PIGE->Release();
    if (m_CrossSec) m_CrossSec->Release();

    if (m_comInitialized)
        CoUninitialize(); // Balance this instance's successful CoInitializeEx call.
}

void SIMNRA::pumpMessages() {
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}


//---------------------- App --------------------------------//
std::wstring SIMNRA::getLastMessage() { return get<std::wstring>(m_App, L"LastMessage"); };

bool SIMNRA::open(const wchar_t* filename, int type) {
    bool ok = invokeMethod<bool>(m_App, L"Open", { CreateVariantFrom<std::wstring>(std::wstring(filename)), CreateVariantFrom<int>(type) });
    pumpMessages();
    if (!ok) {
        std::wstring msg = getLastMessage();
        throw std::runtime_error("SIMNRA::open failed: " + w2s(msg));
    }
    return true;
}

bool SIMNRA::saveAs(const wchar_t* filename, int type) {
    bool ok = invokeMethod<bool>(m_App, L"SaveAs", { CreateVariantFrom<std::wstring>(std::wstring(filename)), CreateVariantFrom<int>(type) });
    pumpMessages();
    if (!ok) {
        std::wstring msg = getLastMessage();
        throw std::runtime_error("SIMNRA::SaveAs failed: " + w2s(msg));
    }
    return true;
}

bool SIMNRA::readSpectrumData(const std::wstring& file, int index) {
    return invokeMethod<bool>(m_App, L"ReadSpectrumData",
        { CreateVariantFrom<std::wstring>(file), CreateVariantFrom<int>(index) });
}

std::wstring SIMNRA::getVersion() {
    return get<std::wstring>(m_App, L"version");
}

bool SIMNRA::calculateSpectrum() {
    bool ok = invokeMethod<bool>(m_App, L"CalculateSpectrum");
    pumpMessages();
    if (!ok) {
        std::wstring msg = getLastMessage();
        throw std::runtime_error("CalculateSpectrum failed: " + w2s(msg));
    }
    return true;
}

bool SIMNRA::calculateSpectrumFast() {
    bool ok = invokeMethod<bool>(m_App, L"CalculateSpectrumFast");
    pumpMessages();
    if (!ok) {
        std::wstring msg = getLastMessage();
        throw std::runtime_error("CalculateSpectrumFast failed: " + w2s(msg));
    }
    return true;
}

std::vector<double>& SIMNRA::getSpectrum( int spid) {
    if (!m_Spectrum) {
        throw std::runtime_error("Spectrum object is null");
    }

    // Call spectrum.GetDataArray(spid)
    DISPID dispid;
    const wchar_t* nameW = L"GetDataArray";
    OLECHAR* name = const_cast<wchar_t*>(nameW);
    HRESULT hr = m_Spectrum->GetIDsOfNames(IID_NULL, &name, 1, LOCALE_USER_DEFAULT, &dispid);
    if (FAILED(hr)) throw std::runtime_error("GetIDsOfNames failed for GetDataArray");

    VARIANTARG args[1];
    VariantInit(&args[0]);
    args[0].vt = VT_I4;
    args[0].lVal = spid;

    DISPPARAMS dp = { args, nullptr, 1, 0 };
    VARIANT result;
    VariantInit(&result);

    hr = m_Spectrum->Invoke(dispid, IID_NULL, LOCALE_USER_DEFAULT, DISPATCH_METHOD, &dp, &result, nullptr, nullptr);
    if (FAILED(hr)) throw std::runtime_error("Invoke(GetDataArray) failed");

    if (result.vt != (VT_ARRAY | VT_VARIANT) && result.vt != (VT_ARRAY | VT_R8)) {
        std::wcerr << L"[ERROR] Unexpected VARIANT type: " << result.vt << std::endl;
        VariantClear(&result);
        throw std::runtime_error("GetDataArray did not return SAFEARRAY of doubles/variants");
    }

    SAFEARRAY* psa = result.parray;
    if (!psa) {
        VariantClear(&result);
        throw std::runtime_error("SAFEARRAY is null");
    }

    LONG lBound, uBound;
    SafeArrayGetLBound(psa, 1, &lBound);
    SafeArrayGetUBound(psa, 1, &uBound);
    LONG count = uBound - lBound + 1;
    //VARTYPE vt;
    //SafeArrayGetVartype(psa, &vt);

    //std::wcout << L"[DEBUG] LBound=" << lBound << L", UBound=" << uBound
    //           << L", Count=" << count << L", VARTYPE=" << vt << std::endl;

    std::vector<double>& data = m_lastSpectrum;
    data.resize(count);
    for (LONG i = lBound; i <= uBound; ++i) {
        double d = 0.0;
        LONG idx = i;
        hr = SafeArrayGetElement(psa, &idx, &d);
        if (FAILED(hr)) {
            std::wcerr << L"[ERROR] SafeArrayGetElement(double) failed at index " << idx << std::endl;
            throw std::runtime_error("SafeArrayGetElement(double) failed");
        }
        data[i - lBound] = d;
    }
    VariantClear(&result);
    return data;
}

bool SIMNRA::setSpectrum(int spid, const std::vector<double>& data) {
    if (!m_Spectrum) {
        throw std::runtime_error("Spectrum object is null");
    }
    if (data.empty()) {
        return false;
    }

    SAFEARRAYBOUND bound;
    bound.lLbound = 0;
    bound.cElements = static_cast<ULONG>(data.size());
    SAFEARRAY* psa = SafeArrayCreate(VT_R8, 1, &bound);
    if (!psa) {
        throw std::runtime_error("SafeArrayCreate failed");
    }

    double* dst = nullptr;
    HRESULT hr = SafeArrayAccessData(psa, reinterpret_cast<void**>(&dst));
    if (FAILED(hr)) {
        SafeArrayDestroy(psa);
        throw std::runtime_error("SafeArrayAccessData failed");
    }
    std::copy(data.begin(), data.end(), dst);
    SafeArrayUnaccessData(psa);

    VariantWrapper arrVar;
    arrVar.v.vt = VT_ARRAY | VT_R8;
    arrVar.v.parray = psa;

    bool ok = invokeMethod<bool>(m_Spectrum, L"SetDataArray",
                                 { CreateVariantFrom<int>(spid), arrVar.v });
    pumpMessages();
    if (!ok) {
        std::wstring msg = getLastMessage();
        throw std::runtime_error("SetDataArray failed: " + w2s(msg));
    }
    return true;
}

int SIMNRA::spectrumIDOfElement(int Z) {
    return invokeMethod<int>(m_Spectrum, L"IDOfElement",
        { CreateVariantFrom<int>(Z) });
}

// ---------------------- Setup, Calibration ----------------------- //
void SIMNRA::setBeamSpread(double fwhm) { set(m_Setup, L"Beamspread", fwhm); }
void SIMNRA::setBeamEnergy(double E) { set(m_Setup, L"Energy", E); }
void SIMNRA::setCalibrationLinear(double val) {set<double>(m_Setup, L"CalibrationLinear", val);}
void SIMNRA::setCalibrationOffset(double val) { set<double>(m_Setup, L"CalibrationOffset", val);}
void SIMNRA::setCalibrationQuadratic(double val) {set<double>(m_Setup, L"CalibrationQuadratic", val);}
void SIMNRA::setParticlesSr(double val) { set<double>(m_Setup, L"ParticlesSr", val);}
void SIMNRA::setDetectorResolution (double val) { set<double>(m_Setup, L"DetectorResolution", val);};
void SIMNRA::setRealTime(double val) { set<double>(m_Setup, L"RealTime", val);}
void SIMNRA::setLiveTime(double val) { set<double>(m_Setup, L"LiveTime", val);}
void SIMNRA::setPileUpCalculation(bool val){ set<bool>(m_Setup, L"PUCalculation", val);}
void SIMNRA::setLiveTimeCorrection(bool val){ set<bool>(m_Setup, L"LTCorrection", val);}

double SIMNRA::getBeamSpread() { return get<double>(m_Setup, L"Beamspread"); }
double SIMNRA::getBeamEnergy() { return get<double>(m_Setup, L"Energy"); }
double SIMNRA::getCalibrationLinear() { return get<double>(m_Setup, L"CalibrationLinear");}
double SIMNRA::getCalibrationOffset() {return get<double>(m_Setup, L"CalibrationOffset");}
double SIMNRA::getCalibrationQuadratic() { return get<double>(m_Setup, L"CalibrationQuadratic");}
double SIMNRA::getParticlesSr() {return get<double>(m_Setup, L"ParticlesSr");}
double SIMNRA::getDetectorResolution () { return get<double>(m_Setup, L"DetectorResolution");};
double SIMNRA::getRealTime() { return get<double>(m_Setup, L"RealTime");}
double SIMNRA::getLiveTime() { return get<double>(m_Setup, L"LiveTime");}
bool SIMNRA::getPileUpCalculation(){ return get<bool>(m_Setup, L"PUCalculation");}
bool SIMNRA::getLiveTimeCorrection(){ return get<bool>(m_Setup, L"LTCorrection");}

// ---------------------- Target, Layer ----------------------------//
bool SIMNRA::addLayer() { return invokeMethod<bool>(m_Target, L"AddLayer");}
bool SIMNRA::deleteLayer(int layerIndex) {return invokeMethod<bool>(m_Target, L"DeleteLayer", { CreateVariantFrom<int>(layerIndex) });}
bool SIMNRA::addElements (int layerIndex, int numberOfElements) {return invokeMethod<bool>(m_Target, L"AddElements", { CreateVariantFrom<int>(layerIndex),  CreateVariantFrom<int>(numberOfElements) });}
bool SIMNRA::addIsotope (int layerIndex, int elementIndex) {return invokeMethod<bool>(m_Target, L"AddIsotope", { CreateVariantFrom<int>(layerIndex),  CreateVariantFrom<int>(elementIndex) });}
bool SIMNRA::deleteIsotope (int layerIndex, int elementIndex, int isotopeIndex) {return invokeMethod<bool>(m_Target, L"DeleteIsotope",{CreateVariantFrom<int>(layerIndex), CreateVariantFrom<int>(elementIndex), CreateVariantFrom<int>(isotopeIndex) }); }
void SIMNRA::addLayerProperties(std::vector<std::wstring> elementNames, std::vector<double> elementCons, double thickness){
    if (elementNames.size() != elementCons.size()) throw std::runtime_error("element names does not match element consentration.");
    addLayer() ;
    int Nlayers = getNumberOfLayers(); 
    if (elementNames.size() > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::length_error("too many elements in SIMNRA layer");
    int Nels = static_cast<int>(elementNames.size());
    addElements(Nlayers, Nels );
    setLayerThickness(Nlayers, thickness);
    for (int i=1; i<= Nels; i++ ) setElementName(Nlayers, i, elementNames[i-1]);
    setElementConcentrationArray(Nlayers, elementCons);
}
void SIMNRA::setNumberOfLayers(int numLayers) { set(m_Target, L"NumberOfLayers", numLayers); }
void SIMNRA::setIsotopeConcentration (int layerIndex, int elementIndex, int isotopeIndex, double val){ set<double>(m_Target, L"IsotopeConcentration", val, {CreateVariantFrom<int>(layerIndex), CreateVariantFrom<int>(elementIndex), CreateVariantFrom<int>(isotopeIndex) });  }
void SIMNRA::setIsotopeMass (int layerIndex, int elementIndex, int isotopeIndex, double val){ set<double>(m_Target, L"IsotopeMass", val, {CreateVariantFrom<int>(layerIndex), CreateVariantFrom<int>(elementIndex), CreateVariantFrom<int>(isotopeIndex) }); }
void SIMNRA::setHasLayerRoughness (int layerIndex, bool status){ set<bool>(m_Target, L"HasLayerRoughness", status, {CreateVariantFrom<int>(layerIndex) }); }
void SIMNRA::setHasLayerPorosity (int layerIndex, bool status){ set<bool>(m_Target, L"HasLayerPorosity", status, {CreateVariantFrom<int>(layerIndex) });}
void SIMNRA::setLayerRoughness (int layerIndex, double val) { set<double>(m_Target, L"LayerRoughness", val, {CreateVariantFrom<int>(layerIndex) }); };
void SIMNRA::setPorosityFraction (int layerIndex, double val) { set<double>(m_Target, L"PorosityFraction", val, {CreateVariantFrom<int>(layerIndex) }); };
void SIMNRA::setPoreDiameter (int layerIndex, double val) { set<double>(m_Target, L"PoreDiameter", val, {CreateVariantFrom<int>(layerIndex) }); };

void SIMNRA::setLayerThickness(int layerIndex, double thick) { set(m_Target, L"LayerThickness", thick, { CreateVariantFrom<int>(layerIndex) }); }
void SIMNRA::setElementName(int layerIndex, int elementIndex, std::wstring elname) { set(m_Target, L"ElementName", elname, { CreateVariantFrom<int>(layerIndex), CreateVariantFrom<int>(elementIndex) }); }
void SIMNRA::setElementZ(int layerIndex, int elementIndex, int Z) { set(m_Target, L"ElementZ", Z, { CreateVariantFrom<int>(layerIndex), CreateVariantFrom<int>(elementIndex) }); }
void SIMNRA::setElementConcentration(int layerIndex, int elementIndex, double conc) { set(m_Target, L"ElementConcentration", conc, { CreateVariantFrom<int>(layerIndex), CreateVariantFrom<int>(elementIndex) }); }
void SIMNRA::setElementConcentrationArray(int layerIndex, const std::vector<double>& concentrations) {
    SAFEARRAYBOUND sab;
    sab.lLbound = 1; // SIMNRA arrays often start at 1
    sab.cElements = static_cast<ULONG>(concentrations.size());
    SAFEARRAY* psa = SafeArrayCreate(VT_R8, 1, &sab);
    if (!psa) throw std::runtime_error("Failed to create SAFEARRAY");
    for (LONG i = 0; i < static_cast<LONG>(concentrations.size()); ++i) {
        LONG idx = i + 1;
        SafeArrayPutElement(psa, &idx, const_cast<double*>(&concentrations[i]));
    }
    VARIANT var;
    VariantInit(&var);
    var.vt = VT_ARRAY | VT_R8;
    var.parray = psa;
    SetPropertyValue(m_Target, L"ElementConcentrationArray", { CreateVariantFrom<int> (layerIndex), var });
    VariantClear(&var);
}
void SIMNRA::setConcentrationMatrix (std::vector< std::vector<double>> matrix){ for (int i =0; i<matrix.size(); i++) setElementConcentrationArray(i+1, matrix[i]);}
void SIMNRA::setLayerThicknessArray (std::vector<double> thicknessArr){ for (int i =0; i<thicknessArr.size(); i++) setLayerThickness(i+1, thicknessArr[i]);  }

int SIMNRA::getNumberOfLayers() { return get<int>(m_Target, L"NumberOfLayers"); }
bool SIMNRA::getHasLayerRoughness (int layerIndex){ return get<bool>(m_Target, L"HasLayerRoughness", {CreateVariantFrom<int>(layerIndex)}); }
bool SIMNRA::getHasLayerPorosity (int layerIndex){ return get<bool>(m_Target, L"HasLayerPorosity", {CreateVariantFrom<int>(layerIndex)});}
double SIMNRA::getLayerRoughness (int layerIndex){ return get<double>(m_Target, L"LayerRoughness", {CreateVariantFrom<int>(layerIndex)});}
double SIMNRA::getPorosityFraction (int layerIndex) { return get<double>(m_Target, L"PorosityFraction",{CreateVariantFrom<int>(layerIndex) }); };
double SIMNRA::getPoreDiameter (int layerIndex) { return get<double>(m_Target, L"PoreDiameter", {CreateVariantFrom<int>(layerIndex) }); };
double SIMNRA::getLayerThickness(int layerIndex) { return get<double>(m_Target, L"LayerThickness", { CreateVariantFrom<int>(layerIndex) }); }
int SIMNRA::getNumberOfElements(int layerIndex) { return get<int>(m_Target, L"NumberOfElements", { CreateVariantFrom<int>(layerIndex) }); }
std::wstring SIMNRA::getElementName(int layerIndex, int elementIndex) { return get<std::wstring>(m_Target, L"ElementName", { CreateVariantFrom<int>(layerIndex), CreateVariantFrom<int>(elementIndex) }); }
int SIMNRA::getElementZ(int layerIndex, int elementIndex) { return get<int>(m_Target, L"ElementZ",  { CreateVariantFrom<int>(layerIndex), CreateVariantFrom<int>(elementIndex) }); }
int SIMNRA::getNumberOfIsotopes(int layerIndex, int elementIndex){return get<int>(m_Target, L"NumberOfIsotopes",  { CreateVariantFrom<int>(layerIndex), CreateVariantFrom<int>(elementIndex) }); }
double SIMNRA::getIsotopeMass(int layerIndex, int elementIndex, int isotopeIndex) {
    return get<double>(m_Target, L"IsotopeMass", {CreateVariantFrom<int>(layerIndex),
        CreateVariantFrom<int>(elementIndex), CreateVariantFrom<int>(isotopeIndex)});
}
double SIMNRA::getIsotopeConcentration(int layerIndex, int elementIndex, int isotopeIndex) {
    return get<double>(m_Target, L"IsotopeConcentration", {CreateVariantFrom<int>(layerIndex),
        CreateVariantFrom<int>(elementIndex), CreateVariantFrom<int>(isotopeIndex)});
}
double SIMNRA::getElementConcentration(int layerIndex, int elementIndex) { return get<double>(m_Target, L"ElementConcentration", { CreateVariantFrom<int>(layerIndex), CreateVariantFrom<int>(elementIndex) }); }
std::vector<double> SIMNRA::getElementConcentrationArray(int layerIndex) {
    VARIANT result = GetPropertyValue(m_Target, L"ElementConcentrationArray", { CreateVariantFrom<int>(layerIndex) });
    if (result.vt != (VT_ARRAY | VT_R8))
        throw std::runtime_error("ElementConcentrationArray did not return a double SAFEARRAY");

    SAFEARRAY* psa = result.parray;
    LONG lbound, ubound;
    SafeArrayGetLBound(psa, 1, &lbound);
    SafeArrayGetUBound(psa, 1, &ubound);

    std::vector<double> vec;
    vec.resize(ubound - lbound + 1);
    for (LONG i = lbound; i <= ubound; ++i)
        SafeArrayGetElement(psa, &i, &vec[i - lbound]);
    VariantClear(&result);
    return vec;
}

// -------------------------- Calc settings --------------------------
void SIMNRA::setCalc_DualScattering(bool val) { set<bool>(m_Calc, L"DualScattering", val); }
void SIMNRA::setCalc_Isotopes(bool val) { set<bool>(m_Calc, L"Isotopes", val); }
void SIMNRA::setCalc_MultipleScattering(bool val) { set<bool>(m_Calc, L"MultipleScattering", val); }
void SIMNRA::setCalc_NuclearStoppingModel(int val) { set<int>(m_Calc, L"NuclearStoppingModel", val); }
void SIMNRA::setCalc_ScreeningModel(int val) { set<int>(m_Calc, L"ScreeningModel", val); }
void SIMNRA::setCalc_StoppingModel(int val) { set<int>(m_Calc, L"StoppingModel", val); }
void SIMNRA::setCalc_Straggling(bool val) { set<bool>(m_Calc, L"Straggling", val); }
void SIMNRA::setCalc_StragglingModel(int val) { set<int>(m_Calc, L"StragglingModel", val); }
void SIMNRA::setCalc_StragglingShape(int val) { set<int>(m_Calc, L"StragglingShape", val); }
void SIMNRA::setCalc_MultipleScatteringModel(int val) { set<int>(m_Calc, L"MultipleScatteringModel", val); }
void SIMNRA::setCalc_CrossSecStraggling(int val) { set<int>(m_Calc, L"CrossSecStraggling", val); }

void SIMNRA::setCalc_SubstrateRoughnessDimension(int val) { set<int>(m_Calc, L"SubstrateRoughnessDimension", val); }
void SIMNRA::setCalc_NumberOfDVariations(int val) { set<int>(m_Calc, L"NumberOfDVariations", val); }
void SIMNRA::setCalc_NumberOfAngleVariations(int val) { set<int>(m_Calc, L"NumberOfAngleVariations", val); }
void SIMNRA::setCalc_DualScatteringRoughness(int val) { set<int>(m_Calc, L"DualScatteringRoughness", val); }
void SIMNRA::setCalc_ElementSpectra(bool val) { set<bool>(m_Calc, L"ElementSpectra", val); }
void SIMNRA::setCalc_PileUpModel(int val) { set<int>(m_Calc, L"PUModel", val); }

bool SIMNRA::getCalc_DualScattering() { return get<bool>(m_Calc, L"DualScattering"); }
bool SIMNRA::getCalc_Isotopes() { return get<bool>(m_Calc, L"Isotopes"); }
bool SIMNRA::getCalc_MultipleScattering() { return get<bool>(m_Calc, L"MultipleScattering"); }
int SIMNRA::getCalc_NuclearStoppingModel() { return get<int>(m_Calc, L"NuclearStoppingModel"); }
int SIMNRA::getCalc_ScreeningModel() { return get<int>(m_Calc, L"ScreeningModel"); }
int SIMNRA::getCalc_StoppingModel() { return get<int>(m_Calc, L"StoppingModel"); }
bool SIMNRA::getCalc_Straggling() { return get<bool>(m_Calc, L"Straggling"); }
int SIMNRA::getCalc_StragglingModel() { return get<int>(m_Calc, L"StragglingModel"); }
int SIMNRA::getCalc_StragglingShape() { return get<int>(m_Calc, L"StragglingShape"); }
int SIMNRA::getCalc_MultipleScatteringModel() { return get<int>(m_Calc, L"MultipleScatteringModel"); }
int SIMNRA::getCalc_CrossSecStraggling() { return get<int>(m_Calc, L"CrossSecStraggling"); }
bool SIMNRA::getCalc_ElementSpectra() { return get<bool>(m_Calc, L"ElementSpectra"); }
int SIMNRA::getCalc_PileUpModel() { return get<int>(m_Calc, L"PUModel"); }

//roughness
int SIMNRA::getCalc_SubstrateRoughnessDimension() { return get<int>(m_Calc, L"SubstrateRoughnessDimension"); }
int SIMNRA::getCalc_NumberOfDVariations() { return get<int>(m_Calc, L"NumberOfDVariations"); }
int SIMNRA::getCalc_NumberOfAngleVariations() { return get<int>(m_Calc, L"NumberOfAngleVariations"); }
int SIMNRA::getCalc_DualScatteringRoughness() { return get<int>(m_Calc, L"DualScatteringRoughness"); }

// --- Stopping ---
double SIMNRA::stoppingStragglingInLayer(int ionZ, double ionMass, double Ein, int layerIndex, int elementIndex) {
    return invokeMethod<double>(m_Stopping, L"StragglingInLayer",
        { CreateVariantFrom<int>(ionZ), CreateVariantFrom<double>(ionMass),
          CreateVariantFrom<double>(Ein), CreateVariantFrom<int>(layerIndex),
          CreateVariantFrom<int>(elementIndex) });
}

double SIMNRA::stoppingEnergyLossInLayer(int ionZ, double ionMass, double Ein, int layerIndex, int elementIndex) {
    return invokeMethod<double>(m_Stopping, L"EnergylossInLayer",
        { CreateVariantFrom<int>(ionZ), CreateVariantFrom<double>(ionMass),
          CreateVariantFrom<double>(Ein), CreateVariantFrom<int>(layerIndex),
          CreateVariantFrom<int>(elementIndex) });
}

void SIMNRA::stoppingClearCache() {
    invokeMethod<void>(m_Stopping, L"ClearCache");
}
