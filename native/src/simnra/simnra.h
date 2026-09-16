#ifndef SIMNRA_H
#define SIMNRA_H

#include "com.h"

/**
 * @class SIMNRA
 * @brief A class to interface with the SIMNRA simulation software for ion beam analysis.
 * 
 * This class provides methods to control the SIMNRA application, manage target layers, 
 * calculate spectra, and configure various simulation parameters such as beam properties, 
 * stopping power, and scattering models
 */
class SIMNRA {
private:
    IDispatch* m_App;          ///< Pointer to the SIMNRA application COM interface
    IDispatch* m_Setup;        ///< Pointer to the setup COM interface
    IDispatch* m_Target;       ///< Pointer to the target COM interface
    IDispatch* m_Calc;         ///< Pointer to the calculation COM interface
    IDispatch* m_Fit;          ///< Pointer to the fitting COM interface
    IDispatch* m_Projectile;   ///< Pointer to the projectile COM interface
    IDispatch* m_Spectrum;     ///< Pointer to the spectrum COM interface
    IDispatch* m_Stopping;     ///< Pointer to the stopping power COM interface
    IDispatch* m_PIGE;         ///< Pointer to the PIGE (Particle-Induced Gamma-ray Emission) COM interface
    IDispatch* m_CrossSec;     ///< Pointer to the cross-section COM interface
    std::vector<double> m_lastSpectrum; ///< Cached spectrum buffer reused across queries

private : 
    void pumpMessages();

public:
    /**
     * @brief Constructs a SIMNRA object and initializes COM interfaces.
     * @param MTA : if true, use mult-threaded apartment. Useful for maximizing cpu utilization. However, the result can sometimes be incorrect. Recommendation : use MTA = True for fitting, MTA=False for data generation.
     * @param threadPriorityIndex : set the thread priority
     *          case -3: THREAD_PRIORITY_IDLE
                case -2: THREAD_PRIORITY_LOWEST
                case -1: THREAD_PRIORITY_BELOW_NORMAL
                case 0:  THREAD_PRIORITY_NORMAL
                case 1: THREAD_PRIORITY_ABOVE_NORMAL
                case 2:  THREAD_PRIORITY_HIGHEST
                case 3:  THREAD_PRIORITY_TIME_CRITICAL
     */
    SIMNRA( bool MTA=true, int threadPriorityIndex = 0);

    /**
     * @brief Destructor that releases COM interfaces.
     */
    ~SIMNRA();

    // ------------------------- App Methods -------------------------
    /**
     * @brief Retrieves the last message from the SIMNRA application.
     * @return The last message as a wide string.
     */
    std::wstring getLastMessage();

    /**
     * @brief Opens a SIMNRA file.
     * @param filename The path to the file to open.
     * @param type The type of file to open (-1 for default).
     * @return True if the file was opened successfully, false otherwise.
     */
    bool open(const wchar_t* filename, int type = -1);

    /**
     * @brief Save a SIMNRA file.
     * @param filename The path to the file to open.
     * @param type The type of file to open (2 for default).
     * @return True if the file was saved successfully, false otherwise.
     */
    bool saveAs(const wchar_t* filename, int filetype = 2);

    /**
     * @brief Calculates the spectrum using the current setup.
     * @return True if the calculation was successful, false otherwise.
     */
    bool calculateSpectrum();

    /**
     * @brief Calculates the spectrum using a faster algorithm.
     * @return True if the calculation was successful, false otherwise.
     */
    bool calculateSpectrumFast();

    /**
     * @brief Reads spectrum data from a file.
     * @param file The path to the spectrum data file.
     * @param index The index of the spectrum to read.
     * @return True if the data was read successfully, false otherwise.
     */
    bool readSpectrumData(const std::wstring& file, int index);

    /**
     * @brief Gets the version of the SIMNRA application.
     * @return The version as a wide string.
     */
    std::wstring getVersion();

    // ------------------------- Spectrum Access Methods -------------------------
    /**
     * @brief Retrieves the spectrum data for a given spectrum ID.
     * @param spID The spectrum ID (default is 1).
     * @return A vector containing the spectrum data.
     */
    std::vector<double>& getSpectrum(int spID = 1);
    /**
     * @brief Sets spectrum data for a given spectrum ID.
     * @param spID The spectrum ID.
     * @param data Spectrum data to set.
     * @return True if the spectrum was set successfully, false otherwise.
     */
    bool setSpectrum(int spID, const std::vector<double>& data);

    /**
     * @brief Gets the spectrum ID associated with an element's atomic number.
     * @param Z The atomic number of the element.
     * @return The spectrum ID.
     */
    int spectrumIDOfElement(int Z);

    // ------------------------- Layer/Target Management Methods -------------------------
    /**
     * @brief Adds a new layer to the target.
     * @return True if the layer was added successfully, false otherwise.
     */
    bool addLayer();

    /**
     * @brief Deletes a layer from the target.
     * @param layerIndex The index of the layer to delete.
     * @return True if the layer was deleted successfully, false otherwise.
     */
    bool deleteLayer(int layerIndex);

    /**
     * @brief Adds elements to a specific layer.
     * @param layerIndex The index of the layer.
     * @param numberOfElements The number of elements to add.
     * @return True if the elements were added successfully, false otherwise.
     */
    bool addElements(int layerIndex, int numberOfElements);

    /**
     * @brief Adds an isotope to an element in a specific layer.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     * @return True if the isotope was added successfully, false otherwise.
     */
    bool addIsotope(int layerIndex, int elementIndex);

    /**
     * @brief Deletes an isotope from an element in a specific layer.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     * @param isotopeIndex The index of the isotope.
     * @return True if the isotope was deleted successfully, false otherwise.
     */
    bool deleteIsotope(int layerIndex, int elementIndex, int isotopeIndex);

    /**
     * @brief Adds properties to a layer, including element names, concentrations, and thickness.
     * @param elementNames Vector of element names.
     * @param elementCons Vector of element concentrations.
     * @param thickness The thickness of the layer.
     */
    void addLayerProperties(std::vector<std::wstring> elementNames, std::vector<double> elementCons, double thickness);

    /**
     * @brief Sets the concentration of an isotope in a specific layer and element.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     * @param isotopeIndex The index of the isotope.
     * @param val The concentration value.
     */
    void setIsotopeConcentration(int layerIndex, int elementIndex, int isotopeIndex, double val);

    /**
     * @brief Sets the mass of an isotope in a specific layer and element.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     * @param isotopeIndex The index of the isotope.
     * @param val The mass value.
     */
    void setIsotopeMass(int layerIndex, int elementIndex, int isotopeIndex, double val);

    /**
     * @brief Sets whether a layer has roughness.
     * @param layerIndex The index of the layer.
     * @param status True to enable roughness, false otherwise.
     */
    void setHasLayerRoughness(int layerIndex, bool status);

    /**
     * @brief Sets whether a layer has porosity.
     * @param layerIndex The index of the layer.
     * @param status True to enable porosity, false otherwise.
     */
    void setHasLayerPorosity(int layerIndex, bool status);

    /**
     * @brief Sets the roughness value of a layer.
     * @param layerIndex The index of the layer.
     * @param val The roughness value.
     */
    void setLayerRoughness(int layerIndex, double val);

    /**
     * @brief Sets the porosity fraction of a layer.
     * @param layerIndex The index of the layer.
     * @param val The porosity fraction value.
     */
    void setPorosityFraction(int layerIndex, double val);

    /**
     * @brief Sets the pore diameter of a layer.
     * @param layerIndex The index of the layer.
     * @param val The pore diameter value.
     */
    void setPoreDiameter(int layerIndex, double val);

    /**
     * @brief Sets the number of layers in the target.
     * @param numLayers The number of layers.
     */
    void setNumberOfLayers(int numLayers);

    /**
     * @brief Sets the thickness of a specific layer.
     * @param layerIndex The index of the layer.
     * @param thick The thickness value.
     */
    void setLayerThickness(int layerIndex, double thick);

    /**
     * @brief Sets the name of an element in a specific layer.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     * @param elname The name of the element (default is empty string).
     */
    void setElementName(int layerIndex , int elementIndex, std::wstring elname);

    /**
     * @brief Sets the Z for an element in a specific layer.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     * @param Z The proton number.
     */
    void setElementZ(int layerIndex, int elementIndex, int Z);

    /**
     * @brief Sets the concentration of an element in a specific layer.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     * @param conc The concentration value .
     */
    void setElementConcentration(int layerIndex, int elementIndex , double conc );

    /**
     * @brief Sets the concentrations of all elements in a specific layer.
     * @param layerIndex The index of the layer.
     * @param concentrations Vector of concentration values.
     */
    void setElementConcentrationArray(int layerIndex, const std::vector<double>& concentrations);

    /**
     * @brief Sets the concentration matrix for all layers.
     * @param matrix A 2D vector representing the concentration matrix.
     */
    void setConcentrationMatrix(std::vector<std::vector<double>> matrix);

    /**
     * @brief Sets the thickness for all layers.
     * @param thicknessArr Vector of thickness values.
     */
    void setLayerThicknessArray(std::vector<double> thicknessArr);

    /**
     * @brief Gets the number of layers in the target.
     * @return The number of layers.
     */
    int getNumberOfLayers();

    /**
     * @brief Gets the number of isotope in a given layer index and element.
     * @param layerIndex The index of the layer
     * @param elementIndex The index of the element
     * @return The number of isotopes.
     */
    int getNumberOfIsotopes(int layerIndex, int elementIndex);

    /**
     * @brief Checks if a layer has roughness.
     * @param layerIndex The index of the layer.
     * @return True if the layer has roughness, false otherwise.
     */
    bool getHasLayerRoughness(int layerIndex);

    /**
     * @brief Checks if a layer has porosity.
     * @param layerIndex The index of the layer.
     * @return True if the layer has porosity, false otherwise.
     */
    bool getHasLayerPorosity(int layerIndex);

    /**
     * @brief Gets the roughness value of a layer.
     * @param layerIndex The index of the layer.
     * @return The roughness value.
     */
    double getLayerRoughness(int layerIndex);

    /**
     * @brief Gets the porosity fraction of a layer.
     * @param layerIndex The index of the layer.
     * @return The porosity fraction value.
     */
    double getPorosityFraction(int layerIndex);

    /**
     * @brief Gets the pore diameter of a layer.
     * @param layerIndex The index of the layer.
     * @return The pore diameter value.
     */
    double getPoreDiameter(int layerIndex);

    /**
     * @brief Gets the thickness of a specific layer.
     * @param layerIndex The index of the layer .
     * @return The thickness value.
     */
    double getLayerThickness(int layerIndex);

    /**
     * @brief Gets the number of elements in a specific layer.
     * @param layerIndex The index of the layer.
     * @return The number of elements.
     */
    int getNumberOfElements(int layerIndex);

    /**
     * @brief Gets the name of an element in a specific layer.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     * @return The element name as a wide string.
     */
    std::wstring getElementName(int layerIndex, int elementIndex );

    /**
     * @brief Gets the Z for an element in a specific layer.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     */
    int getElementZ(int layerIndex, int elementIndex);

    /**
     * @brief Gets the concentration of an element in a specific layer.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     * @return The concentration value.
     */
    double getElementConcentration(int layerIndex, int elementIndex);

    /**
     * @brief Gets the concentrations of all elements in a specific layer.
     * @param layerIndex The index of the layer.
     * @return A vector of concentration values.
     */
    std::vector<double> getElementConcentrationArray(int layerIndex);

    // ------------------------- Calibration and Setup Methods -------------------------
    /**
     * @brief Sets the beam spread.
     * @param spread The beam spread value.
     */
    void setBeamSpread(double spread);

    /**
     * @brief Sets the beam energy.
     * @param E The beam energy value.
     */
    void setBeamEnergy(double E);

    /**
     * @brief Sets the number of particles per steradian.
     * @param val The particles per steradian value.
     */
    void setParticlesSr(double val);

    /**
     * @brief Sets the detector resolution.
     * @param val The detector resolution value.
     */
    void setDetectorResolution(double val);

    /**
     * @brief Sets the linear calibration factor.
     * @param val The linear calibration value.
     */
    void setCalibrationLinear(double val);

    /**
     * @brief Sets the calibration offset.
     * @param val The calibration offset value.
     */
    void setCalibrationOffset(double val);

    /**
     * @brief Sets the quadratic calibration factor.
     * @param val The quadratic calibration value.
     */
    void setCalibrationQuadratic(double val);

    /**
     * @brief Sets the real time for the measurement.
     * @param val The real time value.
     */
    void setRealTime(double val);

    /**
     * @brief Sets the live time for the measurement.
     * @param val The live time value.
     */
    void setLiveTime(double val);

     /**
     * @brief Set pileup calculation status.
     * @param val true if pileup is calculated, otherwise false.
     */
    void setPileUpCalculation(bool val);

    /**
     * @brief Set whether to use live time correction.
     * @param val true if LT is calculated, otherwise false.
     */
    void setLiveTimeCorrection(bool val);


    /**
     * @brief Gets the beam spread.
     * @return The beam spread value.
     */
    double getBeamSpread();

    /**
     * @brief Gets the beam energy.
     * @return The beam energy value.
     */
    double getBeamEnergy();

    /**
     * @brief Gets the number of particles per steradian.
     * @return The particles per steradian value.
     */
    double getParticlesSr();

    /**
     * @brief Gets the detector resolution.
     * @return The detector resolution value.
     */
    double getDetectorResolution();

    /**
     * @brief Gets the linear calibration factor.
     * @return The linear calibration value.
     */
    double getCalibrationLinear();

    /**
     * @brief Gets the calibration offset.
     * @return The calibration offset value.
     */
    double getCalibrationOffset();

    /**
     * @brief Gets the quadratic calibration factor.
     * @return The quadratic calibration value.
     */
    double getCalibrationQuadratic();

    /**
     * @brief Gets the real time for the measurement.
     * @return The real time value.
     */
    double getRealTime();

    /**
     * @brief Gets the live time for the measurement.
     * @return The live time value.
     */
    double getLiveTime();

    /**
     * @brief Gets pileup calculation status.
     * @return true if pileup is calculated, otherwise false.
     */
    bool getPileUpCalculation();

    /**
     * @brief Gets whether to use live time correction.
     * @return val true if LT is calculated, otherwise false.
     */
    bool getLiveTimeCorrection();

    // ------------------------- Calculation Settings Methods -------------------------
    /**
     * @brief Enables or disables dual scattering calculation.
     * @param val True to enable, false to disable.
     */
    void setCalc_DualScattering(bool val);

    /**
     * @brief Enables or disables isotope calculation.
     * @param val True to enable, false to disable.
     */
    void setCalc_Isotopes(bool val);

    /**
     * @brief Enables or disables multiple scattering calculation.
     * @param val True to enable, false to disable.
     */
    void setCalc_MultipleScattering(bool val);

    /**
     * @brief Sets the nuclear stopping model.
     * @param val The nuclear stopping model value.
     */
    void setCalc_NuclearStoppingModel(int val);

    /**
     * @brief Sets the screening model.
     * @param val The screening model value.
     */
    void setCalc_ScreeningModel(int val);

    /**
     * @brief Sets the stopping model.
     * @param val The stopping model value.
     */
    void setCalc_StoppingModel(int val);

    /**
     * @brief Enables or disables straggling calculation.
     * @param val True to enable, false to disable.
     */
    void setCalc_Straggling(bool val);

    /**
     * @brief Sets the straggling model.
     * @param val The straggling model value.
     */
    void setCalc_StragglingModel(int val);

    /**
     * @brief Sets the straggling shape.
     * @param val The straggling shape value.
     */
    void setCalc_StragglingShape(int val);

    /**
     * @brief Sets the multiple scattering model.
     * @param val The multiple scattering model value.
     */
    void setCalc_MultipleScatteringModel(int val);

    /**
     * @brief Sets the cross-section straggling.
     * @param val The cross-section straggling value.
     */
    void setCalc_CrossSecStraggling(int val);

    /**
     * @brief Sets the substrate roughness dimension.
     * @param val The substrate roughness dimension value.
     */
    void setCalc_SubstrateRoughnessDimension(int val);

    /**
     * @brief Sets the number of depth variations.
     * @param val The number of depth variations.
     */
    void setCalc_NumberOfDVariations(int val);

    /**
     * @brief Sets the number of angle variations.
     * @param val The number of angle variations.
     */
    void setCalc_NumberOfAngleVariations(int val);

    /**
     * @brief Sets the dual scattering roughness.
     * @param val The dual scattering roughness value.
     */
    void setCalc_DualScatteringRoughness(int val);

    /**
     * @brief Enables or disables element spectra calculation.
     * @param flag True to enable, false to disable.
     */
    void setCalc_ElementSpectra(bool flag);

     /**
     * @brief Sets pile up calculation model
     * @param val 0 for accurate, 1 for fast.
     */
    void setCalc_PileUpModel(int val);

    /**
     * @brief Gets whether dual scattering calculation is enabled.
     * @return True if enabled, false otherwise.
     */
    bool getCalc_DualScattering();

    /**
     * @brief Gets whether isotope calculation is enabled.
     * @return True if enabled, false otherwise.
     */
    bool getCalc_Isotopes();

    /**
     * @brief Gets whether multiple scattering calculation is enabled.
     * @return True if enabled, false otherwise.
     */
    bool getCalc_MultipleScattering();

    /**
     * @brief Gets the nuclear stopping model.
     * @return The nuclear stopping model value.
     */
    int getCalc_NuclearStoppingModel();

    /**
     * @brief Gets the screening model.
     * @return The screening model value.
     */
    int getCalc_ScreeningModel();

    /**
     * @brief Gets the stopping model.
     * @return The stopping model value.
     */
    int getCalc_StoppingModel();

    /**
     * @brief Gets whether straggling calculation is enabled.
     * @return True if enabled, false otherwise.
     */
    bool getCalc_Straggling();

    /**
     * @brief Gets the straggling model.
     * @return The straggling model value.
     */
    int getCalc_StragglingModel();

    /**
     * @brief Gets the straggling shape.
     * @return The straggling shape value.
     */
    int getCalc_StragglingShape();

    /**
     * @brief Gets the multiple scattering model.
     * @return The multiple scattering model value.
     */
    int getCalc_MultipleScatteringModel();

    /**
     * @brief Gets the cross-section straggling.
     * @return The cross-section straggling value.
     */
    int getCalc_CrossSecStraggling();

    /**
     * @brief Gets the substrate roughness dimension.
     * @return The substrate roughness dimension value.
     */
    int getCalc_SubstrateRoughnessDimension();

    /**
     * @brief Gets the number of depth variations.
     * @return The number of depth variations.
     */
    int getCalc_NumberOfDVariations();

    /**
     * @brief Gets the number of angle variations.
     * @return The number of angle variations.
     */
    int getCalc_NumberOfAngleVariations();

    /**
     * @brief Gets the dual scattering roughness.
     * @return The dual scattering roughness value.
     */
    int getCalc_DualScatteringRoughness();

    /**
     * @brief Gets whether element spectra calculation is enabled.
     * @return True if enabled, false otherwise.
     */
    bool getCalc_ElementSpectra();

     /**
     * @brief Gets pile up calculation model
     * @return 0 for accurate, 1 for fast.
     */
    int getCalc_PileUpModel();

    // ------------------------- Stopping Methods -------------------------
    /**
     * @brief Calculates the stopping straggling in a layer for a given ion.
     * @param ionZ The atomic number of the ion.
     * @param ionMass The mass of the ion.
     * @param Ein The incident energy of the ion.
     * @param layerIndex The index of the layer.
     * @param elementIndex The index of the element.
     * @return The stopping straggling value.
     */
    double stoppingStragglingInLayer(int ionZ, double ionMass, double Ein, int layerIndex, int elementIndex);

    /**
     * @brief Calculates the energy loss in a layer for a given ion.
     * @param ionZ The atomic number of the ion.
     * @param ionMass The mass of the ion.
     * @param Ein The incident energy of the ion.
     * @param layerIndex The index of the layer.
     * @param elementElement The index of the element.
     * @return The energy loss value.
     */
    double stoppingEnergyLossInLayer(int ionZ, double ionMass, double Ein, int layerIndex, int elementIndex);

    /**
     * @brief Clears the stopping power cache.
     */
    void stoppingClearCache();
};

#endif
