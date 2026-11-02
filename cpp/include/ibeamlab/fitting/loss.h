#pragma once
#include <ibeamlab/export.h>
namespace ibeamlab::fitting {
/** DataChiSquare uses data+floor; Neyman/Pearson use max(data/prediction,1).
 * Poisson and Gaussian retain the established floor-at-one formulas. */
enum class Loss { DataChiSquare, VarianceChiSquare, Neyman, Pearson, Mse, Poisson, Gaussian };
enum class RegionNormalization { Sum, WeightSum };
enum class MissingPrediction { Reject, Zero };
IBEAMLAB_API double loss_value(Loss, double data, double prediction, double variance = 0,
                               double floor = 1);
} // namespace ibeamlab::fitting
