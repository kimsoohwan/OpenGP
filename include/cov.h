#ifndef _COVARIANCE_FUNCTION_H_
#define _COVARIANCE_FUNCTION_H_

// Isotropic and dealing with derivative observations
//#include "../src/cov/isotropic.hpp"
//#include "../src/cov/dealingwithderivativeobservations.hpp"

// Squared Exponential
#include "../src/cov/CovSEiso.hpp"
//#include "../src/cov/covseisobase.hpp"
//#include "../src/cov/covseisoderbase.hpp"

namespace GP {
//typedef Isotropic<float, CovSEIsoBase>											CovSEIsoSlow;
//typedef template<typename Scalar> DealingWithDerivativeObservations<Scalar, CovSEIsoDerBase>		CovSEIsoDerSlow;
}

#endif