#ifndef _COVARIANCE_FUNCTION_H_
#define _COVARIANCE_FUNCTION_H_

// Isotropic and dealing with derivative observations
//#include "../src/cov/isotropic.hpp"
//#include "../src/cov/dealingwithderivativeobservations.hpp"
#include "../src/cov/CovDerObs.hpp"

// Squared Exponential
#include "../src/cov/covSEiso/CovSEiso.hpp"
#include "../src/cov/covSEiso/CovSEisoDerObs.hpp"
//#include "../src/cov/covseisobase.hpp"
//#include "../src/cov/covseisoderbase.hpp"

namespace GP {
//typedef Isotropic<float, CovSEIsoBase>											CovSEIsoSlow;
//typedef template<typename Scalar> DealingWithDerivativeObservations<Scalar, CovSEIsoDerBase>		CovSEIsoDerSlow;
//typedef CovDerObs<float, _CovSEisoDerObs> CovSEisoDerObsf;

//template <typename Scalar>
//using CovSEisoDerObs = CovDerObs<Scalar, _CovSEisoDerObs>;

template<typename Scalar>
class CovSEisoDerObs : public CovDerObs<Scalar, _CovSEisoDerObs> {};

//template<typename Scalar> 
//struct _CovDerObs
//{
//   typedef CovDerObs<Scalar, _CovSEisoDerObs> __CovSEisoDerObs;
//};
//template<typename Scalar> using CovSEisoDerObs = typename _CovDerObs::__CovSEisoDerObs;
}

#endif