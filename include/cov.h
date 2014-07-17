#ifndef _COVARIANCE_FUNCTION_H_
#define _COVARIANCE_FUNCTION_H_

// Isotropic and dealing with derivative observations
//#include "../src/cov/isotropic.hpp"
//#include "../src/cov/dealingwithderivativeobservations.hpp"
#include "../src/cov/CovDerObs.hpp"

// Squared Exponential
#include "../src/cov/covSEiso/CovSEiso.hpp"
#include "../src/cov/covSEiso/CovSEisoDerObsBase.hpp"
//#include "../src/cov/covseisobase.hpp"
//#include "../src/cov/covseisoderbase.hpp"

namespace GP {

/**
 * @defgroup	Cov
 * @brief		Covariance Functions\n
 *					All covariance classes should have public static member functions.
 *					-# K: \f$\mathbf{K}(\mathbf{X}, \mathbf{X})\f$
 *					-# Ks: \f$\mathbf{K}(\mathbf{X}, \mathbf{X}_*)\f$
 *					-# Kss: \f$\mathbf{K}(\mathbf{X}_*, \mathbf{X}_*)\f$
 *					.
 *					Also, no covariance class contains any data.
 *					Instead, data are stored in data classes such as
 *					-# TrainingData
 *					-# DerivativeTrainingData
 *					-# TestData
 *					.
 *					Assertions are checked only in those public static member functions
 *					which can be accessed outside.
 */

//typedef Isotropic<float, CovSEIsoBase>											CovSEIsoSlow;
//typedef template<typename Scalar> DealingWithDerivativeObservations<Scalar, CovSEIsoDerBase>		CovSEIsoDerSlow;
//typedef CovDerObs<float, CovSEisoDerObsBase> CovSEisoDerObsf;

//template <typename Scalar>
//using CovSEisoDerObs = CovDerObs<Scalar, CovSEisoDerObsBase>;

template<typename Scalar>
class CovSEisoDerObs : public CovDerObs<Scalar, CovSEisoDerObsBase> {};

//template<typename Scalar> 
//struct _CovDerObs
//{
//   typedef CovDerObs<Scalar, CovSEisoDerObsBase> __CovSEisoDerObs;
//};
//template<typename Scalar> using CovSEisoDerObs = typename _CovDerObs::__CovSEisoDerObs;
}

#endif