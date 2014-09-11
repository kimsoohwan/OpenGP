#ifndef _MEAN_FUNCTION_H_
#define _MEAN_FUNCTION_H_

// MeanZero
#include "../src/mean/MeanZero.hpp"

// MeanGP
#include "../src/mean/MeanGP.hpp"

#include "cov.h"
#include "inf.h"

//#include "../src/cov/covSEiso/CovSEiso.hpp"
//#include "../src/lik/LikGauss.hpp"
#include "../src/lik/LikGaussDerObs.hpp"

namespace GP {

/**
 * @class		MeanGlobalGP
 * @brief		Mean function with a global Gaussian process\n
 *					It inherits from MeanGP whith specialized template arguments for the global Gaussian process.\n
 *					The mean/cov/lik function for the global Gaussian process are MeanZero, CovSEiso, and LikGauss, respectively.
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-Mean
 * @author		Soohwan Kim
 * @date			10/09/2014
 */
template<typename Scalar>
class MeanGlobalGP : public MeanGP<Scalar, MeanZeroDerObs, CovRQisoDerObs, LikGaussDerObs, InfExactDerObs> {};
//class MeanGlobalGP : public MeanGP<Scalar, MeanZeroDerObs, CovSEisoDerObs, LikGaussDerObs, InfExactDerObs> {};
//class MeanGlobalGP : public MeanGP<Scalar, MeanZero, CovSEiso, LikGauss> {};

}

#endif