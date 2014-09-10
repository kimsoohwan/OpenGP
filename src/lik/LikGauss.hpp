#ifndef _LIKELIHOOD_FUNCTION_GAUSSIAN_HPP_
#define _LIKELIHOOD_FUNCTION_GAUSSIAN_HPP_

#include <cmath>

#include "../util/macros.h"
#include "../data/TrainingData.hpp"
#include "../data/TestData.hpp"

namespace GP{

/**
  * @class		LikGauss
  * @brief		Likelihood function
  * @ingroup	-Lik
  * @author		Soohwan Kim
  * @date		02/07/2014
  */
template<typename Scalar>
class LikGauss
{
/**@brief Number of hyperparameters */
public: static const int N = 1;

// define matrix types
protected:	TYPE_DEFINE_VECTOR(Scalar);

// define hyperparameters
public:		TYPE_DEFINE_HYP(Scalar, N); // sigma_n

	// diagonal vector
	static VectorPtr lik(const Hyp							&logHyp, 
								const TrainingData<Scalar>		&trainingData, 
								const int							pdHypIndex = -1)
	{
		assert(pdHypIndex < logHyp.size());

		// number of training data
		VectorPtr pD(new Vector(trainingData.N()));

		// some constants
		const Scalar sn2 = exp(static_cast<Scalar>(2.f) * logHyp(0));

		// derivatives w.r.t log(sn)
		if(pdHypIndex == 0)	pD->fill(static_cast<Scalar>(2.f) * sn2);

		// likelihood
		else					pD->fill(sn2);

		return pD;
	}

	// diagonal vector
	static VectorPtr lik(const Hyp &logHyp, 
								const TestData<Scalar> &testData)
	{
		// some constants
		const int M  = testData.M();

		// number of test data
		VectorPtr pD(new Vector(M));

		// sigma_n^2
		const Scalar sn2 = exp(static_cast<Scalar>(2.f) * logHyp(0));
		pD->fill(sn2);

		return pD;
	}

	//// diagonal matrix
	//MatrixPtr operator()(MatrixConstPtr pX, const Hyp &logHyp, const int pdHypIndex = -1) const
	//{
	//	// number of training data
	//	const int N = getN();
	//	MatrixPtr pD(new Matrix(N, N));
	//	pD->setZero();

	//	// derivatives w.r.t sn
	//	if(pdHypIndex == 0)			pD->diagonal().fill(((Scalar) 2.f) * exp((Scalar) 2.f * logHyp(0)));

	//	// likelihood
	//	else								pD->diagonal().fill(exp((Scalar) 2.f * logHyp(0)));

	//	return pD;
	//}
};

}

#endif