#ifndef LIKELIHOOD_FUNCTION_GAUSSIAN_HPP
#define LIKELIHOOD_FUNCTION_GAUSSIAN_HPP

#include <cmath>

#include "../../util/macros.hpp"
#include "../../data/TrainingData.hpp"
#include "../../data/TestData.hpp"

namespace GPOM{

template<Scalar>
class LikGauss
{
// define matrix types
protected:	TYPE_DEFINE_VECTOR(Scalar);

// define hyperparameters
public:		TYPE_DEFINE_HYP(Scalar, 1); // sigma_n

	// diagonal vector
	static VectorPtr lik(const Hyp &logHyp, const TrainingData<Scalar> trainingData, const int pdHypIndex = -1)
	{
		assert(pdHypIndex < logHyp.size());

		// number of training data
		VectorPtr pD(new Vector(trainingData.N()));

		// some constants
		const Scalar sn2 = exp(static_cast<Scalar>(2.f) * logHyp(0));

		// derivatives w.r.t log(sn)
		if(pdIndex == 0)	pD->fill(static_cast<Scalar>(2.f) * sn2);

		// likelihood
		else					pD->fill(sn2);

		return pD;
	}

	//// diagonal matrix
	//MatrixPtr operator()(MatrixConstPtr pX, const Hyp &logHyp, const int pdIndex = -1) const
	//{
	//	// number of training data
	//	const int N = getN();
	//	MatrixPtr pD(new Matrix(N, N));
	//	pD->setZero();

	//	// derivatives w.r.t sn
	//	if(pdIndex == 0)			pD->diagonal().fill(((Scalar) 2.f) * exp((Scalar) 2.f * logHyp(0)));

	//	// likelihood
	//	else								pD->diagonal().fill(exp((Scalar) 2.f * logHyp(0)));

	//	return pD;
	//}
};

}

#endif