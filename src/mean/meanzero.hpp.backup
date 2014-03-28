#ifndef _MEAN_FUNCTION_ZERO_HPP_
#define _MEAN_FUNCTION_ZERO_HPP_

#include "../data/typetraits.hpp"
#include "../data/trainingdata.hpp"
#include "../data/trainingdatasettable.hpp"

namespace GP{

/**
	* @class		MeanZero
	* @brief		A zero mean function.
	* 				It inherits from TrainingDataSetter
	* 				to be able to set a training data.
	* @author	Soohwan Kim
	* @date		26/03/2014
	*/
template<typename Scalar>
class MeanZero : public TrainingDataSettable<Scalar>
{
public:
	// Hyperparameters
	typedef	TypeTraits<Scalar>::Hyp0		Hyp;

	/**
		* @brief	The mean vector at the training positions. f(X)
		* @param [in] pdCoord	(Optional) flag for derivatives (-1: function value, 1: function derivative)
		* @return	The mean vector.
		*/
	VectorPtr operator()(const Hyp &logHyp, const int pdCoord = -1) const
	{
		// number of training data
		const int n = m_pTrainingData->N();
		VectorPtr mu(new Vector(n));
		mu->setZero();
		return mu;
	}

	/**
		* @brief	The mean vector at the test positions. f(X*)
		* @return	The mean vector.
		*/
	//VectorPtr operator()(const TestPositionsConstPtr pXs, const Hyp &logHyp) const
	VectorPtr operator()(const MatrixConstPtr pXs, const Hyp &logHyp) const
	{
		// number of training data
		const int m = pXs->M();
		VectorPtr mu(new Vector(m));
		mu->setZero();
		return mu;
	}
};

}

#endif