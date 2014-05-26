#ifndef _MEAN_FUNCTION_ZERO_HPP_
#define _MEAN_FUNCTION_ZERO_HPP_

#include "../data/typetraits.hpp"
#include "../data/trainingdata.hpp"

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
class MeanZero : public TypeTraits<Scalar>
{
public:
	// Hyperparameters
	typedef	TypeTraits<Scalar>::Hyp0		Hyp;

	/**
		* @brief	The mean vector at the training positions. f(X)
	   * @param	[in] logHyp 			The log hyperparameters, nothing for MeanZero.
	   * @param	[in] trainingData 	The training data.
		* @param [in] pdCoord			(Optional) flag for derivatives (-1: function value, 1: function derivative)
		* @return	The mean vector.
		*/
	VectorPtr operator()(const Hyp &logHyp, const TrainingData<Scalar> &trainingData, const int pdCoord = -1) const
	{
		// Zero vector
		VectorPtr pMu(new Vector(trainingData.N()));
		pMu->setZero();
		return pMu;
	}

	/**
		* @brief	The mean vector at the test positions. f(X*)
	   * @param	[in] logHyp 			The log hyperparameters, nothing for MeanZero.
	   * @param	[in] pXs 				The test inputs.
		* @return	The mean vector.
		*/
	//VectorPtr operator()(const TestPositionsConstPtr pXs, const Hyp &logHyp) const
	VectorPtr operator()(const Hyp &logHyp, const MatrixConstPtr pXs) const
	{
		// Zero vector
		VectorPtr pMu(new Vector(pXs->M()));
		pMu->setZero();
		return pMu;
	}
};

}

#endif