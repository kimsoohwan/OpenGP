#ifndef _MEAN_FUNCTION_ZERO_HPP_
#define _MEAN_FUNCTION_ZERO_HPP_

#include "../../util/macros.hpp"
#include "../../data/DerivativeTrainingData.hpp"
#include "../../data/TestData.hpp"

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
class MeanZero
{
// define matrix types
protected:	TYPE_DEFINE_MATRIX(Scalar);

// define hyperparameters
public:		TYPE_DEFINE_HYP(Scalar, 0); // No hyperparameter

	/**
		* @brief	The mean vector at the training positions. f(X)
	   * @param	[in] logHyp 			The log hyperparameters, nothing for MeanZero.
	   * @param	[in] trainingData 	The training data.
		* @return	The mean vector.
		*/
	static VectorPtr m(const Hyp &logHyp, const TrainingData<Scalar> &trainingData)
	{
		// Zero vector
		VectorPtr pMu(new Vector(trainingData.N()));
		pMu->setZero();
		return pMu;
	}

	static VectorPtr m(const Hyp &logHyp, const DerivativeTrainingData<Scalar> &derivativeTrainingData)
	{
		// Zero vector
		const int d		= derivativeTrainingData.D();
		const int n		= derivativeTrainingData.N();
		const int nd	= derivativeTrainingData.Nd();
		const int nn	= n + nd*d;

		VectorPtr pMu(new Vector(nn));
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
	static VectorPtr m(const Hyp &logHyp, const TestData &testData)
	{
		// Zero vector
		VectorPtr pMu(new Vector(testData.M()));
		pMu->setZero();
		return pMu;
	}
};

}

#endif