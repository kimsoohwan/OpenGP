#ifndef _MEAN_FUNCTION_ZERO_HPP_
#define _MEAN_FUNCTION_ZERO_HPP_

#include "../util/macros.h"
#include "../data/DerivativeTrainingData.hpp"
#include "../data/TestData.hpp"

namespace GP{

/**
  * @class		MeanZero
  * @brief		A zero mean function.
  * 				It inherits from TrainingDataSetter
  * 				to be able to set a training data.
  * @ingroup	-Mean
  * @author		Soohwan Kim
  * @date		26/03/2014
  */
template<typename Scalar>
class MeanZero
{
/**@brief Number of hyperparameters */
public: static const int N = 0;

// define matrix types
protected:	TYPE_DEFINE_VECTOR(Scalar);

// define hyperparameters
public:		TYPE_DEFINE_HYP(Scalar, N); // No hyperparameter

	/**
		* @brief	The mean vector at the training positions. f(X)
	   * @param	[in] logHyp 					The log hyperparameters, nothing for MeanZero.
	   * @param	[in] generalTrainingData 	The training data or derivative training data
		* @return	The mean vector.
		*/
	template<template<typename> class GeneralTrainingData>
	static VectorPtr m(const Hyp									&logHyp, 
							 const GeneralTrainingData<Scalar>	&generalTrainingData, 
							 const int									pdHypIndex = -1)
	{
		// Zero vector
		VectorPtr pMu(new Vector(generalTrainingData.NN()));
		pMu->setZero();
		return pMu;
	}

	/**
		* @brief	The mean vector at the test positions. f(X*)
	   * @param	[in] logHyp 			The log hyperparameters, nothing for MeanZero.
	   * @param	[in] pXs 				The test inputs.
		* @return	The mean vector.
		*/
	static VectorPtr ms(const Hyp &logHyp, const TestData<Scalar> &testData)
	{
		// Zero vector
		VectorPtr pMu(new Vector(testData.M()));
		pMu->setZero();
		return pMu;
	}
};

//using MeanZeroDerObs = MeanZero;
template<typename Scalar>
class MeanZeroDerObs : public MeanZero<Scalar> {};

}

#endif