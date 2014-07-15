#ifndef _MEAN_FUNCTION_GAUSSIAN_PROCESS_HPP_
#define _MEAN_FUNCTION_GAUSSIAN_PROCESS_HPP_

#include "../util/macros.h"
#include "../data/DerivativeTrainingData.hpp"
#include "../data/TestData.hpp"

namespace GP{

/**
	* @class		MeanGP
	* @brief		A Gaussian process mean function.
	* 				It inherits from TrainingDataSetter
	* 				to be able to set a training data.
	* @author	Soohwan Kim
	* @date		26/03/2014
	*/
template<typename Scalar>
class MeanGP
{
// define matrix types
protected:	TYPE_DEFINE_VECTOR(Scalar);

// define hyperparameters
public:		TYPE_DEFINE_HYP(Scalar, 0); // No hyperparameter

	/**
		* @brief	The mean vector at the training positions. f(X)
	   * @param	[in] logHyp 					The log hyperparameters, nothing for MeanGP.
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
	   * @param	[in] logHyp 			The log hyperparameters, nothing for MeanGP.
	   * @param	[in] pXs 				The test inputs.
		* @return	The mean vector.
		*/
	//VectorPtr operator()(const TestPositionsConstPtr pXs, const Hyp &logHyp) const
	static VectorPtr ms(const Hyp &logHyp, const TestData<Scalar> &testData)
	{
		// Zero vector
		VectorPtr pMu(new Vector(testData.M()));
		pMu->setZero();
		return pMu;
	}
};

//using MeanZeroDerObs = MeanGP;
template<typename Scalar>
class MeanZeroDerObs : public MeanGP<Scalar> {};

}

#endif