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
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template<typename, 
						template<typename> class,
						template<typename> class,
						template<typename> class> class InfMethod,
			template<template<typename> class GlobalTrainingData>
class MeanGP
{
// define matrix types
protected:	TYPE_DEFINE_VECTOR(Scalar);
	typedef	typename InfMethod<Scalar, MeanFunc, CovFunc, LikFunc>		InfType;

// define hyperparameters
public:		TYPE_DEFINE_HYP(Scalar, 0);			// No local hyperparameter
				typedef InfType::Hyp		GlobalHyp;	// global hyperparameters

// define shared pointers
protected:
	typedef boost::shared_ptr<GlobalTrainingData<Scalar> >					GlobalTrainingDataPtr;
	typedef boost::shared_ptr<const GlobalTrainingData<Scalar> >			GlobalTrainingDataConstPtr;
	typedef boost::shared_ptr<GlobalHyp>											GlobalHypPtr;
	typedef boost::shared_ptr<const GlobalHyp>									GlobalHypConstPtr;

public:
	MeanGP(const GlobalTrainingDataPtr pGlobalTrainingData, const GlobalHypConstPtr pGlobalHyp)
		: m_pGlobalTrainingData(pGlobalTrainingData),
		  m_pGlobalHyp(pGlobalHyp)
	{}

	template<template<typename> class GeneralLocalTrainingData>
	static VectorPtr m(const Hyp											&logHyp, 
							 const GeneralLocalTrainingData<Scalar>	&generalLocalTrainingData, 
							 const int											pdHypIndex = -1)
	{
		// set the global test inputs with the local taining inputs
		TestData<Scalar> globalTestData;
		globalTestData.set(generalLocalTrainingData.pX());

		// predict
		InfType::predict /* throw (Exception) */
							 (*m_pGlobalHyp, 
							  *m_pGlobalTrainingData, 
							  globalTestData);

		// return the global mean
		return globalTestData.pMu();
	}


	/**
		* @brief	The mean vector at the test positions. f(X*)
	   * @param	[in] logHyp 			The log hyperparameters, nothing for MeanGP.
	   * @param	[in] pXs 				The test inputs.
		* @return	The mean vector.
		*/
	//VectorPtr operator()(const TestPositionsConstPtr pXs, const Hyp &logHyp) const
	static VectorPtr ms(const Hyp &logHyp, const TestData<Scalar> &localTestData)
	{
		// set the global test inputs with the local taining inputs
		TestData<Scalar> globalTestData;
		globalTestData.set(localTestData.pXs());

		// predict
		InfType::predict /* throw (Exception) */
							 (*m_pGlobalHyp, 
							  *m_pGlobalTrainingData, 
							  globalTestData);

		// return the global mean
		return globalTestData.pMu();
	}

protected:
	static GlobalTrainingData	m_pGlobalTrainingData;
	static GlobalHyp				m_pGlobalHyp;
};

//using MeanZeroDerObs = MeanGP;
template<typename Scalar>
class MeanZeroDerObs : public MeanGP<Scalar> {};

}

#endif