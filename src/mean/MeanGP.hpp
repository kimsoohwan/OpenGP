#ifndef _MEAN_FUNCTION_GAUSSIAN_PROCESS_HPP_
#define _MEAN_FUNCTION_GAUSSIAN_PROCESS_HPP_

#include "../util/macros.h"
#include "../data/TrainingData.hpp"
//#include "../data/DerivativeTrainingData.hpp"		// TODO
#include "../data/TestData.hpp"
#include "../inf/InfExactGeneral.hpp"	// InfExactGeneral
#include "../gp/GaussianProcess.hpp"	// GaussianProcess

namespace GP{

/**
  * @class		MeanGP
  * @brief		A Gaussian process mean function.
  * @ingroup	-Mean
  * @author		Soohwan Kim
  * @date		10/09/2014
  */
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
			//template<template<typename> class GlobalTrainingData> // TODO
class MeanGP : protected InfMethod<Scalar, MeanFunc, CovFunc, LikFunc>
{
/**@brief Number of hyperparameters */
//public: static const int N = MeanFunc<Scalar>::N + CovFunc<Scalar>::N + LikFunc<Scalar>::N;
public: static const int N = 0;

// define matrix types
//protected:	TYPE_DEFINE_VECTOR(Scalar);
//	typedef	InfExactGeneral<Scalar, MeanFunc, CovFunc, LikFunc>	InfType;
public:		TYPE_DEFINE_MATRIX(Scalar);
				TYPE_DEFINE_VECTOR(Scalar);
				TYPE_DEFINE_CHOLESKYFACTOR();

// define hyperparameters
public:		TYPE_DEFINE_HYP(Scalar, N);	// hyperparameter vector for the local Gaussian process

protected:
	typedef	InfMethod<Scalar, MeanFunc, CovFunc, LikFunc>	InfType;		// a set of hyperparameters for the global Gaussian process
public:
	//typedef	InfType::Hyp	GlobalHyp;		// a set of hyperparameters for the global Gaussian process
	typedef	typename InfType::Hyp	GlobalHyp;		// a set of hyperparameters for the global Gaussian process
	//typedef GP::GaussianProcess<Scalar, MeanFunc, CovFunc, LikFunc, InfExactGeneral>		GPType;
	//typedef GPType::Hyp																						GlobalHyp;

//// define shared pointers
//protected:
//	typedef boost::shared_ptr<TrainingData<Scalar> >					GlobalTrainingDataPtr;
//	typedef boost::shared_ptr<const TrainingData<Scalar> >			GlobalTrainingDataConstPtr;
//	typedef boost::shared_ptr<GlobalHyp>											GlobalHypPtr;
//	typedef boost::shared_ptr<const GlobalHyp>									GlobalHypConstPtr;

public:
	/** @brief	Set the global hyperparameters and training data,
	  *			then precompute the Cholesky factor and alpha
	  * @note	Use protected methods of InfExactGeneral */
	//template <template <typename> class GeneralTrainingData>
	static void set(const GlobalHyp &logGlobalHyp, const DerivativeTrainingData<Scalar> &globalDerivativeTrainingData, const bool fTrainGlobalHyp = false, const int maxIter = 0)
	{
		// global hyperparameters
		s_logGlobalHyp = logGlobalHyp;

		// training data
		s_globalDerivativeTrainingData	= globalDerivativeTrainingData;
		//s_globalDerivativeTrainingData.set(globalDerivativeTrainingData.pX(), globalDerivativeTrainingData.pY());

		// train hyperparameters
		if(fTrainGlobalHyp)
		{
			// log file
			LogFile logFile;
			logFile << "Before training global hyperparameters" << std::endl;
			int j = 0;
			for(int i = 0; i < s_logGlobalHyp.mean.size(); i++)		logFile << "- mean[" << i << "] = " << expf(s_logGlobalHyp.mean(i)) << std::endl;
			for(int i = 0; i < s_logGlobalHyp.cov.size();  i++)		logFile << "- cov["  << i << "] = " << expf(s_logGlobalHyp.cov(i))  << std::endl;
			for(int i = 0; i < s_logGlobalHyp.lik.size();  i++)		logFile << "- lik["  << i << "] = " << expf(s_logGlobalHyp.lik(i))  << std::endl;

			// training
			typedef GaussianProcess<Scalar, MeanFunc, CovFunc, LikFunc, InfMethod> GPType;
			DlibScalar nlZ = GPType::train<BOBYQA, NoStopping>(s_logGlobalHyp, s_globalDerivativeTrainingData, maxIter);

			//logFile << "nlZ = " << nlZ << std::endl;
			logFile << "After training global hyperparameters" << std::endl;
			j = 0;
			for(int i = 0; i < s_logGlobalHyp.mean.size(); i++)		logFile << "- mean[" << i << "] = " << expf(s_logGlobalHyp.mean(i)) << std::endl;
			for(int i = 0; i < s_logGlobalHyp.cov.size();  i++)		logFile << "- cov["  << i << "] = " << expf(s_logGlobalHyp.cov(i))  << std::endl;
			for(int i = 0; i < s_logGlobalHyp.lik.size();  i++)		logFile << "- lik["  << i << "] = " << expf(s_logGlobalHyp.lik(i))  << std::endl;
		}

		// precompute the Cholesky factor and alpha
		const bool fDoNotThrowException = true;
		s_pL = choleskyFactor(s_logGlobalHyp, s_globalDerivativeTrainingData, fDoNotThrowException);
		const VectorConstPtr pY_M = y_m(s_logGlobalHyp.mean, s_globalDerivativeTrainingData);
		s_pAlpha	= alpha(s_pL, pY_M);

		// set the flag
		s_bInitialized = true;
	}

	///** @brief Predict with the global training data */
	//static void predict /* throw (Exception) */
	//						 (TestData<Scalar>				&testData,
	//						  const bool						fVarianceVector = true,
	//						  const int							perBatch = 1000)
	//{
	//	InfType::predict(s_logGlobalHyp, s_globalDerivativeTrainingData, testData, fVarianceVector, perBatch);
	//}

	/**
		* @brief	The mean vector at the training positions. f(X)
	   * @param	[in] logHyp 					The log hyperparameters, nothing for MeanZero.
	   * @param	[in] generalTrainingData 	The training data or derivative training data
		* @return	The mean vector.
		*/
	//template<template<typename> class GeneralLocalTrainingData>
	//static VectorPtr m(const Hyp											&logHyp, 
	//						 const GeneralLocalTrainingData<Scalar>	&generalLocalTrainingData, 
	//						 const int											pdHypIndex = -1)
	template <template <typename> class GeneralTrainingData>
	static VectorPtr m(const Hyp											&logHyp, 
							 const GeneralTrainingData<Scalar>			&localTrainingData, 
							 const int											pdHypIndex = -1)
	{
		// check initialized
		assert(s_bInitialized);

		// set the global test inputs with the local taining inputs
		TestData<Scalar> globalTestData;
		globalTestData.set(localTrainingData.pX());
		//globalTestData.set(generalLocalTrainingData.pX(), generalLocalTrainingData.pXd()); // TODO

		// predict with precalulated
		const bool fVarianceVector = true;
		predict_given_precalculation(s_logGlobalHyp, s_globalDerivativeTrainingData, globalTestData, s_pL, s_pAlpha, fVarianceVector);
		//InfType::predict(s_logGlobalHyp, s_globalDerivativeTrainingData, globalTestData, fVarianceVector);

		// return the global mean
		return globalTestData.pMu();
	}


	/**
		* @brief	The mean vector at the test positions. f(X*)
	   * @param	[in] logHyp 			The log hyperparameters, nothing for MeanGP.
	   * @param	[in] pXs 				The test inputs.
		* @return	The mean vector.
		*/
	static VectorPtr ms(const Hyp &logHyp, const TestData<Scalar> &localTestData)
	{
		// check initialized
		assert(s_bInitialized);

		// set the global test inputs with the local taining inputs
		TestData<Scalar> globalTestData;
		globalTestData.set(localTestData.pXs());

		// predict with precalulated
		const bool fVarianceVector = true;
		predict_given_precalculation(s_logGlobalHyp, s_globalDerivativeTrainingData, globalTestData, s_pL, s_pAlpha, fVarianceVector);
		//InfType::predict(s_logGlobalHyp, s_globalDerivativeTrainingData, globalTestData, fVarianceVector);

		// return the global mean
		return globalTestData.pMu();
	}

//protected:
//
//	/** @brief	Conversion from a Mean hyperparameter vector to a GP hyperparameter set */
//	static inline void MeanHyp2GPHyp(const Hyp			&MeanHyp,
//									  InfType::Hyp		&GPHyp)
//	{
//		int j = 0; // hyperparameter index
//		for(int i = 0; i < GPHyp.mean.size(); i++)		GPHyp.mean(i) = MeanHyp(j++);
//		for(int i = 0; i < GPHyp.cov.size();  i++)		GPHyp.cov(i)  = MeanHyp(j++);
//		for(int i = 0; i < GPHyp.lik.size();  i++)		GPHyp.lik(i)  = MeanHyp(j++);
//	}
//
//	/** @brief	Conversion from a GP hyperparameter set to a Mean hyperparameter vector */
//	static inline void GPHyp2MeanHyp(const InfType::Hyp		&GPHyp,
//									  Hyp							&MeanHyp)
//	{
//		int j = 0; // hyperparameter index
//		for(int i = 0; i < GPHyp.mean.size(); i++)		MeanHyp(j++) = GPHyp.mean(i);
//		for(int i = 0; i < GPHyp.cov.size();  i++)		MeanHyp(j++) = GPHyp.cov(i);
//		for(int i = 0; i < GPHyp.lik.size();  i++)		MeanHyp(j++) = GPHyp.lik(i);
//	}

protected:
	/** @brief Flag for initializing global GP */
	static bool								s_bInitialized;

	/** @brief Training data for global GP */
	static DerivativeTrainingData<Scalar>	s_globalDerivativeTrainingData;

	/** @brief Hyperparameters for global GP */
	static GlobalHyp						s_logGlobalHyp;

	/** @brief Precalculated Cholesky factor for global GP */
	static CholeskyFactorConstPtr		s_pL;

	/** @brief Precalculated alpha for global GP */
	static VectorConstPtr				s_pAlpha;
};

template<typename Scalar, template<typename> class MeanFunc, template<typename> class CovFunc, template<typename> class LikFunc, template <typename, template<typename> class, template<typename> class, template<typename> class> class InfMethod>
bool																													MeanGP<Scalar, MeanFunc, CovFunc, LikFunc, InfMethod>::s_bInitialized = false;

template<typename Scalar, template<typename> class MeanFunc, template<typename> class CovFunc, template<typename> class LikFunc, template <typename, template<typename> class, template<typename> class, template<typename> class> class InfMethod>
DerivativeTrainingData<Scalar>																				MeanGP<Scalar, MeanFunc, CovFunc, LikFunc, InfMethod>::s_globalDerivativeTrainingData;

template<typename Scalar, template<typename> class MeanFunc, template<typename> class CovFunc, template<typename> class LikFunc, template <typename, template<typename> class, template<typename> class, template<typename> class> class InfMethod>
typename MeanGP<Scalar, MeanFunc, CovFunc, LikFunc, InfMethod>::GlobalHyp						MeanGP<Scalar, MeanFunc, CovFunc, LikFunc, InfMethod>::s_logGlobalHyp;

template<typename Scalar, template<typename> class MeanFunc, template<typename> class CovFunc, template<typename> class LikFunc, template <typename, template<typename> class, template<typename> class, template<typename> class> class InfMethod>
typename MeanGP<Scalar, MeanFunc, CovFunc, LikFunc, InfMethod>::CholeskyFactorConstPtr		MeanGP<Scalar, MeanFunc, CovFunc, LikFunc, InfMethod>::s_pL;

template<typename Scalar, template<typename> class MeanFunc, template<typename> class CovFunc, template<typename> class LikFunc, template <typename, template<typename> class, template<typename> class, template<typename> class> class InfMethod>
typename MeanGP<Scalar, MeanFunc, CovFunc, LikFunc, InfMethod>::VectorConstPtr				MeanGP<Scalar, MeanFunc, CovFunc, LikFunc, InfMethod>::s_pAlpha;

}

#endif