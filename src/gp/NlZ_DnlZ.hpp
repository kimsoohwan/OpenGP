#ifndef _NEGATIVE_LOG_MARGINALIZATION_AND_ITS_DERIVATIVES_HPP_
#define _NEGATIVE_LOG_MARGINALIZATION_AND_ITS_DERIVATIVES_HPP_

#include <dlib/optimization.h>			// for dlib::find_min
#include "../util/macros.h"

namespace GP {

//typedef	Scalar																	DlibScalar;
typedef	double																		DlibScalar;
typedef	dlib::matrix<DlibScalar, 0, 1>										DlibVector;	

/** @brief	conversion from GP hyperparameters to a Dlib vector */
template <typename Scalar, 
	       template<typename> class MeanFunc, 
			 template<typename> class CovFunc, 
			 template<typename> class LikFunc>
inline void Hyp2Dlib(const Hyp<Scalar, MeanFunc, CovFunc, LikFunc>	&logHyp,
							DlibVector													&logDlib)
{
	int j = 0; // hyperparameter index
	for(int i = 0; i < logHyp.mean.size(); i++)		logDlib(j++, 0) = logHyp.mean(i);
	for(int i = 0; i < logHyp.cov.size();  i++)		logDlib(j++, 0) = logHyp.cov(i);
	for(int i = 0; i < logHyp.lik.size();  i++)		logDlib(j++, 0) = logHyp.lik(i);
}

/** @brief	conversion from a Dlib vector to GP hyperparameters */
template <typename Scalar, 
	       template<typename> class MeanFunc, 
			 template<typename> class CovFunc, 
			 template<typename> class LikFunc>
inline void Dlib2Hyp(const DlibVector										&logDlib,
							Hyp<Scalar, MeanFunc, CovFunc, LikFunc>		&logHyp)
{
	int j = 0; // hyperparameter index
	for(int i = 0; i < logHyp.mean.size(); i++)		logHyp.mean(i) = logDlib(j++, 0);
	for(int i = 0; i < logHyp.cov.size();  i++)		logHyp.cov(i)  = logDlib(j++, 0);
	for(int i = 0; i < logHyp.lik.size();  i++)		logHyp.lik(i)  = logDlib(j++, 0);
}


/**
 * @class		Negative log marginalization
 */
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template<typename, 
						template<typename> class,
						template<typename> class,
						template<typename> class> class InfMethod,
			template<typename> class GeneralTrainingData>
class NlZ
{
protected:	
	// define vector types
	TYPE_DEFINE_VECTOR(Scalar);

	// GP
	typedef	typename InfMethod<Scalar, MeanFunc, CovFunc, LikFunc>		InfType;
	typedef	typename InfType::Hyp													Hyp;

public:
	NlZ(GeneralTrainingData<Scalar> &generalTrainingData)
		: m_generalTrainingData(generalTrainingData)
	{
	}

public:
	DlibScalar operator()(const DlibVector &logDlib) const
	{
		// convert a Dlib vector to GP hyperparameters
		Hyp	logHyp;
		Dlib2Hyp<Scalar, MeanFunc, CovFunc, LikFunc>(logDlib, logHyp);
		std::cout << "hyp.mean = " << std::endl << logHyp.mean.array().exp().matrix() << std::endl << std::endl;
		std::cout << "hyp.cov = "  << std::endl << logHyp.cov.array().exp().matrix()  << std::endl << std::endl;
		std::cout << "hyp.lik = "  << std::endl << logHyp.lik.array().exp().matrix()  << std::endl << std::endl;

		// calculate nlZ only
		Scalar			nlZ;
		//GPType::negativeLogMarginalLikelihood(logHyp, 
		InfType::negativeLogMarginalLikelihood(logHyp, 
															m_generalTrainingData,
															nlZ, 
															VectorPtr(),
															1);

		std::cout << "nlz = " << nlZ << std::endl;
		return nlZ;
	}
protected:
	GeneralTrainingData<Scalar> &m_generalTrainingData;
};

/**
 * @class		Derivatives of negative log marginalization with respect to hyperparameters
 */
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template<typename, 
						template<typename> class,
						template<typename> class,
						template<typename> class> class InfMethod,
			template<typename> class GeneralTrainingData>
class DnlZ
{
protected:	
	// define vector types
	TYPE_DEFINE_VECTOR(Scalar);

	// GP
	typedef	typename InfMethod<Scalar, MeanFunc, CovFunc, LikFunc>		InfType;
	typedef	typename InfType::Hyp													Hyp;

public:
	template<template<typename> class GeneralTrainingData>
	DnlZ(GeneralTrainingData<Scalar> &generalTrainingData)
		: m_generalTrainingData(generalTrainingData)
	{
	}

public:
	DlibVector operator()(const DlibVector &logDlib) const
	{
		// convert a Dlib vector to GP hyperparameters
		Hyp	logHyp;
		Dlib2Hyp<Scalar, MeanFunc, CovFunc, LikFunc>(logDlib, logHyp);
		std::cout << "hyp.mean = " << std::endl << logHyp.mean.array().exp().matrix() << std::endl << std::endl;
		std::cout << "hyp.cov = "  << std::endl << logHyp.cov.array().exp().matrix()  << std::endl << std::endl;
		std::cout << "hyp.lik = "  << std::endl << logHyp.lik.array().exp().matrix()  << std::endl << std::endl;

		// calculate dnlZ only
		Scalar			nlZ;
		VectorPtr		pDnlZ;
		InfType::negativeLogMarginalLikelihood(logHyp, 
														m_generalTrainingData,
														nlZ, //Scalar(),
														pDnlZ,
														-1);

		std::cout << "dnlz = " << std::endl << *pDnlZ << std::endl << std::endl;

		DlibVector dnlZ(logHyp.size());
		Eigen2Dlib(pDnlZ, dnlZ);
		return dnlZ;
	}

protected:
	/** @brief	conversion from a Eigen vector to a Dlib vector */
	inline void Eigen2Dlib(const VectorConstPtr	&pVector,
								  DlibVector				&vec) const
	{
		for(int i = 0; i < pVector->size(); i++) vec(i, 0) = (*pVector)(i);
	}

protected:
	GeneralTrainingData<Scalar> &m_generalTrainingData;
};

}

#endif