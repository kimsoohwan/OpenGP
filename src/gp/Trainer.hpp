#ifndef _HYPER_PARAMETER_TRAINER_HPP_
#define _HYPER_PARAMETER_TRAINER_HPP_

#include <limits>								// for std::numeric_limits<Scalar>::infinity()
#include <dlib/optimization.h>			// for dlib::find_min

#include "../util/macros.h"

namespace GP{

// Gaussian Process
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
class _GP;

// Search Strategy
class CG		{	public:		typedef dlib::cg_search_strategy			Type; };
class BFGS	{	public:		typedef dlib::bfgs_search_strategy		Type; };
class LBFGS	{	public:		typedef dlib::lbfgs_search_strategy		Type; };

// Stopping Strategy
class DeltaFunc		{	public:		typedef dlib::objective_delta_stop_strategy		Type; };
class GradientNorm	{	public:		typedef dlib::gradient_norm_stop_strategy			Type; };

// Trainer
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod,
			template<typename> class GeneralTrainingData>
class Trainer
{
// define vector types
protected:	TYPE_DEFINE_VECTOR(Scalar);

// typedef
public:
	typedef	typename _GP<Scalar, MeanFunc, CovFunc, LikFunc, InfMethod>		GPType;
	typedef	typename GPType::Hyp															Hyp;

	//typedef	Scalar																		DlibScalar;
	typedef	double																			DlibScalar;
	typedef	dlib::matrix<DlibScalar, 0, 1>											DlibVector;	

public:
	template<template<typename> class GeneralTrainingData>
	Trainer(const GeneralTrainingData<Scalar> &generalTrainingData)
		: m_generalTrainingData(generalTrainingData)

// inner class
public:
	// nlZ
	class NlZ
	{
	public:
		DlibScalar operator()(const DlibVector &hypDlib) const
		{
			// conversion from Dlib to Eigen vectors
			Hyp	hypEigen;
			Dlib2Eigen(hypDlib, hypEigen);

			// calculate nlZ only
			Scalar			nlZ;
			GPType::negativeLogMarginalLikelihood(hypEigen, 
															  m_generalTrainingData,
															  nlZ, 
															  VectorPtr(),
															  1);

			//std::cout << "hyp.mean = " << std::endl << meanLogHyp << std::endl << std::endl;
			//std::cout << "hyp.cov = " << std::endl << covLogHyp << std::endl << std::endl;
			//std::cout << "hyp.lik = " << std::endl << likCovLogHyp << std::endl << std::endl;
			//std::cout << "nlz = " << nlZ << std::endl;
			return nlZ;
		}
	};

	// dnlZ
	class DnlZ
	{
	public:
		DlibVector operator()(const DlibVector &hypDlib) const
		{
			// conversion from Dlib to Eigen vectors
			Hyp	hypEigen;
			Dlib2Eigen(hypDlib, hypEigen);

			// calculate dnlZ only
			VectorPtr		pDnlZ;
			m_gp.negativeLogMarginalLikelihood(hypEigen, 
														  m_generalTrainingData,
														  Scalar(),
														  pDnlZ,
														  -1);

			//std::cout << "hyp.mean = " << std::endl << meanLogHyp << std::endl << std::endl;
			//std::cout << "hyp.cov = " << std::endl << covLogHyp << std::endl << std::endl;
			//std::cout << "hyp.lik = " << std::endl << likCovLogHyp << std::endl << std::endl;
			//std::cout << "dnlz = " << std::endl << *pDnlZ << std::endl << std::endl;

			DlibVector dnlZ(hypEigen.size());
			Eigen2Dlib(pDnlZ, dnlZ);
			return dnlZ;
		}
	};

// method
public:

	// train hyperparameters
	template<class SearchStrategy, class StoppingStrategy>
	void train(Hyp					&hypEigen,
				  const int			maxIter = 0,
				  const double		minValue = 1e-7)
	{
		// maxIter
		// [+]:		max iteration criteria on
		// [0, -]:		max iteration criteria off

		// set training data
		NlZ								nlZ(gp);
		DnlZ							dnlZ(gp);

		// hyperparameters
		DlibVector hypDlib;
		hyp.set_size(hypEigen.size());
		//std::cout << "number of hyperparameters" << std::endl;
		//std::cout << "mean: " << pMeanLogHyp->size() << std::endl;
		//std::cout << "cov: " << pCovLogHyp->size() << std::endl;
		//std::cout << "lik: " << pLikCovLogHyp->size() << std::endl;
		//std::cout << "total number of hyperparameters: " << pMeanLogHyp->size() + pCovLogHyp->size() + pLikCovLogHyp->size() << std::endl;

		// initialization
		Eigen2Dlib(hypEigen, hypDlib);

		// find minimum
		if(maxIter <= 0)
		{
			dlib::find_min(SearchStrategy::Type(),
								StoppingStrategy::Type(minValue).be_verbose(),
								nlZ, 
								dnlZ,
								hypDlib,
								-std::numeric_limits<DlibScalar>::infinity());
		}
		else
		{
			dlib::find_min(SearchStrategy::Type(),
								StoppingStrategy::Type(minValue, maxIter).be_verbose(),
								nlZ, 
								dnlZ,
								hypDlib,
								-std::numeric_limits<DlibScalar>::infinity());
		}

		// conversion  from Dlib to Eigen vectors
		Dlib2Eigen(hypDlib, hypEigen);
	}

protected:
	// conversion between Eigen and Dlib vectors
	inline static void Eigen2Dlib(const Hyp	&hypEigen,
											DlibVector	&hypDlib)
	{
		int j = 0; // hyperparameter index
		for(int i = 0; i < hypEigen.mean.size(); i++)		hypDlib(j++, 0) = hypEigen.mean(i);
		for(int i = 0; i < hypEigen.cov.size();  i++)		hypDlib(j++, 0) = hypEigen.cov(i);
		for(int i = 0; i < hypEigen.lik.size();  i++)		hypDlib(j++, 0) = hypEigen.lik(i);
	}

	inline static void Dlib2Eigen(const DlibVector	&hypDlib,
											Hyp					&hypEigen)
	{
		int j = 0; // hyperparameter index
		for(int i = 0; i < hypEigen.mean.size(); i++)		hypEigen.mean(i) = hypDlib(j++, 0);
		for(int i = 0; i < hypEigen.cov.size();  i++)		hypEigen.cov(i)  = hypDlib(j++, 0);
		for(int i = 0; i < hypEigen.lik.size();  i++)		hypEigen.lik(i)  = hypDlib(j++, 0);
	}

	inline static void Eigen2Dlib(const VectorConstPtr	pVector,
											DlibVector				&vec)
	{
		for(int i = 0; i < pVector->size(); i++) vec(i, 0) = (*pVector)(i);
	}

protected:
	const GeneralTrainingData<Scalar> &m_generalTrainingData;
};

}

#endif 