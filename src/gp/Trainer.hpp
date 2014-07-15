#ifndef _HYPER_PARAMETER_TRAINER_HPP_
#define _HYPER_PARAMETER_TRAINER_HPP_

#include <limits>								// for std::numeric_limits<Scalar>::infinity()
#include <dlib/optimization.h>			// for dlib::find_min

#include "../util/macros.h"

namespace GP{

// Search Strategy
class CG		{	public:		typedef dlib::cg_search_strategy			Type; };
class BFGS	{	public:		typedef dlib::bfgs_search_strategy		Type; };
//class LBFGS	{	public:		typedef dlib::lbfgs_search_strategy		Type; };
class LBFGS	: public dlib::lbfgs_search_strategy
{
public:		typedef LBFGS		Type;
public:
	LBFGS()
		: dlib::lbfgs_search_strategy(10)	// The 10 here is basically a measure of how much memory L-BFGS will use.
	{
	}
};

// Stopping Strategy
class DeltaFunc		{	public:		typedef dlib::objective_delta_stop_strategy		Type; };
class GradientNorm	{	public:		typedef dlib::gradient_norm_stop_strategy			Type; };

// Trainer
// refer to http://dlib.net/optimization_ex.cpp.html
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
protected:
	typedef	typename InfMethod<Scalar, MeanFunc, CovFunc, LikFunc>		InfType;
	typedef	typename InfType::Hyp															Hyp;

	//typedef	Scalar																		DlibScalar;
	typedef	double																			DlibScalar;
	typedef	dlib::matrix<DlibScalar, 0, 1>											DlibVector;	

public:
	template<template<typename> class GeneralTrainingData>
	Trainer(GeneralTrainingData<Scalar> &generalTrainingData)
		: m_generalTrainingData(generalTrainingData)
	{
	}

// inner class
protected:
	// nlZ
	class NlZ
	{
	public:
		template<template<typename> class GeneralTrainingData>
		NlZ(GeneralTrainingData<Scalar> &generalTrainingData)
			: m_generalTrainingData(generalTrainingData)
		{
		}

	public:
		DlibScalar operator()(const DlibVector &hypDlib) const
		{
			// conversion from Dlib to Eigen vectors
			Hyp	hypEigen;
			Dlib2Eigen(hypDlib, hypEigen);
			std::cout << "hyp.mean = " << std::endl << hypEigen.mean.array().exp().matrix() << std::endl << std::endl;
			std::cout << "hyp.cov = " << std::endl << hypEigen.cov.array().exp().matrix() << std::endl << std::endl;
			std::cout << "hyp.lik = " << std::endl << hypEigen.lik.array().exp().matrix() << std::endl << std::endl;

			// calculate nlZ only
			Scalar			nlZ;
			//GPType::negativeLogMarginalLikelihood(hypEigen, 
			InfType::negativeLogMarginalLikelihood(hypEigen, 
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

	// dnlZ
	class DnlZ
	{
	public:
		template<template<typename> class GeneralTrainingData>
		DnlZ(GeneralTrainingData<Scalar> &generalTrainingData)
			: m_generalTrainingData(generalTrainingData)
		{
		}

	public:
		DlibVector operator()(const DlibVector &hypDlib) const
		{
			// conversion from Dlib to Eigen vectors
			Hyp	hypEigen;
			Dlib2Eigen(hypDlib, hypEigen);
			std::cout << "hyp.mean = " << std::endl << hypEigen.mean.array().exp().matrix() << std::endl << std::endl;
			std::cout << "hyp.cov = " << std::endl << hypEigen.cov.array().exp().matrix() << std::endl << std::endl;
			std::cout << "hyp.lik = " << std::endl << hypEigen.lik.array().exp().matrix() << std::endl << std::endl;

			// calculate dnlZ only
			Scalar			nlZ;
			VectorPtr		pDnlZ;
			InfType::negativeLogMarginalLikelihood(hypEigen, 
														  m_generalTrainingData,
														  nlZ, //Scalar(),
														  pDnlZ,
														  -1);

			std::cout << "dnlz = " << std::endl << *pDnlZ << std::endl << std::endl;

			DlibVector dnlZ(hypEigen.size());
			Eigen2Dlib(pDnlZ, dnlZ);
			return dnlZ;
		}
	protected:
		GeneralTrainingData<Scalar> &m_generalTrainingData;
	};

// method
public:

	// train hyperparameters
	template<class SearchStrategy, class StoppingStrategy>
	void train(Hyp					&hypEigen,
				  const int			maxIter,
				  const double		minValue,
				  const bool		fUseApproxDer = true)
	{
		// maxIter
		// [+]:		max iteration criteria on
		// [0, -]:		max iteration criteria off

		// hyperparameters
		DlibVector hypDlib;
		hypDlib.set_size(hypEigen.size());

		// initialization
		Eigen2Dlib(hypEigen, hypDlib);

		// Training
		if(fUseApproxDer)
		{
			// set training data
			NlZ							nlZ(m_generalTrainingData);

			// find minimum
			if(maxIter <= 0)
			{
				dlib::find_min_using_approximate_derivatives(SearchStrategy::Type(),
																			StoppingStrategy::Type(minValue).be_verbose(),
																			nlZ,
																			hypDlib,
																			-std::numeric_limits<DlibScalar>::infinity());
			}
			else
			{
				dlib::find_min_using_approximate_derivatives(SearchStrategy::Type(),
																			StoppingStrategy::Type(minValue, maxIter).be_verbose(),
																			nlZ, 
																			hypDlib,
																			-std::numeric_limits<DlibScalar>::infinity());
			}
		}
		else
		{
			// set training data
			NlZ							nlZ(m_generalTrainingData);
			DnlZ							dnlZ(m_generalTrainingData);


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
	GeneralTrainingData<Scalar> &m_generalTrainingData;
};

}

#endif 