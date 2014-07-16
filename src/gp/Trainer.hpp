#ifndef _HYPER_PARAMETER_TRAINER_HPP_
#define _HYPER_PARAMETER_TRAINER_HPP_

#include <limits>								// for std::numeric_limits<Scalar>::infinity()
#include <dlib/optimization.h>			// for dlib::find_min

#include "../util/macros.h"

namespace GP{

// Search Strategy
typedef enum SearchStrategyType
{
	NEWTONS_METHOD_TYPE,
//	TRUST_REGION_TYPE, // requires Hessian
	BOBOYA_TYPE
} SearchStrategyType;

// Newton Method
class	NEWTON_METHOD
{
public:
	static const SearchStrategyType TYPE = SearchStrategyType::NEWTONS_METHOD_TYPE;
};

class CG		: public NEWTON_METHOD	{	public:		typedef dlib::cg_search_strategy			Type; };
class BFGS	: public NEWTON_METHOD	{	public:		typedef dlib::bfgs_search_strategy		Type; };
//class LBFGS	{	public:		typedef dlib::lbfgs_search_strategy		Type; };

//class CG		: public NEWTON_METHOD, public dlib::cg_search_strategy		{};
//class BFGS	: public NEWTON_METHOD, public dlib::bfgs_search_strategy	{};
class LBFGS	: public NEWTON_METHOD, public dlib::lbfgs_search_strategy
{
public:
	typedef LBFGS			Type;
	LBFGS()
		: dlib::lbfgs_search_strategy(10)	// The 10 here is basically a measure of how much memory L-BFGS will use.
	{}
};

//// Trust Region
//class TRUST_REGION
//{
//public:
//	static const SearchStrategyType TYPE = SearchStrategyType::TRUST_REGION_TYPE;
//};

// Boboya
class BOBOYA
{
public:
	typedef LBFGS			Type;
	static const SearchStrategyType TYPE = SearchStrategyType::BOBOYA_TYPE;
};

// BOBOYA

// Stopping Strategy
//class DeltaFunc : public dlib::objective_delta_stop_strategy {};
//class GradientNorm : public dlib::gradient_norm_stop_strategy {};
class DeltaFunc		{	public:		typedef dlib::objective_delta_stop_strategy	Type; };
class GradientNorm	{	public:		typedef dlib::gradient_norm_stop_strategy		Type; };
class MaxFuncEval : public DeltaFunc {};

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
// define OptimizationMode
public:
	enum OptimizationMethod
	{
		NORMAL,
		TRUST_REGION
	};

	enum DerivativeType
	{
		EXACT_DERIVATIVES,
		APPROXIMATE_DERIVATIVES
	};

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
				  int					maxIter,
				  const double		minValue,
				  const bool		fExactDerivatives = true)
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
		switch(SearchStrategy::TYPE)
		{
			// NEWTONS_METHOD_TYPE
			case NEWTONS_METHOD_TYPE:
			{
				if(fExactDerivatives)
				{
					// find minimum
					if(maxIter <= 0) // max_iter can't be 0
					{
						dlib::find_min(SearchStrategy::Type(),
											StoppingStrategy::Type(minValue).be_verbose(),
											NlZ(m_generalTrainingData), 
											DnlZ(m_generalTrainingData),
											hypDlib,
											-std::numeric_limits<DlibScalar>::infinity());
					}
					else
					{
						dlib::find_min(SearchStrategy::Type(),
											StoppingStrategy::Type(minValue, maxIter).be_verbose(),
											NlZ(m_generalTrainingData), 
											DnlZ(m_generalTrainingData),
											hypDlib,
											-std::numeric_limits<DlibScalar>::infinity());
					}
				}
				else
				{
					// find minimum
					if(maxIter <= 0) // max_iter can't be 0
					{
						dlib::find_min_using_approximate_derivatives(SearchStrategy::Type(),
																					StoppingStrategy::Type(minValue).be_verbose(),
																					NlZ(m_generalTrainingData), 
																					hypDlib,
																					-std::numeric_limits<DlibScalar>::infinity());
					}
					else
					{
						dlib::find_min_using_approximate_derivatives(SearchStrategy::Type(),
																					StoppingStrategy::Type(minValue, maxIter).be_verbose(),
																					NlZ(m_generalTrainingData), 
																					hypDlib,
																					-std::numeric_limits<DlibScalar>::infinity());
					}
				}
				break;
			}

			//// TRUST_REGION_TYPE
			//case TRUST_REGION_TYPE:
			//{
			//	// set training data
			//	NlZ							nlZ(m_generalTrainingData);

			//	// find minimum
			//	dlib::find_min_trust_region(StoppingStrategy::Type(minValue, maxIter).be_verbose(),
			//										 nlZ, 
			//										 hypDlib,
			//										 10); \\ trust region radius
			//	break;
			//}

			// BOBOYA_TYPE
			case BOBOYA_TYPE:
			{
				if(maxIter <= 0) maxIter = 1000;

				// find minimum
				try
				{
					dlib::find_min_bobyqa(NlZ(m_generalTrainingData), 
												 hypDlib, 
												 9,    // number of interpolation points
												 dlib::uniform_matrix<double>(hypDlib.nr(), 1, -1e100),  // lower bound constraint
												 dlib::uniform_matrix<double>(hypDlib.nr(), 1,  1e100),  // upper bound constraint
												 1,    // initial trust region radius: 10
												 1e-15,  // stopping trust region radius: 1e-6
												 maxIter    // max number of objective function evaluations
												 );
				}
				catch(const std::exception& e)
				{
					std::cout << e.what() << std::endl;
				}
				break;
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