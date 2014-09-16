#ifndef _HYPER_PARAMETER_TRAINER_HPP_
#define _HYPER_PARAMETER_TRAINER_HPP_

#include <algorithm>
#include <limits>								// for std::numeric_limits<Scalar>::infinity()
#include <dlib/optimization.h>			// for dlib::find_min

#include "../util/macros.h"
#include "../util/LogFile.hpp"
#include "NlZ_DnlZ.hpp"

namespace GP{

/*****************************************************************/
/*                      Search Strategy                          */
/*****************************************************************/

/**
 * @defgroup	-SearchStrategy
 * @brief		Search strategies for the hyperparameter trainer
 * @ingroup		-Trainer
 */ 

/**
 * @defgroup	-NewtonsMethod
 * @brief		Newton's method for search strategies
 * @ingroup		-SearchStrategy
 */ 

/**
 * @class		CG
 * @brief		Conjugate Gradient
 * @ingroup		-NewtonsMethod
 */
class CG		{	public:		typedef dlib::cg_search_strategy			Type; };
//class CG		: public dlib::cg_search_strategy		{};

/**
 * @class		BFGS
 * @ingroup		-NewtonsMethod
 */
class BFGS	{	public:		typedef dlib::bfgs_search_strategy		Type; };
//class BFGS	: public dlib::bfgs_search_strategy		{};

/**
 * @class		LBFGS
 * @ingroup		-NewtonsMethod
 */
class LBFGS	:public dlib::lbfgs_search_strategy
{
public:
	typedef LBFGS			Type;
	LBFGS()
		: dlib::lbfgs_search_strategy(10)	// The 10 here is basically a measure of how much memory L-BFGS will use.
	{}
};
//class LBFGS	{	public:		typedef dlib::lbfgs_search_strategy		Type; };


/**
 * @defgroup	-TrustRegion
 * @brief		Trust region method for search strategies\n
 *					which is currently not implemented
 *					because it requires Hessian
 *					for dlib::find_min_trust_region.
 * @ingroup		-SearchStrategy
 */ 

/**
 * @defgroup	-BOBYQA
 * @brief		Bound Optimization BY Quadratic Approximation
 * @ingroup		-SearchStrategy
 */ 

/**
 * @class		BOBYQA
 * @ingroup		-BOBYQA
 */
class BOBYQA {};


/*****************************************************************/
/*                      Stopping Strategy                        */
/*****************************************************************/

/**
 * @defgroup	-StoppingStrategy
 * @brief		Stopping strategies for the hyperparameter trainer
 * @ingroup		-Trainer
 */ 

/**
 * @class		NoStopping
 * @ingroup		-StoppingStrategy
 */
class NoStopping		{};

/**
 * @class		DeltaFunc
 * @ingroup		-StoppingStrategy
 */
class DeltaFunc		{	public:		typedef dlib::objective_delta_stop_strategy	Type; };
//class DeltaFunc : public dlib::objective_delta_stop_strategy {};

/**
 * @class		GradientNorm
 * @ingroup		-StoppingStrategy
 */
class GradientNorm	{	public:		typedef dlib::gradient_norm_stop_strategy		Type; };
//class GradientNorm : public dlib::gradient_norm_stop_strategy {};


/**
 * @class		Trainer
 * @note			refer to http://dlib.net/optimization_ex.cpp.html
 * @ingroup		-Trainer
 */
template <class NlZ_T, class DnlZ_T>
class TrainerUsingExactDerivatives
{
public:
	TrainerUsingExactDerivatives() {}

// method
public:

	// train hyperparameters
	template<class SearchStrategy, class StoppingStrategy>
	static double train(DlibVector		&hypDlib,
							  NlZ_T				&nlZ,
							  DnlZ_T				&dnlZ,
							  int					maxIter = 0,
							  const double		minValue = 1e-15)
	{
		// maxIter
		// [+]:		max iteration criteria on
		// [0, -]:	max iteration criteria off

		// find minimum
		if(maxIter <= 0) // max_iter can't be 0
		{
			return dlib::find_min(SearchStrategy::Type(),
										 StoppingStrategy::Type(minValue).be_verbose(),
										 nlZ, 
										 dnlZ,
										 hypDlib,
										 -std::numeric_limits<DlibScalar>::infinity());
		}
		else
		{
			return dlib::find_min(SearchStrategy::Type(),
										 StoppingStrategy::Type(minValue, maxIter).be_verbose(),
										 nlZ,
										 dnlZ,
										 hypDlib,
										 -std::numeric_limits<DlibScalar>::infinity());
		}
	}
};


/**
 * @class		Trainer
 * @note			refer to http://dlib.net/optimization_ex.cpp.html
 * @ingroup		-Trainer
 */
template <class NlZ_T>
class TrainerUsingApproxDerivatives
{
public:
	TrainerUsingApproxDerivatives() {}

// method
public:

	// train hyperparameters
	template<class SearchStrategy, class StoppingStrategy>
	static double train(DlibVector			&hypDlib,
							  NlZ_T					&nlZ,
							  int						maxIter,
							  const double			minValue,
							  long					NPT = 0)
	{
		// maxIter
		// [+]:		max iteration criteria on
		// [0, -]:	max iteration criteria off

		// find minimum
		if(maxIter <= 0) // max_iter can't be 0
		{
			return dlib::find_min_using_approximate_derivatives(SearchStrategy::Type(),
																				 StoppingStrategy::Type(minValue).be_verbose(),
																				 nlZ, 
																				 hypDlib,
																				 -std::numeric_limits<DlibScalar>::infinity());
		}
		else
		{
			return dlib::find_min_using_approximate_derivatives(SearchStrategy::Type(),
																				 StoppingStrategy::Type(minValue, maxIter).be_verbose(),
																				 nlZ, 
																				 hypDlib,
																				 -std::numeric_limits<DlibScalar>::infinity());
		}
	}

	// train hyperparameters
	template<>
	static double train<BOBYQA, NoStopping>(DlibVector			&hypDlib,
													  NlZ_T					&nlZ,
													  int						maxIter,
													  const double			minValue,
													  long					NPT)
	{
		// maxIter
		// [+]:		max iteration criteria on
		// [0, -]:		max iteration criteria off

		// Training
		if(maxIter <= 0) maxIter = 1000;

		// find minimum
		try
		{
			// N is the number of parameters
			// it must be at least two
			const long N = hypDlib.size();

			// NPT is the number of interpolation conditions
			// It must be in the interval of [N+2,(N+1)(N+2)/2].
			// Choices that exceed 2*N+1 are not recommended.
			if(NPT == 0) NPT = std::min<long>(static_cast<long>((N+1)*(N+2)/2.f), 2*N+1); // previously fixed as 9

			// train
			return dlib::find_min_bobyqa(nlZ, 
												  hypDlib, 
												  NPT,    // number of interpolation points
												  dlib::uniform_matrix<double>(hypDlib.nr(), 1, -1e100),  // lower bound constraint
												  dlib::uniform_matrix<double>(hypDlib.nr(), 1,  1e100),  // upper bound constraint
												  1,				// initial trust region radius: 1, 10
												  minValue,		// stopping trust region radius: 1e-15, 1e-6
												  maxIter		// max number of objective function evaluations
												  );
		}
		catch(const std::exception& e)
		{
			// log file
			LogFile logFile;
			logFile << e.what() << std::endl;
			return 0.0;
		}
	}
};

}

#endif 