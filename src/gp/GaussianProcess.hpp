#ifndef _GAUSSIAN_PROCESSES_HPP_
#define _GAUSSIAN_PROCESSES_HPP_

#include "Trainer.hpp"

namespace GP{

/**
  * @class		GaussianProcess
  * @brief		A Gaussian process
  * 				It inherits from InfMethod
  * 				to call predict and negativeLogMarginalLikelihood.
  * @ingroup	-GP
  * @author		Soohwan Kim
  * @date		15/07/2014
  */
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
class GaussianProcess : public InfMethod<Scalar, MeanFunc, CovFunc, LikFunc>
{
public:
	//typedef typename InfMethod<Scalar, MeanFunc, CovFunc, LikFunc>::Hyp Hyp;

public:
	/** @brief Train hyperparameters */
	template<class SearchStrategy, class StoppingStrategy, template<typename> class GeneralTrainingData>
	//static DlibScalar train(Hyp			&logHyp,
	static DlibScalar train(typename InfMethod<Scalar, MeanFunc, CovFunc, LikFunc>::Hyp			&logHyp,
	//static DlibScalar train(Hyp<Scalar, MeanFunc, CovFunc, LikFunc>									&logHyp,
									GeneralTrainingData<Scalar>													&generalTrainingData,
									const int																			maxIter = 0,
									const DlibScalar																	minValue = 1e-15)
	{
		assert(generalTrainingData.N() > 0);
		if(generalTrainingData.N() <= 0) return 0.0;

		// conversion from GP hyperparameters to a Dlib vector
		DlibVector logDlib;
		logDlib.set_size(logHyp.size());
		Hyp2Dlib<Scalar, MeanFunc, CovFunc, LikFunc>(logHyp, logDlib);

		// trainer
		DlibScalar minNlZ = TrainerUsingApproxDerivatives<NlZ <Scalar, MeanFunc, CovFunc, LikFunc, InfMethod, GeneralTrainingData> >::train<SearchStrategy, StoppingStrategy>(logDlib, 
																																																		  NlZ <Scalar, MeanFunc, CovFunc, LikFunc, InfMethod, GeneralTrainingData>(generalTrainingData),
																																																		  maxIter, minValue);

		// conversion from a Dlib vector to GP hyperparameters
		Dlib2Hyp<Scalar, MeanFunc, CovFunc, LikFunc>(logDlib, logHyp);

		return minNlZ;
	}
};

}

#endif