#ifndef _HYPER_PARAMETERS_HPP_
#define _HYPER_PARAMETERS_HPP_

namespace GP {

/**
 * @class	Hyp
 *
 * @brief	Logarithm of Hyperparameters.
 *
 * @author	Soohwankim
 * @date	26/03/2014
 */
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc>
class Hyp
{
public:
	typedef	typename MeanFunc<Scalar>::Hyp	MeanHyp;
	typedef	typename CovFunc<Scalar>::Hyp		CovHyp;
	typedef	typename LikFunc<Scalar>::Hyp		LikHyp;


	int size() const
	{
		return logMeanHyp.size() + logCovHyp.size() + logLikHyp.size();
	}

	MeanHyp	mean;	/// Logarithm of the hyperparameter of the mean function
	CovHyp	cov;	/// Logarithm of the hyperparameter of the covariance function
	LikHyp	lik;	/// Logarithm of the hyperparameter of the likelihood function
};

}

#endif