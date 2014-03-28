#ifndef _HYPER_PARAMETERS_HPP_
#define _HYPER_PARAMETERS_HPP_

/**
 * @class	Hyp
 *
 * @brief	Logarithm of Hyperparameters.
 *
 * @author	Soohwankim
 * @date	26/03/2014
 */
template<typename MeanFunc, typename CovFunc, typename LikFunc>
class Hyp
{
public:
		typedef	typename MeanFunc::Hyp		MeanHyp;
		typedef	typename CovFunc::Hyp		CovHyp;
		typedef	typename LikFunc::Hyp		LikHyp;

		MeanHyp	logMeanHyp;	/// Logarithm of the hyperparameter of the mean function
		CovHyp	logCovHyp;	/// Logarithm of the hyperparameter of the covariance function
		LikHyp	logLikHyp;	/// Logarithm of the hyperparameter of the likelihood function
};

#endif