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
	typedef	typename MeanFunc<Scalar>::Hyp	Mean;
	typedef	typename CovFunc<Scalar>::Hyp		Cov;
	typedef	typename LikFunc<Scalar>::Hyp		Lik;


	int size() const
	{
		return mean.size() + cov.size() + lik.size();
	}

	Mean	mean;	/// Logarithm of the hyperparameter of the mean function
	Cov	cov;	/// Logarithm of the hyperparameter of the covariance function
	Lik	lik;	/// Logarithm of the hyperparameter of the likelihood function
};

}

#endif