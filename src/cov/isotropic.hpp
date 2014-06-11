#ifndef _ISOTROPIC_COVARIANCE_FUNCTION_HPP_
#define _ISOTROPIC_COVARIANCE_FUNCTION_HPP_

#include "../data/trainingdata.hpp"

namespace GP {

 /**
	* @class		Isotropic
	* @brief		Isotropic covariance function.
	*				k(x, x') = k(r) = sigma_f^2 f(s), s = s(r), r = |x-x'|
	*
	*				An isotropic covariance functions is defined with f(s) and s(r).
	*				For example, a squared exponential is
	*				f(s) = exp(s), s(r) = - r^2/(2*ell)
	*
	*				For calculating derivatives of k(x, x') 
	*				w.r.t hyperparameters, log(ell), log(sigma_f),
	*
	*				    dk         dk         dell              dk                        df      ds            dk      ds
	*          ---------- = ------ * ----------- = ell * ------ = ell * sigma_f^2 * ---- * ------ = ell * ---- * ------
	*           dlog(ell)    dell     dlog(ell)           dell                       ds     dell           ds     dell
	*				
	*				     dk             dk          dsigma_f                     dk
	*          --------------- = ---------- * --------------- = sigma_f * ---------- = 2*k
	*           dlog(sigma_f)     dsigma_f     dlog(sigma_f)               dsigma_f
	*
	*				which requires to define dk/ds and ds/dell.
	* @author	Soohwan Kim
	* @date		26/03/2014
	*/
template<typename Scalar, template<typename> class CovBase>
class Isotropic : public CovBase<Scalar>
{
	/**
	 * @brief	K: Self covariance matrix between the training data.
	 * 			[(K), K* ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, K**]
	 *          supports three calculations: K, dK_dlog(ell), and dK_dlog(sigma_f)
	 * @note		CRTP (Curiously Recursive Template Pattern)
	 *				This class takes the corresponding covariance function class as a template.
	 *				Thus, Cov<Scalar>::f(), Cov<Scalar>::s(), Cov<Scalar>::dk_ds(), and Cov<Scalar>::ds_dell()
	 *				should be accessable from this class.
	 *				In other words, they should be public, or this class and Cov<Scalar> should be friends.
	 * @param	[in] logHyp 			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] trainingData 	The training data.
	 * @param	[in] pdHypIndex		(Optional) Hyperparameter index.
	 * 										It returns the partial derivatives of the covariance matrix
	 * 										with respect to this hyperparameter. 
	 * 										The partial derivatives are required for learning hyperparameters.
	 * 										(Example) pdHypIndex = 0: pd[K]/pd[log(ell)], pdHypIndex = 1: pd[K]/pd[log(sigma_f)]
	 * 										(Default = -1) K
	 * @return	An NxN matrix pointer.
	 * 			N: The number of training data.
	 */
	static MatrixPtr K(const Hyp &logHyp, TrainingData<Scalar> &trainingData, const int pdHypIndex = -1) 
	{
		// assertion: only once at the public functions
		assert(pdHypIndex < logHyp.size());

		return k(logHyp, pSqDist, pdHypIndex);
	}

	/**
	 * @brief	K*: Cross covariance matrix between the training data and test data.
	 * 			[K,    (K*)]: covariance matrix of the marginal Gaussian distribution 
	 * 			[(K*T), K**]
	 * @note		No pdHypIndex parameter is passed,
	 * 			because the partial derivatives of the covariance matrix
	 * 			is only required for learning hyperparameters.
	 * @param	[in] logHyp 			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] trainingData 	The training data.
	 * @param	[in] pXs 				The test inputs.
	 * @return	An NxM matrix pointer.
	 * 			N: The number of training data.
	 * 			M: The number of test data.
	 */
	static MatrixPtr Ks(const Hyp &logHyp, TrainingData<Scalar> &trainingData, const MatrixConstPtr pXs)
	{
		return k(logHyp, trainingData.sqDist(pXs));
	}

	/**
	 * @brief	K**: Self [co]variance matrix between the test data.
	 * 			[K,    K*  ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, (K**)]
	 * @note		No pdHypIndex parameter is passed,
	 * 			because the partial derivatives of the covariance matrix
	 * 			is only required for learning hyperparameters.
	 * @param	[in] logHyp 			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pXs 				The test inputs.
	 * @param	[in] fVarianceVector	Flag for the return value.
	 * @return	fVarianceVector == true : An Mx1 matrix pointer.
	 * 			fVarianceVector == false: An MxM matrix pointer.
	 * 			M: The number of test data.
	 */
	static MatrixPtr Kss(const Hyp &logHyp, const MatrixConstPtr pXs, const bool fVarianceVector = true)
	{
		// constants
		const Scalar sigma_f2 = exp(static_cast<Scalar>(2.0) * logHyp(1)); // sigma_f^2

		// K: self-variance vector (Mx1).
		if(fVarianceVector)
		{			
			// memory allocation
			const int M = pXs->rows();			// number of test data.
			MatrixPtr pK(new Matrix(M, 1));	// Mx1 matrix

			// k(x, x') = sigma_f^2
			pK->fill(sigma_f2);

			return pK;
		}
		else
		{
			// memory allocation
			MatrixPtr pK = f(logHyp, pXs);

			// k(x, x') = sigma_f^2 * f(s)
			(*pK).noalias() = sigma_f2 * (*pK);

			return pK;
		}
	}

protected:
	/**
	 * @brief	K: Self covariance matrix between the training data.
	 * 			[(K), K* ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, K**]
	 *          supports three calculations: K, dK_dlog(ell), and dK_dlog(sigma_f)
	 * @note		CRTP (Curiously Recursive Template Pattern)
	 *				This class takes the corresponding covariance function class as a template.
	 *				Thus, Cov<Scalar>::f(), Cov<Scalar>::s(), Cov<Scalar>::dk_ds(), and Cov<Scalar>::ds_dell()
	 *				should be accessable from this class.
	 *				In other words, they should be public, or this class and Cov<Scalar> should be friends.
	 * @param	[in] logHyp 			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDist 			The squared distance between data, r^2.
	 * @param	[in] pdHypIndex		(Optional) Hyperparameter index.
	 * 										It returns the partial derivatives of the covariance matrix
	 * 										with respect to this hyperparameter. 
	 * 										The partial derivatives are required for learning hyperparameters.
	 * 										(Example) pdHypIndex = 0: pd[K]/pd[log(ell)], pdHypIndex = 1: pd[K]/pd[log(sigma_f)]
	 * 										(Default = -1) K
	 * @return	An NxN matrix pointer.
	 * 			N: The number of training data.
	 */
	static MatrixPtr k(const Hyp &logHyp, const MatrixConstPtr pSqDist, const int pdHypIndex) 
	{
		// derivative of K w.r.t a hyperparameter
		switch(pdHypIndex)
		{
			// derivative w.r.t log(ell)
	 		//	    dk         dk         dell              dk                        df      ds            dk      ds
	      // ---------- = ------ * ----------- = ell * ------ = ell * sigma_f^2 * ---- * ------ = ell * ---- * ------
	      //  dlog(ell)    dell     dlog(ell)           dell                       ds     dell           ds     dell
			case 0:
			{
				// constants
				const Scalar ell = exp(logHyp(0)); // ell

				// memory allocation
				MatrixPtr pK = dk_ds(logHyp, pSqDist);

				// dk/dlog(ell) = ell * dk/ds * ds/dell
				(*pK).noalias() = ell*(pK->cwiseProduct(*ds_dell(logHyp, pSqDist)));

				return pK;	
			}

			// derivative w.r.t log(sigma_f)
	 		//	      dk             dk          dsigma_f                     dk
	      // --------------- = ---------- * --------------- = sigma_f * ---------- = 2*k
	      //  dlog(sigma_f)     dsigma_f     dlog(sigma_f)               dsigma_f
			case 1:
			{
				// memory allocation
				MatrixPtr pK = k(logHyp, pSqDist);

				// dk/dlog(sigma_f) = sigma_f * k(x, x')
				(*pK).noalias() = static_cast<Scalar>(2.0) * (*pK);

				return pK;
			}
		}

		// original K

	 	//                                                     r^2
	   // k(r) = sigma_f^2 * f(s), f(s) = exp(s), s(r) = - ---------, r = |x-x'|
	   //                                                   2*ell^2

		// constants
		const Scalar sigma_f2 = exp(static_cast<Scalar>(2.0) * logHyp(1)); // sigma_f^2

		// memory allocation
		MatrixPtr pK = f(logHyp, pSqDist);

		// k(x, x') = sigma_f^2 * f(s)
		(*pK).noalias() = sigma_f2 * (*pK);

		return pK;
	}
};

}

#endif