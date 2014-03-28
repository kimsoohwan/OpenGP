#ifndef _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_
#define _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_

#include "usingsqdist.hpp"

namespace GP{

/**
	* @class		CovSEIso
	* @brief		Isotropic squared exponential covariance function
	* 				It inherits from TrainingDataSetter
	* 				to be able to set a training data.
	* 				k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
	* @author	Soohwankim
	* @date		26/03/2014
	*/
template<typename Scalar>
class CovSEIso : public UsingSqDist<Scalar>
{
public:
	/**
	 * @typedef	Hyp2 Hyp
	 * @brief	Defines an alias representing the hyperparameters; ell, sigma_f
	 */
	typedef	Hyp2	Hyp;

	/**
	 * @brief	K:NxN, Self covariance matrix between the training data.
	 * 			[(K), K* ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, K**]
	 * @param	[in] logHyp 	The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pdIndexHyp	(Optional) Flag for partial derivatives of the covariance matrix
	 * 									with respect to this index of hyperparameters. 
	 * 									Needed for learning hyperparameters.
	 * 									(Example) pdIndexHyp = 0: pd[K]/pd[log(sigma_f)], pdIndexHyp = 1: pd[K]/pd[log(ell)]
	 * 									(Default = -1) K
	 * @return	An NxN matrix pointer.
	 * 			N: the number of training data
	 */
	MatrixPtr K(const Hyp &logHyp, const int pdIndexHyp = -1) const 
	{
		// The pairwise squared distances between the trainig inputs
		// is already calculated in m_pSqDist when the training data was set.
		return K(m_pSqDist, logHyp, pdIndexHyp);
	}

	/**
	 * @brief	K*:NxM, Cross covariance matrix between the training data and test data.
	 * 			[K,    (K*)]: covariance matrix of the marginal Gaussian distribution 
	 * 			[(K*T), K**]
	 * 			Note that no pdIndexHyp parameter is passed,
	 * 			because the partial derivatives of the covariance matrix
	 * 			is only required for learning hyperparameters.
	 * @param	[in] pXs 		The test inputs.
	 * @param	[in] logHyp 	The log hyperparameters, log([ell, sigma_f]).
	 * @return	An NxM matrix pointer.
	 * 			N: the number of training data
	 * 			M: the number of test data
	 */
	MatrixPtr Ks(const MatrixConstPtr pXs, const Hyp &logHyp) const
	{
		// Calculate the pairwise squared distances
		// between the training inputs and test inputs.
		MatrixPtr pSqDist = PariwiseOp<Scalar>::sqDist(m_pTrainingData->pX(), pXs); // NxM

		// Calculate the cross covariance matrix
		// given the pairwise squared distances.
		return K(pSqDist, logHyp);
	}

	/**
	 * @brief	K**:MxM, Self [co]variance matrix between the test data.
	 * 			[K,    K*  ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, (K**)]
	 * 			Note that no pdIndexHyp parameter is passed,
	 * 			because the partial derivatives of the covariance matrix
	 * 			is only required for learning hyperparameters.
	 * @param	[in] pXs 				The test inputs.
	 * @param	[in] logHyp 			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] fVarianceVector	Flag for the return value.
	 * @return	fVarianceVector == true : An Mx1 matrix pointer.
	 * 			fVarianceVector == false: An MxM matrix pointer.
	 * 			M: the number of test data
	 */
	MatrixPtr Kss(const MatrixConstPtr pXs, const Hyp &logHyp, const bool fVarianceVector = true) const
	{
		// The number of test data
		const int M = pXs->rows();

		// some constant values
		const Scalar sigma_f2 = exp(static_cast<Scalar>(2.0) * logHyp(1)); // sigma_f^2

		// output
		MatrixPtr pKss;

		// K: self-variance vector (Mx1)
		if(fVarianceVector)
		{
			// k(x, x') = sigma_f^2
			pKss.reset(new Matrix(M, 1));
			pKss->fill(sigma_f2);
		}
		// K: self-covariance matrix (MxM)
		else					
		{
			// Calculate the pairwise squared distances
			// between the test inputs
			MatrixPtr pSqDist = PairwiseOp<Scalar>::sqDist(pXs);

			// Calculate the covariance matrix
			// given the pairwise squared distances
			pKss = K(pSqDist, logHyp);
		}

		return pKss;
	}

protected:
	/**
	 * @brief	This is the core function of this class.
	 * 			Calculates the covariance matrix, given the pairwise squared distances.
	 * 			Designed to be used for calculating any covariance matrix, K, K*, K**.
	 * @param	[in] logHyp 	The log hyperparameters, log([ell, sigma_f])
	 * @param	[in] pdIndexHyp	(Optional) Flag for partial derivatives of the covariance matrix
	 * 									with respect to this index of hyperparameters. 
	 * 									Needed for learning hyperparameters.
	 * 									(Example) pdIndexHyp = 0: pd[K]/pd[log(ell)], pdIndexHyp = 1: pd[K]/pd[log(sigma_f)]
	 * 									(Default = -1) K
	 * @return	An matrix pointer of the same size of the pairwise squared distance matrix.
	 */
	MatrixPtr K(const MatrixConstPtr pSqDist, const Hyp &logHyp, const int pdIndexHyp = -1) const
	{
		// pdIndexHyp should be greater than the number of hyperparameters
		assert(pdIndexHyp < 2); // logHyp.size() == 2;

		// Output
		// K: same size of the pairwise squared distance matrix
		// NxN for K, NxM for Ks
		MatrixPtr pK(new Matrix(pSqDist->rows(), pSqDist->cols()));

		// some constant values
		const Scalar inv_ell2					= exp(static_cast<Scalar>(-2.0) * logHyp(0));	// 1/ell^2
		const Scalar sigma_f2					= exp(static_cast<Scalar>( 2.0) * logHyp(1));	// sigma_f^2
		const Scalar neg_half_inv_ell2		= static_cast<Scalar>(-0.5) * inv_ell2;			// -1/(2*ell^2)
		const Scalar sigma_f2_inv_ell2		= sigma_f2 * inv_ell2;									// sigma_f^2/ell^2
		const Scalar twice_sigma_f2			= static_cast<Scalar>(2.0) * sigma_f2;				// 2*sigma_f^2

		// mode
		switch(pdIndexHyp)
		{
		// pd[K]/pd[log(ell)]: derivatives of covariance matrix w.r.t log(ell)
		case 0:
			{
				//				k(x, x')	 = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				// pd[k]/pd[log(ell)] = sigma_f^2 * exp(-r^2/(2*ell^2)) * (r^2/ell^3) * ell
				//					       = sigma_f^2 * exp(-r^2/(2*ell^2)) * (r^2/ell^2)
				(*pK).noalias() = sigma_f2_inv_ell2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp() * (*pSqDist).array()).matrix();
				break;
			}

		// pd[K]/pd[log(sigma_f)]: derivatives of covariance matrix w.r.t log(sigma_f)
		case 1:
			{
				//			        k(X, X') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				// pd[k]/pd[log(sigma_f)] = 2 * sigma_f * exp(-r^2/(2*ell^2)) * sigma_f
				//								  = 2 * sigma_f^2 * exp(-r^2/(2*ell^2))
				(*pK).noalias() = twice_sigma_f2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp()).matrix();
				break;
			}

		// K: covariance matrix
		default:
			{
				// k(X, X') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				(*pK).noalias() = sigma_f2 * (neg_half_inv_ell2 * (*pSqDist)).array().exp().matrix();
				break;
			}
		}

		return pK;
	}
};

}

#endif