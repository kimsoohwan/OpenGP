#ifndef _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_
#define _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_

#include "../data/trainingdata.hpp"

namespace GP{

 /**
	* @class		CovSEIso
	* @brief		Isotropic squared exponential covariance function.
	* 				It inherits from TrainingDataSetter
	* 				to be able to set a training data.
	* 				k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
	* @author	Soohwan Kim
	* @date		26/03/2014
	*/
template<typename Scalar>
class CovSEIso : public TypeTraits<Scalar>
{
public:
	/**
	 * @typedef	Hyp2 Hyp
	 * @brief	Defines an alias representing the hyperparameters; ell, sigma_f.
	 */
	typedef	Hyp2	Hyp;

	/**
	 * @brief	K: Self covariance matrix between the training data.
	 * 			[(K), K* ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, K**]
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
		assert(pdHypIndex < logHyp.size());

		// The pairwise squared distances between the trainig inputs
		// is already calculated in m_pSqDist when the training data was set.
		return K(logHyp, trainingData.sqDist(), pdHypIndex);
	}

	/**
	 * @brief	K*: Cross covariance matrix between the training data and test data.
	 * 			[K,    (K*)]: covariance matrix of the marginal Gaussian distribution 
	 * 			[(K*T), K**]
	 * 			Note that no pdHypIndex parameter is passed,
	 * 			because the partial derivatives of the covariance matrix
	 * 			is only required for learning hyperparameters.
	 * @param	[in] logHyp 			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] trainingData 	The training data.
	 * @param	[in] pXs 				The test inputs.
	 * @return	An NxM matrix pointer.
	 * 			N: The number of training data.
	 * 			M: The number of test data.
	 */
	static MatrixPtr Ks(const Hyp &logHyp, const TrainingData<Scalar> &trainingData, const MatrixConstPtr pXs)
	{
		// Calculate the cross covariance matrix
		// given the pairwise squared distances
		// between the training inputs and test inputs.
		return K(logHyp, trainingData.sqDist(pXs));
	}

	/**
	 * @brief	K**: Self [co]variance matrix between the test data.
	 * 			[K,    K*  ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, (K**)]
	 * 			Note that no pdHypIndex parameter is passed,
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
		// The number of test data.
		const int M = pXs->rows();

		// Some constant values.
		const Scalar sigma_f2 = exp(static_cast<Scalar>(2.0) * logHyp(1)); // sigma_f^2

		// Output.
		MatrixPtr pKss;

		// K: self-variance vector (Mx1).
		if(fVarianceVector)
		{
			// k(x, x') = sigma_f^2
			pKss.reset(new Matrix(M, 1));
			pKss->fill(sigma_f2);
		}
		// K: self-covariance matrix (MxM).
		else					
		{
			// Calculate the pairwise squared distances
			// between the test inputs.
			MatrixPtr pSqDist = PairwiseOp<Scalar>::sqDist(pXs);

			// Calculate the covariance matrix
			// given the pairwise squared distances.
			pKss = K(logHyp, pSqDist);
		}

		return pKss;
	}

protected:
	/**
	 * @brief	This is the core function of this class.
	 * 			Calculates the covariance matrix, given the pairwise squared distances.
	 * 			Designed to be used for calculating any covariance matrix, K, K*, K**.
	 * @param	[in] logHyp 	The log hyperparameters, log([ell, sigma_f])
	 * @param	[in] pSqDist 	The pairwise squared distances.
	 * @param	[in] pdHypIndex	(Optional) Hyperparameter index.
	 * 									It returns the partial derivatives of the covariance matrix
	 * 									with respect to this hyperparameter. 
	 * 									The partial derivatives are required for learning hyperparameters.
	 * 									(Example) pdHypIndex = 0: pd[K]/pd[log(ell)], pdHypIndex = 1: pd[K]/pd[log(sigma_f)]
	 * 									(Default = -1) K
	 * @return	An matrix pointer of the same size of the pairwise squared distance matrix.
	 */
	static MatrixPtr K(const Hyp &logHyp, const MatrixConstPtr pSqDist, const int pdHypIndex = -1)
	{
		// pdHypIndex should be greater than the number of hyperparameters
		assert(pdHypIndex < 2); // logHyp.size() == 2;

		// Output.
		// K: of the same size as the pairwise squared distance matrix
		// NxN for K, NxM for Ks
		MatrixPtr pK(new Matrix(pSqDist->rows(), pSqDist->cols()));

		// some constant values
		const Scalar inv_ell2					= exp(static_cast<Scalar>(-2.0) * logHyp(0));	// 1/ell^2
		const Scalar sigma_f2					= exp(static_cast<Scalar>( 2.0) * logHyp(1));	// sigma_f^2
		const Scalar neg_half_inv_ell2		= static_cast<Scalar>(-0.5) * inv_ell2;			// -1/(2*ell^2)
		const Scalar sigma_f2_inv_ell2		= sigma_f2 * inv_ell2;									// sigma_f^2/ell^2
		const Scalar twice_sigma_f2			= static_cast<Scalar>(2.0) * sigma_f2;				// 2*sigma_f^2

		// mode
		switch(pdHypIndex)
		{
		// pd[k]/pd[log(ell)]: partial derivative of covariance function w.r.t log(ell).
		case 0:
			{
				//				 k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				// pd[k]/pd[log(ell)] = sigma_f^2 * exp(-r^2/(2*ell^2)) * (r^2/ell^3) * ell
				//					       = sigma_f^2 * exp(-r^2/(2*ell^2)) * (r^2/ell^2)
				(*pK).noalias() = sigma_f2_inv_ell2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp() * (*pSqDist).array()).matrix();
				break;
			}

		// pd[k]/pd[log(sigma_f)]: partial derivative of covariance function w.r.t log(sigma_f).
		case 1:
			{
				//			        k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				// pd[k]/pd[log(sigma_f)] = 2 * sigma_f * exp(-r^2/(2*ell^2)) * sigma_f
				//								  = 2 * sigma_f^2 * exp(-r^2/(2*ell^2))
				(*pK).noalias() = twice_sigma_f2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp()).matrix();
				break;
			}

		// k: covariance function.
		default:
			{
				// k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				(*pK).noalias() = sigma_f2 * (neg_half_inv_ell2 * (*pSqDist)).array().exp().matrix();
				break;
			}
		}

		return pK;
	}

};


}

#endif