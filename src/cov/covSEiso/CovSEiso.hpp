#ifndef _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_
#define _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_

#include "../../util/macros.h"
#include "../../data/TrainingData.hpp"

namespace GP{

/**
 * @class		CovSEiso
 * @brief		Squared Exponential Covariance Function
 *					\f[
 *					k(\mathbf{x}, \mathbf{z}) = \sigma_f^2 \exp\left(-\frac{r^2}{2l^2}\right), r = |\mathbf{x}-\mathbf{z}|
 *					\f]
 *					All covariance classes should have public static member functions.
 *					<CENTER>
 *					Public Static Member Functions | Corresponding Mathematical Equations
 *					-------------------------------|-------------------------------------
 *					CovSEiso::K							 | \f$\mathbf{K} = \mathbf{K}(\mathbf{X}, \mathbf{X})\f$
 *					CovSEiso::Ks						 | \f$\mathbf{K}_* = \mathbf{K}(\mathbf{X}, \mathbf{X}_*)\f$
 *					CovSEiso::Kss						 | \f$[\mathbf{k}_{**}]_i = k([\mathbf{X}_*]_i, \mathbf{X}_*]_i)\f$ or \f$\mathbf{K}_{**} = \mathbf{K}(\mathbf{X}_*, \mathbf{X}_*)\f$
 *					</CENTER>
 *					for \f[
 *					\mathbf{K} = 
 *					\begin{bmatrix}
 *					\mathbf{K} & \mathbf{k}_*\\ 
 *					\mathbf{k}_*^\text{T} & k_{**}
 *					\end{bmatrix}
 *					\text{, or }
 *					\mathbf{K} = 
 *					\begin{bmatrix}
 *					\mathbf{K} & \mathbf{K}_*\\ 
 *					\mathbf{K}_*^\text{T} & \mathbf{K}_{**}
 *					\end{bmatrix}
 *					\f]
 *					Also, no covariance class contains any data.
 *					Instead, data are stored in data classes such as
 *					-# TrainingData
 *					-# DerivativeTrainingData
 *					-# TestData
 *					.
 *					Assertions are checked only in those public static member functions
 *					which can be accessed outside.
 * @ingroup		Cov
 * @tparam		Scalar	Datatype such as float and double
 * @author		Soohwan Kim
 * @date			26/03/2014
 */
template<typename Scalar>
class CovSEiso
{
// define types
protected:	TYPE_DEFINE_MATRIX(Scalar);

/**@detailed
  * - Hyp(0) = \f$\log(l)\f$
  * - Hyp(1) = \f$\log(\sigma_f)\f$					.
  */
public:		TYPE_DEFINE_HYP(Scalar, 2);

// public static member functions
public:

	/**
	 * @brief	Self covariance matrix between the training data or
	 *				its partial derivative with respective to the i-th hyperparameter\n
	 *				Only this function returns partial derivatives
	 *				since they are used for learning hyperparameters with training data.
	 * @note		The public member functions, CovSEiso::K, CovSEiso::Ks and CovSEiso::Kss call 
	 *				a protected general member function, CovSEiso::K(const Hyp, const MatrixConstPtr, const int)
	 *				which only depends on pair-wise squared distances.
	 * @param	[in] logHyp 			The log hyperparameters, \f$\log([l, \sigma_f])\f$
	 * @param	[in] trainingData 	The training data
	 * @param	[in] pdHypIndex		(Optional) Hyperparameter index\n
	 * 										- pdHypIndex = -1 (default): return \f$\mathbf{K}(\mathbf{X}, \mathbf{X})\f$
	 *											- pdHypIndex =  0: return \f$\frac{\partial \mathbf{K}}{\partial \log(l)}\f$
	 *											- pdHypIndex =  1: return \f$\frac{\partial \mathbf{K}}{\partial \log(\sigma_f)}\f$
	 * @return	An NxN matrix pointer\n
	 * 			N: The number of training data
	 */
	static MatrixPtr K(const Hyp					&logHyp, 
							 TrainingData<Scalar>	&trainingData, 
							 const int					pdHypIndex = -1) 
	{
		// Assertions only in the begining of the public static member functions which can be accessed outside.
		// The hyparparameter index should be less than the number of hyperparameters
		assert(pdHypIndex < logHyp.size());

		// K(r)
		return K(logHyp, trainingData.pSqDistXX(), pdHypIndex);
	}

	/**
	 * @brief	Cross covariance matrix between the training data and test data, Ks(x, x*)
	 * @note		The public member functions, CovSEiso::K, CovSEiso::Ks and CovSEiso::Kss call 
	 *				a protected general member function, CovSEiso::K(const Hyp, const MatrixConstPtr, const int)
	 *				which only depends on pair-wise squared distances.
	 * @param	[in] logHyp 			The log hyperparameters, \f$\log([l, \sigma_f])\f$
	 * @param	[in] trainingData 	The training data
	 * @param	[in] testData 			The test data
	 * @return	An NxM matrix pointer, \f$\mathbf{k}_* = \mathbf{K}(\mathbf{X}, \mathbf{x}_*)\f$ or \f$\mathbf{K}_* = \mathbf{K}(\mathbf{X}, \mathbf{X}_*)\f$\n
	 * 			N: The number of training data\n
	 * 			M: The number of test data
	 */
	static MatrixPtr Ks(const Hyp								&logHyp, 
							  const TrainingData<Scalar>		&trainingData, 
							  const TestData<Scalar>			&testData)
	{
		// K(r)
		return K(logHyp, trainingData.pSqDistXXs(testData));
	}

	/**
	 * @brief	Self [co]variance matrix between the test data, Kss(x*, x*)
	 * @param	[in] logHyp 				The log hyperparameters, log([ell, sigma_f])
	 * @param	[in] testData 				The test data
	 * @param	[in] fVarianceVector		Flag for the return value
	 * @return	An Mx1 matrix pointer (fVarianceVector == true)
	 * 			An MxM matrix pointer (fVarianceVector == false)
	 * 			M: The number of test data.
	 */
	static MatrixPtr Kss(const Hyp						&logHyp, 
								const TestData<Scalar>		&testData, 
								const bool						fVarianceVector = true)
	{
		// The number of test data.
		const int M = testData.M();

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
			// K(r)
			pKss = K(logHyp, PairwiseOp<Scalar>::sqDist(testData.pXs()));
		}

		return pKss;
	}

protected:
	/**
	 * @brief	Covariance matrix given pair-wise squared distances
	 * @note		This is the core function of this class which is called from
	 *				other public static member functions, K(x, x), Ks(x, x*) and Kss(x*, x*).
	 * @param	[in] logHyp 			The log hyperparameters, log([ell, sigma_f])
	 * @param	[in] pSqDist 			The pairwise squared distances
	 * @param	[in] pdHypIndex		(Optional) Hyperparameter index
	 * 										pdHypIndex = -1: K(x, x),  (default)
	 *											pdHypIndex =  0: pd[K]/pd[log(ell)],
	 *											pdHypIndex =  1: pd[K]/pd[log(sigma_f)]
	 * @return	An matrix pointer of the same size of the pairwise squared distance matrix
	 */
	static MatrixPtr K(const Hyp						&logHyp, 
							 const MatrixConstPtr		pSqDist, 
							 const int						pdHypIndex = -1)
	{
		// Output
		// K: of the same size as the pairwise squared distance matrix
		// K(x, x):     NxN
		// Ks(x, x*):   NxM
		// Kss(x*, x*): MxM
		MatrixPtr pK(new Matrix(pSqDist->rows(), pSqDist->cols()));

		// some constant values
		const Scalar inv_ell2					= exp(static_cast<Scalar>(-2.f) * logHyp(0));	// 1/ell^2
		const Scalar sigma_f2					= exp(static_cast<Scalar>( 2.f) * logHyp(1));	// sigma_f^2
		const Scalar neg_half_inv_ell2		= static_cast<Scalar>(-0.5f) * inv_ell2;			// -1/(2*ell^2)
		const Scalar sigma_f2_inv_ell2		= sigma_f2 * inv_ell2;									// sigma_f^2/ell^2
		const Scalar twice_sigma_f2			= static_cast<Scalar>(2.f) * sigma_f2;				// 2*sigma_f^2

		// hyperparameter index for partial derivatives
		switch(pdHypIndex)
		{
		// pd[k]/pd[log(ell)]: partial derivative of covariance function w.r.t log(ell).
		case 0:
			{
				//				 k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				// pd[k]/pd[log(ell)] = sigma_f^2 * exp(-r^2/(2*ell^2)) * (r^2/ell^3) * ell
				//					       = sigma_f^2 * exp(-r^2/(2*ell^2)) * (r^2/ell^2)
				pK->noalias() = sigma_f2_inv_ell2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp() * (*pSqDist).array()).matrix();
				break;
			}

		// pd[k]/pd[log(sigma_f)]: partial derivative of covariance function w.r.t log(sigma_f).
		case 1:
			{
				//			        k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				// pd[k]/pd[log(sigma_f)] = 2 * sigma_f * exp(-r^2/(2*ell^2)) * sigma_f
				//								  = 2 * sigma_f^2 * exp(-r^2/(2*ell^2))
				pK->noalias() = twice_sigma_f2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp()).matrix();
				break;
			}

		// k: covariance function.
		default:
			{
				// k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				pK->noalias() = sigma_f2 * (neg_half_inv_ell2 * (*pSqDist)).array().exp().matrix();
				break;
			}
		}

		return pK;
	}

};


}

#endif