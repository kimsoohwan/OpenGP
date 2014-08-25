#ifndef _COVARIANCE_FUNCTION_MATERN_ISO_HPP_
#define _COVARIANCE_FUNCTION_MATERN_ISO_HPP_

#include "../../util/macros.h"
#include "../../data/TrainingData.hpp"

namespace GP{

/**
 * @class		CovMaterniso
 * @brief		Matern covariance function with isotropic distances, \f$\nu = 3/2\f$
 *					\f[
 *					k(\mathbf{x}, \mathbf{z}) = \sigma_f^2 \left(1+\frac{\sqrt{3}r}{l}\right)\exp\left(-\frac{\sqrt{3}r}{l}\right), \quad r = |\mathbf{x}-\mathbf{z}|
 *					\f]
 *					All covariance classes should have public static member functions as follows.
 *					<CENTER>
 *					Public Static Member Functions | Corresponding Covariance Functions
 *					-------------------------------|-------------------------------------
 *					+CovMaterniso::K			| \f$\mathbf{K} = \mathbf{K}(\mathbf{X}, \mathbf{X}) \in \mathbb{R}^{N \times N}\f$
 *					+CovMaterniso::Ks			| \f$\mathbf{K}_* = \mathbf{K}(\mathbf{X}, \mathbf{Z}) \in \mathbb{R}^{N \times M}\f$
 *					+CovMaterniso::Kss		| \f$\mathbf{k}_{**} \in \mathbb{R}^{M \times 1}, \mathbf{k}_{**}^i = k(\mathbf{Z}_i, \mathbf{Z}_i)\f$ or \f$\mathbf{K}_{**} = \mathbf{K}(\mathbf{Z}, \mathbf{Z}) \in \mathbb{R}^{M \times M}\f$
 *					</CENTER>
 *					where \f$N\f$: the number of training data and \f$M\f$: the number of test data given
 *					\f[
 *						\mathbf{\Sigma} = 
 *						\begin{bmatrix}
 *						\mathbf{K} & \mathbf{k}_*\\ 
 *						\mathbf{k}_*^\text{T} & k_{**}
 *						\end{bmatrix}
 *						\text{,   or   }
 *						\mathbf{\Sigma} = 
 *						\begin{bmatrix}
 *						\mathbf{K} & \mathbf{K}_*\\ 
 *						\mathbf{K}_*^\text{T} & \mathbf{K}_{**}
 *						\end{bmatrix}
 *					\f]
 *
 *					The public static member functions, K, Ks and Kss call 
 *					a protected general member function, K(const Hyp, const MatrixConstPtr, const int)
 *					which only depends on pair-wise squared distances.\n\n
 *
 * 				In addition, no covariance class contains any data.
 *					Instead, data are stored in data classes such as
 *					-# TrainingData
 *					-# DerivativeTrainingData
 *					-# TestData
 *					.
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-Materniso
 * @author		Soohwan Kim
 * @date			25/08/2014
 */
template<typename Scalar>
class CovMaterniso
{
/**@brief Number of hyperparameters */
public: static const int N = 2;

/**@brief Define Matrix, MatrixPtr, MatrixConstPtr */
protected:	TYPE_DEFINE_MATRIX(Scalar);

/**@detailed
  * - Hyp(0) = \f$\log(l)\f$
  * - Hyp(1) = \f$\log(\sigma_f)\f$
  */
public:		TYPE_DEFINE_HYP(Scalar, N);

// public static member functions
public:

	/**
	 * @brief	Self covariance matrix between the training data, K(X, X) or its partial derivative
	 * @note		It calls the protected general member function, 
	 *				CovMaterniso::K(const Hyp, const MatrixConstPtr, const int)
	 *				which only depends on pair-wise absolute distances.\n\n
	 *				Only this function returns partial derivatives
	 *				since they are used for learning hyperparameters with training data.
	 * @param	[in] logHyp 			The log hyperparameters
	 *											- logHyp(0) = \f$\log(l)\f$
	 *											- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] trainingData 	The training data
	 * @param	[in] pdHypIndex		(Optional) Hyperparameter index for partial derivatives
	 * 										- pdHypIndex = -1: return \f$\mathbf{K}(\mathbf{X}, \mathbf{X})\f$ (default)
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
		return K(logHyp, trainingData.pAbsDistXX(), pdHypIndex);
	}

	/**
	 * @brief	Cross covariance matrix between the training and test data, Ks(X, Z)
	 * @note		It calls the protected general member function, 
	 *				CovMaterniso::K(const Hyp, const MatrixConstPtr, const int)
	 *				which only depends on pair-wise absolute distances.
	 * @param	[in] logHyp 				The log hyperparameters
	 *												- logHyp(0) = \f$\log(l)\f$
	 *												- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] trainingData 		The training data
	 * @param	[in] testData 				The test data
	 * @return	An NxM matrix pointer, \f$\mathbf{K}_* = \mathbf{K}(\mathbf{X}, \mathbf{Z})\f$\n
	 * 			N: The number of training data\n
	 * 			M: The number of test data
	 */
	static MatrixPtr Ks(const Hyp								&logHyp, 
							  const TrainingData<Scalar>		&trainingData, 
							  const TestData<Scalar>			&testData)
	{
		// K(r)
		return K(logHyp, trainingData.pAbsDistXXs(testData));
	}

	/**
	 * @brief	Self [co]variance matrix between the test data, Kss(Z, Z)
	 * @param	[in] logHyp 				The log hyperparameters
	 *												- logHyp(0) = \f$\log(l)\f$
	 *												- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] testData 				The test data
	 * @param	[in] fVarianceVector		Flag for the return value
	 * 											- fVarianceVector = true : return \f$\mathbf{k}_{**} \in \mathbb{R}^{M \times 1}, \mathbf{k}_{**}^i = k(\mathbf{Z}_i, \mathbf{Z}_i)\f$ (default)
	 *												- fVarianceVector = false: return \f$\mathbf{K}_{**} = \mathbf{K}(\mathbf{Z}, \mathbf{Z}) \in \mathbb{R}^{M \times M}\f$,\n
    *																					which can be used for Bayesian Committee Machines.
	 * @return	A matrix pointer\n
	 *				- Mx1 (fVarianceVector == true)
	 * 			- MxM (fVarianceVector == false)\n
	 * 			M: The number of test data
	 */
	static MatrixPtr Kss(const Hyp						&logHyp, 
								const TestData<Scalar>		&testData, 
								const bool						fVarianceVector = true)
	{
		// The number of test data
		const int M = testData.M();

		// Some constant values
		const Scalar sigma_f2 = exp(static_cast<Scalar>(2.0) * logHyp(1)); // sigma_f^2

		// Output
		MatrixPtr pKss;

		// K: self-variance vector (Mx1)
		if(fVarianceVector)
		{
			// k(z, z) = sigma_f^2
			pKss.reset(new Matrix(M, 1));
			pKss->fill(sigma_f2);
		}

		// K: self-covariance matrix (MxM)
		else					
		{
			// K(r)
			MatrixPtr pAbsDistXsXs = PairwiseOp<Scalar>::sqDist(testData.pXs()); // MxM
			pAbsDistXsXs->noalias() = pAbsDistXsXs->cwiseSqrt();	
			pKss = K(logHyp, pAbsDistXsXs);
		}

		return pKss;
	}

protected:
	/**
	 * @brief	Covariance matrix given pair-wise squared distances, K(R.^2)
	 * @note		This is the core function of this class which is called from
	 *				other public static member functions, CovMaterniso::K, CovMaterniso::Ks and CovMaterniso::Kss.
	 * @param	[in] logHyp 		The log hyperparameters
	 *										- logHyp(0) = \f$\log(l)\f$
	 *										- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] pAbsDist 		The shared pointer to the pairwise absolute distance matrix
	 * @param	[in] pdHypIndex	(Optional) Hyperparameter index
	 * 									- pdHypIndex = -1: return \f$\mathbf{K}(\mathbf{X}, \mathbf{X})\f$ (default)
	 *										- pdHypIndex =  0: return \f$\frac{\partial \mathbf{K}}{\partial \log(l)}\f$
	 *										- pdHypIndex =  1: return \f$\frac{\partial \mathbf{K}}{\partial \log(\sigma_f)}\f$
	 * @return	A matrix pointer of the same size of the pairwise squared distance matrix
	 */
	static MatrixPtr K(const Hyp						&logHyp, 
							 const MatrixConstPtr		pAbsDist, 
							 const int						pdHypIndex = -1)
	{
		// Output
		// K: of the same size as the pairwise squared distance matrix
		// 1. K(X, X):    NxN
		// 2. Ks(X, X):   NxM
		// 3. Kss(Z, Z):	MxM
		MatrixPtr pK(new Matrix(pAbsDist->rows(), pAbsDist->cols()));

		// some constant values
		const Scalar inv_ell						= exp(static_cast<Scalar>(-1.f) * logHyp(0));	// 1/ell
		const Scalar sigma_f2					= exp(static_cast<Scalar>( 2.f) * logHyp(1));	// sigma_f^2
		const Scalar neg_sqrt3_inv_ell		= static_cast<Scalar>(-1.732050807568877f) * inv_ell;		// -sqrt(3)/ell, sqrt(3) = -1.732050807568877f
		const Scalar twice_sigma_f2			= static_cast<Scalar>(2.f) * sigma_f2;				// 2*sigma_f^2

		// s = -sqrt(3)*r/ell
		pK->noalias() = neg_sqrt3_inv_ell * (*pAbsDist);

		// hyperparameter index for the partial derivatives
		switch(pdHypIndex)
		{
		// pd[k]/pd[log(ell)]: partial derivative of covariance function w.r.t log(ell)
		case 0:
			{
				//				 k(x, z) = sigma_f^2 *(1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell), r = |x-z|
				// pd[k]/pd[log(ell)] = sigma_f^2 * (3*r^2/ell^2) * exp(-sqrt(3)*r/ell)
				pK->noalias() = (sigma_f2 * pK->array().square() * pK->array().exp()).matrix();
				break;
			}

		// pd[k]/pd[log(sigma_f)]: partial derivative of covariance function w.r.t log(sigma_f)
		case 1:
			{
				//					  k(x, z) = sigma_f^2 *(1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell), r = |x-z|
				// pd[k]/pd[log(sigma_f)] = 2 * sigma_f *(1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell)
				pK->noalias() = (twice_sigma_f2 * (static_cast<Scalar>(1.f) - pK->array()) * pK->array().exp()).matrix();
				break;
			}

		// k: covariance function
		default:
			{
				//	k(x, z) = sigma_f^2 *(1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell), r = |x-z|
				pK->noalias() = (sigma_f2 * (static_cast<Scalar>(1.f) - pK->array()) * pK->array().exp()).matrix();
				break;
			}
		}

		return pK;
	}

};


}

#endif