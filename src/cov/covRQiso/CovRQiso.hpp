#ifndef _COVARIANCE_FUNCTION_RATIONAL_QUADRATIC_ISO_HPP_
#define _COVARIANCE_FUNCTION_RATIONAL_QUADRATIC_ISO_HPP_

#include "../../util/macros.h"
#include "../../data/TrainingData.hpp"

namespace GP{

/**
 * @class		CovRQiso
 * @brief		Rational quadratic covariance function with isotropic distances
 *					\f[
 *					k(\mathbf{x}, \mathbf{z}) = \sigma_f^2 \left(1 + \frac{r^2}{2\alpha l^2} \right)^{-\alpha}, \quad r = |\mathbf{x}-\mathbf{z}|
 *					\f]
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-RQiso
 * @author		Soohwan Kim
 * @date			11/09/2014
 */
template<typename Scalar>
class CovRQiso
{
/**@brief Number of hyperparameters */
public: static const int N = 3;	// ell, alpha, sigma_f

/**@brief Define Matrix, MatrixPtr, MatrixConstPtr */
protected:	TYPE_DEFINE_MATRIX(Scalar);

/**@detailed
  * - Hyp(0) = \f$\log(l)\f$
  * - Hyp(1) = \f$\log(\alpha)\f$
  * - Hyp(2) = \f$\log(\sigma_f)\f$
  */
public:		TYPE_DEFINE_HYP(Scalar, N);

// public static member functions
public:

	/**
	 * @brief	Self covariance matrix between the training data, K(X, X) or its partial derivative
	 * @note		It calls the protected general member function, 
	 *				CovRQiso::K(const Hyp, const MatrixConstPtr, const int)
	 *				which only depends on pair-wise squared distances.\n\n
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
		return K(logHyp, trainingData.pSqDistXX(), pdHypIndex);
	}

	/**
	 * @brief	Cross covariance matrix between the training and test data, Ks(X, Z)
	 * @note		It calls the protected general member function, 
	 *				CovRQiso::K(const Hyp, const MatrixConstPtr, const int)
	 *				which only depends on pair-wise squared distances.
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
		return K(logHyp, trainingData.pSqDistXXs(testData));
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
		const Scalar sigma_f2 = exp(static_cast<Scalar>(2.0) * logHyp(2)); // sigma_f^2

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
			// K(r^2)
			pKss = K(logHyp, PairwiseOp<Scalar>::sqDist(testData.pXs()));
		}

		return pKss;
	}

protected:
	/**
	 * @brief	Covariance matrix given pair-wise squared distances, K(R.^2)
	 * @note		This is the core function of this class which is called from
	 *				other public static member functions, CovRQiso::K, CovRQiso::Ks and CovRQiso::Kss.
	 * @param	[in] logHyp 			The log hyperparameters
	 *											- logHyp(0) = \f$\log(l)\f$
	 *											- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] pSqDist 			The shared pointer to the pairwise squared distance matrix
	 * @param	[in] pdHypIndex		(Optional) Hyperparameter index
	 * 										- pdHypIndex = -1: return \f$\mathbf{K}(\mathbf{X}, \mathbf{X})\f$ (default)
	 *											- pdHypIndex =  0: return \f$\frac{\partial \mathbf{K}}{\partial \log(l)}\f$
	 *											- pdHypIndex =  1: return \f$\frac{\partial \mathbf{K}}{\partial \log(\sigma_f)}\f$
	 * @return	A matrix pointer of the same size of the pairwise squared distance matrix
	 */
	static MatrixPtr K(const Hyp						&logHyp, 
							 const MatrixConstPtr		pSqDist, 
							 const int						pdHypIndex = -1)
	{
		// Output
		// K: of the same size as the pairwise squared distance matrix
		// 1. K(X, X):    NxN
		// 2. Ks(X, X):   NxM
		// 3. Kss(Z, Z):	MxM
		MatrixPtr pK(new Matrix(pSqDist->rows(), pSqDist->cols()));

		// some constant values
		const Scalar inv_ell2					= exp(static_cast<Scalar>(-2.f) * logHyp(0));	// 1/ell^2
		const Scalar alpha						= exp(logHyp(1));											// alpha
		const Scalar sigma_f2					= exp(static_cast<Scalar>( 2.f) * logHyp(2));	// sigma_f^2
		const Scalar inv_double_alpha_ell2	= static_cast<Scalar>(0.5f) * inv_ell2 / alpha;	// 1/(2*alpha*ell^2)

		// hyperparameter index for the partial derivatives
		switch(pdHypIndex)
		{
		// pd[k]/pd[log(ell)]: partial derivative of covariance function w.r.t log(ell)
		case 0:
			{
				assert(false); // Not implemented yet!
				break;
			}

		// pd[k]/pd[log(sigma_f)]: partial derivative of covariance function w.r.t log(sigma_f)
		case 1:
			{
				assert(false); // Not implemented yet!
				break;
			}

		// k: covariance function
		default:
			{
				// k(x, z) = sigma_f^2 [1 + r^2 / (2*alpha*ell^2)]^(-alpha), r = |x-z|
				pK->noalias() = (sigma_f2 * (1 + inv_double_alpha_ell2 * pSqDist->array()).pow(-alpha)).matrix();
				break;
			}
		}

		return pK;
	}

};

//template<typename Scalar> const int CovRQiso::N = 2;
}

#endif