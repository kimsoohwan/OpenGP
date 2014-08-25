#ifndef _COVARIANCE_FUNCTION_SPARSE_ISO_HPP_
#define _COVARIANCE_FUNCTION_SPARSE_ISO_HPP_

#define _USE_MATH_DEFINES
#include <cmath>	// sin, cos, M_PI

#include "../../util/macros.h"
#include "../../data/TrainingData.hpp"

namespace GP{

/**
 * @class		CovSparseiso
 * @brief		Sparse covariance function with isotropic distances
 *					\f[
 *					k(\mathbf{x}, \mathbf{z}) = 
 *					\begin{cases} 
 *					\sigma_f^2 \left( \frac{2+\cos(2\pi \frac{r}{l})}{3} \left(1-\frac{r}{l}\right) + \frac{1}{2\pi} \sin\left(2\pi \frac{r}{l}\right) \right) & r < l\\
 *					0 & r \ge l
 *					\end{cases}
 *					\f]
 *					All covariance classes should have public static member functions as follows.
 *					<CENTER>
 *					Public Static Member Functions | Corresponding Covariance Functions
 *					-------------------------------|-------------------------------------
 *					+CovSparseiso::K			| \f$\mathbf{K} = \mathbf{K}(\mathbf{X}, \mathbf{X}) \in \mathbb{R}^{N \times N}\f$
 *					+CovSparseiso::Ks			| \f$\mathbf{K}_* = \mathbf{K}(\mathbf{X}, \mathbf{Z}) \in \mathbb{R}^{N \times M}\f$
 *					+CovSparseiso::Kss		| \f$\mathbf{k}_{**} \in \mathbb{R}^{M \times 1}, \mathbf{k}_{**}^i = k(\mathbf{Z}_i, \mathbf{Z}_i)\f$ or \f$\mathbf{K}_{**} = \mathbf{K}(\mathbf{Z}, \mathbf{Z}) \in \mathbb{R}^{M \times M}\f$
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
 * @ingroup		-Sparseiso
 * @author		Soohwan Kim
 * @date			25/08/2014
 */
template<typename Scalar>
class CovSparseiso
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
	 *				CovSparseiso::K(const Hyp, const MatrixConstPtr, const int)
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
	 *				CovSparseiso::K(const Hyp, const MatrixConstPtr, const int)
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
	 *				other public static member functions, CovSparseiso::K, CovSparseiso::Ks and CovSparseiso::Kss.
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
		const Scalar inv_ell				= exp(static_cast<Scalar>(-1.f) * logHyp(0));	// 1/ell
		const Scalar sigma_f2			= exp(static_cast<Scalar>( 2.f) * logHyp(1));	// sigma_f^2
		const Scalar twice_sigma_f2	= static_cast<Scalar>(2.f) * sigma_f2;				// 2*sigma_f^2
		const Scalar twice_sigma_f2_inv_three	= twice_sigma_f2 / static_cast<Scalar>(3.f);				// 2*sigma_f^2/3

		const Scalar pi				= static_cast<Scalar>(M_PI);				// pi
		const Scalar two_pi			= static_cast<Scalar>(2.f) * pi;			// 2*pi
		const Scalar inv_two_pi		= static_cast<Scalar>(1.f) / two_pi;	// 1/(2*pi)
		const Scalar inv_three		= static_cast<Scalar>(1.f) / static_cast<Scalar>(3.f);	// 1/3

		// scaled distance
		Matrix S(pAbsDist->rows(), pAbsDist->cols());	// s = r/ell
		S.noalias() = inv_ell * (*pAbsDist);

		Matrix two_pi_S(S);	// 2*pi*r/ell
		two_pi_S *= two_pi;

		// sparse mask
		Mask Mask_S_greater_than_one = S.array() >= static_cast<Scalar>(1.f);	// if the distance, r is greater or equal to ell, k(r) = 0

		// hyperparameter index for the partial derivatives
		switch(pdHypIndex)
		{
		// pd[k]/pd[log(ell)]: partial derivative of covariance function w.r.t log(ell)
		case 0:
			{
				//	k(x, z) = sigma_f^2 * [(2+cos(2*pi*r/ell))/3 * (1-r/ell) + (1/(2*pi))*sin(2*pi*r/ell)]
				//         = sigma_f^2 * [(2+cos(2*pi*s))/3 * (1-s) + (1/(2*pi))*sin(2*pi*s)], s = r/ell
				//
				// dk/ds = sigma_f^2 * [-2*pi*sin(2*pi*s)/3 * (1-s) - (2+cos(2*pi*s))/3 + cos(2*pi*s)]
				//       = (2*sigma_f^2/3) * [cos(2*pi*s) - pi*sin(2*pi*s)*(1-s) - 1]
				//
				// ds/dlog(ell) = -r/ell = -s
				//
				// dk/dlog(ell) = dk/ds * ds/dlog(ell)
				//              = (2*sigma_f^2/3) * [pi*sin(2*pi*s)*(1-s) - cos(2*pi*s) + 1] * s
				pK->noalias() = (twice_sigma_f2_inv_three * (pi * two_pi_S.array().sin() * (static_cast<Scalar>(1.f) - S.array())
					                                          - two_pi_S.array().cos() + static_cast<Scalar>(1.f))
																		* S.array()).matrix();
				break;
			}

		// pd[k]/pd[log(sigma_f)]: partial derivative of covariance function w.r.t log(sigma_f)
		case 1:
			{
				//	k(x, z) = sigma_f^2 * [(2+cos(2*pi*r/ell))/3 * (1-r/ell) + (1/(2*pi))*sin(2*pi*r/ell)]
				// pd[k]/pd[log(sigma_f)] = 2 * k(x, z)
				pK->noalias() = (twice_sigma_f2 * (inv_three * (static_cast<Scalar>(2.f) + two_pi_S.array().cos())
																			* (static_cast<Scalar>(1.f) - S.array())
												          + inv_two_pi * two_pi_S.array().sin())).matrix();
				break;
			}

		// k: covariance function
		default:
			{
				//	k(x, z) = sigma_f^2 * [(2+cos(2*pi*r/ell))/3 * (1-r/ell) + (1/(2*pi))*sin(2*pi*r/ell)]
				pK->noalias() = (sigma_f2 * (inv_three * (static_cast<Scalar>(2.f) + two_pi_S.array().cos())
																  * (static_cast<Scalar>(1.f) - S.array())
												    + inv_two_pi * two_pi_S.array().sin())).matrix();
				break;
			}
		}

		// sparse
		for(int row = 0; row < Mask_S_greater_than_one.rows(); row++)
			for(int col = 0; col < Mask_S_greater_than_one.cols(); col++)
				if(Mask_S_greater_than_one(row, col)) (*pK)(row, col) = static_cast<Scalar>(0.f);

		return pK;
	}

};


}

#endif