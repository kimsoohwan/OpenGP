#ifndef _COVARIANCE_FUNCTION_RATIONAL_QUADRATIC_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_
#define _COVARIANCE_FUNCTION_RATIONAL_QUADRATIC_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_

#include "CovRQiso.hpp"

namespace GP{

/**
 * @class		CovRQisoDerObsBase
 * @brief		A base class for CovRQisoDerObs which will be passed to CovDerObs
 *					as a template parameter.
 *					Thus, CovRQisoDerObs is a combination of CovDerObs and CovRQisoDerObsBase.
 * @note			It inherits from CovRQiso to use CovRQiso::K.
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-RQiso
 * @author	Soohwan Kim
 * @date		11/09/2014
 */
template<typename Scalar>
class CovRQisoDerObsBase : public CovRQiso<Scalar>
{
protected:
	/// define itself as a parent class to CovRQisoDerObs
	typedef CovRQiso<Scalar> CovParent;

// for CovDerObs or CovNormalPoints
protected:
	/**
	 * @brief	Covariance matrix between the functional and derivative training data
	 *				or its partial derivative
	 * @note		It calls the protected general member function, 
	 *				CovRQisoDerObsBase::K_FD(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta, const int pdHypIndex = -1)
	 *				which only depends on pair-wise squared distances and differences.
	 * @param	[in] logHyp 							The log hyperparameters
	 *															- logHyp(0) = \f$\log(l)\f$
	 *															- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] derivativeTrainingData 		The functional and derivative training data
	 * @param	[in] coord_j 							The partial derivative coordinate of Z
	 * @param	[in] pdHypIndex						(Optional) Hyperparameter index for partial derivatives
	 * 														- pdHypIndex = -1: return \f$\frac{\partial \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \mathbf{Z}_j}\f$ (default)
	 *															- pdHypIndex =  0: return \f$\frac{\partial^2 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \log(l) \partial \mathbf{Z}_j}\f$
	 *															- pdHypIndex =  1: return \f$\frac{\partial^2 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \log(\sigma_f) \partial \mathbf{Z}_j}\f$
	 * @return	An NNxNN matrix pointer\n
	 * 			NN: The number of functional and derivative training data
	 */
	static MatrixPtr K_FD(const Hyp									&logHyp, 
								 DerivativeTrainingData<Scalar>		&derivativeTrainingData, 
								 const int									coord_j, 
								 const int									pdHypIndex)
	{
		return K_FD(logHyp, 
						derivativeTrainingData.pSqDistXXd(), 
						derivativeTrainingData.pDeltaXXd(coord_j), 
						pdHypIndex);
	}

	/**
	 * @brief	Covariance matrix between the derivative training data
	 *				or its partial derivative
	 * @note		It calls the protected general member function, 
	 *				CovRQisoDerObsBase::K_DD(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta_i, const MatrixConstPtr pDelta_j, const bool fSameCoord, const int pdHypIndex = -1)
	 *				which only depends on pair-wise squared distances and differences.
	 * @param	[in] logHyp 							The log hyperparameters
	 *															- logHyp(0) = \f$\log(l)\f$
	 *															- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] derivativeTrainingData 		The functional and derivative training data
	 * @param	[in] coord_i 							The partial derivative coordinate of X
	 * @param	[in] coord_j 							The partial derivative coordinate of Z
	 * @param	[in] pdHypIndex						(Optional) Hyperparameter index for partial derivatives
	 * 														- pdHypIndex = -1: return \f$\frac{\partial^2 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \mathbf{X}_i \partial \mathbf{Z}_j}\f$ (default)
	 *															- pdHypIndex =  0: return \f$\frac{\partial^3 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \log(l) \partial \mathbf{X}_i \partial \mathbf{Z}_j}\f$
	 *															- pdHypIndex =  1: return \f$\frac{\partial^3 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \log(\sigma_f) \partial \mathbf{X}_i \partial \mathbf{Z}_j}\f$
	 * @return	An NNxNN matrix pointer\n
	 * 			NN: The number of functional and derivative training data
	 */
	static MatrixPtr K_DD(const Hyp									&logHyp, 
								 DerivativeTrainingData<Scalar>		&derivativeTrainingData, 
								 const int									coord_i, 
								 const int									coord_j, 
								 const int									pdHypIndex)
	{
		return K_DD(logHyp, 
						derivativeTrainingData.pSqDistXdXd(), 
						derivativeTrainingData.pDeltaXdXd(coord_i),
						derivativeTrainingData.pDeltaXdXd(coord_j),
						coord_i == coord_j,
						pdHypIndex);
	}

	/**
	 * @brief	Cross covariance matrix between the derivative and functional training data
	 * @note		It calls the protected static member function, CovRQisoDerObsBase::K_FD
	 *				to utilize the symmetry property.
	 * @param	[in] logHyp 							The log hyperparameters
	 *															- logHyp(0) = \f$\log(l)\f$
	 *															- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] derivativeTrainingData 		The functional and derivative training data
	 * @param	[in] testData 							The test data
	 * @param	[in] coord_i 							The partial derivative coordinate of X
	 * @return	An NNxNN matrix pointer, \f$\frac{\partial \mathbf{K}_*(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{X}_i} = \mathbf{K}(\mathbf{X}, \mathbf{Z})\f$\n
	 * 			NN: The number of functional and derivative training data
	 */
	static MatrixPtr Ks_DF(const Hyp											&logHyp, 
								  const DerivativeTrainingData<Scalar>		&derivativeTrainingData, 
								  const TestData<Scalar>						&testData, 
								  const int											coord_i)
	{
		// squared distance: FD
		MatrixPtr pSqDistXsXd = derivativeTrainingData.pSqDistXdXs(testData);
		pSqDistXsXd->transposeInPlace();

		// delta: FD
		MatrixPtr pDeltaXsXd = derivativeTrainingData.pDeltaXdXs(testData, coord_i);
		pDeltaXsXd->transposeInPlace();
		(*pDeltaXsXd) *= static_cast<Scalar>(-1.f);

		// K_DF
		MatrixPtr K = K_FD(logHyp, pSqDistXsXd, pDeltaXsXd);
		K->transposeInPlace();

		return K;
	}

protected:
	/**
	 * @brief	Covariance matrix between the functional and derivative training data
	 *				or its partial derivative
	 *				given pair-wise squared distances and differences
	 * @param	[in] logHyp 				The log hyperparameters
	 *												- logHyp(0) = \f$\log(l)\f$
	 *												- logHyp(1) = \f$\log(\alpha)\f$
	 *												- logHyp(2) = \f$\log(\sigma_f)\f$
	 * @param	[in] pSqDist 				The pair-wise squared distances between the functional and derivative training data
	 * @param	[in] pDelta 				The pair-wise differences between the functional and derivative training data
	 * @param	[in] pdHypIndex			(Optional) Hyperparameter index for partial derivatives
	 * 											- pdHypIndex = -1: return \f$\frac{\partial \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \mathbf{Z}_j}\f$ (default)
	 *												- pdHypIndex =  0: return \f$\frac{\partial^2 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \log(l) \partial \mathbf{Z}_j}\f$
	 *												- pdHypIndex =  1: return \f$\frac{\partial^2 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \log(\sigma_f) \partial \mathbf{Z}_j}\f$
	 * @return	An NNxNN matrix pointer\n
	 * 			NN: The number of functional and derivative training data
	 */
	static MatrixPtr K_FD(const Hyp						&logHyp, 
								 const MatrixConstPtr		pSqDist, 
								 const MatrixConstPtr		pDelta, 
								 const int						pdHypIndex = -1)
	{
		// K: same size with the squared distances
		MatrixPtr pK(new Matrix(pSqDist->rows(), pSqDist->cols()));

		// constants
		const Scalar inv_ell2					= exp(static_cast<Scalar>(-2.f) * logHyp(0));	// (1/ell^2)
		const Scalar alpha						= exp(logHyp(1));											// alpha
		const Scalar sigma_f2					= exp(static_cast<Scalar>( 2.f) * logHyp(2));	// sigma_f^2
		const Scalar inv_double_alpha_ell2	= static_cast<Scalar>(0.5f) * inv_ell2 / alpha;	// 1/(2*alpha*ell^2)
		const Scalar sigma_f2_inv_ell2		= sigma_f2 * inv_ell2;									// sigma_f^2/ell^2

		// pre-calculation
		// k(x, z) = sigma_f^2 * [1 + r^2/(2*alpha*ell^2)]^(-alpha),	r = |x-z|
		// k(s)    = sigma_f^2 * s^(-alpha),									s = 1 + r^2/(2*alpha*ell^2)
		//
		// s = 1 + r^2/(2*alpha*ell^2) = 1 + (1/(2*alpha*ell^2)) * sum_{i=1}^d (xi - zi)^2
		// ds/dzj  = - (xj - zj) / (alpha*ell^2)
		//
		// dk/ds		  = sigma_f^2 * (-alpha) * s^(-alpha-1)
		// dk(s)/dzj  = dk/ds * ds/dzj
		//            = (sigma_f^2/ell^2) * s^(-alpha-1) * (xj - zj)
		pK->noalias() = (sigma_f2_inv_ell2 * (1 + inv_double_alpha_ell2 * pSqDist->array()).pow(-alpha-static_cast<Scalar>(1.f)) * pDelta->array()).matrix();

		// mode
		switch(pdHypIndex)
		{
		// derivatives of covariance matrix w.r.t log ell
		case 0:
			{
				assert(false); // Not implemented yet!
				break;
			}

		// derivatives of covariance matrix w.r.t log sigma_f
		case 1:
			{
				assert(false); // Not implemented yet!
				break;
			}
		// covariance matrix, dk/dzj
		default:
			{
				break;
			}
		}

		return pK;
	}

	/**
	 * @brief	Covariance matrix between the derivative training data
	 *				or its partial derivative
	 *				given pair-wise squared distances and differences
	 * @param	[in] logHyp 				The log hyperparameters
	 *												- logHyp(0) = \f$\log(l)\f$
	 *												- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] pSqDist 				The pair-wise squared distances between the functional and derivative training data
	 * @param	[in] pDelta_i 				The pair-wise differences of the i-th components between the functional and derivative training data
	 * @param	[in] pDelta_j 				The pair-wise differences of the j-th components between the functional and derivative training data
	 * @param	[in] fSameCoord 			i == j
	 * @param	[in] pdHypIndex			(Optional) Hyperparameter index for partial derivatives
	 * 											- pdHypIndex = -1: return \f$\frac{\partial^2 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \mathbf{X}_i \partial \mathbf{Z}_j}\f$ (default)
	 *												- pdHypIndex =  0: return \f$\frac{\partial^3 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \log(l) \partial \mathbf{X}_i \partial \mathbf{Z}_j}\f$
	 *												- pdHypIndex =  1: return \f$\frac{\partial^3 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \log(\sigma_f) \partial \mathbf{X}_i \partial \mathbf{Z}_j}\f$
	 * @return	An NNxNN matrix pointer\n
	 * 			NN: The number of functional and derivative training data
	 */
	static MatrixPtr K_DD(const Hyp &logHyp, 
								 const MatrixConstPtr pSqDist, 
								 const MatrixConstPtr pDelta_i, 
								 const MatrixConstPtr pDelta_j,
								 const bool fSameCoord,
								 const int pdHypIndex = -1)
	{
		// K: same size with the squared distances
		MatrixPtr pK(new Matrix(pSqDist->rows(), pSqDist->cols()));

		// constants
		const Scalar inv_ell2					= exp(static_cast<Scalar>(-2.f) * logHyp(0));	// (1/ell^2)
		const Scalar alpha						= exp(logHyp(1));											// alpha
		const Scalar sigma_f2					= exp(static_cast<Scalar>( 2.f) * logHyp(2));	// sigma_f^2
		const Scalar inv_double_alpha_ell2	= static_cast<Scalar>(0.5f) * inv_ell2 / alpha;	// 1/(2*alpha*ell^2)
		const Scalar sigma_f2_inv_ell2		= sigma_f2 * inv_ell2;									// sigma_f^2/ell^2
		const Scalar alpha_factor				= inv_ell2*(-alpha-static_cast<Scalar>(1.f))/(alpha); // (-alpha-1)/(alpha*ell^2)

		// delta
		const Scalar delta = fSameCoord ? static_cast<Scalar>(1.f) : static_cast<Scalar>(0.f);	// delta(i, j)

		// pre-calculation
		// k(x, z) = sigma_f^2 * [1 + r^2/(2*alpha*ell^2)]^(-alpha),	r = |x-z|
		// k(s)    = sigma_f^2 * s^(-alpha),									s = 1 + r^2/(2*alpha*ell^2)
		//
		// s = 1 + r^2/(2*alpha*ell^2) = 1 + (1/(2*alpha*ell^2)) * sum_{i=1}^d (xi - zi)^2
		// ds/dzj  = - (xj - zj) / (alpha*ell^2)
		//
		// dk/ds		  = sigma_f^2 * (-alpha) * s^(-alpha-1)
		// dk(s)/dzj  = dk/ds * ds/dzj
		//            = (sigma_f^2/ell^2) * s^(-alpha-1) * (xj - zj)
		//
		// d^2k/dzj dxi = (sigma_f^2/ell^2) * [(-alpha-1) * s^(-alpha-2) * (xi - zi) / (alpha*ell^2) * (xj - zj)
		//                                                + s^(-alpha-1) * delta(i, j)]
		//              = (sigma_f^2/ell^2) * [(-alpha-1)/(alpha*ell^2) * s^(-alpha-2) * (xi - zi) * (xj - zj)
		//                                                              + s^(-alpha-1) * delta(i, j)]
		Matrix S(pSqDist->rows(), pSqDist->cols());
		S.noalias() = (1 + inv_double_alpha_ell2 * pSqDist->array()).matrix();

		pK->noalias() = (sigma_f2_inv_ell2 * (alpha_factor * S.array().pow(-alpha-static_cast<Scalar>(2.f)) * pDelta_i->array() * pDelta_j->array()
			                                                + S.array().pow(-alpha-static_cast<Scalar>(1.f)) * delta)).matrix();
	
		// mode
		switch(pdHypIndex)
		{
		// derivatives of covariance matrix w.r.t log ell
		case 0:
			{
				assert(false); // Not implemented yet!
				break;
			}

		// derivatives of covariance matrix w.r.t log sigma_f
		case 1:
			{
				assert(false); // Not implemented yet!
				break;
			}
		// covariance matrix, dk/dzj
		default:
			{
				break;
			}
		}

		return pK;
	}
};

}

#endif