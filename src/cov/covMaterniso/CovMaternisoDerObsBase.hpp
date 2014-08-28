#ifndef _COVARIANCE_FUNCTION_MATERN_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_
#define _COVARIANCE_FUNCTION_MATERN_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_

#include "CovMaterniso.hpp"

namespace GP{

/**
 * @class		CovMaternisoDerObsBase
 * @brief		A base class for CovMaternisoDerObs which will be passed to CovDerObs
 *					as a template parameter.
 *					Thus, CovMaternisoDerObs is a combination of CovDerObs and CovMaternisoDerObsBase.
 * @note			It inherits from CovMaterniso to use CovMaterniso::K.
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-Materniso
 * @author	Soohwan Kim
 * @date		25/08/2014
 */
template<typename Scalar>
class CovMaternisoDerObsBase : public CovMaterniso<Scalar>
{
protected:
	/// define itself as a parent class to CovMaternisoDerObs
	typedef CovMaterniso<Scalar> CovParent;

// for CovDerObs or CovNormalPoints
protected:
	/**
	 * @brief	Covariance matrix between the functional and derivative training data
	 *				or its partial derivative
	 * @note		It calls the protected general member function, 
	 *				CovMaternisoDerObsBase::K_FD(const Hyp &logHyp, const MatrixConstPtr pAbsDist, const MatrixConstPtr pDelta, const int pdHypIndex = -1)
	 *				which only depends on pair-wise absolute distances and differences.
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
						derivativeTrainingData.pAbsDistXXd(), 
						derivativeTrainingData.pDeltaXXd(coord_j), 
						pdHypIndex);
	}

	/**
	 * @brief	Covariance matrix between the derivative training data
	 *				or its partial derivative
	 * @note		It calls the protected general member function, 
	 *				CovMaternisoDerObsBase::K_DD(const Hyp &logHyp, const MatrixConstPtr pAbsDist, const MatrixConstPtr pDelta_i, const MatrixConstPtr pDelta_j, const bool fSameCoord, const int pdHypIndex = -1)
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
						derivativeTrainingData.pAbsDistXdXd(), 
						derivativeTrainingData.pDeltaXdXd(coord_i),
						derivativeTrainingData.pDeltaXdXd(coord_j),
						coord_i == coord_j,
						pdHypIndex);
	}

	/**
	 * @brief	Cross covariance matrix between the derivative and functional training data
	 * @note		It calls the protected static member function, CovMaternisoDerObsBase::K_FD
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
		MatrixPtr pAbsDistXsXd = derivativeTrainingData.pAbsDistXdXs(testData);
		pAbsDistXsXd->transposeInPlace();

		// delta: FD
		MatrixPtr pDeltaXsXd = derivativeTrainingData.pDeltaXdXs(testData, coord_i);
		pDeltaXsXd->transposeInPlace();
		(*pDeltaXsXd) *= static_cast<Scalar>(-1.f);

		// K_DF
		MatrixPtr K = K_FD(logHyp, pAbsDistXsXd, pDeltaXsXd);
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
	 *												- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] pAbsDist 				The pair-wise absolute distances between the functional and derivative training data
	 * @param	[in] pDelta 				The pair-wise differences between the functional and derivative training data
	 * @param	[in] pdHypIndex			(Optional) Hyperparameter index for partial derivatives
	 * 											- pdHypIndex = -1: return \f$\frac{\partial \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \mathbf{Z}_j}\f$ (default)
	 *												- pdHypIndex =  0: return \f$\frac{\partial^2 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \log(l) \partial \mathbf{Z}_j}\f$
	 *												- pdHypIndex =  1: return \f$\frac{\partial^2 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \log(\sigma_f) \partial \mathbf{Z}_j}\f$
	 * @return	An NNxNN matrix pointer\n
	 * 			NN: The number of functional and derivative training data
	 */
	static MatrixPtr K_FD(const Hyp						&logHyp, 
								 const MatrixConstPtr		pAbsDist, 
								 const MatrixConstPtr		pDelta, 
								 const int						pdHypIndex = -1)
	{
		// K: same size with the squared distances
		MatrixPtr pK(new Matrix(pAbsDist->rows(), pAbsDist->cols()));

		// constants
		const Scalar inv_ell							= exp(static_cast<Scalar>(-1.f) * logHyp(0));	// (1/ell)
		const Scalar inv_ell2						= inv_ell * inv_ell;										// (1/ell^2)
		const Scalar neg_sqrt3_inv_ell			= static_cast<Scalar>(-1.732050807568877f) * inv_ell;		// -sqrt(3)/ell, sqrt(3) = 1.732050807568877f
		const Scalar three_sigma_f2_inv_ell2	= static_cast<Scalar>(3.f) * exp(static_cast<Scalar>(2.f) * logHyp(1)) * inv_ell2; // 3*sigma_f^2/ell^2

		// pre-calculation: S = - sqrt(3) * r / ell
		Matrix S(pAbsDist->rows(), pAbsDist->cols());
		S.noalias() = neg_sqrt3_inv_ell * (*pAbsDist);

		// k(x, z) = sigma_f^2 * (1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell),		r = |x - z|
		// k(s)    = sigma_f^2 * (1 - s) * exp(s),										s = sqrt(3)*r/ell
		//
		// s^2    = (3/ell^2) * sum_{i=1}^d (xi - zi)^2
		// ds/dzj = -(3/ell^2)*(xj - zj)/s
		//
		// dk/ds		 = sigma_f^2 * exp(s) * (-s)
		// dk(s)/dzj = dk/ds * ds/dzj
		//           = 3*sigma_f^2 * exp(s) * (xj - zj)/ell^2
		pK->noalias() = (three_sigma_f2_inv_ell2 * S.array().exp() * pDelta->array()).matrix(); // dk/dzj

		// mode
		switch(pdHypIndex)
		{
		// derivatives of covariance matrix w.r.t log ell
		case 0:
			{
				// dk/dzj             = 3*sigma_f^2 * exp(s) * (xj - zj)/ell^2
				// d^2k/dzj dlog(ell) = 3*sigma_f^2 * exp(s) * [(-s) * (xj - zj)/ell^2 - 2*(xj - zj)/ell^2]
				//                    = 3*sigma_f^2 * exp(s) * [(xj - zj)/ell^2] * (-2-s)
				//                    = dk/dzj * (-2-s)
				pK->noalias() = (pK->array() * (static_cast<Scalar>(-2.f) - S.array())).matrix();
				break;
			}

		// derivatives of covariance matrix w.r.t log sigma_f
		case 1:
			{
				// d^2k/dzj dlog(sigma_f) = 2 * dk/dzj
				(*pK) *= static_cast<Scalar>(2.f);
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
	 *				given pair-wise absolute distances and differences
	 * @param	[in] logHyp 				The log hyperparameters
	 *												- logHyp(0) = \f$\log(l)\f$
	 *												- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] pAbsDist 				The pair-wise absolute distances between the functional and derivative training data
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
								 const MatrixConstPtr pAbsDist, 
								 const MatrixConstPtr pDelta_i, 
								 const MatrixConstPtr pDelta_j,
								 const bool fSameCoord,
								 const int pdHypIndex = -1)
	{
		// K: same size with the squared distances
		MatrixPtr pK(new Matrix(pAbsDist->rows(), pAbsDist->cols()));

		// hyperparameters
		const Scalar inv_ell  = exp(static_cast<Scalar>(-1.f) * logHyp(0));	// 1/ell
		const Scalar inv_ell2 = inv_ell  * inv_ell;									// 1/ell^2
		const Scalar inv_ell4 = inv_ell2 * inv_ell2;									// 1/ell^4

		const Scalar three_inv_ell4		= static_cast<Scalar>( 3.f) * inv_ell4;	// 3/ell^4
		const Scalar neg_nine_inv_ell4	= static_cast<Scalar>(-9.f) * inv_ell4;	// -9/ell^4
		const Scalar neg_sqrt3_inv_ell	= static_cast<Scalar>(-1.732050807568877f) * inv_ell;		// -sqrt(3)/ell, sqrt(3) = 1.732050807568877f

		const Scalar sigma_f2			= exp(static_cast<Scalar>( 2.f) * logHyp(1));	// sigma_f^2
		const Scalar three_sigma_f2	= static_cast<Scalar>(3.f) * sigma_f2;	// 3*sigma_f^2

		// delta
		const Scalar delta_inv_ell2 = fSameCoord ? inv_ell2 : static_cast<Scalar>(0.f);			// delta(i, j)/ell^2
		const Scalar neg_double_delta_inv_ell2 = static_cast<Scalar>(-2.f) * delta_inv_ell2;	// -2*delta(i, j)/ell^2

		const Scalar neg_six_sigma_f2_delta_inv_ells = three_sigma_f2 * neg_double_delta_inv_ell2;	// -6*sigma_f^2*delta(i, j)/ell^2
		const Scalar six_sigma_f2_delta_inv_ells = - neg_six_sigma_f2_delta_inv_ells;						//  6*sigma_f^2*delta(i, j)/ell^2

		// pre-calculation: S = -sqrt(3)*r/ell
		Matrix S(pAbsDist->rows(), pAbsDist->cols());
		S.noalias() = neg_sqrt3_inv_ell * (*pAbsDist);
		Mask Mask_S_less_than_eps = S.array() > - Epsilon<Scalar>::value;	// avoiding division by 0

		// pre-calculation
		// k(x, z) = sigma_f^2 * (1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell),		r = |x - z|
		// k(s)    = sigma_f^2 * (1 - s) * exp(s),										s = sqrt(3)*r/ell
		//
		// s^2    = (3/ell^2) * sum_{i=1}^d (xi - zi)^2
		// ds/dzj = -(3/ell^2)*(xj - zj)/s
		//
		// dk/ds		 = sigma_f^2 * exp(s) * (-s)
		// dk(s)/dzj = dk/ds * ds/dzj
		//           = 3*sigma_f^2 * exp(s) * (xj - zj)/ell^2
		//
		// d^2k/dzj dxi = 3*sigma_f^2 * exp(s) * {[(3/ell^2)*(xi - zi)/s] * (xj - zj)/ell^2 + delta(i, j)/ell^2}
		//              = 3*sigma_f^2 * exp(s) * {delta(i, j)/ell^2 + [3/(s*ell^4)]*(xi - zi)*(xj - zj)}
		pK->noalias() = (three_sigma_f2 * S.array().exp() * (delta_inv_ell2 + three_inv_ell4 * pDelta_i->array() * pDelta_j->array() / S.array())).matrix();
	
		// mode
		switch(pdHypIndex)
		{
		// derivatives of covariance matrix w.r.t log ell
		case 0:
			{
				// d^2k/dxi dzj       = 3*sigma_f^2 * exp(s) * {delta(i, j)/ell^2 + [3/(s*ell^4)]*(xi - zi)*(xj - zj)}
				// d^3k/dzj dlog(ell) = d^2k/dxi dzj * (-s)
				//                    + 3*sigma_f^2 * exp(s) * {-2*delta(i, j)/ell^2 - [9/(s*ell^4)]*(xi - zi)*(xj - zj)}
				pK->noalias() = (three_sigma_f2 * S.array().exp() * (neg_double_delta_inv_ell2 + neg_nine_inv_ell4 * pDelta_i->array() * pDelta_j->array() / S.array())
									  - pK->array() * S.array()).matrix();

				// if s = 0 ?
				for(int row = 0; row < Mask_S_less_than_eps.rows(); row++)
					for(int col = 0; col < Mask_S_less_than_eps.cols(); col++)
						if(Mask_S_less_than_eps(row, col)) (*pK)(row, col) = neg_six_sigma_f2_delta_inv_ells;
				break;
 			}

		// derivatives of covariance matrix w.r.t log sigma_f
		case 1:
			{
				// d^2k/dzj dlog(sigma_f) = 2 * dk/dzj
				pK->noalias() = static_cast<Scalar>(2.f) * (*pK);

				// if s = 0 ?
				for(int row = 0; row < Mask_S_less_than_eps.rows(); row++)
					for(int col = 0; col < Mask_S_less_than_eps.cols(); col++)
						if(Mask_S_less_than_eps(row, col)) (*pK)(row, col) = six_sigma_f2_delta_inv_ells;
				break;
			}

		// covariance matrix, d^2k/dxi dzj
		default:
			{
				// d^2k/dxi dzj = 3*sigma_f^2 * exp(s) * {delta(i, j)/ell^2 + [3/(s*ell^4)]*(xi - zi)*(xj - zj)}
				// if s = 0 ?
				for(int row = 0; row < Mask_S_less_than_eps.rows(); row++)
					for(int col = 0; col < Mask_S_less_than_eps.cols(); col++)
						if(Mask_S_less_than_eps(row, col)) (*pK)(row, col) = three_sigma_f2 * delta_inv_ell2;
				break;
			}
		}

		return pK;
	}
};

}

#endif