#ifndef _COVARIANCE_FUNCTION_SPARSE_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_
#define _COVARIANCE_FUNCTION_SPARSE_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_

#include "CovSparseiso.hpp"

namespace GP{

/**
 * @class		CovSparseisoDerObsBase
 * @brief		A base class for CovSparseisoDerObs which will be passed to CovDerObs
 *					as a template parameter.
 *					Thus, CovSparseisoDerObs is a combination of CovDerObs and CovSparseisoDerObsBase.
 * @note			It inherits from CovSparseiso to use CovSparseiso::K.
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-Sparseiso
 * @author	Soohwan Kim
 * @date		25/08/2014
 */
template<typename Scalar>
class CovSparseisoDerObsBase : public CovSparseiso<Scalar>
{
protected:
	/// define itself as a parent class to CovSparseisoDerObs
	typedef CovSparseiso<Scalar> CovParent;

// for CovDerObs or CovNormalPoints
protected:
	/**
	 * @brief	Covariance matrix between the functional and derivative training data
	 *				or its partial derivative
	 * @note		It calls the protected general member function, 
	 *				CovSparseisoDerObsBase::K_FD(const Hyp &logHyp, const MatrixConstPtr pAbsDist, const MatrixConstPtr pDelta, const int pdHypIndex = -1)
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
	 *				CovSparseisoDerObsBase::K_DD(const Hyp &logHyp, const MatrixConstPtr pAbsDist, const MatrixConstPtr pDelta_i, const MatrixConstPtr pDelta_j, const bool fSameCoord, const int pdHypIndex = -1)
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
	 * @note		It calls the protected static member function, CovSparseisoDerObsBase::K_FD
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
		const Scalar inv_ell				= exp(static_cast<Scalar>(-1.f) * logHyp(0));	// 1/ell
		const Scalar inv_ell2			= inv_ell * inv_ell;										// 1/ell^2
		const Scalar neg_inv_ell2		= - inv_ell2;												// -1/ell^2
		const Scalar sigma_f2			= exp(static_cast<Scalar>( 2.f) * logHyp(1));	// sigma_f^2
		const Scalar twice_sigma_f2	= static_cast<Scalar>(2.f) * sigma_f2;				// 2*sigma_f^2
		const Scalar twice_sigma_f2_inv_three	= twice_sigma_f2 / static_cast<Scalar>(3.f); // 2*sigma_f^2/3

		const Scalar pi				= static_cast<Scalar>(M_PI);				// pi
		const Scalar two_pi			= static_cast<Scalar>(2.f) * pi;			// 2*pi
		const Scalar neg_two_pi_sigma_f2_inv_three	= - pi * twice_sigma_f2_inv_three; // -2*pi*sigma_f^2/3

		// scaled distance
		Matrix S(pAbsDist->rows(), pAbsDist->cols());	// s = r/ell
		S.noalias() = inv_ell * (*pAbsDist);

		Matrix two_pi_S(S);	// 2*pi*r/ell
		two_pi_S *= two_pi;

		// mask
		Mask Mask_S_greater_than_one	= S.array() >= static_cast<Scalar>(1.f);	// if the distance, r is greater or equal to ell, k(r) = 0
		Mask Mask_S_less_than_eps		= S.array() <  Epsilon<Scalar>::value;		// avoiding division by 0

		//	k(x, z) = sigma_f^2 * [(2+cos(2*pi*r/ell))/3 * (1-r/ell) + (1/(2*pi))*sin(2*pi*r/ell)]
		//         = sigma_f^2 * [(2+cos(2*pi*s))/3 * (1-s) + (1/(2*pi))*sin(2*pi*s)], s = r/ell
		//
		// s^2    = (1/ell^2) * sum_{i=1}^d (xi - zi)^2
		// ds/dzj = -(1/ell^2)*(xj - zj)/s
		//
		// dk/ds = sigma_f^2 * [-2*pi*sin(2*pi*s)/3 * (1-s) - (2+cos(2*pi*s))/3 + cos(2*pi*s)]
		//       = (2*sigma_f^2/3) * [cos(2*pi*s) - pi*sin(2*pi*s)*(1-s) - 1]
		//
		// dk(s)/dzj = dk/ds * ds/dzj
		//           = (2*sigma_f^2/(3*ell^2)) * [pi*sin(2*pi*s)*(1-s) - cos(2*pi*s) + 1]*(xj - zj)/s
		Matrix dK_dS(pAbsDist->rows(), pAbsDist->cols());
		dK_dS.noalias() = (twice_sigma_f2_inv_three * (two_pi_S.array().cos()
					                                      - pi * two_pi_S.array().sin() * (static_cast<Scalar>(1.f) - S.array()) 
																	  - static_cast<Scalar>(1.f))).matrix();
		pK->noalias() = (neg_inv_ell2 * dK_dS.array() * pDelta->array() / S.array()).matrix();

		// avoiding division by 0
		for(int row = 0; row < Mask_S_less_than_eps.rows(); row++)
			for(int col = 0; col < Mask_S_less_than_eps.cols(); col++)
				if(Mask_S_less_than_eps(row, col)) (*pK)(row, col) = static_cast<Scalar>(0.f);

		// mode
		switch(pdHypIndex)
		{
		// derivatives of covariance matrix w.r.t log ell
		case 0:
			{
				// dk/dzj = (2*sigma_f^2/(3*ell^2)) * [pi*sin(2*pi*s)*(1-s) - cos(2*pi*s) + 1]*(xj - zj)/s, s = r/ell
				// ds/dlog(ell) = -r/ell = -s
				//
				// d^2k/dzj dlog(ell) = (-2) * dk/dzj
				//                      + (2*sigma_f^2/(3*ell^2)) * [-2*pi^2*cos(2*pi*s)*(1-s) + pi*sin(2*pi*s) - 2*pi*sin(2*pi*s)]*(xj - zj)
				//                      + dk/dzj
				//                    = (-2*pi*sigma_f^2/(3*ell^2)) * [2*pi*cos(2*pi*s)*(1-s) + sin(2*pi*s)]*(xj - zj)
				//                      - dk/dzj
				Matrix d2K_dS2(pAbsDist->rows(), pAbsDist->cols());
				d2K_dS2.noalias() = (neg_two_pi_sigma_f2_inv_three * (two_pi_S.array().sin()
																						+ two_pi * two_pi_S.array().cos() * (static_cast<Scalar>(1.f) - S.array()))).matrix();
				pK->noalias() = (inv_ell2 * d2K_dS2.array() * pDelta->array() - pK->array()).matrix();
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

		// sparse
		for(int row = 0; row < Mask_S_greater_than_one.rows(); row++)
			for(int col = 0; col < Mask_S_greater_than_one.cols(); col++)
				if(Mask_S_greater_than_one(row, col)) (*pK)(row, col) = static_cast<Scalar>(0.f);

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
		MatrixPtr pK = K(logHyp, pAbsDist, pdHypIndex);

		// constants
		const Scalar inv_ell				= exp(static_cast<Scalar>(-1.f) * logHyp(0));	// 1/ell
		const Scalar inv_ell2			= inv_ell * inv_ell;										// 1/ell^2
		const Scalar neg_inv_ell2		= - inv_ell2;												// -1/ell^2
		const Scalar inv_ell4			= inv_ell2 * inv_ell2;									// 1/ell^4
		const Scalar neg_inv_ell4		= - inv_ell4;												// -1/ell^4

		const Scalar sigma_f2			= exp(static_cast<Scalar>( 2.f) * logHyp(1));			// sigma_f^2
		const Scalar twice_sigma_f2	= static_cast<Scalar>(2.f) * sigma_f2;						// 2*sigma_f^2
		const Scalar twice_sigma_f2_inv_three	= twice_sigma_f2 / static_cast<Scalar>(3.f); // 2*sigma_f^2/3
	
		const Scalar pi				= static_cast<Scalar>(M_PI);				// pi
		const Scalar two_pi			= static_cast<Scalar>(2.f)*pi;			// 2*pi
		const Scalar neg_two_pi_sigma_f2_inv_three	= - pi * twice_sigma_f2_inv_three;									// -2*pi*sigma_f^2/3
		const Scalar eight_pi3_sigma_f2_inv_three		= static_cast<Scalar>(8.f/3.f) * pi * pi * pi * sigma_f2;	// 8*pi^3*sigma_f^2/3

		// delta
		const Scalar delta_inv_ell2 = fSameCoord ? inv_ell2 : static_cast<Scalar>(0.f);		// delta(i, j)/ell^2
		const Scalar neg_delta_inv_ell2 = - delta_inv_ell2;											// -delta(i, j)/ell^2
		const Scalar four_pi2_sigma_f2_delta_inv_three_ell2 = static_cast<Scalar>(4.f/3.f) * pi * pi * sigma_f2 * delta_inv_ell2;			// 4*pi^2*sigma_f^2*delta(i, j)/(3*ell^2)
		const Scalar neg_eight_pi2_sigma_f2_delta_inv_three_ell2 = static_cast<Scalar>(-8.f/3.f) * pi * pi * sigma_f2 * delta_inv_ell2;	// -8*pi^2*sigma_f^2*delta(i, j)/(3*ell^2)

		// scaled distance
		Matrix S(pAbsDist->rows(), pAbsDist->cols());	// s = r/ell
		S.noalias() = inv_ell * (*pAbsDist);

		Matrix two_pi_S(S);	// 2*pi*r/ell
		two_pi_S *= two_pi;

		// mask
		Mask Mask_S_greater_than_one	= S.array() >= static_cast<Scalar>(1.f);	// if the distance, r is greater or equal to ell, k(r) = 0
		Mask Mask_S_less_than_eps		= S.array() <  Epsilon<Scalar>::value;		// avoiding division by 0

		//	k(x, z) = sigma_f^2 * [(2+cos(2*pi*r/ell))/3 * (1-r/ell) + (1/(2*pi))*sin(2*pi*r/ell)]
		// k(s)    = sigma_f^2 * [(2+cos(2*pi*s))/3 * (1-s) + (1/(2*pi))*sin(2*pi*s)], s = r/ell
		//
		// s^2    = (1/ell^2) * sum_{i=1}^d (xi - zi)^2
		// ds/dzj = -(1/ell^2)*(xj - zj)/s
		//
		// dk/ds = sigma_f^2 * [-2*pi*sin(2*pi*s)/3 * (1-s) - (2+cos(2*pi*s))/3 + cos(2*pi*s)]
		//       = (2*sigma_f^2/3) * [cos(2*pi*s) - pi*sin(2*pi*s)*(1-s) - 1]
		//
		// dk(s)/dzj = dk/ds * ds/dzj
		//           = (2*sigma_f^2/(3*ell^2)) * [pi*sin(2*pi*s)*(1-s) - cos(2*pi*s) + 1]*(xj - zj)/s
		//
		// d^2k/dzj dxi = (2*sigma_f^2/(3*ell^2)) * {[2*pi^2*cos(2*pi*s)*(1-s) - pi*sin(2*pi*s)]*(1/ell^2)*(xi - zi)*(xj - zj)/s^2
		//                                           + [pi*sin(2*pi*s)*(1-s) - cos(2*pi*s) + 1]*[delta(i, j)/s + (1/ell^2)*(xi - zi)(xj - zj)/s^3]}
		//
		//              = (2*pi*sigma_f^2/(3*ell^4)) * [2*pi*cos(2*pi*s)*(1-s) - sin(2*pi*s)]*(xi - zi)*(xj - zj)/s^2
		//              + dk/ds*[delta(i, j)/s + (1/ell^2)*(xi - zi)(xj - zj)/s^3]
		Matrix dK_dS(pAbsDist->rows(), pAbsDist->cols());
		dK_dS.noalias() = (twice_sigma_f2_inv_three * (two_pi_S.array().cos()
					                                      - pi * two_pi_S.array().sin() * (static_cast<Scalar>(1.f) - S.array()) 
																	  - static_cast<Scalar>(1.f))).matrix();
		Matrix d2K_dS2(pAbsDist->rows(), pAbsDist->cols());
		d2K_dS2.noalias() = (neg_two_pi_sigma_f2_inv_three * (two_pi_S.array().sin()
																			   + two_pi * two_pi_S.array().cos() * (static_cast<Scalar>(1.f) - S.array()))).matrix();

		pK->noalias() = (neg_inv_ell4 * d2K_dS2.array() * pDelta_i->array() * pDelta_j->array() / (S.array() * S.array())
							 + dK_dS.array() * (neg_delta_inv_ell2 / S.array() + inv_ell4 * pDelta_i->array() * pDelta_j->array()  / (S.array() * S.array() * S.array()))).matrix();

		// avoiding division by 0
		for(int row = 0; row < Mask_S_less_than_eps.rows(); row++)
			for(int col = 0; col < Mask_S_less_than_eps.cols(); col++)
				if(Mask_S_less_than_eps(row, col)) (*pK)(row, col) = four_pi2_sigma_f2_delta_inv_three_ell2;

		// mode
		switch(pdHypIndex)
		{
		// derivatives of covariance matrix w.r.t log ell
		case 0:
			{
				// d^3k/ds^3
				Matrix d3K_dS3(pAbsDist->rows(), pAbsDist->cols());
				d3K_dS3.noalias() = (eight_pi3_sigma_f2_inv_three * two_pi_S.array().sin() * (static_cast<Scalar>(1.f) - S.array())).matrix();

				// d^3k/dxi dzj dlog(ell)
				pK->noalias() = (inv_ell4 * d3K_dS3.array() * pDelta_i->array() * pDelta_j->array() / S.array()
									 + d2K_dS2.array() * (delta_inv_ell2 
															    + inv_ell4 * pDelta_i->array() * pDelta_j->array() / (S.array() * S.array()))
									 + dK_dS.array() * (delta_inv_ell2 / S.array()
															  - inv_ell4 * pDelta_i->array() * pDelta_j->array() / (S.array() * S.array() * S.array()))).matrix();
				// if s = 0 ?
				for(int row = 0; row < Mask_S_less_than_eps.rows(); row++)
					for(int col = 0; col < Mask_S_less_than_eps.cols(); col++)
						if(Mask_S_less_than_eps(row, col)) (*pK)(row, col) = neg_eight_pi2_sigma_f2_delta_inv_three_ell2;
				break;
 			}

		// derivatives of covariance matrix w.r.t log sigma_f
		case 1:
			{
				// d^2k/dzj dlog(sigma_f) = 2 * dk/dzj
				pK->noalias() = static_cast<Scalar>(2.f) * (*pK);
				break;
			}

		// covariance matrix, d^2k/dxi dzj
		default:
			{
				// d^2k/dxi dzj = 3*sigma_f^2 * exp(s) * {delta(i, j)/ell^2 + [3/(s*ell^4)]*(xi - zi)*(xj - zj)}
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