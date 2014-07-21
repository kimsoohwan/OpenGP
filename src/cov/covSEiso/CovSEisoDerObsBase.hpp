#ifndef _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_
#define _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_

#include "CovSEiso.hpp"

namespace GP{

/**
 * @class		CovSEisoDerObsBase
 * @brief		A base class for CovSEisoDerObs which will be passed to CovDerObs
 *					as a template parameter.
 *					Thus, CovSEisoDerObs is a combination of CovDerObs and CovSEisoDerObsBase.
 * @note			It inherits from CovSEiso to use CovSEiso::K.
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-SEiso
 * @author	Soohwan Kim
 * @date		30/06/2014
 */
template<typename Scalar>
class CovSEisoDerObsBase : public CovSEiso<Scalar>
{
protected:
	/// define itself as a parent class to CovSEisoDerObs
	typedef CovSEiso<Scalar> CovParent;

// for CovDerObs or CovNormalPoints
protected:
	/**
	 * @brief	Covariance matrix between the functional and derivative training data
	 *				or its partial derivative
	 * @note		It calls the protected general member function, 
	 *				CovSEisoDerObsBase::K_FD(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta, const int pdHypIndex = -1)
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
	 *				CovSEisoDerObsBase::K_DD(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta_i, const MatrixConstPtr pDelta_j, const bool fSameCoord, const int pdHypIndex = -1)
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
	 * @note		It calls the protected static member function, CovSEisoDerObsBase::K_FD
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
	static MatrixPtr Ks_DF(const Hyp									&logHyp, 
								  DerivativeTrainingData<Scalar>		&derivativeTrainingData, 
								  const TestData<Scalar>				&testData, 
								  const int									coord_i)
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
	 *												- logHyp(1) = \f$\log(\sigma_f)\f$
	 * @param	[in] MatrixConstPtr 		The pair-wise squared distances between the functional and derivative training data
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
		MatrixPtr pK = K(logHyp, pSqDist);

		// hyperparameters
		Scalar inv_ell2 = exp(static_cast<Scalar>(-2.f) * logHyp(0));	// (1/ell^2)

		// pre-calculation
		pK->noalias() = (inv_ell2 * (*pK).array() * (*pDelta).array()).matrix();

		// mode
		switch(pdHypIndex)
		{
		// covariance matrix
		case -1:
			{
				// k(x, z) = K(x, z) * ((x_j - z_j) / ell^2)
				//pK->noalias() = (inv_ell2 * (*pK).array() * (*pDelta).array()).matrix();
				//pK->noalias() = *(pK->cwiseProduct(inv_ell2 * (*pDelta)));
				//pK->noalias() = *(pDelta->cwiseProduct(inv_ell2 * (*pK)));
				//std::cout << "K_FD = " << std::endl << *pK_FD << std::endl << std::endl;
				break;
			}

		// derivatives of covariance matrix w.r.t log ell
		case 0:
			{
				// dk/dlog(ell) = k * (-2s-2), s = (-1/2)*r^2/ell^2
				pK->noalias() = ((*pK).array() * (inv_ell2*(pSqDist->array()) - static_cast<Scalar>(2.f))).matrix();
				//pK->noalias() = *(pK->cwiseProduct(neg_half_inv_ell2*(*pSqDist) + static_cast<Scalar>(-2.f)));
				//std::cout << "K_FD_log(ell) = " << std::endl << *pK_FD << std::endl << std::endl;
				break;
			}

		// derivatives of covariance matrix w.r.t log sigma_f
		case 1:
			{
				// k_log(sigma_f) = 2*k
				//pK->noalias() = (double_inv_ell2 * (*pK).array() * (*pDelta).array()).matrix();
				pK->noalias() = static_cast<Scalar>(2.f) * (*pK);
				//std::cout << "K_FD_log(sigma_f) = " << std::endl << *pK_FD << std::endl << std::endl;
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
	 * @param	[in] MatrixConstPtr 		The pair-wise squared distances between the functional and derivative training data
	 * @param	[in] pDelta 				The pair-wise differences between the functional and derivative training data
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
		MatrixPtr pK = K(logHyp, pSqDist, pdHypIndex);

		// hyperparameters
		Scalar inv_ell2				= exp(static_cast<Scalar>(-2.f) * logHyp(0));	// 1/ell^2
		Scalar inv_ell4				= exp(static_cast<Scalar>(-4.f) * logHyp(0));	// 1/ell^4

		// delta
		Scalar delta = fSameCoord ? inv_ell2 : static_cast<Scalar>(0.f);

		// simplified version

		// for all cases
		pK->noalias() = ((*pK).array() * (delta - inv_ell4*(pDelta_i->array())*(pDelta_j->array()))).matrix();

		// particularly, derivatives of covariance matrix w.r.t log ell
		if(pdHypIndex == 0)
		{
			MatrixPtr pK0 = K(logHyp, pSqDist);
			pK->noalias() += ((*pK0).array() * (static_cast<Scalar>(-2.f)*delta 
															 + static_cast<Scalar>(4.f)*inv_ell4*(pDelta_i->array())*(pDelta_j->array()))).matrix();
		}

		return pK;
	}
};

}

#endif