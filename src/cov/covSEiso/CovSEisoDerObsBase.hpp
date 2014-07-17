#ifndef _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_
#define _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_

#include "CovSEiso.hpp"

namespace GP{

/**
 * @class		CovSEisoDerObsBase
 * @brief		A base class for CovSEisoDerObs
 *					which is the squared exponential covariance function with isotropic distances
 *					and deals with derivative observations.\n\n
 *					It provides protected static member functions which will be called
 *					in CovDerObs as follows
 *					<CENTER>
 *					Protected Static Member Functions | Corresponding Mathematical Equations
 *					----------------------------------|-------------------------------------
 *					CovSEisoDerObsBase::K_FD			 | \f$\mathbf{K} = \mathbf{K}(\mathbf{X}, \mathbf{X}) \in \mathbb{R}^{N \times N}\f$
 *					CovSEisoDerObsBase::K_DD			 | \f$\mathbf{K}_* = \mathbf{K}(\mathbf{X}, \mathbf{Z}) \in \mathbb{R}^{N \times M}\f$
 *					CovSEisoDerObsBase::K_DD			 | \f$\mathbf{k}_{**} \in \mathbb{R}^{M \times 1}, \mathbf{k}_{**}^i = k(\mathbf{Z}_i, \mathbf{Z}_i)\f$ or \f$\mathbf{K}_{**} = \mathbf{K}(\mathbf{Z}, \mathbf{Z}) \in \mathbb{R}^{M \times M}\f$
 *					CovSEisoDerObsBase::Ks_DF			 | \f$\mathbf{k}_{**} \in \mathbb{R}^{M \times 1}, \mathbf{k}_{**}^i = k(\mathbf{Z}_i, \mathbf{Z}_i)\f$ or \f$\mathbf{K}_{**} = \mathbf{K}(\mathbf{Z}, \mathbf{Z}) \in \mathbb{R}^{M \times M}\f$
 *					</CENTER>
 *					
 *					Thus, CovSEisoDerObs is a combination of CovDerObs and CovSEisoDerObsBase.
 * @note			It inherits from CovSEiso to use CovSEiso::K.
 * @ingroup		CovDerObs
 * @author	Soohwan Kim
 * @date		30/06/2014
 */
template<typename Scalar>
class CovSEisoDerObsBase : public CovSEiso<Scalar>
{
protected:
	typedef CovSEiso<Scalar> CovParent;

// for CovDerObs or CovNormalPoints
protected:
	static MatrixPtr K_FD(const Hyp &logHyp, 
								 DerivativeTrainingData<Scalar> &derivativeTrainingData, 
								 const int coord_j, const int pdHypIndex)
	{
		return K_FD(logHyp, 
						derivativeTrainingData.pSqDistXXd(), 
						derivativeTrainingData.pDeltaXXd(coord_j), 
						pdHypIndex);
	}

	static MatrixPtr K_DD(const Hyp &logHyp, 
								 DerivativeTrainingData<Scalar> &derivativeTrainingData, 
								 const int coord_i, const int coord_j, const int pdHypIndex)
	{
		return K_DD(logHyp, 
						derivativeTrainingData.pSqDistXdXd(), 
						derivativeTrainingData.pDeltaXdXd(coord_i),
						derivativeTrainingData.pDeltaXdXd(coord_j),
						coord_i == coord_j,
						pdHypIndex);
	}

	static MatrixPtr Ks_DF(const Hyp &logHyp, 
								  DerivativeTrainingData<Scalar> &derivativeTrainingData, 
								  const TestData<Scalar> &testData, 
								  const int coord_i)
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
	static MatrixPtr K_FD(const Hyp &logHyp, 
								 const MatrixConstPtr pSqDist, 
								 const MatrixConstPtr pDelta, 
								 const int pdHypIndex = -1)
	{
		// input
		// pSqDist (nxm): squared distances = r^2
		// pDelta (nxm): delta = x_i - x_i'
		// pLogHyp: log hyperparameters
		// pdHypIndex: partial derivatives with respect to this parameter index

		// output
		// K: nxm matrix
		// if pdHypIndex == -1:		K_FF
		// else							partial K_FF / partial theta_i
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

	static MatrixPtr K_DD(const Hyp &logHyp, 
								 const MatrixConstPtr pSqDist, 
								 const MatrixConstPtr pDelta_i, 
								 const MatrixConstPtr pDelta_j,
								 const bool fSameCoord,
								 const int pdHypIndex = -1)
	{
		// input
		// pSqDist (nxm): squared distances = r^2
		// pDelta1 (nxm): delta = x_i - x_i'
		// i: index for delta1
		// pDelta2 (nxm): delta = x_j - x_j'
		// j: index for delta2
		// pLogHyp: log hyperparameters
		// pdHypIndex: partial derivatives with respect to this parameter index

		// output
		// K: nxm matrix
		// if pdHypIndex == -1:		K_FF
		// else							partial K_FF / partial theta_i
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