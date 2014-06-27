#ifndef COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_
#define COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_WITH_DERIVATIVE_OBSERVATIONS_HPP_

#include "CovSEiso.hpp"

namespace GP{

template<typename Scalar>
class CovSEisoDiff : public CovSEiso<Scalar>
{
// for CovDerObs or CovNormalPoints
protected:
	static MatrixPtr K_FD(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const int coord_j, const int pdIndex)
	{
		return K_FD(logHyp, derivativeTrainingData.pSqDistXXd(), derivativeTrainingData.pDeltaXXd(coord_j), pdIndex);
	}

	static MatrixPtr K_DD(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const int coord_i, const int coord_j, const int pdIndex)
	{
		return K_FD(logHyp, derivativeTrainingData.pSqDistXdXd(), derivativeTrainingData.pDeltaXdXd(coord_i, coord_j), pdIndex);
	}

	static MatrixPtr Ks_DF(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const TestData<Scalar> &testData, const int coord_i)
	{
		// squared distance: FD
		MatrixPtr pSqDistXdXs = derivativeTrainingData.sqDistXdXs(testData);
		pSqDistXdXs->transposeInPlace();

		// delta: FD
		MatrixPtr deltaXdXs = derivativeTrainingData.deltaXdXs(testData, coord_i);
		deltaXdXs->transposeInPlace();

		// K_DF
		MatrixPtr K = K_FD(logHyp, pSqDistXdXs, deltaXdXs);
		K->transposeInPlace();

		return K;
	}

protected:
	static MatrixPtr K_FD(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta, const int pdIndex)
	{
		// input
		// pSqDist (nxm): squared distances = r^2
		// pDelta (nxm): delta = x_i - x_i'
		// pLogHyp: log hyperparameters
		// pdIndex: partial derivatives with respect to this parameter index

		// output
		// K: nxm matrix
		// if pdIndex == -1:		K_FF
		// else							partial K_FF / partial theta_i
		MatrixPtr pK_FD = K_FF(pSqDist, pLogHyp, pdIndex);

		// hyperparameters
		Scalar inv_ell2 = exp(((Scalar) -2.f) * (*pLogHyp)(0));

		// mode
		switch(pdIndex)
		{
		// covariance matrix
		case -1:
			{
				// k(X, X') = ((x_i - x_i') / ell^2) * K_FF(X, X')
				(*pK_FD) = inv_ell2 * pDelta->cwiseProduct(*pK_FD);
				//std::cout << "K_FD = " << std::endl << *pK_FD << std::endl << std::endl;
				break;
			}

		// derivatives of covariance matrix w.r.t log ell
		case 0:
			{
				// k_log(ell)	 = ((x_i - x_i') / ell^2) * (K_FF_log(ell) - 2K_FF)
				MatrixPtr pK_FF = K_FF(pSqDist, pLogHyp); // K_FF
				(*pK_FD) = inv_ell2 * pDelta->cwiseProduct(*pK_FD - (((Scalar) 2.f) * (*pK_FF)));
				//std::cout << "K_FD_log(ell) = " << std::endl << *pK_FD << std::endl << std::endl;
				break;
			}

		// derivatives of covariance matrix w.r.t log sigma_f
		case 1:
			{
				// k_log(sigma_f) = ((x_i - x_i') / ell^2) * K_FF_log(sigma_f)
				(*pK_FD) = inv_ell2 * pDelta->cwiseProduct(*pK_FD);
				//std::cout << "K_FD_log(sigma_f) = " << std::endl << *pK_FD << std::endl << std::endl;
				break;
			}
		}

		return pK_FD;
	}

	static MatrixPtr K_DD(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta, const int pdIndex)
	{
		// input
		// pSqDist (nxm): squared distances = r^2
		// pDelta1 (nxm): delta = x_i - x_i'
		// i: index for delta1
		// pDelta2 (nxm): delta = x_j - x_j'
		// j: index for delta2
		// pLogHyp: log hyperparameters
		// pdIndex: partial derivatives with respect to this parameter index

		// output
		// K: nxm matrix
		// if pdIndex == -1:		K_FF
		// else							partial K_FF / partial theta_i
		MatrixPtr pK_DD = K_FF(pSqDist, pLogHyp, pdIndex);

		// hyperparameters
		Scalar inv_ell2				= exp(((Scalar) -2.f) * (*pLogHyp)(0));
		Scalar inv_ell4				= exp(((Scalar) -4.f) * (*pLogHyp)(0));
		Scalar neg2_inv_ell2	= ((Scalar) -2.f) * inv_ell2;
		Scalar four_inv_ell4		= ((Scalar) 4.f) * inv_ell4;

		// delta
		Scalar delta = (i == j) ? (Scalar) 1.f  : (Scalar) 0.f;

#if 0
		// mode
		switch(pdIndex)
		{
		// covariance matrix
		case -1:
			{
				// k(X, X') = [ delta / ell^2 - ((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF(X, X')
				(*pK_DD) = (inv_ell2*delta - inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_DD->array();
				//std::cout << "K_DD = " << std::endl << *pK_DD << std::endl << std::endl;
				break;
			}

		// derivatives of covariance matrix w.r.t log ell
		case 0:
			{
				// k_log(ell)	 = [ -2*delta / ell^2 + 4*((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF(X, X')
				//                   + [ delta / ell^2 - ((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF_log(ell)(X, X')
				MatrixPtr pK_FF = K_FF(pSqDist, pLogHyp); // K_FF
				(*pK_DD) = (neg2_inv_ell2*delta + four_inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_FF->array()
									+ (inv_ell2*delta - inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_DD->array();
				//std::cout << "K_DD_log(ell) = " << std::endl << *pK_DD << std::endl << std::endl;
				break;
			}

		// derivatives of covariance matrix w.r.t log sigma_f
		case 1:
			{
				// k_log(sigma_f) = [ delta / ell^2 - ((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF_log(sigma_f)(X, X')
				(*pK_DD) = (inv_ell2*delta - inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_DD->array();
				//std::cout << "K_DD_log(sigma_f) = " << std::endl << *pK_DD << std::endl << std::endl;
				break;
			}
		}
#else
		// simplified version

		// for all cases
		(*pK_DD) = (inv_ell2*delta - inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_DD->array();

		// particularly, derivatives of covariance matrix w.r.t log ell
		if(pdIndex == 0)
			(*pK_DD) += (neg2_inv_ell2*delta + four_inv_ell4*(pDelta1->array())*(pDelta2->array())).matrix().cwiseProduct(*(K_FF(pSqDist, pLogHyp)));
			//(*pK_DD) += (neg2_inv_ell2*delta + four_inv_ell4*(pDelta1->array())*(pDelta2->array())) * (K_FF(pSqDist, pLogHyp)->array());
#endif

		return pK_DD;
	}
};

}

#endif