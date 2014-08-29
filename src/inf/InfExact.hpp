#ifndef _INFERENCE_METHOD_EXACT_HPP_
#define _INFERENCE_METHOD_EXACT_HPP_

#include <iostream>		// for std::cerr
#include <limits>			// for std::numeric_limits<Scalar>::infinity();

#include "../util/macros.h"
#include "../util/Exception.hpp"
#include "../util/LogFile.hpp"
#include "../data/TrainingData.hpp"
#include "../data/TestData.hpp"
#include "Hyp.hpp"

namespace GP{

/**
  * @class		InfExact
  * @brief		Exact inference
  * @ingroup	-Inf
  * @author		Soohwan Kim
  * @date		02/07/2014
  */
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc>
class InfExact
{
// define matrix, vector and cholesky factor types
protected:	TYPE_DEFINE_MATRIX(Scalar);
				TYPE_DEFINE_VECTOR(Scalar);
				TYPE_DEFINE_CHOLESKYFACTOR();

// define hyperparameter type
public:		TYPE_DEFINE_ALL_HYP(Scalar, MeanFunc, CovFunc, LikFunc);

public:
	/**
		* @brief	Predict the mean and [co]variance.
		* @note	mu = ms + ks' * inv(Kn) * (y - m) 
		*           = ms + ks' * alpha
		*        sigma^2 = kss + ks' * inv(Kn) * ks
		*                = kss + v' * v
		* @param [in]		logHyp				The log hyperparameters.
		* @param [in]		pXs				 	The test positions.
		* @param [out]		pMu	 				The mean vector.
		* @param [out]		pSigma 				The covariance matrix or variance vector.
		* @param [in]		fVarianceVector 	(Optional) flag for true: variance vector, false: covariance matrix
		* @param [in]		fBatchProcessing	(Optional) flag for the batch processing.
		*/
	static void predict /* throw (Exception) */
							 (const Hyp							&logHyp, 
							 TrainingData<Scalar>			&trainingData, 
							 TestData<Scalar>					&testData,
							 const bool							fVarianceVector = true,
							 const int							perBatch = 1000)
	{
		// number of data
		const int N = trainingData.N();
		const int M = testData.M();
		assert(N > 0 && M > 0);
		if(N <= 0 || M <= 0) return;

		// some constants
		// Note that we make the cholesky factor not to throw an exception even the covariance matrix is numerically singular
		const bool fDoNotThrowException = true;
		//const VectorConstPtr				pInvSqrtD	= invSqrtD(logHyp.lik, trainingData);
		//const CholeskyFactorConstPtr	pL				= choleskyFactor(logHyp.cov, trainingData, pInvSqrtD, fDoNotThrowException);
		//const VectorConstPtr				pY_M			= y_m(logHyp.mean, trainingData);
		//const VectorConstPtr				pAlpha		= alpha(pInvSqrtD, pL, pY_M);
		const VectorConstPtr				pInvSqrtD	= invSqrtD(logHyp.lik, trainingData);
		const CholeskyFactorConstPtr	pL				= choleskyFactor(logHyp.cov, logHyp.lik, trainingData, fDoNotThrowException);
		const VectorConstPtr				pY_M			= y_m(logHyp.mean, trainingData);
		const VectorConstPtr				pAlpha		= alpha(logHyp.lik, pL, pY_M);

		// too many test points: batch
		if(!fVarianceVector || perBatch <= 0)
		{
			predict(logHyp, trainingData, testData, pInvSqrtD, pL, pAlpha, fVarianceVector);
		}
		else
		{
			// memory allocation
			testData.pMu().reset(new Vector(M));
			testData.pSigma().reset(new Matrix(M, 1)); // variance vector (Mx1)

			// batch processing
			int from	= 0;
			int to	= -1;
			while(to < M-1)
			{
				// range
				from	= to + 1;
				to		= (M-1 < from + perBatch - 1) ? M-1 : from + perBatch - 1;
				const int MM = to - from + 1;

				// part of test data
				TestData<Scalar> testDataPart(testData, from, MM);

				// predict
				predict(logHyp, trainingData, testDataPart, pInvSqrtD, pL, pAlpha, fVarianceVector);

				// copy
				testData.pMu()->segment(from, MM).noalias()			= (*testDataPart.pMu());
				testData.pSigma()->middleRows(from, MM).noalias()	= (*testDataPart.pSigma());
			}
		}
	}

	// nlZ, dnlZ
	static void negativeLogMarginalLikelihood /* throw (Exception) */
														  (const Hyp					&logHyp,
															TrainingData<Scalar>		&trainingData, 
															Scalar						&nlZ, 
															VectorPtr					&pDnlZ,
															const int					calculationMode = 0)
	{
		// calculationMode
		// [0]: calculate both nlZ and pDnlZ
		// [+]: calculate nlZ only
		// [-]: calculate pDnlZ only

		// number of training data
		const int N = trainingData.N();
		assert(N > 0);
		if(N <= 0) return;

		// some constants
		const Scalar sn = exp(logHyp.lik(0));	// sn

		CholeskyFactorConstPtr	pL;
		//try
		//{
			//pL = choleskyFactor(logHyp.cov, trainingData, pInvSqrtD);
			pL	= choleskyFactor(logHyp.cov, logHyp.lik, trainingData);
		//}
		//catch(Exception &e) // if Kn is non positivie definite, nlZ = Inf, dnlZ = zeros
		//{
		//	std::cerr << e.what() << std::endl;
		//	nlZ = std::numeric_limits<Scalar>::infinity();
		//	pDnlZ.reset(new Vector(logHyp.size()));
		//	pDnlZ->setZero();
		//	return;
		//}
		const VectorConstPtr		pY_M		= y_m(logHyp.mean, trainingData);
		const VectorConstPtr		pAlpha	= alpha(logHyp.lik, pL, pY_M);

		// marginal likelihood
		// p(y) = N(m, Kn) = (2pi)^(-n/2) * |Kn|^(-1/2) * exp[(-1/2) * (y-m)' * inv(Kn) * (y-m)]
		// nlZ  = (1/2) * (y-m)' * inv(Kn) * (y-m)	+ (1/2) * log |Kn|									+ (n/2) * log(2pi)
		//      = (1/2) * (y-m)' * alpha					+ (1/2) * log |D^(1/2)*L*L'*D^(1/2)|			+ (n/2) * log(2pi)
		//      = (1/2) * (y-m)' * alpha					+ (1/2) * log |D^(1/2)|*|L|*|L'|*|D^(1/2)|	+ (n/2) * log(2pi)
		//      = (1/2) * (y-m)' * alpha					+ log |L|      + log |D^(1/2)|					+ (n/2) * log(2pi)
		//      = (1/2) * (y-m)' * alpha					+ log |L|      - log |D^(-1/2)|					+ (n/2) * log(2pi)
		//      = (1/2) * (y-m)' * alpha					+ tr[log (L)]	- tr[log(D^(-1/2))]				+ (n/2) * log(2pi)
		if(calculationMode >= 0)
		{
			nlZ = static_cast<Scalar>(0.5f) * (*pY_M).dot(*pAlpha)
					//+ L.diagonal().array().log().sum()
					+ pL->matrixL().nestedExpression().diagonal().array().log().sum()
					+ static_cast<Scalar>(N) * log(sn)
					+ static_cast<Scalar>(N) * static_cast<Scalar>(0.918938533204673f); // log(2pi)/2 = 0.918938533204673
		}

		// partial derivatives w.r.t hyperparameters
		if(calculationMode <= 0)
		{
			// derivatives (f_j = partial f / partial x_j)
			int j = 0; // partial derivative index
			pDnlZ.reset(new Vector(logHyp.size()));

			// (1) w.r.t the mean parameters
			// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m)
			//       = - m' * inv(Kn) * y + (1/2) m' * inv(Kn) * m
			// nlZ_i = - m_i' * inv(Kn) * y + m_i' * inv(Kn) * m
			//       = - m_i' * inv(Kn) (y - m)
			//       = - m_i' * alpha
			for(int i = 0; i < logHyp.mean.size(); i++)
			{
				(*pDnlZ)(j++) = - MeanFunc<Scalar>::m(logHyp.mean, trainingData, i)->dot(*pAlpha);
			}

			// (2) w.r.t the cov parameters
			// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m) + (1/2) * log |Kn|
			// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * K_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * K_j]
			//          = (-1/2) * alpha' * K_j * alpha							+ (1/2) * tr[inv(Kn) * K_j]
			//          = (-1/2) * tr[(alpha*alpha') * K_j]						+ (1/2) * tr[inv(Kn) * K_j]
			//          = (1/2) tr[(inv(Kn) - alpha*alpha') * K_j]
			//          = (1/2) tr[Q * K_j]
			//
			// Q = inv(Kn) - alpha*alpha'
			//
			// Kn * inv(Kn) = I
			// => D^(1/2) * LL' * D^(1/2) * inv(Kn) = I
			// => LL' * D^(1/2) * inv(Kn) = D^(-1/2)
			// => D^(1/2) * inv(Kn) = L.solve(D^(-1/2))
			// => inv(Kn) = D^(-1/2) * L.solve(D^(-1/2))

			MatrixPtr pQ = q(logHyp.lik, pL, pAlpha);

			for(int i = 0; i < logHyp.cov.size(); i++)
			{
				(*pDnlZ)(j++) = static_cast<Scalar>(0.5f) * pQ->cwiseProduct(*(CovFunc<Scalar>::K(logHyp.cov, trainingData, i))).sum();
			}

			// (3) w.r.t the lik parameters
			// nlZ = (1/2) * (y-m)' * inv(K + D) * (y-m) + (1/2) * log |K + D|
			// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * D_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * D_j]
			//          = (-1/2) * alpha' * D_j * alpha							+ (1/2) * tr[inv(Kn) * D_j]
			//          = (-1/2) * tr[(alpha' * alpha) * D_j]					+ (1/2) * tr[inv(Kn) * D_j]
			//          = (1/2) tr[(inv(Kn) - alpha*alpha') * D_j]
			//          = (1/2) tr[Q * D_j]

			//(*pDnlZ)(j++) = dnlZWRTLikHyp(logHyp.lik, trainingData, pQ);
			(*pDnlZ)(j++) = dnlZWRTLikHyp(logHyp.lik, pQ);

		}
	}

	// nlZ, dnlZ
	static void negativeLogMarginalLikelihood2 /* throw (Exception) */
															(const Hyp					&logHyp,
															 TrainingData<Scalar>	&trainingData, 
															 Scalar						&nlZ, 
															 VectorPtr					&pDnlZ,
															 const int					calculationMode = 0)
	{
		// calculationMode
		// [0]: calculate both nlZ and pDnlZ
		// [+]: calculate nlZ only
		// [-]: calculate pDnlZ only

		// number of training data
		const int N = trainingData.N();
		assert(N > 0);
		if(N <= 0) return;

		// some constants
		const VectorConstPtr pInvSqrtD = invSqrtD(logHyp.lik, trainingData);
		const Scalar sn = exp(logHyp.lik(0));	// sn

		CholeskyFactorConstPtr	pL;
		//try
		//{
			pL = choleskyFactor(logHyp.cov, trainingData, pInvSqrtD);
		//}
		//catch(Exception &e) // if Kn is non positivie definite, nlZ = Inf, dnlZ = zeros
		//{
		//	std::cerr << e.what() << std::endl;
		//	nlZ = std::numeric_limits<Scalar>::infinity();
		//	pDnlZ.reset(new Vector(logHyp.size()));
		//	pDnlZ->setZero();
		//	return;
		//}
		const VectorConstPtr		pY_M		= y_m(logHyp.mean, trainingData);
		const VectorConstPtr		pAlpha	= alpha(pInvSqrtD, pL, pY_M);

		// marginal likelihood
		// p(y) = N(m, Kn) = (2pi)^(-n/2) * |Kn|^(-1/2) * exp[(-1/2) * (y-m)' * inv(Kn) * (y-m)]
		// nlZ  = (1/2) * (y-m)' * inv(Kn) * (y-m)	+ (1/2) * log |Kn|									+ (n/2) * log(2pi)
		//      = (1/2) * (y-m)' * alpha					+ (1/2) * log |D^(1/2)*L*L'*D^(1/2)|			+ (n/2) * log(2pi)
		//      = (1/2) * (y-m)' * alpha					+ (1/2) * log |D^(1/2)|*|L|*|L'|*|D^(1/2)|	+ (n/2) * log(2pi)
		//      = (1/2) * (y-m)' * alpha					+ log |L|      + log |D^(1/2)|					+ (n/2) * log(2pi)
		//      = (1/2) * (y-m)' * alpha					+ log |L|      - log |D^(-1/2)|					+ (n/2) * log(2pi)
		//      = (1/2) * (y-m)' * alpha					+ tr[log (L)]	- tr[log(D^(-1/2))]				+ (n/2) * log(2pi)
		if(calculationMode >= 0)
		{
			nlZ = static_cast<Scalar>(0.5f) * (*pY_M).dot(*pAlpha)
					//+ L.diagonal().array().log().sum()
					+ pL->matrixL().nestedExpression().diagonal().array().log().sum()
					- pInvSqrtD->array().log().sum()
					+ static_cast<Scalar>(N) * static_cast<Scalar>(0.918938533204673f); // log(2pi)/2 = 0.918938533204673
		}

		// partial derivatives w.r.t hyperparameters
		if(calculationMode <= 0)
		{
			// derivatives (f_j = partial f / partial x_j)
			int j = 0; // partial derivative index
			pDnlZ.reset(new Vector(logHyp.size()));

			// (1) w.r.t the mean parameters
			// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m)
			//       = - m' * inv(Kn) * y + (1/2) m' * inv(Kn) * m
			// nlZ_i = - m_i' * inv(Kn) * y + m_i' * inv(Kn) * m
			//       = - m_i' * inv(Kn) (y - m)
			//       = - m_i' * alpha
			for(int i = 0; i < logHyp.mean.size(); i++)
			{
				(*pDnlZ)(j++) = - MeanFunc<Scalar>::m(logHyp.mean, trainingData, i)->dot(*pAlpha);
			}

			// (2) w.r.t the cov parameters
			// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m) + (1/2) * log |Kn|
			// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * K_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * K_j]
			//          = (-1/2) * alpha' * K_j * alpha							+ (1/2) * tr[inv(Kn) * K_j]
			//          = (-1/2) * tr[(alpha*alpha') * K_j]						+ (1/2) * tr[inv(Kn) * K_j]
			//          = (1/2) tr[(inv(Kn) - alpha*alpha') * K_j]
			//          = (1/2) tr[Q * K_j]
			//
			// Q = inv(Kn) - alpha*alpha'
			//
			// Kn * inv(Kn) = I
			// => D^(1/2) * LL' * D^(1/2) * inv(Kn) = I
			// => LL' * D^(1/2) * inv(Kn) = D^(-1/2)
			// => D^(1/2) * inv(Kn) = L.solve(D^(-1/2))
			// => inv(Kn) = D^(-1/2) * L.solve(D^(-1/2))

			MatrixPtr pQ = q(pInvSqrtD, pL, pAlpha);

			for(int i = 0; i < logHyp.cov.size(); i++)
			{
				(*pDnlZ)(j++) = static_cast<Scalar>(0.5f) * pQ->cwiseProduct(*(CovFunc<Scalar>::K(logHyp.cov, trainingData, i))).sum();
			}

			// (3) w.r.t the lik parameters
			// nlZ = (1/2) * (y-m)' * inv(K + D) * (y-m) + (1/2) * log |K + D|
			// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * D_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * D_j]
			//          = (-1/2) * alpha' * D_j * alpha							+ (1/2) * tr[inv(Kn) * D_j]
			//          = (-1/2) * tr[(alpha' * alpha) * D_j]					+ (1/2) * tr[inv(Kn) * D_j]
			//          = (1/2) tr[(inv(Kn) - alpha*alpha') * D_j]
			//          = (1/2) tr[Q * D_j]

			//(*pDnlZ)(j++) = dnlZWRTLikHyp(logHyp.lik, trainingData, pQ);
			(*pDnlZ)(j++) = dnlZWRTLikHyp(logHyp.lik, pQ);

		}
	}

protected:
	/** @brief	Predict the mean and the [co]variance 
	  * @note	Please note that the [co]variance is 
	  *			not for latent function outputs but for function outputs.
	  *			This is because the Kss is sometimes numerically singular.
	  */
	static void predict /* throw (Exception) */
							 (const Hyp								&logHyp,
							  TrainingData<Scalar>				&trainingData, 
							  TestData<Scalar>					&testData,
							  const VectorConstPtr				pInvSqrtD,
							  const CholeskyFactorConstPtr	pL, 
							  const VectorConstPtr				pAlpha,
							  const bool							fVarianceVector)
	{
		// number of data
		const int N = trainingData.N();
		const int M = testData.M();

		// Ks, Kss
		MatrixConstPtr pKs		= CovFunc<Scalar>::Ks (logHyp.cov, trainingData, testData); // N x M
		MatrixConstPtr pKss	= CovFunc<Scalar>::Kss(logHyp.cov, testData, fVarianceVector); // Vector (Mx1) or Matrix (MxM)
		//MatrixPtr pKss				= CovFunc<Scalar>::Kss(logHyp.cov, testData, fVarianceVector); // Vector (Mx1) or Matrix (MxM)

		// D = sn2*I
		// not for latent function outputs but for function outputs
		//VectorPtr pD = LikFunc<Scalar>::lik(logHyp.lik, testData);
		//if(fVarianceVector)	(*pKss) += (*pD);
		//else						(*pKss) += pD->asDiagonal();

		// [1] predictive mean
		// mu = ms + Ks' * inv(Kn) * (y-m) = ms + ks' * alpha
		// Mx1  Mx1  MxN    NxN       Nx1
		testData.pMu().reset(new Vector(M));
		testData.pMu()->noalias() = *(MeanFunc<Scalar>::ms(logHyp.mean, testData))
										  + (pKs->transpose()) * (*pAlpha);

		// [2] predictive variance
		// (1) scalar
		// 1x1     1x1   1xN   NxN      Nx1
		// sigma = kss - ks' * inv(Kn) * ks
		//       = kss - ks' * inv(D^(1/2) * (D^(-1/2) * K * D^(-1/2) + I) * D^(1/2)) * ks
		//       = kss - ks' * inv(D^(1/2) * LL' * D^(1/2)) * ks
		//       = kss - ks' * D^(-1/2) * inv(L') * inv(L) * D^(-1/2) * ks
		//       = kss - (inv(L) * D^(-1/2) * ks)' * (inv(L) * D^(-1/2) * ks)
		//       = kss - v' * v
		//         1x1  1xN  Nx1
		//
		// (2) vector
		// V = (inv(L) * D^(-1/2) * Ks = [v1, v2, ..., vM]
		// NxM  NxN      NxN       NxM   Nx1  Nx1      Nx1
		//
		// Mx1     Mx1      Mx1
		// sigma = kss - [v1' * v1] = kss - sum(V.*V, 1)'
		//               [v2' * v2]
		//                   ...
		//               [vM' * vM]
		//
		// MxM     MxM   MxN   NxN      NxM
		// Sigma = Kss - Ks' * inv(Kn) * Ks
		//       = Kss - Ks' * inv(D^(1/2) * (D^(-1/2) * K * D^(-1/2) + I) * D^(1/2)) * Ks
		//       = Kss - Ks' * inv(D^(1/2) * LL' * D^(1/2)) * Ks
		//       = Kss - Ks' * D^(-1/2) * inv(L') * inv(L) * D^(-1/2) * Ks
		//       = Kss - (inv(L) * D^(-1/2) * Ks)' * (inv(L) * D^(-1/2) * Ks)
		//       = Kss - V' * V
		//               MxN  NxM

		// V = inv(L) * D^(-1/2) * Ks
		// NxN   NxN      NxN     NxM
		Matrix V(N, M); // N x M
		V.noalias() = pL->matrixL().solve(pInvSqrtD->replicate(1, M).cwiseProduct(*pKs));
		//V.noalias() = pL->matrixL().solve(pInvSqrtD->asDiagonal() * (*pKs));
		//V.noalias() = pL->getL().solve(m_pInvSqrtD->asDiagonal() * (*pKs));

		if(fVarianceVector)
		{
			// sigma2 = kss - v' * v
			testData.pSigma().reset(new Matrix(M, 1)); // variance vector (Mx1)
			testData.pSigma()->noalias() = (*pKss) - V.transpose().array().square().matrix().rowwise().sum();
		}
		else
		{
			// Sigma = Kss - V' *V
			testData.pSigma().reset(new Matrix(M, M)); // covariance matrix (MxM)
			testData.pSigma()->noalias() = (*pKss) - V.transpose() * V;
		}
	}

protected:
	static VectorPtr invSqrtD(const typename Hyp::Lik	&logHyp,
									  TrainingData<Scalar>		&trainingData)
	{
		// D = sW = sn2*I
		VectorPtr pInvSqrtD = LikFunc<Scalar>::lik(logHyp, trainingData);

		// D^(-1/2)
		(*pInvSqrtD) = pInvSqrtD->cwiseSqrt().cwiseInverse();

		return pInvSqrtD;
	}

	static CholeskyFactorPtr choleskyFactor /* throw (Exception) */
														(const typename Hyp::Cov	&logHyp,
														 TrainingData<Scalar>		&trainingData,
														 const VectorConstPtr		pInvSqrtD,
														 const bool						fDoNotThrowException = false)
	{
		// number of training data
		const int N = trainingData.N();

		// K
		MatrixPtr pKn = CovFunc<Scalar>::K(logHyp, trainingData);

		// Kn = K + sn*I
		//    = K + D
		//    = D^(1/2) * (D^(-1/2)*K*D^(-1/2) + I) * D^(1/2)
		//    = D^(1/2) * (L*L') * D^(1/2)
		//
		// instead of						LL' = K + D
		// for numerical stability,	LL' = D^(-1/2)*K*D^(-1/2) + I 
		(*pKn) = pInvSqrtD->asDiagonal() * (*pKn) * pInvSqrtD->asDiagonal() + Matrix::Identity(N, N);

		// cholesky factor
		//CholeskyFactorPtr pL(new CholeskyFactor());
		//pL->compute(*pKn);	// compute the Cholesky decomposition of Kn
		CholeskyFactorPtr pL(new CholeskyFactor(*pKn));
		if(fDoNotThrowException)
		{
			// add the diagonal term until it is numerically non-singular
			int num_iters(-1);
			float factor;
			while(pL->info() != Eigen::/*ComputationInfo::*/Success)
			{
				num_iters++;
				factor = powf(10.f, static_cast<float>(num_iters)) * Epsilon<float>::value;
				pKn->noalias() += factor * Matrix::Identity(pKn->rows(), pKn->cols());
				pL->compute(*pKn);
			}
			if(num_iters > 0)
			{
				LogFile logFile;
				logFile << "InfExact::choleskyFactor::num_iters: " << num_iters << "(" << factor << ")" << std::endl;
			}
		}
		if(pL->info() != Eigen::/*ComputationInfo::*/Success)
		{
			Exception e;
			switch(pL->info())
			{
				case Eigen::/*ComputationInfo::*/NumericalIssue :
				{
					e = "InfExact::choleskyFactor1::NumericalIssue";
					break;
				}
				case Eigen::/*ComputationInfo::*/NoConvergence :
				{
					e = "InfExact::choleskyFactor1::NoConvergence";
					break;
				}
#if EIGEN_VERSION_AT_LEAST(3,2,0)
				case Eigen::/*ComputationInfo::*/InvalidInput :
				{
					e = "InfExact::choleskyFactor1::InvalidInput";
					break;
				}
#endif
			}
			throw e;
		}

		return pL;
	}

	static CholeskyFactorPtr choleskyFactor /* throw (Exception) */
														(const typename Hyp::Cov	&covLogHyp,
														 const typename Hyp::Lik	&likLogHyp,
														 TrainingData<Scalar>		&trainingData,
														 const bool						fDoNotThrowException = false)
	{
		// number of training data
		const int N = trainingData.N();

		// constants
		const Scalar sn2 = exp(static_cast<Scalar>(2.f) * likLogHyp(0));	// sn^2

		// K
		MatrixPtr pKn = CovFunc<Scalar>::K(covLogHyp, trainingData);

		// Kn = K + sn^2*I
		//    = sn^2 * (K/sn^2 + I)
		//    = sn^2 * (L*L')
		//
		// instead of						LL' = K + D
		// for numerical stability,	LL' = K/sn2 + I 
		(*pKn) = (*pKn)/sn2 + Matrix::Identity(N, N);

		// cholesky factor
		//CholeskyFactorPtr pL(new CholeskyFactor());
		//pL->compute(*pKn);	// compute the Cholesky decomposition of Kn
		CholeskyFactorPtr pL(new CholeskyFactor(*pKn));
		if(fDoNotThrowException)
		{
			// add the diagonal term until it is numerically non-singular
			int num_iters(-1);
			float factor;
			while(pL->info() != Eigen::/*ComputationInfo::*/Success)
			{
				num_iters++;
				factor = powf(10.f, static_cast<float>(num_iters)) * Epsilon<float>::value;
				pKn->noalias() += factor * Matrix::Identity(pKn->rows(), pKn->cols());
				pL->compute(*pKn);
			}
			if(num_iters > 0)
			{
				LogFile logFile;
				logFile << "InfExact::choleskyFactor2::num_iters: " << num_iters << "(" << factor << ")" << std::endl;
			}
		}
		if(pL->info() != Eigen::/*ComputationInfo::*/Success)
		{
			Exception e;
			switch(pL->info())
			{
				case Eigen::/*ComputationInfo::*/NumericalIssue :
				{
					e = "InfExact::choleskyFactor2::NumericalIssue";
					break;
				}
				case Eigen::/*ComputationInfo::*/NoConvergence :
				{
					e = "InfExact::choleskyFactor2::NoConvergence";
					break;
				}
#if EIGEN_VERSION_AT_LEAST(3,2,0)
				case Eigen::/*ComputationInfo::*/InvalidInput :
				{
					e = "InfExact::choleskyFactor2::InvalidInput";
					break;
				}
#endif
			}
			throw e;
		}

		return pL;
	}

	static VectorPtr y_m(const typename Hyp::Mean	&logHyp,
								TrainingData<Scalar>			&trainingData)
	{
		// number of training data
		const int N = trainingData.N();

		// memory allocation
		VectorPtr pY_M(new Vector(N));

		// y - m
		pY_M->noalias() = (*trainingData.pY()) - (*(MeanFunc<Scalar>::m(logHyp, trainingData)));

		return pY_M;
	}

	static VectorPtr alpha(const VectorConstPtr				pInvSqrtD,
								  const CholeskyFactorConstPtr	pL,
								  const VectorConstPtr				pY_M)
	{
		// memory allocation
		VectorPtr pAlpha(new Vector(pY_M->size()));

		// alpha = inv(Kn)*(y-m)
		// => (K + D) * alpha = y - m
		// =>  D^(1/2) * (D^(-1/2) * K * D^(-1/2) + I ) * D^(1/2) * alpha = y - m
		// =>  D^(1/2) * L * L' * D^(1/2) * alpha = y - m
		// => L * L' * D^(1/2) * alpha = D^(-1/2) * (y - m)
		// => D^(1/2) * alpha = L.solve(D^(-1/2) * (y - m))
		// => alpha = D^(-1/2) * L.solve(D^(-1/2) * (y - m))
		pAlpha->noalias() = pInvSqrtD->asDiagonal()
								  * (pL->solve(pInvSqrtD->asDiagonal() * (*pY_M)));
		return pAlpha;
	}

	static VectorPtr alpha(const typename Hyp::Lik			&logHyp,
								  const CholeskyFactorConstPtr	pL,
								  const VectorConstPtr				pY_M)
	{
		// memory allocation
		VectorPtr pAlpha(new Vector(pY_M->size()));

		// alpha = inv(Kn) * (y-m)
		//       = inv(sn2*(K/sn2 + I)) * (y-m)
		//       = (1/sn2)*inv(LL')*(y-m)
		//       = (1/sn2)*L.solve(y-m)
		pAlpha->noalias() = pL->solve(*pY_M) / exp(static_cast<Scalar>(2.f) * logHyp(0));
		return pAlpha;
	}

protected:
	static MatrixPtr q(const VectorConstPtr				pInvSqrtD,
							 const CholeskyFactorConstPtr		pL,
							 const VectorConstPtr				pAlpha)
	{
		// Q = inv(Kn) - alpha*alpha'
		//
		// Kn * inv(Kn) = I
		// => D^(1/2) * LL' * D^(1/2) * inv(Kn) = I
		// => LL' * D^(1/2) * inv(Kn) = D^(-1/2)
		// => D^(1/2) * inv(Kn) = L.solve(D^(-1/2))
		// => inv(Kn) = D^(-1/2) * L.solve(D^(-1/2))
		const int N = pAlpha->size();
		MatrixPtr pQ(new Matrix(N, N)); // nxn
		pQ->noalias() = pInvSqrtD->asDiagonal() * (pL->solve(Matrix(pInvSqrtD->asDiagonal())))
						  - (*pAlpha) * (pAlpha->transpose());

		return pQ;
	}

	static MatrixPtr q(const typename Hyp::Lik			&logHyp,
							 const CholeskyFactorConstPtr		pL,
							 const VectorConstPtr				pAlpha)
	{
		// Q = inv(Kn) - alpha*alpha'
		//
		// Kn * inv(Kn) = I
		// => D^(1/2) * LL' * D^(1/2) * inv(Kn) = I
		// => LL' * D^(1/2) * inv(Kn) = D^(-1/2)
		// => D^(1/2) * inv(Kn) = L.solve(D^(-1/2))
		// => inv(Kn) = D^(-1/2) * L.solve(D^(-1/2))
		// => inv(Kn) = D^(-1) * L.solve(eye)
		// => inv(Kn) = (1/sn2) * L.solve(eye)
		const int N = pAlpha->size();
		MatrixPtr pQ(new Matrix(N, N)); // nxn
		pQ->noalias() = pL->solve(Matrix::Identity(N, N)) / exp(static_cast<Scalar>(2.f) * logHyp(0))
						  - (*pAlpha) * (pAlpha->transpose());
		return pQ;
	}

	static Scalar dnlZWRTLikHyp(const typename Hyp::Lik	&logHyp,
										 TrainingData<Scalar>		&trainingData,
										 const MatrixConstPtr		pQ)
	{
		// derivative w.r.t the lik parameters
		// nlZ = (1/2) * (y-m)' * inv(K + D) * (y-m) + (1/2) * log |K + D|
		// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * D_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * D_j]
		//          = (-1/2) * alpha' * D_j * alpha							+ (1/2) * tr[inv(Kn) * D_j]
		//          = (-1/2) * tr[(alpha' * alpha) * D_j]					+ (1/2) * tr[inv(Kn) * D_j]
		//          = (1/2) tr[(inv(Kn) - alpha*alpha') * D_j]
		//          = (1/2) tr[Q * D_j]
		return static_cast<Scalar>(0.5f) * pQ->cwiseProduct(Matrix(LikFunc<Scalar>::lik(logHyp, trainingData, 0)->asDiagonal())).sum();
	}

	static Scalar dnlZWRTLikHyp(const typename Hyp::Lik	&logHyp,
										 const MatrixConstPtr		pQ)
	{
		// derivative w.r.t the lik parameters
		// nlZ = (1/2) * (y-m)' * inv(K + D) * (y-m) + (1/2) * log |K + D|
		// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * D_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * D_j]
		//          = (-1/2) * alpha' * D_j * alpha							+ (1/2) * tr[inv(Kn) * D_j]
		//          = (-1/2) * tr[(alpha' * alpha) * D_j]					+ (1/2) * tr[inv(Kn) * D_j]
		//          = (1/2) tr[(inv(Kn) - alpha*alpha') * D_j]
		//          = (1/2) tr[Q * D_j]
		//          = (1/2) tr[Q * (2*sn2)*I]
		//          = (sn2) tr[Q]
		return exp(static_cast<Scalar>(2.f) * logHyp(0)) * pQ->trace();
	}
};
}

#endif