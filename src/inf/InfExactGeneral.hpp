#ifndef _INFERENCE_METHOD_EXACT_GENERAL_VERSION_HPP_
#define _INFERENCE_METHOD_EXACT_GENERAL_VERSION_HPP_

#include <iostream>		// for std::cerr
#include <limits>			// for std::numeric_limits<Scalar>::infinity();
#include <cmath>			// powf

#include "../util/macros.h"
#include "../util/Exception.hpp"
#include "../util/LogFile.hpp"
#include "../data/DerivativeTrainingData.hpp"
#include "../data/TestData.hpp"
#include "Hyp.hpp"

namespace GP{

/**
  * @class	InfExactGeneral
  * @brief	Exact inference with TrainingData or DerivativeTrainingData
  * @ingroup	-Inf
  * @author		Soohwan Kim
  * @date		02/07/2014
  */
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc>
class InfExactGeneral
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
	template<template<typename> class GeneralTrainingData>
	static void predict /* throw (Exception) */
							 (const Hyp							&logHyp, 
							  GeneralTrainingData<Scalar>	&generalTrainingData, 
							  TestData<Scalar>				&testData,
							  const bool						fVarianceVector = true,
							  const int							perBatch = 1000)
	{
		// number of data
		const int NN = generalTrainingData.NN();
		const int M  = testData.M();
		assert(NN > 0 && M > 0);
		if(NN <= 0 || M <= 0) return;

		// some constants
		// Note that we make the cholesky factor not to throw an exception even the covariance matrix is numerically singular
		const bool			fDoNotThrowException = true;
		const CholeskyFactorConstPtr	pL			= choleskyFactor(logHyp, generalTrainingData, fDoNotThrowException);
		const VectorConstPtr				pY_M		= y_m(logHyp.mean, generalTrainingData);
		const VectorConstPtr				pAlpha	= alpha(pL, pY_M);

		predict_given_precalculation(logHyp, generalTrainingData, testData, pL, pAlpha, fVarianceVector, perBatch);
	}

	// nlZ, dnlZ
	// TODO: separate nlZ and dnlZ into different methods
	template<template<typename> class GeneralTrainingData>
	static void negativeLogMarginalLikelihood /* throw (Exception) */
														  (const Hyp							&logHyp,
															GeneralTrainingData<Scalar>	&generalTrainingData, 
															Scalar								&nlZ, 
															VectorPtr							&pDnlZ,
															const int							calculationMode = 0)
	{
		// calculationMode
		// [0]: calculate both nlZ and pDnlZ
		// [+]: calculate nlZ only
		// [-]: calculate pDnlZ only

		// number of training data
		const int NN = generalTrainingData.NN();

		// some constants
		CholeskyFactorConstPtr	pL;
		//try
		//{
			pL = choleskyFactor(logHyp, generalTrainingData);
		//}
		//catch(Exception &e) // if Kn is non positivie definite, nlZ = Inf, dnlZ = zeros
		//{
		//	std::cerr << e.what() << std::endl;
		//	nlZ = std::numeric_limits<Scalar>::infinity();
		//	pDnlZ.reset(new Vector(logHyp.size()));
		//	pDnlZ->setZero();
		//	return;
		//}
		const VectorConstPtr pY_M		= y_m(logHyp.mean, generalTrainingData);
		const VectorConstPtr pAlpha	= alpha(pL, pY_M);


		// marginal likelihood
		// p(y) = N(m, Kn) = (2pi)^(-n/2) * |Kn|^(-1/2) * exp[(-1/2) * (y-m)' * inv(Kn) * (y-m)]
		// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m) + (1/2) * log |Kn|		+ (n/2) * log(2pi)
		//     = (1/2) * (y-m)' * alpha           + (1/2) * log |L*L'|		+ (n/2) * log(2pi)
		//     = (1/2) * (y-m)' * alpha           + (1/2) * log |L|*|L'|	+ (n/2) * log(2pi)
		//     = (1/2) * (y-m)' * alpha           + log |L||					+ (n/2) * log(2pi)
		//     = (1/2) * (y-m)' * alpha           + tr[log (L)]				+ (n/2) * log(2pi)
		if(calculationMode >= 0)
		{
			nlZ = static_cast<Scalar>(0.5f) * (*pY_M).dot(*pAlpha)
					+ pL->matrixL().nestedExpression().diagonal().array().log().sum()
					+ static_cast<Scalar>(NN) * static_cast<Scalar>(0.918938533204673f); // log(2pi)/2 = 0.918938533204673
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
				(*pDnlZ)(j++) = - MeanFunc<Scalar>::m(logHyp.mean, generalTrainingData, i)->dot(*pAlpha);
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

			MatrixPtr pQ = q(pL, pAlpha);

			for(int i = 0; i < logHyp.cov.size(); i++)
			{
				(*pDnlZ)(j++) = static_cast<Scalar>(0.5f) * pQ->cwiseProduct(*(CovFunc<Scalar>::K(logHyp.cov, generalTrainingData, i))).sum();
			}

			// (3) w.r.t the cov parameters
			// nlZ = (1/2) * (y-m)' * inv(K + D) * (y-m) + (1/2) * log |K + D|
			// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * D_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * D_j]
			//          = (-1/2) * alpha' * D_j * alpha							+ (1/2) * tr[inv(Kn) * D_j]
			//          = (-1/2) * tr[(alpha' * alpha) * D_j]					+ (1/2) * tr[inv(Kn) * D_j]
			//          = (1/2) tr[(inv(Kn) - alpha*alpha') * D_j]
			//          = (1/2) tr[Q * D_j]
			for(int i = 0; i < logHyp.lik.size(); i++)
			{
				//(*pDnlZ)(j++) = dnlZWRTLikHyp(logHyp.lik, generalTrainingData, pQ, i);
				(*pDnlZ)(j++) = dnlZWRTLikHyp2(logHyp.lik, generalTrainingData, pQ, i);
			}
		}
	}

protected:
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
	template<template<typename> class GeneralTrainingData>
	static void predict_given_precalculation /* throw (Exception) */
														 (const Hyp								&logHyp, 
														 GeneralTrainingData<Scalar>		&generalTrainingData, 
														 TestData<Scalar>						&testData,
														 const CholeskyFactorConstPtr		&pL, 
														 const VectorConstPtr				&pAlpha,
														 const bool								fVarianceVector = true,
														 const int								perBatch = 1000)
	{
		// number of data
		const int NN = generalTrainingData.NN();
		const int M  = testData.M();

		// some constants
		// Note that we make the cholesky factor not to throw an exception even the covariance matrix is numerically singular
		//const bool			fDoNotThrowException = true;
		//const CholeskyFactorConstPtr	pL			= choleskyFactor(logHyp, generalTrainingData, fDoNotThrowException);
		//const VectorConstPtr				pY_M		= y_m(logHyp.mean, generalTrainingData);
		//const VectorConstPtr				pAlpha	= alpha(pL, pY_M);

		// too many test points: batch
		if(!fVarianceVector || perBatch <= 0)
		{
			predict(logHyp, generalTrainingData, testData, pL, pAlpha, fVarianceVector);
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
				predict(logHyp, generalTrainingData, testDataPart, pL, pAlpha, fVarianceVector);

				// copy
				testData.pMu()->segment(from, MM).noalias()			= (*testDataPart.pMu());
				testData.pSigma()->middleRows(from, MM).noalias()	= (*testDataPart.pSigma());
			}
		}
	}

	/** @brief	Predict the mean and the [co]variance 
	  * @note	Please note that the [co]variance is 
	  *			not for latent function outputs but for function outputs.
	  *			This is because the Kss is sometimes numerically singular.
	  */
	template<template<typename> class GeneralTrainingData>
	static void predict /* throw (Exception) */
							 (const Hyp								&logHyp,
							  GeneralTrainingData<Scalar>		&generalTrainingData,
							  TestData<Scalar>					&testData,
							  const CholeskyFactorConstPtr	pL, 
							  const VectorConstPtr				pAlpha,
							  const bool							fVarianceVector)
	{
		// number of data
		const int NN = generalTrainingData.NN();
		const int M  = testData.M();

		// Ks, Kss
		MatrixConstPtr pKs		= CovFunc<Scalar>::Ks (logHyp.cov, generalTrainingData, testData); // NN x M
		MatrixConstPtr pKss	= CovFunc<Scalar>::Kss(logHyp.cov, testData, fVarianceVector); // Vector (Mx1) or Matrix (MxM)
		//MatrixPtr pKss				= CovFunc<Scalar>::Kss(logHyp.cov, testData, fVarianceVector); // Vector (Mx1) or Matrix (MxM)

		// D = sn2*I
		// not for latent function outputs but for function outputs
		//VectorPtr pD = LikFunc<Scalar>::lik(logHyp.lik, testData);
		//if(fVarianceVector)	(*pKss) += (*pD);
		//else						(*pKss) += pD->asDiagonal();

		// check pKss
		//CholeskyFactor	L(*pKss);
		//if(L.info() != Eigen::Success) std::cout << "Kss is not PSD!" << std::endl;

		// [1] predictive mean
		// mu = ms + Ks' * inv(Kn) * (y-m)
		//    = ms + Ks' * alpha
		testData.pMu().reset(new Vector(M));
		testData.pMu()->noalias() = *(MeanFunc<Scalar>::ms(logHyp.mean, testData))
										  + (pKs->transpose()) * (*pAlpha);

		// [2] predictive variance
		// Sigma = Kss - Ks' * inv(Kn) * Ks
		//       = Kss - Ks' * inv(D^(1/2) * (D^(-1/2) * K * D^(-1/2) + I) * D^(1/2)) * Ks
		//       = Kss - Ks' * inv(D^(1/2) * LL' * D^(1/2)) * Ks
		//       = Kss - Ks' * D^(-1/2) * inv(L') * inv(L) * D^(-1/2) * Ks
		//       = Kss - (inv(L) * D^(-1/2) * Ks)' * (inv(L) * D^(-1/2) * Ks)
		//       = Kss - V' * V

		// V = inv(L) * Ks 
		//  (NN x NN) * (NN x M)
		Matrix V(NN, M); // NN x M
		V = pL->matrixL().solve(*pKs);

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
			//testData.pSigma()->noalias() = (*pKss) - Matrix(V.transpose()) * V;
			//L.compute(*(testData.pSigma()));
			//if(L.info() != Eigen::Success) std::cout << "Sigma is not PSD!" << std::endl;
		}
	}

protected:
	template<template<typename> class GeneralTrainingData>
	static CholeskyFactorPtr choleskyFactor /* throw (Exception) */
														(const Hyp							&logHyp,
														 GeneralTrainingData<Scalar>	&generalTrainingData,
														 const bool							fDoNotThrowException = false)
	{
		// number of training data
		const int NN = generalTrainingData.NN();

		// K
		MatrixPtr pKn = CovFunc<Scalar>::K(logHyp.cov, generalTrainingData);

		// D = sn2*I
		VectorPtr pD = LikFunc<Scalar>::lik(logHyp.lik, generalTrainingData);

		// Kn = K + D = LL'
		(*pKn) += pD->asDiagonal();

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
				logFile << "InfExactGeneral::choleskyFactor::num_iters: " << num_iters << "(" << factor << ")" << std::endl;
			}
		}
		if(pL->info() != Eigen::/*ComputationInfo::*/Success)
		{
			Exception e;
			switch(pL->info())
			{
				case Eigen::/*ComputationInfo::*/NumericalIssue :
				{
					e = "InfExactGeneral::choleskyFactor::NumericalIssue";
					break;
				}
				case Eigen::/*ComputationInfo::*/NoConvergence :
				{
					e = "InfExactGeneral::choleskyFactor::NoConvergence";
					break;
				}
#if EIGEN_VERSION_AT_LEAST(3,2,0)
				case Eigen::/*ComputationInfo::*/InvalidInput :
				{
					e = "InfExactGeneral::choleskyFactor::InvalidInput";
					break;
				}
#endif
			}
			throw e;
		}

		return pL;
	}

	template<template<typename> class GeneralTrainingData>
	static VectorPtr y_m(const typename Hyp::Mean		&logHyp,
								GeneralTrainingData<Scalar>	&generalTrainingData)
	{
		// number of training data
		const int NN = generalTrainingData.NN();

		// memory allocation
		VectorPtr pY_M(new Vector(NN));

		// y - m
		pY_M->noalias() = (*generalTrainingData.pY()) - (*(MeanFunc<Scalar>::m(logHyp, generalTrainingData)));

		return pY_M;
	}

	static VectorPtr alpha(const CholeskyFactorConstPtr	pL,
								  const VectorConstPtr				pY_M)
	{
		// memory allocation
		VectorPtr pAlpha(new Vector(pY_M->size()));

		// alpha = inv(Kn)*(y-m)
		pAlpha->noalias() = pL->solve(*pY_M);

		return pAlpha;
	}

	static MatrixPtr q(const CholeskyFactorConstPtr		pL,
							 const VectorConstPtr				pAlpha)
	{
		// Q = inv(Kn) - alpha*alpha'
		const int NN = pAlpha->size();
		MatrixPtr pQ(new Matrix(NN, NN)); // NN x NN
		pQ->noalias() = pL->solve(Matrix::Identity(NN, NN))
							- (*pAlpha) * (pAlpha->transpose());
		return pQ;
	}

	template<template<typename> class GeneralTrainingData>
	static Scalar dnlZWRTLikHyp(const typename Hyp::Lik		&logHyp,
										 GeneralTrainingData<Scalar>	&generalTrainingData,
										 const MatrixConstPtr			pQ,
										 const int							pdHypIndex)
	{
		// derivative w.r.t the lik parameters
		// nlZ = (1/2) * (y-m)' * inv(K + D) * (y-m) + (1/2) * log |K + D|
		// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * D_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * D_j]
		//          = (-1/2) * alpha' * D_j * alpha							+ (1/2) * tr[inv(Kn) * D_j]
		//          = (-1/2) * tr[(alpha' * alpha) * D_j]					+ (1/2) * tr[inv(Kn) * D_j]
		//          = (1/2) tr[(inv(Kn) - alpha*alpha') * D_j]
		//          = (1/2) tr[Q * D_j]
		return static_cast<Scalar>(0.5f) * pQ->cwiseProduct(Matrix(LikFunc<Scalar>::lik(logHyp, generalTrainingData, pdHypIndex)->asDiagonal())).sum();
	}

	template<template<typename> class GeneralTrainingData>
	static Scalar dnlZWRTLikHyp2(const typename Hyp::Lik		&logHyp,
										  GeneralTrainingData<Scalar>	&generalTrainingData,
										  const MatrixConstPtr			pQ,
										  const int							pdHypIndex)
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

		const int N = generalTrainingData.N();
		const int NN = generalTrainingData.NN();
		if(pdHypIndex == 0)	return exp(static_cast<Scalar>(2.f) * logHyp(0)) * pQ->topLeftCorner(N, N).trace();
									return exp(static_cast<Scalar>(2.f) * logHyp(1)) * pQ->bottomRightCorner(NN-N, NN-N).trace();
	}
};

template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc>
class InfExactDerObs : public InfExactGeneral<Scalar, MeanFunc, CovFunc, LikFunc> {};

}

#endif