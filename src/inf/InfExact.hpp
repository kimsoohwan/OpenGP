#ifndef _INFERENCE_METHOD_EXACT_HPP_
#define _INFERENCE_METHOD_EXACT_HPP_

#include <exception>		// for throw std::exception during cholesky decomposition
#include <limits>			// for std::numeric_limits<Scalar>::infinity();

#include "../util/macros.hpp"
#include "../data/DerivativeTrainingData.hpp"
#include "../data/TestData.hpp"
#include "../util/Hyp.hpp"

namespace GP{

/**
	* @class	InfExact
	* @brief	Exact inference
	* @author	Soohwan Kim
	* @date	26/03/2014
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
				TYPE_DEFINE_CHOLESKYFACTOR(Scalar);

public:
	/**
		* @brief	Predict the mean and [co]variance.
		* @note	mu = ms + ks' * inv(Kn) * (y - m) 
		*           = ms + ks' * alpha
		*        sigma^2 = kss + ks' * inv(Kn) * ks
		*                = kss + v' * v
		* @param [in]		logHyp				The log hyperparameters.
		* @param [in]		pXs				 	The test positions.
		* @param [out]	pMu	 				The mean vector.
		* @param [out]	pSigma 				The covariance matrix or variance vector.
		* @param [in]		fVarianceVector 	(Optional) flag for true: variance vector, false: covariance matrix
		* @param [in]		fBatchProcessing	(Optional) flag for the batch processing.
		*/
	template<typename GeneralTrainingData>
	void predict /* throw std::exception */
					(const Hyp<Scalar, MeanFunc, CovFunc, LikFunc>	&logHyp, 
					       GeneralTrainingData<Scalar>					&generalTrainingData, 
					 const TestData<Scalar>									&testData,
					 const bool													fVarianceVector = true)
	{
		// number of data
		const int NN = generalTrainingData.NN();
		const int M  = testData.M();

		// some constants
		const VectorConstPtr				pInvSqrtD	= invSqrtD(logHyp.lik, generalTrainingData);
		const CholeskyFactorConstPtr	pL				= choleskyFactor(logHyp.cov, generalTrainingData, pInvSqrtD);
		const VectorConstPtr				pY_M			= y_m(logHyp.mean, generalTrainingData);
		const MatrixConstPtr				pAlpha		= alpha(pInvSqrtD, pL, pY_M);

		// Ks, Kss
		MatrixConstPtr pKs		= CovFunc<Scalar>::Ks (logHyp.cov, generalTrainingData, testData); // NN x M
		MatrixConstPtr pKss		= CovFunc<Scalar>::Kss(logHyp.cov, generalTrainingData, testData, fVarianceVector); // Vector (Mx1) or Matrix (MxM)
		//std::cout << "Ks = " << std::endl << *pKs << std::endl << std::endl;
		//std::cout << "Kss = " << std::endl << *pKss << std::endl << std::endl;

		// predictive mean
		// mu = ms + Ks' * inv(Kn) * (y-m)
		//    = ms + Ks' * alpha
		testData.pMu().reset(new Vector(M));
		testData.pMu()->noalias() = *(MeanFunc<Scalar>::ms(logHyp.mean, generalTrainingData, testData))
										  + (pKs->transpose()) * (*pAlpha);
		//std::cout << "InfExact::Mu" << std::endl;
		//std::cout << "Mu = " << std::endl << *pMu << std::endl << std::endl;

		// predictive variance
		// Sigma = Kss - Ks' * inv(Kn) * Ks
		//       = Kss - Ks' * inv(D^(1/2) * (D^(-1/2) * K * D^(-1/2) + I) * D^(1/2)) * Ks
		//       = Kss - Ks' * inv(D^(1/2) * LL' * D^(1/2)) * Ks
		//       = Kss - Ks' * D^(-1/2) * inv(L') * inv(L) * D^(-1/2) * Ks
		//       = Kss - (inv(L) * D^(-1/2) * Ks)' * (inv(L) * D^(-1/2) * Ks)
		//       = Kss - V' * V

		// V = inv(L) * D^(-1/2) * Ks
		//        (nxn)  *    (nxn)    * (nxm)
		Matrix V(NN, M); // nxm
		V = (*pL).matrixL().solve(pInvSqrtD->asDiagonal() * (*pKs));
		//std::cout << "InfExact::V" << std::endl;
		//V = m_L.getL().solve(m_pInvSqrtD->asDiagonal() * (*pKs));
		//std::cout << "V = " << std::endl << V << std::endl << std::endl;

		if(fVarianceVector)
		{
			// sigma2 = kss - v' * v
			testData.pSigma().reset(new Matrix(M, 1));					// variance vector (mx1)
			(*(testData.pSigma())).noalias() = (*pKss) - V.transpose().array().square().matrix().rowwise().sum();
		}
		else
		{
			// Sigma = Kss - V' *V
			testData.pSigma().reset(new Matrix(M, M));				// covariance matrix (mxm)
			(*(testData.pSigma())).noalias() = (*pKss) - V.transpose() * V;
		}
		//std::cout << "Sigma = " << std::endl << *pSigma << std::endl << std::endl;
		//std::cout << "InfExact::Sigma" << std::endl;
	}

	// nlZ, dnlZ
	template<typename GeneralTrainingData>
	void negativeLogMarginalLikelihood(const Hyp				&logHyp,
												  GeneralTrainingData<Scalar>	&generalTrainingData, 
												  Scalar					&nlZ, 
												  VectorPtr				&pDnlZ,
												  const int				calculationMode = 0)
	{
		// calculationMode
		// [0]: calculate both nlZ and pDnlZ
		// [+]: calculate nlZ only
		// [-]: calculate pDnlZ only

		// number of training data
		const int NN = generalTrainingData.NN();

		// some constants
		const VectorConstPtr pInvSqrtD = invSqrtD(logHyp.lik, generalTrainingData);
		CholeskyFactorConstPtr	pL;
		try
		{
			pL = choleskyFactor(logHyp.cov, generalTrainingData, pInvSqrtD);
		}
		catch(e) // if Kn is non positivie definite, nlZ = Inf, dnlZ = zeros
		{
			nlZ = std::numeric_limits<Scalar>::infinity();
			pDnlZ.reset(new Vector(logHyp.size()));
			pDnlZ.setZero();
			return;
		}
		const VectorConstPtr pY_M		= y_m(logHyp.mean, generalTrainingData);
		const MatrixConstPtr pAlpha	= alpha(pInvSqrtD, pL, pY_M);


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
			//std::cout << "nlZ" << std::endl;
			//std::cout << "meanLogHyp = " << std::endl << meanLogHyp << std::endl << std::endl;
			//std::cout << "covLogHyp = " << std::endl << covLogHyp << std::endl << std::endl;
			//std::cout << "likCovLogHyp = " << std::endl << likCovLogHyp << std::endl << std::endl;

			//Matrix L(m_L.matrixL());
			nlZ = static_cast<Scalar>(0.5f) * (*pY_M).dot(*pAlpha)
					//+ L.diagonal().array().log().sum()
					+ (*pL).matrixL().nestedExpression().diagonal().array().log().sum()
					- pInvSqrtD->array().log().sum()
					+ static_cast<Scalar>(NN) * static_cast<Scalar>(0.918938533204673f); // log(2pi)/2 = 0.918938533204673
			//nlZ = ((Scalar) 0.5f) * ((*m_pY_M).transpose() * (*m_pAlpha)).sum()
			//	  + log(det)
			//	  + ((Scalar) n) * ((Scalar) 0.918938533204673f); // log(2pi)/2 = 0.918938533204673
			//std::cout << "1 = " << std::endl << ((Scalar) 0.5f) * (*m_pY_M).dot(*m_pAlpha) << std::endl << std::endl;
			////std::cout << "2 = " << std::endl << L.diagonal().array().log().sum() << std::endl << std::endl;
			//std::cout << "2 = " << std::endl << m_L.matrixL().nestedExpression().diagonal().array().log().sum() << std::endl << std::endl;
			//std::cout << "3 = " << std::endl <<  - m_pInvSqrtD->array().log().sum() << std::endl << std::endl;
			//std::cout << "4 = " << std::endl << ((Scalar) n) * ((Scalar) 0.918938533204673f) << std::endl << std::endl;
		}

		// partial derivatives w.r.t hyperparameters
		if(calculationMode <= 0)
		{
			//std::cout << "dnlZ" << std::endl;

			// derivatives (f_j = partial f / partial x_j)
			int j = 0; // partial derivative index
			pDnlZ.reset(new Vector(logHyp.size()));

			// (1) w.r.t the mean parameters
			// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m)
			//       = - m' * inv(Kn) * y + (1/2) m' * inv(Kn) * m
			// nlZ_i = - m_i' * inv(Kn) * y + m_i' * inv(Kn) * m
			//          = - m_i' * inv(Kn) (y - m)
			//          = - m_i' * alpha
			for(int i = 0; i < logHyp.mean.size(); i++)
			{
				(*pDnlZ)(j++) = MeanFunc<Scalar>::m(logHyp.mean, generalTrainingData, i)->dot(*pAlpha);
				//std::cout << "DnlZ[ " << j-1 << " ] =  " << (*pDnlZ)(j-1) << std::endl;
			}

			// (2) w.r.t the cov parameters
			// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m) + (1/2) * log |Kn|
			// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * K_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * K_j]
			//          = (-1/2) * alpha' * K_j * alpha							+ (1/2) * tr[inv(Kn) * K_j]
			//          = (-1/2) * tr[(alpha' * alpha) * K_j]					+ (1/2) * tr[inv(Kn) * K_j]
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

			Matrix Q(NN, NN); // nxn
			Q.noalias() = pInvSqrtD->asDiagonal() * ((*pL).solve(Matrix(pInvSqrtD->asDiagonal())));
			Q -= (*pAlpha) * (pAlpha->transpose());
			for(int i = 0; i < logHyp.cov.size(); i++)
			{
				//(*pDnlZ)(j++) = ((Scalar) 0.5f) * (Q * (m_CovFunc(covLogHyp, i)->selfadjointView<Eigen::Upper>())).trace(); // [CAUTION] K: upper triangular matrix
				(*pDnlZ)(j++) = static_cast<Scalar>(0.5f) * Q.cwiseProduct(*(CovFunc<Scalar>::K(logHyp.cov, generalTrainingData, i))).sum();
				//std::cout << "DnlZ[ " << j-1 << " ] =  " << (*pDnlZ)(j-1) << std::endl;
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
				//(*pDnlZ)(j++) = (Scalar) 0.5f * (Q * (*m_LikFunc(likCovLogHyp, i))).trace();
				//(*pDnlZ)(j++) = ((Scalar) 0.5f) * Q.cwiseProduct(*(m_LikFunc(likCovLogHyp, i))).sum(); // [CAUTION] K: upper triangular matrix
				//(*pDnlZ)(j++) = ((Scalar) 0.5f) * (Q.array() * Matrix(m_LikFunc(likCovLogHyp, i)->asDiagonal()).array()).sum(); // [CAUTION] K: upper triangular matrix
				(*pDnlZ)(j++) = static_cast<Scalar>(0.5f) * Q.cwiseProduct(Matrix(LikFunc<Scalar>::lik(logHyp.lik, generalTrainingData, i)->asDiagonal())).sum(); // [CAUTION] K: upper triangular matrix
				//std::cout << "DnlZ[ " << j-1 << " ] =  " << (*pDnlZ)(j-1) << std::endl;
			}
		}
	}

protected:
	template<typename GeneralTrainingData>
	VectorPtr invSqrtD(const typename Hyp::LikHyp			&logHyp,
							 const GeneralTrainingData<Scalar>	&generalTrainingData)
	{
		// D = sW = sn2*I
		VectorPtr pInvSqrtD = LikFunc<Scalar>::lik(logHyp, generalTrainingData);

		// D^(-1/2)
		(*pInvSqrtD) = pInvSqrtD->cwiseSqrt().cwiseInverse();

		return pInvSqrtD;
	}

	template<typename GeneralTrainingData>
	CholeskyFactorPtr choleskyFactor /* throw (std::exception) */
											  (const typename Hyp::CovHyp				&logHyp,
												const GeneralTrainingData<Scalar>	&generalTrainingData,
												const VectorConstPtr						pInvSqrtD)
	{
		// number of training data
		const int NN = generalTrainingData.NN();

		// K
		MatrixPtr pKn = CovFunc<Scalar>::K(logHyp, generalTrainingData);

		// Kn = K + sn*I
		//    = K + D
		//    = D^(1/2) * (D^(-1/2)*K*D^(-1/2) + I) * D^(1/2)
		//    = D^(1/2) * (L*L') * D^(1/2)
		//
		// instead of						LL' = K + D
		// for numerical stability,	LL' = D^(-1/2)*K*D^(-1/2) + I 
		(*pKn) = pInvSqrtD->asDiagonal() * (*pKn) * pInvSqrtD->asDiagonal() + Matrix(NN, NN).setIdentity();

		// cholesky factor
		CholeskyFactorPtr pL(new CholeskyFactor());
		(*pL).compute(*pKn);	// compute the Cholesky decomposition of Kn

		if((*pL).info() != Eigen::ComputationInfo::Success)
		{
			std::exception e;
			switch((*pL).info())
			{
				case Eigen::ComputationInfo::NumericalIssue :
				{
					e = "NumericalIssue";
					break;
				}
				case Eigen::ComputationInfo::NoConvergence :
				{
					e = "NoConvergence";
					break;
				}
				case Eigen::ComputationInfo::InvalidInput :
				{
					e = "InvalidInput";
					break;
				}
				throw e;
			}
		}

		return pL;
	}

	template<typename GeneralTrainingData>
	VectorPtr y_m(const typename Hyp::MeanHyp			&logHyp,
					  const GeneralTrainingData<Scalar>	&generalTrainingData)
	{
		// number of training data
		const int NN = generalTrainingData.NN();

		// memory allocation
		VectorPtr pY_M  (new Vector(NN));

		// y - m
		(*pY_M).noalias() = (*generalTrainingData.pY()) - (*(MeanFunc<Scalar>::m(logHyp, generalTrainingData)));

		return pY_M;
	}

	template<typename GeneralTrainingData>
	VectorPtr alpha(const VectorConstPtr						pInvSqrtD,
						 const CholeskyFactorConstPtr				pL,
						 const VectorConstPtr						pY_M)
	{
		// memory allocation
		VectorPtr pAlpha(new Vector(pY_M.size()));

		// alpha = inv(Kn)*(y-m)
		// => (K + D) * alpha = y - m
		// =>  D^(1/2) * (D^(-1/2) * K * D^(-1/2) + I ) * D^(1/2) * alpha = y - m
		// =>  D^(1/2) * L * L' * D^(1/2) * alpha = y - m
		// => L * L' * D^(1/2) * alpha = D^(-1/2) * (y - m)
		// => D^(1/2) * alpha = L.solve(D^(-1/2) * (y - m))
		// => alpha = D^(-1/2) * L.solve(D^(-1/2) * (y - m))
		(*pAlpha).noalias() = pInvSqrtD->asDiagonal()
								  * ((*pL).solve(pInvSqrtD->asDiagonal() * (*pY_M)));

		return pAlpha;
	}
};
}

#endif