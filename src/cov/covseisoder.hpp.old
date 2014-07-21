#ifndef _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_DERIVATIVE_TRAINING_DATA_HPP_
#define _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_DERIVATIVE_TRAINING_DATA_HPP_

#include "covseiso.hpp"
#include "handlingderivativetrainingdata"

namespace GP{

/**
	* @class		CovSEIsoDer
	* @brief		Isotropic squared exponential covariance function
	* 				which handles derivative training data.
	* 				It inherits from CovSEIso to handle function training data
	* 				and from HandlingDerivativeTrainingData to handle derivative training data.
		// k(f, f) = sigma_f^2 exp(r), r = -1/(2*ell^2) * \Sum_{j=0}^D (x_k - x'_k)^2
		// 
		// dk/dr		= k
		// d2k/dr2	= k
		// d3k/dr3	= k
		// 
		// dr/dx_i			= -(x_i - x'_i)/ell^2
		// dr/dx'_j			=  (x_j - x'_j)/ell^2
		// dr/dlog(ell)	= -2r
		// 
		// d2r/dx_i,dx'_j			= d(i,j)/ell^2
		// d2r/dlog(ell),dx_i	=  2(x_i - x'_i)/ell^3
		// d2r/dlog(ell),dx'_j	= -2(x_j - x'_j)/ell^3
		// 
		// d3r/dell,dx_i,dx'_j = -2d(i,j)/ell^3
		//
		// dk/dsigma_f = 2*k/sigma_f
		
	* @author	Soohwan Kim
	* @date		02/04/2014
	*/
template<typename Scalar>
class CovSEIsoDer : public CovSEIso<Scalar>, public HandlingDerivativeTrainingData<CovSEIsoDer<Scalar> >
{
protected:
	inline
	MatrixPtr dk_dr(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData)
	{
		// dk/dr = k
		return K(logHyp, derivativeTrainingData.sqDist());
	}

	inline
	MatrixPtr d2k_dr2(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData)
	{
		// d2k/dr2 = k
		return K(logHyp, derivativeTrainingData.sqDist());
	}

	inline
	MatrixPtr d3k_dr3(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData)
	{
		// d3k/dr3 = k
		return K(logHyp, derivativeTrainingData.sqDist());
	}

	// K(f, f)
protected:
	static MatrixPtr K_FF(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const int pdHypIndex = -1)
	{
		return K(logHyp, derivativeTrainingData.sqDistXd(), pdHypIndex);
	}

	//K(f, df)
	static MatrixPtr K_FDf(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const int coord, const int pdHypIndex = -1)
	{
		assert(coord >= 0 && coord < derivativeTrainingData.D());

		// Output
		// K: NdxNd matrix
		MatrixPtr pK_FDf = K(logHyp, derivativeTrainingData.sqDistXd(), pdHypIndex);

		// Some constants
		const Scalar inv_ell2			= exp(static_cast<Scalar>(-2.0) * logHyp(0));	// 1/ell^2
		const Scalar twice_inv_ell2	= static_cast<Scalar>(2.0) * inv_ell2;				// 2/ell^2

		// [Delta]_ij = Xd_i(cood) - Xd_j(cood)
		const MatrixPtr pDelta = derivativeTrainingData.deltaXd(coord);

		// mode
		switch(pdHypIndex)
		{
		// pd[k(f, df')]/pd[log(ell)]: partial derivative  w.r.t log(ell) of covariance function between function and partial derivative.
		case 0:
			{
				// dk(f, df'/dx'_j)/dlog(ell) = ell * dk(f, df'/dx'_j)/dell
				// = ell * d[(dk/dr)*(dr/dx'_j)]/dell
				// = ell * [(d2k/dr2)*(dr/dell)*(dr/dx'_j) + (dk/dr)*(d2r/dell,dx'_j)]
				// = ell * [  (dk/dr)*(dr/dell)*(dr/dx'_j) + (dk/dr)*(d2r/dell,dx'_j)]
				// = ell * [          (dk/dell)*(dr/dx'_j) + (dk/dr)*(d2r/dell,dx'_j)]
				// = ell*(dk/dell)*(dr/dx'_j) + ell*(dk/dr)*(d2r/dell,dx'_j)]
				// = dk/dlog(ell)*(dr/dx'_j) + ell*(dk/dr)*(d2r/dell,dx'_j)]
				// = dk/dlog(ell)*(x_j - x'_j)/ell^2 + ell*k*(-2(x_j - x'_j)/ell^3)
				// = (x_j - x'_j)/ell^2 * [dk/dlog(ell) - 2k]
				(*pK_FDf) = inv_ell2 * pDelta->cwiseProduct(*K(logHyp, derivativeTrainingData.sqDistXd(), 0) - static_cast<Scalar>(-2.0)*(*pK_FDf));
				break;
			}

		// pd[K(f, df)]/pd[log(sigma_f)]: partial derivatives w.r.t log(sigma_f) of covariance function between function and partial derivative.
		case 1:
			{
				// dk(f, df'/dx'_j)/dlog(sigma_f) = sigma_f * dk(f, df'/dx'_j)/dsigma_f
				// = 2 * dk(f, df'/dx'_j)
				(*pK_FDf) = twice_inv_ell2 * pDelta->cwiseProduct(*pK_FDf);
				break;
			}

		// k(f, df') = covariance function between function and partial derivative.
		default:
			{
				// k(f, df'/dx'_j) = dk/dx'_j
				// = (dk/dr)*(dr/dx'_j)
				// = k*(x_j - x'_j)/ell^2
				(*pK_FDf) = inv_ell2 * pDelta->cwiseProduct(*pK_FDf);
				break;
			}
		}

		return pK_FDf;
	}

	//K(dfi, df'j)
	static MatrixPtr K_DfDf(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const int coordi, const int coordj, const int pdHypIndex = -1)
	{
		assert(coordi >= 0 && coordi < derivativeTrainingData.D());
		assert(coordj >= 0 && coordj < derivativeTrainingData.D());

		// Output
		// K: NdxNd matrix
		MatrixPtr K_DfDf = K(logHyp, derivativeTrainingData.sqDistXd(), pdHypIndex);

		// Some constants
		const Scalar inv_ell2				= exp(static_cast<Scalar>(-2.0) * logHyp(0));	// 1/ell^2
		const Scalar neg_twice_inv_ell2	= static_cast<Scalar>(-2.0) * inv_ell2;			// -2/ell^2
		const Scalar inv_ell4				= exp(static_cast<Scalar>(-4.0) * logHyp(0));	// 1/ell^4
		const Scalar neg_inv_ell4			= -inv_ell4;												// -1/ell^4
		const Scalar four_inv_ell4			= static_cast<Scalar>(4.0) * inv_ell4;				// 4/ell^2

		// dirac delta
		const Scalar dirac_delta = (coordi == coordj) ? static_cast<Scalar>(1.0) : static_cast<Scalar>(0.0);

		// mode
		switch(pdHypIndex)
		{
		// pd[k(df, df')]/pd[log(ell)]: partial derivative  w.r.t log(ell) of covariance function between partial derivatives.
		case 0:
			{
				// dk(df/dx_i, df'/dx'_j)/dlog(ell) = ell * dk(df/dx_i, df'/dx'_j)/dell
				// = ell * d3k/dx_i,dx'_j,dell
				// = ell * d[d[(dk/dr)*(dr/dx_i)]/dx'_j]/dell
				// = ell * d[(d2k/dr2)*(dr/dx_i)*(dr/dx'_j) + (dk/dr)*(d2r/dx_i,dx'_j)]/dell
				// = ell * [(d3k/dr3)*(dr/dell)*(dr/dx_i)*(dr/dx'_j)
				//        + (d2k/dr2)*(d2r/dell,dx_i)*(dr/dx'_j)
				//        + (d2k/dr2)*(dr/dx_i)*(d2r/dell,dx'_j)
				//        + (d2k/dr2)*(dr/dell)*(d2r/dx_i,dx'_j)
				//        + (dk/dr)*(d3r/dell,dx_i,dx'_j)]
				//        + 
				// = ell * k * [(-2r/ell) * (-(x_i - x'_i)/ell^2) * (x_j - x'_j)/ell^2
				//            +  2(x_i - x'_i)/ell^3 *     (x_j - x'_j)/ell^2
				//            + (-(x_i - x'_i)/ell^2) * (-2(x_j - x'_j)/ell^3)
				//            + (-2r/ell) * d(i,j)/ell^2
				//            + (-2(x_j - x'_j)/ell^3)]
				// = ell * [  (dk/dr)*(dr/dell)*(dr/dx'_j) + (dk/dr)*(d2r/dell,dx'_j)]
				// = ell * [          (dk/dell)*(dr/dx'_j) + (dk/dr)*(d2r/dell,dx'_j)]
				// = ell*(dk/dell)*(dr/dx'_j) + ell*(dk/dr)*(d2r/dell,dx'_j)]
				// = dk/dlog(ell)*(dr/dx'_j) + ell*(dk/dr)*(d2r/dell,dx'_j)]
				// = dk/dlog(ell)*(x_j - x'_j)/ell^2 + ell*k*(-2(x_j - x'_j)/ell^3)
				// = (x_j - x'_j)/ell^2 * [dk/dlog(ell) - 2k]
				(*pK_FDf) = inv_ell2 * pDelta->cwiseProduct(*K(logHyp, derivativeTrainingData.sqDistXd(), 0) - static_cast<Scalar>(-2.0)*(*pK_FDf));
				break;
			}

		// pd[K(f, df)]/pd[log(sigma_f)]: partial derivatives w.r.t log(sigma_f) of covariance function between function and partial derivative.
		case 1:
			{
				// dk(f, df'/dx'_j)/dlog(sigma_f) = sigma_f * dk(f, df'/dx'_j)/dsigma_f
				// = 2 * dk(f, df'/dx'_j)
				(*pK_FDf) = twice_inv_ell2 * pDelta->cwiseProduct(*pK_FDf);
				break;
			}

		// k(df, df') = covariance function between partial derivatives.
		default:
			{
				// k(df/dx_i, df'/dx'_j) = d2k/dx_i,dx'_j
				// = d[(dk/dr)*(dr/dx_i)]/dx'_j
				// = (d2k/dr2)*(dr/dx_i)*(dr/dx'_j) + (dk/dr)*(d2r/dx_i,dx'_j)
				// = k*[-(x_i - x'_i)/ell^2]*(x_j - x'_j)/ell^2 + k*d(i,j)/ell^2
				// = k*[d(i,j)/ell^2 - (x_i - x'_i)*(x_j - x'_j)/ell^4]
				if(coord1 == coord2) (*pK_FDf) = *K_DfDf->cwiseProduct(           neg_inv_ell4*(pDelta1->cwiseProduct(*pDelta2)));
				else						(*pK_FDf) = *K_DfDf->cwiseProduct(inv_ell2 + neg_inv_ell4*(pDelta1->cwiseProduct(*pDelta2)));
				break;
			}
		}

		return pK_FDf;
	}

	/**
	 * @brief	K*:NxM, Cross covariance matrix between the training data and test data.
	 * 			[K,    (K*)]: covariance matrix of the marginal Gaussian distribution 
	 * 			[(K*T), K**]
	 * 			Note that no pdHypIndex parameter is passed,
	 * 			because the partial derivatives of the covariance matrix
	 * 			is only required for learning hyperparameters.
	 * @param	[in] logHyp 			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] trainingData 	The training data.
	 * @param	[in] pXs 				The test inputs.
	 * @return	An NxM matrix pointer.
	 * 			N: the number of training data
	 * 			M: the number of test data
	 */
	static MatrixPtr Ks(const Hyp &logHyp, const TrainingData<Scalar> &trainingData, const MatrixConstPtr pXs)
	{
		// Calculate the cross covariance matrix
		// given the pairwise squared distances
		// between the training inputs and test inputs.
		return K(logHyp, trainingData.sqDist(pXs));
	}

	/**
	 * @brief	K**:MxM, Self [co]variance matrix between the test data.
	 * 			[K,    K*  ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, (K**)]
	 * 			Note that no pdHypIndex parameter is passed,
	 * 			because the partial derivatives of the covariance matrix
	 * 			is only required for learning hyperparameters.
	 * @param	[in] logHyp 			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pXs 				The test inputs.
	 * @param	[in] fVarianceVector	Flag for the return value.
	 * @return	fVarianceVector == true : An Mx1 matrix pointer.
	 * 			fVarianceVector == false: An MxM matrix pointer.
	 * 			M: the number of test data
	 */
	static MatrixPtr Kss(const Hyp &logHyp, const MatrixConstPtr pXs, const bool fVarianceVector = true)
	{
		// The number of test data.
		const int M = pXs->rows();

		// Some constant values.
		const Scalar sigma_f2 = exp(static_cast<Scalar>(2.0) * logHyp(1)); // sigma_f^2

		// Output.
		MatrixPtr pKss;

		// K: self-variance vector (Mx1).
		if(fVarianceVector)
		{
			// k(x, x') = sigma_f^2
			pKss.reset(new Matrix(M, 1));
			pKss->fill(sigma_f2);
		}
		// K: self-covariance matrix (MxM).
		else					
		{
			// Calculate the pairwise squared distances
			// between the test inputs.
			MatrixPtr pSqDist = PairwiseOp<Scalar>::sqDist(pXs);

			// Calculate the covariance matrix
			// given the pairwise squared distances.
			pKss = K(logHyp, pSqDist);
		}

		return pKss;
	}

	/**
	 * @brief	K:NxN, Self covariance matrix between the derivative training data.
	 * 			[(K), K* ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, K**]
	 * @param	[in] logHyp 						The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] derivativeTrainingData 	The derivative training data with respect to each dimension.
	 * @param	[in] pdHypIndex		(Optional) Hyperparameter index.
	 * 										It returns the partial derivatives of the covariance matrix
	 * 										with respect to this hyperparameter. 
	 * 										The partial derivatives are required for learning hyperparameters.
	 * 										(Example) pdHypIndex = 0: pd[K]/pd[log(ell)], pdHypIndex = 1: pd[K]/pd[log(sigma_f)]
	 * 										(Default = -1) K
	 * @return	An NxN matrix pointer.
	 * 			N: the number of training data
	 */
	static MatrixPtr K(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const int pdHypIndex = -1) 
	{
		// The pairwise squared distances between the trainig inputs
		// is already calculated in m_pSqDist when the training data was set.
		return K(logHyp, trainingData.sqDist(), pdHypIndex);
	}
protected:
	/**
	 * @brief	This is the core function of this class.
	 * 			Calculates the covariance matrix, given the pairwise squared distances.
	 * 			Designed to be used for calculating any covariance matrix, K, K*, K**.
	 * @param	[in] logHyp 	The log hyperparameters, log([ell, sigma_f])
	 * @param	[in] pSqDist 	The pairwise squared distances.
	 * @param	[in] pdHypIndex	(Optional) Hyperparameter index.
	 * 									It returns the partial derivatives of the covariance matrix
	 * 									with respect to this hyperparameter. 
	 * 									The partial derivatives are required for learning hyperparameters.
	 * 									(Example) pdHypIndex = 0: pd[K]/pd[log(ell)], pdHypIndex = 1: pd[K]/pd[log(sigma_f)]
	 * 									(Default = -1) K
	 * @return	An matrix pointer of the same size of the pairwise squared distance matrix.
	 */
	static MatrixPtr K(const Hyp &logHyp, const MatrixConstPtr pSqDist, const int pdHypIndex = -1)
	{
		// pdHypIndex should be greater than the number of hyperparameters
		assert(pdHypIndex < logHyp.size());

		// Output.
		// K: of the same size as the pairwise squared distance matrix
		// NxN for K, NxM for Ks
		MatrixPtr pK(new Matrix(pSqDist->rows(), pSqDist->cols()));

		// some constant values
		const Scalar inv_ell2					= exp(static_cast<Scalar>(-2.0) * logHyp(0));	// 1/ell^2
		const Scalar sigma_f2					= exp(static_cast<Scalar>( 2.0) * logHyp(1));	// sigma_f^2
		const Scalar neg_half_inv_ell2		= static_cast<Scalar>(-0.5) * inv_ell2;			// -1/(2*ell^2)
		const Scalar sigma_f2_inv_ell2		= sigma_f2 * inv_ell2;									// sigma_f^2/ell^2
		const Scalar twice_sigma_f2			= static_cast<Scalar>(2.0) * sigma_f2;				// 2*sigma_f^2

		// mode
		switch(pdHypIndex)
		{
		// pd[K]/pd[log(ell)]: derivatives of covariance matrix w.r.t log(ell).
		case 0:
			{
				//				 k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				// pd[k]/pd[log(ell)] = sigma_f^2 * exp(-r^2/(2*ell^2)) * (r^2/ell^3) * ell
				//					       = sigma_f^2 * exp(-r^2/(2*ell^2)) * (r^2/ell^2)
				pK->noalias() = sigma_f2_inv_ell2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp() * (*pSqDist).array()).matrix();
				break;
			}

		// pd[K]/pd[log(sigma_f)]: derivatives of covariance matrix w.r.t log(sigma_f).
		case 1:
			{
				//			        k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				// pd[k]/pd[log(sigma_f)] = 2 * sigma_f * exp(-r^2/(2*ell^2)) * sigma_f
				//								  = 2 * sigma_f^2 * exp(-r^2/(2*ell^2))
				pK->noalias() = twice_sigma_f2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp()).matrix();
				break;
			}

		// K: covariance matrix.
		default:
			{
				// k(x, x') = sigma_f^2 * exp(-r^2/(2*ell^2)), r = |x-x'|
				pK->noalias() = sigma_f2 * (neg_half_inv_ell2 * (*pSqDist)).array().exp().matrix();
				break;
			}
		}

		return pK;
	}

	static MatrixPtr K(const Hyp								&logHyp, 
							 TrainingData<Scalar>				&trainingData, 
							 DerivativeTrainingData<Scalar>	&derivativeTrainingData,
							 const int								pdHypIndex = -1) 

};


}

#endif