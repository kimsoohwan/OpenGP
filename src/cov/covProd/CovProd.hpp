#ifndef _COVARIANCE_FUNCTION_PRODUCT_HPP_
#define _COVARIANCE_FUNCTION_PRODUCT_HPP_

namespace GP{

/**
 * @class		CovProd
 * @brief		Sparse covariance function with isotropic distances
 * @ingroup		-Cov
 * @author		Soohwan Kim
 * @date			26/03/2014
 */
template<typename Scalar, 
			template <typename> class _Cov1,
			template <typename> class _Cov2>
class CovProd
{
protected:
	typedef _Cov1<Scalar>	Cov1;
	typedef _Cov2<Scalar>	Cov2;

/**@brief Number of hyperparameters */
public: static const int N = Cov1::N + Cov2::N;

/**@brief Define Matrix, MatrixPtr, MatrixConstPtr */
protected:	TYPE_DEFINE_MATRIX(Scalar);

/**@brief Define Hyp */
public:		TYPE_DEFINE_HYP(Scalar, N);

public:
	template <template<typename> class GeneralTrainingData>
	static MatrixPtr K(const Hyp							&logHyp, 
							 GeneralTrainingData<Scalar>	&generalTrainingData, 
							 const int							pdHypIndex = -1) 
	{
		// Assertions only in the begining of the public static member functions which can be accessed outside.
		// The hyparparameter index should be less than the number of hyperparameters
		assert(pdHypIndex < logHyp.size());

		// copy hyperparameters
		Cov1::Hyp	logHyp1;
		Cov2::Hyp	logHyp2;
		copy(logHyp, logHyp1, logHyp2);

		// output
		MatrixPtr pK;

		// covariance matrix
		if(pdHypIndex < 0)
		{
			// Cov = Cov1 * Cov2
			pK = Cov1::K(logHyp1, generalTrainingData, pdHypIndex);											// Cov1
			pK->noalias() = pK->cwiseProduct(*Cov2::K(logHyp2, generalTrainingData, pdHypIndex));	// Cov2
		}

		// partial derivatives of covariance matrix 
		else
		{
			// dCov1
			if(pdHypIndex < Cov1::N)
			{
				// dCov = dCov1*Cov2
				pK = Cov1::K(logHyp1, generalTrainingData, pdHypIndex);									// dCov1
				pK->noalias() = pK->cwiseProduct(*Cov2::K(logHyp2, generalTrainingData, -1));		// Cov2
			}
			// dCov2
			else
			{
				// dCov = Cov1*dCov2
				pK = Cov1::K(logHyp1, generalTrainingData, -1);																		// Cov1
				pK->noalias() = pK->cwiseProduct(*Cov2::K(logHyp2, generalTrainingData, pdHypIndex - Cov1::N));		// dCov2
			}
		}
		
		return pK;
	}

	template <template<typename> class GeneralTrainingData>
	static MatrixPtr Ks(const Hyp										&logHyp, 
							  const GeneralTrainingData<Scalar>		&generalTrainingData, 
							  const TestData<Scalar>					&testData)
	{
		// copy hyperparameters
		Cov1::Hyp	logHyp1;
		Cov2::Hyp	logHyp2;
		copy(logHyp, logHyp1, logHyp2);

		// Cov = Cov1 * Cov2
		MatrixPtr pKs = Cov1::Ks(logHyp1, generalTrainingData, testData);							// Cov1
		pKs->noalias() = pKs->cwiseProduct(*Cov2::Ks(logHyp2, generalTrainingData, testData));	// Cov2

		return pKs;
	}

	static MatrixPtr Kss(const Hyp						&logHyp, 
								const TestData<Scalar>		&testData, 
								const bool						fVarianceVector = true)
	{
		// copy hyperparameters
		Cov1::Hyp	logHyp1;
		Cov2::Hyp	logHyp2;
		copy(logHyp, logHyp1, logHyp2);

		// Cov = Cov1 * Cov2
		MatrixPtr pKss = Cov1::Kss(logHyp1, testData, fVarianceVector);								// Cov1
		pKss->noalias() = pKss->cwiseProduct(*Cov2::Kss(logHyp2, testData, fVarianceVector));	// Cov2

		return pKss;
	}

protected:
	static void copy(const Hyp					&logHyp,
						  typename Cov1::Hyp		&logHyp1,
						  typename Cov2::Hyp		&logHyp2)
	{
		int j = 0;
		for(int i = 0; i < logHyp1.size(); i++)	logHyp1(i) = logHyp(j++);
		for(int i = 0; i < logHyp2.size(); i++)	logHyp2(i) = logHyp(j++);
	}

};

}

#endif