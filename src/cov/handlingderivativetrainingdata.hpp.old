#ifndef _HANDLING_DERIVATIVE_TRAINING_DATA_HPP_
#define _HANDLING_DERIVATIVE_TRAINING_DATA_HPP_

#include "../data/trainingdata.hpp"
#include "../data/derivativetrainingdata.hpp"

namespace GP{

/**
 * @class	HandlingDerivativeTrainingData
 * @brief	Enables a covariance function to handle derivative training data.
 * 			It assumes that the covariance function class, Cov has K_FF(K), K_FD, and K_DD,
 * 			and provide the common way to combine them for derivative training data.
 * @note		The curiously recurring template pattern (CRTP) is used
 * 			because the same procedure is applied for every covariance function,
 * 			while how to calculate K, Ks, and Kss varies covariance to covariance.
 * @author	Soohwan Kim
 * @date	1/04/2014
 */
template<template<typename Scalar> class Cov>
class HandlingDerivativeTrainingData : public TypeTraits<Scalar>
{
public:
	/**
	 * @brief	K: Self covariance matrix between the training data (function derivatives and function values).
	 * 			[(K), K* ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, K**]
	 * @param	[in] logHyp 						The log hyperparameters.
	 * @param	[in] derivativeTrainingData 	The derivative training data.
	 * @param	[in] pdHypIndex		(Optional) Hyperparameter index.
	 * 										It returns the partial derivatives of the covariance matrix
	 * 										with respect to this hyperparameter. 
	 * 										The partial derivatives are required for learning hyperparameters.
	 * 										(Example) pdHypIndex = 0: pd[K]/pd[log(hyp(0))], pdHypIndex = 1: pd[K]/pd[log(hyp(1))]
	 * 										(Default = -1) K
	 * @return	An (Nd*(D+1)+N)x(Nd*(D+1)+N) matrix pointer.
	 * 			Nd: the number of derivative training data.
	 * 			N: the number of function training data.
	 * 			D: the number of dimensions.
	 */
	static MatrixPtr K(const Cov<Scalar>::Hyp				&logHyp, 
							 DerivativeTrainingData<Scalar>	&derivativeTrainingData,
							 const int								pdHypIndex = -1) 
	{
		assert(pdHypIndex < logHyp.size());

		// Output
		// K: (Nd*(D+1)+N) x (Nd*(D+1)+N)
		// 
		// for example, when D = 3
		//                 | F (Nd) | Df1 (Nd) | Df2 (Nd) | Df3 (Nd) | F0 (N) |
		// K = -------------------------------------------------------------
		//        F   (Nd) |     FF,     FDf1,       FDf2,      FDf3 |  FF0
		//        Df1 (Nd) |      -,   Df1Df1,     Df1Df2,    Df1Df3 |  Df1F0
		//        Df2 (Nd) |      -,        -,     Df2Df2,    Df2Df3 |  Df2F0
		//        Df3 (Nd) |      -,        -,          -,    Df3Df3 |  Df3F0
		//       -----------------------------------------------------------
		//        F0  (N)  |      -,        -,          -,         - |  F0F0=K
		       
		// The number of function training data.
		const int N = derivativeTrainingData.N();

		// The number of derivative training data.
		const int Nd = derivativeTrainingData.Nd();

		// The number of dimensions
		const int D = derivativeTrainingData.D();

		// The covariance matrix
		MatrixPtr pK(new Matrix(Nd*(D+1) + N, Nd*(D+1) + N)); // K: (Nd*(D+1)+N) x (Nd*(D+1)+N)

		// Fill the block matrices of FF, FD and DD in order.
		if(Nd > 0)
		{
			for(int row = 0; row <= D; row++)
			{
				const int startingRow = Nd*row;
				for(int col = row; col <= D; col++)
				{
					const int startingCol = Nd*col;

					// Calculate the upper triangle.
					if(row == 0)
					{
						// F1F1
						if(col == 0)	pK->block(startingRow, startingCol, Nd, Nd) = *(K(logHyp, static_cast<TrainingData<Scalar> >(derivativeTrainingData), pdIndex));

						// F1D*
						else				pK->block(startingRow, startingCol, Nd, Nd) = *(K_FD(logHyp, derivativeTrainingData, col-1, pdIndex));
					}
					else
					{
						// D*D*
											pK->block(startingRow, startingCol, Nd, Nd) = *(K_DD(logHyp, derivativeTrainingData, row-1, col-1, pdIndex));
					}

					// copy its transpose
					if(row != col)	pK->block(startingCol, startingRow, Nd, Nd) = pK->block(startingRow, startingCol, Nd, Nd).transpose();
				}

				if(N > 0)
				{
					const int startingCol = Nd*(D+1);

					// F1F2
					if(row == 0)		pK->block(startingRow, startingCol, Nd, N) = *(K(logHyp, static_cast<TrainingData<Scalar> >(derivativeTrainingData), trainingData, pdIndex));

					// D*F2
					else					pK->block(startingRow, startingCol, Nd, N) = *(K_DF(logHyp, derivativeTrainingData, trainingData, row-1, pdIndex));

					// copy its transpose
					pK->block(startingCol, startingRow, N, Nd) = pK->block(startingRow, startingCol, Nd, N).transpose();
				}
			}
		}

		if(N > 0)
		{
			const int startingRow = Nd*(D+1);

			// F2F2
			pK->block(startingRow, startingRow, N, N) = *(K_FF(logHyp, trainingData, pdIndex));
		}

		return pK;
	}

	/**
	 * @brief	K:NxN, Self covariance matrix between the derivative training data.
	 * 			[(K), K* ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, K**]
	 * @param	[in] logHyp 						The log hyperparameters.
	 * @param	[in] derivativeTrainingData 	The derivative training data with respect to each dimension.
	 * @param	[in] pdHypIndex		(Optional) Hyperparameter index.
	 * 										It returns the partial derivatives of the covariance matrix
	 * 										with respect to this hyperparameter. 
	 * 										The partial derivatives are required for learning hyperparameters.
	 * 										(Example) pdHypIndex = 0: pd[K]/pd[log(hyp(0))], pdHypIndex = 1: pd[K]/pd[log(hyp(1))]
	 * 										(Default = -1) K
	 * @return	An Nd*(D+1)xNd*(D+1) matrix pointer.
	 * 			Nd: the number of derivative training data.
	 * 			D: the number of dimensions.
	 */
	static MatrixPtr K(const Cov<Scalar>::Hyp				&logHyp, 
							 DerivativeTrainingData<Scalar>	&derivativeTrainingData,
							 const int								pdHypIndex = -1) 
	{
		return K(logHyp, derivativeTrainingData, TrainingData<Scalar>(), pdHypIndex);
	}

	/**
	 * @brief	K*:NxM, Cross covariance matrix 
	 * 			        between the training data (function derivatives and function values)
	 * 			        and the test data.
	 * 			[K,    (K*)]: covariance matrix of the marginal Gaussian distribution 
	 * 			[(K*T), K**]
	 * 			Note that no pdHypIndex parameter is passed,
	 * 			because the partial derivatives of the covariance matrix
	 * 			is only required for learning hyperparameters.
	 * @param	[in] logHyp 						The log hyperparameters.
	 * @param	[in] derivativeTrainingData 	The derivative training data with respect to each dimension.
	 * @param	[in] trainingData 				The training data.
	 * @param	[in] pXs 							The test inputs.
	 * @return	An (Nd*(D+1)+N) x M matrix pointer.
	 * 			Nd: the number of derivative training data
	 * 			N: the number of function training data
	 * 			D: the number of dimensions
	 * 			M: the number of test inputs.
	 */
	static MatrixPtr Ks(const Cov<Scalar>::Hyp						&logHyp, 
							  const DerivativeTrainingData<Scalar>		&derivativeTrainingData,
							  const TrainingData<Scalar>					&trainingData,
							  const MatrixConstPtr							pXs)
	{
		// Output
		// Ks: (Nd*(D+1)+N) x M

		//                |  F (M)  |
		// Ks = ---------------------
		//        F1 (Nd) |    F1F
		//        D1 (Nd) |    D1F
		//        D2 (Nd) |    D2F
		//        D3 (Nd) |    D3F
		//     ---------------------   
		//        F2 (N)  |    F2F=Ks

		// The number of derivative training data.
		const int Nd = derivativeTrainingData.N();

		// The number of dimensions
		const int D = derivativeTrainingData.D();

		// The number of function training data.
		const int N = trainingData.N();

		// The number of test positions.
		const int M = pXs->rows();

		// covariance matrix
		MatrixPtr pKs(new Matrix(Nd*(D+1) + N, M)); // Ks: (Nd*(D+1) + N) x M

		if(Nd > 0)
		{
			// F1F
			pKs->block(0, 0, Nd, M) = *(K_FF(logHyp, static_cast<TrainingData<Scalar> >(derivativeTrainingData), pXs));

			// D1F, D2F, D3F
			for(int row = 1; row <= D; row++)
				pKs->block(Nd*row, 0, Nd, M) = *(K_DF(logHyp, derivativeTrainingData, pXs, row-1));
		}

		if(N > 0)
		{
			// F2F
			pKs->block(Nd*(D+1), 0, N, M) = *(K_FF(logHyp, trainingData, pXs));
		}

		return pKs;
	}

	/**
	 * @brief	K*:NxM, Cross covariance matrix 
	 * 			        between the training data (function derivatives and function values)
	 * 			        and the test data.
	 * 			[K,    (K*)]: covariance matrix of the marginal Gaussian distribution 
	 * 			[(K*T), K**]
	 * 			Note that no pdHypIndex parameter is passed,
	 * 			because the partial derivatives of the covariance matrix
	 * 			is only required for learning hyperparameters.
	 * @param	[in] logHyp 						The log hyperparameters.
	 * @param	[in] derivativeTrainingData 	The derivative training data with respect to each dimension.
	 * @param	[in] pXs 							The test inputs.
	 * @return	An (Nd*(D+1)+N) x M matrix pointer.
	 * 			Nd: the number of derivative training data
	 * 			D: the number of dimensions
	 * 			M: the number of test inputs.
	 */
	static MatrixPtr Ks(const Cov<Scalar>::Hyp						&logHyp, 
							  const DerivativeTrainingData<Scalar>		&derivativeTrainingData,
							  const MatrixConstPtr							pXs)
	{
		return Ks(logHyp, derivativeTrainingData, TrainingData<Scalar>(), pXs);
	}

};

}

#endif