#ifndef _COVARIANCE_FUNCTION_WITH_DERIVATIVE_OBSERVATIONS_HPP_
#define _COVARIANCE_FUNCTION_WITH_DERIVATIVE_OBSERVATIONS_HPP_

#include "../data/DerivativeTrainingData.hpp"
#include "../data/TestData.hpp"

namespace GP{

template<typename Scalar, template<typename> class Cov>
class CovDerObs : public Cov<Scalar>
{
// define matrix types
protected:	TYPE_DEFINE_MATRIX(Scalar);

public:
	static MatrixPtr K(const typename Cov<Scalar>::Hyp &logHyp, 
							 DerivativeTrainingData<Scalar> &derivativeTrainingData, 
							 const int pdHypIndex = -1)
	{
		// input
		// pSqDist (nxn): squared distances
		// deltaList: list of delta (nxn)
		// D: dimension of training inputs
		// pLogHyp: log hyperparameters
		// pdHypIndex: partial derivatives with respect to this parameter index

		// output
		// K: NN by NN, NN = N + Nd*D
		// 
		// for example, when D = 3
		//                  | f(x) | df(xd)/dx_1, df(xd)/dx_2, df(xd)/dx_3
		//                  |  N   |     Nd            Nd           Nd
		// ---------------------------------------------------------------
		// f(x)        : N  |  FF  |     FD1,         FD2,         FD3
		// df(xd)/dx_1 : Nd |   -  |    D1D1,        D1D2,        D1D3  
		// df(xd)/dx_2 : Nd |   -  |      - ,        D2D2,        D2D3  
		// df(xd)/dx_3 : Nd |   -  |      - ,          - ,        D3D3

		const int D		= derivativeTrainingData.D();
		const int N		= derivativeTrainingData.N();
		const int Nd	= derivativeTrainingData.Nd();
		const int NN	= derivativeTrainingData.NN();
		const int numBlocks	= Nd > 0 ? 1 + D : 1;

		// covariance matrix
		MatrixPtr pK(new Matrix(NN, NN)); // NN by NN, NN = N + Nd*D

		// fill block matrices of FF, FD and DD in order
		for(int rowBlock = 0; rowBlock < numBlocks; rowBlock++)
		{
			// constants
			const int startRow	= rowBlock == 0 ? 0 : N + Nd*(rowBlock-1);
			const int numRows		= rowBlock == 0 ? N : Nd;

			for(int colBlock = rowBlock; colBlock < numBlocks; colBlock++)
			{
				// constants
				const int startCol	= colBlock == 0 ? 0 : N + Nd*(colBlock-1);
				const int numCols		= colBlock == 0 ? N : Nd;

				// calculate the upper triangle
				if(rowBlock == 0)
				{
					// F-F
					if(colBlock == 0)	
						pK->block(startRow, startCol, numRows, numCols) 
						= *(CovParent::K(logHyp, static_cast<TrainingData<Scalar> >(derivativeTrainingData), pdHypIndex));

					// F-D
					else					
						pK->block(startRow, startCol, numRows, numCols) 
						= *(K_FD(logHyp, derivativeTrainingData, colBlock-1, pdHypIndex));
				}
				else
				{
					// D-D
						pK->block(startRow, startCol, numRows, numCols) 
						= *(K_DD(logHyp, derivativeTrainingData, rowBlock-1, colBlock-1, pdHypIndex));
				}

				// copy its transpose
				if(rowBlock != colBlock)	
						pK->block(startCol, startRow, numCols, numRows).noalias()
						= pK->block(startRow, startCol, numRows, numCols).transpose();
			}
		}

		return pK;
	}

	static MatrixPtr Ks(const typename Cov<Scalar>::Hyp &logHyp, 
							  DerivativeTrainingData<Scalar> &derivativeTrainingData, 
							  const TestData<Scalar> &testData)
	{
		// output
		// K: NN x M, NN  = N  + Nd*D
		// 
		// for example, when D = 3
		// K
		//             | f(z)
		// -------------------
		// f(x)        | FF
		// df(xd)/dx_1 | D1F
		// df(xd)/dx_2 | D2F
		// df(xd)/dx_3 | D3F

		const int D		= derivativeTrainingData.D();
		const int N		= derivativeTrainingData.N();
		const int Nd	= derivativeTrainingData.Nd();
		const int NN	= derivativeTrainingData.NN();
		const int M		= testData.M();

		const int numBlocks	= Nd > 0 ? 1 + D : 1;

		// covariance matrix
		MatrixPtr pK(new Matrix(NN, M)); // NN x M, NN = N + Nd*D

		// fill block matrices of FF, FD and DD in order
		// constants
		const int startCol	= 0;
		const int numCols		= M;
		for(int rowBlock = 0; rowBlock < numBlocks; rowBlock++)
		{
			// constants
			const int startRow	= rowBlock == 0 ? 0 : N + Nd*(rowBlock-1);
			const int numRows		= rowBlock == 0 ? N : Nd;

			// F-F
			if(rowBlock == 0)		
				pK->block(startRow, startCol, numRows, numCols)
				= *(CovParent::Ks(logHyp, static_cast<TrainingData<Scalar> >(derivativeTrainingData), testData));

			// D-F
			else						
				pK->block(startRow, startCol, numRows, numCols)
				= *(Ks_DF(logHyp, derivativeTrainingData, testData, rowBlock-1));
		}

		return pK;
	}
};

}

#endif