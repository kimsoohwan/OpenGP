#ifndef COVARIANCE_FUNCTION_WITH_DERIVATIVE_OBSERVATIONS_HPP_
#define COVARIANCE_FUNCTION_WITH_DERIVATIVE_OBSERVATIONS_HPP_

#include "../data/derivativetrainingdata.hpp"

namespace GP{

template<typename Scalar, template<> class Cov>
class CovDiff : Cov<Scalar>
{
public:
	static MatrixPtr K(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const int pdHypIndex = -1)
	{
		// input
		// pSqDist (nxn): squared distances
		// deltaList: list of delta (nxn)
		// d: dimension of training inputs
		// pLogHyp: log hyperparameters
		// pdIndex: partial derivatives with respect to this parameter index

		// output
		// K: nn by nn, nn = n + nd*d
		// 
		// for example, when d = 3
		//                  | f(x) | df(xd)/dx_1, df(xd)/dx_2, df(xd)/dx_3
		//                  |  n   |     nd            nd           nd
		// ---------------------------------------------------------------
		// f(x)        : n  |  FF  |     FD1,         FD2,         FD3
		// df(xd)/dx_1 : nd |   -  |    D1D1,        D1D2,        D1D3  
		// df(xd)/dx_2 : nd |   -  |      - ,        D2D2,        D2D3  
		// df(xd)/dx_3 : nd |   -  |      - ,          - ,        D3D3

		const int d		= derivativeTrainingData.D();
		const int nd	= derivativeTrainingData.Nd();
		const int n		= derivativeTrainingData.N();

		const int nn			= n + nd*d;
		const int numBlocks	= nd > 0 ? 1 + d : 1;

		// covariance matrix
		MatrixPtr pK(new Matrix(nn, nn)); // nn by nn, nn = n + nd*d

		// fill block matrices of FF, FD and DD in order
		for(int rowBlock = 0; rowBlock < numBlocks; rowBlock++)
		{
			// constants
			const int startRow	= rowBlock == 0 ? 0 : n + nd*(row_block-1);
			const int numRows		= rowBlock == 0 ? n : nd;

			for(int colBlock = rowBlock; colBlock < numBlocks; colBlock++)
			{
				// constants
				const int startCol	= colBlock == 0 ? 0 : n + nd*(col_block-1);
				const int numCols		= colBlock == 0 ? n : nd;

				// calculate the upper triangle
				if(rowBlock == 0)
				{
					// F-F
					if(colBlock == 0)	pK->block(startRow, startCol, numRows, numCols) = *(K(pLogHyp, static_cast<TrainingData<Scalar> >(derivativeTrainingData), pdIndex));

					// F-D
					else					pK->block(startRow, startCol, numRows, numCols) = *(K_FD(pLogHyp, derivativeTrainingData, colBlock-1, pdIndex));
				}
				else
				{
					// D-D
											pK->block(startRow, startCol, numRows, numCols) = *(K_DD(pLogHyp, derivativeTrainingData, rowBlock-1, colBlock-1, pdIndex));
				}

				// copy its transpose
				if(rowBlock != colBlock)	pK->block(startCol, startRow, numCols, numRows).noalias() = pK->block(startRow, startCol, numRows, numCols).transpose();
			}
		}

		return pK;
	}

	static MatrixPtr Ks(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const MatrixConstPtr pXs)
	{
		// output
		// K: nn x m, nn  = n  + nd*d
		// 
		// for example, when d = 3
		// K
		//             | f(z)
		// -------------------
		// f(x)        | FF
		// df(xd)/dx_1 | D1F
		// df(xd)/dx_2 | D2F
		// df(xd)/dx_3 | D3F

		const int n		= derivativeTrainingData.N();
		const int n		= derivativeTrainingData.N();

		// covariance matrix
		MatrixPtr pK(new Matrix(n*(d+1), m)); // n(d+1) x m

		// fill block matrices of FF, FD and DD in order
		for(int rowBlock = 0; rowBlock <= d; rowBlock++)
		{
			// F-F
			if(rowBlock == 0)		pK->block(n*rowBlock, 0, n, m) = *(K_FF(pSqDist, pLogHyp));

			// D-F
			else					pK->block(n*rowBlock, 0, n, m) = ((Scalar) -1.f) * (*(K_FD(pSqDist, deltaList[rowBlock-1], pLogHyp)));
		}

		return pK;
	}
};

}

#endif