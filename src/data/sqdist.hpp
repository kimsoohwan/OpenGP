#ifndef _SQUARED_DISTANCES_HPP_
#define _SQUARED_DISTANCES_HPP_

#include <assert.h>

#include "datatypes.hpp"

namespace GP{
	// self
	MatrixPtr selfSqDistances(MatrixConstPtr pX)
	{
		// pX: nxd

		// number of training data
		const int n  = PointMatrixDirection::fRowWisePointsMatrix ? pX->rows()   : pX->cols();

		// squared distances: nxn
		MatrixPtr pSqDist(new Matrix(n, n));

		// initialization: diagonal is always zero
#if 0
		for(int row = 0; row < n; row++)
		{
			(*pSqDist)(row, row) = (Scalar) 0.f;
		}
#else
		(*pSqDist).setZero();
#endif

		// upper triangle
		if(PointMatrixDirection::fRowWisePointsMatrix)	
		{
			for(int row = 0; row < n; row++)
				for(int col = row + 1; col < n; col++)
					(*pSqDist)(row, col) = (pX->row(row) - pX->row(col)).array().square().sum();			//(*pSqDist)(row, col) = (pX->row(row) - pX->row(col)).squaredNorm();
		}
		else
		{
			for(int row = 0; row < n; row++)
				for(int col = row + 1; col < n; col++)
					(*pSqDist)(row, col) = (pX->col(row) - pX->col(col)).array().square().sum();
		}

		// lower triangle
#if 0
		for(int row = 0; row < n; row++)
		{
			for(int col = 0; col < row; col++)
			{
				(*pSqDist)(row, col) = (*pSqDist)(col, row);
			}
		}
#else
		pSqDist->triangularView<Eigen::StrictlyLower>() = pSqDist->transpose().eval().triangularView<Eigen::StrictlyLower>();
#endif

		return pSqDist;
	}

	// cross
	MatrixPtr crossSqDistances(MatrixConstPtr pX, MatrixConstPtr pXs)
	{
		// check if the dimensions are same
		assert(PointMatrixDirection::fRowWisePointsMatrix ? pX->cols()  == pXs->cols() : pX->rows() == pXs->rows());

		// X: nxd
		// Xs: mxd
		// number of data
		const int n  = PointMatrixDirection::fRowWisePointsMatrix ? pX->rows()   : pX->cols();
		const int m = PointMatrixDirection::fRowWisePointsMatrix ? pXs->rows() : pXs->cols();

		// squared distances: nxn
		MatrixPtr pSqDist(new Matrix(n, m));

		// dense
		if(PointMatrixDirection::fRowWisePointsMatrix)	
		{
			for(int row = 0; row < n; row++)
				for(int col = 0; col < m; col++)
					(*pSqDist)(row, col) = (pX->row(row) - pXs->row(col)).array().square().sum();
		}
		else
		{
			for(int row = 0; row < n; row++)
				for(int col = 0; col < m; col++)
					(*pSqDist)(row, col) = (pX->col(row) - pXs->col(col)).array().square().sum();
		}

		return pSqDist;
	}

	// self
	MatrixPtr selfDelta(MatrixConstPtr pX, const int i)
	{
		// [input]
		// X: mxd
		// i: coordinate index

		// [output]
		// x_i - x_i'

		assert(PointMatrixDirection::fRowWisePointsMatrix ? i < pX->cols() : i < pX->rows());

		// number of training data
		const int n  = PointMatrixDirection::fRowWisePointsMatrix ? pX->rows()   : pX->cols();

		// squared distances: nxn
		MatrixPtr pDelta(new Matrix(n, n));

		// initialization: diagonal is always zero
		(*pDelta).setZero();

		// upper triangle
		if(PointMatrixDirection::fRowWisePointsMatrix)	
		{
			for(int row = 0; row < n; row++)
				for(int col = row + 1; col < n; col++)
					(*pDelta)(row, col) = (*pX)(row, i) - (*pX)(col, i);
		}
		else
		{
			for(int row = 0; row < n; row++)
				for(int col = row + 1; col < n; col++)
					(*pDelta)(row, col) = (*pX)(i, row) - (*pX)(i, col);
		}

		// lower triangle (CAUTION: not symmetric but skew symmetric)
		pDelta->triangularView<Eigen::StrictlyLower>() = (((Scalar) -1.f) * (*pDelta)).transpose().eval().triangularView<Eigen::StrictlyLower>();

		return pDelta;
	}

	// cross
	MatrixPtr crossDelta(MatrixConstPtr pX, MatrixConstPtr pXs, const int i)
	{
		// [input]
		// X: nxd
		// Xs: mxd
		// i: coordinate index

		// [output]
		// x_i - x_i'

		// check if the dimensions are same
		assert(PointMatrixDirection::fRowWisePointsMatrix ? pX->cols()  == pXs->cols() : pX->rows() == pXs->rows());
		assert(PointMatrixDirection::fRowWisePointsMatrix ? i < pX->cols() : i < pX->rows());

		// X: nxd
		// Xs: mxd
		// number of data
		const int n  = PointMatrixDirection::fRowWisePointsMatrix ? pX->rows()   : pX->cols();
		const int m = PointMatrixDirection::fRowWisePointsMatrix ? pXs->rows() : pXs->cols();

		// squared distances: nxn
		MatrixPtr pDelta(new Matrix(n, m));

		// dense
		if(PointMatrixDirection::fRowWisePointsMatrix)	
		{
			for(int row = 0; row < n; row++)
				for(int col = 0; col < m; col++)
					(*pDelta)(row, col) = (*pX)(row, i) - (*pXs)(col, i);
		}
		else
		{
			for(int row = 0; row < n; row++)
				for(int col = 0; col < m; col++)
					(*pDelta)(row, col) = (*pX)(i, row) - (*pXs)(i, col);
		}

		return pDelta;
	}
}

#endif