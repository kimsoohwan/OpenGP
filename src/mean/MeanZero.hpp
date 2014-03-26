#ifndef MEAN_FUNCTION_ZERO_HPP
#define MEAN_FUNCTION_ZERO_HPP

#include "GP.h"

namespace GP{

	template<typename Scalar>
	class MeanZero
	{
	public:
		// Data Types
		typedef	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>		Matrix;
		typedef	Eigen::Matrix<Scalar, Eigen::Dynamic, 1>						Vector;
		typedef	boost::shared_ptr<Matrix>											MatrixPtr;
		typedef	boost::shared_ptr<Vector>											VectorPtr;
		typedef	boost::shared_ptr<const Matrix>									MatrixConstPtr;
		typedef	boost::shared_ptr<const Vector>									VectorConstPtr;

		// Hyperparameters
		typedef	Eigen::Matrix<Scalar, 0, 1> Hyp;

		// mean and derivatives
		VectorPtr operator()(const Hyp &logHyp, const int pdIndex = -1) const
		{
			// number of training data
			const int n = getN();
			VectorPtr mu(new Vector(n));
			mu->setZero();
			return mu;
		}

		// Ms
		VectorPtr Ms(MatrixConstPtr pXs, const Hyp &logHyp) const
		{
			// number of training data
			const int m = PointMatrixDirection::fRowWisePointsMatrix ? pXs->rows() : pXs->cols();
			VectorPtr mu(new Vector(m));
			mu->setZero();
			return mu;
		}
	};
}

#endif MEAN_FUNCTION_ZERO_HPP