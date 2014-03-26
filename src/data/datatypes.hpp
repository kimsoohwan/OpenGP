#ifndef _DATA_TYPES_HPP_
#define _DATA_TYPES_HPP_

// Boost
#include <boost/shared_ptr.hpp>

// Eigen
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#endif

template<Scalar>
struct DataTypes
{
	// Matrix and Vector
	typedef	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>		Matrix;
	typedef	Eigen::Matrix<Scalar, Eigen::Dynamic, 1>						Vector;

	typedef	boost::shared_ptr<Matrix>											MatrixPtr;
	typedef	boost::shared_ptr<Vector>											VectorPtr;

	typedef	boost::shared_ptr<const Matrix>									MatrixConstPtr;
	typedef	boost::shared_ptr<const Vector>									VectorConstPtr;

	// Hyperparameters
	typedef	Eigen::Matrix<Scalar, 0, 1> Hyp0;
	typedef	Eigen::Matrix<Scalar, 1, 1> Hyp1;
	typedef	Eigen::Matrix<Scalar, 2, 1> Hyp2;
};

#endif