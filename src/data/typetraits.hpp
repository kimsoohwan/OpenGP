#ifndef _TYPE_TRAITS_HPP_
#define _TYPE_TRAITS_HPP_

// Boost
#include <boost/shared_ptr.hpp>

// Eigen
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#endif

namespace GP{

template<typename Scalar>
struct TypeTraits
{
	// Matrix and Vector
	typedef	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>		Matrix;
	typedef	Eigen::Matrix<Scalar, Eigen::Dynamic, 1>						Vector;
	typedef	Eigen::Matrix<Scalar, 1, Eigen::Dynamic>						RowVector;

	typedef	boost::shared_ptr<Matrix>											MatrixPtr;
	typedef	boost::shared_ptr<Vector>											VectorPtr;
	typedef	boost::shared_ptr<RowVector>										RowVectorPtr;

	typedef	boost::shared_ptr<const Matrix>									MatrixConstPtr;
	typedef	boost::shared_ptr<const Vector>									VectorConstPtr;
	typedef	boost::shared_ptr<const RowVector>								RowVectorConstPtr;

	// Hyperparameters
	typedef	Eigen::Matrix<Scalar, 0, 1> Hyp0;
	typedef	Eigen::Matrix<Scalar, 1, 1> Hyp1;
	typedef	Eigen::Matrix<Scalar, 2, 1> Hyp2;
};

typedef TypeTraits<float>::Matrix				MatrixXf;
typedef TypeTraits<float>::Vector				VectorXf;
typedef TypeTraits<float>::RowVector			RowVectorXf;

typedef TypeTraits<float>::MatrixPtr			MatrixXfPtr;
typedef TypeTraits<float>::VectorPtr			VectorXfPtr;
typedef TypeTraits<float>::RowVectorPtr		RowVectorXfPtr;

typedef TypeTraits<float>::MatrixConstPtr		MatrixXfConstPtr;
typedef TypeTraits<float>::VectorConstPtr		VectorXfConstPtr;
typedef TypeTraits<float>::RowVectorConstPtr	RowVectorXfConstPtr;
}

#endif