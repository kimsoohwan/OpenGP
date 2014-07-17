#ifndef _CUSTOM_MACROS_H_
#define _CUSTOM_MACROS_H_

// Boost
#include <boost/shared_ptr.hpp>

// Eigen
#define EIGEN_NO_DEBUG		// to speed up
//#define EIGEN_USE_MKL_ALL	// to use Intel Math Kernel Library
// TODO: check if Intel is correctly intalled and running.
#include <Eigen/Dense>

namespace GP{

/** @brief	Define matrix and its shared pointer types */
//#define TYPE_DEFINE_MATRIX(Scalar)		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>	Matrix;\
//													typedef boost::shared_ptr<Matrix>										MatrixPtr;\
//													typedef boost::shared_ptr<const Matrix>								MatrixConstPtr;

#define TYPE_DEFINE_MATRIX(Scalar)																\
	/** @brief Matrix type */																		\
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>	Matrix;			\
	/** @brief Matrix shared pointer type */													\
	typedef boost::shared_ptr<Matrix>										MatrixPtr;		\
	/** @brief Matrix const shared pointer type */											\
	typedef boost::shared_ptr<const Matrix>								MatrixConstPtr;

/** @brief	Define vector and its shared pointer types */
#define TYPE_DEFINE_VECTOR(Scalar)		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1>					Vector;\
													typedef boost::shared_ptr<Vector>										VectorPtr;\
													typedef boost::shared_ptr<const Vector>								VectorConstPtr;


/** @brief	Define row vector and its shared pointer types */
#define TYPE_DEFINE_ROW_VECTOR(Scalar)	typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic>					RowVector;\
													typedef boost::shared_ptr<RowVector>									RowVectorPtr;\
													typedef boost::shared_ptr<const RowVector>							RowVectorConstPtr;

#define TYPE_DEFINE_HYP(Scalar, N)					\
	/** @brief	Hyperparameter type */				\
	typedef Eigen::Matrix<Scalar, N, 1>		Hyp;

#define TYPE_DEFINE_ALL_HYP(Scalar, MeanFunc, CovFunc, LikFunc)	\
	/** @brief	All hyperparameter type */									\
	typedef Hyp<Scalar, MeanFunc, CovFunc, LikFunc>		Hyp;

/** @brief	Define cholesky factor type and its shared pointer types */
#define TYPE_DEFINE_CHOLESKYFACTOR(Scalar)		typedef Eigen::LLT<Matrix>											CholeskyFactor;\
																typedef boost::shared_ptr<CholeskyFactor>						CholeskyFactorPtr;\
																typedef boost::shared_ptr<const CholeskyFactor>				CholeskyFactorConstPtr;

//typedef	Eigen::ConjugateGradient<Matrix, Eigen::Upper>					CholeskyFactor;
//typedef	Eigen::ConjugateGradient<SparseMatrix, Eigen::Upper>			SparseCholeskyFactor;
//typedef	Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>				SparseCholeskyFactor;
//typedef	boost::shared_ptr<SparseCholeskyFactor>							SparseCholeskyFactorPtr;
//typedef	boost::shared_ptr<const SparseCholeskyFactor>					SparseCholeskyFactorConstPtr;
}

#endif