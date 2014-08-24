#ifndef _LLT_MKL_HPP_
#define _LLT_MKL_HPP_

namespace GP{

/** @class	LLT_MKL
  * @brief	Wrapper of LLT for using Intel MKL
  * @note	Refer to Eigen::LLT
  */
template<typename _MatrixType, int _UpLo = Eigen::Lower>
class LLT_MKL : public Eigen::LLT<_MatrixType, _UpLo>
{
public:
	LLT_MKL(const MatrixType &A)
      : Eigen::LLT<_MatrixType, _UpLo>()
	{
		compute(A);
	}

	/**@ brief Cholesky decomposition */
	Eigen::LLT<_MatrixType, _UpLo>& compute(const MatrixType& A)
	{
#if defined(EIGEN_USE_LAPACKE) || defined(EIGEN_USE_LAPACKE_STRICT)
		// resize
		eigen_assert(A.rows() == A.cols());
		const Index size = A.rows();
		m_matrix.resize(size, size);
		m_matrix = A;
		m_isInitialized = true;

		// set up parameters for ?potrf
		const int StorageOrder = MatrixType::Flags & Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor;
		const int matrix_order = (StorageOrder==Eigen::RowMajor) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

		// lhs
		const lapack_int n = m_matrix.rows();
		const char uplo = (_UpLo == Eigen::Lower)	? 'L' : 'U';
		MatrixType::Scalar* a = &(m_matrix.coeffRef(0,0));
		const lapack_int lda = m_matrix.outerStride();

		// potrf
		const lapack_int info = potrf<MatrixType::Scalar>(matrix_order, uplo, n, a, lda);
		m_info = (info==0) ? Eigen::Success : Eigen::NumericalIssue;

		return static_cast<Eigen::LLT<_MatrixType, _UpLo> >(*this);
#else
		return Eigen::LLT<_MatrixType, _UpLo>::compute(A);
#endif
	}

protected:

#if defined(EIGEN_USE_LAPACKE) || defined(EIGEN_USE_LAPACKE_STRICT)
	template <typename T>
	inline lapack_int potrf(int matrix_order, char uplo, lapack_int n, T* a, lapack_int lda);

	template <>
	inline lapack_int potrf<float>(int matrix_order, char uplo, lapack_int n, float* a, lapack_int lda)
	{
		return LAPACKE_spotrf(matrix_order, uplo, n, a, lda);
	}

	template <>
	inline lapack_int potrf<double>(int matrix_order, char uplo, lapack_int n, double* a, lapack_int lda)
	{
		return LAPACKE_dpotrf(matrix_order, uplo, n, a, lda);
	}
#endif
};

}
#endif