#ifndef _TEST_DATA_HPP_
#define _TEST_DATA_HPP_

#include "../util/macros.h"

namespace GP{

/**
 * @class	TestData
 * @brief	A test data.
 * 			M: number of test data
 * 			D: number of dimensions
 * 			{X_i}_i=1^M, X \in R^D
 * @ingroup		-Data
 * @author	Soohwan Kim
 * @date		26/06/2014
 * @todo		DerivativeTestData to predict surface normals
 */
template<typename Scalar>
class TestData
{
// define matrix and vector types
protected:	TYPE_DEFINE_MATRIX(Scalar);
				TYPE_DEFINE_VECTOR(Scalar);

public:
	// default constructor
	TestData()
	{
	}

	// copy constructor for batch processing
	TestData(const TestData &other, const int startRow, const int n)
	{
		assert(other.M() > 0);
		assert(startRow <= other.M());
		assert(n > 0);
		if(other.M() > 0 && startRow <= other.M() && n > 0)
		{
			//m_pXs.reset(new Matrix(n, other.D()));
			//m_pXs->noalias() = other.m_pXs->middleRows(startRow, n);
			m_pXs.reset(new Matrix(other.m_pXs->middleRows(startRow, n)));
		}
	}

	/**	
	 * @brief	Gets the number of test data, M.
	 * @return	the number of test data
	 */
	inline int M() const
	{
		if(!m_pXs) return 0;
		return m_pXs->rows();
	}

	/**
	 * @brief	Gets the number of dimensions, D.
	 * @return	the number of dimensions.
	 */
	inline int D() const
	{
		if(!m_pXs) return 0;
		return m_pXs->cols();
	}

	/**
	 * @brief	Resets the test data.
	 * @param	[in] pXs	The test inputs.
	 */
	void set(const MatrixConstPtr pXs)
	{
		m_pXs = pXs;
	}

	/**
	 * @brief	Gets the pointer to the output means.
	 * @return	A matrix const pointer.
	 */
	VectorPtr& pMu()
	{
		return m_pMu;
	}

	/**
	 * @brief	Gets the pointer to the output (co)variances.
	 * @return	A matrix const pointer.
	 */
	MatrixPtr& pSigma()
	{
		return m_pSigma;
	}

	/**
	 * @brief	Gets the const pointer to the training inputs.
	 * @return	A matrix const pointer.
	 */
	const MatrixConstPtr pXs() const
	{
		return m_pXs;
	}

	///**
	// * @brief	Gets the const pointer to the training outputs.
	// * @return	A vector const pointer.
	// */
	//const VectorConstPtr pY() const
	//{
	//	return m_pYs;
	//}


protected:
	/** @brief function test inputs */
	MatrixConstPtr m_pXs;	// MxD matrix

	/** @brief derivative test inputs */
	//MatrixConstPtr m_pXds;	// MdxD matrix, TODO

	/** @brief test outputs */
	VectorPtr m_pMu;		// Mx1 vector
	MatrixPtr m_pSigma;	// Mx1 vector or MxM matrix
};

}
#endif