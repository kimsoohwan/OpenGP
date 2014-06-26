#ifndef _TEST_DATA_HPP_
#define _TEST_DATA_HPP_

#include "../util/macros.hpp"

namespace GP{

/**
 * @class	TestData
 * @brief	A test data.
 * 			M: number of test data
 * 			D: number of dimensions
 * 			{X_i}_i=1^M, X \in R^D
 * @author	Soohwan Kim
 * @date		26/06/2014
 */
template<typename Scalar>
class TestData
{
// define matrix and vector types
protected:	TYPE_DEFINE_MATRIX(Scalar);
				TYPE_DEFINE_VECTOR(Scalar);

public:
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
	void set(MatrixPtr pXs)
	{
		m_pX = pXs;
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
	/** @brief test inputs */
	MatrixPtr m_pXs;	// MxD matrix

	/** @brief test outputs */
	VectorPtr m_pYs;	// Mx1 vector
};

}
#endif