#ifndef _TRAINING_DATA_HPP_
#define _TRAINING_DATA_HPP_

#include "../util/macros.hpp"
#include "TestData.hpp"

namespace GP{

/**
 * @class	TrainingData
 * @brief	A training data.
 * 			N: number of training data
 * 			D: number of dimensions
 * 			{(X, y)_i}_i=1^N, X \in R^D, y \in R
 * @author	Soohwan Kim
 * @date		26/03/2014
 */
template<typename Scalar>
class TrainingData
{
// define matrix and vector types
protected:	TYPE_DEFINE_MATRIX(Scalar);
				TYPE_DEFINE_VECTOR(Scalar);

public:
	/**	
	 * @brief	Default constructor.
	 * 			Initialize calculation flags to be zero
	 */
	TrainingData() :
		m_fSqDist(false),
	   m_fDeltaList(false)
	{
	}

	/**	
	 * @brief	Gets the number of training data, N.
	 * @return	the number of training data.
	 */
	inline int N() const
	{
		if(!m_pX || !m_pY) return 0;
		assert(m_pX->rows() == m_pY->size());
		return m_pX->rows();
	}

	/**
	 * @brief	Gets the number of dimensions, D.
	 * @return	the number of dimensions.
	 */
	inline int D() const
	{
		if(!m_pX) return 0;
		return m_pX->cols();
	}

	/**
	 * @brief	Resets the training data.
	 * @param	[in] pX	The training inputs.
	 * @param	[in] pY	The training outputs.
	 */
	void set(MatrixPtr pX, VectorPtr pY)
	{
		m_pX = pX;
		m_pY = pY;
		m_fSqDist = false;
		m_fDeltaList = false;
	}

	///**
	// * @brief	Gets the const pointer to the training inputs.
	// * @return	A matrix const pointer.
	// */
	//const MatrixConstPtr pX() const
	//{
	//	return m_pX;
	//}

	///**
	// * @brief	Gets the const pointer to the training outputs.
	// * @return	A vector const pointer.
	// */
	//const VectorConstPtr pY() const
	//{
	//	return m_pY;
	//}

	/**
	 * @brief	Gets the const pointer to the pre-calculated
	 * 			self squared distances between the training inputs.
	 * @return	An NxN matrix const pointer.
	 */
	const MatrixConstPtr pSqDistXX()
	{
		assert(m_pX);

		// Calculate it only once.
		if(!m_fSqDist)
		{
			m_pSqDist = PairwiseOp<Scalar>::sqDist(m_pX);
			m_fSqDist = true;
		}

		return m_pSqDist;
	}

	/**
	 * @brief	Gets the const pointer to the pre-calculated
	 * 			self differences between the training inputs.
	 * @param	[in] coord	Corresponding coordinate. [result]_ij = Xi_coord - Xj_coord
	 * @return	An NxN matrix const pointer.
	 */
	const MatrixConstPtr pDeltaXX(const int coord)
	{
		assert(m_pX);
		assert(coord >= 0 && coord < D());

		// Calculate it only once.
		if(!m_fDeltaList)
		{
			m_pDeltaList.resize(D());
			for(int d = 0; d < D(); i++)
				m_pDeltaList[d] = PairwiseOp<Scalar>::sqDelta(m_pX, d);
			m_fDeltaList = true;
		}

		return m_pDeltaList[coord];
	}

	/**
	 * @brief	Gets the pointer to the cross squared distances
	 * 			between the training inputs and test inputs.
	 * @param	[in] pXs		The test inputs. A MxD matrix.
	 * @return	An NxM matrix const pointer.
	 */
	const MatrixPtr pSqDistXXs(const TestData<Scalar> &testData) const
	{
		assert(m_pX && testData.M() > 0);
		assert(D() == testData.D());
		return PairwiseOp<Scalar>::sqDist(m_pX, testData.pXs()); // NxM
	}

	/**
	 * @brief	Gets the pointer to the cross differences
	 * 			between the training inputs and test inputs.
	 * @param	[in] pXs		The test inputs. A MxD matrix.
	 * @param	[in] coord	Corresponding coordinate. [result]_ij = Xi_coord - Xsj_coord
	 * @return	An NxM matrix const pointer.
	 */
	const MatrixPtr pDeltaXXs(const TestData<Scalar> &testData, const int coord) const
	{
		assert(m_pX && testData.M() > 0);
		assert(D() == testData.D());
		return PairwiseOp<Scalar>::delta(m_pX, testData.pXs(), coord); // NxM
	}

protected:
	/** @brief training inputs */
	MatrixPtr m_pX;	// NxD matrix

	/** @brief training outputs */
	VectorPtr m_pY;	// Nx1 vector


	/** @brief	Pre-calculated self squared distances between the training inputs. */
	MatrixConstPtr m_pSqDist;	// NxN matrix

	/** @brief	Pre-calculated self differences between the training inputs. */
	std::vector<MatrixConstPtr> m_pDeltaList;	// NxN matrix per each dimension


	/** @brief	Flag for the pre-calculated self squared distances between the training inputs. */
	bool m_fSqDist;

	/** @brief	Flag for the pre-calculated self differences between the training inputs. */
	bool m_fDeltaList;
};

}
#endif