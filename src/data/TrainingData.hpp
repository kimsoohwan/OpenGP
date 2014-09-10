#ifndef _TRAINING_DATA_HPP_
#define _TRAINING_DATA_HPP_

#include "../util/macros.h"
#include "TestData.hpp"

namespace GP{

/**
 * @class		TrainingData
 * @brief		A training input and output data
 *					\f[
 * 				\{(\mathbf{X}, \mathbf{y}_i)\}_{i=1}^N, \quad \mathbf{X} \in \mathbb{R}^D, \; \mathbf{y} \in \mathbb{R}
 *					\f]
 * 				where \f$N\f$: number of training data, 
 * 						\f$D\f$: number of dimensions.\n\n
 *					It also holds the pair-wise squared distances, absolute distances and differences between the training inputs
 *					which will be repeatedly used in covariance functioins.					
 * @tparam		Scalar	Datatype such as float and double
 * @author		Soohwan Kim
 * @ingroup		-Data
 * @date			26/03/2014
 */
template<typename Scalar>
class TrainingData
{
// define matrix and vector types
protected:	TYPE_DEFINE_MATRIX(Scalar);
				TYPE_DEFINE_VECTOR(Scalar);

public:
	/**	
	 * @brief		Default constructor
	 * @details		Initialize calculation flags to be zero
	 */
	TrainingData() :
		m_fSqDistXX(false),
		m_fAbsDistXX(false),
	   m_fDeltaXXList(false)
	{
	}

	/**	
	 * @brief		Copy constructor
	 * @details		Copy const pointer only
	 */
	TrainingData(const TrainingData &other)
	{
		(*this) = other;
	}

	/**	
	 * @brief		Assignment operator
	 * @details		Copy const pointer only
	 */
	inline TrainingData& operator=(const TrainingData &other)
	{
		set(other.m_pX, other.m_pY);
		return *this;
	}


	/**	
	 * @brief	Gets the number of functional observations
	 * @return	The number of functional observations
	 */
	inline int N() const
	{
		if(!m_pX || !m_pY) return 0;
		//assert(m_pX->rows() == m_pY->size());
		return m_pX->rows();
	}

	/**	
	 * @brief	Gets the number of training data for the size of covariance matrix
	 * @note		Required to be consistent with DerivativeTrainingData
	 * @return	The number of training data
	 */
	inline int NN() const
	{
		return N();
	}

	/**
	 * @brief	Gets the number of dimensions
	 * @return	The number of dimensions
	 */
	inline int D() const
	{
		if(!m_pX) return 0;
		return m_pX->cols();
	}

	/**
	 * @brief	Resets the training data
	 * @param	[in] pX	The training inputs, \f$\mathbf{X} \in \mathbb{R}^{N \times D}\f$
	 * @param	[in] pY	The training outputs, \f$\mathbf{y} \in \mathbb{R}^{N}\f$
	 */
	void set(const MatrixConstPtr pX, const VectorConstPtr pY)
	{
		m_pX = pX;
		m_pY = pY;
		m_fSqDistXX			= false;
		m_fAbsDistXX		= false;
		m_fDeltaXXList		= false;
	}

	/**
	 * @brief	Gets the training inputs
	 * @return	A const matrix const pointer
	 */
	const MatrixConstPtr pX() const
	{
		return m_pX;
	}

	/**
	 * @brief	Gets the training outputs
	 * @return	A const vector const pointer
	 */
	const VectorConstPtr pY() const
	{
		return m_pY;
	}

	/**
	 * @brief	Gets the self squared distances between the training inputs
	 * @return	A const matrix const pointer
	 *				\f[
	 *				\mathbf{R^2} \in \mathbb{R}^{N \times N}, \quad
	 *				\mathbf{R^2}_{ij} = (\mathbf{x}_i - \mathbf{x}_j)^\text{T}(\mathbf{x}_i - \mathbf{x}_j)
	 *				\f]
	 */
	const MatrixConstPtr pSqDistXX()
	{
		assert(m_pX);

		// Calculate it only once.
		if(!m_fSqDistXX)
		{
			m_pSqDistXX = PairwiseOp<Scalar>::sqDist(m_pX);
			m_fSqDistXX = true;
		}

		return m_pSqDistXX;
	}

	/**
	 * @brief	Gets the self absolute distances between the training inputs
	 * @return	A const matrix const pointer
	 *				\f[
	 *				\mathbf{R} \in \mathbb{R}^{N \times N}, \quad
	 *				\mathbf{R}_{ij} = |\mathbf{x}_i - \mathbf{x}_j|
	 *				\f]
	 */
	const MatrixConstPtr pAbsDistXX()
	{
		// Calculate it only once.
		if(!m_fAbsDistXX)
		{
			m_pAbsDistXX.reset(new Matrix(*pSqDistXX()));
			m_pAbsDistXX->noalias() = m_pAbsDistXX->cwiseSqrt();	
			m_fAbsDistXX = true;
		}

		return m_pAbsDistXX;
	}

	/**
	 * @brief	Gets the self differences between the training inputs
	 * @param	[in] coord	Corresponding coordinate
	 * @return	An const matrix const pointer
	 *				\f[
	 *				\mathbf{D} \in \mathbb{R}^{N \times N}, \quad
	 *				\mathbf{D}_{ij} = \mathbf{x}_i^c - \mathbf{x}_j^c
	 *				\f]
	 */
	const MatrixConstPtr pDeltaXX(const int coord)
	{
		assert(m_pX);
		assert(coord >= 0 && coord < D());

		// Calculate it only once.
		if(!m_fDeltaXXList)
		{
			m_pDeltaXXList.resize(D());
			for(int d = 0; d < D(); i++)
				m_pDeltaXXList[d] = PairwiseOp<Scalar>::sqDelta(m_pX, d);
			m_fDeltaXXList = true;
		}

		return m_pDeltaXXList[coord];
	}

	/**
	 * @brief	Gets the cross squared distances between the training and test inputs
	 * @param	[in] pXs		The M test inputs
	 * @return	An matrix pointer
	 *				\f[
	 *				\mathbf{R^2} \in \mathbb{R}^{N \times M}, \quad
	 *				\mathbf{R^2}_{ij} = (\mathbf{x}_i - \mathbf{z}_j)^\text{T}(\mathbf{x}_i - \mathbf{z}_j)
	 *				\f]
	 * @todo		Include this matrix as a member variable like m_pSqDistXX
	 */
	MatrixPtr pSqDistXXs(const TestData<Scalar> &testData) const
	{
		assert(m_pX && testData.M() > 0);
		assert(D() == testData.D());
		return PairwiseOp<Scalar>::sqDist(m_pX, testData.pXs()); // NxM
	}

	/**
	 * @brief	Gets the cross absolute distances between the training and test inputs
	 * @param	[in] pXs		The M test inputs
	 * @return	An matrix pointer
	 *				\f[
	 *				\mathbf{R} \in \mathbb{R}^{N \times M}, \quad
	 *				\mathbf{R}_{ij} = |\mathbf{x}_i - \mathbf{z}_j|
	 *				\f]
	 * @todo		Include this matrix as a member variable like m_pDistXX
	 */
	MatrixPtr pAbsDistXXs(const TestData<Scalar> &testData) const
	{
		MatrixPtr pAbsDist = pSqDistXXs(testData); // NxM
		pAbsDist->noalias() = pAbsDist->cwiseSqrt();	
		return pAbsDist;
	}

	/**
	 * @brief	Gets the cross differences between the training and test inputs.
	 * @param	[in] pXs		The M test inputs
	 * @param	[in] coord	Corresponding coordinate
	 * @return	An matrix pointer
	 *				\f[
	 *				\mathbf{D} \in \mathbb{R}^{N \times M}, \quad
	 *				\mathbf{D}_{ij} = \mathbf{x}_i^c - \mathbf{z}_j^c
	 *				\f]
	 * @todo		Include this matrix as a member variable like m_pDeltaXXList
	 */
	MatrixPtr pDeltaXXs(const TestData<Scalar> &testData, const int coord) const
	{
		assert(m_pX && testData.M() > 0);
		assert(D() == testData.D());
		return PairwiseOp<Scalar>::delta(m_pX, testData.pXs(), coord); // NxM
	}

protected:
	/** @brief Training inputs: NxD matrix */
	MatrixConstPtr m_pX;

	/** @brief Training outputs: Nx1 vector */
	VectorConstPtr m_pY;

	/** @brief	Flag for the pre-calculated self squared distances between the training inputs */
	bool m_fSqDistXX;

	/** @brief	Pre-calculated self squared distances between the training inputs:  NxN matrix */
	MatrixConstPtr m_pSqDistXX;

	/** @brief	Flag for the pre-calculated self distances between the training inputs */
	bool m_fAbsDistXX;

	/** @brief	Pre-calculated self distances between the training inputs:  NxN matrix */
	MatrixPtr m_pAbsDistXX;

	/** @brief	Flag for the pre-calculated self differences between the training inputs */
	bool m_fDeltaXXList;

	/** @brief	Pre-calculated self differences between the training inputs: NxN matrices per each dimension*/
	std::vector<MatrixConstPtr> m_pDeltaXXList;
};

}
#endif