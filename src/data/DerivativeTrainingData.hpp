#ifndef _DERIVATIVE_TRAINING_DATA_HPP_
#define _DERIVATIVE_TRAINING_DATA_HPP_

#include <vector>

#include "TrainingData.hpp"

namespace GP{

/**
 * @class	DerivativeTrainingData
 * @brief	A training data which have two kinds of traing outputs,
 * 			function values and function derivatives w.r.t each dimension
 * 			at the same training inputs
 * 			as well as the training data from TrainingData.
 * 			N: number of function value training data.
 * 			Nd: number of derivative value training data.
 * 			D: number of dimensions.
 * 			{(X, y, dy/dx1, ..., dy/dxD)_j}_i=1^Nd, X \in R^D, y \in R.
 * 			{(X, y)_i}_i=1^N, X \in R^D, y \in R from TrainingData.
 * @note		For example, suppose that you have a set of points and
 * 			some part of them have surface normals. Then, you can divide them
 * 			into a set of points which have function values (0) and surface normals
 * 			and another set of points which have function values (0) only.
 * 			This is why DerivativeTrainingData inherits from TrainingData.
 * @author	Soohwan Kim
 * @date		01/04/2014
 */
template<typename Scalar>
class DerivativeTrainingData : public TrainingData<Scalar>
{
public:
	/**	
	 * @brief	Default constructor.
	 * 			Initialize calculation flags to be zero
	 */
	DerivativeTrainingData() :
		m_fSqDistXd(false),
		m_fSqDistXXd(false),
		m_fDeltaListXd(false),
	   m_fDeltaListXXd(false)
	{
	}

	/**	
	 * @brief	Gets the number of derivative training data, Nd.
	 * @return	The number of derivative training data.
	 */
	inline int Nd() const
	{
		if(!m_pXd || !m_pYd) return 0;
		assert(m_pXd->rows() == m_pYd->size());
		return m_pXd->rows();
	}

	/**
	 * @brief	Gets the number of dimensions, D.
	 * @note		Overloading the function D().
	 * @return	The number of dimensions.
	 */
	inline int D() const
	{
		if(!m_pX || !m_pXd) return 0;
		assert(m_pX && m_pXd ? m_pX->cols() == m_pXd->cols() : true);
		if(m_pX)		return TrainingData<Scalar>::D();
		if(m_pXd)	return m_pXd->cols();
	}

	/**
	 * @brief	Resets the training inputs.
	 * @param	[in] pXd		The training inputs.
	 * @param	[in] pYd		The training outputs.
	 */
	void set(MatrixPtr pXd, VectorPtr pYd, std::vector<VectorPtr> pdYList)
	{
		m_pXd			= pXd;
		m_pYd			= pYd;
		m_pdYList	= pdYList;
		m_fSqDistXd			= false;
		m_fSqDistXXd		= false;
		m_fDeltaListXd		= false;
		m_fDeltaListXXd	= false;
	}

	/**
	 * @brief	Gets the const pointer to the pre-calculated
	 * 			self squared distances between the derivative training inputs.
	 * @return	An NdxNd matrix const pointer.
	 */
	const MatrixConstPtr pSqDistXdXd()
	{
		assert(m_pXd);

		// Calculate it only once.
		if(!m_fSqDistXd)
		{
			m_pSqDistXd = PairwiseOp<Scalar>::sqDist(m_pXd);
			m_fSqDistXd = true;
		}

		return m_pSqDistXd;
	}

	/**
	 * @brief	Gets the const pointer to the pre-calculated
	 * 			cross squared distances between the derivative training inputs and training inputs.
	 * @return	An NxNd matrix const pointer.
	 */
	const MatrixConstPtr pSqDistXXd()
	{
		assert(m_pX && m_pXd);

		// Calculate it only once.
		if(!m_fSqDistXXd)
		{
			m_pSqDistXXd = PairwiseOp<Scalar>::sqDist(m_pX, m_pXd);
			m_fSqDistXXd = true;
		}

		return m_pSqDistXXd;
	}

	/**
	 * @brief	Gets the const pointer to the pre-calculated
	 * 			self differences between the derivative training inputs.
	 * @param	[in] coord	Corresponding coordinate. [result]_ij = Xdi_coord - Xdj_coord
	 * @return	An NdxNd matrix const pointer.
	 */
	const MatrixConstPtr pDeltaXdXd(const int coord)
	{
		assert(m_pXd);
		assert(coord >= 0 && coord < D());

		// Calculate it only once.
		if(!m_fDeltaListXd)
		{
			m_pDeltaListXd.resize(D());
			for(int d = 0; d < D(); d++)
				m_pDeltaListXd[d] = PairwiseOp<Scalar>::delta(m_pXd, d);
			m_fDeltaListXd = true;
		}

		return m_pDeltaListXd[coord];
	}

	/**
	 * @brief	Gets the const pointer to the pre-calculated
	 * 			cross differences between the derivative training inputs and training inputs.
	 * @param	[in] coord	Corresponding coordinate. [result]_ij = Xj_coord - Xdi_coord
	 * @return	An NxNd matrix const pointer.
	 */
	const MatrixConstPtr pDeltaXXd(const int coord)
	{
		assert(m_pX && m_pXd);
		assert(coord >= 0 && coord < D());

		// Calculate it only once.
		if(!m_fDeltaListXXd)
		{
			m_pDeltaListXXd.resize(D());
			for(int d = 0; d < D(); d++)
				m_pDeltaListXXd[d] = PairwiseOp<Scalar>::delta(m_pX, m_pXd, d);
			m_fDeltaListXXd = true;
		}

		return m_pDeltaListXXd[coord];
	}

	/**
	 * @brief	Gets the pointer to the cross squared distances
	 * 			between the derivative training inputs and test inputs.
	 * @param	[in] pXs		The test inputs. A MxD matrix.
	 * @return	An NdxM matrix const pointer.
	 */
	const MatrixPtr pSqDistXdXs(const TestData<Scalar> &testData) const
	{
		assert(m_pXd && testData.M() > 0);
		assert(D() == testData.D());
		return PairwiseOp<Scalar>::sqDist(m_pXd, testData.pXs()); // NdxM
	}

	/**
	 * @brief	Gets the pointer to the cross differences
	 * 			between the derivative training inputs and test inputs.
	 * @param	[in] pXs		The test inputs. A MxD matrix.
	 * @param	[in] coord	Corresponding coordinate. [result]_ij = Xdi_coord - Xsj_coord
	 * @return	An NdxM matrix const pointer.
	 */
	const MatrixPtr pDeltaXdXs(const TestData<Scalar> &testData, const int coord) const
	{
		assert(m_pXd && testData.M() > 0);
		assert(D() == testData.D());
		return PairwiseOp<Scalar>::delta(m_pXd, testData.pXs(), coord); // NdxM
	}

protected:
	/** @brief Training inputs */
	MatrixPtr m_pXd;	// NdxD matrix

	/** @brief Training outputs */
	VectorPtr m_pYd;	// Ndx1 vector

	/** @brief Derivative training outputs with respect to each dimension */
	std::vector<VectorPtr>	m_pdYList;		// Ndx1 vector per each dimension


	/** @brief	Pre-calculated self squared distances between the derivative training inputs. */
	MatrixConstPtr m_pSqDistXd;	// NdxNd matrix

	/** @brief	Pre-calculated cross quared distances between the derivative training inputs and function training inputs. */
	//MatrixConstPtr m_pSqDistXXd;	// NdxN matrix
	MatrixConstPtr m_pSqDistXXd;	// NxNd matrix

	/** @brief	Pre-calculated self differences between the derivative training inputs. */
	std::vector<MatrixConstPtr> m_pDeltaXdList;	// NdxNd matrix per each dimension

	/** @brief	Pre-calculated cross differences between the derivative training inputs and training inputs. */
	std::vector<MatrixConstPtr> m_pDeltaXXdList;	// NxNd matrix per each dimension


	/** @brief	Flag for the pre-calculated self squared distances between the derivative training inputs. */
	bool m_fSqDistXd;

	/** @brief	Flag for the pre-calculated cross quared distances between the function training inputs and derivative training inputs. */
	bool m_fSqDistXXd;

	/** @brief	Flag for the pre-calculated self differences between the derivative training inputs. */
	bool m_fDeltaListXd;

	/** @brief	Flag for the pre-calculated cross differences between the function training inputs and derivativetraining inputs. */
	bool m_fDeltaListXXd;
};

}
#endif