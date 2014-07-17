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
	 * 			Initialize calculation flags to be false
	 */
	DerivativeTrainingData() :
		m_fSqDistXdXd(false),
		m_fSqDistXXd(false),
		m_fDeltaXdXdList(false),
	   m_fDeltaXXdList(false)
	{
	}

	/**	
	 * @brief	Gets the number of derivative training data, Nd.
	 * @return	The number of derivative training data.
	 */
	inline int Nd() const
	{
		//if(!m_pXd || !m_pYd) return 0;
		//assert(m_pXd->rows() == m_pYd->size());
		if(!m_pXd) return 0;
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
		else			return m_pXd->cols();
	}

	inline int NN() const
	{
		return N() + Nd()*D();
	}

	///**
	// * @brief	Resets the training inputs.
	// * @param	[in] pXd		The training inputs.
	// * @param	[in] pYd		The training outputs.
	// */
	//void set(MatrixPtr pXd, VectorPtr pYd, std::vector<VectorPtr> pdYList)
	//{
	//	m_pXd			= pXd;
	//	m_pYd			= pYd;
	//	m_pdYList	= pdYList;
	//	m_fSqDistXdXd			= false;
	//	m_fSqDistXXd		= false;
	//	m_fDeltaXdXdList		= false;
	//	m_fDeltaXXdList	= false;
	//}

	/**
	 * @brief	Resets the training inputs.
	 * @param	[in] pX		The function training inputs.
	 * @param	[in] pXd		The derivative training inputs.
	 * @param	[in] pYYd	The function/derivative training outputs.
	 */
	void set(MatrixPtr pX, MatrixPtr pXd, VectorPtr pYYd)
	{
		TrainingData<Scalar>::set(pX, pYYd);
		m_pXd = pXd;
		m_fSqDistXdXd		= false;
		m_fSqDistXXd		= false;
		m_fDeltaXdXdList	= false;
		m_fDeltaXXdList	= false;
	}

	/**
	 * @brief	Resets the training inputs.
	 * @param	[in] pXXd	The function/derivative training inputs.
	 * @param	[in] pYYd	The function/derivative training outputs.
	 */
	void set(MatrixPtr pXXd, VectorPtr pYYd)
	{
		set(pXXd, pXXd, pYYd);
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
		if(!m_fSqDistXdXd)
		{
			m_pSqDistXdXd = PairwiseOp<Scalar>::sqDist(m_pXd);
			m_fSqDistXdXd = true;
		}

		return m_pSqDistXdXd;
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
		if(!m_fDeltaXdXdList)
		{
			m_pDeltaXdXdList.resize(D());
			for(int d = 0; d < D(); d++)
				m_pDeltaXdXdList[d] = PairwiseOp<Scalar>::delta(m_pXd, d);
			m_fDeltaXdXdList = true;
		}

		return m_pDeltaXdXdList[coord];
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
		if(!m_fDeltaXXdList)
		{
			m_pDeltaXXdList.resize(D());
			for(int d = 0; d < D(); d++)
				m_pDeltaXXdList[d] = PairwiseOp<Scalar>::delta(m_pX, m_pXd, d);
			m_fDeltaXXdList = true;
		}

		return m_pDeltaXXdList[coord];
	}

	/**
	 * @brief	Gets the pointer to the cross squared distances
	 * 			between the derivative training inputs and test inputs.
	 * @param	[in] pXs		The test inputs. A MxD matrix.
	 * @return	An NdxM matrix const pointer.
	 */
	MatrixPtr pSqDistXdXs(const TestData<Scalar> &testData) const
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
	MatrixPtr pDeltaXdXs(const TestData<Scalar> &testData, const int coord) const
	{
		assert(m_pXd && testData.M() > 0);
		assert(D() == testData.D());
		return PairwiseOp<Scalar>::delta(m_pXd, testData.pXs(), coord); // NdxM
	}

protected:
	/** @brief Training derivative inputs */
	MatrixPtr m_pXd;	// NdxD matrix

	///** @brief Training function/derivative outputs */
	//VectorPtr m_pYd;	// Ndx1 vector

	///** @brief Derivative training outputs with respect to each dimension */
	//std::vector<VectorPtr>	m_pdYList;		// Ndx1 vector per each dimension


	/** @brief	Pre-calculated self squared distances between the derivative training inputs. */
	MatrixConstPtr m_pSqDistXdXd;	// NdxNd matrix

	/** @brief	Pre-calculated cross quared distances between the derivative training inputs and function training inputs. */
	MatrixConstPtr m_pSqDistXXd;	// NxNd matrix
	//MatrixConstPtr m_pSqDistXdX;	// NdxN matrix

	/** @brief	Pre-calculated self differences between the derivative training inputs. */
	std::vector<MatrixConstPtr> m_pDeltaXdXdList;	// NdxNd matrix per each dimension

	/** @brief	Pre-calculated cross differences between the derivative training inputs and training inputs. */
	std::vector<MatrixConstPtr> m_pDeltaXXdList;	// NxNd matrix per each dimension


	/** @brief	Flag for the pre-calculated self squared distances between the derivative training inputs. */
	bool m_fSqDistXdXd;

	/** @brief	Flag for the pre-calculated cross quared distances between the function training inputs and derivative training inputs. */
	bool m_fSqDistXXd;

	/** @brief	Flag for the pre-calculated self differences between the derivative training inputs. */
	bool m_fDeltaXdXdList;

	/** @brief	Flag for the pre-calculated cross differences between the function training inputs and derivativetraining inputs. */
	bool m_fDeltaXXdList;
};

}
#endif