#ifndef _DERIVATIVE_TRAINING_DATA_HPP_
#define _DERIVATIVE_TRAINING_DATA_HPP_

#include <vector>

#include "TrainingData.hpp"

namespace GP{

/**
 * @class		DerivativeTrainingData
 * @brief		Functional and derivative training data.\n\n
 *					Note that it has two kinds of training inputs
 *					and single training outputs as follows.
 *					<CENTER>
 *					Training Data | Description
 *					--------------|-------------
 *					\f$\mathbf{X}   \in \mathbb{R}^{N \times D}\f$				| Training inputs of functional observations
 *					\f$\mathbf{X}_d \in \mathbb{R}^{N_d \times D}}\f$		| Training inputs of derivative observations
 *					\f$\mathbf{yy}_d   \in \mathbb{R}^{N + N_d \times D}\f$	| Training outputs of both functional and derivative observations
 *					</CENTER>
 *					or
 *					\f[
 *						\{\mathbf{X}, \mathbf{X}_d, \mathbf{yy}_d\} =
 * 					\{(\mathbf{X}, \mathbf{y})_i\}_{i=1}^N
 *						\cup
 *						\left\{\left(\mathbf{X}_d, \frac{dy}{d\mathbf{x}_1}, ..., \frac{dy}{d\mathbf{x}_D}\right)_i\right\}_{i=1}^{N_d},
 *					\f]
 * 				where\n
 *					\f$N\f$: number of function observations\n
 * 				\f$N_d\f$: number of derivative observations\n
 * 				\f$D\f$: number of dimensions\n\n
 *					This is for making it simple to retrieve the training outputs;
 *					if functional and derivative outputs are stored in separate variables,
 *					they need to be concatenated when retrieved from mean functions.\n\n
 *					Note that the order of training outputs is 
 *					\f{bmatrix}{
 *						\mathbf{y}\\
 *						\frac{dy}{d\mathbf{x}_1}\\
 *						\vdots\\
 *						\frac{dy}{d\mathbf{x}_D}\\
 *					\f}
 *					For example, suppose that you have a set of points, and
 * 				some part of them have surface normals. Then, the training inputs are
 *					devided into two parts, functional and derivative observations,
 *					while the traing outputs are combined into one.
 * 				This is why DerivativeTrainingData inherits from TrainingData
 *					but has its own derivative inputs.\n\n
 *					Also note that the functional inputs and derivative inputs can be the same.
 * @tparam		Scalar	Datatype such as float and double
 * @todo			Instead of having m_pXd, 
 *					how about a mask for the derivative inputs
 *					among the functional inputs
 * @ingroup		-Data
 * @author		Soohwan Kim
 * @date			01/04/2014
 */
template<typename Scalar>
class DerivativeTrainingData : public TrainingData<Scalar>
{
public:
	/**	
	 * @brief		Default constructor
	 * @details		Initialize calculation flags to be false
	 */
	DerivativeTrainingData() :
		m_fSqDistXdXd(false),
		m_fSqDistXXd(false),
		m_fAbsDistXdXd(false),
		m_fAbsDistXXd(false),
		m_fDeltaXdXdList(false),
	   m_fDeltaXXdList(false)
	{
	}

	/**	
	 * @brief		Copy constructor
	 * @details		Copy const pointer only
	 */
	DerivativeTrainingData(const DerivativeTrainingData &other)
	{
		(*this) = other;
	}

	/**	
	 * @brief		Assignment operator
	 * @details		Copy const pointer only
	 */
	inline DerivativeTrainingData& operator=(const DerivativeTrainingData &other)
	{
		set(other.m_pX, other.m_pXd, other.m_pY);
		return *this;
	}

	/**	
	 * @brief	Gets the number of derivative obervations
	 * @return	The number of derivative observations
	 */
	inline int Nd() const
	{
		//if(!m_pXd || !m_pYd) return 0;
		//assert(m_pXd->rows() == m_pYd->size());
		if(!m_pXd) return 0;
		return m_pXd->rows();
	}


	/**	
	 * @brief	Gets the number of training data for the size of covariance matrix
	 * @note		The number of training data, NN
	 *				is not the same as the number of functional observations, N.
	 * @return	The number of training data
	 */
	inline int NN() const
	{
		return N() + Nd()*D();
	}


	/**
	 * @brief	Gets the number of dimensions, D
	 * @note		Overloading TrainingData<Scalar>::D()
	 * @return	The number of dimensions
	 */
	inline int D() const
	{
		if(!m_pX || !m_pXd) return 0;
		assert(m_pX && m_pXd ? m_pX->cols() == m_pXd->cols() : true);
		if(m_pX)		return TrainingData<Scalar>::D();
		else			return m_pXd->cols();
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
	 * @brief	Resets the training data
	 * @param	[in] pX		The functional inputs, \f$\mathbf{X}   \in \mathbb{R}^{N \times D}\f$
	 * @param	[in] pXd		The derivative inputs, \f$\mathbf{X}_d \in \mathbb{R}^{N_d \times D}}\f$
	 * @param	[in] pYYd	The functional and derivative outputs, \f$\mathbf{yy}_d   \in \mathbb{R}^{N + N_d \times D}\f$
	 */
	void set(MatrixConstPtr pX, MatrixConstPtr pXd, VectorConstPtr pYYd)
	{
		TrainingData<Scalar>::set(pX, pYYd);
		m_pXd = pXd;
		m_fSqDistXdXd		= false;
		m_fSqDistXXd		= false;
		m_fAbsDistXdXd		= false;
		m_fAbsDistXXd		= false;
		m_fDeltaXdXdList	= false;
		m_fDeltaXXdList	= false;

		assert(NN() == pYYd->size());
	}

	/**
	 * @brief	Resets the training data when the functional and derivative inputs are the same
	 * @param	[in] pXXd	The function/derivative training inputs, \f$\mathbf{X} = \f$\mathbf{X}_d in \mathbb{R}^{N \times D}, \; N = N_d\f$
	 * @param	[in] pYYd	The function/derivative training outputs, \f$\mathbf{yy}_d \in \mathbb{R}^{N + N_d \times D}\f$
	 */
	void set(MatrixConstPtr pXXd, VectorConstPtr pYYd)
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
	 * 			self absolute distances between the derivative training inputs.
	 * @return	An NdxNd matrix const pointer.
	 */
	const MatrixConstPtr pAbsDistXdXd()
	{
		// Calculate it only once.
		if(!m_fAbsDistXdXd)
		{
			m_pAbsDistXdXd.reset(new Matrix(*pSqDistXdXd()));
			m_pAbsDistXdXd->noalias() = m_pAbsDistXdXd->cwiseSqrt();	
			m_fAbsDistXdXd = true;
		}

		return m_pAbsDistXdXd;
	}

	/**
	 * @brief	Gets the const pointer to the pre-calculated
	 * 			cross absolute distances between the derivative training inputs and training inputs.
	 * @return	An NxNd matrix const pointer.
	 */
	const MatrixConstPtr pAbsDistXXd()
	{
		// Calculate it only once.
		if(!m_fAbsDistXXd)
		{
			m_pAbsDistXXd.reset(new Matrix(*pSqDistXXd()));
			m_pAbsDistXXd->noalias() = m_pAbsDistXXd->cwiseSqrt();	
			m_fAbsDistXXd = true;
		}

		return m_pAbsDistXXd;
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
	 * @brief	Gets the pointer to the cross absolute distances
	 * 			between the derivative training inputs and test inputs.
	 * @param	[in] pXs		The test inputs. A MxD matrix.
	 * @return	An NdxM matrix const pointer.
	 */
	MatrixPtr pAbsDistXdXs(const TestData<Scalar> &testData) const
	{
		MatrixPtr pAbsDist = pSqDistXdXs(testData); // NdxM
		pAbsDist->noalias() = pAbsDist->cwiseSqrt();	
		return pAbsDist;
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
	MatrixConstPtr m_pXd;	// NdxD matrix

	///** @brief Training function/derivative outputs */
	//VectorPtr m_pYd;	// Ndx1 vector

	///** @brief Derivative training outputs with respect to each dimension */
	//std::vector<VectorPtr>	m_pdYList;		// Ndx1 vector per each dimension

	/** @brief	Flag for the pre-calculated self squared distances between the derivative training inputs. */
	bool m_fSqDistXdXd;

	/** @brief	Pre-calculated self squared distances between the derivative training inputs. */
	MatrixConstPtr m_pSqDistXdXd;	// NdxNd matrix

	/** @brief	Flag for the pre-calculated cross squared distances between the function training inputs and derivative training inputs. */
	bool m_fSqDistXXd;

	/** @brief	Pre-calculated cross squared distances between the derivative training inputs and function training inputs. */
	MatrixConstPtr m_pSqDistXXd;	// NxNd matrix
	//MatrixConstPtr m_pSqDistXdX;	// NdxN matrix

	/** @brief	Flag for the pre-calculated self absolute distances between the derivative training inputs. */
	bool m_fAbsDistXdXd;

	/** @brief	Pre-calculated self absolute distances between the derivative training inputs. */
	MatrixPtr m_pAbsDistXdXd;	// NdxNd matrix

	/** @brief	Flag for the pre-calculated cross absolute distances between the function training inputs and derivative training inputs. */
	bool m_fAbsDistXXd;

	/** @brief	Pre-calculated cross absolute distances between the derivative training inputs and function training inputs. */
	MatrixPtr m_pAbsDistXXd;	// NxNd matrix
	//MatrixConstPtr m_pAbsDistXdX;	// NdxN matrix

	/** @brief	Flag for the pre-calculated self differences between the derivative training inputs. */
	bool m_fDeltaXdXdList;

	/** @brief	Pre-calculated self differences between the derivative training inputs. */
	std::vector<MatrixConstPtr> m_pDeltaXdXdList;	// NdxNd matrix per each dimension

	/** @brief	Flag for the pre-calculated cross differences between the function training inputs and derivativetraining inputs. */
	bool m_fDeltaXXdList;

	/** @brief	Pre-calculated cross differences between the derivative training inputs and training inputs. */
	std::vector<MatrixConstPtr> m_pDeltaXXdList;	// NxNd matrix per each dimension
};

}
#endif