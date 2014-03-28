#ifndef _TRAINING_DATA_HPP_
#define _TRAINING_DATA_HPP_

#include "typetraits.hpp"

namespace GP{

/**
 * @class	TrainingData
 * @brief	A training data.
 * 			N: number of training data
 * 			D: number of dimensions
 * 			{(X, y)_i}_i=1^N, X \in R^D, y \in R
 * @author	Soohwankim
 * @date		26/03/2014
 */
template<typename Scalar>
class TrainingData : public TypeTraits<Scalar>
{
public:
	// Boost shared pointers
	typedef	boost::shared_ptr<TrainingData<Scalar> >			Ptr;
	typedef	boost::shared_ptr<const TrainingData<Scalar> >	ConstPtr;

	/**	
	 * @brief	Gets the number of training data, N.
	 * @return	the number of training data.
	 */
	inline int N() const
	{
		assert(m_X.rows() == m_y.size());
		return m_X.rows();
	}

	/**
	 * @brief	Gets the number of dimensions, D.
	 * @return	the number of dimensions.
	 */
	inline int D() const
	{
		return m_X.cols();
	}

	/**
	 * @brief	Gets the const pointer to the training inputs.
	 * @return	A matrix const pointer.
	 */
	const MatrixConstPtr pX() const
	{
		return MatrixConstPtr(&m_X);
	}

protected:
	Matrix m_X; /// [NxD] training inputs
	Vector m_y; /// [Nx1] training outputs
};

}
#endif