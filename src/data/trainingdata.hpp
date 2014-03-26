#ifndef _TRAINING_DATA_HPP_
#define _TRAINING_DATA_HPP_

#include "DataTypes.hpp"

/**
 * @class	TrainingData
 * @brief	A training data.
 * 			N: number of training data
 * 			D: number of dimensions
 * 			{(X, y)_i}_i=1^N, X \in R^D, y \in R
 * @author	Soohwankim
 * @date		26/03/2014
 */
template<Scalar>
class TrainingData
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
		assert(X_.rows() == y_.size());
		return X_.rows();
	}

	/**
	 * @brief	Gets the number of dimensions, D.
	 * @return	the number of dimensions.
	 */
	inline int D() const
	{
		return X_.cols();
	}

protected:
	DataTypes<Scalar>::Matrix X_; /// [NxD] training inputs
	DataTypes<Scalar>::Vector y_; /// [Nx1] training outputs
};

#endif