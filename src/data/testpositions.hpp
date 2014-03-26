#ifndef _TEST_POSITIONS_HPP_
#define _TEST_POSITIONS_HPP_

#include "typetraits.hpp"

/**
 * @class	TestPositions
 * @brief	A test positions.
 * 			M: number of test positions
 * 			D: number of dimensions
 * 			{(X*)_i}_i=1^M, X \in R^D
 * @author	Soohwankim
 * @date		26/03/2014
 */
template<Scalar>
class TestPositions
{
public:
   // Boost shared pointers
	typedef	boost::shared_ptr<TestPositions<Scalar> >				Ptr;
	typedef	boost::shared_ptr<const TestPositions<Scalar> >		ConstPtr;
	
	/**	
	 * @brief	Gets the number of test positions, M.
	 * @return	the number of test positions.
	 */
	inline int M() const
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
	TypeTraits<Scalar>::Matrix Xs_; /// [MxD] training inputs
};

#endif