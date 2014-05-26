#ifndef _USING_SQUARED_DISTANCE_HPP_
#define _USING_SQUARED_DISTANCE_HPP_

#include "../data/trainingdatasettable.hpp"
#include "../util/pairwiseop.hpp"

namespace GP{

template<typename Scalar>
class UsingSqDist : public TrainingDataSettable<Scalar>
{
public:
	/**
		* @brief		Sets a training data and pre-calculate
		* 				the squared distance between the training inputs.
		* 				Overloading the setter of TrainingDataSettable.
		* @param	pX	The training inputs. An NxD matrix.
		* @param	pY	The training outputs. An Nx1 vector.
		*/
	bool set(const MatrixConstPtr pX, VectorConstPtr pY)
	{
		// Set the training data.
		if(!TrainingDataSettable<Scalar>::set(pX, pY)) return false;

		// Pre-calculate the squared distances between the training inputs.
		m_pSqDist = PairwiseOp<Scalar>::sqDist(pX);

		return true;
	}

protected:

	/** @brief	Pre-calculated squared distances between the training inputs. */
	MatrixConstPtr m_pSqDist; // NxN matrix
};

}

#endif