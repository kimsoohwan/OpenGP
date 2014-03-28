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
		* 				the squared distance between training inputs.
		* 				Overloading the setter function.
		* @param	data	The data.
		*/
	bool set(const TrainingDataConstPtr data)
	{
		if(!TrainingDataSettable<Scalar>::set(data)) return false;

		m_sqDist = PairwiseOp<Scalar>::sqDist(m_pTrainingData->pX());
	}

protected:
	MatrixConstPtr m_sqDist;
};

}

#endif