#ifndef _SQUARED_DISTANCE_HPP_
#define _SQUARED_DISTANCE_HPP_

#include "../gp/trainingdatasetter.hpp"

namespace GP{

typename<Scalar>
class TrainingDataSetterSqDist : public TrainingDataSetter
{
public:
	/**
		* @brief		Sets a training data and pre-calculate
		* 				the squared distance between training inputs.
		* @param	data	The data.
		*/
	void set(const TrainingDataConstPtr data)
	{
		TrainingDataSetter<Scalar>::set(data);
		m_pTrainingData = data;
	}
};

}

#endif