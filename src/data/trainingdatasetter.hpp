#ifndef _TRAINING_DATA_SETTER_HPP_
#define _TRAINING_DATA_SETTER_HPP_

#include "datatypes.hpp"
#include "trainingdata.hpp"

namespace GP{

/**
 * @class	TrainingDataSetter
 * @brief	Setting training data.
 * 			The mean/cov/lik functions need to set training data.
 * 			This is a base class to avoid duplicating 
 * 			the same feature to set training data for them.
 * @author	Soohwankim
 * @date	26/03/2014
 */
typename<Scalar>
class TrainingDataSetter
{
public:
	/**
		* @brief	Sets a training data.
		* @param	data	The data.
		*/
	void set(const TrainingDataConstPtr data)
	{
		m_pTrainingData = data;
	}

protected:

	/** @brief	The training data. */
	TrainingData<Scalar>::ConstPtr		m_pTrainingData;
};

}
#endif