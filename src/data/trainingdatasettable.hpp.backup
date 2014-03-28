#ifndef _TRAINING_DATA_SETTABLE_HPP_
#define _TRAINING_DATA_SETTABLE_HPP_

#include "typetraits.hpp"
#include "trainingdata.hpp"

namespace GP{

/**
 * @class	TrainingDataSettable
 * @brief	Setting training data.
 * 			The mean/cov/lik functions need to set training data.
 * 			This is a base class to avoid duplicating 
 * 			the same feature to set training data for them.
 * @author	Soohwankim
 * @date	26/03/2014
 */
template<typename Scalar>
class TrainingDataSettable : public TypeTraits<Scalar>
{
public:
	// typedef
	typedef typename TrainingData<Scalar>::ConstPtr TrainingDataConstPtr;

	/**
		* @brief	Sets a training data.
		* @param	data	The data.
		*/
	bool set(const TrainingDataConstPtr data)
	{
		if(m_pTrainingData == data) return false;

		m_pTrainingData = data;
		return true;
	}

protected:

	/** @brief	The training data. */
	TrainingDataConstPtr		m_pTrainingData;
};

}
#endif