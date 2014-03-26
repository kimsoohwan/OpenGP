#ifndef _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_
#define _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_

#include "../gp/datatypes.hpp"
#include "../gp/trainingdata.hpp"
#include "../gp/trainingdatasetter.hpp"

namespace GP{

/**
	* @class		CovSEIso
	* @brief		Isotropic squared exponential covariance function
	* 				It inherits from TrainingDataSetter
	* 				to be able to set a training data.
	* @author	Soohwankim
	* @date		26/03/2014
	*/
template<Scalar>
class CovSEIso : public TrainingDataSetter<Scalar>
{
public:
	// Hyperparameters: sf, l
	typedef	DataTypes<Scalar>::Hyp2		Hyp;
};

}

#endif