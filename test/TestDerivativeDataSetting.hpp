#ifndef _TEST_DERIVATIVE_DATA_SETTING_HPP_
#define _TEST_DERIVATIVE_DATA_SETTING_HPP_

#include "TestFunctionDataSetting.hpp"

/**
 * @class	TestDerivativeDataSetting
 * @brief	A data setting test fixture.
 * 			Initialize the training data, test positions, and hyperparameters
 * 			which are commonly used in mean/cov/lik/inf functions.
 * @author	Soohwan Kim
 * @date		02/07/2014
 */
class TestDerivativeDataSetting : public TestFunctionDataSetting
{
protected:
	/** @brief	Constructor. */
	TestDerivativeDataSetting() :
		pXd (new Matrix(4, 3)),	// Training inputs,	Nd=4,  D=3
		pYYd(new Vector(17))		// Training outputs, NN=17, D=1
	{
	}

	/** @brief	Set up the test fixture. */
	virtual void SetUp()
	{
		TestFunctionDataSetting::SetUp();

		// Initialize the derivative training inputs. A 4x3 matrix.
		(*pXd) << 0.850712674289007f, 0.929608866756663f, 0.582790965175840f,
					 0.560559527354885f, 0.696667200555228f, 0.815397211477421f, 
					 0.089950678770581f, 0.495177019089661f, 0.054974146906188f, 
					 0.560559527354885f, 0.696667200555228f, 0.815397211477421f;

		// Initialize the training outputs. A 17x1 vector.
		(*pYYd) <<  0.346448761300360f, 
						0.886543861760306f, 
						0.454694864991908f, 
						0.413427289020815f, 
						0.217732068357300f, 
						0.125654587362626f, 
						0.308914593566815f, 
						0.726104431664832f, 
						0.782872072979123f, 
						0.693787614986897f, 
						0.009802252263062f, 
						0.843213338010510f, 
						0.922331997796276f, 
						0.770954220673925f, 
						0.042659855935049f, 
						0.378186137050219f, 
						0.704339624483368f;

		// Set the derivative training data
		derivativeTrainingData.set(pX, pXd, pYYd);
	}

	//virtual void TearDown() {}

protected:
	/** @brief The Derivative Training data. */
	DerivativeTrainingData<TestType>	derivativeTrainingData;

	/** @brief The Derivative Training inputs. */
	MatrixPtr pXd;

	/** @brief The Function and Derivative Training outputs. */
	VectorPtr pYYd;
};

#endif