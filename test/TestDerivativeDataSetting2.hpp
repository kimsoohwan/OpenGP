#ifndef _TEST_DERIVATIVE_DATA_SETTING2_HPP_
#define _TEST_DERIVATIVE_DATA_SETTING2_HPP_

#include "TestFunctionDataSetting2.hpp"

/**
 * @class	TestDerivativeDataSetting2
 * @brief	A data setting test fixture.
 * 			Initialize the training data, test positions, and hyperparameters
 * 			which are commonly used in mean/cov/lik/inf functions.
 * @author	Soohwan Kim
 * @date		15/07/2014
 */
class TestDerivativeDataSetting2 : public TestFunctionDataSetting
{
protected:
	/** @brief	Constructor. */
	TestDerivativeDataSetting2() :
		pYYd(new Vector(20))		// Training outputs, NN=20, D=1
	{
	}

	/** @brief	Set up the test fixture. */
	virtual void SetUp()
	{
		TestFunctionDataSetting::SetUp();

		// Initialize the training outputs. A 17x1 vector.
		(*pYYd) <<   1.162470824949019f, 
						 2.649518079414324f, 
						 0.826178856519631f, 
						 -0.296449902131397f, 
						 -2.925889279464160f, 
						 -3.059479729867036f, 
						 -2.684443970935226f, 
						 -0.724953106178071f, 
						 -0.871446982536995f, 
						 -1.909489684161842f, 
						 41.704618124666638f, 
						 -28.757549228476215f, 
						 14.122022625746855f, 
						 -52.574386176465111f, 
						 15.885794730520876f, 
						 -27.330344334593999f, 
						 43.308255238127209f, 
						 -20.443387861589134f, 
						 22.877410410575429f, 
						 -52.447400235726114f;

		// Set the derivative training data
		derivativeTrainingData.set(pX, pYYd);
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