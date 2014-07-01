#ifndef _TEST_DATA_SETTING_HPP_
#define _TEST_DATA_SETTING_HPP_

// STL
#include <iostream>
using namespace std;

// Google Test
#include "gtest/gtest.h"

// GP
#include "gp.h"
using namespace GP;

/**
 * @class	DataSetting
 * @brief	A data setting test fixture.
 * 			Initialize the training data, test positions, and hyperparameters
 * 			which are commonly used in mean/cov/lik/inf functions.
 * @author	Soohwan Kim
 * @date		28/03/2014
 */
class TestDataSetting : public ::testing::Test
{
protected:
	/** @brief	Constructor. */
	TestDataSetting() :
		pX  (new MatrixXf(5, 3)),	// Training inputs,	N =5, D=3
		pXd (new MatrixXf(4, 3)),	// Training inputs,	Nd=4, D=3
		pY  (new VectorXf(5)),		// Training outputs, N =5, D=1
	   pXs (new MatrixXf(3, 3)),	// Test inputs,		M =3, D=3
		ell(0.5f),						// Settint the hyperparameter, ell
		sigma_f(1.5f),					// Settint the hyperparameter, sigma_f
		sigma_n(0.1f)					// Settint the hyperparameter, sigma_n
	{
	}

	/** @brief	Set up the test fixture. */
	virtual void SetUp()
	{
		// Initialize the training inputs. A 5x3 matrix.
		(*pX) << 0.789963029944531f, 0.111705744193203f, 0.189710406017580f, 
					0.318524245398992f, 0.136292548938299f, 0.495005824990221f, 
					0.534064127370726f, 0.678652304800188f, 0.147608221976689f, 
					0.089950678770581f, 0.495177019089661f, 0.054974146906188f, 
					0.089950678770581f, 0.495177019089661f, 0.054974146906188f;

		// Initialize the derivative training inputs. A 4x3 matrix.
		(*pXd) << 0.850712674289007f, 0.929608866756663f, 0.582790965175840f,
					 0.560559527354885f, 0.696667200555228f, 0.815397211477421f, 
					 0.089950678770581f, 0.495177019089661f, 0.054974146906188f, 
					 0.560559527354885f, 0.696667200555228f, 0.815397211477421f;

		// Initialize the training outputs. A 5x1 vector.
		(*pY) << 0.913337361501670f,
			      0.152378018969223f,
					0.825816977489547f,
					0.538342435260057f,
					0.996134716626885f;

		// Initialize the test inputs. A 3x3 matrix.
		(*pXs) << 0.879013904597178f, 0.988911616079589f, 0.000522375356945f, 
					 0.089950678770581f, 0.495177019089661f, 0.054974146906188f, 
					 0.560559527354885f, 0.696667200555228f, 0.815397211477421f;

		// Set the training data
		trainingData.set(pX, pY);

		// Set the derivative training data
		derivativeTrainingData.set(pX, pXd, pY);

		// Set the test data
		testData.set(pXs);
	}

	//virtual void TearDown() {}

protected:
	/** @brief The Training data. */
	TrainingData<float>				trainingData;

	/** @brief The Derivative Training data. */
	DerivativeTrainingData<float>	derivativeTrainingData;

	/** @brief The Test data. */
	TestData<float>		testData;

	/** @brief The Training inputs. */
	MatrixXfPtr pX;

	/** @brief The Derivative Training inputs. */
	MatrixXfPtr pXd;

	/** @brief The Training outputs. */
	VectorXfPtr pY;

	/** @brief The Test inputs. */
	MatrixXfPtr pXs;

	/** @brief Characteristic length-scale hyperparameter, ell */
	const float ell;

	/** @brief Signal variance hyperarameter, sigma_f^2 */
	const float sigma_f;

	/** @brief Noise variance hyperparameter: sigma_n^2 */
	const float sigma_n;
};

#endif