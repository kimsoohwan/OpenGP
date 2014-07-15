#ifndef _TEST_FUNCTION_DATA_SETTING_HPP_
#define _TEST_FUNCTION_DATA_SETTING_HPP_

#include "TestMacros.hpp"

/**
 * @class	TestFunctionDataSetting
 * @brief	A data setting test fixture.
 * 			Initialize the training data, test positions, and hyperparameters
 * 			which are commonly used in mean/cov/lik/inf functions.
 * @author	Soohwan Kim
 * @date		28/03/2014
 */
class TestFunctionDataSetting : public ::testing::Test
{
// define matrix and vector types
protected: TYPE_DEFINE_MATRIX(TestType);
			  TYPE_DEFINE_VECTOR(TestType);

protected:
	/** @brief	Constructor. */
	TestFunctionDataSetting() :
		pX  (new Matrix(5, 3)),	// Training inputs,	N =5,  D=3
		pY  (new Vector(5)),		// Training outputs, N =5,  D=1
	   pXs (new Matrix(3, 3)),	// Test inputs,		M =3,  D=3
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

		// Initialize the training outputs. A 5x1 vector.
		(*pY) <<  0.729513045504647f, 
					 0.224277070664514f, 
					 0.269054731773365f, 
					 0.673031165004119f, 
					 0.477492197726861f;

		// Initialize the test inputs. A 3x3 matrix.
		(*pXs) << 0.879013904597178f, 0.988911616079589f, 0.000522375356945f, 
					 0.089950678770581f, 0.495177019089661f, 0.054974146906188f, 
					 0.560559527354885f, 0.696667200555228f, 0.815397211477421f;

		// Set the training data
		trainingData.set(pX, pY);

		// Set the test data
		testData.set(pXs);
	}

	//virtual void TearDown() {}

protected:
	/** @brief The Training data. */
	TrainingData<TestType>				trainingData;

	/** @brief The Test data. */
	TestData<TestType>					testData;

	/** @brief The Function Training inputs. */
	MatrixPtr pX;

	/** @brief The Function Training outputs. */
	VectorPtr pY;

	/** @brief The Test inputs. */
	MatrixPtr pXs;

	/** @brief Characteristic length-scale hyperparameter, ell */
	const TestType ell;

	/** @brief Signal variance hyperarameter, sigma_f^2 */
	const TestType sigma_f;

	/** @brief Noise variance hyperparameter: sigma_n^2 */
	const TestType sigma_n;
};

#endif