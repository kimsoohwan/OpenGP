#ifndef _TEST_FUNCTION_DATA_SETTING2_HPP_
#define _TEST_FUNCTION_DATA_SETTING2_HPP_

#include "TestMacros.hpp"

/**
 * @class	TestFunctionDataSetting2
 * @brief	A data setting test fixture.
 * 			Initialize the training data, test positions, and hyperparameters
 * 			which are commonly used in mean/cov/lik/inf functions.
 * @author	Soohwan Kim
 * @date		15/07/2014
 */
class TestFunctionDataSetting2 : public ::testing::Test
{
// define matrix and vector types
protected: TYPE_DEFINE_MATRIX(TestType);
			  TYPE_DEFINE_VECTOR(TestType);

protected:
	/** @brief	Constructor. */
	TestFunctionDataSetting2() :
		pX  (new Matrix(10, 1)),	// Training inputs,	N =10,  D=1
		pY  (new Vector(10)),		// Training outputs, N =10,  D=1
	   pXs (new Matrix(101, 1)),	// Test inputs,		M =101, D=1
		ell(0.5f),						// Settint the hyperparameter, ell
		sigma_f(1.5f),					// Settint the hyperparameter, sigma_f
		sigma_n(0.1f)					// Settint the hyperparameter, sigma_n
	{
	}

	/** @brief	Set up the test fixture. */
	virtual void SetUp()
	{
		// Initialize the training inputs. A 5x3 matrix.
		(*pX) <<  0.040000000000000f, 
					 0.140000000000000f, 
					 0.240000000000000f, 
					 0.340000000000000f, 
					 0.440000000000000f, 
					 0.540000000000000f, 
					 0.640000000000000f, 
					 0.740000000000000f, 
					 0.840000000000000f, 
					 0.940000000000000f;

		// Initialize the training outputs. A 5x1 vector.
		(*pY) <<   1.162470824949019f, 
					  2.649518079414324f, 
					  0.826178856519631f, 
					  -0.296449902131397f, 
					  -2.925889279464160f, 
					  -3.059479729867036f, 
					  -2.684443970935226f, 
					  -0.724953106178071f, 
					  -0.871446982536995f, 
					  -1.909489684161842f;

		// Initialize the test inputs. A 3x3 matrix.
		(*pXs) <<   0.000000000000000f,  0.010000000000000f,  0.020000000000000f,  0.030000000000000f,  0.040000000000000f,  0.050000000000000f,  0.060000000000000f,  0.070000000000000f,  0.080000000000000f,  0.090000000000000f, 
						0.100000000000000f,  0.110000000000000f,  0.120000000000000f,  0.130000000000000f,  0.140000000000000f,  0.150000000000000f,  0.160000000000000f,  0.170000000000000f,  0.180000000000000f,  0.190000000000000f, 
						0.200000000000000f,  0.210000000000000f,  0.220000000000000f,  0.230000000000000f,  0.240000000000000f,  0.250000000000000f,  0.260000000000000f,  0.270000000000000f,  0.280000000000000f,  0.290000000000000f, 
						0.300000000000000f,  0.310000000000000f,  0.320000000000000f,  0.330000000000000f,  0.340000000000000f,  0.350000000000000f,  0.360000000000000f,  0.370000000000000f,  0.380000000000000f,  0.390000000000000f, 
						0.400000000000000f,  0.410000000000000f,  0.420000000000000f,  0.430000000000000f,  0.440000000000000f,  0.450000000000000f,  0.460000000000000f,  0.470000000000000f,  0.480000000000000f,  0.490000000000000f, 
						0.500000000000000f,  0.510000000000000f,  0.520000000000000f,  0.530000000000000f,  0.540000000000000f,  0.550000000000000f,  0.560000000000000f,  0.570000000000000f,  0.580000000000000f,  0.590000000000000f, 
						0.600000000000000f,  0.610000000000000f,  0.620000000000000f,  0.630000000000000f,  0.640000000000000f,  0.650000000000000f,  0.660000000000000f,  0.670000000000000f,  0.680000000000000f,  0.690000000000000f, 
						0.700000000000000f,  0.710000000000000f,  0.720000000000000f,  0.730000000000000f,  0.740000000000000f,  0.750000000000000f,  0.760000000000000f,  0.770000000000000f,  0.780000000000000f,  0.790000000000000f, 
						0.800000000000000f,  0.810000000000000f,  0.820000000000000f,  0.830000000000000f,  0.840000000000000f,  0.850000000000000f,  0.860000000000000f,  0.870000000000000f,  0.880000000000000f,  0.890000000000000f, 
						0.900000000000000f,  0.910000000000000f,  0.920000000000000f,  0.930000000000000f,  0.940000000000000f,  0.950000000000000f,  0.960000000000000f,  0.970000000000000f,  0.980000000000000f,  0.990000000000000f, 
						1.000000000000000f;


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