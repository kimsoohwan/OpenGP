#ifndef _TEST_CASE_COVARIANCE_FUNCTION_PRODUCT_HPP_
#define _TEST_CASE_COVARIANCE_FUNCTION_PRODUCT_HPP_

#include "TestFunctionDataSetting.hpp"


/**
 * @class	TestCaseCovSEisoMaterniso
 * @brief	Test fixture for testing CovSEisoMaterniso class.
 * @note		Inherits from TestDataSetting
 * 			to use the initialized training data and test positions.
 * @author	Soohwan Kim
 * @date		25/08/2014
 */
class TestCaseCovSEisoMaterniso : public TestFunctionDataSetting
{
protected:
	/** @brief	Overloading the test fixture set up. */
	virtual void SetUp()
	{
		// Call the parent set up.
		TestFunctionDataSetting::SetUp();

		// Set the hyperparameters.
		logHyp(0) = log(ell);
		logHyp(1) = log(sigma_f);
		logHyp(2) = log(ell);
		logHyp(3) = log(sigma_f);
	}

protected:
	/** @brief Log hyperparameters: log([ell, sigma_f]). */
	CovSEisoMaterniso<TestType>::Hyp logHyp;
};

/** @brief	K: (NxN) self covariance matrix between the training data. */  
TEST_F(TestCaseCovSEisoMaterniso, KTest)
{
	// Expected value
	Matrix K1(5, 5);
	K1 <<  5.062500000000000f,  1.131095299996123f,  0.848131038115009f,  0.314566189277918f,  0.314566189277918f,
			 1.131095299996123f,  5.062500000000000f,  0.641644478569553f,  0.895908946106318f,  0.895908946106318f,
			 0.848131038115009f,  0.641644478569553f,  5.062500000000000f,  1.551400421143818f,  1.551400421143818f,
			 0.314566189277918f,  0.895908946106318f,  1.551400421143818f,  5.062500000000000f,  5.062500000000000f,
			 0.314566189277918f,  0.895908946106318f,  1.551400421143818f,  5.062500000000000f,  5.062500000000000f;

	// Actual value
	MatrixPtr pK2 = CovSEisoMaterniso<TestType>::K(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(ell)]: (NxN) partial derivative of K with respect to log(ell). */  
TEST_F(TestCaseCovSEisoMaterniso, dKdlogell1Test)
{
	// Expected value
	Matrix K1(5, 5);
	K1 << 0.000000000000000f,  1.429995518913143f,  1.318624080768839f,  0.824441957959973f,  0.824441957959973f,
			1.429995518913143f,  0.000000000000000f,  1.183953768356236f,  1.342687327213515f,  1.342687327213515f,
			1.318624080768839f,  1.183953768356236f,  0.000000000000000f,  1.486123743927200f,  1.486123743927200f,
			0.824441957959973f,  1.342687327213515f,  1.486123743927200f,  0.000000000000000f,  0.000000000000000f,
			0.824441957959973f,  1.342687327213515f,  1.486123743927200f,  0.000000000000000f,  0.000000000000000f;

	// Actual value
	MatrixPtr pK2 = CovSEisoMaterniso<TestType>::K(logHyp, trainingData, 0);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(sigma_f)]: (NxN) partial derivative of K with respect to log(sigma_f). */  
TEST_F(TestCaseCovSEisoMaterniso, dKdlogsigmaf1Test)
{
	// Expected value
	Matrix K1(5, 5);
	K1 <<  10.125000000000000f,  2.262190599992246f,  1.696262076230018f,  0.629132378555836f,  0.629132378555836f,
			 2.262190599992246f,  10.125000000000000f,  1.283288957139107f,  1.791817892212636f,  1.791817892212636f,
			 1.696262076230018f,  1.283288957139107f,  10.125000000000000f,  3.102800842287636f,  3.102800842287636f,
			 0.629132378555836f,  1.791817892212636f,  3.102800842287636f,  10.125000000000000f,  10.125000000000000f,
			 0.629132378555836f,  1.791817892212636f,  3.102800842287636f,  10.125000000000000f,  10.125000000000000f;

	// Actual value
	MatrixPtr pK2 = CovSEisoMaterniso<TestType>::K(logHyp, trainingData, 1);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(ell)]: (NxN) partial derivative of K with respect to log(ell). */  
TEST_F(TestCaseCovSEisoMaterniso, dKdlogell2Test)
{
	// Expected value
	Matrix K1(5, 5);
	K1 << 0.000000000000000f,  1.455464183237136f,  1.251984607548945f,  0.650183490495004f,  0.650183490495004f,
			1.455464183237136f,  0.000000000000000f,  1.059378923646432f,  1.290883362123079f,  1.290883362123079f,
			1.251984607548945f,  1.059378923646432f,  0.000000000000000f,  1.654176885623780f,  1.654176885623780f,
			0.650183490495004f,  1.290883362123079f,  1.654176885623780f,  0.000000000000000f,  0.000000000000000f,
			0.650183490495004f,  1.290883362123079f,  1.654176885623780f,  0.000000000000000f,  0.000000000000000f;

	// Actual value
	MatrixPtr pK2 = CovSEisoMaterniso<TestType>::K(logHyp, trainingData, 2);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(sigma_f)]: (NxN) partial derivative of K with respect to log(sigma_f). */  
TEST_F(TestCaseCovSEisoMaterniso, dKdlogsigmaf2Test)
{
	// Expected value
	Matrix K1(5, 5);
	K1 <<  10.125000000000000f,  2.262190599992246f,  1.696262076230018f,  0.629132378555836f,  0.629132378555836f,
			 2.262190599992246f,  10.125000000000000f,  1.283288957139107f,  1.791817892212636f,  1.791817892212636f,
			 1.696262076230018f,  1.283288957139107f,  10.125000000000000f,  3.102800842287636f,  3.102800842287636f,
			 0.629132378555836f,  1.791817892212636f,  3.102800842287636f,  10.125000000000000f,  10.125000000000000f,
			 0.629132378555836f,  1.791817892212636f,  3.102800842287636f,  10.125000000000000f,  10.125000000000000f;

	// Actual value
	MatrixPtr pK2 = CovSEisoMaterniso<TestType>::K(logHyp, trainingData, 3);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	Ks: (NxM) cross covariance matrix between the training data and test data. */  
TEST_F(TestCaseCovSEisoMaterniso, KsTest)
{
	// Expected value
	Matrix Ks1(5, 3);
	Ks1 << 0.180569379524433f,  0.314566189277918f,  0.198220440904747f,
			 0.037542741377696f,  0.895908946106318f,  0.608673899524569f,
			 1.568527403854430f,  1.551400421143818f,  0.677542687023012f,
			 0.148884011301953f,  5.062500000000000f,  0.164469465529416f,
			 0.148884011301953f,  5.062500000000000f,  0.164469465529416f;

	// Actual value
	MatrixPtr pKs2 = CovSEisoMaterniso<TestType>::Ks(logHyp, trainingData, testData);

	// Test
	TEST_MACRO::COMPARE(Ks1, *pKs2, __FILE__, __LINE__);
}

/** @brief	Kss: (Nx1) self variance matrix between the test data. */  
TEST_F(TestCaseCovSEisoMaterniso, KssTest)
{
	// Expected value
	Matrix Kss1(3, 1);
	Kss1 << 5.062500000000000f,
			  5.062500000000000f,
			  5.062500000000000f;

	// Actual value
	MatrixPtr pKss2 = CovSEisoMaterniso<TestType>::Kss(logHyp, testData);

	// Test
	TEST_MACRO::COMPARE(Kss1, *pKss2, __FILE__, __LINE__);
}
#endif