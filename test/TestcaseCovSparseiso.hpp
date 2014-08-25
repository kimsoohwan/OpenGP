#ifndef _TEST_CASE_COVARIANCE_FUNCTION_SPARSE_ISO_HPP_
#define _TEST_CASE_COVARIANCE_FUNCTION_SPARSE_ISO_HPP_

#include "TestFunctionDataSetting.hpp"


/**
 * @class	TestCaseCovSparseiso
 * @brief	Test fixture for testing CovSparseiso class.
 * @note		Inherits from TestDataSetting
 * 			to use the initialized training data and test positions.
 * @author	Soohwan Kim
 * @date		25/08/2014
 */
class TestCaseCovSparseiso : public TestFunctionDataSetting
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
	}

protected:
	/** @brief Log hyperparameters: log([ell, sigma_f]). */
	CovSparseiso<TestType>::Hyp logHyp;
};

/** @brief	K: (NxN) self covariance matrix between the training data. */  
TEST_F(TestCaseCovSparseiso, KTest)
{
	// Expected value
	Matrix K1(5, 5);
	K1 <<  2.250000000000000f,  0.000000000000000f,  0.000000000000000f,  0.000000000000000f,  0.000000000000000f,
			 0.000000000000000f,  2.250000000000000f,  0.000000000000000f,  0.000000000000000f,  0.000000000000000f,
			 0.000000000000000f,  0.000000000000000f,  2.250000000000000f,  0.000000084623653f,  0.000000084623653f,
			 0.000000000000000f,  0.000000000000000f,  0.000000084623653f,  2.250000000000000f,  2.250000000000000f,
			 0.000000000000000f,  0.000000000000000f,  0.000000084623653f,  2.250000000000000f,  2.250000000000000f;

	// Actual value
	MatrixPtr pK2 = CovSparseiso<TestType>::K(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(ell)]: (NxN) partial derivative of K with respect to log(ell). */  
TEST_F(TestCaseCovSparseiso, dKdlogellTest)
{
	// Expected value
	Matrix K1(5, 5);
	K1 <<  -0.000000000000000f,  0.000000000000000f,  0.000000000000000f,  0.000000000000000f,  0.000000000000000f,
			  0.000000000000000f,  -0.000000000000000f,  0.000000000000000f,  0.000000000000000f,  0.000000000000000f,
			  0.000000000000000f,  0.000000000000000f,  -0.000000000000000f,  0.000019468536748f,  0.000019468536748f,
			  0.000000000000000f,  0.000000000000000f,  0.000019468536748f,  -0.000000000000000f,  -0.000000000000000f,
			  0.000000000000000f,  0.000000000000000f,  0.000019468536748f,  -0.000000000000000f,  -0.000000000000000f;

	// Actual value
	MatrixPtr pK2 = CovSparseiso<TestType>::K(logHyp, trainingData, 0);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(sigma_f)]: (NxN) partial derivative of K with respect to log(sigma_f). */  
TEST_F(TestCaseCovSparseiso, dKdlogsigmafTest)
{
	// Expected value
	Matrix K1(5, 5);
	K1 <<  4.500000000000000f,  0.000000000000000f,  0.000000000000000f,  0.000000000000000f,  0.000000000000000f,
			 0.000000000000000f,  4.500000000000000f,  0.000000000000000f,  0.000000000000000f,  0.000000000000000f,
			 0.000000000000000f,  0.000000000000000f,  4.500000000000000f,  0.000000169247307f,  0.000000169247307f,
			 0.000000000000000f,  0.000000000000000f,  0.000000169247307f,  4.500000000000000f,  4.500000000000000f,
			 0.000000000000000f,  0.000000000000000f,  0.000000169247307f,  4.500000000000000f,  4.500000000000000f;

	// Actual value
	MatrixPtr pK2 = CovSparseiso<TestType>::K(logHyp, trainingData, 1);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	Ks: (NxM) cross covariance matrix between the training data and test data. */  
TEST_F(TestCaseCovSparseiso, KsTest)
{
	// Expected value
	Matrix Ks1(5, 3);
	Ks1 << 0.000000000000000f,  0.000000000000000f,  0.000000000000000f,
			 0.000000000000000f,  0.000000000000000f,  0.000000000000000f,
			 0.000000258242034f,  0.000000084623653f,  0.000000000000000f,
			 0.000000000000000f,  2.250000000000000f,  0.000000000000000f,
			 0.000000000000000f,  2.250000000000000f,  0.000000000000000f;

	// Actual value
	MatrixPtr pKs2 = CovSparseiso<TestType>::Ks(logHyp, trainingData, testData);

	// Test
	TEST_MACRO::COMPARE(Ks1, *pKs2, __FILE__, __LINE__);
}

/** @brief	Kss: (Nx1) self variance matrix between the test data. */  
TEST_F(TestCaseCovSparseiso, KssTest)
{
	// Expected value
	Matrix Kss1(3, 1);
	Kss1 << 2.250000000000000f, 
			  2.250000000000000f, 
			  2.250000000000000f;

	// Actual value
	MatrixPtr pKss2 = CovSparseiso<TestType>::Kss(logHyp, testData);

	// Test
	TEST_MACRO::COMPARE(Kss1, *pKss2, __FILE__, __LINE__);
}
#endif