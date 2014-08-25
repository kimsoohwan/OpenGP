#ifndef _TEST_CASE_COVARIANCE_FUNCTION_MATERN_ISO_HPP_
#define _TEST_CASE_COVARIANCE_FUNCTION_MATERN_ISO_HPP_

#include "TestFunctionDataSetting.hpp"


/**
 * @class	TestCaseCovMaterniso
 * @brief	Test fixture for testing CovMaterniso class.
 * @note		Inherits from TestDataSetting
 * 			to use the initialized training data and test positions.
 * @author	Soohwan Kim
 * @date		25/08/2014
 */
class TestCaseCovMaterniso : public TestFunctionDataSetting
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
	CovMaterniso<TestType>::Hyp logHyp;
};

/** @brief	K: (NxN) self covariance matrix between the training data. */  
TEST_F(TestCaseCovMaterniso, KTest)
{
	// Expected value
	Matrix K1(5, 5);
	K1 <<  2.250000000000000f,  0.945903145706138f,  0.820140300147686f,  0.518379208154154f,  0.518379208154154f,
			 0.945903145706138f,  2.250000000000000f,  0.717445827143238f,  0.842397675075848f,  0.842397675075848f,
			 0.820140300147686f,  0.717445827143238f,  2.250000000000000f,  1.113145531082548f,  1.113145531082548f,
			 0.518379208154154f,  0.842397675075848f,  1.113145531082548f,  2.250000000000000f,  2.250000000000000f,
			 0.518379208154154f,  0.842397675075848f,  1.113145531082548f,  2.250000000000000f,  2.250000000000000f;

	// Actual value
	MatrixPtr pK2 = CovMaterniso<TestType>::K(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(ell)]: (NxN) partial derivative of K with respect to log(ell). */  
TEST_F(TestCaseCovMaterniso, dKdlogellTest)
{
	// Expected value
	Matrix K1(5, 5);
	K1 << 0.000000000000000f,  1.217163707948694f,  1.210665552457046f,  1.071448917416644f,  1.071448917416644f,
			1.217163707948694f,  0.000000000000000f,  1.184529772356234f,  1.213780873349521f,  1.213780873349521f,
			1.210665552457046f,  1.184529772356234f,  0.000000000000000f,  1.186888686348668f,  1.186888686348668f,
			1.071448917416644f,  1.213780873349521f,  1.186888686348668f,  0.000000000000000f,  0.000000000000000f,
			1.071448917416644f,  1.213780873349521f,  1.186888686348668f,  0.000000000000000f,  0.000000000000000f;

	// Actual value
	MatrixPtr pK2 = CovMaterniso<TestType>::K(logHyp, trainingData, 0);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(sigma_f)]: (NxN) partial derivative of K with respect to log(sigma_f). */  
TEST_F(TestCaseCovMaterniso, dKdlogsigmafTest)
{
	// Expected value
	Matrix K1(5, 5);
	K1 << 4.500000000000000f,  1.891806291412276f,  1.640280600295373f,  1.036758416308309f,  1.036758416308309f,
			1.891806291412276f,  4.500000000000000f,  1.434891654286475f,  1.684795350151696f,  1.684795350151696f,
			1.640280600295373f,  1.434891654286475f,  4.500000000000000f,  2.226291062165095f,  2.226291062165095f,
			1.036758416308309f,  1.684795350151696f,  2.226291062165095f,  4.500000000000000f,  4.500000000000000f,
			1.036758416308309f,  1.684795350151696f,  2.226291062165095f,  4.500000000000000f,  4.500000000000000f;

	// Actual value
	MatrixPtr pK2 = CovMaterniso<TestType>::K(logHyp, trainingData, 1);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	Ks: (NxM) cross covariance matrix between the training data and test data. */  
TEST_F(TestCaseCovMaterniso, KsTest)
{
	// Expected value
	Matrix Ks1(5, 3);
	Ks1 << 0.408139686053389f,  0.518379208154154f,  0.424549766729713f,
			 0.218283235552083f,  0.842397675075848f,  0.699842546493200f,
			 1.119607160578329f,  1.113145531082548f,  0.736183711504438f,
			 0.376514509003230f,  2.250000000000000f,  0.392455499736434f,
			 0.376514509003230f,  2.250000000000000f,  0.392455499736434f;

	// Actual value
	MatrixPtr pKs2 = CovMaterniso<TestType>::Ks(logHyp, trainingData, testData);

	// Test
	TEST_MACRO::COMPARE(Ks1, *pKs2, __FILE__, __LINE__);
}

/** @brief	Kss: (Nx1) self variance matrix between the test data. */  
TEST_F(TestCaseCovMaterniso, KssTest)
{
	// Expected value
	Matrix Kss1(3, 1);
	Kss1 << 2.250000000000000f, 
			  2.250000000000000f, 
			  2.250000000000000f;

	// Actual value
	MatrixPtr pKss2 = CovMaterniso<TestType>::Kss(logHyp, testData);

	// Test
	TEST_MACRO::COMPARE(Kss1, *pKss2, __FILE__, __LINE__);
}
#endif