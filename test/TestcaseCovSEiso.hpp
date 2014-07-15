#ifndef _TEST_CASE_COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_
#define _TEST_CASE_COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_

#include "TestFunctionDataSetting.hpp"


/**
 * @class	TestCaseCovSEiso
 * @brief	Test fixture for testing CovSEiso class.
 * @note		Inherits from TestDataSetting
 * 			to use the initialized training data and test positions.
 * @author	Soohwan Kim
 * @date		28/03/2014
 */
class TestCaseCovSEiso : public TestFunctionDataSetting
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
	CovSEiso<TestType>::Hyp logHyp;
};

/** @brief	K: (NxN) self covariance matrix between the training data. */  
TEST_F(TestCaseCovSEiso, KTest)
{
	// Expected value
	Matrix K1(5, 5);
	K1 << 2.250000000000000f, 1.195783421516941f, 1.034129206871412f, 0.606826401078133f, 0.606826401078133f, 
			1.195783421516941f, 2.250000000000000f, 0.894345544003630f, 1.063522576822939f, 1.063522576822939f, 
			1.034129206871412f, 0.894345544003630f, 2.250000000000000f, 1.393708529409504f, 1.393708529409504f, 
			0.606826401078133f, 1.063522576822939f, 1.393708529409504f, 2.250000000000000f, 2.250000000000000f, 
			0.606826401078133f, 1.063522576822939f, 1.393708529409504f, 2.250000000000000f, 2.250000000000000f;

	// Actual value
	MatrixPtr pK2 = CovSEiso<TestType>::K(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(ell)]: (NxN) partial derivative of K with respect to log(ell). */  
TEST_F(TestCaseCovSEiso, dKdlogellTest)
{
	// Expected value
	Matrix K1(5, 5);
	K1 << 0.000000000000000f, 1.511777950421784f, 1.607803055808120f, 1.590422503432665f, 1.590422503432665f, 
			1.511777950421784f, 0.000000000000000f, 1.650234378072228f, 1.593887740837630f, 1.593887740837630f, 
			1.607803055808120f, 1.650234378072228f, 0.000000000000000f, 1.335066891462005f, 1.335066891462005f, 
			1.590422503432665f, 1.593887740837630f, 1.335066891462005f, 0.000000000000000f, 0.000000000000000f, 
			1.590422503432665f, 1.593887740837630f, 1.335066891462005f, 0.000000000000000f, 0.000000000000000f;

	// Actual value
	MatrixPtr pK2 = CovSEiso<TestType>::K(logHyp, trainingData, 0);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(sigma_f)]: (NxN) partial derivative of K with respect to log(sigma_f). */  
TEST_F(TestCaseCovSEiso, dKdlogsigmafTest)
{
	// Expected value
	Matrix K1(5, 5);
	K1 << 4.500000000000000f, 2.391566843033882f, 2.068258413742825f, 1.213652802156266f, 1.213652802156266f, 
			2.391566843033882f, 4.500000000000000f, 1.788691088007260f, 2.127045153645877f, 2.127045153645877f, 
			2.068258413742825f, 1.788691088007260f, 4.500000000000000f, 2.787417058819007f, 2.787417058819007f, 
			1.213652802156266f, 2.127045153645877f, 2.787417058819007f, 4.500000000000000f, 4.500000000000000f, 
			1.213652802156266f, 2.127045153645877f, 2.787417058819007f, 4.500000000000000f, 4.500000000000000f;

	// Actual value
	MatrixPtr pK2 = CovSEiso<TestType>::K(logHyp, trainingData, 1);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	Ks: (NxM) cross covariance matrix between the training data and test data. */  
TEST_F(TestCaseCovSEiso, KsTest)
{
	// Expected value
	Matrix Ks1(5, 3);
	Ks1 << 0.442420538101783f, 0.606826401078133f, 0.466895653792557f, 
			 0.171990951493561f, 1.063522576822939f, 0.869729773610560f, 
			 1.400962283095986f, 1.393708529409504f, 0.920344577630508f, 
			 0.395427022709171f, 2.250000000000000f, 0.419077999008476f, 
			 0.395427022709171f, 2.250000000000000f, 0.419077999008476f;

	// Actual value
	MatrixPtr pKs2 = CovSEiso<TestType>::Ks(logHyp, trainingData, testData);

	// Test
	TEST_MACRO::COMPARE(Ks1, *pKs2, __FILE__, __LINE__);
}

/** @brief	Kss: (Nx1) self variance matrix between the test data. */  
TEST_F(TestCaseCovSEiso, KssTest)
{
	// Expected value
	Matrix Kss1(3, 1);
	Kss1 << 2.250000000000000f, 
			  2.250000000000000f, 
			  2.250000000000000f;

	// Actual value
	MatrixPtr pKss2 = CovSEiso<TestType>::Kss(logHyp, testData);

	// Test
	TEST_MACRO::COMPARE(Kss1, *pKss2, __FILE__, __LINE__);
}
#endif