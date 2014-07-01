#ifndef _TEST_CASE_COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_
#define _TEST_CASE_COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP_

#include "TestDataSetting.hpp"


/**
 * @class	TestCaseCovSEiso
 * @brief	Test fixture for testing CovSEiso class.
 * @note		Inherits from TestDataSetting
 * 			to use the initialized training data and test positions.
 * @author	Soohwan Kim
 * @date		28/03/2014
 */
class TestCaseCovSEiso : public TestDataSetting
{
protected:
	/** @brief	Overloading the test fixture set up. */
	virtual void SetUp()
	{
		// Call the parent set up.
		TestDataSetting::SetUp();

		// Set the hyperparameters.
		logHyp(0) = log(ell);
		logHyp(1) = log(sigma_f);
	}

protected:
	/** @brief Log hyperparameters: log([ell, sigma_f]). */
	CovSEiso<float>::Hyp logHyp;
};

/** @brief	K: (NxN) self covariance matrix between the training data. */  
TEST_F(TestCaseCovSEiso, KTest)
{
	// Expected value
	MatrixXf K1(5, 5);
	K1 << 2.250000000000000f, 1.195783421516941f, 1.034129206871412f, 0.606826401078133f, 0.606826401078133f, 
			1.195783421516941f, 2.250000000000000f, 0.894345544003630f, 1.063522576822939f, 1.063522576822939f, 
			1.034129206871412f, 0.894345544003630f, 2.250000000000000f, 1.393708529409504f, 1.393708529409504f, 
			0.606826401078133f, 1.063522576822939f, 1.393708529409504f, 2.250000000000000f, 2.250000000000000f, 
			0.606826401078133f, 1.063522576822939f, 1.393708529409504f, 2.250000000000000f, 2.250000000000000f;

	// Actual value
	MatrixXfPtr pK2 = CovSEiso<float>::K(logHyp, trainingData);

	// Test
	EXPECT_TRUE(K1.isApprox(*pK2))
		<< "Expected: " << endl << K1 << endl << endl 
		<< "Actual: " << endl << *pK2 << endl << endl;
}

/** @brief	pd[K]/pd[log(ell)]: (NxN) partial derivative of K with respect to log(ell). */  
TEST_F(TestCaseCovSEiso, dKdlogellTest)
{
	// Expected value
	MatrixXf K1(5, 5);
	K1 << 0.000000000000000f, 1.511777950421784f, 1.607803055808120f, 1.590422503432665f, 1.590422503432665f, 
			1.511777950421784f, 0.000000000000000f, 1.650234378072228f, 1.593887740837630f, 1.593887740837630f, 
			1.607803055808120f, 1.650234378072228f, 0.000000000000000f, 1.335066891462005f, 1.335066891462005f, 
			1.590422503432665f, 1.593887740837630f, 1.335066891462005f, 0.000000000000000f, 0.000000000000000f, 
			1.590422503432665f, 1.593887740837630f, 1.335066891462005f, 0.000000000000000f, 0.000000000000000f;

	// Actual value
	MatrixXfPtr pK2 = CovSEiso<float>::K(logHyp, trainingData, 0);

	// Test
	EXPECT_TRUE(K1.isApprox(*pK2))
		<< "Expected: " << endl << K1 << endl << endl 
		<< "Actual: " << endl << *pK2 << endl << endl;
}

/** @brief	pd[K]/pd[log(sigma_f)]: (NxN) partial derivative of K with respect to log(sigma_f). */  
TEST_F(TestCaseCovSEiso, dKdlogsigmafTest)
{
	// Expected value
	MatrixXf K1(5, 5);
	K1 << 4.500000000000000f, 2.391566843033882f, 2.068258413742825f, 1.213652802156266f, 1.213652802156266f, 
			2.391566843033882f, 4.500000000000000f, 1.788691088007260f, 2.127045153645877f, 2.127045153645877f, 
			2.068258413742825f, 1.788691088007260f, 4.500000000000000f, 2.787417058819007f, 2.787417058819007f, 
			1.213652802156266f, 2.127045153645877f, 2.787417058819007f, 4.500000000000000f, 4.500000000000000f, 
			1.213652802156266f, 2.127045153645877f, 2.787417058819007f, 4.500000000000000f, 4.500000000000000f;

	// Actual value
	MatrixXfPtr pK2 = CovSEiso<float>::K(logHyp, trainingData, 1);

	// Test
	EXPECT_TRUE(K1.isApprox(*pK2))
		<< "Expected: " << endl << K1 << endl << endl 
		<< "Actual: " << endl << *pK2 << endl << endl;
}

/** @brief	Ks: (NxM) cross covariance matrix between the training data and test data. */  
TEST_F(TestCaseCovSEiso, KsTest)
{
	// Expected value
	MatrixXf Ks1(5, 3);
	Ks1 << 0.442420538101783f, 0.606826401078133f, 0.466895653792557f, 
			 0.171990951493561f, 1.063522576822939f, 0.869729773610560f, 
			 1.400962283095986f, 1.393708529409504f, 0.920344577630508f, 
			 0.395427022709171f, 2.250000000000000f, 0.419077999008476f, 
			 0.395427022709171f, 2.250000000000000f, 0.419077999008476f;

	// Actual value
	MatrixXfPtr pKs2 = CovSEiso<float>::Ks(logHyp, trainingData, testData);

	// Test
	EXPECT_TRUE(Ks1.isApprox(*pKs2))
		<< "Expected: " << endl << Ks1 << endl << endl 
		<< "Actual: " << endl << *pKs2 << endl << endl;
}

/** @brief	Kss: (Nx1) self variance matrix between the test data. */  
TEST_F(TestCaseCovSEiso, KssTest)
{
	// Expected value
	MatrixXf Kss1(3, 1);
	Kss1 << 2.250000000000000f, 
			  2.250000000000000f, 
			  2.250000000000000f;

	// Actual value
	MatrixXfPtr pKss2 = CovSEiso<float>::Kss(logHyp, testData);

	// Test
	EXPECT_TRUE(Kss1.isApprox(*pKss2))
		<< "Expected: " << endl << Kss1 << endl << endl 
		<< "Actual: " << endl << *pKss2 << endl << endl;
}
#endif