#ifndef _TEST_CASE_COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_SLOW_HPP_
#define _TEST_CASE_COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_SLOW_HPP_

#include "testdatasetting.hpp"

/**
 * @class	TestCaseCovSEIsoSlow
 * @brief	Test fixture for testing CovSEIsoSlow class.
 * @note		Inherits from TestDataSetting
 * 			to use the initialized training data and test positions.
 * @author	Soohwan Kim
 * @date		03/06/2014
 */
class TestCaseCovSEIsoSlow : public TestDataSetting
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
	//CovSEIsoSlow<float>::Hyp logHyp;
	CovSEIsoSlow::Hyp logHyp;
};

/** @brief	K: (NxN) self covariance matrix between the training data. */  
TEST_F(TestCaseCovSEIsoSlow, KTest)
{
	// Expected value
	MatrixXf K1(5, 5);
	K1 << 2.250000000000000f, 0.958218774745736f, 0.431288446325351f, 0.560933083942909f, 0.308631547122618f, 
		   0.958218774745736f, 2.250000000000000f, 0.639672531767379f, 0.493453485948981f, 0.593443307553922f, 
			0.431288446325351f, 0.639672531767379f, 2.250000000000000f, 0.659656655969132f, 0.834888531682446f, 
			0.560933083942909f, 0.493453485948981f, 0.659656655969132f, 2.250000000000000f, 1.851836243293809f, 
			0.308631547122618f, 0.593443307553922f, 0.834888531682446f, 1.851836243293809f, 2.250000000000000f;

	// Actual value
	//MatrixXfPtr pK2 = CovSEIsoSlow<float>::K(logHyp, trainingData);
	MatrixXfPtr pK2 = CovSEIsoSlow::K(logHyp, trainingData);

	// Test
	EXPECT_TRUE(K1.isApprox(*pK2))
		<< "Expected: " << endl << K1 << endl << endl 
		<< "Actual: " << endl << *pK2 << endl << endl;
}

/** @brief	pd[K]/pd[log(ell)]: (NxN) partial derivative of K with respect to log(ell). */  
TEST_F(TestCaseCovSEIsoSlow, dKdlogellTest)
{
	// Expected value
	MatrixXf K1(5, 5);
	K1 << 0.000000000000000f, 1.635889063018654f, 1.424897997301291f, 1.558366205680773f, 1.226216181565493f,
		   1.635889063018654f, 0.000000000000000f, 1.609069539535710f, 1.497391407009958f, 1.581815787018512f, 
			1.424897997301291f, 1.609069539535710f, 0.000000000000000f, 1.618752595258672f, 1.655395731454394f, 
			1.558366205680773f, 1.497391407009958f, 1.618752595258672f, 0.000000000000000f, 0.721299496126858f, 
			1.226216181565493f, 1.581815787018512f, 1.655395731454394f, 0.721299496126858f, 0.000000000000000f;

	// Actual value
	//MatrixXfPtr pK2 = CovSEIsoSlow<float>::K(logHyp, trainingData, 0);
	MatrixXfPtr pK2 = CovSEIsoSlow::K(logHyp, trainingData, 0);

	// Test
	EXPECT_TRUE(K1.isApprox(*pK2))
		<< "Expected: " << endl << K1 << endl << endl 
		<< "Actual: " << endl << *pK2 << endl << endl;
}

/** @brief	pd[K]/pd[log(sigma_f)]: (NxN) partial derivative of K with respect to log(sigma_f). */  
TEST_F(TestCaseCovSEIsoSlow, dKdlogsigmafTest)
{
	// Expected value
	MatrixXf K1(5, 5);
	K1 << 4.500000000000000f, 1.916437549491473f, 0.862576892650702f, 1.121866167885818f, 0.617263094245237f,
		   1.916437549491473f, 4.500000000000000f, 1.279345063534759f, 0.986906971897962f, 1.186886615107843f, 
			0.862576892650702f, 1.279345063534759f, 4.500000000000000f, 1.319313311938264f, 1.669777063364892f, 
			1.121866167885817f, 0.986906971897962f, 1.319313311938264f, 4.500000000000000f, 3.703672486587619f, 
			0.617263094245237f, 1.186886615107843f, 1.669777063364892f, 3.703672486587619f, 4.500000000000000f;

	// Actual value
	//MatrixXfPtr pK2 = CovSEIsoSlow<float>::K(logHyp, trainingData, 1);
	MatrixXfPtr pK2 = CovSEIsoSlow::K(logHyp, trainingData, 1);

	// Test
	EXPECT_TRUE(K1.isApprox(*pK2))
		<< "Expected: " << endl << K1 << endl << endl 
		<< "Actual: " << endl << *pK2 << endl << endl;
}


/** @brief	Ks: (NxM) cross covariance matrix between the training data and test data. */  
TEST_F(TestCaseCovSEIsoSlow, KsTest)
{
	// Expected value
	MatrixXf Ks1(5, 4);
	Ks1 << 0.954130271843582f, 0.252272178204362f, 1.482432301701979f, 0.477606866068634f, 
		    0.438755577442086f, 1.319847987061875f, 0.902419962795361f, 0.475952898778052f, 
			 0.703986975482761f, 0.872942358738421f, 0.807864890700496f, 1.984336721955273f, 
			 1.885432860740457f, 0.604127429327114f, 1.687835119341133f, 1.082845369530834f, 
			 1.173863554051925f, 1.148821830108717f, 1.148148008255321f, 1.126543080103098f;

	// Actual value
	//MatrixXfPtr pKs2 = CovSEIsoSlow<float>::Ks(logHyp, trainingData, pXs);
	MatrixXfPtr pKs2 = CovSEIsoSlow::Ks(logHyp, trainingData, pXs);

	// Test
	EXPECT_TRUE(Ks1.isApprox(*pKs2))
		<< "Expected: " << endl << Ks1 << endl << endl 
		<< "Actual: " << endl << *pKs2 << endl << endl;
}

/** @brief	Kss: (Nx1) self variance matrix between the test data. */  
TEST_F(TestCaseCovSEIsoSlow, KssTest)
{
	// Expected value
	MatrixXf Kss1(4, 1);
	Kss1 << 2.250000000000000f, 
		     2.250000000000000f, 
			  2.250000000000000f, 
			  2.250000000000000f;

	// Actual value
	//MatrixXfPtr pKss2 = CovSEIsoSlow<float>::Kss(logHyp, pXs);
	MatrixXfPtr pKss2 = CovSEIsoSlow::Kss(logHyp, pXs);

	// Test
	EXPECT_TRUE(Kss1.isApprox(*pKss2))
		<< "Expected: " << endl << Kss1 << endl << endl 
		<< "Actual: " << endl << *pKss2 << endl << endl;
}
#endif