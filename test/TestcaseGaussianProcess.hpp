#ifndef _TEST_CASE_GAUSSIAN_PROCESS_HPP_
#define _TEST_CASE_GAUSSIAN_PROCESS_HPP_

#include "TestFunctionDataSetting2.hpp"

/**
 * @class	TestCaseGP
 * @brief	Test fixture for testing GP class.
 * @note		Inherits from TestFunctionDataSetting2
 * 			to use the initialized training data and test positions.
 * @author	Soohwan Kim
 * @date		15/07/2014
 */
class TestCaseGP : public TestFunctionDataSetting2
{
// define matrix and vector types
protected:	TYPE_DEFINE_MATRIX(TestType);
				TYPE_DEFINE_VECTOR(TestType);

protected:
	typedef GaussianProcess<TestType, MeanZero, CovSEiso, LikGauss, InfExact> GPType;

public:
	TestCaseGP()
		: EPS_SEARCH(static_cast<TestType>(1e-3f))
	{}

protected:
	/** @brief	Overloading the test fixture set up. */
	virtual void SetUp()
	{
		// Call the parent set up.
		TestFunctionDataSetting2::SetUp();

		// Set the hyperparameters.
		logHyp.cov(0) = log(ell);
		logHyp.cov(1) = log(sigma_f);
		logHyp.lik(0) = log(sigma_n);
	}

protected:
	/** @brief Epsilon for the Eigen solver */
	const TestType EPS_SEARCH;

	/** @brief Log hyperparameters: log([ell, sigma_f, sigma_n]). */
	GPType::Hyp logHyp;
};

/** @brief	Training test: BOBOYA */  
TEST_F(TestCaseGP, Training_BOBOYA_MaxFuncEval_Test)
{
	// Expected value
	const TestType ell(0.178039338440386f);
	const TestType sigma_f(1.99551833536411f);
	const TestType sigma_n(0.550806723661735f);

	// Actual value
	GPType::train<BOBOYA, MaxFuncEval>(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__, EPS_SEARCH);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__, EPS_SEARCH);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__, EPS_SEARCH);
}

/** @brief	Training test: CG, DeltaFunc */  
TEST_F(TestCaseGP, Training_CG_DeltaFunc_Test)
{
	// Expected value
	const TestType ell(0.178039338440386f);
	const TestType sigma_f(1.99551833536411f);
	const TestType sigma_n(0.550806723661735f);

	// Actual value
	GPType::train<CG, DeltaFunc>(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
}

/** @brief	Training test: CG, GradientNorm */  
TEST_F(TestCaseGP, Training_CG_GradientNorm_Test)
{
	// Expected value
	const TestType ell(0.178039338440386f);
	const TestType sigma_f(1.99551833536411f);
	const TestType sigma_n(0.550806723661735f);

	// Actual value
	GPType::train<CG, GradientNorm>(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
}

/** @brief	Training test: BFGS, DeltaFunc */  
TEST_F(TestCaseGP, Training_BFGS_DeltaFunc_Test)
{
	// Expected value
	const TestType ell(0.178039338440386f);
	const TestType sigma_f(1.99551833536411f);
	const TestType sigma_n(0.550806723661735f);

	// Actual value
	GPType::train<BFGS, DeltaFunc>(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
}

/** @brief	Training test: BFGS, GradientNorm */  
TEST_F(TestCaseGP, Training_BFGS_GradientNorm_Test)
{
	// Expected value
	const TestType ell(0.178039338440386f);
	const TestType sigma_f(1.99551833536411f);
	const TestType sigma_n(0.550806723661735f);

	// Actual value
	GPType::train<BFGS, GradientNorm>(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
}

/** @brief	Training test: LBFGS, DeltaFunc */  
TEST_F(TestCaseGP, Training_LBFGS_DeltaFunc_Test)
{
	// Expected value
	const TestType ell(0.178039338440386f);
	const TestType sigma_f(1.99551833536411f);
	const TestType sigma_n(0.550806723661735f);

	// Actual value
	GPType::train<LBFGS, DeltaFunc>(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
}

/** @brief	Training test: LBFGS, GradientNorm */  
TEST_F(TestCaseGP, Training_LBFGS_GradientNorm_Test)
{
	// Expected value
	const TestType ell(0.178039338440386f);
	const TestType sigma_f(1.99551833536411f);
	const TestType sigma_n(0.550806723661735f);

	// Actual value
	GPType::train<LBFGS, GradientNorm>(logHyp, trainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
}

#endif