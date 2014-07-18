#ifndef _TEST_CASE_GAUSSIAN_PROCESS_WITH_DERIVATIVE_OBSERVATIONS_HPP_
#define _TEST_CASE_GAUSSIAN_PROCESS_WITH_DERIVATIVE_OBSERVATIONS_HPP_

#include "TestDerivativeDataSetting2.hpp"

/**
 * @class	TestCaseGPDerObs
 * @brief	Test fixture for testing CovSEiso class.
 * @note		Inherits from TestDerivativeDataSetting2
 * 			to use the initialized training data and test positions.
 * @author	Soohwan Kim
 * @date		16/07/2014
 */
class TestCaseGPDerObs : public TestDerivativeDataSetting2 
{
// define matrix and vector types
protected:	TYPE_DEFINE_MATRIX(TestType);
				TYPE_DEFINE_VECTOR(TestType);

protected:
	typedef GaussianProcess<TestType, MeanZeroDerObs, CovSEisoDerObs, LikGaussDerObs, InfExactDerObs> GPType;

public:
	TestCaseGPDerObs()
		: sigma_nd(0.2f),					// Settint the hyperparameter, sigma_nd
		  EPS_SEARCH(static_cast<TestType>(1e-3f))
	{}

protected:
	/** @brief	Overloading the test fixture set up. */
	virtual void SetUp()
	{
		// Call the parent set up.
		TestDerivativeDataSetting2::SetUp();

		// Set the hyperparameters.
		logHyp.cov(0) = log(ell);
		logHyp.cov(1) = log(sigma_f);
		logHyp.lik(0) = log(sigma_n);
		logHyp.lik(1) = log(sigma_nd);
	}

protected:
	/** @brief Epsilon for the Eigen solver */
	const TestType EPS_SEARCH;

	/** @brief Noise variance hyperparameter: sigma_nd^2 */
	const TestType sigma_nd;

	/** @brief Log hyperparameters: log([ell, sigma_f, sigma_n, sigma_nd]). */
	GPType::Hyp logHyp;
};

/** @brief	Training test: BOBYQA */  
TEST_F(TestCaseGPDerObs, Training_BOBYQA_MaxFuncEval_Test)
{
	// Expected value
	const TestType ell(0.0857795061678744f);
	const TestType sigma_f(2.89187436909207f);
	const TestType sigma_n(0.00950266014474696f);
	const TestType sigma_nd(0.0446410213159447f);

	// Actual value
	GPType::train<BOBYQA, NoStopping>(logHyp, derivativeTrainingData, 10000);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__, EPS_SEARCH);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__, EPS_SEARCH);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__, EPS_SEARCH);
	TEST_MACRO::COMPARE(sigma_nd,	exp(logHyp.lik(1)), __FILE__, __LINE__, EPS_SEARCH);
}

/** @brief	Training test: CG, DeltaFunc */  
TEST_F(TestCaseGPDerObs, Training_CG_DeltaFunc_Test)
{
	// Expected value
	const TestType ell(0.0857795061678744f);
	const TestType sigma_f(2.89187436909207f);
	const TestType sigma_n(0.00950266014474696f);
	const TestType sigma_nd(0.0446410213159447f);

	// Actual value
	GPType::train<CG, DeltaFunc>(logHyp, derivativeTrainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_nd,	exp(logHyp.lik(1)), __FILE__, __LINE__);
}

/** @brief	Training test: CG, GradientNorm */  
TEST_F(TestCaseGPDerObs, Training_CG_GradientNorm_Test)
{
	// Expected value
	const TestType ell(0.0857795061678744f);
	const TestType sigma_f(2.89187436909207f);
	const TestType sigma_n(0.00950266014474696f);
	const TestType sigma_nd(0.0446410213159447f);

	// Actual value
	GPType::train<CG, GradientNorm>(logHyp, derivativeTrainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_nd,	exp(logHyp.lik(1)), __FILE__, __LINE__);
}

/** @brief	Training test: BFGS, DeltaFunc */  
TEST_F(TestCaseGPDerObs, Training_BFGS_DeltaFunc_Test)
{
	// Expected value
	const TestType ell(0.0857795061678744f);
	const TestType sigma_f(2.89187436909207f);
	const TestType sigma_n(0.00950266014474696f);
	const TestType sigma_nd(0.0446410213159447f);

	// Actual value
	GPType::train<BFGS, DeltaFunc>(logHyp, derivativeTrainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_nd,	exp(logHyp.lik(1)), __FILE__, __LINE__);
}

/** @brief	Training test: BFGS, GradientNorm */  
TEST_F(TestCaseGPDerObs, Training_BFGS_GradientNorm_Test)
{
	// Expected value
	const TestType ell(0.0857795061678744f);
	const TestType sigma_f(2.89187436909207f);
	const TestType sigma_n(0.00950266014474696f);
	const TestType sigma_nd(0.0446410213159447f);

	// Actual value
	GPType::train<BFGS, GradientNorm>(logHyp, derivativeTrainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_nd,	exp(logHyp.lik(1)), __FILE__, __LINE__);
}

/** @brief	Training test: LBFGS, DeltaFunc */  
TEST_F(TestCaseGPDerObs, Training_LBFGS_DeltaFunc_Test)
{
	// Expected value
	const TestType ell(0.0857795061678744f);
	const TestType sigma_f(2.89187436909207f);
	const TestType sigma_n(0.00950266014474696f);
	const TestType sigma_nd(0.0446410213159447f);

	// Actual value
	GPType::train<LBFGS, DeltaFunc>(logHyp, derivativeTrainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_nd,	exp(logHyp.lik(1)), __FILE__, __LINE__);
}

/** @brief	Training test: LBFGS, GradientNorm */  
TEST_F(TestCaseGPDerObs, Training_LBFGS_GradientNorm_Test)
{
	// Expected value
	const TestType ell(0.0857795061678744f);
	const TestType sigma_f(2.89187436909207f);
	const TestType sigma_n(0.00950266014474696f);
	const TestType sigma_nd(0.0446410213159447f);

	// Actual value
	GPType::train<LBFGS, GradientNorm>(logHyp, derivativeTrainingData);

	// Test
	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(sigma_nd,	exp(logHyp.lik(1)), __FILE__, __LINE__);
}

#endif