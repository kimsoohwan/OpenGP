#ifndef _TEST_CASE_INFERENCE_METHOD_EXACT_HPP_
#define _TEST_CASE_INFERENCE_METHOD_EXACT_HPP_

#include "TestFunctionDataSetting.hpp"

/**
 * @class	TestCaseInfExact
 * @brief	Test fixture for testing CovSEiso class.
 * @note		Inherits from TestDataSetting
 * 			to use the initialized training data and test positions.
 * @author	Soohwan Kim
 * @date		03/07/2014
 */
class TestCaseInfExact : public TestFunctionDataSetting, 
								 public InfExact<TestType, MeanZero, CovSEiso, LikGauss> // for predicted member function test
{
// define matrix and vector types
protected:	TYPE_DEFINE_MATRIX(TestType);
				TYPE_DEFINE_VECTOR(TestType);

protected:
	typedef InfExact<TestType, MeanZero, CovSEiso, LikGauss> InfExactType;

public:
	TestCaseInfExact()
		: EPS_SOLVER(static_cast<TestType>(1e-4f)),
		  EPS_SOLVER_SOLVER(static_cast<TestType>(1e-3f)) {}

protected:
	/** @brief	Overloading the test fixture set up. */
	virtual void SetUp()
	{
		// Call the parent set up.
		TestFunctionDataSetting::SetUp();

		// Set the hyperparameters.
		logHyp.cov(0) = log(ell);
		logHyp.cov(1) = log(sigma_f);
		logHyp.lik(0) = log(sigma_n);

		// Some constants
		N					= trainingData.N();

		pInvSqrtD		= invSqrtD(logHyp.lik, trainingData);
		pL1				= choleskyFactor(logHyp.cov, logHyp.lik, trainingData);
		pL2				= choleskyFactor(logHyp.cov, trainingData, pInvSqrtD);
		pY_M				= y_m(logHyp.mean, trainingData);

		pAlpha1			= alpha(pInvSqrtD, pL1, pY_M);
		pAlpha2			= alpha(logHyp.lik, pL1, pY_M);
		pQ1				= q(pInvSqrtD, pL1, pAlpha1);
		pQ2				= q(logHyp.lik, pL1, pAlpha2);
		dnlZWRTLikHyp1 = dnlZWRTLikHyp(logHyp.lik, trainingData, pQ1);
		dnlZWRTLikHyp2 = dnlZWRTLikHyp(logHyp.lik, pQ2);
	}

protected:
	/** @brief Epsilon for the Eigen solver */
	const TestType EPS_SOLVER;
	const TestType EPS_SOLVER_SOLVER;

	/** @brief Log hyperparameters: log([ell, sigma_f]). */
	InfExactType::Hyp logHyp;

	/** @brief Some constants */
	int N;
	VectorConstPtr				pInvSqrtD;
	CholeskyFactorConstPtr	pL1;
	CholeskyFactorConstPtr	pL2;
	VectorConstPtr				pY_M;
	VectorConstPtr				pAlpha1, pAlpha2;
	MatrixConstPtr				pQ1, pQ2;
	TestType						dnlZWRTLikHyp1, dnlZWRTLikHyp2;
};


/** @brief	Sqrt of noise precision vector test */  
TEST_F(TestCaseInfExact, InvSqrtDTest)
{
	// Expected value
	Vector InvSqrtD(5);
	InvSqrtD <<  9.999999999999996f, 
			       9.999999999999996f, 
			       9.999999999999996f, 
			       9.999999999999996f, 
			       9.999999999999996f;

	// Actual value
	// pInvSqrtD

	// Test
	TEST_MACRO::COMPARE(InvSqrtD, *pInvSqrtD, __FILE__, __LINE__);
}

/** @brief	Cholesky factor test */  
TEST_F(TestCaseInfExact, CholeskyFactorTest)
{
	// Expected value
	Matrix L1(5, 5);
	L1 << 15.033296378372905f, 0.000000000000000f, 0.000000000000000f, 0.000000000000000f, 0.000000000000000f, 
		   7.954233000003977f, 12.756573888848354f, 0.000000000000000f, 0.000000000000000f, 0.000000000000000f, 
		   6.878925159482144f, 2.721575651484489f, 13.087146924501544f, 0.000000000000000f, 0.000000000000000f, 
		   4.036549175942017f, 5.820105434875480f, 7.317399015148106f, 11.058404728090039f, 0.000000000000000f, 
		   4.036549175942017f, 5.820105434875480f, 7.317399015148106f, 10.967975771600516f, 1.411319454917352f;

	// Actual value
	const Matrix L21(pL1->matrixL());
	const Matrix L22(pL2->matrixL());

	// Test
	TEST_MACRO::COMPARE(L21, L22, __FILE__, __LINE__);
	TEST_MACRO::COMPARE(L1,  L21, __FILE__, __LINE__);
	TEST_MACRO::COMPARE(L1,  L22, __FILE__, __LINE__);
}

/** @brief	y-m test */  
TEST_F(TestCaseInfExact, Y_MTest)
{
	// Expected value
	Vector Y_M(5);
	Y_M <<   0.729513045504647f, 
		      0.224277070664514f, 
			   0.269054731773365f, 
			   0.673031165004119f, 
			   0.477492197726861f;

	// Actual value
	// pY_M

	// Test
	TEST_MACRO::COMPARE(Y_M, *pY_M, __FILE__, __LINE__);
}

/** @brief	alpha test */  
TEST_F(TestCaseInfExact, AlphaTest)
{
	// Expected value
	Vector Alpha(5);
	Alpha <<   0.445390157193336f,
			    -0.225664997833041f, 
			    -0.233476969822487f, 
			     9.969938572168159f, 
			    -9.583958155557667f; 

	// Actual value
	// pAlpha1, pAlpha2

	// Test
	TEST_MACRO::COMPARE(*pAlpha1, *pAlpha2, __FILE__, __LINE__);
	TEST_MACRO::COMPARE(Alpha, *pAlpha1, __FILE__, __LINE__, EPS_SOLVER); // maxAbsDiff = 1.411e-4
	TEST_MACRO::COMPARE(Alpha, *pAlpha2, __FILE__, __LINE__, EPS_SOLVER); // maxAbsDiff = 1.449e-4
}

/** @brief	Prediction test */  
TEST_F(TestCaseInfExact, PredictionTest)
{
	// Expected value
	Vector Fmu(3);
	Fmu << -0.016227926413896f, 
			  0.573331779282441f, 
			 -0.041440201369751f;

	Matrix Fs2(3, 1);
	Fs2 << 1.189715097844089f, 
			 0.004979472579145f, 
			 1.643456665149181f;

	// Actual value
	predict(logHyp, trainingData, testData, true, 0);

	// Test
	TEST_MACRO::COMPARE(Fmu, *(testData.pMu()), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(Fs2, *(testData.pSigma()), __FILE__, __LINE__);

	// Actual value
	predict(logHyp, trainingData, testData, true, 2);

	// Test
	TEST_MACRO::COMPARE(Fmu, *(testData.pMu()), __FILE__, __LINE__);
	TEST_MACRO::COMPARE(Fs2, *(testData.pSigma()), __FILE__, __LINE__);
}

/** @brief	Negative log marginalLikelihood test */  
TEST_F(TestCaseInfExact, NlZTest)
{
	// Expected value
	const TestType factor11 = 1.172651245740942f;
	const TestType factor21 = 10.575660633423778f;
	const TestType factor31 = -11.512925464970227f;
	const TestType factor41 = 4.594692666023363f;

	const TestType nlZ1		= 4.830079080217857f;

	// Actual value
	const TestType factor12 = static_cast<TestType>(0.5f) * (*pY_M).dot(*pAlpha1);
	const TestType factor22 = pL1->matrixL().nestedExpression().diagonal().array().log().sum();
	const TestType factor321 = static_cast<TestType>(N) * log(sigma_n);
	const TestType factor322 = - pInvSqrtD->array().log().sum();
	const TestType factor42 = static_cast<TestType>(N) * 0.918938533204673f;

	TestType nlZ21, nlZ22;
	negativeLogMarginalLikelihood (logHyp, trainingData, nlZ21, VectorPtr(), 1);
	negativeLogMarginalLikelihood2(logHyp, trainingData, nlZ22, VectorPtr(), 1);

	// Test
	TEST_MACRO::COMPARE(factor11,  factor12,  __FILE__, __LINE__, EPS_SOLVER); // maxAbsDiff = 1.382e-5
	TEST_MACRO::COMPARE(factor21,  factor22,  __FILE__, __LINE__);
	TEST_MACRO::COMPARE(factor321, factor322, __FILE__, __LINE__);
	TEST_MACRO::COMPARE(factor31,  factor321, __FILE__, __LINE__);
	TEST_MACRO::COMPARE(factor31,  factor322, __FILE__, __LINE__);
	TEST_MACRO::COMPARE(factor41,  factor42,  __FILE__, __LINE__);
	TEST_MACRO::COMPARE(nlZ21, nlZ22, __FILE__, __LINE__);
	TEST_MACRO::COMPARE(nlZ1,  nlZ21, __FILE__, __LINE__); // 4.8300757
	TEST_MACRO::COMPARE(nlZ1,  nlZ22, __FILE__, __LINE__); // 4.8300714
}

/** @brief	Derivatives of negative log marginalLikelihood test test */  
TEST_F(TestCaseInfExact, QTest)
{
	// Expected value
	Matrix Q(5, 5);
	Q <<   0.508037206651325f, -0.227521833525396f, -0.173304161314830f, -4.372515811653288f,  4.336597325650786f, 
		   -0.227521833525396f,  0.683391439667285f, -0.022556946036755f,  2.111529724178329f, -2.301100338511428f, 
		   -0.173304161314830f, -0.022556946036755f,  0.786045568326524f,  2.098201657931753f, -2.467182898279517f, 
		   -4.372515811653288f,  2.111529724178328f,  2.098201657931754f, -49.194400924248164f,  45.756748297698124f, 
			 4.336597325650785f, -2.301100338511430f, -2.467182898279517f,  45.756748297698131f, -41.646979718922019f;

	// Actual value
	VectorPtr pAlpha0(new Vector(5));
	(*pAlpha0) <<  0.445390157193336f,
			        -0.225664997833041f, 
			        -0.233476969822487f, 
			         9.969938572168159f, 
			        -9.583958155557667f;
	MatrixPtr pQ3 = q(pInvSqrtD, pL1, pAlpha0);
	MatrixPtr pQ4 = q(logHyp.lik, pL1, pAlpha0);

	// Test
	TEST_MACRO::COMPARE(*pQ1, *pQ2, __FILE__, __LINE__);
	TEST_MACRO::COMPARE(Q, *pQ1, __FILE__, __LINE__, EPS_SOLVER_SOLVER); // maxAbsDiff: 2.071e-3
	TEST_MACRO::COMPARE(Q, *pQ2, __FILE__, __LINE__, EPS_SOLVER_SOLVER); // maxAbsDiff: 2.151e-3
	TEST_MACRO::COMPARE(Q, *pQ3, __FILE__, __LINE__, EPS_SOLVER);			// maxAbsDiff: 7.400e-4
	TEST_MACRO::COMPARE(Q, *pQ4, __FILE__, __LINE__, EPS_SOLVER);			// maxAbsDiff: 7.362e-4
}

/** @brief	Derivatives of negative log marginalLikelihood test test */  
TEST_F(TestCaseInfExact, dnlZWRTLikHypTest)
{
	// Expected value
	TestType dnlZWRTLikHyp0 = static_cast<TestType>(-0.888639064285251f);

	// Actual value
	// dnlZWRTLikHyp1, dnlZWRTLikHyp2
	MatrixPtr pQ0(new Matrix(5, 5));
	(*pQ0) <<   0.508037206651325f, -0.227521833525396f, -0.173304161314830f, -4.372515811653288f,  4.336597325650786f, 
		   -0.227521833525396f,  0.683391439667285f, -0.022556946036755f,  2.111529724178329f, -2.301100338511428f, 
		   -0.173304161314830f, -0.022556946036755f,  0.786045568326524f,  2.098201657931753f, -2.467182898279517f, 
		   -4.372515811653288f,  2.111529724178328f,  2.098201657931754f, -49.194400924248164f,  45.756748297698124f, 
			 4.336597325650785f, -2.301100338511430f, -2.467182898279517f,  45.756748297698131f, -41.646979718922019f;
	TestType dnlZWRTLikHyp3 = dnlZWRTLikHyp(logHyp.lik, pQ0);

	// Test
	TEST_MACRO::COMPARE(dnlZWRTLikHyp0, dnlZWRTLikHyp1, __FILE__, __LINE__, EPS_SOLVER);	// maxAbsDiff: 4.041e-5
	TEST_MACRO::COMPARE(dnlZWRTLikHyp0, dnlZWRTLikHyp2, __FILE__, __LINE__, EPS_SOLVER);	// maxAbsDiff: 4.202e-5
	TEST_MACRO::COMPARE(dnlZWRTLikHyp0, dnlZWRTLikHyp3, __FILE__, __LINE__);
}

/** @brief	Derivatives of negative log marginalLikelihood test test */  
TEST_F(TestCaseInfExact, DnlZTest)
{
	// Expected value
	Vector DnlZ1(3);
	DnlZ1 << -1.511720183318730f, 
			    3.543336572803365f, 
			   -0.888639064285251f;

	// Actual value
	TestType nlZ;
	VectorPtr pDnlZ21(new Vector(3));
	VectorPtr pDnlZ22(new Vector(3));
	negativeLogMarginalLikelihood (logHyp, trainingData, nlZ, pDnlZ21, -1);
	negativeLogMarginalLikelihood2(logHyp, trainingData, nlZ, pDnlZ22, -1);

	// Test
	TEST_MACRO::COMPARE(*pDnlZ21, *pDnlZ22, __FILE__, __LINE__);
	TEST_MACRO::COMPARE(DnlZ1, *pDnlZ21, __FILE__, __LINE__, EPS_SOLVER);	// maxAbsDiff: 4.202e-5
	TEST_MACRO::COMPARE(DnlZ1, *pDnlZ22, __FILE__, __LINE__, EPS_SOLVER);	// maxAbsDiff: 4.202e-5
}

#endif