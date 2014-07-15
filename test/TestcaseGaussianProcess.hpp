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
		: EPS_SOLVER(static_cast<TestType>(1e-4f)),
		  EPS_SOLVER_SOLVER(static_cast<TestType>(1e-3f)) {}

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
	const TestType EPS_SOLVER;
	const TestType EPS_SOLVER_SOLVER;

	/** @brief Log hyperparameters: log([ell, sigma_f]). */
	GPType::Hyp logHyp;
};


///** @brief	Training test: CG, DeltaFunc */  
//TEST_F(TestCaseGP, Training_CG_DeltaFunc_Test)
//{
//	// Expected value
//	const TestType ell(0.178039338440386f);
//	const TestType sigma_f(1.99551833536411f);
//	const TestType sigma_n(0.550806723661735f);
//
//	// Actual value
//	GPType::train<CG, DeltaFunc>(logHyp, trainingData);
//	std::cout << exp(logHyp.cov(0)) << "\t" << exp(logHyp.cov(1)) << "\t" << exp(logHyp.lik(0)) << std::endl;
//
//	// Test
//	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
//	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
//	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
//}
//
///** @brief	Training test: CG, GradientNorm */  
//TEST_F(TestCaseGP, Training_CG_GradientNorm_Test)
//{
//	// Expected value
//	const TestType ell(0.178039338440386f);
//	const TestType sigma_f(1.99551833536411f);
//	const TestType sigma_n(0.550806723661735f);
//
//	// Actual value
//	GPType::train<CG, GradientNorm>(logHyp, trainingData);
//
//	// Test
//	TEST_MACRO::COMPARE(ell,		exp(logHyp.cov(0)), __FILE__, __LINE__);
//	TEST_MACRO::COMPARE(sigma_f,	exp(logHyp.cov(1)), __FILE__, __LINE__);
//	TEST_MACRO::COMPARE(sigma_n,	exp(logHyp.lik(0)), __FILE__, __LINE__);
//}

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

///** @brief	Prediction test */  
//TEST_F(TestCaseGP, PredictionTest)
//{
//	// Expected value
//	Vector Fmu(101);
//	Fmu <<  0.998395143146455f,  1.109177856467577f,  1.218972146344127f,  1.326566430401895f,  1.430714326042003f,  1.530151226901907f,  1.623611563445468f,  1.709846406065347f,  1.787641060830121f,  1.855832310014746f, 
//		     1.913324961725065f,  1.959107394779877f,  1.992265815687255f,  2.011996982844361f,  2.017619197510584f,  2.008581409934477f,  1.984470340388638f,  1.945015566843907f,  1.890092581674095f,  1.819723867301702f, 
//			  1.734078083427439f,  1.633467495035041f,  1.518343799632323f,  1.389292533433548f,  1.247026249030462f,  1.092376661544666f,  0.926285956676630f,  0.749797443183541f,  0.564045715149015f,  0.370246467217282f, 
//			  0.169686080202246f, -0.036288933292164f, -0.246282561789990f, -0.458859971981563f, -0.672558219880464f, -0.885897006599120f, -1.097389504749719f, -1.305553291166325f, -1.508921424982238f, -1.706053706853589f, 
//			  -1.895548145373719f, -2.076052640849987f, -2.246276875270292f, -2.405004371375564f, -2.551104654392459f, -2.683545418450414f, -2.801404567410559f, -2.903881968239809f, -2.990310725656130f, -3.060167760997953f, 
//			  -3.113083457494153f, -3.148850119558908f, -3.167428986461489f, -3.168955541563006f, -3.153742867864422f, -3.122282819193048f, -3.075244804000295f, -3.013472015179854f, -2.937974983967891f, -2.849922387979948f, 
//			  -2.750629101614899f, -2.641541539994388f, -2.524220413660636f, -2.400321078593908f, -2.271571732766442f, -2.139749774377268f, -2.006656696052655f, -1.874091941643659f, -1.743826195920580f, -1.617574610760398f, 
//			  -1.496970492909842f, -1.383539986962294f, -1.278678282056982f, -1.183627851647690f, -1.099459202583716f, -1.027054563233949f, -0.967094881444027f, -0.920050433148806f, -0.886175263255086f, -0.865505594084217f, 
//			  -0.857862245609748f, -0.862857018520193f, -0.879902898463435f, -0.908227850381391f, -0.946891888233221f, -0.994807030073039f, -1.050759683575136f, -1.113434954534195f, -1.181442332060201f, -1.253342180142373f, 
//			  -1.327672456500028f, -1.402975086207340f, -1.477821439022766f, -1.550836394760405f, -1.620720529073157f, -1.686270010972528f, -1.746393871282412f, -1.800128375784687f, -1.846648315692819f, -1.885275108859136f, 
//			  -1.915481685385363f;
//
//
//	Matrix Fs2(101, 1);
//	Fs2 <<  0.429770526189355f,  0.366018614604934f,  0.312877234609039f,  0.269596794959717f,  0.235265559472393f,  0.208856106360815f,  0.189273590846671f,  0.175403141143387f,  0.166153899936337f,  0.160497569111093f, 
//		     0.157499793001175f,  0.156343283111870f,  0.156342197377017f,  0.156947890210169f,  0.157746699534707f,  0.158450894214314f,  0.158884240645581f,  0.158963843792116f,  0.158679971778132f,  0.158075493013373f, 
//			  0.157226360223817f,  0.156224294440999f,  0.155162486625989f,  0.154124779307879f,  0.153178447680447f,  0.152370396605471f,  0.151726347696487f,  0.151252421860746f,  0.150938431804840f,  0.150762182762429f, 
//			  0.150694128317886f,  0.150701827224180f,  0.150753779429373f,  0.150822367384215f,  0.150885775548387f,  0.150928892935848f,  0.150943310256853f,  0.150926598533458f,  0.150881097868609f,  0.150812454864967f, 
//			  0.150728129465393f,  0.150636053185629f,  0.150543568355537f,  0.150456719766382f,  0.150379913129639f,  0.150315904911447f,  0.150266049808056f,  0.150230708067336f,  0.150209706046210f,  0.150202749278837f, 
//			  0.150209706046211f,  0.150230708067335f,  0.150266049808056f,  0.150315904911446f,  0.150379913129638f,  0.150456719766383f,  0.150543568355538f,  0.150636053185629f,  0.150728129465393f,  0.150812454864967f, 
//			  0.150881097868610f,  0.150926598533458f,  0.150943310256852f,  0.150928892935848f,  0.150885775548387f,  0.150822367384214f,  0.150753779429373f,  0.150701827224180f,  0.150694128317886f,  0.150762182762430f, 
//			  0.150938431804842f,  0.151252421860747f,  0.151726347696487f,  0.152370396605471f,  0.153178447680447f,  0.154124779307879f,  0.155162486625989f,  0.156224294440998f,  0.157226360223815f,  0.158075493013376f, 
//			  0.158679971778135f,  0.158963843792120f,  0.158884240645583f,  0.158450894214316f,  0.157746699534710f,  0.156947890210170f,  0.156342197377023f,  0.156343283111871f,  0.157499793001178f,  0.160497569111093f, 
//			  0.166153899936343f,  0.175403141143392f,  0.189273590846671f,  0.208856106360817f,  0.235265559472396f,  0.269596794959720f,  0.312877234609036f,  0.366018614604943f,  0.429770526189352f,  0.504678429236414f, 
//			  0.591048622141944f;
//
//	// Actual value
//	predict(logHyp, trainingData, testData, true, 0);
//
//	// Test
//	TEST_MACRO::COMPARE(Fmu, *(testData.pMu()), __FILE__, __LINE__);
//	TEST_MACRO::COMPARE(Fs2, *(testData.pSigma()), __FILE__, __LINE__);
//
//	// Actual value
//	predict(logHyp, trainingData, testData, true, 2);
//
//	// Test
//	TEST_MACRO::COMPARE(Fmu, *(testData.pMu()), __FILE__, __LINE__);
//	TEST_MACRO::COMPARE(Fs2, *(testData.pSigma()), __FILE__, __LINE__);
//}
//
///** @brief	Negative log marginalLikelihood test */  
//TEST_F(TestCaseGP, NlZTest)
//{
//	// Expected value
//	const TestType factor11 = 1.172651245740942f;
//	const TestType factor21 = 10.575660633423778f;
//	const TestType factor31 = -11.512925464970227f;
//	const TestType factor41 = 4.594692666023363f;
//
//	const TestType nlZ1		= 4.830079080217857f;
//
//	// Actual value
//	const TestType factor12 = static_cast<TestType>(0.5f) * (*pY_M).dot(*pAlpha1);
//	const TestType factor22 = pL->matrixL().nestedExpression().diagonal().array().log().sum();
//	const TestType factor32 = - pInvSqrtD->array().log().sum();
//	const TestType factor42 = static_cast<TestType>(N) * 0.918938533204673f;
//
//	TestType nlZ2;
//	negativeLogMarginalLikelihood(logHyp, trainingData, nlZ2, VectorPtr(), 1);
//
//	// Test
//	TEST_MACRO::COMPARE(factor11, factor12, __FILE__, __LINE__, EPS_SOLVER); // maxAbsDiff = 1.382e-5
//	TEST_MACRO::COMPARE(factor21, factor22, __FILE__, __LINE__);
//	TEST_MACRO::COMPARE(factor31, factor32, __FILE__, __LINE__);
//	TEST_MACRO::COMPARE(factor41, factor42, __FILE__, __LINE__);
//	TEST_MACRO::COMPARE(nlZ1, nlZ2, __FILE__, __LINE__);
//}
//
///** @brief	Derivatives of negative log marginalLikelihood test test */  
//TEST_F(TestCaseGP, QTest)
//{
//	// Expected value
//	Matrix Q(5, 5);
//	Q <<   0.508037206651325f, -0.227521833525396f, -0.173304161314830f, -4.372515811653288f,  4.336597325650786f, 
//		   -0.227521833525396f,  0.683391439667285f, -0.022556946036755f,  2.111529724178329f, -2.301100338511428f, 
//		   -0.173304161314830f, -0.022556946036755f,  0.786045568326524f,  2.098201657931753f, -2.467182898279517f, 
//		   -4.372515811653288f,  2.111529724178328f,  2.098201657931754f, -49.194400924248164f,  45.756748297698124f, 
//			 4.336597325650785f, -2.301100338511430f, -2.467182898279517f,  45.756748297698131f, -41.646979718922019f;
//
//	// Actual value
//	VectorPtr pAlpha0(new Vector(5));
//	(*pAlpha0) <<  0.445390157193336f,
//			        -0.225664997833041f, 
//			        -0.233476969822487f, 
//			         9.969938572168159f, 
//			        -9.583958155557667f;
//	MatrixPtr pQ3 = q(pInvSqrtD, pL, pAlpha0);
//	MatrixPtr pQ4 = q(logHyp.lik, pL, pAlpha0);
//
//	// Test
//	TEST_MACRO::COMPARE(*pQ1, *pQ2, __FILE__, __LINE__);
//	TEST_MACRO::COMPARE(Q, *pQ1, __FILE__, __LINE__, EPS_SOLVER_SOLVER); // maxAbsDiff: 2.071e-3
//	TEST_MACRO::COMPARE(Q, *pQ2, __FILE__, __LINE__, EPS_SOLVER_SOLVER); // maxAbsDiff: 2.151e-3
//	TEST_MACRO::COMPARE(Q, *pQ3, __FILE__, __LINE__, EPS_SOLVER);			// maxAbsDiff: 7.400e-4
//	TEST_MACRO::COMPARE(Q, *pQ4, __FILE__, __LINE__, EPS_SOLVER);			// maxAbsDiff: 7.362e-4
//}
//
///** @brief	Derivatives of negative log marginalLikelihood test test */  
//TEST_F(TestCaseGP, dnlZWRTLikHypTest)
//{
//	// Expected value
//	TestType dnlZWRTLikHyp0 = static_cast<TestType>(-0.888639064285251f);
//
//	// Actual value
//	// dnlZWRTLikHyp1, dnlZWRTLikHyp2
//	MatrixPtr pQ0(new Matrix(5, 5));
//	(*pQ0) <<   0.508037206651325f, -0.227521833525396f, -0.173304161314830f, -4.372515811653288f,  4.336597325650786f, 
//		   -0.227521833525396f,  0.683391439667285f, -0.022556946036755f,  2.111529724178329f, -2.301100338511428f, 
//		   -0.173304161314830f, -0.022556946036755f,  0.786045568326524f,  2.098201657931753f, -2.467182898279517f, 
//		   -4.372515811653288f,  2.111529724178328f,  2.098201657931754f, -49.194400924248164f,  45.756748297698124f, 
//			 4.336597325650785f, -2.301100338511430f, -2.467182898279517f,  45.756748297698131f, -41.646979718922019f;
//	TestType dnlZWRTLikHyp3 = dnlZWRTLikHyp(logHyp.lik, pQ0);
//
//	// Test
//	TEST_MACRO::COMPARE(dnlZWRTLikHyp0, dnlZWRTLikHyp1, __FILE__, __LINE__, EPS_SOLVER);	// maxAbsDiff: 4.041e-5
//	TEST_MACRO::COMPARE(dnlZWRTLikHyp0, dnlZWRTLikHyp2, __FILE__, __LINE__, EPS_SOLVER);	// maxAbsDiff: 4.202e-5
//	TEST_MACRO::COMPARE(dnlZWRTLikHyp0, dnlZWRTLikHyp3, __FILE__, __LINE__);
//}
//
///** @brief	Derivatives of negative log marginalLikelihood test test */  
//TEST_F(TestCaseGP, DnlZTest)
//{
//	// Expected value
//	Vector DnlZ1(3);
//	DnlZ1 << -1.511720183318730f, 
//			    3.543336572803365f, 
//			   -0.888639064285251f;
//
//	// Actual value
//	TestType nlZ;
//	VectorPtr pDnlZ2(new Vector(3));
//	negativeLogMarginalLikelihood(logHyp, trainingData, nlZ, pDnlZ2, -1);
//
//	// Test
//	TEST_MACRO::COMPARE(DnlZ1, *pDnlZ2, __FILE__, __LINE__, EPS_SOLVER);	// maxAbsDiff: 4.202e-5
//}

#endif