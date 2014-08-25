#ifndef _TEST_CASE_COVARIANCE_FUNCTION_MATERN_ISO_WIDE_DERIVATIVE_OBSERVATIONS_HPP_
#define _TEST_CASE_COVARIANCE_FUNCTION_MATERN_ISO_WIDE_DERIVATIVE_OBSERVATIONS_HPP_

#include "TestDerivativeDataSetting.hpp"
#include <fstream>

/**
 * @class	TestCaseCovMaternisoDerObs
 * @brief	Test fixture for testing CovMaternisoDerObs class.
 * @note		Inherits from TestDataSetting
 * 			to use the initialized training data and test positions.
 * @author	Soohwan Kim
 * @date		25/08/2014
 */
class TestCaseCovMaternisoDerObs : public TestDerivativeDataSetting
{
protected:
	/** @brief	Overloading the test fixture set up. */
	virtual void SetUp()
	{
		// Call the parent set up.
		TestDerivativeDataSetting::SetUp();

		// Set the hyperparameters.
		logHyp(0) = log(ell);
		logHyp(1) = log(sigma_f);
	}

protected:
	/** @brief Log hyperparameters: log([ell, sigma_f]). */
	CovMaternisoDerObs<TestType>::Hyp logHyp;
};

/** @brief	K: (NxN) self covariance matrix between the training data. */  
TEST_F(TestCaseCovMaternisoDerObs, KTest)
{
	// Expected value
	Matrix K1(17, 17);
	K1 <<  2.250000000000000f,  0.945903145706138f,  0.820140300147686f,  0.518379208154154f,  0.518379208154154f, -0.070248528146921f,  0.287032941082086f,  1.144693244451851f,  0.287032941082086f, -0.945791389346269f, -0.731912134487307f, -0.627070332515083f, -0.731912134487307f, -0.454543084523379f, -0.782868273326437f,  0.220327091841839f, -0.782868273326437f,
			 0.945903145706138f,  2.250000000000000f,  0.717445827143238f,  0.842397675075848f,  0.842397675075848f, -0.517845318067194f, -0.599934936345073f,  0.740483301607387f, -0.599934936345073f, -0.771935499960762f, -1.389005471559554f, -1.162636438120254f, -1.389005471559554f, -0.085419226298868f, -0.794156886981196f,  1.425519646063060f, -0.794156886981196f,
			 0.820140300147686f,  0.717445827143238f,  2.250000000000000f,  1.113145531082548f,  1.113145531082548f, -1.092861922295394f, -0.070587851743076f,  2.201064848412299f, -0.070587851743076f, -0.866136520699639f, -0.047994474191143f,  0.909319461508675f, -0.047994474191143f, -1.501963782594417f, -1.779093360157565f,  0.459102526720465f, -1.779093360157565f,
			 0.518379208154154f,  0.842397675075848f,  1.113145531082548f,  2.250000000000000f,  2.250000000000000f, -0.594155468569309f, -0.530791245643619f,  0.000000000000000f, -0.530791245643619f, -0.339291472934476f, -0.227257147261031f,  0.000000000000000f, -0.227257147261031f, -0.412225177946828f, -0.857667480911376f,  0.000000000000000f, -0.857667480911376f,
			 0.518379208154154f,  0.842397675075848f,  1.113145531082548f,  2.250000000000000f,  2.250000000000000f, -0.594155468569309f, -0.530791245643619f,  0.000000000000000f, -0.530791245643619f, -0.339291472934476f, -0.227257147261031f,  0.000000000000000f, -0.227257147261031f, -0.412225177946828f, -0.857667480911376f,  0.000000000000000f, -0.857667480911376f,
			 -0.070248528146921f, -0.517845318067194f, -1.092861922295394f, -0.594155468569309f, -0.594155468569309f,  27.000000000000000f,  1.980377514588473f, -0.749935380242812f,  1.980377514588473f,  0.000000000000000f, -3.150524800391490f, -0.874238315954504f, -3.150524800391490f,  0.000000000000000f,  3.145988262423565f, -1.062163579430354f,  3.145988262423565f,
			 0.287032941082086f, -0.599934936345073f, -0.070587851743076f, -0.530791245643619f, -0.530791245643619f,  1.980377514588473f,  27.000000000000000f,  0.183921958512266f,  27.000000000000000f, -3.150524800391490f,  0.000000000000000f, -0.404154488461174f,  0.000000000000000f,  3.145988262423565f,  0.000000000000000f, -1.525277273763264f,  0.000000000000000f,
			 1.144693244451851f,  0.740483301607387f,  2.201064848412299f,  0.000000000000000f,  0.000000000000000f, -0.749935380242812f,  0.183921958512266f,  27.000000000000000f,  0.183921958512266f, -0.874238315954504f, -0.404154488461174f,  0.000000000000000f, -0.404154488461174f, -1.062163579430354f, -1.525277273763264f,  0.000000000000000f, -1.525277273763264f,
			 0.287032941082086f, -0.599934936345073f, -0.070587851743076f, -0.530791245643619f, -0.530791245643619f,  1.980377514588473f,  27.000000000000000f,  0.183921958512266f,  27.000000000000000f, -3.150524800391490f,  0.000000000000000f, -0.404154488461174f,  0.000000000000000f,  3.145988262423565f,  0.000000000000000f, -1.525277273763264f,  0.000000000000000f,
			 -0.945791389346269f, -0.771935499960762f, -0.866136520699639f, -0.339291472934476f, -0.339291472934476f,  0.000000000000000f, -3.150524800391490f, -0.874238315954504f, -3.150524800391490f,  27.000000000000000f,  3.375370557751488f,  0.281768152103641f,  3.375370557751488f,  0.000000000000000f,  2.525672237032427f, -0.606546711132797f,  2.525672237032427f,
			 -0.731912134487307f, -1.389005471559554f, -0.047994474191143f, -0.227257147261031f, -0.227257147261031f, -3.150524800391490f,  0.000000000000000f, -0.404154488461174f,  0.000000000000000f,  3.375370557751488f,  27.000000000000000f,  0.954844104132812f,  27.000000000000000f,  2.525672237032427f,  0.000000000000000f, -0.653044233231860f,  0.000000000000000f,
			 -0.627070332515083f, -1.162636438120254f,  0.909319461508675f,  0.000000000000000f,  0.000000000000000f, -0.874238315954504f, -0.404154488461174f,  0.000000000000000f, -0.404154488461174f,  0.281768152103641f,  0.954844104132812f,  27.000000000000000f,  0.954844104132812f, -0.606546711132797f, -0.653044233231860f,  0.000000000000000f, -0.653044233231860f,
			 -0.731912134487307f, -1.389005471559554f, -0.047994474191143f, -0.227257147261031f, -0.227257147261031f, -3.150524800391490f,  0.000000000000000f, -0.404154488461174f,  0.000000000000000f,  3.375370557751488f,  27.000000000000000f,  0.954844104132812f,  27.000000000000000f,  2.525672237032427f,  0.000000000000000f, -0.653044233231860f,  0.000000000000000f,
			 -0.454543084523379f, -0.085419226298868f, -1.501963782594417f, -0.412225177946828f, -0.412225177946828f,  0.000000000000000f,  3.145988262423565f, -1.062163579430354f,  3.145988262423565f,  0.000000000000000f,  2.525672237032427f, -0.606546711132797f,  2.525672237032427f,  27.000000000000000f,  3.382649388959333f,  0.044071165213277f,  3.382649388959333f,
			 -0.782868273326437f, -0.794156886981196f, -1.779093360157565f, -0.857667480911376f, -0.857667480911376f,  3.145988262423565f,  0.000000000000000f, -1.525277273763264f,  0.000000000000000f,  2.525672237032427f,  0.000000000000000f, -0.653044233231860f,  0.000000000000000f,  3.382649388959333f,  27.000000000000000f, -1.336704091061321f,  27.000000000000000f,
			 0.220327091841839f,  1.425519646063060f,  0.459102526720465f,  0.000000000000000f,  0.000000000000000f, -1.062163579430354f, -1.525277273763264f,  0.000000000000000f, -1.525277273763264f, -0.606546711132797f, -0.653044233231860f,  0.000000000000000f, -0.653044233231860f,  0.044071165213277f, -1.336704091061321f,  27.000000000000000f, -1.336704091061321f,
			 -0.782868273326437f, -0.794156886981196f, -1.779093360157565f, -0.857667480911376f, -0.857667480911376f,  3.145988262423565f,  0.000000000000000f, -1.525277273763264f,  0.000000000000000f,  2.525672237032427f,  0.000000000000000f, -0.653044233231860f,  0.000000000000000f,  3.382649388959333f,  27.000000000000000f, -1.336704091061321f,  27.000000000000000f;

	// Actual value
	MatrixPtr pK2 = CovMaternisoDerObs<TestType>::K(logHyp, derivativeTrainingData);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(ell)]: (NxN) partial derivative of K with respect to log(ell). */  
TEST_F(TestCaseCovMaternisoDerObs, dKdlogellTest)
{
	// Expected value
	Matrix K1(17, 17);
	K1 <<  0.000000000000000f,  1.217163707948694f,  1.210665552457046f,  1.071448917416644f,  1.071448917416644f, -0.080825057852219f,  0.307619618890702f,  0.920382446240606f,  0.307619618890702f, -1.088188546814318f, -0.784406594670530f, -0.504191432423029f, -0.784406594670530f, -0.522978517444403f, -0.839017427666706f,  0.177152428168279f, -0.839017427666706f,
			 1.217163707948694f,  0.000000000000000f,  1.184529772356234f,  1.213780873349521f,  1.213780873349521f, -0.685191122741211f, -0.232834247747322f,  0.089148188516652f, -0.232834247747322f, -1.021392553815232f, -0.539071863455313f, -0.139972004955256f, -0.539071863455313f, -0.113023123951103f, -0.308211624580680f,  0.171621012743379f, -0.308211624580680f,
			 1.210665552457046f,  1.184529772356234f,  0.000000000000000f,  1.186888686348668f,  1.186888686348668f, -0.062374207835915f, -0.022302227548366f, -0.670839738001976f, -0.022302227548366f, -0.049434039428261f, -0.015163851258302f, -0.277142052292824f, -0.015163851258302f, -0.085723364705389f, -0.562104441036671f, -0.139925099873073f, -0.562104441036671f,
			 1.071448917416644f,  1.213780873349521f,  1.186888686348668f,  0.000000000000000f,  0.000000000000000f, -0.916791635692321f, -0.623942633287635f,  -0.000000000000000f, -0.623942633287635f, -0.523532309139674f, -0.267139716525556f,  -0.000000000000000f, -0.267139716525556f, -0.636070212521060f, -1.008184123074843f,  -0.000000000000000f, -1.008184123074843f,
			 1.071448917416644f,  1.213780873349521f,  1.186888686348668f,  0.000000000000000f,  0.000000000000000f, -0.916791635692321f, -0.623942633287635f,  -0.000000000000000f, -0.623942633287635f, -0.523532309139674f, -0.267139716525556f,  -0.000000000000000f, -0.267139716525556f, -0.636070212521060f, -1.008184123074843f,  -0.000000000000000f, -1.008184123074843f,
			 -0.080825057852219f, -0.685191122741211f, -0.062374207835915f, -0.916791635692321f, -0.916791635692321f, -54.000000000000000f,  2.973905910924679f,  0.373773245275823f,  2.973905910924679f,  -0.000000000000000f,  4.662490666027520f, -0.474725748280704f,  4.662490666027520f,  -0.000000000000000f, -4.655777001710735f, -0.576772249442139f, -4.655777001710735f,
			 0.307619618890702f, -0.232834247747322f, -0.022302227548366f, -0.623942633287635f, -0.623942633287635f,  2.973905910924679f, -54.000000000000000f,  1.160159437888751f, -54.000000000000000f,  4.662490666027520f,  -0.000000000000000f, -0.070927227477283f,  -0.000000000000000f, -4.655777001710735f,  -0.000000000000000f, -0.267679046629048f,  -0.000000000000000f,
			 0.920382446240606f,  0.089148188516652f, -0.670839738001976f,  -0.000000000000000f,  -0.000000000000000f,  0.373773245275823f,  1.160159437888751f, -54.000000000000000f,  1.160159437888751f, -0.474725748280704f, -0.070927227477283f,  -0.000000000000000f, -0.070927227477283f, -0.576772249442139f, -0.267679046629048f,  -0.000000000000000f, -0.267679046629048f,
			 0.307619618890702f, -0.232834247747322f, -0.022302227548366f, -0.623942633287635f, -0.623942633287635f,  2.973905910924679f, -54.000000000000000f,  1.160159437888751f, -54.000000000000000f,  4.662490666027520f,  -0.000000000000000f, -0.070927227477283f,  -0.000000000000000f, -4.655777001710735f,  -0.000000000000000f, -0.267679046629048f,  -0.000000000000000f,
			 -1.088188546814318f, -1.021392553815232f, -0.049434039428261f, -0.523532309139674f, -0.523532309139674f,  -0.000000000000000f,  4.662490666027520f, -0.474725748280704f,  4.662490666027520f, -54.000000000000000f,  0.909442859526808f,  0.934005189425139f,  0.909442859526808f,  -0.000000000000000f, -3.737765603097376f, -0.329364815125196f, -3.737765603097376f,
			 -0.784406594670530f, -0.539071863455313f, -0.015163851258302f, -0.267139716525556f, -0.267139716525556f,  4.662490666027520f,  -0.000000000000000f, -0.070927227477283f,  -0.000000000000000f,  0.909442859526808f, -54.000000000000000f,  1.295452678347112f, -54.000000000000000f, -3.737765603097376f,  -0.000000000000000f, -0.114606216695806f,  -0.000000000000000f,
			 -0.504191432423029f, -0.139972004955256f, -0.277142052292824f,  -0.000000000000000f,  -0.000000000000000f, -0.474725748280704f, -0.070927227477283f,  -0.000000000000000f, -0.070927227477283f,  0.934005189425139f,  1.295452678347112f, -54.000000000000000f,  1.295452678347112f, -0.329364815125196f, -0.114606216695806f,  -0.000000000000000f, -0.114606216695806f,
			 -0.784406594670530f, -0.539071863455313f, -0.015163851258302f, -0.267139716525556f, -0.267139716525556f,  4.662490666027520f,  -0.000000000000000f, -0.070927227477283f,  -0.000000000000000f,  0.909442859526808f, -54.000000000000000f,  1.295452678347112f, -54.000000000000000f, -3.737765603097376f,  -0.000000000000000f, -0.114606216695806f,  -0.000000000000000f,
			 -0.522978517444403f, -0.113023123951103f, -0.085723364705389f, -0.636070212521060f, -0.636070212521060f,  -0.000000000000000f, -4.655777001710735f, -0.576772249442139f, -4.655777001710735f,  -0.000000000000000f, -3.737765603097376f, -0.329364815125196f, -3.737765603097376f, -54.000000000000000f,  0.898670850189837f,  0.804931825896802f,  0.898670850189837f,
			 -0.839017427666706f, -0.308211624580680f, -0.562104441036671f, -1.008184123074843f, -1.008184123074843f, -4.655777001710735f,  -0.000000000000000f, -0.267679046629048f,  -0.000000000000000f, -3.737765603097376f,  -0.000000000000000f, -0.114606216695806f,  -0.000000000000000f,  0.898670850189837f, -54.000000000000000f,  0.893296659408563f, -54.000000000000000f,
			 0.177152428168279f,  0.171621012743379f, -0.139925099873073f,  -0.000000000000000f,  -0.000000000000000f, -0.576772249442139f, -0.267679046629048f,  -0.000000000000000f, -0.267679046629048f, -0.329364815125196f, -0.114606216695806f,  -0.000000000000000f, -0.114606216695806f,  0.804931825896802f,  0.893296659408563f, -54.000000000000000f,  0.893296659408563f,
			 -0.839017427666706f, -0.308211624580680f, -0.562104441036671f, -1.008184123074843f, -1.008184123074843f, -4.655777001710735f,  -0.000000000000000f, -0.267679046629048f,  -0.000000000000000f, -3.737765603097376f,  -0.000000000000000f, -0.114606216695806f,  -0.000000000000000f,  0.898670850189837f, -54.000000000000000f,  0.893296659408563f, -54.000000000000000f;

	// Actual value
	MatrixPtr pK2 = CovMaternisoDerObs<TestType>::K(logHyp, derivativeTrainingData, 0);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	pd[K]/pd[log(sigma_f)]: (NxN) partial derivative of K with respect to log(sigma_f). */  
TEST_F(TestCaseCovMaternisoDerObs, dKdlogsigmafTest)
{
	// Expected value
	Matrix K1(17, 17);
	K1 <<  4.500000000000000f,  1.891806291412276f,  1.640280600295373f,  1.036758416308309f,  1.036758416308309f, -0.140497056293841f,  0.574065882164172f,  2.289386488903701f,  0.574065882164172f, -1.891582778692537f, -1.463824268974614f, -1.254140665030166f, -1.463824268974614f, -0.909086169046759f, -1.565736546652874f,  0.440654183683678f, -1.565736546652874f,
			 1.891806291412276f,  4.500000000000000f,  1.434891654286475f,  1.684795350151696f,  1.684795350151696f, -1.035690636134389f, -1.199869872690146f,  1.480966603214774f, -1.199869872690146f, -1.543870999921524f, -2.778010943119108f, -2.325272876240508f, -2.778010943119108f, -0.170838452597736f, -1.588313773962392f,  2.851039292126120f, -1.588313773962392f,
			 1.640280600295373f,  1.434891654286475f,  4.500000000000000f,  2.226291062165095f,  2.226291062165095f, -2.185723844590788f, -0.141175703486152f,  4.402129696824598f, -0.141175703486152f, -1.732273041399279f, -0.095988948382286f,  1.818638923017350f, -0.095988948382286f, -3.003927565188834f, -3.558186720315130f,  0.918205053440929f, -3.558186720315130f,
			 1.036758416308309f,  1.684795350151696f,  2.226291062165095f,  4.500000000000000f,  4.500000000000000f, -1.188310937138618f, -1.061582491287239f,  0.000000000000000f, -1.061582491287239f, -0.678582945868953f, -0.454514294522061f,  0.000000000000000f, -0.454514294522061f, -0.824450355893655f, -1.715334961822752f,  0.000000000000000f, -1.715334961822752f,
			 1.036758416308309f,  1.684795350151696f,  2.226291062165095f,  4.500000000000000f,  4.500000000000000f, -1.188310937138618f, -1.061582491287239f,  0.000000000000000f, -1.061582491287239f, -0.678582945868953f, -0.454514294522061f,  0.000000000000000f, -0.454514294522061f, -0.824450355893655f, -1.715334961822752f,  0.000000000000000f, -1.715334961822752f,
			 -0.140497056293841f, -1.035690636134389f, -2.185723844590788f, -1.188310937138618f, -1.188310937138618f,  54.000000000000000f,  3.960755029176946f, -1.499870760485625f,  3.960755029176946f,  0.000000000000000f, -6.301049600782981f, -1.748476631909009f, -6.301049600782981f,  0.000000000000000f,  6.291976524847129f, -2.124327158860709f,  6.291976524847129f,
			 0.574065882164172f, -1.199869872690146f, -0.141175703486152f, -1.061582491287239f, -1.061582491287239f,  3.960755029176946f,  54.000000000000000f,  0.367843917024531f,  54.000000000000000f, -6.301049600782981f,  0.000000000000000f, -0.808308976922349f,  0.000000000000000f,  6.291976524847129f,  0.000000000000000f, -3.050554547526527f,  0.000000000000000f,
			 2.289386488903701f,  1.480966603214774f,  4.402129696824598f,  0.000000000000000f,  0.000000000000000f, -1.499870760485625f,  0.367843917024531f,  54.000000000000000f,  0.367843917024531f, -1.748476631909009f, -0.808308976922349f,  0.000000000000000f, -0.808308976922349f, -2.124327158860709f, -3.050554547526527f,  0.000000000000000f, -3.050554547526527f,
			 0.574065882164172f, -1.199869872690146f, -0.141175703486152f, -1.061582491287239f, -1.061582491287239f,  3.960755029176946f,  54.000000000000000f,  0.367843917024531f,  54.000000000000000f, -6.301049600782981f,  0.000000000000000f, -0.808308976922349f,  0.000000000000000f,  6.291976524847129f,  0.000000000000000f, -3.050554547526527f,  0.000000000000000f,
			 -1.891582778692537f, -1.543870999921524f, -1.732273041399279f, -0.678582945868953f, -0.678582945868953f,  0.000000000000000f, -6.301049600782981f, -1.748476631909009f, -6.301049600782981f,  54.000000000000000f,  6.750741115502977f,  0.563536304207282f,  6.750741115502977f,  0.000000000000000f,  5.051344474064854f, -1.213093422265595f,  5.051344474064854f,
			 -1.463824268974614f, -2.778010943119108f, -0.095988948382286f, -0.454514294522061f, -0.454514294522061f, -6.301049600782981f,  0.000000000000000f, -0.808308976922349f,  0.000000000000000f,  6.750741115502977f,  54.000000000000000f,  1.909688208265624f,  54.000000000000000f,  5.051344474064854f,  0.000000000000000f, -1.306088466463720f,  0.000000000000000f,
			 -1.254140665030166f, -2.325272876240508f,  1.818638923017350f,  0.000000000000000f,  0.000000000000000f, -1.748476631909009f, -0.808308976922349f,  0.000000000000000f, -0.808308976922349f,  0.563536304207282f,  1.909688208265624f,  54.000000000000000f,  1.909688208265624f, -1.213093422265595f, -1.306088466463720f,  0.000000000000000f, -1.306088466463720f,
			 -1.463824268974614f, -2.778010943119108f, -0.095988948382286f, -0.454514294522061f, -0.454514294522061f, -6.301049600782981f,  0.000000000000000f, -0.808308976922349f,  0.000000000000000f,  6.750741115502977f,  54.000000000000000f,  1.909688208265624f,  54.000000000000000f,  5.051344474064854f,  0.000000000000000f, -1.306088466463720f,  0.000000000000000f,
			 -0.909086169046759f, -0.170838452597736f, -3.003927565188834f, -0.824450355893655f, -0.824450355893655f,  0.000000000000000f,  6.291976524847129f, -2.124327158860709f,  6.291976524847129f,  0.000000000000000f,  5.051344474064854f, -1.213093422265595f,  5.051344474064854f,  54.000000000000000f,  6.765298777918667f,  0.088142330426553f,  6.765298777918667f,
			 -1.565736546652874f, -1.588313773962392f, -3.558186720315130f, -1.715334961822752f, -1.715334961822752f,  6.291976524847129f,  0.000000000000000f, -3.050554547526527f,  0.000000000000000f,  5.051344474064854f,  0.000000000000000f, -1.306088466463720f,  0.000000000000000f,  6.765298777918667f,  54.000000000000000f, -2.673408182122642f,  54.000000000000000f,
			 0.440654183683678f,  2.851039292126120f,  0.918205053440929f,  0.000000000000000f,  0.000000000000000f, -2.124327158860709f, -3.050554547526527f,  0.000000000000000f, -3.050554547526527f, -1.213093422265595f, -1.306088466463720f,  0.000000000000000f, -1.306088466463720f,  0.088142330426553f, -2.673408182122642f,  54.000000000000000f, -2.673408182122642f,
			 -1.565736546652874f, -1.588313773962392f, -3.558186720315130f, -1.715334961822752f, -1.715334961822752f,  6.291976524847129f,  0.000000000000000f, -3.050554547526527f,  0.000000000000000f,  5.051344474064854f,  0.000000000000000f, -1.306088466463720f,  0.000000000000000f,  6.765298777918667f,  54.000000000000000f, -2.673408182122642f,  54.000000000000000f;

	// Actual value
	MatrixPtr pK2 = CovMaternisoDerObs<TestType>::K(logHyp, derivativeTrainingData, 1);

	// Test
	TEST_MACRO::COMPARE(K1, *pK2, __FILE__, __LINE__);
}

/** @brief	Ks: (NxM) cross covariance matrix between the training data and test data. */  
TEST_F(TestCaseCovMaternisoDerObs, KsTest)
{
	// Expected value
	Matrix Ks1(17, 3);
	Ks1 << 0.408139686053389f,  0.518379208154154f,  0.424549766729713f,
			 0.218283235552083f,  0.842397675075848f,  0.699842546493200f,
			 1.119607160578329f,  1.113145531082548f,  0.736183711504438f,
			 0.376514509003230f,  2.250000000000000f,  0.392455499736434f,
			 0.376514509003230f,  2.250000000000000f,  0.392455499736434f,
			 0.100373808651576f, -0.594155468569309f, -1.713262885812514f,
			 0.352127263457594f, -0.530791245643619f,  0.000000000000000f,
			 0.842837187243444f,  0.000000000000000f,  0.530791245643619f,
			 0.352127263457594f, -0.530791245643619f,  0.000000000000000f,
			 0.210324524702135f, -0.339291472934476f, -1.375447123283683f,
			 0.323145899863267f, -0.227257147261031f,  0.000000000000000f,
			 0.527382173381396f,  0.000000000000000f,  0.227257147261031f,
			 0.323145899863267f, -0.227257147261031f,  0.000000000000000f,
			 -2.065087467290447f, -0.412225177946828f,  1.373466574488486f,
			 -0.901038473982862f, -0.857667480911376f,  0.000000000000000f,
			 -0.058162611652456f,  0.000000000000000f,  0.857667480911376f,
			 -0.901038473982862f, -0.857667480911376f,  0.000000000000000f;

	// Actual value
	MatrixPtr pKs2 = CovMaternisoDerObs<TestType>::Ks(logHyp, derivativeTrainingData, testData);

	// Test
	TEST_MACRO::COMPARE(Ks1, *pKs2, __FILE__, __LINE__);
}

/** @brief	Kss: (Nx1) self variance matrix between the test data. */  
TEST_F(TestCaseCovMaternisoDerObs, KssTest)
{
	// Expected value
	Matrix Kss1(3, 1);
	Kss1 <<   2.250000000000000f, 
		       2.250000000000000f, 
			    2.250000000000000f;

	// Actual value
	MatrixPtr pKss2 = CovMaternisoDerObs<TestType>::Kss(logHyp, testData);

	// Test
	TEST_MACRO::COMPARE(Kss1, *pKss2, __FILE__, __LINE__);
}
#endif