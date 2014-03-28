#ifndef _PAIRWISE_OPERATION_TEST_HPP_
#define _PAIRWISE_OPERATION_TEST_HPP_

#include "testdatasetting.hpp"

/**
 * @class	PairwiseOpTestCase
 * @brief	A pairwise operation test fixture.
 *
 * @author	Soohwankim
 * @date	28/03/2014
 */
class PairwiseOpTestCase : public DataSetting
{
};

/** @brief	Self squared distances between the training inputs. */
TEST_F(PairwiseOpTestCase, SelfSqDistTest)
{
	// Expected value
	MatrixXf SqDist1(5, 5);
	SqDist1 <<                0.f,   0.426804688588141f,   0.825954189963619f,   0.694541938374676f,   0.993268666957043f, 
		        0.426804688588141f,                  0.f,   0.628864559452765f,   0.758628447081624f,   0.666371903972809f,
				  0.825954189963619f,   0.628864559452765f,                  0.f,   0.613483006883516f,   0.495693637124971f,
				  0.694541938374676f,   0.758628447081624f,   0.613483006883516f,                  0.f,   0.097376252724688f,
				  0.993268666957043f,   0.666371903972809f,   0.495693637124971f,   0.097376252724688f,                  0.f;

	// Actual value
	MatrixXfPtr pSqDist2 = PairwiseOp<float>::sqDist(pX);

	// Test
	EXPECT_TRUE(SqDist1.isApprox(*pSqDist2))
		<< "Expected: " << endl << SqDist1 << endl << endl 
		<< "Actual: " << endl << *pSqDist2 << endl << endl;
}

/** @brief	Cross squared distances between the training inputs and test inputs. */
TEST_F(PairwiseOpTestCase, CrossSqDistTest)
{
	// Expected value
	MatrixXf SqDist1(5, 4);
	SqDist1 << 0.428942639888786f,   1.094088459152940f,   0.208623015162883f,   0.774948778516402f,
		        0.817371504190236f,   0.266706823783192f,   0.456802746403667f,   0.776683299005042f,
				  0.580962819973139f,   0.473407984092142f,   0.512145332569427f,   0.062822751513393f,
				  0.088386393666608f,   0.657450171904375f,   0.143741751464412f,   0.365669019070698f,
				  0.325314862344639f,   0.336096647276545f,   0.336389999788830f,   0.345888246811163f;

	// Actual value
	MatrixXfPtr pSqDist2 = PairwiseOp<float>::sqDist(pX, pXs);

	// Test
	EXPECT_TRUE(SqDist1.isApprox(*pSqDist2))
		<< "Expected: " << endl << SqDist1 << endl << endl 
		<< "Actual: " << endl << *pSqDist2 << endl << endl;
}

/** @brief	Self differences between the training inputs. */
TEST_F(PairwiseOpTestCase, SelfDeltaTest)
{
	// Expected value
	MatrixXf SqDist1(5, 5);
	SqDist1 <<                0.f,  -0.379366370423766f,  -0.840746276957704f,  -0.221388045107756f,  -0.466270069421400f,
              0.379366370423766f,                  0.f,  -0.461379906533938f,   0.157978325316010f,  -0.086903698997634f,
				  0.840746276957704f,   0.461379906533938f,                  0.f,   0.619358231849948f,   0.374476207536304f,
				  0.221388045107756f,  -0.157978325316010f,  -0.619358231849948f,                  0.f,  -0.244882024313644f,
				  0.466270069421400f,   0.086903698997634f,  -0.374476207536304f,   0.244882024313644f,                  0.f;

	// Actual value
	MatrixXfPtr pSqDist2 = PairwiseOp<float>::delta(pX, 0);

	// Test
	EXPECT_TRUE(SqDist1.isApprox(*pSqDist2))
		<< "Expected: " << endl << SqDist1 << endl << endl 
		<< "Actual: " << endl << *pSqDist2 << endl << endl;
}

/** @brief	Cross differences between the training inputs and test inputs. */
TEST_F(PairwiseOpTestCase, CrossDeltaTest)
{
	// Expected value
	MatrixXf SqDist1(5, 4);
	SqDist1 << -0.138510572565359f,  -0.721719574425286f,  -0.135284497413154f,  -0.695287144510439f,
		         0.240855797858407f,  -0.342353204001520f,   0.244081873010612f,  -0.315920774086673f,
					0.702235704392345f,   0.119026702532418f,   0.705461779544550f,   0.145459132447265f,
					0.082877472542397f,  -0.500331529317530f,   0.086103547694602f,  -0.473899099402683f,
					0.327759496856041f,  -0.255449505003886f,   0.330985572008246f,  -0.229017075089039f;


	// Actual value
	MatrixXfPtr pSqDist2 = PairwiseOp<float>::delta(pX, pXs, 0);

	// Test
	EXPECT_TRUE(SqDist1.isApprox(*pSqDist2))
		<< "Expected: " << endl << SqDist1 << endl << endl 
		<< "Actual: " << endl << *pSqDist2 << endl << endl;
}

#endif