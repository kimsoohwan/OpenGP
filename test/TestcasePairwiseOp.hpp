#ifndef _TEST_CASE_PAIRWISE_OPERATION_HPP_
#define _TEST_CASE_PAIRWISE_OPERATION_HPP_

#include "TestDataSetting.hpp"

/**
 * @class	TestCasePairwiseOp
 * @brief	Test fixture for testing PairwiseOp class
 * @author	Soohwankim
 * @date	28/03/2014
 */
class TestCasePairwiseOp : public TestDataSetting
{
};

/** @brief	Self squared distances between the training inputs. */
TEST_F(TestCasePairwiseOp, SelfSqDistTest)
{
	// Expected value
	MatrixXf SqDist1(5, 5);
	SqDist1 << 0.000000000000000f, 0.316064331387029f, 0.388685244823581f, 0.655221369986129f, 0.655221369986129f, 
				  0.316064331387029f, 0.000000000000000f, 0.461296640078504f, 0.374671815994507f, 0.374671815994507f, 
				  0.388685244823581f, 0.461296640078504f, 0.000000000000000f, 0.239481007558240f, 0.239481007558240f, 
				  0.655221369986129f, 0.374671815994507f, 0.239481007558240f, 0.000000000000000f, 0.000000000000000f, 
				  0.655221369986129f, 0.374671815994507f, 0.239481007558240f, 0.000000000000000f, 0.000000000000000f;

	// Actual value
	MatrixXfPtr pSqDist2 = PairwiseOp<float>::sqDist(pX);

	// Test
	EXPECT_TRUE(SqDist1.isApprox(*pSqDist2))
		<< "Expected: " << endl << SqDist1 << endl << endl 
		<< "Actual: " << endl << *pSqDist2 << endl << endl;
}

/** @brief	Cross squared distances between the training inputs and test inputs. */
TEST_F(TestCasePairwiseOp, CrossSqDistTest)
{
	// Expected value
	MatrixXf SqDist1(5, 3);
	SqDist1 << 0.813212310893605f, 0.655221369986129f, 0.786289850956120f, 
				  1.285621813682207f, 0.374671815994507f, 0.475251468421454f, 
				  0.236885435319994f, 0.239481007558240f, 0.446968677187794f, 
				  0.869359622041533f, 0.000000000000000f, 0.840314218724777f, 
				  0.869359622041533f, 0.000000000000000f, 0.840314218724777f;

	// Actual value
	MatrixXfPtr pSqDist2 = PairwiseOp<float>::sqDist(pX, pXs);

	// Test
	EXPECT_TRUE(SqDist1.isApprox(*pSqDist2))
		<< "Expected: " << endl << SqDist1 << endl << endl 
		<< "Actual: " << endl << *pSqDist2 << endl << endl;
}

/** @brief	Self differences between the training inputs. */
TEST_F(TestCasePairwiseOp, SelfDeltaTest)
{
	// Expected value
	MatrixXf SqDist1(5, 5);
	SqDist1 << 0.000000000000000f, 0.471438784545539f, 0.255898902573805f, 0.700012351173950f, 0.700012351173950f, 
				  -0.471438784545539f, 0.000000000000000f, -0.215539881971734f, 0.228573566628411f, 0.228573566628411f, 
				  -0.255898902573805f, 0.215539881971734f, 0.000000000000000f, 0.444113448600145f, 0.444113448600145f, 
				  -0.700012351173950f, -0.228573566628411f, -0.444113448600145f, 0.000000000000000f, 0.000000000000000f, 
				  -0.700012351173950f, -0.228573566628411f, -0.444113448600145f, 0.000000000000000f, 0.000000000000000f;

	// Actual value
	MatrixXfPtr pSqDist2 = PairwiseOp<float>::delta(pX, 0);

	// Test
	EXPECT_TRUE(SqDist1.isApprox(*pSqDist2))
		<< "Expected: " << endl << SqDist1 << endl << endl 
		<< "Actual: " << endl << *pSqDist2 << endl << endl;
}

/** @brief	Cross differences between the training inputs and test inputs. */
TEST_F(TestCasePairwiseOp, CrossDeltaTest)
{
	// Expected value
	MatrixXf SqDist1(5, 3);
	SqDist1 << -0.089050874652647f, 0.700012351173950f, 0.229403502589646f, 
				  -0.560489659198186f, 0.228573566628411f, -0.242035281955893f, 
				  -0.344949777226452f, 0.444113448600145f, -0.026495399984159f, 
				  -0.789063225826597f, 0.000000000000000f, -0.470608848584304f, 
				  -0.789063225826597f, 0.000000000000000f, -0.470608848584304f;

	// Actual value
	MatrixXfPtr pSqDist2 = PairwiseOp<float>::delta(pX, pXs, 0);

	// Test
	EXPECT_TRUE(SqDist1.isApprox(*pSqDist2))
		<< "Expected: " << endl << SqDist1 << endl << endl 
		<< "Actual: " << endl << *pSqDist2 << endl << endl;
}

#endif