#ifndef _TEST_MACROS_HPP_
#define _TEST_MACROS_HPP_

// STL
#include <iostream>
#include <cmath>			// for std::abs
using namespace std;

// Google Test
#include "gtest/gtest.h"

// GP
#include "gp.h"
using namespace GP;

typedef float	TestType;
//typedef double TestType;

class TEST_MACRO
{
// define matrix, vector and cholesky factor types
protected:	TYPE_DEFINE_MATRIX(TestType);
				TYPE_DEFINE_VECTOR(TestType);

public:
	// Eigen::NumTraits<Scalar>::dummy_precision()
	// - float:  1e-5f
	// - double: 1e-12f
	inline static void COMPARE(const TestType &v1, const TestType &v2, const char* line, int line_number, const TestType eps = Eigen::NumTraits<TestType>::dummy_precision()) 
	{
		// abs difference
		TestType absDiff = abs(v1 - v2);

		// warning
		if(eps > Eigen::NumTraits<TestType>::dummy_precision())
			cerr << "Warning: eps = " << eps << endl;

		// comparison
		EXPECT_TRUE(absDiff < eps)
			<< line << "(" << line_number << "): error" << endl
			//<< "Expected: " << endl << v1 << endl << endl 
			///<< "Actual: "   << endl << v2 << endl << endl
			//<< "Abs Difference: " << endl << absDiff << endl << endl
			<< "Abs Difference: absDiff = " << absDiff << endl << endl;
	}

	inline static void COMPARE(const Vector &v1, const Vector &v2, const char* line, int line_number, const TestType eps = Eigen::NumTraits<TestType>::dummy_precision()) 
	{
		// max abs difference
		int i;
		Vector absDiff = (v1 - v2).cwiseAbs();
		TestType maxAbsDiff = absDiff.maxCoeff(&i);

		// warning
		if(eps > Eigen::NumTraits<TestType>::dummy_precision())
			cerr << "Warning: eps = " << eps << endl;

		// comparison
		EXPECT_TRUE(v1.isApprox(v2, eps))
			<< line << "(" << line_number << "): error" << endl
			//<< "Expected: " << endl << v1 << endl << endl 
			//<< "Actual: "   << endl << v2 << endl << endl
			//<< "Abs Difference: " << endl << absDiff << endl << endl
			<< "Max Abs Difference: absDiff["   << i << "] = " << maxAbsDiff << endl << endl;
	}

	inline static void COMPARE(const Matrix &m1, const Matrix &m2, const char* line, int line_number, const TestType eps = Eigen::NumTraits<TestType>::dummy_precision()) 
	{
		// max abs difference
		int i, j;
		Matrix absDiff = (m1 - m2).cwiseAbs();
		TestType maxAbsDiff = absDiff.maxCoeff(&i, &j);

		// warining
		if(eps > Eigen::NumTraits<TestType>::dummy_precision())
			cerr << "Warning: eps = " << eps << endl;

		// comparison
		EXPECT_TRUE(m1.isApprox(m2, eps))
			<< line << "(" << line_number << "): error" << endl
			//<< "Expected: " << endl << m1 << endl << endl 
			//<< "Actual: "   << endl << m2 << endl << endl
			//<< "Abs Difference: " << endl << absDiff << endl << endl
			<< "Max Abs Difference: absDiff["   << i << ", " << j << "] = " << maxAbsDiff << endl << endl;
	}
};

#endif