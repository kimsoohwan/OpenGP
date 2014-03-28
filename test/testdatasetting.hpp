#ifndef _TEST_DATA_SETTING_HPP_
#define _TEST_DATA_SETTING_HPP_

// STL
#include <iostream>
using namespace std;

// Google Test
#include "gtest/gtest.h"

// GP
#include "gp.h"
using namespace GP;

/**
 * @class	DataSetting
 * @brief	A data setting test fixture
 * @author	Soohwankim
 * @date	28/03/2014
 */
class DataSetting : public ::testing::Test
{
protected:
	// Constructor
	DataSetting() :
		pX (new MatrixXf(5, 3)),	// Training inputs,	N = 5, D = 3
	   pXs(new MatrixXf(4, 3))		// Test inputs,		M = 4, D = 3
	{
	}

	virtual void SetUp()
	{
		// Training inputs
		(*pX) << 0.118997681558377f,   0.223811939491137f,   0.890903252535799f,
				   0.498364051982143f,   0.751267059305653f,   0.959291425205444f,
					0.959743958516081f,   0.255095115459269f,   0.547215529963803f,
					0.340385726666133f,   0.505957051665142f,   0.138624442828679f,
					0.585267750979777f,   0.699076722656686f,   0.149294005559057f;

		// Test inputs
		(*pXs) << 0.257508254123736f,   0.243524968724989f,   0.251083857976031f,
			       0.840717255983663f,   0.929263623187228f,   0.616044676146639f,
					 0.254282178971531f,   0.349983765984809f,   0.473288848902729f,
					 0.814284826068816f,   0.196595250431208f,   0.351659507062997f;
	}

	//virtual void TearDown() {}

protected:
	MatrixXfPtr pX;		/// Training inputs
	MatrixXfPtr pXs;		/// Test inputs
};

#endif