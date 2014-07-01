#include "TestcasePairwiseop.hpp"
#include "TestcaseCovSEiso.hpp"
#include "TestcaseCovSEisoDerObs.hpp"
//#include "testcasecovseisoslow.hpp"

int main(int argc, char** argv) 
{ 
	// Initialize test environment
	::testing::InitGoogleTest(&argc, argv);
		
	// Test
	int ret = RUN_ALL_TESTS(); 

	system("pause");
	return ret;
}