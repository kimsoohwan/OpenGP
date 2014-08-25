#define EIGEN_USE_MKL_ALL
#include "TestcasePairwiseop.hpp"
#include "TestcaseCovSEiso.hpp"
#include "TestcaseCovSEisoDerObs.hpp"
#include "TestcaseCovMaterniso.hpp"
#include "TestcaseCovMaternisoDerObs.hpp"
#include "TestcaseCovSparseiso.hpp"
#include "TestcaseCovSparseisoDerObs.hpp"
#include "TestcaseCovProd.hpp"
#include "TestcaseCholeskyFactorSolver.hpp"
#include "TestcaseInfExact.hpp"
#include "TestcaseInfExactDerObs.hpp"
#include "TestcaseGaussianProcess.hpp"
#include "TestcaseGaussianProcessDerObs.hpp"

int main(int argc, char** argv) 
{ 
	// Initialize test environment
	::testing::InitGoogleTest(&argc, argv);
		
	// Test
	int ret = RUN_ALL_TESTS(); 

	system("pause");
	return ret;
}