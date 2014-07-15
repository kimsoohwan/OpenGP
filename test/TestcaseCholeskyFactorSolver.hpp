#ifndef _TEST_CASE_CHOLESKY_FACTOR_SOLVER_HPP_
#define _TEST_CASE_CHOLESKY_FACTOR_SOLVER_HPP_

#include "TestcaseCovSEiso.hpp"

/**
 * @class	TestCaseCholeskyFactorSolver
 * @brief	Test fixture for testing PairwiseOp class
 * @author	Soohwankim
 * @date	03/07/2014
 */
class TestCaseCholeskyFactorSolver : public TestCaseCovSEiso
{
// define cholesky factor types
protected:	TYPE_DEFINE_CHOLESKYFACTOR(TestType);

public:
	TestCaseCholeskyFactorSolver()
		: sigma_n(static_cast<TestType>(0.1f)),
		  EPS_SOLVER(static_cast<TestType>(1e-4f)) {}

protected:
	/** @brief Log hyperparameters: log([ell, sigma_f]). */
	const TestType sigma_n;

	/** @brief Epsilon for the Eigen solver */
	const TestType EPS_SOLVER;
};

/** @brief	Self squared distances between the training inputs. */
TEST_F(TestCaseCholeskyFactorSolver, InverseTest)
{
	const int N = trainingData.N();

	// Expected value
	Matrix I1 = Matrix::Identity(N, N);

	// Actual value
	MatrixPtr pK = CovSEiso<TestType>::K(logHyp, trainingData);
	(*pK) += sigma_n * sigma_n * Matrix::Identity(N, N); // diagonal terms

	//CholeskyFactor L;
	//L.compute(*pK);
	CholeskyFactor L(*pK);
	if(L.info() != Eigen::ComputationInfo::Success)
	{
		switch(L.info())
		{
			case Eigen::ComputationInfo::NumericalIssue :
			{
				std::cerr << "NumericalIssue" << std::endl;
				break;
			}
			case Eigen::ComputationInfo::NoConvergence :
			{
				std::cerr << "NoConvergence" << std::endl;
				break;
			}
			case Eigen::ComputationInfo::InvalidInput :
			{
				std::cerr << "InvalidInput" << std::endl;
				break;
			}
		}
	}

	Matrix I2 = L.solve(Matrix::Identity(N, N)) * (*pK);

	// Test
	TEST_MACRO::COMPARE(I1, I2, __FILE__, __LINE__, EPS_SOLVER); // maxAbsDiff = 2.288e-5
}

#endif