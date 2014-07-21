/*
//#include "GP.h"
//using namespace GP;
//
//int main()
//{
//	GP<float, MeanZero, CovSEIso, InfExact> gp;
//	gp.hyp(hyp);
//	gp.setTrainingData(Training Data);
//	gp.train(10);
//	gp.predict(Test Data)
//
//	return 0;
//}

#include "../src/util/pairwiseop.hpp"
using namespace GP;

class Empty
{
};

class A
{
public:
	typedef float real;
};

class B : public A
{
public:
	real a;
};

int main()
{
	B b;
	std::cout << "Empty: " << sizeof(Empty) << std::endl;
	std::cout << "A: " << sizeof(A) << std::endl;
	std::cout << "B: " << sizeof(B) << std::endl;

	TypeTraits<float>::MatrixPtr pX(new TypeTraits<float>::Matrix(4, 2));
	TypeTraits<float>::MatrixPtr pXs(new TypeTraits<float>::Matrix(3, 2));
	(*pX) << 1, 5,
		      2, 6,
		      3, 7,
		      4, 8;
	(*pXs) << 3, 1,
		       2, 2,
		       1, 3;

	TypeTraits<float>::MatrixPtr pSqDist = PairwiseOp<float>::sqDist(pX);
	std::cout << (*pSqDist) << std::endl;

	TypeTraits<float>::MatrixPtr pSqDist2 = PairwiseOp<float>::sqDist(pX, pXs);
	std::cout << (*pSqDist2) << std::endl;

	TypeTraits<float>::MatrixPtr pDelta = PairwiseOp<float>::delta(pX, 0);
	std::cout << (*pDelta) << std::endl;

	TypeTraits<float>::MatrixPtr pDelta2 = PairwiseOp<float>::delta(pX, pXs, 1);
	std::cout << (*pDelta2) << std::endl;

	//// X: NxD
	//// X = [X1'] = [x1, y1, z1]
	////     [X2']   [x2, y2, z2]
	////     [...]   [    ...   ]
	////     [Xn']   [xn, yn, zn]
	//MatrixXf X(4, 2);
	//X << 1, 5,
	//	  2, 6,
	//	  3, 7,
	//	  4, 8;
	//std::cout << "X: " << std::endl;
	//std::cout << X << std::endl;

	//// mu: 1xD
	//// mu = [x, y, z]
	//// Matlab: mu = mean(X, 1); 
	//RowVectorXf mu(2);
	//mu.noalias() = X.colwise().mean();			
	//std::cout << "mu: " << std::endl;
	//std::cout << mu << std::endl;

	//// XX: NxD, shifted X for numerical stability
	//// XX = [XX1'] = [x1-x, y1-y, z1-z] = [xx1, yy1, zz1]
	////      [XX2']   [x2-x, y2-y, z2-z]   [xx2, yy2, zz2]
	////      [... ]   [       ...      ]   [     ...     ]
	////      [XXn']   [xn-x, yn-y, zn-z]   [xxn, yyn, zzn]
	//// Matlab: XX = bsxfun(@minus, X, mu); or
	////         XX = X - repmat(mu, N, 1);
	//MatrixXf XX(4, 2);
	//XX.noalias() = X.rowwise() - mu;
	//std::cout << "XX: " << std::endl;
	//std::cout << XX << std::endl;
 //
	//// XX2: Nx1, [XX2]i = XXi'*XXi, squared sum
	//// XX2 = [xx1^2 + yy1^2 + zz1^2]
	////       [xx2^2 + yy2^2 + zz2^2]
	////       [          ...        ]
	////       [xxn^2 + yyn^2 + zzn^2]
	//// Matlab: XX2 = sum(XX.*XX, 2);
	//MatrixXf XX2(4, 1);
	//XX2.noalias() = XX.array().square().matrix().rowwise().sum();
	//std::cout << "sum(XX.*XX, 2): " << std::endl;
	//std::cout << XX2 << std::endl;

	//// SqDist: NxN
	//// [SqDist]_ij = (Xi - Xj)'*(Xi - Xj)
	////             = (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2
	////             = (xi^2 + yi^2 + zi^2) + (xj^2 + yj^2 + zj^2)
	////               -2(xi*xj + yi*yj + zi*zj)
	//MatrixXf SqDist(4, 4);
	//SqDist.noalias() = XX2.replicate(1, 4) + XX2.transpose().replicate(4, 1) - 2*XX * XX.transpose();
	//std::cout << "sq_dist: " << std::endl;
	//std::cout << SqDist << std::endl;

	//std::cout << "repmat(sum(a.*a,1)',1,m): " << std::endl;
	//std::cout << x2y2z2.transpose().replicate(1, 4) << std::endl;

	//std::cout << "repmat(sum(b.*b,1),n,1): " << std::endl;
	//std::cout << x2y2z2.replicate(4, 1) << std::endl;

	//MatrixXf xyz(4, 4);
	//xyz = A.transpose() * A;
	//std::cout << "a'*b: " << std::endl;
	//std::cout << xyz << std::endl;

	//MatrixXf sqDist(4, 4);
	//sqDist = x2y2z2.transpose().replicate(1, 4) + x2y2z2.replicate(4, 1) - 2*A.transpose() * A;
	//std::cout << "sq_dist: " << std::endl;
	//std::cout << sqDist << std::endl;

	//MatrixXf A(2, 4);
	// A << 1, 2, 3, 4,
	//	  5, 6, 7, 8;
	// std::cout << "A: " << std::endl;
	// std::cout << A << std::endl;
	//
	// VectorXf mean(2);
	// mean = A.rowwise().mean();
	// std::cout << "mean: " << std::endl;
	// std::cout << mean << std::endl;
	//
	// A = A.colwise() - mean;
	// std::cout << "A = A - mu: " << std::endl;
	// std::cout << A << std::endl;
	//
	// MatrixXf x2y2z2(1, 4);
	// x2y2z2 = A.array().square().colwise().sum();
	// std::cout << "sum(A.*A, 2): " << std::endl;
	// std::cout << x2y2z2 << std::endl;
	//
	// std::cout << "repmat(sum(a.*a,1)',1,m): " << std::endl;
	// std::cout << x2y2z2.transpose().replicate(1, 4) << std::endl;
	//
	// std::cout << "repmat(sum(b.*b,1),n,1): " << std::endl;
	// std::cout << x2y2z2.replicate(4, 1) << std::endl;
	//
	// MatrixXf xyz(4, 4);
	// xyz = A.transpose() * A;
	// std::cout << "a'*b: " << std::endl;
	// std::cout << xyz << std::endl;
	//
	// MatrixXf sqDist(4, 4);
	// sqDist = x2y2z2.transpose().replicate(1, 4) + x2y2z2.replicate(4, 1) - 2*A.transpose() * A;
	//std::cout << "sq_dist: " << std::endl;
	//std::cout << sqDist << std::endl;

	// MatrixXf data(2,4);
	// MatrixXf means(2,2);
	//
	// / data points
	// data << 1, 23, 6, 9,
	//		  3, 11, 7, 2;
	//
	// / means
	// means << 2, 20,
	//			3, 10;
	//
	// std::cout << "Data: " << std::endl;
	// std::cout << data.replicate(2,1) << std::endl;
	//
	// VectorXf temp1(4);
	// temp1 = Eigen::Map<VectorXf>(means.data(),4);
	//
	// std::cout << "Means: " << std::endl;
	// std::cout << temp1.replicate(1,4) << std::endl;
	//
	// MatrixXf temp2(4,4);
	// temp2 = (data.replicate(2,1) - temp1.replicate(1,4));
	// std::cout << "Differences: " << std::endl;
	// std::cout << temp2 << std::endl;
	//
	// MatrixXf temp3(2,8);
	// temp3 = Eigen::Map<MatrixXf>(temp2.data(),2,8);
	// std::cout << "Remap to 2xF: " << std::endl;
	// std::cout << temp3 << std::endl;
	//
	// MatrixXf temp4(1,8);
	// temp4 = temp3.colwise().squaredNorm();
	// std::cout << "Squared norm: " << std::endl;
	// std::cout << temp4 << std::endl;//.minCoeff(&index);
	//
	// MatrixXf temp5(2,4);
	// temp5 = Eigen::Map<MatrixXf>(temp4.data(),2,4);
	//std::cout << "Squared norm result, the distances: " << std::endl;
	//std::cout << temp5.transpose() << std::endl;

	//std::cout << "Cannot get the indices: " << std::endl;
	//std::cout << temp5.transpose().colwise().minCoeff() << std::endl; // .minCoeff(&x,&y);

	system("pause");
	return 0;
}
*/