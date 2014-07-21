#ifndef _PAIRWISE_OPERATIONS_HPP_
#define _PAIRWISE_OPERATIONS_HPP_

#include <assert.h>
#include "macros.h"

namespace GP{

/**
 * @class		PairwiseOp
 * @brief		Pairwise operations.
 * @ingroup		-Util
 * @author		Soohwankim
 * @date			27/03/2014
 */
template<typename Scalar>
class PairwiseOp
{
// define matrix and row vector types
protected:	TYPE_DEFINE_MATRIX(Scalar);
				TYPE_DEFINE_ROW_VECTOR(Scalar);

public:	
	/**
	 * @brief	Pairwise squared distances between N training inputs
	 * @param	pX	training inputs (NxD)
	 * 				N: The number of training inputs
	 * 				D: The number of dimensions
	 * @return	NxN matrix
	 */
	static MatrixPtr sqDist(const MatrixConstPtr pX)
	{
		// X: NxD
		// X = [X1'] = [x1, y1, z1] in 3D
		//     [X2']   [x2, y2, z2]
		//     [...]   [    ...   ]
		//     [Xn']   [xn, yn, zn]

		// Number of training data
		const int N  = pX->rows();
		const int D  = pX->cols();

		// mu: 1xD
		// mu = [x, y, z]
		// Matlab: mu = mean(X, 1); 
		RowVector mu(D);
		mu.noalias() = pX->colwise().mean();			

		// XX: NxD, shifted X for numerical stability
		// XX = [XX1'] = [x1-x, y1-y, z1-z] = [xx1, yy1, zz1]
		//      [XX2']   [x2-x, y2-y, z2-z]   [xx2, yy2, zz2]
		//      [... ]   [       ...      ]   [     ...     ]
		//      [XXn']   [xn-x, yn-y, zn-z]   [xxn, yyn, zzn]
		// Matlab: XX = bsxfun(@minus, X, mu); or
		//         XX = X - repmat(mu, N, 1);
		Matrix XX(N, D);
		XX.noalias() = pX->rowwise() - mu;

		// XX2: Nx1, [XX2]i = XXi'*XXi, squared sum
		// XX2 = [xx1^2 + yy1^2 + zz1^2]
		//       [xx2^2 + yy2^2 + zz2^2]
		//       [          ...        ]
		//       [xxn^2 + yyn^2 + zzn^2]
		// Matlab: XX2 = sum(XX.*XX, 2);
		Matrix XX2(N, 1);
		XX2.noalias() = XX.array().square().matrix().rowwise().sum();

		// SqDist: NxN
		// [SqDist]_ij = (Xi - Xj)'*(Xi - Xj)
		//             = (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2
		//             = (xi^2 + yi^2 + zi^2) + (xj^2 + yj^2 + zj^2)
		//               -2(xi*xj + yi*yj + zi*zj)
		MatrixPtr pSqDist(new Matrix(N, N));
		pSqDist->noalias() = XX2.replicate(1, N) + XX2.transpose().replicate(N, 1) - 2*XX * XX.transpose();

		return pSqDist;
	}

	/**
	 * @brief	Pairwise squared distances between N training inputs and M test inputs
	 * @param	pX		training inputs (NxD)
	 * 					N: The number of training inputs
	 * 					D: The number of dimensions
	 * @param	pXs	test inputs (MxD)
	 * 					M: The number of test inputs
	 * 					D: The number of dimensions
	 * @return	NxM matrix
	 */
	static MatrixPtr sqDist(const MatrixConstPtr pX, const MatrixConstPtr pXs)
	{
		assert(pX->cols() == pXs->cols());

		// X: NxD
		// X = [X1'] = [x1, y1, z1] in 3D
		//     [X2']   [x2, y2, z2]
		//     [...]   [    ...   ]
		//     [Xn']   [xn, yn, zn]
		//     
		// X*: MxD
		// X* = [X*1'] = [x*1, y*1, z*1] in 3D
		//      [X*2']   [x*2, y*2, z*2]
		//      [ ...]   [     ...     ]
		//      [X*m']   [x*m, y*m, z*m]

		// Number of training data
		const int N  = pX->rows();
		const int D  = pX->cols();
		const int M  = pXs->rows();

		// mu: 1xD
		// mu = [x, y, z]
		// Matlab: mu = (n/(n+m))*mean(X,1) + (m/(n+m))*mean(Xs,1);
		RowVector mu(D);
		mu.noalias() = (static_cast<Scalar>(N)/static_cast<Scalar>(N+M)) * (pX->colwise().mean())
						 + (static_cast<Scalar>(M)/static_cast<Scalar>(N+M)) * (pXs->colwise().mean());			

		// XX: NxD, shifted X for numerical stability
		// XX = [XX1'] = [x1-x, y1-y, z1-z] = [xx1, yy1, zz1]
		//      [XX2']   [x2-x, y2-y, z2-z]   [xx2, yy2, zz2]
		//      [... ]   [       ...      ]   [     ...     ]
		//      [XXn']   [xn-x, yn-y, zn-z]   [xxn, yyn, zzn]
		// Matlab: XX = bsxfun(@minus, X, mu); or
		//         XX = X - repmat(mu, N, 1);
		Matrix XX(N, D), XXs(M, D);
		XX.noalias() = pX->rowwise() - mu;
		XXs.noalias() = pXs->rowwise() - mu;

		// XX2: Nx1, [XX2]i = XXi'*XXi, squared sum
		// XX2 = [xx1^2 + yy1^2 + zz1^2]
		//       [xx2^2 + yy2^2 + zz2^2]
		//       [          ...        ]
		//       [xxn^2 + yyn^2 + zzn^2]
		// Matlab: XX2 = sum(XX.*XX, 2);
		//TypeTraits<Scalar>::Matrix XX2(N, 1), XXs2(M, 1);
		//XX2.noalias() = XX.array().square().matrix().rowwise().sum();
		//XXs2.noalias() = XXs.array().square().matrix().rowwise().sum();

		// SqDist: NxN
		// [SqDist]_ij = (Xi - Xj)'*(Xi - Xj)
		//             = (xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2
		//             = (xi^2 + yi^2 + zi^2) + (xj^2 + yj^2 + zj^2)
		//               -2(xi*xj + yi*yj + zi*zj)
		MatrixPtr pSqDist(new Matrix(N, M));
		pSqDist->noalias() = XX.array().square().matrix().rowwise().sum().replicate(1, M)						// XX2.replicate(1, M)
								 + XXs.array().square().matrix().rowwise().sum().transpose().replicate(N, 1)		// XXs2.transpose().replicate(N, 1)
								 - 2*XX * XXs.transpose();

		return pSqDist;
	}

	/**
	 * @brief	Pairwise differences between the coord-th coordinates of N training inputs
	 * @param	pX			Training inputs (NxD)
	 * 						N: The number of training inputs
	 * 						D: The number of dimensions
	 * @param	coord		Corresponding coordinate. [result]_ij = Xi_coord - Xj_coord
	 * @return	NxN matrix
	 */
	static MatrixPtr delta(const MatrixConstPtr pX, const int coord)
	{
		return delta(pX, pX, coord);
	}

	/**
	 * @brief	Pairwise differences between the coord-th coordinates of N training inputs and M test inputs
	 * @param	pX			Training inputs (NxD)
	 * 						N: The number of training inputs
	 * 						D: The number of dimensions
	 * @param	pXs		Test inputs (MxD)
	 * 						M: The number of test inputs
	 * 						D: The number of dimensions
	 * @param	coord		Corresponding coordinate. [result]_ij = Xi_coord - X'j_coord
	 * @return	NxM matrix
	 */
	static MatrixPtr delta(const MatrixConstPtr pX, const MatrixConstPtr pXs, const int coord)
	{
		assert(pX->cols() == pXs->cols());
		assert(coord >= 0 && coord < pX->cols() && coord < pXs->cols());

		// X: NxD
		// X = [X1'] = [x1, y1, z1] in 3D
		//     [X2']   [x2, y2, z2]
		//     [...]   [    ...   ]
		//     [Xn']   [xn, yn, zn]
		//     
		// X*: MxD
		// X* = [X*1'] = [x*1, y*1, z*1] in 3D
		//      [X*2']   [x*2, y*2, z*2]
		//      [ ...]   [     ...     ]
		//      [X*m']   [x*m, y*m, z*m]

		// Number of training/test data
		const int N  = pX->rows();
		const int M  = pXs->rows();

		// Delta: NxM
		// [Delta]_jk = Xj(i) - Xk(i)
		// Delta(1) = [x1-x'1, x1-x'2, ..., x1-x'm] in 3D
		//				  [x2-x'1, x2-x'2, ..., x2-x'm]
		//				  [             ...           ]
		//				  [xn-x'1, xn-x'2, ..., xn-x'm]
		MatrixPtr pDelta(new Matrix(N, M));
		pDelta->noalias() = pX->col(coord).replicate(1, M) - pXs->col(coord).transpose().replicate(N, 1);

		return pDelta;
	}

};

}

#endif