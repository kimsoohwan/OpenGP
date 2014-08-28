#ifndef _COVARIANCE_FUNCTION_WITH_DERIVATIVE_OBSERVATIONS_HPP_
#define _COVARIANCE_FUNCTION_WITH_DERIVATIVE_OBSERVATIONS_HPP_

#include "../data/DerivativeTrainingData.hpp"
#include "../data/TestData.hpp"

namespace GP{

/**
 * @class		CovDerObs
 * @brief		Host class for covariance functions dealing with derivative observations
 *					It is the host class which accept and interit a base class
 *					for each covariance function as a template parameter
 *					and use their public and protected static member functions as follows.
 *					<CENTER>
 *					Public Static Member Functions | Corresponding Covariance Functions
 *					-------------------------------|-------------------------------------
 *					+CovBase::K					| \f$\mathbf{K}(\mathbf{X}, \mathbf{X})\f$
 *					+CovBase::Ks				| \f$\mathbf{K}(\mathbf{X}, \mathbf{Z})\f$
 *					#CovBase::K_FD				| \f$\frac{\partial \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \mathbf{Z}_j}\f$
 *					#CovBase::K_DD				| \f$\frac{\partial^2 \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \mathbf{X}_i \partial \mathbf{Z}_j}\f$
 *					#CovBase::K_DF				| \f$\frac{\partial \mathbf{K}(\mathbf{X}, \mathbf{Z})}{\partial \mathbf{X}_i}\f$
 *					</CENTER>
 * @tparam		Scalar	Datatype such as float and double
 * @tparam		CovBase	Base class for each covariance function
 * @ingroup		-Cov
 * @author		Soohwan Kim
 * @date			30/06/2014
 */
template<typename Scalar, template<typename> class CovBase>
class CovDerObs : public CovBase<Scalar>
{
// define matrix types
protected:	TYPE_DEFINE_MATRIX(Scalar);

public:

	/**
	 * @brief	Self covariance matrix between the derivative training data, K(X, X) or its partial derivative
	 * @note		Only this function returns partial derivatives
	 *				since they are used for learning hyperparameters with training data.
	 * @param	[in] logHyp 							The log hyperparameters
	 * @param	[in] derivativeTrainingData 		The functional and derivative training data
	 * @param	[in] pdHypIndex						(Optional) Hyperparameter index for partial derivatives
	 * 														- pdHypIndex = -1: return \f$\mathbf{K}(\mathbf{X}, \mathbf{X})\f$ (default)
	 *															- pdHypIndex =  0: return \f$\frac{\partial \mathbf{K}}{\partial \log(l)}\f$
	 *															- pdHypIndex =  1: return \f$\frac{\partial \mathbf{K}}{\partial \log(\sigma_f)}\f$
	 * @return	An NNxNN matrix pointer\n
	 * 			NN: The number of functional and derivative training data
	 */
	static MatrixPtr K(const typename CovBase<Scalar>::Hyp	&logHyp, 
							 DerivativeTrainingData<Scalar>			&derivativeTrainingData, 
							 const int										pdHypIndex = -1)
	{
		// N: number of functional training data only
		// NN: number of functional and derivative training data
		// D: dimension of training inputs

		// output
		// K: NN by NN, NN = N + Nd*D
		// 
		// for example, when D = 3
		//                  | f(x) | df(xd)/dx_1, df(xd)/dx_2, df(xd)/dx_3
		//                  |  N   |     Nd            Nd           Nd
		// ---------------------------------------------------------------
		// f(x)        : N  |  FF  |     FD1,         FD2,         FD3
		// df(xd)/dx_1 : Nd |   -  |    D1D1,        D1D2,        D1D3  
		// df(xd)/dx_2 : Nd |   -  |      - ,        D2D2,        D2D3  
		// df(xd)/dx_3 : Nd |   -  |      - ,          - ,        D3D3

		const int D		= derivativeTrainingData.D();
		const int N		= derivativeTrainingData.N();
		const int Nd	= derivativeTrainingData.Nd();
		const int NN	= derivativeTrainingData.NN();
		const int numBlocks	= Nd > 0 ? 1 + D : 1;

		// covariance matrix
		MatrixPtr pK(new Matrix(NN, NN)); // NN by NN, NN = N + Nd*D

		// fill block matrices of FF, FD and DD in order
		for(int rowBlock = 0; rowBlock < numBlocks; rowBlock++)
		{
			// constants
			const int startRow	= rowBlock == 0 ? 0 : N + Nd*(rowBlock-1);
			const int numRows		= rowBlock == 0 ? N : Nd;

			for(int colBlock = rowBlock; colBlock < numBlocks; colBlock++)
			{
				// constants
				const int startCol	= colBlock == 0 ? 0 : N + Nd*(colBlock-1);
				const int numCols		= colBlock == 0 ? N : Nd;

				// calculate the upper triangle
				if(rowBlock == 0)
				{
					// F-F
					if(colBlock == 0)	
						pK->block(startRow, startCol, numRows, numCols) 
						= *(CovParent::K(logHyp, static_cast<TrainingData<Scalar> >(derivativeTrainingData), pdHypIndex));

					// F-D
					else					
						pK->block(startRow, startCol, numRows, numCols) 
						= *(K_FD(logHyp, derivativeTrainingData, colBlock-1, pdHypIndex));
				}
				else
				{
					// D-D
						pK->block(startRow, startCol, numRows, numCols) 
						= *(K_DD(logHyp, derivativeTrainingData, rowBlock-1, colBlock-1, pdHypIndex));
				}

				// copy its transpose
				if(rowBlock != colBlock)	
						pK->block(startCol, startRow, numCols, numRows).noalias()
						= pK->block(startRow, startCol, numRows, numCols).transpose();
			}
		}

		return pK;
	}

	/**
	 * @brief	Cross covariance matrix between the training and test data, Ks(X, Z)
	 * @param	[in] logHyp 				The log hyperparameters
	 * @param	[in] trainingData 		The training data
	 * @param	[in] testData 				The test data
	 * @return	An NNxM matrix pointer, \f$\mathbf{K}_* = \mathbf{K}(\mathbf{X}, \mathbf{Z})\f$\n
	 * 			NN: The number of functional and derivative training data\n
	 * 			M: The number of test data
	 * @todo		DerivativeTestData to predict surface normals\n
	 *				Then it returns K: NN x MM,
	 *				where NN = N + Nd*D and MM = M + Md*D.\n
	 *				In that case, Kss should be overloaded for DerivativeTestData.
	 */
	static MatrixPtr Ks(const typename CovBase<Scalar>::Hyp		&logHyp, 
							  const DerivativeTrainingData<Scalar>		&derivativeTrainingData, 
							  const TestData<Scalar>						&testData)
	{
		// N: number of functional training data only
		// NN: number of functional and derivative training data
		// D: dimension of training inputs

		// output
		// K: NN x M, NN  = N  + Nd*D
		// 
		// for example, when D = 3
		// K
		//             | f(z)
		// -------------------
		// f(x)        | FF
		// df(xd)/dx_1 | D1F
		// df(xd)/dx_2 | D2F
		// df(xd)/dx_3 | D3F

		const int D		= derivativeTrainingData.D();
		const int N		= derivativeTrainingData.N();
		const int Nd	= derivativeTrainingData.Nd();
		const int NN	= derivativeTrainingData.NN();
		const int M		= testData.M();

		const int numBlocks	= Nd > 0 ? 1 + D : 1;

		// covariance matrix
		MatrixPtr pK(new Matrix(NN, M)); // NN x M, NN = N + Nd*D

		// fill block matrices of FF, FD and DD in order
		// constants
		const int startCol	= 0;
		const int numCols		= M;
		for(int rowBlock = 0; rowBlock < numBlocks; rowBlock++)
		{
			// constants
			const int startRow	= rowBlock == 0 ? 0 : N + Nd*(rowBlock-1);
			const int numRows		= rowBlock == 0 ? N : Nd;

			// F-F
			if(rowBlock == 0)		
				pK->block(startRow, startCol, numRows, numCols)
				= *(CovParent::Ks(logHyp, static_cast<TrainingData<Scalar> >(derivativeTrainingData), testData));

			// D-F
			else						
				pK->block(startRow, startCol, numRows, numCols)
				= *(Ks_DF(logHyp, derivativeTrainingData, testData, rowBlock-1));
		}

		return pK;
	}
};

}

#endif