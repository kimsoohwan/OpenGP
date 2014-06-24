#ifndef _ISOTROPIC_AND_DIFFERENTIABLE_COVARIANCE_FUNCTION_WITH_RESPECT_TO_INPUT_COORDINATES_HPP_
#define _ISOTROPIC_AND_DIFFERENTIABLE_COVARIANCE_FUNCTION_WITH_RESPECT_TO_INPUT_COORDINATES_HPP_

#include "../data/derivativetrainingdata.hpp"
#include "isotropic.hpp"

namespace GP {

 /**
	* @class		IsotropicAndDifferentiable
	* @brief		Isotropic covariance function which is differentiable with repect to input coordinates.
	*				It means that it deal with derivative observations as well as function observations.
	*
	*				Covariance functions of partial derivatives with respective to input coordinates
	*				are partial derivatives of covariance functions with respective to input coordinates.
	*
	*				k(x, x') = k(r) = sigma_f^2 f(s), s = s(r), r = |x-x'|
	*
	*				(1) Partial derivatives with respective to input coordinates
	*
	*                      dx           dk(x, x')     dk      ds
	*              (a) k(------, dx') = ---------- = ---- * ------
	*                     dx_i             dx_i       ds     dx_i
	*               
	*                          dx'       dk(x, x')     dk      ds
	*              (b) k(dx, -------) = ---------- = ---- * --------
	*                         dx'_j        dx'_j       ds     dx'_j
	*
	*                      dx      dx'       dk(x, x')      d^2k    ds    ds      dk     d^2s
	*              (c) k(------, -------) = ------------ = ------*-----*------ + ----*-----------
	*                     dx_i    dx'_j      dx_i dx'_j     ds^2   dx_i  dx'_j    ds   dx_i dx'_j
	*
	*					which requires four components, ds/dx_i, ds/dx'_j, d^2k/ds^2 and d^2s/dx_i dx'_j
	*					as well as dk/ds which is already required in Isotropic.
	*
	*				(2) Partial derivatives with respective to input coordinates and hyperparameters for learning
	*
	*                        d
	*              (0) -------------- k(?x, ?x') = 2 * k(?x, ?x')
	*                   dlog(sigma_f)
	*               
	*                       d         dx             d^2 k(x, x')           d^2k    ds     ds      dk     d^2s
	*              (a) ---------- k(------, dx') = ---------------- = ell*(------*------*------ + ----*----------)
	*                   dlog(ell)    dx_i           dlog(ell) dx_i          ds^2   dell   dx_i     ds   dell dx_i 
	*               
	*                       d            dx'        d^2 k(x, x')           d^2k    ds     ds       dk     d^2s
	*              (b) ---------- k(x, ------) = ----------------- = ell*(------*------*------- + ----*-----------)
	*                   dlog(ell)       dx'_j     dlog(ell) dx'_j          ds^2   dell   dx'_j     ds   dell dx'_j 
	*
	*                       d         dx      dx'            d^3k    ds     ds    ds      d^2k     d^2s      ds       d^2k    ds      d^2s        d^2k    ds      d^2s        dk        d^3s
	*              (c) ---------- k(------, -------) = ell*(------*------*-----*------ + ------*----------*------- + ------*-----*------------ + ------*------*----------- + ----*-----------------)
	*                   dlog(ell)    dx_i    dx'_j           ds^3   dell   dx_i  dx'_j    ds^2   dell dx_i  dx'_j     ds^2   dx_i  dell dx'_j     ds^2   dell   dx_i dx'_j    ds   dell dx_i dx'_j
	*
	*                                                        d^3k    ds     ds    ds      d^2k      d^2s      ds        ds      d^2s          ds      d^2s         dk        d^3s
	*                                                = ell*(------*------*-----*------ + ------*(----------*------- + ------*------------ + ------*-----------) + ----*-----------------)
	*                                                        ds^3   dell   dx_i  dx'_j    ds^2    dell dx_i  dx'_j     dx_i  dell dx'_j      dell   dx_i dx'_j     ds   dell dx_i dx'_j
	*
	*					which additionally requires four components, d^2s/dell dx_i, d^2s/dell dx'_j, d^3k/ds^3 and d^3s/dell dx_i dx'_j
	*					as well as ds/dell which are already required in Isotropic.
	* @author	Soohwan Kim
	* @date		10/06/2014
	*/
template<typename Scalar, template<typename> class Cov>
class IsotropicAndDifferentiable : public Isotropic<Scalar, Cov>
{
public:
	/**
	 * @brief	K: Self covariance matrix between the training data.
	 * 			[(K), K* ]: covariance matrix of the marginal Gaussian distribution 
	 * 			[K*T, K**]
	 *          supports three calculations: K, dK_dlog(ell), and dK_dlog(sigma_f)
	 * @note		CRTP (Curiously Recursive Template Pattern)
	 *				This class takes the corresponding covariance function class as a template.
	 *				Thus, Cov<Scalar>::f(), Cov<Scalar>::s(), Cov<Scalar>::dk_ds(), and Cov<Scalar>::ds_dell()
	 *				should be accessable from this class.
	 *				In other words, they should be public, or this class and Cov<Scalar> should be friends.
	 * @param	[in] logHyp 						The log hyperparameters, log([ell, sigma_f])
	 * @param	[in] derivativeTrainingData 	The derivative training data
	 * @param	[in] pdHypIndex					(Optional) Hyperparameter index.
	 * 													It returns the partial derivatives of the covariance matrix
	 * 													with respect to this hyperparameter. 
	 * 													The partial derivatives are required for learning hyperparameters.
	 * 													(Example) pdHypIndex = 0: pd[K]/pd[log(ell)], pdHypIndex = 1: pd[K]/pd[log(sigma_f)]
	 * 													(Default = -1) K
	 * @return	An NNxNN matrix pointer
	 *				NN = N + Nd*(D+1)
	 *				N:  The number of function training data
	 *				Nd: The number of derivative training data
	 */
	static MatrixPtr K(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const int pdHypIndex = -1) 
	{
		// Output
		// K: (N + Nd*(D+1)) x (N + Nd*(D+1))
		// 
		// for example, when D = 3
		//                 |  F (N) | Fd (Nd) | Df1 (Nd) | Df2 (Nd) | Df3 (Nd) |
		// K = ----------------------------------------------------------------
		//        F   (N)  |    F-F |     F-Fd,     F-Df1,     F-Df2,      F-Df3
		//       --------------------------------------------------------------
		//        Fd  (Nd) |      - |    Fd-Fd,    Fd-Df1,    Fd-Df2,     Fd-Df3
		//        Df1 (Nd) |      - |        -,   Df1-Df1,   Df1-Df2,    Df1-Df3
		//        Df2 (Nd) |      - |        -,         -,   Df2-Df2,    Df2-Df3
		//        Df3 (Nd) |      - |        -,         -,         -,    Df3-Df3


		// constants		
		const int N		= derivativeTrainingData.N();		// number of function training data
		const int Nd	= derivativeTrainingData.Nd();	// number of derivative training data
		const int D		= derivativeTrainingData.D();		// number of dimensions
		const int NN	= N + Nd*(D+1);

		// assertion: only once at the public functions
		assert(pdHypIndex < logHyp.size());
		assert(N > 0 || Nd > 0);
		assert(D > 0);

		// memory allocation
		MatrixPtr pK(new Matrix(NN, NN));

		// fill the block matrix
		int startRow,  startCol;
		int blockRows, blockCols;
		for(int row = 0; row < D+2; row++)
		{
			// size
			if(row == 0)	{ if(N  <= 0) continue;		blockRows = N;		startRow = 0;					}
			else				{ if(Nd <= 0) break;			blockRows = Nd;	startRow = N + (row-1)*Nd;	}

			for(int col = 0; col < D+2; col++)
			{
				// size
				if(col == 0)	{ if(N  <= 0) continue;	blockCols = N;		startCol = 0;					}	
				else				{ if(Nd <= 0) break;		blockCols = Nd;	startCol = N + (col-1)*Nd;	}

				// block-wise symmetric
				if(row > col) 
				{
					pK->block(startRow, startCol, blockRows, blockCols) = pK->block(startCol, startRow, blockCols, blockRows).transpose();
					continue;
				}

				// for each row
				switch(row)
				{
				// first row
				case 0:
				{
					switch(col)
					{
					case 0:	pK->block(startRow, startCol, blockRows, blockCols) = *(k(logHyp, derivativeTrainingData.sqDist(), pdIndex));						break;
					case 1:	pK->block(startRow, startCol, blockRows, blockCols) = *(k(logHyp, derivativeTrainingData.sqDistXXd(), pdIndex));					break;
					default:	pK->block(startRow, startCol, blockRows, blockCols) = *(dk_dxj(logHyp, derivativeTrainingData.sqDistXXd(), derivativeTrainingData.deltaXXd(col-2), pdIndex));
					}
					break;
				}

				// second row
				case 1:
				{
					switch(col)
					{
					case 1:	pK->block(startRow, startCol, blockRows, blockCols) = *(k(logHyp, derivativeTrainingData.sqDistXd(), pdIndex));					break;
					default:	pK->block(startRow, startCol, blockRows, blockCols) = *(dk_dxj(logHyp, derivativeTrainingData.sqDistXd(), derivativeTrainingData.deltaXd(col-2), pdIndex));
					}
					break;
				}

				// other rows
				default:		pK->block(startRow, startCol, blockRows, blockCols) = *(d2k_dxi_dxj(logHyp, derivativeTrainingData, row-2, col-2, pdIndex));
				}
			}
		}

		return pK;
	}

	/**
	 * @brief	K*: Cross covariance matrix between the training data and test data.
	 * 			[K,    (K*)]: covariance matrix of the marginal Gaussian distribution 
	 * 			[(K*T), K**]
	 * @note		No pdHypIndex parameter is passed,
	 * 			because the partial derivatives of the covariance matrix
	 * 			is only required for learning hyperparameters.
	 * @param	[in] logHyp 						The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] derivativeTrainingData 	The derivative training data
	 * @param	[in] pXs 							The test inputs.
	 * @return	An (N + Nd*(D+1)) x M matrix pointer.
	 * 			N: the number of function training data
	 * 			Nd: the number of derivative training data
	 * 			D: the number of dimensions
	 * 			M: the number of test inputs.
	 */
	static MatrixPtr Ks(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const MatrixConstPtr pXs)
	{
		// Output
		// Ks: (N + Nd*(D+1)) x M

		// for example, when D = 3
		//                 |  Fs (M)  |
		// Ks = ------------------------
		//        F   (N)  |    F-Fs
		//        Fd  (Nd) |   Fd-Fs
		//        Df1 (Nd) |  Df1-Fs
		//        Df2 (Nd) |  Df2-Fs
		//        Df3 (Nd) |  Df3-Fs

		// constants		
		const int N		= derivativeTrainingData.N();		// number of function training data
		const int Nd	= derivativeTrainingData.Nd();	// number of derivative training data
		const int D		= derivativeTrainingData.D();		// number of dimensions
		const int M		= pXs->rows();							// number of test positions
		const int NN	= N + Nd*(D+1);

		// assertion: only once at the public functions
		assert(N > 0 || Nd > 0);
		assert(D > 0);

		// memory allocation
		MatrixPtr pK(new Matrix(NN, M));

		// fill the block matrix
		int startRow;	const int startCol  = 0;
		int blockRows;	const int blockCols = M;
		for(int row = 0; row < D+2; row++)
		{
			// size
			if(row == 0)	{ if(N  <= 0) continue;		blockRows = N;		startRow = 0;					}
			else				{ if(Nd <= 0) break;			blockRows = Nd;	startRow = N + (row-1)*Nd;	}

			// for each row
			switch(row)
			{
			// first row
			case 0:
			{
				pK->block(startRow, startCol, blockRows, blockCols) = *(k(logHyp, derivativeTrainingData.sqDist(pXs)));
				break;
			}

			// second row
			case 1:
			{
				pK->block(startRow, startCol, blockRows, blockCols) = *(k(logHyp, derivativeTrainingData.sqDistXd(pXs)));
				break;
			}

			// other rows
			default:
				MatrixPtr pSqDistXsXd(new Matrix(*derivativeTrainingData.sqDistXd(pXs)));	pSqDistXsXd->transposeInPlace();
				MatrixPtr pDeltaXsXd(new Matrix(*derivativeTrainingData.deltaXd(pXs)));		pDeltaXsXd->transposeInPlace();
				pK->block(startRow, startCol, blockRows, blockCols).noalias() = *(dk_dxj(logHyp, pSqDistXsXd, pDeltaXsXd, row-2)->transpose());
			}
		}

		return pM;
	}

protected:
	/**
	 * @brief	 dk(x, x')      d^2 k(x, x')           d^2 k(x, x')
	 *				-----------, ----------------- and ---------------------
	 *				   dx'_j      dlog(ell) dx'_j       dlog(sigma_f) dx'_j
	 *
	 *				 dk(x, x')      d^2 k(x, x')           d^2 k(x, x')
	 *				-----------, ----------------- and --------------------- are omitted due to symmetry.
	 *				    dx_i       dlog(ell) dx_i        dlog(sigma_f) dx_i
	 *				
	 *	@note		The reason dk/dx'_j is implemented rather than dk/dx_i
	 *				is that the squared distance is implemented only between function inputs and derivative inputs.
	 *				which is derivativeTrainingData.sqDistXXd().
	 *				Therefore, the second term is a derivative observation, and so is the partial coordinate, coord_j.
	 * @param	[in] logHyp 			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDist			The squared distance between inputs, i.e.) pSqDistXXd and pSqDistXd
	 * @param	[in] pDelta				The difference between inputs, i.e.) pDeltaXXd and pDeltaXd
	 * @param	[in] pdHypIndex		(Optional) Hyperparameter index.
	 * 										It returns the partial derivatives of the covariance matrix
	 * 										with respect to this hyperparameter. 
	 * 										The partial derivatives are required for learning hyperparameters.
	 * 										(Example) pdHypIndex = 0: pd[K]/pd[log(ell)], pdHypIndex = 1: pd[K]/pd[log(sigma_f)]
	 * 										(Default = -1) K
	 * @return	A matrix pointer
	 */
	static MatrixPtr dk_dxj(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta, const int pdIndex)
	{
		// derivative of K w.r.t a hyperparameter
		switch(pdHypIndex)
		{
			// derivative w.r.t log(ell)
			//      d            dx'        d^2 k(x, x')           d^2k    ds     ds       dk     d^2s
			// ---------- k(x, ------) = ----------------- = ell*(------*------*------- + ----*-----------)
			//  dlog(ell)       dx'_j     dlog(ell) dx'_j          ds^2   dell   dx'_j     ds   dell dx'_j 
			case 0:
			{
				// constants
				const Scalar ell = exp(logHyp(0)); // ell

				// memory allocation
				MatrixPtr pK(new Matrix(pSqDist->rows(), pSqDist->cols()));

				// components
				MatrixPtr pd2k_ds2		= d2k_ds2(logHyp, pSqDist);
				MatrixPtr pds_dell		= ds_dell(logHyp, pSqDist);
				MatrixPtr pds_dxj			= ds_dxj(logHyp, pSqDist, pDelta);

				MatrixPtr pdk_ds			= dk_ds(logHyp, pSqDist);
				MatrixPtr pd2s_dell_dxj	= d2s_dell_dxj(logHyp, pSqDist, pDelta);

				// calculation
				(*pK).noalias() = ell * ( *(pd2k_ds2->cwiseProduct(*(pds_dell->cwiseProduct(*pds_dxj))))
					                     + *(dk_ds->cwiseProduct(*pd2s_dell_dxj)) );

				return pK;	
			}

			// derivative w.r.t log(sigma_f)
			//      d                dx'                 dx' 
			// -------------- k(x, -------) = 2 * k(x, -------)
			//  dlog(sigma_f)       dx'_j               dx'_j 
			case 1:
			{
				// memory allocation
				MatrixPtr pK = dk_dxj(logHyp, pSqDist, pDelta);

				// dk/dlog(sigma_f) = sigma_f * k(x, x')
				(*pK).noalias() = static_cast<Scalar>(2.0) * (*pK);

				return pK;
			}
		}

		// original dK/dx_i

		//         dx'       dk(x, x')     dk      ds
		// k(dx, -------) = ---------- = ---- * --------
		//        dx'_j        dx'_j       ds     dx'_j

		// memory allocation
		MatrixPtr pK = dk_ds(logHyp, pSqDist);

		// calculation
		(*pK).noalias() = *(pK->cwiseProduct(*(ds_dxj(logHyp, pSqDist, pDelta)));

		return pK;
	}


	static MatrixPtr d2k_dxi_dxj(const Hyp &logHyp, DerivativeTrainingData<Scalar> &derivativeTrainingData, const int coord_i, const int coord_j, const int pdIndex)
	{
		// squared distance
		const MatrixConstPtr pSqDistXd	= derivativeTrainingData.sqDistXd();

		// derivative of K w.r.t a hyperparameter
		switch(pdHypIndex)
		{
			// derivative w.r.t log(ell)
			//      d         dx       dx'           d^3k    ds     ds    ds      d^2k      d^2s      ds        ds      d^2s          ds      d^2s         dk        d^3s
			// ---------- k(------, -------) = ell*(------*------*-----*------ + ------*(----------*------- + ------*------------ + ------*-----------) + ----*-----------------)
			//  dlog(ell)    dx_i    dx'_j           ds^3   dell   dx_i  dx'_j    ds^2    dell dx_i  dx'_j     dx_i  dell dx'_j      dell   dx_i dx'_j     ds   dell dx_i dx'_j
			case 0:
			{
				// constants
				const Scalar ell = exp(logHyp(0)); // ell

				// memory allocation
				MatrixPtr pK(new Matrix(pSqDistXd->rows(), pSqDist->cols()));

				// difference
				const MatrixConstPtr pDeltaXd_i	= derivativeTrainingData.deltaXd(coord_i);
				const MatrixConstPtr pDeltaXd_j	= derivativeTrainingData.deltaXd(coord_j);
				const bool fSameCoords				= coord_i == coord_j;

				// components
				MatrixPtr pd3k_ds3			= d3k_ds3(logHyp, pSqDistXd);
				MatrixPtr pds_dell			= ds_dell(logHyp, pSqDistXd);
				MatrixPtr pds_dxi				= ds_dxj(logHyp, pSqDistXd, pDeltaXd_i);
				(*pds_dxi).noalias()			= static_cast<Scalar>(-1.0) * (*pds_dxi);
				MatrixPtr pds_dxj				= ds_dxj(logHyp, pSqDistXd, pDeltaXd_j);

				MatrixPtr pd2k_ds2				= d2k_ds2(logHyp, pSqDistXd);
				MatrixPtr pd2s_dell_dxi			= d2s_dell_dxj(logHyp, pSqDistXd, pDeltaXd_i);
				(*pd2s_dell_dxi).noalias()		= static_cast<Scalar>(-1.0) * (*pd2s_dell_dxi);
				MatrixPtr pd2s_dell_dxj			= d2s_dell_dxj(logHyp, pSqDistXd, pDeltaXd_j);

				MatrixPtr pd2s_dxi_dxj			= d2s_dxi_dxj(logHyp, pSqDistXd, pDeltaXd_i, pDeltaXd_j, fSameCoords);

				MatrixPtr pdk_ds					= dk_ds(logHyp, pSqDistXd);
				MatrixPtr pd3s_dell_dxi_dxj	= d3s_dell_dxi_dxj(logHyp, pSqDistXd,  pDeltaXd_i, pDeltaXd_j, fSameCoords);

				// calculation
				(*pK).noalias() = ell * ( *(pd3k_ds3->cwiseProduct(*(pds_dell->cwiseProduct(*(pds_dxi->cwiseProduct(*pds_dxj))))))
											   + *(pd2k_ds2->cwiseProduct( *(pd2s_dell_dxi->cwiseProduct(*(pds_dxj)))
																				  + *(pds_dxi->cwiseProduct(*(pd2s_dell_dxj)))
																				  + *(pds_dell->cwiseProduct(*(pd2s_dxi_dxj))) ))
												+ *(pdk_ds->cwiseProduct(*(pd3s_dell_dxi_dxj))) );

				return pK;	
			}

			// derivative w.r.t log(sigma_f)
			//      d             dx      dx'             dx      dx' 
			// -------------- k(------, -------) = 2 * k(-----, -------)
			//  dlog(sigma_f)    dx_i    dx'_j            dx_i   dx'_j 
			case 1:
			{
				// memory allocation
				MatrixPtr pK(new Matrix(pSqDistXd->rows(), pSqDist->cols()));

				// calculation
				(*pK).noalias() = static_cast<Scalar>(2.0) * (*pK);

				return pK;
			}
		}

		// original dK/dx_i dx'_j

		//     dx      dx'       dk(x, x')      d^2k    ds    ds      dk     d^2s
		// k(------, -------) = ------------ = ------*-----*------ + ----*-----------
		//    dx_i    dx'_j      dx_i dx'_j     ds^2   dx_i  dx'_j    ds   dx_i dx'_j

		// memory allocation
		MatrixPtr pK = dk_ds(logHyp, pSqDistXd);

		// difference
		const MatrixConstPtr pDeltaXd_i	= derivativeTrainingData.deltaXd(coord_i);
		const MatrixConstPtr pDeltaXd_j	= derivativeTrainingData.deltaXd(coord_j);
		const bool fSameCoords				= coord_i == coord_j;

		// components
		MatrixPtr pd2k_ds2		= d2k_ds2(logHyp, pSqDistXd);
		MatrixPtr pds_dxi			= ds_dxi(logHyp, pSqDistXd, coord_i);
		MatrixPtr pds_dxj			= ds_dxj(logHyp, pSqDistXd, coord_j);

		MatrixPtr pdk_ds			= dk_ds(logHyp, pSqDistXd);
		MatrixPtr pd2s_dxi_dxj	= d2s_dxi_dxj(logHyp, pSqDistXd, pDeltaXd_i, pDeltaXd_j, fSameCoords);

		// calculation
		(*pK).noalias() = ( *(pd2k_ds2->cwiseProduct(*(pds_dxi->cwiseProduct(*(pds_dxj)))))
								+ *(pdk_ds->cwiseProduct(*pd2s_dxi_dxj)) );

		return pK;
	}
};

}

#endif