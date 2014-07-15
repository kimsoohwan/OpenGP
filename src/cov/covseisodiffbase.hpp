#ifndef _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_DERIVATIVE_OBSERVATIONS_BASE_HPP_
#define _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_DERIVATIVE_OBSERVATIONS_BASE_HPP_

#include "../util/macros.h"
#include "../data/trainingdata.hpp"
#include "covseisobase.hpp"

namespace GP {

/**
	* @class		CovSEIsoDiffBase
	* @brief		Base class for the isotropic and differentiable squared exponential covariance function.
	*
	*				It defines component functions used in class: IsotropicAndDifferentiable.
	*				It is rather slow in computation but easy to fix bugs.
	*				So, it can be used as a reference for a faster version.
	*
	* 				k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	*
	*				It defines f(s) which is required for calculating k(x, x'),
	*				            d^2k     ds      ds              d2s 
	*				It defines ------, ------, ------- and  ------------ for calculating k(x, x'),
	*				            ds^2    dx_i    dx'_j        dx_i dx'_j
	*
	*				     d^3k       d^2s         d^2s                 d^3s
	*				and ------, -----------, ------------- and ----------------- for for dk/dlog(ell), dk/dlog(sigma_f).
	*				     ds^3    dell dx_i    dell dx'_j        dell dx_i dx'_j
	*				For details, please refer to class: isotropic.
	* @note		CRTP (Curiously Recursive Template Pattern)
	*				Isotropic<Scalar, CovSEIsoBase> class the component functions.
	*				Thus, they should be public, or class: IsotropicAndDifferentiable is a friend of this class.
	* @author	Soohwan Kim
	* @date		11/06/2014
	*/
template<typename Scalar>
class CovSEIsoDiffBase : public CovSEIsoBase<Scalar>
{
// component functions
protected:

/**************************************************************************************/
/* [1] For calculaing derivatives w.r.t input coords,                                 */
/*                                                                                    */
/*		  d^2k     ds      ds              d2s                                          */
/*     ------, ------, ------- and  ------------ are required,                        */
/*      ds^2    dx_i    dx'_j        dx_i dx'_j                                       */
/*                                                                                    */
/*     which will be used in class: IsotropicAndDifferentiable                        */
/**************************************************************************************/

	/**
	 * @brief	 d^2k
	 *          ------ = k
	 *           ds^2
	 * @note		k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	 * @param	[in] logHyp			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDist		The squared distance between inputs, i.e.) pSqDist, pSqDistXXd and pSqDistXd
	 * @return	A matrix pointer.
	 */
	static MatrixPtr d2k_ds2(const Hyp &logHyp, const MatrixConstPtr pSqDist) 
	{
		return k(logHyP, pSqDist);
	}

	///**
	// * @brief	  ds        x_i - x'_i        ds
	// *          ------ = - ------------ = - -------
	// *           dx_i         ell^2          dx'_i
	// * @note		k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	// * @param	[in] logHyp		The log hyperparameters, log([ell, sigma_f]).
	// * @param	[in] pSqDist	The squared distance between inputs, i.e.) pSqDistXXd and pSqDistXd
	// * @param	[in] pDelta		The difference between inputs, i.e.) pDeltaXXd and pDeltaXd
	// * @return	A matrix pointer.
	// */
	//static MatrixPtr ds_dxi(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta) 
	//{
	//	// memory allocation
	//	MatrixPtr pK = ds_dxj(logHyp, pSqDist, pDelta);

	//	// ds/dx_i = - ds/dx'_j
	//	pK->noalias() *= static_cast<Scalar>(-1.0);

	//	return pK;
	//}

	/**
	 * @brief	   ds      x_j - x'_j        ds
	 *          ------- = ------------ = - ------
	 *           dx'_j       ell^2          dx_j
	 * @note		k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	 * @param	[in] logHyp		The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDist	The squared distance between inputs, i.e.) pSqDistXXd and pSqDistXd
	 * @param	[in] pDelta		The difference between inputs, i.e.) pDeltaXXd and pDeltaXd
	 * @return	A matrix pointer.
	 */
	static MatrixPtr ds_dxj(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta) 
	{
		// constants
		const Scalar inv_ell2 = exp(static_cast<Scalar>(-2.0) * logHyp(0));	// 1/ell^2

		// memory allocation
		MatrixPtr pK(new Matrix(*pDelta));

		// ds/dx_i = (1/ell^2) * (x_i - x'_i)
		pK->noalias() *= inv_ell2;

		return pK;
	}

	/**
	 * @brief	    d^2s        d(i, j)
	 *          ------------ = ---------
	 *           dx_i dx'_j      ell^2
	 * @param	[in] logHyp				The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDistXd			The squared distance between derivative inputs
	 * @param	[in] pDeltaXd_i		The i-th coordinate difference between derivative inputs
	 * @param	[in] pDeltaXd_j		The j-th coordinate difference between derivative inputs
	 * @param	[in] fSameCoords		true: i == j, false: i != j
	 * @return	An NdxNd matrix pointer.
	 *				Nd: The number of derivative training data
	 */
	static MatrixPtr d2s_dxi_dxj(const Hyp &logHyp, const MatrixConstPtr pSqDistXd, const MatrixConstPtr pDeltaXd_i, const MatrixConstPtr pDeltaXd_j, const bool fSameCoords) 
	{
		// memory allocation
		MatrixPtr pK(new Matrix(pSqDistXd->rows(), pSqDistXd->cols()));

		// d(i, j)
		if(fSameCoords)
		{
			// constants
			const Scalar inv_ell2 = exp(static_cast<Scalar>(-2.0) * logHyp(0));	// 1/ell^2

			// d^2s / dx_i dx'_j = 1/ell^2
			pK->setConstant(inv_ell2);
		}
		else
		{
			// d^2s / dx_i dx'_j = 0
			pK->setZero();
		}

		return pK;
	}


/**************************************************************************************/
/* [2] For calculaing derivative w.r.t input coords and hyperparameters,              */
/*                                                                                    */
/*		   d^3k       d^2s         d^2s                 d^3s                            */
/*      ------, -----------, ------------- and ----------------- are required,        */
/*       ds^3    dell dx_i    dell dx'_j        dell dx_i dx'_j                       */
/*                                                                                    */
/*     which will be used in class: IsotropicAndDifferentiable                        */
/**************************************************************************************/

	/**
	 * @brief	 d^3k
	 *          ------ = k
	 *           ds^3
	 * @note		k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	 * @param	[in] logHyp			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDistXd		The squared distance between derivative inputs
	 * @return	An NdxNd matrix pointer
	 *				Nd: The number of derivative training data
	 */
	static MatrixPtr d3k_ds3(const Hyp &logHyp, const MatrixConstPtr pSqDistXd) 
	{
		return k(logHyP, pSqDistXd);
	}

	///**
	// * @brief	    d^2s           2       ds
	// *          ------------ = - ----- * -------
	// *           dell dx_i        ell      dx_i
	// * @note		k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	// * @param	[in] logHyp			The log hyperparameters, log([ell, sigma_f]).
	// * @param	[in] pSqDist	The squared distance between inputs, i.e.) pSqDistXXd and pSqDistXd
	// * @param	[in] pDelta		The difference between inputs, i.e.) pDeltaXXd and pDeltaXd
	// * @return	A matrix pointer.
	// */
	//static MatrixPtr d2s_dell_dxi(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta) 
	//{
	//	// constants
	//	const Scalar negative_double_inv_ell = static_cast<Scalar>(-2.0) 
	//													 * exp(static_cast<Scalar>(-1.0) * logHyp(0));	// -2/ell

	//	// memory allocation
	//	MatrixPtr pK(*ds_dxi(logHyp, pSqDist, pDelta));

	//	// d^2s/dell dx'_j = (-2/ell) * ds/dx'_j
	//	pK->noalias() *= negative_double_inv_ell;

	//	return pK;
	//}

	/**
	 * @brief	    d^2s           2       ds           d^2s
	 *          ------------ = - ----- * ------- = - -----------
	 *           dell dx'_j       ell     dx'_j       dell dx_j
	 * @note		k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	 * @param	[in] logHyp			The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDist	The squared distance between inputs, i.e.) pSqDistXXd and pSqDistXd
	 * @param	[in] pDelta		The difference between inputs, i.e.) pDeltaXXd and pDeltaXd
	 * @return	A matrix pointer.
	 */
	static MatrixPtr d2s_dell_dxj(const Hyp &logHyp, const MatrixConstPtr pSqDist, const MatrixConstPtr pDelta) 
	{
		// constants
		const Scalar negative_double_inv_ell = static_cast<Scalar>(-2.0) 
														 * exp(static_cast<Scalar>(-1.0) * logHyp(0));	// -2/ell

		// memory allocation
		MatrixPtr pK(*ds_dxj(logHyp, pSqDist, pDelta));

		// d^2s/dell dx'_j = (-2/ell) * ds/dx'_j
		pK->noalias() *= negative_double_inv_ell;

		return pK;
	}

	/**
	 * @brief	       d^3s             2         d^2s
	 *          ----------------- = - ----- * ------------
	 *           dell dx_i dx'_j       ell     dx_i dx'_j
	 * @param	[in] logHyp				The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDistXd			The squared distance between derivative inputs
	 * @param	[in] pDeltaXd_i		The i-th coordinate difference between derivative inputs
	 * @param	[in] pDeltaXd_j		The j-th coordinate difference between derivative inputs
	 * @return	An NdxNd matrix pointer.
	 *				Nd: The number of derivative training data
	 */
	static MatrixPtr d3s_dell_dxi_dxj(const Hyp &logHyp, const MatrixConstPtr pSqDistXd, const MatrixConstPtr pDeltaXd_i, const MatrixConstPtr pDeltaXd_j, const bool fSameCoords)
	{
		// constants
		const Scalar negative_double_inv_ell = static_cast<Scalar>(-2.0) 
														 * exp(static_cast<Scalar>(-1.0) * logHyp(0));	// -2/ell

		// memory allocation
		MatrixPtr pK(*d2s_dxi_dxj(logHyp, pSqDistXd, pDeltaXd_i, pDeltaXd_j, fSameCoords));

		// d^3s / dell dx_i dx'_j = (-2/ell) * d^2s / dx_i dx'_j
		pK->noalias() *= negative_double_inv_ell;

		return pK;
	}
};

}

#endif