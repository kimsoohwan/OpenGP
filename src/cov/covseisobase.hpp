#ifndef _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_BASE_HPP_
#define _COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_BASE_HPP_

#include "../util/macros.hpp"
#include "../data/trainingdata.hpp"

namespace GP {

/**
	* @class		CovSEIsoBase
	* @brief		Base class for the isotropic squared exponential covariance function.
	*
	*				It defines component functions used in class: Isotropic.
	*				It is rather slow in computation but easy to fix bugs.
	*				So, it can be used as a reference for a faster version.
	*
	* 				k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	*
	*				It defines f(s) which is required for calculating k(x, x'),
	*				and dk/ds and ds/dell for dk/dlog(ell), dk/dlog(sigma_f).
	*				For details, please refer to class: isotropic.
	* @note		CRTP (Curiously Recursive Template Pattern)
	*				Isotropic<Scalar, CovSEIsoBase> class the component functions.
	*				Thus, they should be public, or class: Isotropic is a friend of this class.
	* @author	Soohwan Kim
	* @date		03/06/2014
	*/
template<typename Scalar>
class CovSEIsoBase
{
// define matrix types
protected:	TYPE_DEFINE_MATRIX(Scalar);

// define hyperparameters
public:		TYPE_DEFINE_HYP(Scalar, 2); // log(ell), log(sigma_f)

// component functions
protected:

/**************************************************************************************/
/* [1] For calculating k(x, x'), f(s) is required,                                    */
/*     which will be used in class: Isotropic                                         */
/**************************************************************************************/
	/**
	 * @brief	f(s) = exp(s) for k(r)
	 *				It calls class: CovSEIsoBase::s
	 * @note		k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	 * @param	[in] logHyp 				The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDist 				The squared distance between data, r^2.
	 * @return	An NxN matrix pointer.	N: The number of training data.
	 */
	static MatrixPtr f(const Hyp &logHyp, const MatrixConstPtr pSqDist) 
	{
		// memory allocation
		MatrixPtr pK = s(logHyp, pSqDist);

		// f(s) = exp(s)
		(*pK) = pK->array().exp();

		return pK;
	}

	/**
	 * @brief	          r^2
	 *          s =  - ---------
	 *                  2*ell^2
	 * @note		k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	 * @param	[in] logHyp 				The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDist 				The squared distance between data, r^2.
	 * @return	An NxN matrix pointer.	N: The number of training data.
	 */
	static MatrixPtr s(const Hyp &logHyp, const MatrixConstPtr pSqDist) 
	{
		// constants
		const Scalar neg_half_inv_ell2	= static_cast<Scalar>(-0.5) 
													* exp(static_cast<Scalar>(-2.0) * logHyp(0)); // -1/(2*ell^2)

		// memory allocation
		MatrixPtr pK(new Matrix(*pSqDist));

		// s = (-1/(2*ell^2))*r^2
		(*pK) *= neg_half_inv_ell2;

		return pK;
	}


/**************************************************************************************/
/* [2] For calculaing derivatives w.r.t parameters,                                   */
/*                                                                                    */
/*		      dk        ds                                                              */
/*         ---- and ------ are required,															  */
/*          ds       dell                                                             */
/*                                                                                    */
/*     which will be used in class: Isotropic                                         */
/**************************************************************************************/

	/**
	 * @brief	 dk
	 *          ---- = k
	 *           ds
	 * @note		k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	 * @param	[in] logHyp 				The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDist 				The squared distance between data, r^2.
	 * @return	An NxN matrix pointer.	N: The number of training data.
	 */
	static MatrixPtr dk_ds(const Hyp &logHyp, const MatrixConstPtr pSqDist) 
	{
		return k(logHyp, pSqDist);
	}

	/**
	 * @brief	  ds
	 *          ------ = (-2/ell) * s
	 *           dell
	 * @note		k(x, x') = sigma_f^2 * f(s), f(s) = exp(s), s(r) = -r^2/(2*ell^2), r = |x-x'|
	 * @param	[in] logHyp 				The log hyperparameters, log([ell, sigma_f]).
	 * @param	[in] pSqDist 				The squared distance between data, r^2.
	 * @return	An NxN matrix pointer.	N: The number of training data.
	 */
	static MatrixPtr ds_dell(const Hyp &logHyp, const MatrixConstPtr pSqDist) 
	{
		// constants
		const Scalar neg_double_inv_ell	= static_cast<Scalar>(-2.0)
													* exp(static_cast<Scalar>(-1.0) * logHyp(0)); // -2/ell

		// memory allocation
		MatrixPtr pK = s(logHyp, pSqDist);

		// ds/dl = (-2/ell) * s
		(*pK) *= neg_double_inv_ell;
		return pK;
	}
};

}

#endif