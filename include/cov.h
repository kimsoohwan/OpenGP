#ifndef _COVARIANCE_FUNCTION_H_
#define _COVARIANCE_FUNCTION_H_

// Derivative observations
#include "../src/cov/CovDerObs.hpp"

// Squared Exponential
#include "../src/cov/covSEiso/CovSEiso.hpp"
#include "../src/cov/covSEiso/CovSEisoDerObsBase.hpp"

// Matern
#include "../src/cov/covMaterniso/CovMaterniso.hpp"
#include "../src/cov/covMaterniso/CovMaternisoDerObsBase.hpp"

// Sparse
#include "../src/cov/covSparseiso/CovSparseiso.hpp"
#include "../src/cov/covSparseiso/CovSparseisoDerObsBase.hpp"

// Rational Quadratic
#include "../src/cov/covRQiso/CovRQiso.hpp"
#include "../src/cov/covRQiso/CovRQisoDerObsBase.hpp"

// Prod
#include "../src/cov/covProd/CovProd.hpp"

namespace GP {

/**
 * @defgroup	-Cov
 * @brief		All covariance classes should have public static member functions as follows.
 *					<CENTER>
 *					Public Static Member Functions | Corresponding Covariance Functions
 *					-------------------------------|-------------------------------------
 *					+K				| \f$\mathbf{K} = \mathbf{K}(\mathbf{X}, \mathbf{X}) \in \mathbb{R}^{N \times N}\f$
 *					+Ks			| \f$\mathbf{K}_* = \mathbf{K}(\mathbf{X}, \mathbf{Z}) \in \mathbb{R}^{N \times M}\f$
 *					+Kss			| \f$\mathbf{k}_{**} \in \mathbb{R}^{M \times 1}, \mathbf{k}_{**}^i = k(\mathbf{Z}_i, \mathbf{Z}_i)\f$ or \f$\mathbf{K}_{**} = \mathbf{K}(\mathbf{Z}, \mathbf{Z}) \in \mathbb{R}^{M \times M}\f$
 *					</CENTER>
 *					where \f$N\f$: the number of training data and \f$M\f$: the number of test data given
 *					\f[
 *					\mathbf{\Sigma} = 
 *					\begin{bmatrix}
 *					\mathbf{K} & \mathbf{k}_*\\ 
 *					\mathbf{k}_*^\text{T} & k_{**}
 *					\end{bmatrix}
 *					\text{,   or   }
 *					\mathbf{\Sigma} = 
 *					\begin{bmatrix}
 *					\mathbf{K} & \mathbf{K}_*\\ 
 *					\mathbf{K}_*^\text{T} & \mathbf{K}_{**}
 *					\end{bmatrix}
 *					\f]
 *
 *					The public static member functions, K, Ks and Kss call 
 *					a protected general member function, K(const Hyp, const MatrixConstPtr, const int)
 *					which only depends on pair-wise squared distances.\n\n
 *
 * 				In addition, no covariance class contains any data.
 *					Instead, data are stored in data classes such as
 *					-# TrainingData
 *					-# DerivativeTrainingData
 *					-# TestData
 *					.
 */

/**
 * @defgroup	-SEiso
 * @brief		Squared exponential covariance functions with isotropic distances\n
 *					\f[
 *					k(\mathbf{x}, \mathbf{z}) = \sigma_f^2 \exp\left(-\frac{r^2}{2l^2}\right), \quad r = |\mathbf{x}-\mathbf{z}|
 *					\f]
 * @ingroup		-Cov
 */ 

/**
 * @class		CovSEisoDerObs
 * @brief		Squared exponential covariance function dealing with derivative observations\n
 *					It inherits from CovDerObs which takes CovSEisoDerObsBase as a template parameter.\n
 *					Thus, CovSEisoDerObs is a combination of CovDerObs and CovSEisoDerObsBase.
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-SEiso
 * @author		Soohwan Kim
 * @date			30/06/2014
 */
template<typename Scalar>
class CovSEisoDerObs : public CovDerObs<Scalar, CovSEisoDerObsBase> {};


//template <typename Scalar>
//using CovSEisoDerObs = CovDerObs<Scalar, CovSEisoDerObsBase>;

//template<typename Scalar> 
//struct _CovDerObs
//{
//   typedef CovDerObs<Scalar, CovSEisoDerObsBase> __CovSEisoDerObs;
//};
//template<typename Scalar> using CovSEisoDerObs = typename _CovDerObs::__CovSEisoDerObs;

/**
 * @defgroup	-Materniso
 * @brief		Matern covariance function with isotropic distances, \f$\nu = 3/2\f$\n
 *					\f[
 *					k(\mathbf{x}, \mathbf{z}) = \sigma_f^2 \left(1+\frac{\sqrt{3}r}{l}\right)\exp\left(-\frac{\sqrt{3}r}{l}\right), \quad r = |\mathbf{x}-\mathbf{z}|
 *					\f]
 * @ingroup		-Cov
 */ 

/**
 * @class		CovMaternisoDerObs
 * @brief		Matern covariance function dealing with derivative observations\n
 *					It inherits from CovDerObs which takes CovMaternisoDerObsBase as a template parameter.\n
 *					Thus, CovMaternisoDerObs is a combination of CovDerObs and CovMaternisoDerObsBase.
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-Materniso
 * @author		Soohwan Kim
 * @date			25/08/2014
 */
template<typename Scalar>
class CovMaternisoDerObs : public CovDerObs<Scalar, CovMaternisoDerObsBase> {};

/**
 * @defgroup	-Sparseiso
 * @brief		Sparse covariance functions with isotropic distances\n
 *					\f[
 *					k(\mathbf{x}, \mathbf{z}) = 
 *					\begin{cases} 
 *					\sigma_f^2 \left( \frac{2+\cos(2\pi \frac{r}{l})}{3} \left(1-\frac{r}{l}\right) + \frac{1}{2\pi} \sin\left(2\pi \frac{r}{l}\right) \right) & r < l\\
 *					0 & r \ge l
 *					\end{cases}
 *					\f]
 * @ingroup		-Cov
 */ 

/**
 * @class		CovSparseisoDerObs
 * @brief		Sparse covariance function dealing with derivative observations\n
 *					It inherits from CovDerObs which takes CovSparseisoDerObsBase as a template parameter.\n
 *					Thus, CovSparseisoDerObs is a combination of CovDerObs and CovSparseisoDerObsBase.
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-Sparseiso
 * @author		Soohwan Kim
 * @date			25/08/2014
 */
template<typename Scalar>
class CovSparseisoDerObs : public CovDerObs<Scalar, CovSparseisoDerObsBase> {};

/**
 * @defgroup	-RQiso
 * @brief		Rational quadratic covariance functions with isotropic distances\n
 *					\f[
 *					k(\mathbf{x}, \mathbf{z}) = \sigma_f^2 \left(1 + \frac{r^2}{2\alpha l^2} \right)^{-\alpha}, \quad r = |\mathbf{x}-\mathbf{z}|
 *					\f]
 * @ingroup		-Cov
 */ 

/**
 * @class		CovRQisoDerObs
 * @brief		Rational quadratic covariance function dealing with derivative observations\n
 *					It inherits from CovDerObs which takes CovRQisoDerObsBase as a template parameter.\n
 *					Thus, CovRQisoDerObs is a combination of CovDerObs and CovRQisoDerObsBase.
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-Sparseiso
 * @author		Soohwan Kim
 * @date			11/09/2014
 */
template<typename Scalar>
class CovRQisoDerObs : public CovDerObs<Scalar, CovRQisoDerObsBase> {};

/**
 * @defgroup	-CovComposite
 * @brief		Product of two covariance functions\n
 * @ingroup		-Cov
 */ 

/**
 * @class		CovSEMaterniso
 * @brief		Product of CovSEiso and CovMaterniso
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-CovComposite
 * @author		Soohwan Kim
 * @date			25/08/2014
 */
template <typename Scalar>
class CovSEMaterniso : public CovProd<Scalar, CovSEiso, CovMaterniso> {};

/**
 * @class		CovSparseMaterniso
 * @brief		Product of CovSparseiso and CovMaterniso
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-CovComposite
 * @author		Soohwan Kim
 * @date			25/08/2014
 */
template <typename Scalar>
class CovSparseMaterniso : public CovProd<Scalar, CovSparseiso, CovMaterniso> {};

/**
 * @class		CovSparseMaternisoDerObs
 * @brief		Product of CovSparseisoDerObs and CovMaternisoDerObs
 * @tparam		Scalar	Datatype such as float and double
 * @ingroup		-CovComposite
 * @author		Soohwan Kim
 * @date			25/08/2014
 */
template <typename Scalar>
class CovSparseMaternisoDerObs : public CovProd<Scalar, CovSparseisoDerObs, CovMaternisoDerObs> {};

}

#endif