#ifndef _TRAINING_DATA_SETTABLE_HPP_
#define _TRAINING_DATA_SETTABLE_HPP_

#include "typetraits.hpp"

namespace GP{

/**
 * @class	TrainingDataSettable
 * @brief	Setting training data.
 * @note		The mean/cov/lik/inf functions need to set training data.
 * 			This is a base class to avoid duplicating 
 * 			the same feature to set training data for them.
 * @author	Soohwankim
 * @date	26/03/2014
 */
template<typename Scalar>
class TrainingDataSettable : public TypeTraits<Scalar>
{
public:
	/**
		* @brief	Sets training inputs.
		* @param	pX	The training inputs. An NxD matrix.
		* @param	pY	The training outputs. An Nx1 vector.
		*/
	bool set(const MatrixConstPtr pX, VectorConstPtr pY)
	{
		assert(pX->rows() == pY->size());

		// Check if the training inputs are the same.
		if(m_pX == pX && m_pY == pY) return false;

		// Set the training data.
		m_pX = pX;
		m_pY = pY;

		return true;
	}

	/**
	 * @brief	Gets the number of training data.
	 * @return	N: the number of training data.
	 */
	int N() const { return m_pX->rows(); }

	/**
	 * @brief	Gets the number of dimensions.
	 * @return	D: the number of dimensions.
	 */
	int D() const { return m_pX->cols(); }

protected:

	/** @brief	The training inputs. */
	MatrixConstPtr		m_pX; // NxD matrix

	/** @brief	The training outputs. */
	VectorConstPtr		m_pY; // Nx1 vector
};

}
#endif