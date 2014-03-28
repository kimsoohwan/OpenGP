#ifndef _INFERENCE_METHOD_EXACT_HPP_
#define _INFERENCE_METHOD_EXACT_HPP_

#include "../data/typetraits.hpp"
#include "../data/trainingdatasettable.hpp"
#include "../gp/Hyp.hpp"

namespace GP{

	/**
	 * @class	InfExact
	 * @brief	Exact inference
	 * @author	Soohwankim
	 * @date	26/03/2014
	 */
	template<typename Scalar, typename MeanFunc, typename CovFunc, typename LikFunc>
	class InfExact : public TrainingDataSettable<Scalar>
	{
	public:
		/**
		 * @brief	Sets the training data.
		 * @param	data	The training data.
		 */
		//void setTrainingData(const typename TrainingData::ConstPtr data)
		bool set(const TrainingDataConstPtr data)
		{
			m_MeanFunc.setTrainingData(data);
			m_CovFunc.setTrainingData(data);
			m_LikFunc.setTrainingData(data);

			return true;
		}

		/**
		 * @brief	Predict the mean and [co]variance.
		 * @param [in]		hyp				 	The hyperparameters.
		 * @param [in]		pXs				 	The test positions.
		 * @param [out]	pMu	 				The mean vector.
		 * @param [out]	pSigma 				The covariance matrix or variance vector.
		 * @param [in]		fVarianceVector 	(Optional) flag for true: variance vector, false: covariance matrix
		 * @param [in]		fBatchProcessing	(Optional) flag for the batch processing.
		 */
		void predict(const Hyp<MeanFunc, CovFunc, LikFunc>	&hyp, 
						 MatrixConstPtr		pXs, 
						 VectorPtr				&pMu, 
						 MatrixPtr				&pSigma, 
						 const bool				fVarianceVector = true,
						 const bool				fBatchProcessing = true)
		{
		}

	protected:
		// GP setting
		MeanFunc				m_MeanFunc;	/// The mean function
		CovFunc				m_CovFunc;	/// The covariance function
		LikFunc				m_LikFunc;	/// The likelihood function
	};
}

#endif