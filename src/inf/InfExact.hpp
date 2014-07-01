#ifndef _INFERENCE_METHOD_EXACT_HPP_
#define _INFERENCE_METHOD_EXACT_HPP_

#include "../util/macros.hpp"
#include "../data/DerivativeTrainingData.hpp"
#include "../data/TestData.hpp"
#include "../gp/Hyp.hpp"

namespace GP{

/**
	* @class	InfExact
	* @brief	Exact inference
	* @author	Soohwan Kim
	* @date	26/03/2014
	*/
template<typename Scalar, 
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc>
class InfExact
{
// define matrix types
protected:	TYPE_DEFINE_MATRIX(Scalar);

// hyperparameters
public:
	//struct Hyp
	//{
	//	typename MeanFunc::Hyp		logMeanHyp;
	//	typename CovFunc::Hyp		logCovHyp;
	//	typename LikFunc::Hyp		logLikHyp;
	//};

	//typedef	typename MeanFunc::Hyp		MeanHyp;
	//typedef	typename CovFunc::Hyp		CovHyp;
	//typedef	typename LikFunc::Hyp		LikHyp;

public:
	/**
		* @brief	Sets the training data.
		* @param	data	The training data.
		*/
	//void setTrainingData(const typename TrainingData::ConstPtr data)
	//bool set(const TrainingDataConstPtr data)
	//{
	//	m_MeanFunc.setTrainingData(data);
	//	m_CovFunc.setTrainingData(data);
	//	m_LikFunc.setTrainingData(data);

	//	return true;
	//}
	//bool set(MatrixConstPtr pX, VectorConstPtr pY)
	//{
	//	// Check if the training data are the same.
	//	if(!TrainingDataSettable::set(pX, pY))		return false;

	//	// Set the training data
	//	m_MeanFunc.set(pX);
	//	m_CovFunc.set(pX);
	//	m_LikFunc.set(pX);

	//	return true;
	//}

	/**
		* @brief	Predict the mean and [co]variance.
		* @param [in]		logHyp				The log hyperparameters.
		* @param [in]		pXs				 	The test positions.
		* @param [out]	pMu	 				The mean vector.
		* @param [out]	pSigma 				The covariance matrix or variance vector.
		* @param [in]		fVarianceVector 	(Optional) flag for true: variance vector, false: covariance matrix
		* @param [in]		fBatchProcessing	(Optional) flag for the batch processing.
		*/
	template<typename GeneralTrainingData>
	void predict(const Hyp<MeanFunc, CovFunc, LikFunc>	&logHyp, 
						GeneralTrainingData<Scalar>	&trainingData, 
						const MatrixConstPtr		pXs, 
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