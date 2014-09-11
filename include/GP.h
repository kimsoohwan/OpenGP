#ifndef _Gaussian_Processes_H_
#define _Gaussian_Processes_H_

/**
 * @defgroup	-Cov
 * @brief		Covariance functions for Gaussian processes
 */ 
#include "cov.h"


/**
 * @defgroup	-Mean
 * @brief		Mean functions for Gaussian processes
 */ 
#include "mean.h"


/**
 * @defgroup	-Lik
 * @brief		Likelihood functions for Gaussian processes
 */ 
#include "../src/lik/LikGauss.hpp"
#include "../src/lik/LikGaussDerObs.hpp"


/**
 * @defgroup	-Inf
 * @brief		Inference methods for Gaussian processes
 */ 
#include "inf.h"


/**
 * @defgroup	-Data
 * @brief		Training and test data set for Gaussain processes
 */ 
#include "../src/data/TrainingData.hpp"
#include "../src/data/DerivativeTrainingData.hpp"
#include "../src/data/TestData.hpp"


/**
 * @defgroup	-GP
 * @brief		Gaussian processes
 */ 
#include "../src/gp/GaussianProcess.hpp"


/**
 * @defgroup	-Trainer
 * @brief		Trainer for hyperparameters
 */ 
#include "../src/gp/Trainer.hpp"


/**
 * @defgroup	-Util
 * @brief		Utilities
 */ 
#include "../src/util/macros.h"
#include "../src/util/PairwiseOp.hpp"
#include "../src/util/LogFile.hpp"


#endif
