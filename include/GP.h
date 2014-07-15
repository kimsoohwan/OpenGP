#ifndef _Gaussian_Processes_H_
#define _Gaussian_Processes_H_

//// Type Traits
//#include "../src/data/typetraits.hpp"

// Util
#include "../src/util/macros.h"
#include "../src/util/PairwiseOp.hpp"

// Data
#include "../src/data/TrainingData.hpp"
#include "../src/data/DerivativeTrainingData.hpp"
#include "../src/data/TestData.hpp"

// GP
#include "../src/gp/GP.hpp"

// Mean Functions
#include "../src/mean/MeanZero.hpp"

// Covariance Functions
#include "cov.h"

// Likelihood Function
#include "../src/lik/LikGauss.hpp"
#include "../src/lik/LikGaussDerObs.hpp"

// Inference Methods
#include "../src/inf/Hyp.hpp"
#include "../src/inf/InfExact.hpp"
#include "../src/inf/InfExactGeneral.hpp"

#endif
