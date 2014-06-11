#ifndef _Gaussian_Processes_H_
#define _Gaussian_Processes_H_

// Type Traits
#include "../src/data/typetraits.hpp"

// Data
#include "../src/data/trainingdata.hpp"

// GP
#include "../src/gp/gp.hpp"

// Mean Functions
#include "../src/mean/meanzero.hpp"

// Covariance Functions
#include "cov.h"

// Inference Methods
#include "../src/inf/infexact.hpp"

// Util
#include "../src/util/pairwiseop.hpp"
#include "../src/util/macros.hpp"

#endif
