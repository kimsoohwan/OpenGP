#include "GP.h"
using namespace GP;

int main()
{
	GP<float, MeanZero, CovSEIso, InfExact> gp;
	gp.hyp(hyp);
	gp.setTrainingData(Training Data);
	gp.train(10);
	gp.predict(Test Data)

	return 0;
}
