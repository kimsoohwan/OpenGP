#include "GP.h"

int main()
{
	GP::GP<float, MeanZero, CovSEIso, InfExact> gp;
	gp.hyp(hyp);
	gp.setTrainingData(Training Data);
	gp.train(10);
	gp.predict(Test Data)

	return 0;
}
