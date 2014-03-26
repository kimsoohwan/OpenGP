#ifndef _GAUSSIAN_PROCESSES_HPP_
#define _GAUSSIAN_PROCESSES_HPP_

namespace GP{
	template<typename MeanFunc, typename CovFunc, typename LikFunc,
		      template <typename, typename, typename> typename InfMethod>
	class GP : public InfMethod<MeanFunc, CovFunc, LikFunc>
	{
	public:
		void train();
	};
}

#endif