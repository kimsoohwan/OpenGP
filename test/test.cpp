//#include "testcasepairwiseop.hpp"
//#include "testcasecovseiso.hpp"
//#include "testcasecovseisoslow.hpp"

//template<typename Scalar>
//class _CovSEIsoBase
//{
//public:
//	static void f()			{};
//	static void dk_ds()		{};
//	static void dk_dell()	{};
//};
//
//template<typename Scalar>
//class _CovSEIsoDerBase : public _CovSEIsoBase<Scalar>
//{
//public:
//	static void d2k_ds2()	{};
//	static void ds_dxi()		{};
//	static void ds_dxj()		{};
//};
//
//template<typename Scalar, template <typename> class Cov>
//class _Isotropic : public Cov<Scalar>
//{
//public:
//	static void K()		{ Cov<Scalar>::f();			}
//	static void Ks()		{ Cov<Scalar>::dk_ds();		}
//	static void Kss()		{ Cov<Scalar>::dk_dell();	}
//};
//
//template<typename Scalar, template <typename> class Cov>
//class _DerivativeObservatioins : public _Isotropic<Scalar, Cov>
//{
//public:
//	static void Kd()		{ Cov<Scalar>::d2k_ds2(); }
//	static void Ksd()		{ Cov<Scalar>::ds_dxi();  }
//	static void Kssd()	{ Cov<Scalar>::ds_dxj();  }
//};
//
//typedef _Isotropic<float, _CovSEIsoBase> _CovSEIsoSlow;
//typedef _DerivativeObservatioins<float, _CovSEIsoDerBase> _CovSEIsoDerSlow;

template<typename Scalar>
class _CovSEIsoBase
{
protected:
	static void f()			{};
	static void dk_ds()		{};
	static void dk_dell()	{};
};

template<typename Scalar>
class _CovSEIsoDerBase : public _CovSEIsoBase<Scalar>
{
protected:
	static void d2k_ds2()	{};
	static void ds_dxi()		{};
	static void ds_dxj()		{};
};

template<typename Scalar, template <typename> class Cov>
class _Isotropic : public Cov<Scalar>
{
public:
	static void K()		{ f();			}
	static void Ks()		{ dk_ds();		}
	static void Kss()		{ dk_dell();	}
};

template<typename Scalar, template <typename> class Cov>
class _DerivativeObservatioins : public _Isotropic<Scalar, Cov>
{
public:
	static void Kd()		{ d2k_ds2(); }
	static void Ksd()		{ ds_dxi();  }
	static void Kssd()	{ ds_dxj();  }
};

typedef _Isotropic<float, _CovSEIsoBase> _CovSEIsoSlow;
typedef _DerivativeObservatioins<float, _CovSEIsoDerBase> _CovSEIsoDerSlow;

int main(int argc, char** argv) 
{ 
	_CovSEIsoSlow::K();
	_CovSEIsoSlow::Ks();
	_CovSEIsoSlow::Kss();

	_CovSEIsoDerSlow::Kd();
	_CovSEIsoDerSlow::Ksd();
	_CovSEIsoDerSlow::Kssd();

	// Initialize test environment
	::testing::InitGoogleTest(&argc, argv);
		
	// Test
	int ret = RUN_ALL_TESTS(); 

	system("pause");
	return ret;
}