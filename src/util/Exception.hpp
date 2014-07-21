#ifndef _CUSTOM_EXCEPTION_HPP_
#define _CUSTOM_EXCEPTION_HPP_

#include <exception>
#include <string>

namespace GP {

/**
  * @class		Exception
  * @brief		Customized exception
  * @ingroup	-Util
  * @author		Soohwan Kim
  * @date		15/07/2014
  */
class Exception : public std::exception
{
public:
	Exception(std::string msg = "Custom Exception")
		: m_strMessage(msg)
	{
	}

	~Exception() throw()
	{
	}

	Exception& operator=(const Exception &e)
	{
		m_strMessage = e.m_strMessage;
		return *this;
	}

	Exception& operator=(const std::string &msg)
	{
		m_strMessage = msg;
		return *this;
	}

	virtual const char* what() const throw()
	{
		return m_strMessage.c_str();
	}

protected:
  std::string m_strMessage;
};

}

#endif