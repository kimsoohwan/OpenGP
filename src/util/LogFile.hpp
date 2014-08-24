#ifndef _GP_LOG_HPP_
#define _GP_LOG_HPP_

#include <string>
#include <fstream>

namespace GP {

class LogFile
{
public:
	/** @brief Constructor */
	LogFile()
	{}

	/** @brief Constructor */
	LogFile(const std::string &strFilePath)
	{
		open(strFilePath);
	}

	/** @brief Destructor */
	virtual ~LogFile()
	{
		//close();
	}

	/** @brief Open a log file */
	static inline bool open(const std::string &strFilePath)
	{
		// if it is already open, close it and open a new file
		if(m_strFilePath.compare(strFilePath) == 0) return false;
		close();

		// open a file
		m_ofs.open(strFilePath.c_str());
		if(!m_ofs.is_open())
		{
			std::cerr << "Error: Can't open a log file, " << strFilePath << std::endl;
		}

		// file name
		m_strFilePath = strFilePath;
		return true;
	}

	/** @brief Close the log file */
	static inline bool close()
	{
		// if it is not open, return false
		if(!m_ofs.is_open()) return false;

		// close the file
		m_ofs.close();

		return true;
	}

	/** @brief Check whether the log file is open or not */
	static inline bool is_open()
	{
		const bool fIsOpen = m_ofs.is_open();
		return fIsOpen;
	}

	/** @brief Set verbose */
	static inline void verbose(const bool fVerbose)
	{
		m_fVerbose = fVerbose;
	}
		
	/** @brief Write down the log */
	template <typename T>
	inline LogFile& operator<< (const T &obj)
	{
		// print on screen
		if(m_fVerbose)		std::cout << obj;

		// print in file
		if(is_open())		m_ofs << obj;

		return *this;
	}

	/** @brief Write down std::endl which is a function template */
	inline LogFile& operator<< (std::ostream& (*pfun)(std::ostream&))
	{
		// print on screen
		if(m_fVerbose)		pfun(std::cout);

		// print in file
		if(is_open())		pfun(m_ofs);

	  return *this;
	}

	inline std::string getFilePath() const
	{
		return m_strFilePath;
	}

protected:
	static std::string		m_strFilePath;
	static std::ofstream		m_ofs;
	static bool					m_fVerbose;
};

// static member initialization
std::string			LogFile::m_strFilePath = std::string();
std::ofstream		LogFile::m_ofs = std::ofstream();
bool					LogFile::m_fVerbose = true;

}

#endif