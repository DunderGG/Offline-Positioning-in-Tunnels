#include "Log.h"
#include "Common.h"
#include <iostream>
#include <string>

using namespace std;

Log::Log (const string dirname)
{
    Common::createDir(dirname);
}

void Log::logInit(const string logname)
{
    logFile.open(logname, std::ios::out);
}

void Log::logWrite(const string line)
{
    logFile << line << flush;
}

void Log::logClose()
{
    logFile.close();
}