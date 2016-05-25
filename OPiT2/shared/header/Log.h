#ifndef __LOG_H_INCLUDED
#define __LOG_H_INCLUDED

#include <iostream>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <string>

using namespace std;

class Log
{
public:
    // constructor
    Log(const string dirname);
    Log();
    
    void logInit(const string logname);
    void logWrite(const string line);
    void logClose();

private:
    ofstream logFile;
};

#endif
