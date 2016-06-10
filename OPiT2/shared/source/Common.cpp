#include "Common.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <thread>
#include <sys/stat.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "Reprojection.h"

#define FLT_THRESHOLD   1.192092896e-07F


using namespace std;
using namespace std::chrono;
using namespace cv;


// constructor
Common::Common()
{
    elapsedTime = 0;
}

// destructor()
Common::~Common()
{

}

// timer
void Common::startTimer()
{
    t1 = high_resolution_clock::now();
}

// timer
void Common::reportTimer()
{
    t2 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds> (t2-t1).count();
    elapsedTime += duration1;
    cout << "  finished in " << duration1 << "ms (" << duration1/1000 << " sec)" << endl;
}

void Common::reportTimer(string verbose)
{
    t2 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds> (t2-t1).count();
    elapsedTime += duration1;
    cout << "  " << verbose << " finished in " << duration1 << "ms (" << duration1/1000 << " sec)" << endl;
}

void Common::reportElapsedTime()
{
    cout << "  finished in " << elapsedTime << "ms (" << elapsedTime/1000 << " sec)" << endl;
}

void Common::reportElapsedTime(string verbose)
{
    cout << "  " << verbose << " finished in " << elapsedTime << "ms (" << elapsedTime/1000 << " sec)" << endl;
}

// log
void  Common::createDir(const string dirname)
{
    char logDirChar[100];
    strcpy(logDirChar, dirname.c_str());
    logDirChar[sizeof(logDirChar) - 1] = 0;

    mkdir(logDirChar, 0777);
}

// prepare map
void Common::prepareMap (string mapCoordinateFile, string mapKeypointsFile, vector<Point2d> &tunnel2Dx, vector<Point3d> &tunnel3Dx, Mat &tunnelDescriptor)
{
    // load descriptor
    string line;
    ifstream mapFile;
    mapFile.open(mapCoordinateFile);

    double temp=0,a=0,b=0,x=0,y=0,z=0;
    if (mapFile.is_open())
    {
        while (getline(mapFile, line) && !mapFile.eof())
        {
            istringstream in(line);

            for (int i=0; i<5; i++)
            {
                in >> temp;
                if (i==0)
                    a = temp;
                if (i==1)
                    b = temp;
                if (i==2)
                    x = temp;
                if (i==3)
                    y = temp;
                if (i==4)
                    z = temp;
            }
            tunnel2Dx.push_back(Point2f(a,b));
            tunnel3Dx.push_back(Point3f(x,y,z));
        }
    }
    mapFile.close();

    // load keypoints
    cv::FileStorage lstorage(mapKeypointsFile, cv::FileStorage::READ);
    lstorage["img"] >> tunnelDescriptor;
    lstorage.release();
}

void Common::readCsvTo3D2D(char *fileName, vector<Point3d> &worldPoints, vector<Point2d> &imagePoints)
{
    string line;
    ifstream file;

    // open the file
    file.open(fileName);

    // temporary output
    double x=0,y=0,z=0,u=0,v=0;

    if (file.is_open())
    {
        while (getline(file, line) && !file.eof())
        {
            istringstream in(line);
            string token;

            int rowCounter = 0;
            while (getline(in, token, ','))
            {
                if (rowCounter==0)
                    x = stod(token);
                if (rowCounter==1)
                    y = stod(token);
                if (rowCounter==2)
                    z = stod(token);
                if (rowCounter==3)
                    u = stod(token);
                if (rowCounter==4)
                    v = stod(token);

                rowCounter++;
            }

            worldPoints.push_back(Point3d(x,y,z));
            imagePoints.push_back(Point2d(u,v));
        }
    }
    file.close();
}

/* ---------------------------------------------------------------------------------------------------------
    lookup table tools
   ---------------------------------------------------------------------------------------------------------*/
void Common::updatelut (vector<Point3d> point3D, Mat descriptor, vector< pair<Point3d, Mat> > &lookuptable)
{
    for (int h = 0; h < point3D.size(); h++)
	{
		lookuptable.push_back(make_pair(point3D[h], descriptor.row(h)));
	}
}

Mat Common::getdescriptor (vector< pair<Point3d, Mat> > lookuptable)
{
    Mat descriptorTemp;
    for (pair<Point3d, Mat> pair : lookuptable)
        descriptorTemp.push_back(pair.second);

    return descriptorTemp;
}


/* ---------------------------------------------------------------------------------------------------------
    multithreading support for backprojection
   ---------------------------------------------------------------------------------------------------------*/

void Common::calcBestPoint ( Mat T, Mat K, vector<KeyPoint> imagepoint, Mat descriptor,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree,
                        int start, int end, int tidx,
                        vector<pair<Point3d, Mat> > &vocabulary,
                        vector<Point3d> &projected3D,
                        vector<Point2d> &projected2D,
                        vector<int> &projectedIndex)
{
    vector <double> temp = {0,0,0,1000};
    Point3d _mp3dcoord;
    for (int i=start; i<end; i++)
    {
        temp = Reprojection::backproject(T, K, Point2d(imagepoint[i].pt.x,imagepoint[i].pt.y), std::ref(cloud), std::ref(kdtree));
        _mp3dcoord.x = temp[0]; _mp3dcoord.y = temp[1]; _mp3dcoord.z = temp[2];
        if ((_mp3dcoord.x > 0.0f) && (_mp3dcoord.y > 0.0f) && (_mp3dcoord.z > 0.0f))
        {
            lock_guard<mutex> lock(g_mutex);
            {
                projected2D.push_back(Point2d(imagepoint[i].pt.x,imagepoint[i].pt.y));
                projected3D.push_back(_mp3dcoord);
                projectedIndex.push_back(i);
                vocabulary.push_back(make_pair(_mp3dcoord, descriptor.row(i)));
            }
        }
    }
}

void Common::threading( int numofthreads, Mat T, Mat K, vector<KeyPoint> detectedkpts, Mat descriptor,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree,
                        vector<pair<Point3d, Mat> > &lookuptable, vector<Point3d> &tunnel3D, vector<Point2d> &tunnel2D, vector<int> &tunnel1D)
{
    int numtask = floor(detectedkpts.size()/numofthreads);
    for (int tidx = 0; tidx < numofthreads; tidx++)
    {
        int start = tidx    * numtask;
        int end   = (tidx+1)* numtask;
        workers.push_back(thread(&Common::calcBestPoint,  this, T, K, detectedkpts, descriptor, std::ref(cloud), std::ref(kdtree),
                                                                start, end, tidx,
                                                                std::ref(lookuptable), std::ref(tunnel3D), std::ref(tunnel2D), std::ref(tunnel1D)));
    }
    for (int tidx = 0; tidx < numofthreads; tidx ++) {workers[tidx].join();} workers.clear();
}
