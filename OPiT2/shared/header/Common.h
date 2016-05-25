#ifndef __COMMON_H_INCLUDED_
#define __COMMON_H_INCLUDED_

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

using namespace std;
using namespace std::chrono;
using namespace cv;

class Common
{
public:
    //constructor & destructor
    Common();
    ~Common();

    //timer
    void startTimer();
    void reportTimer();

    //logging
    static void createDir(const string dirname);

    //preparemap
    void prepareMap (char* mapCoordinateFile, char* mapKeypointsFile, vector<Point2d> &tunnel2Dx, vector<Point3d> &tunnel3Dx, Mat &tunnelDescriptor);
    void updatelut (vector<Point3d>, Mat, vector< pair<Point3d, Mat> > &);
    void threading(int numofthreads, Mat T, Mat K, vector<KeyPoint> detectedkpts, Mat descriptor,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree,
                   vector<pair<Point3d, Mat> > &lookuptable, vector<Point3d> &tunnel3D, vector<Point2d> &tunnel2D, vector<int> &tunnel1D);
    Mat getdescriptor (vector< pair<Point3d, Mat> >);

private:
    high_resolution_clock::time_point t1, t2;
    mutex                  g_mutex;
    void calcBestPoint ( Mat T, Mat K, vector<KeyPoint> imagepoint, Mat descriptor,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree,
                    int start, int end, int tidx,
                    vector<pair<Point3d, Mat> > &vocabulary,
                    vector<Point3d> &projected3D,
                    vector<Point2d> &projected2D,
                    vector<int> &projectedIndex);
    vector<thread> workers;
};

#endif
