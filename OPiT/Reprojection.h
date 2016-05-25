#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

class Reprojection
{
public:
	Reprojection();

	cv::Mat foo(cv::Mat,cv::Mat, cv::Mat, cv::Mat, cv::Mat, cv::Mat);

	static std::vector<double> backproject(cv::Mat T, cv::Mat K, cv::Point2d imagepoint, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree);
    static std::vector<double> backprojectRadius(cv::Mat T, cv::Mat	K, cv::Point2d imagepoint, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree);
    static std::vector<double> LinearInterpolation(std::vector<double> bestPoint, cv::Mat origin, cv::Mat vectorPoint);

private:

};
