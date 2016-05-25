#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

class PCLCloudSearch
{
public:
	static std::vector<double> FindClosestPoint(double,double,double, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree);
    static std::vector<double> FindClosestPointRadius(double,double,double,double,double, pcl::PointCloud<pcl::PointXYZ>::Ptr&, pcl::KdTreeFLANN<pcl::PointXYZ>&, cv::Mat);
	static void VoxelizeCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered);
private:
};
