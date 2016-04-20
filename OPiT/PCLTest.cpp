#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "PCLTest.h"

void PCLTest::foo()
{
	pcl::PointCloud<pcl::PointXYZ> cloud;
	// F i l l i n t h e c l o u d d a t a
	cloud.width = 500;
	cloud.height = 1;
	cloud.is_dense = false;
	cloud.points.resize(cloud.width * cloud.height);
	for(size_t i = 0; i < cloud.points.size(); ++i)
	{
		cloud.points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud.points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud.points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
	}
	pcl::io::savePCDFileASCII("testpcd.pcd", cloud);

}

std::vector<double> PCLTest::test(double x, double y, double z, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	//srand(time(NULL));


	/*
	// Generate pointcloud data
	cloud->width = 1000;
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);

	for (size_t i = 0; i < cloud->points.size(); ++i)
	{
		cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	}
	*/

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

	kdtree.setInputCloud(cloud);

	pcl::PointXYZ searchPoint;

	//searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	//searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	//searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);

	searchPoint.x = x;
	searchPoint.y = y;
	searchPoint.z = z;

	// K nearest neighbor search

	int K = 1;

	std::vector<double> bestPoint {0, 0, 0, 1000};

	std::vector<int> pointIdxNKNSearch(K);
	std::vector<float> pointNKNSquaredDistance(K);

	// std::cout << "K nearest neighbor search at: \n(" << searchPoint.x
	// 	<< " " << searchPoint.y
	// 	<< " " << searchPoint.z
	// 	<< ") with K=" << K << std::endl;

	if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
		{
			// std::cout	<< " " << cloud->points[pointIdxNKNSearch[i]].x
			// 			<< " " << cloud->points[pointIdxNKNSearch[i]].y
			// 			<< " " << cloud->points[pointIdxNKNSearch[i]].z
			// 			<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;

			if (pointNKNSquaredDistance[i] < bestPoint[3])
			{
				bestPoint[0] = cloud->points[pointIdxNKNSearch[i]].x;
				bestPoint[1] = cloud->points[pointIdxNKNSearch[i]].y;
				bestPoint[2] = cloud->points[pointIdxNKNSearch[i]].z;
				bestPoint[3] = pointNKNSquaredDistance[i];
			}
		}
	}


	/*
	// Neighbors within radius search

	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;

	float radius = 1;

	std::cout << "Neighbors within radius search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with radius=" << radius << std::endl;


	if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
		{
			std::cout << "    " << cloud->points[pointIdxRadiusSearch[i]].x
				<< " " << cloud->points[pointIdxRadiusSearch[i]].y
				<< " " << cloud->points[pointIdxRadiusSearch[i]].z
				<< " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;

			if (pointRadiusSquaredDistance[i] < bestPoint[3])
			{
				bestPoint[0] = cloud->points[pointIdxNKNSearch[i]].x;
				bestPoint[1] = cloud->points[pointIdxNKNSearch[i]].y;
				bestPoint[2] = cloud->points[pointIdxNKNSearch[i]].z;
				bestPoint[3] = pointNKNSquaredDistance[i];
			}
		}
	}
	*/
	return bestPoint;


}
