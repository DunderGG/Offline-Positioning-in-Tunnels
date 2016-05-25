#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

#include "PCLCloudSearch.h"


std::vector<double> PCLCloudSearch::FindClosestPoint(double x, double y, double z, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree)
{
	pcl::PointXYZ searchPoint;

	searchPoint.x = x;
	searchPoint.y = y;
	searchPoint.z = z;

	// K nearest neighbor search, we want only the nearest point.
	int K = 1;

	std::vector<double> bestPoint{0, 0, 0, 1000};

	std::vector<int> pointIdxNKNSearch(K);
	std::vector<float> pointNKNSquaredDistance(K);

	//std::cout << "K nearest neighbor search at (" << searchPoint.x
	//	<< " " << searchPoint.y
	//	<< " " << searchPoint.z
	//	<< ") with K=" << K << std::endl;

	if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
		{
			//std::cout	<< "    " << cloud->points[pointIdxNKNSearch[i]].x
			//			<< " " << cloud->points[pointIdxNKNSearch[i]].y
			//			<< " " << cloud->points[pointIdxNKNSearch[i]].z
			//			<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
			//
			if (pointNKNSquaredDistance[i] < bestPoint[3])
			{
				bestPoint[0] = cloud->points[pointIdxNKNSearch[i]].x;
				bestPoint[1] = cloud->points[pointIdxNKNSearch[i]].y;
				bestPoint[2] = cloud->points[pointIdxNKNSearch[i]].z;
				bestPoint[3] = pointNKNSquaredDistance[i];
			}
		}
	}

	return bestPoint;
}

std::vector<double> PCLCloudSearch::FindClosestPointRadius(double x, double y, double z, double radius, double threshold,
                                                           pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                           pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree,
                                                           cv::Mat origin)
{
	pcl::PointXYZ searchPoint;
    cv::Mat searchPointMat = (cv::Mat_<double>(4, 1) << x,y,z,1);

	searchPoint.x = x;
	searchPoint.y = y;
	searchPoint.z = z;

	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;

	std::vector<double> bestPoint{0,0,0,1000};

    double curr_best    = INFINITY;
    double l2_lerp      = INFINITY;
	double curr_best_d 	= INFINITY;
    int best_idx;

	if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
	{
		bool best_status    = FALSE;

//        std::cout << cloud->points[596225] << std::endl;

		// perform lerp to get the projection of those points on the line
        for (int it = 0; it < pointIdxRadiusSearch.size(); it++)
        {
            // basically if known two points in 3D A and B, and a point P (bestPoint) that does not belong to vector AB
            // a perpendicular vector xP has 90 degree angle from AB and has length of bestPoint[3].
            //
            // x is the point in which we need to return. Ax is basically the orthogonal projection of AP
            // to vector AB.
            //
            // it given as
            // 		x = A + dot(AP,AB) / dot(AB,AB) * AB
            //				with
            //				 A   = origin
            //				 P   = bestPoint
            //				 B   = 3D point in line
            //				 AP  = vector from origin to the point P (bestPoint)
            // 				 AB  = vector from origin to the AB
            //				 dot = dot product
            // 				 *   = scalar multiplication

			cv::Mat P 		 = (cv::Mat_<double>(4, 1) << cloud->points[pointIdxRadiusSearch[it]].x,
														  cloud->points[pointIdxRadiusSearch[it]].y,
														  cloud->points[pointIdxRadiusSearch[it]].z,
														  1);
            cv::Mat vectorAP = P - origin;
            cv::Mat vectorAB = searchPointMat - origin;
            cv::Mat output   = origin + (vectorAP.dot(vectorAB) / vectorAB.dot(vectorAB)) * vectorAB;

//            std::cout << P << std::endl;
//            std::cout << origin << std::endl;
//            std::cout << vectorAP << std::endl;
//            std::cout << vectorAB << std::endl;
//            std::cout << output << std::endl;

			// get the square root distance of neigbours after we got the lerp-ed xyz (output)
            double y = cv::norm(searchPointMat, output, cv::NORM_L2);
            double d = sqrt(pow(pointRadiusSquaredDistance[it],2) - pow(y,2));

            // if we detect a neighbour that conforms with the threshold requirement
            // check the distance of it from the origin as well, and take the smallest one
            if (d < threshold)
            {
                // get the distance from the origin to the interpolated (lerp) point
                l2_lerp = cv::norm(output, origin, cv::NORM_L2);

                // update the best point
                if (l2_lerp < curr_best)
                {
					//std::cout << "-----GOT SOMETHING!-----" << std::endl;
                    curr_best   = l2_lerp;
					curr_best_d = d;
                    best_idx    = pointIdxRadiusSearch[it];
                    best_status = TRUE;
                }
            }
        }

		// if no best point
        if (best_status == TRUE)
		{
			//std::cout << cloud->points[best_idx].x << " " << cloud->points[best_idx].y << " " << cloud->points[best_idx].z << std::endl;
            // until now, we've got the best point
			bestPoint[0] = cloud->points[best_idx].x;
			bestPoint[1] = cloud->points[best_idx].y;
			bestPoint[2] = cloud->points[best_idx].z;
			bestPoint[3] = curr_best_d;

            return bestPoint;
		}
        else
        	return {0,0,0,1000};
	}
    else
        return {0,0,0,1000};
}

void PCLCloudSearch::VoxelizeCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered)
{
	std::cerr 	<< "PointCloud before filtering: " << cloud->width * cloud->height
       			<< " data points (" << pcl::getFieldsList (*cloud) << ")" << std::endl;

	// Create the filtering object
    pcl::VoxelGrid<pcl::PointXYZ> voxelized;
	voxelized.setInputCloud (cloud);
	voxelized.setLeafSize (0.05f, 0.05f, 0.05f);
	voxelized.filter (*cloudFiltered);

	std::cerr 	<< "PointCloud after filtering: " << cloudFiltered->width * cloudFiltered->height
       			<< " data points (" << pcl::getFieldsList (*cloudFiltered) << ")" << std::endl;

    pcl::io::savePCDFile("cloud-voxelized.pcd", *cloudFiltered);
	std::cerr << "Saved " << cloudFiltered->points.size () << " data points to [voxelized.pcd]" << std::endl;
}
