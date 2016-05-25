#ifndef __PNPSOLVER_H_INCLUDED__   // if x.h hasn't been included yet...
#define __PNPSOLVER_H_INCLUDED__

#include <opencv2/features2d.hpp>
#include <initializer_list>
#include <iostream>
class PnPSolver
{
public:
	PnPSolver();
	PnPSolver(int iterCount, int repError, double confidence);

	void setPnPParam(int iterCount, int repError, double confidence);
	void setWorldPoints();
	void setWorldPoints(std::vector<cv::Point3d> WP);
	std::vector<cv::Point3d> getWorldPoints();

	void setImagePoints();
	void setImagePoints(std::vector<cv::Point2d> IP);
	std::vector<cv::Point2d> getImagePoints();

	cv::Mat getCameraPose();
	cv::Mat getCameraPosition();

	cv::Mat getRotationVector();
	cv::Mat getTranslationVector();

	cv::Mat getRotationMatrix();
	cv::Mat getTranslationMatrix();

	cv::Mat getEssentialMatrix();
	cv::Mat getFundamentalMatrix();

	void setVoVImagePoints();
	std::vector< std::vector<cv::Point2d> > getVoVImagePoints();

	int run(int verbal);

	std::vector<cv::Mat> camPositions;
	cv::Mat cameraPose34;
	cv::Mat R, t;

private:

	std::vector<cv::Point2d> imagePoints, imagePoints2;
	std::vector<cv::Point3d> worldPoints;
	std::vector<std::vector<cv::Point2d> > VoVImagePoints;

	cv::Mat essentialMatrix, fundamentalMatrix;

	cv::Mat rVec, tVec;
	cv::Mat rMat, tMat;
	cv::Mat cameraPose, cameraPosition;

	int paramIterCount;
	int paramRepError;
	double paramConfidence;
};

#endif
