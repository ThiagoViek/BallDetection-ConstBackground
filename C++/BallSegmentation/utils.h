#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class ImageSegmentation
{
public:
	String savePath;
	String img1Path;
	String img2Path;

	ImageSegmentation(String imgFolder, String saveFolder)
	{
		savePath = saveFolder;
		img1Path = imgFolder + "img1.jpg";
		img2Path = imgFolder + "img2.jpg";
	}

	void drawMask()
	{
		Mat img1 = imread(this->img1Path, IMREAD_GRAYSCALE);
		Mat img2 = imread(this->img2Path, IMREAD_GRAYSCALE);

		Mat subtractedImg = Mat::zeros(img2.size(), CV_32F);
		subtract(img1, img2, subtractedImg);

		Mat filteredImg = Mat::zeros(img2.size(), CV_32F);
		Mat cannyImg = Mat::zeros(img2.size(), CV_32F);
		bilateralFilter(subtractedImg, filteredImg, 5, 5, 5);
		Canny(filteredImg, cannyImg, 50, 100);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(cannyImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		int maxX, minX, maxY, minY;
		findKeyPoints(contours, maxX, minX, maxY, minY);

		int radius = ((maxX - minX) + (maxY - minY)) / 4;
		Point center((maxX + minX) / 2, (maxY + minY) / 2);
		Mat maskImg = img1.clone();
		drawCircle(maskImg, center, radius);

		imshow("Ball Contours", maskImg);
		waitKey();
	}

	void findKeyPoints(vector<vector<Point> > contours, int& maxX, int& minX, int& maxY, int& minY)
	{
		list<int> xPoints, yPoints;

		for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
			for (size_t contourPoints = 0; contourPoints < contours[contourIdx].size(); contourPoints++)
			{
				Point coordinates = contours[contourIdx][contourPoints];
				xPoints.push_back(coordinates.x);
				yPoints.push_back(coordinates.y);
			}

		xPoints.sort();
		yPoints.sort();

		maxX = xPoints.back();
		minX = xPoints.front();
		maxY = yPoints.back();
		minY = yPoints.front();
	}

	void drawCircle(Mat& img, Point& center, int& radius)
	{
		const Scalar color = 255;
		int thickness = 2;

		circle(img, center, radius, color, thickness);
	}
};