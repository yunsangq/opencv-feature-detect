#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

Mat harris(Mat gray) {
	Mat output;
	Mat gradx, grady;
	int patchSize = 2;
	float k = 0.04, thresh = 10;
	int scale = 1, delta = 0, ddepth = CV_32F;
	Sobel(gray, gradx, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(gray, grady, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	
	Mat gradxx, gradyy, gradxy, det, trace, response, temp;

	multiply(gradx, gradx, gradxx);
	multiply(grady, grady, gradyy);
	multiply(gradx, grady, gradxy);

	for (int j = 0; j < gradxx.rows - patchSize; j++) {
		for (int i = 0; i < gradxx.cols - patchSize; i++) {
			for (int patch_y = j; patch_y < patchSize; patch_y++) {
				for (int patch_x = i; patch_x < patchSize; patch_x++) {
					if (!(patch_y == j&&patch_x == i)) {
						
					}
				}
			}
		}
	}






	return output;
}



int main() {
	Mat input;
	VideoCapture vc(0);
	if (!vc.isOpened()) return 0;
	time_t harris_start, harris_end, fast_start, fast_end;
	long double frame = 0.0;
	time(&harris_start);
	time(&fast_start);
	while (1) {
		vc >> input;
		if (input.empty()) break;
		frame++;
		int col = input.cols;
		int row = input.rows;
		Mat img1(row, col, CV_8UC3);
		Mat img2(row, col, CV_8UC3);
		Mat gray;
		cvtColor(input, gray, COLOR_BGR2GRAY);
		
		img1 = harris(gray);



		//FAST/////////////////////
		img2 = input.clone();
		vector<KeyPoint> points;
		FAST(gray, points, 10);
		for (int i = 0; i < points.size(); i++) {
			circle(img2, Point(points[i].pt.x, points[i].pt.y), 3, Scalar(0, 255, 0), 1, 8, 0);
		}		
		time(&fast_end);
		double fast_seconds = difftime(fast_end, fast_start);
		double fast_fps = frame / fast_seconds;
		string s_fps = format("%.2f", fast_fps);
		string s_points = format("%d", points.size());
		putText(img2, "FAST FPS:" + s_fps + "  Threshold:10" + "  Points:" + s_points, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
		/////////////////////
		
		Mat disp;
		hconcat(input, img2, disp);
		
		imshow("detection", disp);
		
		if (waitKey(10) == 27) break;
	}
	cv::destroyAllWindows();

	return 0;
}