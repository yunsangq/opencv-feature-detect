#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <cmath>
#include <iomanip>
#include <time.h>
#include <iostream>

using namespace cv;
using namespace std;

void Gaussian(float kernel[][3]) {
	float sum = 0.0, temp, sigma = 1;
	for (int x = -1; x <= 1; x++) {
		for (int y = -1; y <= 1; y++) {
			temp = exp(-(x*x + y*y) / (2 * pow(sigma, 2)));
			kernel[x + 1][y + 1] = temp / ((float)CV_PI * 2 * pow(sigma, 2));
			sum += kernel[x + 1][y + 1];
		}
	}
}

void Gaussian_filter(Mat &image) {
	float kernel[3][3];
	float* img = (float*)image.data;
	Gaussian(kernel);
	for (int j = 1; j < image.rows - 1; j++) {
		for (int i = 1; i < image.cols - 1; i++) {
			float sum = 0;
			for (int y = 0; y < 3; y++) {
				for (int x = 0; x < 3; x++) {
					sum += img[(j - 1 + y)*image.cols + (i - 1 + x)] * kernel[x][y];
				}
				img[j*image.cols + i] = sum;
			}
		}
	}
}

Mat harris(Mat gray) {
	Mat gradx, grady;
	int patchSize = 2;
	float k = 0.04f;
	int scale = 1, delta = 0, ddepth = CV_32F;
	Sobel(gray, gradx, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(gray, grady, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	
	Mat gradxx, gradyy, gradxy, det, trace, response, temp;

	multiply(gradx, gradx, gradxx);
	multiply(grady, grady, gradyy);
	multiply(gradx, grady, gradxy);

	float* xx = (float*)gradxx.data;
	float* yy = (float*)gradyy.data;
	float* xy = (float*)gradxy.data;
	for (int j = 0; j < gradxx.rows - patchSize; j++) {
		for (int i = 0; i < gradxx.cols - patchSize; i++) {
			for (int patch_y = j; patch_y < patchSize; patch_y++) {
				for (int patch_x = i; patch_x < patchSize; patch_x++) {
					if (!(patch_y == j&&patch_x == i)) {
						xx[j*gradxx.cols + i] += xx[patch_y*gradxx.cols + patch_x];
						xy[j*gradxx.cols + i] += xy[patch_y*gradxx.cols + patch_x];
						xx[j*gradxx.cols + i] += xx[patch_y*gradxx.cols + patch_x];
					}
				}
			}
		}
	}

	Gaussian_filter(gradxx);
	Gaussian_filter(gradyy);
	Gaussian_filter(gradxy);
	//imshow("gradxx", gradxx);
	//imshow("gradyy", gradyy);
	//imshow("gradxy", gradxy);

	multiply(gradxx, gradyy, det);
	//imshow("det", det);
	multiply(gradxy, gradxy, temp);
	det -= temp;
	trace = gradxx + gradyy;
	multiply(trace, trace, trace, k);
	response = det - trace;
	//imshow("response", response);
	normalize(response, response, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	
	return response;
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
		Mat response;
		cvtColor(input, gray, COLOR_BGR2GRAY);
		///*
		img1 = input.clone();
		int cnt = 0;
		response = harris(gray);

		float* R = (float*)response.data;
		for (int j = 0; j < response.rows; j++) {
			for (int i = 0; i < response.cols; i++) {
				if ((int)R[j*response.cols + i] > 180) {
					circle(img1, Point2f(i, j), 3, Scalar(0, 0, 255), 2, 8, 0);
					cnt++;
				}
			}
		}

		time(&harris_end);
		double harris_seconds = difftime(harris_end, harris_start);
		double harris_fps = frame / harris_seconds;
		string h_fps = format("%.2f", harris_fps);
		string h_points = format("%d", cnt);
		putText(img1, "HARRIS FPS:" + h_fps + "  Threshold:200" + "  Points:" + h_points, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
		//*/
		//FAST/////////////////////
		///*
		img2 = input.clone();
		vector<KeyPoint> points;
		FAST(gray, points, 100);
		for (int i = 0; i < points.size(); i++) {
			circle(img2, Point(points[i].pt.x, points[i].pt.y), 3, Scalar(0, 0, 255), 2, 8, 0);
		}		
		time(&fast_end);
		double fast_seconds = difftime(fast_end, fast_start);
		double fast_fps = frame / fast_seconds;
		string s_fps = format("%.2f", fast_fps);
		string s_points = format("%d", points.size());
		putText(img2, "FAST FPS:" + s_fps + "  Threshold:10" + "  Points:" + s_points, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
		//*/
		/////////////////////
		
		Mat disp;
		hconcat(img1, img2, disp);
		imshow("detection", disp);

		//imshow("detection", img1);
		//imshow("detection", img2);

		if (waitKey(33) == 27) break;
	}
	cv::destroyAllWindows();
	return 0;
}