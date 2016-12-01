#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <time.h>
#include <iostream>
#include <thread>

using namespace cv;
using namespace std;

Mat harriscorner(Mat gray) {
	int depth = gray.depth();
	double scale = 1;
	float k = 0.04f;
	Mat dst = Mat::zeros(gray.size(), CV_32FC1);
	Mat Dx, Dy, dst_norm;
	Sobel(gray, Dx, CV_32F, 1, 0, 3, scale, 0, 4);
	Sobel(gray, Dy, CV_32F, 0, 1, 3, scale, 0, 4);
	Size size = gray.size();
	Mat cov(size, CV_32FC3);
	
	for (int i = 0; i < size.height; i++) {
		float* cov_data = cov.ptr<float>(i);
		float* dxdata = Dx.ptr<float>(i);
		float* dydata = Dy.ptr<float>(i);
		for (int j = 0; j < size.width; j++) {
			float dx = dxdata[j];
			float dy = dydata[j];

			cov_data[j * 3] = dx*dx;
			cov_data[j * 3+1] = dx*dy;
			cov_data[j * 3+2] = dy*dy;
		}
	}

	boxFilter(cov, cov, cov.depth(), Size(2, 2), Point(-1, -1), false, 4);
	
	Size cov_size = cov.size();

	for (int i = 0; i < cov_size.height; i++) {
		float* cov_data = cov.ptr<float>(i);
		float* dst_data = dst.ptr<float>(i);
		
		for (int j = 0; j < cov_size.width; j++) {
			float a = cov_data[j * 3];
			float b = cov_data[j * 3+1];
			float c = cov_data[j * 3+2];
			dst_data[j] = (float)(a*c - b*b - k*(a + c)*(a + c));			
		}
	}
	normalize(dst, dst_norm, 0, 1, NORM_MINMAX, CV_32FC1, Mat());

	return dst_norm;
}

void fast(VideoCapture vc, time_t start) {
	long double frame = 0.0;
	while (waitKey(20) != 27) {
		Mat img2;
		vc >> img2;
		frame++;
		Mat gray;
		cvtColor(img2, gray, COLOR_BGR2GRAY);
		time_t end;
		vector<KeyPoint> points;
		int fast_thresh = 50;
		FAST(gray, points, fast_thresh, true);
		for (int i = 0; i < points.size(); i++) {
			circle(img2, Point(points[i].pt.x, points[i].pt.y), 3, Scalar(0, 0, 255), 2, 8, 0);
		}
		time(&end);
		double fast_seconds = difftime(end, start);
		double fast_fps = frame / fast_seconds;
		string s_fps = format("%.2f", fast_fps);
		string s_points = format("%d", points.size());
		string s_thresh = format("%d", fast_thresh);
		putText(img2, "FAST FPS:" + s_fps + "  Threshold:" + s_thresh + "  Points:" + s_points, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
		imshow("FAST", img2);
	}
}

void cornerHarris_demo(VideoCapture vc, time_t start) {
	long double frame = 0.0;
	while (waitKey(20) != 27) {
		Mat src;
		vc >> src;
		frame++;
		Mat gray;
		time_t end;
		cvtColor(src, gray, COLOR_BGR2GRAY);
		Mat dst, dst_norm, dst_norm_scaled;
		dst = Mat::zeros(src.size(), CV_32FC1);

		/// Detector parameters
		int blockSize = 2;
		int apertureSize = 3;
		double k = 0.04;

		/// Detecting corners
		cornerHarris(gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

		/// Normalizing
		normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(dst_norm, dst_norm_scaled);

		/// Drawing a circle around corners
		int cnt = 0;
		for (int j = 0; j < dst_norm.rows; j++)
		{
			for (int i = 0; i < dst_norm.cols; i++)
			{
				if ((int)dst_norm.at<float>(j, i) > 200)
				{
					circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
					cnt++;
				}
			}
		}
		time(&end);
		double harris_seconds = difftime(end, start);
		double harris_fps = frame / harris_seconds;
		string h_fps = format("%.2f", harris_fps);
		string h_points = format("%d", cnt);
		putText(dst_norm_scaled, "HARRIS FPS:" + h_fps + "  Threshold:200  Points:" + h_points, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(1), 1);
		imshow("CV Harris", dst_norm_scaled);
	}
}

void harris(VideoCapture vc, time_t start) {
	long double frame = 0.0;
	while (waitKey(20) != 27) {
		Mat img1;
		vc >> img1;
		frame++;
		Mat gray;
		cvtColor(img1, gray, COLOR_BGR2GRAY);
		time_t end;
		Mat response;
		int cnt = 0;
		int thresh_max = 3000;
		double harris_thresh = 0.55;
		response = harriscorner(gray);
		for (int j = 0; j < response.rows; j++) {
			if (cnt > thresh_max) break;
			float* res = response.ptr<float>(j);
			for (int i = 0; i < response.cols; i++) {
				if (res[i] > harris_thresh) {
					circle(img1, Point(i, j), 3, Scalar(0, 0, 255), 2, 8, 0);
					cnt++;
				}
			}
		}
		time(&end);
		double harris_seconds = difftime(end, start);
		double harris_fps = frame / harris_seconds;
		string h_fps = format("%.2f", harris_fps);
		string h_points = format("%d", cnt);
		string h_thresh = format("%.2f", harris_thresh);
		putText(img1, "HARRIS FPS:" + h_fps + "  Threshold:" + h_thresh + "  Points:" + h_points, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
		imshow("harris", img1);
	}
}

int main() {
	VideoCapture vc(0);
	if (!vc.isOpened()) return 0;
	time_t start;
	time(&start);
	thread t1(&harris, vc, start);
	thread t2(&fast, vc, start);
	thread t3(&cornerHarris_demo, vc, start);

	t1.join();
	t2.join();
	t3.join();
	return 0;
}