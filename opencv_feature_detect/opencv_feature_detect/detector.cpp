#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

Mat harris(Mat input) {
	int col = input.cols;
	int row = input.rows;
	int ch = input.channels();
	Mat output(row, col, CV_8UC3);

	for (int y = 0; y < row; ++y) {
		for (int x = 0; x < col; ++x) {

		}
	}


	return output;
}

int main() {
	Mat input;
	VideoCapture vc(0);
	if (!vc.isOpened()) return 0;

	while (1) {
		vc >> input;
		if (input.empty()) break;
		int col = input.cols;
		int row = input.rows;
		Mat img1(row, col, CV_8UC3);
		Mat img2(row, col, CV_8UC3);
		
		img1 = harris(img1);

		

		
		
		putText(img1, "FPS : ", Point2f(0, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 1);
		putText(img2, "FPS : ", Point2f(0, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 1);
		Mat disp;
		hconcat(img1, img2, disp);
		
		imshow("detection", disp);
		if (waitKey(10) == 27) break;
	}
	destroyAllWindows();

	return 0;
}