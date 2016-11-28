/// Global variables
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

const char* source_window = "Source image";
const char* corners_window = "Corners detected";

/// Function header
void cornerHarris_demo(int, void*);

/**
* @function main
*/
int main(int, char** argv)
{
	/// Load source image and convert it to gray
	src = imread(argv[1], 1);
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	/// Create a window and a trackbar
	namedWindow(source_window, WINDOW_AUTOSIZE);
	createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
	imshow(source_window, src);

	cornerHarris_demo(0, 0);

	waitKey(0);
	return(0);
}

/**
* @function cornerHarris_demo
* @brief Executes the corner detection and draw a circle around the possible corners
*/
void cornerHarris_demo(int, void*)
{

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result
	namedWindow(corners_window, WINDOW_AUTOSIZE);
	imshow(corners_window, dst_norm_scaled);
}

void cv::cornerHarris(InputArray _src, OutputArray _dst, int blockSize, int ksize, double k, int borderType)
{

	Mat src = _src.getMat();
	_dst.create(src.size(), CV_32FC1);
	Mat dst = _dst.getMat();


	cornerEigenValsVecs(src, dst, blockSize, ksize, HARRIS, k, borderType);
}

static void
cornerEigenValsVecs(const Mat& src, Mat& eigenv, int block_size,
	int aperture_size, int op_type, double k = 0.,
	int borderType = BORDER_DEFAULT)
{
	int depth = src.depth();
	double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
	if (aperture_size < 0)
		scale *= 2.0;
	if (depth == CV_8U)
		scale *= 255.0;
	scale = 1.0 / scale;

	CV_Assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);

	Mat Dx, Dy;
	if (aperture_size > 0)
	{
		Sobel(src, Dx, CV_32F, 1, 0, aperture_size, scale, 0, borderType);
		Sobel(src, Dy, CV_32F, 0, 1, aperture_size, scale, 0, borderType);
	}
	else
	{
		Scharr(src, Dx, CV_32F, 1, 0, scale, 0, borderType);
		Scharr(src, Dy, CV_32F, 0, 1, scale, 0, borderType);
	}

	Size size = src.size();
	Mat cov(size, CV_32FC3);
	int i, j;

	for (i = 0; i < size.height; i++)
	{
		float* cov_data = cov.ptr<float>(i);
		const float* dxdata = Dx.ptr<float>(i);
		const float* dydata = Dy.ptr<float>(i);
		j = 0;

		for (; j < size.width; j++)
		{
			float dx = dxdata[j];
			float dy = dydata[j];

			cov_data[j * 3] = dx*dx;
			cov_data[j * 3 + 1] = dx*dy;
			cov_data[j * 3 + 2] = dy*dy;
		}
	}

	boxFilter(cov, cov, cov.depth(), Size(block_size, block_size),
		Point(-1, -1), false, borderType);

	if (op_type == MINEIGENVAL)
		calcMinEigenVal(cov, eigenv);
	else if (op_type == HARRIS)
		calcHarris(cov, eigenv, k);
	else if (op_type == EIGENVALSVECS)
		calcEigenValsVecs(cov, eigenv);
}

void cv::Sobel(InputArray _src, OutputArray _dst, int ddepth, int dx, int dy,
	int ksize, double scale, double delta, int borderType)
{
	int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
	if (ddepth < 0)
		ddepth = sdepth;
	int dtype = CV_MAKE_TYPE(ddepth, cn);
	_dst.create(_src.size(), dtype);

#ifdef HAVE_TEGRA_OPTIMIZATION
	if (tegra::useTegra() && scale == 1.0 && delta == 0)
	{
		Mat src = _src.getMat(), dst = _dst.getMat();
		if (ksize == 3 && tegra::sobel3x3(src, dst, dx, dy, borderType))
			return;
		if (ksize == -1 && tegra::scharr(src, dst, dx, dy, borderType))
			return;
	}
#endif

	CV_IPP_RUN(true, ipp_sobel(_src, _dst, ddepth, dx, dy, ksize, scale, delta, borderType));

	int ktype = std::max(CV_32F, std::max(ddepth, sdepth));

	Mat kx, ky;
	getDerivKernels(kx, ky, dx, dy, ksize, false, ktype);
	if (scale != 1)
	{
		// usually the smoothing part is the slowest to compute,
		// so try to scale it instead of the faster differenciating part
		if (dx == 0)
			kx *= scale;
		else
			ky *= scale;
	}
	sepFilter2D(_src, _dst, ddepth, kx, ky, Point(-1, -1), delta, borderType);
}

static void getSobelKernels(OutputArray _kx, OutputArray _ky,
	int dx, int dy, int _ksize, bool normalize, int ktype)
{
	int i, j, ksizeX = _ksize, ksizeY = _ksize;
	if (ksizeX == 1 && dx > 0)
		ksizeX = 3;
	if (ksizeY == 1 && dy > 0)
		ksizeY = 3;

	CV_Assert(ktype == CV_32F || ktype == CV_64F);

	_kx.create(ksizeX, 1, ktype, -1, true);
	_ky.create(ksizeY, 1, ktype, -1, true);
	Mat kx = _kx.getMat();
	Mat ky = _ky.getMat();

	if (_ksize % 2 == 0 || _ksize > 31)
		CV_Error(CV_StsOutOfRange, "The kernel size must be odd and not larger than 31");
	std::vector<int> kerI(std::max(ksizeX, ksizeY) + 1);

	CV_Assert(dx >= 0 && dy >= 0 && dx + dy > 0);

	for (int k = 0; k < 2; k++)
	{
		Mat* kernel = k == 0 ? &kx : &ky;
		int order = k == 0 ? dx : dy;
		int ksize = k == 0 ? ksizeX : ksizeY;

		CV_Assert(ksize > order);

		if (ksize == 1)
			kerI[0] = 1;
		else if (ksize == 3)
		{
			if (order == 0)
				kerI[0] = 1, kerI[1] = 2, kerI[2] = 1;
			else if (order == 1)
				kerI[0] = -1, kerI[1] = 0, kerI[2] = 1;
			else
				kerI[0] = 1, kerI[1] = -2, kerI[2] = 1;
		}
		else
		{
			int oldval, newval;
			kerI[0] = 1;
			for (i = 0; i < ksize; i++)
				kerI[i + 1] = 0;

			for (i = 0; i < ksize - order - 1; i++)
			{
				oldval = kerI[0];
				for (j = 1; j <= ksize; j++)
				{
					newval = kerI[j] + kerI[j - 1];
					kerI[j - 1] = oldval;
					oldval = newval;
				}
			}

			for (i = 0; i < order; i++)
			{
				oldval = -kerI[0];
				for (j = 1; j <= ksize; j++)
				{
					newval = kerI[j - 1] - kerI[j];
					kerI[j - 1] = oldval;
					oldval = newval;
				}
			}
		}

		Mat temp(kernel->rows, kernel->cols, CV_32S, &kerI[0]);
		double scale = !normalize ? 1. : 1. / (1 << (ksize - order - 1));
		temp.convertTo(*kernel, ktype, scale);
	}
}

}
