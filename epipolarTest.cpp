#include <opencv2/opencv.hpp>
#include <vector>
#include <typeinfo>

void drawEpilines(cv::Mat img1
		, cv::Mat img2
		, std::vector<cv::Vec3f> lines
		, std::vector<cv::Point2f> pts1
		, std::vector<cv::Point2f> pts2
		, cv::Mat& img3
		, cv::Mat& img4)
{
	cv::cvtColor(img1, img3, cv::COLOR_GRAY2BGR);
	cv::cvtColor(img2, img4, cv::COLOR_GRAY2BGR);

	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Scalar clr(rand() % 255,rand() % 255,rand() % 255);
		cv::Point p0(0, -lines[i][2]/lines[i][1]);
		cv::Point p1(img1.cols, -(lines[i][2] + lines[i][0]*img1.cols)/lines[i][1]);
		cv::line(img3, p0, p1, clr, 1);
		cv::circle(img3, pts1[i], 5, clr, -1);
		cv::circle(img4, pts2[i], 5, clr, -1);
	}
}

int main(void)
{
	cv::Mat img_left, dist_left, img_right, dist_right;
 
	img_left = cv::imread("../img/left.jpg", 0);
	img_right = cv::imread("../img/right.jpg", 0);	
 
	// 特徴点検出アルゴリズムの選択
	cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 2);
	
	// 検出したキーポイント（特徴点）を格納する配列
	// std::vector<cv::KeyPoint> keyAkaze, key_left, keySurf;
	std::vector<cv::KeyPoint> key_left, key_right;
	cv::Mat des_left, des_right; //特徴量

	// キーポイント検出
	// akaze->detectAndCompute(img_left, cv::noArray(), keyAkaze);
	orb->detectAndCompute(img_left , cv::noArray(), key_left , des_left );
	orb->detectAndCompute(img_right, cv::noArray(), key_right, des_right);
 
 	//-- Step 2: Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
	// std::vector< cv::DMatch > matches;
	// matcher.match( des_left, des_right, matches, 2);
	// cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<cv::DMatch> > matches;
	matcher.knnMatch( des_left, des_right, matches, 2);

	std::cout << "match done." << std::endl;
	std::cout << "size " << matches.size() << std::endl;

	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.75f;
	std::vector<cv::DMatch> good_matches;
	std::vector<cv::Point2f> pts_left, pts_right;

	for (size_t i = 0; i < matches.size(); i++)
	{
        	if ( !matches[i].empty() && matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        	{
			std::cout << i << " dist " << matches[i][0].distance << "/" << ratio_thresh * matches[i][1].distance << std::endl;
			good_matches.push_back(matches[i][0]);
			pts_right.push_back(key_right[matches[i][0].trainIdx].pt);
			pts_left.push_back(key_left[matches[i][0].queryIdx].pt);
        	}
    	}

	std::vector<uchar> mask;
	std::vector<cv::Point2f> good_pts_left, good_pts_right;

	cv::Mat fundamental_matrix = cv::findFundamentalMat(pts_left, pts_right, mask, cv::FM_LMEDS);
	for (size_t i = 0; i < mask.size(); i++)
	{
		if(mask[i] == 1)
		{
			good_pts_left.push_back(pts_left[i]);
			good_pts_right.push_back(pts_right[i]);
		}
	}

	std::vector<cv::Vec3f> lines_left;
	cv::computeCorrespondEpilines(good_pts_right, 2, fundamental_matrix, lines_left);

	cv::Mat img1, img2;
	drawEpilines(img_left, img_right, lines_left, good_pts_left, good_pts_right, img1, img2);

	std::vector<cv::Vec3f> lines_right;
	cv::computeCorrespondEpilines(good_pts_left, 1, fundamental_matrix, lines_right);

	cv::Mat img3, img4;
	drawEpilines(img_right, img_left, lines_right, good_pts_right, good_pts_left, img3, img4);

	cv::imshow("img1", img1);
	cv::imshow("img3", img3);
 
	cv::waitKey(0);
	return 0;
}
