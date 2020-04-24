#include <opencv2/opencv.hpp>

int main(void)
{
	cv::Mat img_left, dist_left, img_right, dist_right;
 
	img_left = cv::imread("../img/left.jpg");
	img_right = cv::imread("../img/right.jpg");	
 
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
 
	// 画像上にキーポイントの場所を描く
	// # DrawMatchesFlags::DRAW_RICH_KEYPOINTS  キーポイントのサイズと方向を描く
	cv::drawKeypoints(img_left , key_left , dist_left , cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::drawKeypoints(img_right, key_right, dist_right, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
 
	cv::imshow("left" , dist_left );
	cv::imshow("right", dist_right);
 
	cv::waitKey(0);
	return 0;
}