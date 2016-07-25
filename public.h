#include <opencv2\opencv.hpp>
using namespace cv;

#include <iostream>
#include <vector>
using namespace std;

#ifndef _IC_PUBLIC_H
#define _IC_PUBLIC_H

struct icPoint
{
	enum PointType
	{
		BORDER = 1,
		INNER = 2,
		OUTER = 3
	};
	Point2i pos;
	Point2i from;
	int num;
	vector<int> neighbors;
	PointType type;
};

#endif