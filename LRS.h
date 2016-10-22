#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <fstream>
#include <ctime>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef _LRS_H
#define _LRS_H

struct Item
{
	wstring name;
	int count;
	Point pos;
	Scalar color;
	Scalar x;
	int r;
	vector<int> inpart;

	Item(wstring _name, int _count, Point _pos, Scalar _color) :
		name(_name), count(_count), pos(_pos), color(_color){}
	Item(wstring _name, int _count, int x, int y, Scalar _color) :
		name(_name), count(_count), pos(Point(x, y)), color(_color){}
	Item() {}
};

typedef vector<set<int>> _Cluster;

struct Cluster : public _Cluster
{
	vector<wstring> namelist;
	bool _withname = false;

	void takename(vector<wstring>& nl);
	void clearname();
	bool withname();
	Mat makemat();
	void print();
	void fprint(string filename);
};

double qnorm(double value, double sigma = 1.0, double mu = 0.0);

Mat GenerateU(Mat data);
Mat GenerateUs(Mat U);
Mat GenerateSigma(Mat U);
Mat GenerateS(Mat U, Mat sigma);
Cluster LRS(Mat _S, double Mt, double M, double Mr = 0.8);

Mat LRSTools_ViewSubjectRlt(Mat data);
Mat LRSTools_GenerateSignedView(Mat data);
Mat LRSTools_GenerateLogView(Mat data);
void LRSTools_GenerateUIView(Mat _S, Mat dataframe, vector<wstring> namelist);

#endif
