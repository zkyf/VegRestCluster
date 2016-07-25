#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <opencv2\opencv.hpp>

#define _LRS_USE_WSTRING
#include "LRS.h"

using namespace std;
using namespace cv;

string inpath;
wfstream infile;
int namecount= 0;
int daycount = 0;
int linecount = 0;
map<wstring, int> namepairs;
map<wstring, int> datepairs;
vector<wstring> namelist;
vector<wstring> datelist;
Mat dataframe;

int main(int argc, char **argv)
{
	if (argc == 1)
	{
		cout << "Please specify the input file: " << endl;
		cin >> inpath;
	}
	else if (argc > 2)
	{
		cout << "The program only accepts 2 parameters" << endl;
		return 1;
	}
	else
	{
		inpath = argv[1];
	}
	infile.open(inpath, ios::in);
	if (infile.fail())
	{
		cout << "Cannot open file " << inpath << endl;
		cout << "Please check the path" << endl;
		return 2;
	}
	wstring name, date;
	wcout << "inputing" << endl;
	infile >> date >> name;
	while (!infile.eof())
	{
		//wcout << date << endl << name << endl;
		//system("pause");
		linecount++;
		if (namepairs.find(name) == namepairs.end())
		{
			namepairs.insert(pair<wstring, int>(name, namecount++));
			namelist.push_back(name);
		}
		if (datepairs.find(date) == datepairs.end())
		{
			datepairs.insert(pair<wstring, int>(date, daycount++));
			datelist.push_back(date);
		}
		infile >> date >> name;
	}
	wcout << "endinput" << endl;
	wcout << "linecount=" << linecount << endl;
	wcout << "namecount=" << namecount << endl;
	wcout << "datecount=" << daycount << endl;
	infile.close();

	infile.open(inpath, ios::in);
	if (infile.fail())
	{
		cout << "Cannot open file " << inpath << endl;
		cout << "Please check the path" << endl;
		return 2;
	}
	dataframe = Mat(daycount, namecount, CV_8UC1, Scalar::all(0));
	Mat toshow = dataframe.clone();
	for (int i = 0; i < linecount; i++)
	{
		infile >> date >> name;
		int daten = datepairs[date];
		int namen = namepairs[name];
		dataframe.at<uchar>(daten, namen) = (uchar)1;
		toshow.at<uchar>(daten, namen) = (uchar)255;
	}
	dataframe = dataframe.t();
	toshow = toshow.t();
	//imshow("dataframe", toshow);
	//waitKey(0);
	//imwrite("dataframe.png", toshow);
	
	LRSTools_ViewSubjectRlt(dataframe);

	Mat U = GenerateU(dataframe);

	Mat sigma = GenerateSigma(U);

	Mat S = GenerateS(U, sigma);

#define thresdate 25000
#define thresname 100000

	//Cluster cluster = LRS(S, thresname, thresname);

	//cluster.takename(namelist);

	//cluster.fprint("result.txt");

	//system("pause");

	LRSTools_GenerateUIView(S, namelist);
	return 0;
}