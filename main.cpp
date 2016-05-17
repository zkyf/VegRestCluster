#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>

using namespace std;

typedef vector<bool> record;

string inpath;
wfstream infile;
int namecount= 0;
int daycount = 0;
map<wstring, int> pairs;
vector<wstring> namelist;
vector<record> table;

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
		if (pairs.find(name)==pairs.end())
		{
			pairs.insert(pair<wstring, int>(name, namecount++));
			namelist.push_back(name);
		}
		infile >> date >> name;
	}
	wcout << "endinput" << endl;
	wcout << "count=" << namecount << endl;
	//for (int i = 0; i < namecount; i++)
	//{
	//	wcout << namelist[i] << endl;
	//}
	infile.seekg(0);
	wstring ldate;
	infile >> date >> name;
	ldate = date;
	record sub = record(namecount);
	for (int i = 0; i < namecount; i++)
	{
		sub[i] = false;
	}
	while (!infile.eof())
	{
		if (ldate == date)
		{
			sub[pairs[name]] = true;
		}
		else
		{
			table.push_back(sub);
			sub = record(namecount);
			for (int i = 0; i < namecount; i++)
			{
				sub[i] = false;
			}
			ldate = date;
			sub[pairs[name]] = true;
		}
		infile >> date >> name;
	}

	system("pause");
	return 0;
}