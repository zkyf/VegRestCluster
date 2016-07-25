#include "LRS.h"

//#define DEBUG
//#define DEBUG_MAT_WRITE
//#define DEBUG_MAT_SHOW

#define randn(x) (rand()%(x))

const double SNE = 1E-10;

static int mouse_event;
static int mouse_flags;
static int mouse_counter;
static Point mouse_pos;

void Cluster::takename(vector<wstring> &nl)
{
	namelist = nl;
	_withname = true;
}

void Cluster::clearname()
{
	_withname = false;
	namelist.clear();
}

bool Cluster::withname()
{
	return _withname;
}

Mat Cluster::makemat()
{
	Mat ret;
	return ret;
}

void Cluster::print()
{
	for (int i = 0; i < size(); i++)
	{
		wcout << "Cluster #" << i << ": " << endl;
		for (set<int>::iterator j = this->at(i).begin();
				 j != this->at(i).end(); j++)
		{
			if (_withname) wcout << *j << " " << namelist[*j] << " " << endl;
			else  wcout << *j << " ";
		}
		wcout << endl;
	}
}

void Cluster::fprint(string filename)
{
	wfstream file(filename, ios::out);
	if (file.fail())
	{
		cerr << "Failed to write to file " << filename << endl;
		return;
	}
	for (int i = 0; i < size(); i++)
	{
		file << "Cluster #" << i << ": " << endl;
		for (set<int>::iterator j = this->at(i).begin();
				 j != this->at(i).end(); j++)
		{
			if (_withname) file << *j << " " << namelist[*j] << " " << endl;
			else  file << *j << " ";
		}
		file << endl;
	}
	file.close();
}

bool CmpByValue(const pair<int, double> a, const pair<int, double> b)
{
	return a.second > b.second;
}

double normalCFD(double value)
{
	return 0.5 * erfc(-value * M_SQRT1_2);
}

double qnorm(double value, double sigma, double mu)
{
	double left = -100;
	double right = 100;
	double m = (left + right) / 2;
	while (true)
	{
		double cfd = normalCFD(m);
		if (fabs(cfd - value) <= SNE)
		{
			return m;
		}
		if (cfd < value)
		{
			left = m;
			m = (left + right) / 2;
			continue;
		}
		if (cfd>value)
		{
			right = m;
			m = (left + right) / 2;
			continue;
		}
	}
}

Mat GenerateU(Mat data)
{
#if defined DEBUG
	cout << "GenerateU()" << endl;
#endif
	int rows = data.rows, cols = data.cols;
	Mat U(rows, cols, CV_64F, Scalar::all(0));
	Mat Un = U.clone();
	double maxe = -10000;
	double mine = 10000;
	for (int i = 0; i < cols; i++)
	{
		vector<int> count;
		int maxlevel = -1;
		for (int j = 0; j < rows; j++)
		{
			int level = data.at<uchar>(j, i);
			while (maxlevel < level)
			{
				count.push_back(0);
				maxlevel++;
			}
			count[level]++;
		}
		vector<double> p;
		for (int l = 0; l <= maxlevel; l++)
		{
			double np = 1.0*count[l] / rows;
			if (l == 0) p.push_back(np);
			else p.push_back(np + p[l - 1]);
		}
		vector<double> q;
		for (int l = 0; l<maxlevel; l++)
		{
			q.push_back(qnorm(p[l]));
		}
		vector<double> e;
		for (int l = 0; l <= maxlevel; l++)
		{
			double ne;
			if (l == 0)
			{
				ne = -exp(-q[l] * q[l] / 2) / (sqrt(2 * M_PI)*p[l]);
			}
			else if (l != maxlevel)
			{
				ne = (exp(-q[l - 1] * q[l - 1] / 2) - exp(-q[l] * q[l] / 2)) / (sqrt(2 * M_PI)*(p[l] - p[l - 1]));
			}
			else
			{
				ne = exp(-q[l - 1] * q[l - 1] / 2) / (sqrt(2 * M_PI)*(p[l] - p[l - 1]));
			}
			e.push_back(ne);
			if (maxe < ne) maxe = ne;
			if (mine > ne) mine = ne;
		}
		for (int j = 0; j < rows; j++)
		{
			U.at<double>(j, i) = e[(int)data.at<uchar>(j, i)];
		}
	}
#if defined DEBUG
	Mat toshow = LRSTools_GenerateSignedView(LRSTools_GenerateLogView(U));
#ifdef DEBUG_MAT_SHOW
	imshow("U", U);
	waitKey(0);
#endif
#ifdef DEBUG_MAT_WRITE
	imwrite("U.png", toshow);
#endif
#endif
	return U;
}

Mat GenerateUs(Mat U)
{
	Mat Us;
	Mat devu(U.cols, U.cols, CV_64F, Scalar::all(0));
	for (int i = 0; i < U.cols; i++)
	{
		Mat stddev, mean;
		meanStdDev(U.colRange(i, i+1), mean, stddev);
		devu.at<double>(i, i) = stddev.at<double>(0, 0);
	}
	Us = U*devu;
#if defined DEBUG
#ifdef DEBUG_MAT_SHOW
	imshow("Us", Us);
	waitKey(0);
#endif
#ifdef DEBUG_MAT_WRITE
	Mat towrite = LRSTools_GenerateSignedView(Us);
	imwrite("diff.png", towrite);
#endif
#endif
	return Us;
}

Mat GenerateSigma(Mat U)
{
#if defined DEBUG
	cout << "GenerateSigma()" << endl;
#endif
	Mat sigma, mean;
	calcCovarMatrix(U, sigma, mean, CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_64F);
#if defined DEBUG
#ifdef DEBUG_MAT_SHOW
	imshow("sigma", sigma);
	waitKey(0);
#endif
#ifdef DEBUG_MAT_WRITE
	Mat towrite = LRSTools_GenerateSignedView(LRSTools_GenerateLogView(sigma));
	imwrite("sigma.png", towrite);
#endif
#endif
	return sigma;
}

Mat GenerateS(Mat U, Mat sigma)
{
#if defined DEBUG
	cout << "GenerateS()" << endl;
#endif
	Mat S = U*sigma*U.t();
#if defined DEBUG
	Mat toshow = LRSTools_GenerateSignedView(LRSTools_GenerateLogView(S));
#ifdef DEBUG_MAT_SHOW
	imshow("S", toshow);
	waitKey(0);
#endif
#ifdef DEBUG_MAT_WRITE
	imwrite("S.png", toshow);
#endif
#endif
	return S;
}

Cluster LRS(Mat S, double Mt, double M, double Mr)
{
#if defined DEBUG
	cout << "LRS(): M=" << M << "Mr=" << Mr << endl;
#endif
	Cluster ret;
#if defined DEBUG
	int _debug_counter = 0;
#endif
	while (true)
	{
#if defined DEBUG
		cout << "LRS() counter #" << _debug_counter << endl;
		_debug_counter++;
#endif
		int rows = S.rows, cols = S.cols;
		if (rows != cols) return ret;

		//Find the subject with the largest self-LRS value as the core
		double maxlrs = 0, maxn = -1;
		for (int i = 0; i < rows; i++)
		{
			double nlrs = S.at<double>(i, i);
			if (nlrs>maxlrs)
			{
				maxlrs = nlrs;
				maxn = i;
			}
		}
#if defined DEBUG
		cout << "maxn = " << maxn << endl;
#endif
		if (maxn == -1) return ret;

		//Find the subjects with positive LRS with the selected core and sort
		vector<pair<int, double>> choices;
		for (int i = 0; i < cols; i++)
		{
			if (S.at<double>(i, maxn)<Mt)
			{
				choices.push_back(pair<int, double>(i, S.at<double>(i, maxn)));
#if defined DEBUG
				cout << "Add choice <" << i << ", " << S.at<double>(i, maxn) << ">" << endl;
#endif
			}
		}
		sort(choices.begin(), choices.end(), CmpByValue);

		//Calculate the ratio of the similar subjects in the present cluster
		set<int> cluster;
		cluster.insert(maxn);
		for (int i = 0; i < choices.size(); i++)
		{
			int count = 0;
			for (set<int>::iterator j = cluster.begin();
					 j != cluster.end(); j++)
			{
				if (S.at<double>(choices[i].first, *j)<M) count++;
			}
			double r = count*1.0 / cluster.size();
#if defined DEBUG
			cout << "For subject #" << choices[i].first << "@" << choices[i].second << ",R(h)=" << r << endl;
#endif
			if (r >= Mr) cluster.insert(choices[i].first);
		}

		//Set LRS values of the chosen subjects zero
		for (set<int>::iterator i = cluster.begin();
				 i != cluster.end(); i++)
		{
			S.at<double>(*i, *i) = 0;
		}
		ret.push_back(cluster);
		//system("pause");
	}
}

Mat LRSTools_ViewSubjectRlt(Mat data)
{
	int rows = data.rows;
	int cols = data.cols;
	Mat ret(rows, rows, CV_64F, Scalar::all(0));
	for (int sub1 = 0; sub1 < rows; sub1++)
	{
		int sub1count = 0;
		for (int i = 0; i < cols; i++)
		{
			if (data.at<uchar>(sub1, i)>0)
			{
				sub1count++;
			}
		}
#ifdef DEBUG
		//cout << "sub#" << sub1 << ": " << sub1count << endl;
#endif
		for (int sub2 = 0; sub2 < rows; sub2++)
		{
			for (int i = 0; i < cols; i++)
			{
				if (data.at<uchar>(sub1, i)>0 && data.at<uchar>(sub2, i)>0)
				{
					ret.at<double>(sub1, sub2)++;
				}
			}
			//ret.at<double>(sub1, sub2) = 1.0*ret.at<double>(sub1, sub2) / sub1count;
		}
	}
#if defined DEBUG
	Mat toshow;
	ret.convertTo(ret, CV_64F);
#ifdef DEBUG_MAT_SHOW
	imshow("LRSTools_ViewSubjectRlt", ret);
	waitKey(0);
#endif
#ifdef DEBUG_MAT_WRITE
	Mat towrite;
	ret.convertTo(towrite, CV_8UC1, 255);
	imwrite("LRSTools_ViewSubjectRlt.png", towrite);
#endif
#endif
	return ret;
}

Mat LRSTools_GenerateSignedView(Mat data)
{
	Mat toshow = data.clone();
	double maxs = -1000000;
	double mins = 1000000;
	for (int i = 0; i<data.rows; i++)
	{
		for (int j = 0; j<data.cols; j++)
		{
			if (toshow.at<double>(i, j) > maxs) maxs = toshow.at<double>(i, j);
			if (toshow.at<double>(i, j) < mins) mins = toshow.at<double>(i, j);
		}
	}
	Mat ret(data.size(), CV_8UC3, Scalar::all(0));
	for (int i = 0; i < data.rows; i++)
	{
		for (int j = 0; j < data.cols; j++)
		{
			ret.at<Vec3b>(i, j)[0] = 255;
			ret.at<Vec3b>(i, j)[1] = 255;
			ret.at<Vec3b>(i, j)[2] = 255;
			if (toshow.at<double>(i, j)>0)
			{
				ret.at<Vec3b>(i, j)[0] = 255 - 255 * toshow.at<double>(i, j) / maxs;
				ret.at<Vec3b>(i, j)[1] = 255 - 255 * toshow.at<double>(i, j) / maxs;
			}
			else
			{
				ret.at<Vec3b>(i, j)[0] = 255 - 255 * toshow.at<double>(i, j) / mins;
				ret.at<Vec3b>(i, j)[2] = 255 - 255 * toshow.at<double>(i, j) / mins;
			}
		}
	}
	return ret;
}

Mat LRSTools_GenerateLogView(Mat data)
{
	Mat ret = data.clone();
	for (int i = 0; i < ret.rows; i++)
	{
		for (int j = 0; j < ret.cols; j++)
		{
			if (ret.at<double>(i, j) >= 0)
			{
				ret.at<double>(i, j) = log(1 + ret.at<double>(i, j));
			}
			else
			{
				ret.at<double>(i, j) = -log(1 - ret.at<double>(i, j));
			}
		}
	}
	return ret;
}

Cluster ui_getItemlist(Mat S, double Mt, double M, double Mr, int cthres,
										int width, int height, vector<Item> &itemlist,
										vector<wstring> namelist)
{
	srand(unsigned int(time(0)));
	int *list;
	Cluster cluster = LRS(S, Mt, M, Mr);
	cluster.takename(namelist);
	cluster.fprint("result.txt");
	list = new int[cluster.namelist.size()];
	memset(list, 0, sizeof(int)* cluster.namelist.size());
	for (int i = 0; i < cluster.size(); i++)
	{
		set<int>& nowset = cluster[i];
		if (nowset.size() < cthres) continue;
		for (set<int>::iterator j = nowset.begin();
				 j != nowset.end(); j++)
		{
			list[*j]++;
		}
	}
	for (int i = 0; i < cluster.namelist.size(); i++)
	{
		if (list[i]>=0)
		{
			int x, y;
			x = randn(width - 40) + 20; y = randn(height - 40) + 20;
			int r, g, b;
			r = randn(255); g = randn(255); b = randn(255);
			itemlist.push_back(Item(cluster.namelist[i], list[i], x, y, Scalar(r, g, b)));
		}
	}
	for (int i = 0; i < cluster.size(); i++)
	{
		set<int>& nowset = cluster[i];
		if (nowset.size() < cthres) continue;
		for (set<int>::iterator j = nowset.begin();
				 j != nowset.end(); j++)
		{
			int item = *j;
			itemlist[item].inpart.push_back(i);
		}
	}

	EndPos:
	if (list) delete[] list;

	return cluster;
}

static void mouseCallBack(int event, int x, int y, int flags, void* ustc)
{
	mouse_event = event;
	mouse_pos = Point(x, y);
	mouse_flags = flags;
	if(mouse_event == CV_EVENT_LBUTTONDOWN) mouse_counter++;
	//cout << "mm " << mouse_counter << endl;
}

void LRSTools_GenerateUIView(Mat _S, vector<wstring> namelist)
{
	const int cr = 3;
	const Scalar line_s = Scalar(0, 0, 225);
	const Scalar line_n = Scalar(200, 200, 200);
	const int minr = 5;


	int counter = 0;
	int width = 640;
	int height = 480;
	int cthres = 3;
	int nowselect = -1;
	int nowmove = -1;
	int showcluster = -1;
	int lastclick = mouse_counter;
	int state = 0;
	double Mt = -100000;
	double M = -100000;
	double Mr = 0.8;
	double Md = -100000;
	double alpha = 0.3;
	Cluster cluster;



	// Initialize content
	Mat S = _S.clone();
	string winname = "Cluster View";
	vector<Item> itemlist;
	cluster = ui_getItemlist(S, Mt, M, Mr, cthres, width, height,itemlist, namelist);
	bool *connected = new bool[itemlist.size()];
	bool *seen = new bool[itemlist.size()];
	bool *available = new bool[itemlist.size()];
	for (int i = 0; i < itemlist.size(); i++)
	{
		available[i] = true;
	}
	Mat display(height, width, CV_8UC3, Scalar::all(255));
	imshow(winname, display);
	setMouseCallback(winname, mouseCallBack);
	//cout << itemlist.size() << endl;
	while (1)
	{
		//memset(connected, false, sizeof(connected));
		for (int i = 0; i < itemlist.size(); i++)
		{
			connected[i] = false;
			seen[i] = false;
		}
		display = Mat(height, width, CV_8UC3, Scalar::all(255));
		// draw
		for (int i = 0; i < cluster.size(); i++)
		{
			if (showcluster != -1)
			{
				if (showcluster != i) continue;
			}
			for (set<int>::iterator j = cluster[i].begin();
					 j != cluster[i].end(); j++)
			{
				for (set<int>::iterator kkk = cluster[i].begin();
						 kkk != cluster[i].end(); kkk++)
				{
					if (*j == *kkk) break;
					if (!available[*j] || !available[*kkk]) continue;
					if (S.at<double>(*j, *kkk) >= Md) continue;
					if (itemlist[*j].count <= 0) continue;
					if (itemlist[*kkk].count <= 0) continue;
					if (*j == nowselect || *kkk == nowselect)
					{
						connected[*j] = true;
						connected[*kkk] = true;
						continue;
					}
					else
					{
						if (state == 1) continue;
						line(display, itemlist[*j].pos, itemlist[*kkk].pos, line_n, 1);
						connected[*j] |= false;
						connected[*kkk] |= false;
					}
					if (showcluster != -1 && state != 1)
					{
						stringstream buffer;
						string num;
						buffer << _S.at<double>(*j, *kkk);
						buffer >> num;
						putText(display, num, (itemlist[*j].pos + itemlist[*kkk].pos)/2,
										CV_FONT_HERSHEY_COMPLEX, 0.4, Scalar::all(0));
					}
				}
			}
		}
		for (int i = 0; i < cluster.size(); i++)
		{
			if (showcluster != -1)
			{
				if (showcluster != i) continue;
			}
			for (set<int>::iterator j = cluster[i].begin();
					 j != cluster[i].end(); j++)
			{
				for (set<int>::iterator kkk = cluster[i].begin();
						 kkk != cluster[i].end(); kkk++)
				{
					if (*j == *kkk) break;
					if (!available[*j] || !available[*kkk]) continue;
					if (S.at<double>(*j, *kkk) >= Md) continue;
					if (itemlist[*j].count <= 0) continue;
					if (itemlist[*kkk].count <= 0) continue;
					if (*j == nowselect || *kkk == nowselect)
					{
						line(display, itemlist[*j].pos, itemlist[*kkk].pos, line_s, 1);
						if (showcluster != -1 || state == 1)
						{
							stringstream buffer;
							string num;
							buffer << _S.at<double>(*j, *kkk);
							buffer >> num;
							putText(display, num, (itemlist[*j].pos + itemlist[*kkk].pos) / 2,
											CV_FONT_HERSHEY_COMPLEX, 0.4, Scalar::all(0));
						}
					}
				}
			}
		}
		for (int i = 0; i < itemlist.size(); i++)
		{
			if (!available[i]) continue;
			if (showcluster != -1)
			{
				bool flag = false;
				for (int j = 0; j < itemlist[i].inpart.size(); j++)
				{
					if (itemlist[i].inpart[j] == showcluster)
					{
						flag = true;
						break;
					}
				}
				if (!flag) continue;
			}
			int r = itemlist[i].count + minr;
			itemlist[i].r = r;
			Scalar color;
			if (connected[i])
				color = itemlist[i].color;
			else
				color = alpha * itemlist[i].color + (1 - alpha)*Scalar(255, 255, 255);
			if (itemlist[i].count>0)
			{
				circle(display, itemlist[i].pos, r, color, -1);
				seen[i] = true;
				if (showcluster != -1 || state == 1)
				{
					stringstream buffer;
					buffer << i;
					string num;
					buffer >> num;
					putText(display, num, itemlist[i].pos, CV_FONT_BLACK,
									itemlist[i].count / 15.0 + 0.4,
									Scalar::all(0));
				}
			}
		}

		// event handler
		if (mouse_event == CV_EVENT_LBUTTONDBLCLK)
		{
			if (nowselect != -1)
			{
				state = 1;
			}
			else
			{
				state = 0;
			}
		}
		else if (mouse_event == CV_EVENT_MOUSEMOVE &&
				(mouse_flags & CV_EVENT_FLAG_LBUTTON) &&
				nowselect != -1)
		{
			itemlist[nowselect].pos = mouse_pos;
		}
		else if (lastclick != mouse_counter &&
				(mouse_event == CV_EVENT_LBUTTONDOWN ||
				(mouse_flags & CV_EVENT_FLAG_LBUTTON)))
		{
			//mouse_event = -1;
			//cout << lastclick << ", " << mouse_counter << endl;
			lastclick = mouse_counter;
			int minr = 100;
			int minn = -1;
			for (int i = 0; i < itemlist.size(); i++)
			{
				int x = mouse_pos.x - itemlist[i].pos.x;
				int y = mouse_pos.y - itemlist[i].pos.y;
				int r = x*x + y*y;
				if (r < itemlist[i].r*itemlist[i].r &&
						seen[i])
				{
					minr = r;
					minn = i;
				}
			}
			if (minn == -1) nowselect = -1;
			else
			{
				if (minn != nowselect)
				{
					nowselect = minn;
					wcout << "Selected: Item #" << minn << endl;
					wcout << "Name: " << itemlist[minn].name << endl;
					wcout << "Count: " << itemlist[minn].count << endl;
					wcout << endl;
				}
			}
		}

		imshow(winname, display);
		char choice = waitKey(10);
		string command;
		bool changed = false;
		switch (choice)
		{
			case 27:
				if (state == 0)
				{
					destroyAllWindows();
					return;
				}
				state--;
				break;
			case '/':
				//destroyAllWindows();
				cout << "Command Line" << endl;
				while (true)
				{
					cout << ">> ";
					cin >> command;
					if (command == "size")
					{
						cin >> width >> height;
						//cout << "Recalculating cluster...";
						//cluster = ui_getItemlist(S, Mt, M, Mr, cthres, width, height, itemlist, namelist);
					}
					else if (command == "mt")
					{
						cin >> Mt;
						changed = true;
						//cout << "Recalculating cluster...";
						//cluster = ui_getItemlist(S, Mt, M, Mr, cthres, width, height, itemlist, namelist);
						//cout << endl;
					}
					else if (command == "m")
					{
						cin >> M;
						changed = true;
						//cout << "Recalculating cluster...";
						//cluster = ui_getItemlist(S, Mt, M, Mr, cthres, width, height, itemlist, namelist);
						//cout << endl;
					}
					else if (command == "mr")
					{
						cin >> Mr;
						changed = true;
						//cout << "Recalculating cluster...";
						//cluster = ui_getItemlist(S, Mt, M, Mr, cthres, width, height, itemlist, namelist);
						//cout << endl;
					}
					else if (command == "md")
					{
						cin >> Md;
						changed = true;
					}
					else if (command == "cthres")
					{
						cin >> cthres;
						changed = true;
					}
					else if (command == "alpha")
					{
						cin >> alpha;
					}
					else if (command == "show")
					{
						if (showcluster == -1)
							cout << "Num of clusters: " << cluster.size() << endl;
						else
							cout << "Showing cluster #" << showcluster << endl;
						cout << "Mt: " << Mt << endl;
						cout << "M: " << M << endl;
						cout << "Mr: " << Mr << endl;
						cout << "Md: " << Md << endl;
						cout << "cthres: " << cthres << endl;
						cout << "Size: " << width << " x " << height << endl;
						cout << endl;
						if (showcluster != -1)
						{
							cout << "Including:" << endl;
							for (set<int>::iterator i = cluster[showcluster].begin();
									 i != cluster[showcluster].end(); i++)
							{
								wcout << "Item #" << *i << endl;
								wcout << "Name: " << itemlist[*i].name << endl;
							}
						}
						cout << "Disable list: " << endl;
						for (int i = 0; i < itemlist.size(); i++)
						{
							if (!available[i])
							{
								cout << "#" << i << " is disabled." << endl;
							}
						}
					}
					else if (command == "cluster")
					{
						int x;
						cin >> x;
						if (x < cluster.size() && x >= 0)
							showcluster = x;
						else
						{
							showcluster = -1;
						}
					}
					else if (command == "s")
					{
						int a, b;
						cin >> a >> b;
						cout << "The value of S[" << a << "][" << b << "] is "
							<< _S.at<double>(a, b) << endl;
					}
					else if (command == "switch")
					{
						int x;
						cin >> x;
						if (x >= 0 && x < itemlist.size())
						{
							available[x] = !available[x];
							string xx = (available[x]) ? "TRUE" : "FALSE";
							cout << "The state of #" << x << " is: "
								<< xx << endl;
						}
					}
					else if (command == "exit")
					{
						//imshow(winname, display);
						//setMouseCallback(winname, mouseCallBack);
						if (changed)
						{
							cout << "Recalculating cluster...";
							cluster = Cluster();
							itemlist = vector<Item>();
							Mat S = _S.clone();
							cluster = ui_getItemlist(S, Mt, M, Mr, cthres, width, height, itemlist, namelist);
							cout << endl;
						}
						break;
					}
					else
					{
						cout << "Supported commands: " << endl;
						cout << "show : display the current settings." << endl;
						cout << "mt Mt_value: set the value of Mt" << endl;
						cout << "m M_value: set the value of M" << endl;
						cout << "mr Mr_value: set the value of Mr" << endl;
						cout << "md Md_value: set the value of Md" << endl;
						cout << "cthres cthres_value: set the value of cthres" << endl;
						cout << "size WIDTH HEIGHT: set the window size." << endl;
						cout << "cluster CLUSTER: set the cluster to show." << endl;
						cout << "s A B: show the value of s[A][B]." << endl;
						cout << "alpha ALPHA: set the alpha for unselected items." << endl;
						cout << "exit: recalculate the clusters and leave command mode." << endl;
					}
				}
				break;
			default:imshow(winname, display);
		}
	}
}