#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <optixu/optixu_math_namespace.h>
#include <stdio.h>
#include "helper_cuda.h"

#include <math.h>
#include <assert.h>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <ctime>

#define border (1000000000000.0)
#define minerBorder (100000000000.0)

const int maxThreads = 256;
__device__ const int maxEdgeCnt = 32;

//二维平面上的点
struct Point {
	double x, y;
	Point(int x) {}
	__device__ Point(double x_ = 0.0, double y_ = 0.0) : x(x_), y(y_) {}
	__device__ double cross(const Point &a, const Point &b) const
	{
		return (a.x - x) * (b.y - y) - (a.y - y) * (b.x - x);
	}
	double cross_host(const Point &a, const Point &b) const
	{
		return (a.x - x) * (b.y - y) - (a.y - y) * (b.x - x);
	}
	__device__ double len2()
	{
		return x * x + y * y;
	}
	__device__ void normalize()
	{
		double l = sqrt(x * x + y * y);
		x /= l;
		y /= l;
	}
	void print() const
	{
		fprintf(stderr, "%.4f %.4f\n", x, y);
	}
	void dprint() const
	{
		fprintf(stdout, "%.4f %.4f\n", x, y);
	}
}; 
bool operator < (const Point &a, const Point &b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }
__device__ Point operator - (const Point &a, const Point &b) { return Point(a.x - b.x, a.y - b.y); }
__device__ Point operator + (const Point &a, const Point &b) { return Point(a.x + b.x, a.y + b.y); }
__device__ Point operator * (const Point &a, const double &b) { return Point(a.x * b, a.y * b); }
__device__ Point operator * (const double &b, const Point &a) { return Point(a.x * b, a.y * b); }
__device__ bool operator == (const Point &a, const Point &b) { return a.x == b.x && a.y == b.y; }
__device__ double dis2(const Point &a, const Point &b) { return (a - b).len2(); }

struct PointWithID : public Point {
	int ID;
};

__device__ const double eps = 1e-8;
int total;

__device__ int dblcmp(double x)
{
	if (fabs(x) < eps) return 0;
	return (x > 0) * 2 - 1;
}

PointWithID *d_CUDA;//点的坐标数据
int *g_CUDA;//每个点的Voronoi图的边构成的双向链表的入口
int *pos_CUDA;//每个点双向链表下标的起始范围
double *Area_CUDA;

struct Edge {
	Point center;
	Point direction;
	__device__ Edge(Point x_ = Point(0, 0), Point y_ = Point(0, 0)) : center(x_), direction(y_) {}
	void print() const
	{
		fprintf(stderr, "Center : ");
		center.print();
		fprintf(stderr, "Direct : ");
		direction.print();
	}
};//中垂线数据：中心、方向

//每个点的Voronoi图的边的具体数据
struct EdgeData {
	Edge e;//中垂线
	int adj, previous, next;//中垂线对面的点的编号、逆时针的前一条边、逆时针的后一条边
	int eadj;//中垂线对面的点的这条边的编号
	void print() const
	{
		fprintf(stderr, "adj = %d; previous = %d; next = %d\n", adj, previous, next);
	}
} *edge_CUDA;

struct Frag {
	int l, r;
	Point a, b, c; 
} *frag, *frag_CUDA;//片元的数据：点在d中的下标范围、三角网的三个顶点的坐标。初始化后不应该被修改。
bool cmp1(const Frag &a, const Frag &b) { return a.l < b.l; }

struct Segment {
	Point a, b;
	__device__ Segment(Point x_ = Point(0, 0), Point y_ = Point(0, 0)) : a(x_), b(y_) {}
};

int random(int a, int b) { return ((rand() << 15) + rand()) % (b - a + 1) + a; }

int qwerty;
Point ranPoint()
{
	Point tmp(1);
	tmp.x = rand() % 1000 * 10;
	tmp.y = rand() % 1000 * 10;
	//tmp.x = (double)rand() / RAND_MAX * 100;
	//tmp.y = (double)rand() / RAND_MAX * 100;
	return tmp;
}

struct myPair {
	bool first;
	Point second;
	__device__ myPair(bool x = false, Point y = 0): first(x), second(y) {}
};

__device__ bool checkIntersect(const Segment &l1, const Segment &l2)
{
	return (dblcmp(l1.a.cross(l2.a, l2.b)) * dblcmp(l1.b.cross(l2.a, l2.b)) < 0 && dblcmp(l2.a.cross(l1.a, l1.b)) * dblcmp(l2.b.cross(l1.a, l1.b)) < 0);
}

__device__ myPair intersectPoint(const Segment &l1, const Segment &l2)
{
	myPair res(0, Point(0, 0));
	bool tmp = checkIntersect(l1, l2);
	if (!tmp) return res;
	double c1 = Point(0, 0).cross(l1.a, l1.b);
	double c2 = Point(0, 0).cross(l2.a, l2.b);
	double a1 = l1.a.y - l1.b.y;
	double b1 = l1.b.x - l1.a.x;
	double a2 = l2.a.y - l2.b.y;
	double b2 = l2.b.x - l2.a.x;
	double D = a1 * b2 - a2 * b1;
	Point p;
	p.x = (b1 * c2 - b2 * c1) / D;
	p.y = (c1 * a2 - c2 * a1) / D;
	res.first = 1;
	res.second = p;
	return res;
}

__device__ void getBorderPoint(const Edge &e, Point &a, Point &b)
{
	if (fabs(e.direction.x) > fabs(e.direction.y))
	{
		a = Point(-border, (-border - e.center.x) * e.direction.y / e.direction.x + e.center.y);
		b = Point(border, (border - e.center.x) * e.direction.y / e.direction.x + e.center.y);
	}
	else {
		a = Point((-border - e.center.y) * e.direction.x / e.direction.y + e.center.x, -border);
		b = Point((border - e.center.y) * e.direction.x / e.direction.y + e.center.x, border);
	}
}

__device__ void getMinerBorderPoint(const Edge &e, Point &a, Point &b)
{
	if (fabs(e.direction.x) > fabs(e.direction.y))
	{
		a = Point(-minerBorder, (-minerBorder - e.center.x) * e.direction.y / e.direction.x + e.center.y);
		b = Point(minerBorder, (minerBorder - e.center.x) * e.direction.y / e.direction.x + e.center.y);
	}
	else {
		a = Point((-minerBorder - e.center.y) * e.direction.x / e.direction.y + e.center.x, -minerBorder);
		b = Point((minerBorder - e.center.y) * e.direction.x / e.direction.y + e.center.x, minerBorder);
	}
}

__device__ myPair intersectPoint(const Edge &e1, const Segment &l2)
{
	Point a1, b1;
	getBorderPoint(e1, a1, b1);
	return intersectPoint(Segment(a1, b1), l2);
}

__device__ myPair intersectPoint(const Edge &e1, const Edge &e2)
{
	Point a1, b1, a2, b2;
	getBorderPoint(e1, a1, b1);
	getBorderPoint(e2, a2, b2);
	return intersectPoint(Segment(a1, b1), Segment(a2, b2));
}

__device__ Edge verticalLine(const Point &a, const Point &b)
{
	Point dir(a.y - b.y, b.x - a.x);
	Point cen = (a + b) * 0.5;
	dir.normalize();
	return Edge(cen, dir);
}

__device__ void calculateUpConnectedTangent(int l1, int r1, int l2, int r2, Point *convexHull, int &x, int &y)
{
	x = (l1 + r1) / 2;
	y = (l2 + r2) / 2;
	double midx = (convexHull[r1].x + convexHull[l2].x) / 2.0;
	bool flag1, flag2;
	while (!(
			(flag1 = (x == l1 || dblcmp(convexHull[x].cross(convexHull[x - 1], convexHull[y])) > 0))
			&&
			(flag2 = (y == r2 || dblcmp(convexHull[y].cross(convexHull[y + 1], convexHull[x])) < 0))
			&&
			(x == r1 || dblcmp(convexHull[x].cross(convexHull[x + 1], convexHull[y])) >= 0)
			&&
			(y == l2 || dblcmp(convexHull[y].cross(convexHull[y - 1], convexHull[x])) <= 0)
			))
	{
		if (!flag1) x = x - 1;
		else if (!flag2) y = y + 1;
		else {
			if (x == r1) y = y - 1;
			else if (y == l2) x = x + 1;
			else {
				Edge e_x = Edge(convexHull[x], convexHull[x + 1] - convexHull[x]);
				e_x.direction.normalize();
				myPair res1 = intersectPoint(e_x, Segment(Point(midx, -border), Point(midx, border)));
				double h1;
				if (res1.first) h1 = res1.second.y; else h1 = ((convexHull[x].y < convexHull[x + 1].y) * 2 - 1) * border;

				Edge e_y = Edge(convexHull[y], convexHull[y - 1] - convexHull[y]);
				e_y.direction.normalize();
				myPair res2 = intersectPoint(e_y, Segment(Point(midx, -border), Point(midx, border)));
				double h2;
				if (res2.first) h2 = res2.second.y; else h2 = ((convexHull[y].y < convexHull[y - 1].y) * 2 - 1) * border;

				if (h1 > h2) x = x + 1;
				else y = y - 1;
			}
		}
	}
}

__device__ const int& mymin(const int &a, const int &b) { return a < b ? a : b; }

__device__ const int& mymax(const int &a, const int &b) { return a > b ? a : b; }

__device__ int getFrameFlag(const Point &p)
{
	if (p.y == -minerBorder) return 1;
	if (p.y == minerBorder) return 3;
	if (p.x == minerBorder) return 2;
	if (p.x == -minerBorder) return 0;
}

__device__ void addPoint(int &top, Point *st, PointWithID *d, const Point &x, EdgeData ed1, EdgeData ed2)
{
const Point frame[] = {
			Point(-minerBorder, -minerBorder),
			Point(minerBorder, -minerBorder),
			Point(minerBorder, minerBorder),
			Point(-minerBorder, minerBorder)
};
	myPair tmp = intersectPoint(ed1.e, ed2.e);
	if (dblcmp(x.cross(d[ed1.adj], d[ed2.adj])) <= 0 || !tmp.first)
	{
		Point a, b, res1, res2;
		getMinerBorderPoint(ed1.e, a, b);
		if (x.cross(ed1.e.center, a) > 0) res1 = a;
		else res1 = b;
		getMinerBorderPoint(ed2.e, a, b);
		if (x.cross(ed2.e.center, a) > 0) res2 = b;
		else res2 = a;
		int f1 = getFrameFlag(res1);
		int f2 = getFrameFlag(res2);
		st[top++] = res1;
		for (int i = f1; i != f2; i = (i + 1) % 4)
			st[top++] = frame[i];
		st[top++] = res2;
	}
	else {
		st[top++] = tmp.second;
	}
}

__device__ void mySwap(Point &a, Point &b)
{
	Point c = a;
	a = b;
	b = c;
}

__device__ void cutLine(int &top, Point *d, Point *d2, const Edge &e, const Point &x)
{
	Point p1, p2;
	getBorderPoint(e, p1, p2);
	if (x.cross(p1, p2) < 0) mySwap(p1, p2);
	int top2 = 0;
	for (int i = 0; i < top; ++i)
	{
		int flag;
		if ((flag = dblcmp(p1.cross(p2, d[i]))) >= 0)
		{
			d2[top2++] = d[i];
		}
		
		if (flag * dblcmp(p1.cross(p2, d[i + 1])) < 0)
		{
			d2[top2++] = intersectPoint(e, Segment(d[i], d[i + 1])).second;
		}
	}
	d2[top2] = d2[0];
	memcpy(d, d2, (top2 + 1) * sizeof(Point));
	top = top2;
}

__global__ void calcAreaOne(PointWithID *d, int *g, EdgeData *edge, Frag *frag, double *Area)
{
	
const Point frame[] = {
			Point(-minerBorder, -minerBorder),
			Point(minerBorder, -minerBorder),
			Point(minerBorder, minerBorder),
			Point(-minerBorder, minerBorder)
};

	int idx = blockIdx.x, idy = threadIdx.x + blockIdx.y * maxThreads;
	++idx;
	int i = frag[idx].l + idy;
	if (i > frag[idx].r) return;
	Point x = frag[idx].b - frag[idx].a;
	Point y = frag[idx].c - frag[idx].b;
	Edge e1(frag[idx].a, x);
	e1.direction.normalize();
	Edge e2(frag[idx].b, y);
	e2.direction.normalize();
	Edge e3(frag[idx].c, x + y);
	e3.direction.normalize();
	Point centerForCutLine = frag[idx].a + (x + y * 0.5) * 0.5;

	Point st[maxEdgeCnt], d2[maxEdgeCnt];
	int top = 0;
	int cnt = 0;
	if (g[i] != 0) 
	{
		++cnt;
		if (edge[g[i]].next != g[i])
			++cnt;
	}

	if (cnt >= 2)
	{
		addPoint(top, st, d, d[i], edge[g[i]], edge[edge[g[i]].next]);
		for (int p = edge[g[i]].next; p != g[i]; p = edge[p].next)
		{
			addPoint(top, st, d, d[i], edge[p], edge[edge[p].next]);
		} 
	}
	else {
		top = 4;
		for (int j = 0; j < 4; ++j)
			st[j] = frame[j];
		st[4] = st[0];
		if (cnt == 1)
		{
			cutLine(top, st, d2, edge[g[i]].e, d[i]);
		}
	}
	st[top] = st[0];
	cutLine(top, st, d2, e1, centerForCutLine);
	cutLine(top, st, d2, e2, centerForCutLine);
	cutLine(top, st, d2, e3, centerForCutLine);
	double area = 0;
	for (int j = 1; j < top; ++j)
		area += st[0].cross(st[j], st[j + 1]);
	Area[d[i].ID] = area / 2.0;
}

extern "C" void calcArea_GPU(int n, int m, PointWithID *d, int *g, EdgeData *edge, Frag *frag, double *Area, int total, int maxcnt)
{
	checkCudaErrors(cudaMalloc((void **) &Area_CUDA, (n + 1) * sizeof(double)));
	
	checkCudaErrors(cudaMalloc((void **) &edge_CUDA, (total + 1) * sizeof(EdgeData)));
	checkCudaErrors(cudaMemcpy(edge_CUDA, edge, (total + 1) * sizeof(EdgeData), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void **) &d_CUDA, (n + 1) * sizeof(PointWithID)));
	checkCudaErrors(cudaMemcpy(d_CUDA, d, (n + 1) * sizeof(PointWithID), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &g_CUDA, (n + 1) * sizeof(int)));
	checkCudaErrors(cudaMemcpy(g_CUDA, g, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &frag_CUDA, (m + 1) * sizeof(Frag)));
	checkCudaErrors(cudaMemcpy(frag_CUDA, frag, (m + 1) * sizeof(Frag), cudaMemcpyHostToDevice));

	dim3 dimGrid(m, maxcnt / maxThreads + 1);
	calcAreaOne<<<dimGrid, maxThreads>>>(d_CUDA, g_CUDA, edge_CUDA, frag_CUDA, Area_CUDA);
	
	checkCudaErrors(cudaMemcpy(Area, Area_CUDA, (n + 1) * sizeof(double), cudaMemcpyDeviceToHost));

	//for (int i = 1; i <= m; ++i)
	//{
	//	double areaRes = fabs(frag[i].a.cross_host(frag[i].b, frag[i].c)) / 2;
	//	double totArea = 0;

	//	for (int j = frag[i].l; j <= frag[i].r; ++j)
	//	{
	//		totArea += Area[j];
	//	}
	//	if (!(fabs(areaRes - totArea) < areaRes * 1e-4))
	//	{
	//		frag[i].a.print();
	//		frag[i].b.print();
	//		frag[i].c.print();
	//		std::cout << "Points" << std::endl;
	//		for (int k = frag[i].l; k <= frag[i].r; ++k)
	//		{
	//			//d[k].print();
	//			printf("%.5lf\n", Area[k]);
	//		}
	//		fprintf(stderr, "%.6lf %.6lf\n", areaRes, totArea);
	//		fprintf(stderr, "Wrong\n");
	//	}
	//}
	
	checkCudaErrors(cudaFree(frag_CUDA));
	checkCudaErrors(cudaFree(g_CUDA));
	checkCudaErrors(cudaFree(edge_CUDA));
	checkCudaErrors(cudaFree(d_CUDA));
	checkCudaErrors(cudaFree(Area_CUDA));
}
