/*
* Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from NVIDIA Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/

///-------------------------------------------------------------------------------
///
///  ppm.cpp -- Progressive photon mapping scene
///
///-------------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <GLUTDisplay.h>
#include <sutil.h>

using namespace optix;
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <iomanip>
#include <set>
#include <map>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>

#include "ppm.h"
#include "select.h"
#include <ImageLoader.h>
#include "PpmObjLoader.h"
#include "random.h"
#include "opengltool.h"
#include "gpc.h"
#include "jfv_gpu.h"

namespace Voronoi_CPU {

static const double border = 1000000000000.0;
static const double minerBorder = border / 10;
int Muls = 100000.0;

//二维平面上的点
struct Point {
	double x, y;
	Point(double x_ = 0, double y_ = 0) : x(x_), y(y_) {}
	double cross(const Point &a, const Point &b) const
	{
		return (a.x - x) * (b.y - y) - (a.y - y) * (b.x - x);
	}
	double len2()
	{
		return x * x + y * y;
	}
	void normalize()
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
Point operator - (const Point &a, const Point &b) { return Point(a.x - b.x, a.y - b.y); }
Point operator + (const Point &a, const Point &b) { return Point(a.x + b.x, a.y + b.y); }
Point operator * (const Point &a, const double &b) { return Point(a.x * b, a.y * b); }
Point operator * (const double &b, const Point &a) { return Point(a.x * b, a.y * b); }
bool operator == (const Point &a, const Point &b) { return a.x == b.x && a.y == b.y; }
double dis2(const Point &a, const Point &b) { return (a - b).len2(); }

struct PointWithID : public Point {
	int ID;
};

const double eps = 1e-8;
int total;

int dblcmp(double x)
{
	if (fabs(x) < eps) return 0;
	return (x > 0) * 2 - 1;
}

int n = 15000;//点的数量
int m = 700;//片元的数量

PointWithID *d;//点的坐标数据
int *g;//每个点的Voronoi图的边构成的双向链表的入口
int *pos;//每个点双向链表下标的起始范围
Point *convexHull;//暂存凸包的变量
int *childID;//凸包的点的ID。
int *edgeCount;
int *convexHullSize;
double *Area;
int maxcnt;

struct Edge {
	Point center;
	Point direction;
	Edge(Point x_ = Point(0, 0), Point y_ = Point(0, 0)) : center(x_), direction(y_) {}
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
} *edge;

struct Frag {
	int l, r;
	Point a, b, c; 
} *frag;//片元的数据：点在d中的下标范围、三角网的三个顶点的坐标。初始化后不应该被修改。
bool cmp1(const Frag &a, const Frag &b) { return a.l < b.l; }

struct Segment {
	Point a, b;
	Segment(Point x_ = Point(0, 0), Point y_ = Point(0, 0)) : a(x_), b(y_) {}
};

int random(int a, int b) { return ((rand() << 15) + rand()) % (b - a + 1) + a; }

int qwerty;
Point ranPoint()
{
	Point tmp;
	tmp.x = rand() % 1000 * 10;
	tmp.y = rand() % 1000 * 10;
	//tmp.x = (double)rand() / RAND_MAX * 100;
	//tmp.y = (double)rand() / RAND_MAX * 100;
	return tmp;
}

void initData()//构造随机的数据
{
	assert(RAND_MAX == 32767);
	d = (PointWithID*) malloc((n + 1) * sizeof(PointWithID));
	if (d == NULL) { fprintf(stderr, "No Enough Memory!\n"); exit(0); }
	frag = (Frag*) malloc((m + 1) * sizeof(Frag));
	if (frag == NULL) { fprintf(stderr, "No Enough Memory!\n"); exit(0); }
	for (int i = 1; i <= m; ++i)
		frag[i].l = random(1, n);
	std::sort(frag + 1, frag + m + 1, cmp1);
	frag[1].l = 1;
	for (int i = 1; i <= m; ++i)
	{
		if (i < m && frag[i + 1].l <= frag[i].l)
			frag[i + 1].l = frag[i].l + 1;
		if (i < m)
			frag[i].r = frag[i + 1].l - 1;
		else
			frag[i].r = n;
	}
	for (int i = 1; i <= m; ++i)
	if (frag[i].l <= frag[i].r && frag[i].r <= n)
	{
		do
		{
			frag[i].a = ranPoint();
			frag[i].b = ranPoint();
			frag[i].c = ranPoint();
		} while (fabs(frag[i].a.cross(frag[i].b, frag[i].c)) < 1e-4);
		/*
		frag[i].a.print();
		frag[i].b.print();
		frag[i].c.print();
		*/
		for (int j = frag[i].l; j <= frag[i].r; ++j)
		{
			double r1 = (double)rand() / RAND_MAX;
			double r2 = (double)rand() / RAND_MAX;
			Point x = frag[i].b - frag[i].a;
			Point y = frag[i].c - frag[i].b;
			d[j].x = (frag[i].a + (x + y * r2) * r1).x;
			d[j].y = (frag[i].a + (x + y * r2) * r1).y;
			d[j].ID = j;
			//d[j].print();
		}
	}
}

bool checkIntersect(const Segment &l1, const Segment &l2)
{
	return (dblcmp(l1.a.cross(l2.a, l2.b)) * dblcmp(l1.b.cross(l2.a, l2.b)) < 0 && dblcmp(l2.a.cross(l1.a, l1.b)) * dblcmp(l2.b.cross(l1.a, l1.b)) < 0);
}

std::pair<bool, Point> intersectPoint(const Segment &l1, const Segment &l2)
{
	bool tmp = checkIntersect(l1, l2);
	if (!tmp) return std::make_pair(0, Point(0, 0));
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
	return std::make_pair(1, p);
}

void getBorderPoint(const Edge &e, Point &a, Point &b)
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

void getMinerBorderPoint(const Edge &e, Point &a, Point &b)
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

std::pair<bool, Point> intersectPoint(const Edge &e1, const Segment &l2)
{
	Point a1, b1;
	getBorderPoint(e1, a1, b1);
	return intersectPoint(Segment(a1, b1), l2);
}

std::pair<bool, Point> intersectPoint(const Edge &e1, const Edge &e2)
{
	Point a1, b1, a2, b2;
	getBorderPoint(e1, a1, b1);
	getBorderPoint(e2, a2, b2);
	return intersectPoint(Segment(a1, b1), Segment(a2, b2));
}

Edge verticalLine(const Point &a, const Point &b)
{
	Point dir(a.y - b.y, b.x - a.x);
	Point cen = (a + b) * 0.5;
	dir.normalize();
	return Edge(cen, dir);
}

void calculateUpConnectedTangent(int l1, int r1, int l2, int r2, Point *convexHull, int &x, int &y)
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
				std::pair<bool, Point> res1 = intersectPoint(e_x, Segment(Point(midx, -border), Point(midx, border)));
				double h1;
				if (res1.first) h1 = res1.second.y; else h1 = ((convexHull[x].y < convexHull[x + 1].y) * 2 - 1) * border;

				Edge e_y = Edge(convexHull[y], convexHull[y - 1] - convexHull[y]);
				e_y.direction.normalize();
				std::pair<bool, Point> res2 = intersectPoint(e_y, Segment(Point(midx, -border), Point(midx, border)));
				double h2;
				if (res2.first) h2 = res2.second.y; else h2 = ((convexHull[y].y < convexHull[y - 1].y) * 2 - 1) * border;

				if (h1 > h2) x = x + 1;
				else y = y - 1;
			}
		}
	}
}

//将pos插入到p的前面，形成双向链表。
int ins(int *pos, int p, const int &adj, const Edge &abLine, EdgeData *edge)
{
	int tpos = *pos;//atomicAdd()
	++(*pos);
	if (p == 0)
	{
		edge[tpos].e = abLine;
		edge[tpos].adj = adj;
		edge[tpos].previous = edge[tpos].next = tpos;
		return tpos;
	}
	edge[tpos].e = abLine;
	edge[tpos].adj = adj;

	edge[tpos].next = p;
	edge[tpos].previous = edge[p].previous;
	edge[edge[p].previous].next = tpos;
	edge[p].previous = tpos;

	return tpos;
}

void del(const int &p, EdgeData *edge, int &p2)
{
	if (p == p2)
	{
		p2 = edge[p].next;
		if (p2 == p) p2 = 0;
	}
	edge[edge[p].previous].next = edge[p].next;
	edge[edge[p].next].previous = edge[p].previous;
}

int getNxt(const int &p, EdgeData *edge)
{
	if (p == 0) return 0;
	if (edge[p].next == p) return 0;
	return edge[p].next;
}

int getPre(const int &p, EdgeData *edge)
{
	if (p == 0) return 0;
	if (edge[p].previous == p) return 0;
	return edge[p].previous;
}

int getFlag(const Point &p)
{
	int res = 0;
	if (p.x > 0 || (p.x == 0 && p.y < 0)) res |= 2;
	if ((p.y > 0) ^ (!res)) res |= 1;
	return res;
}

void checkLeader(const int &pos, EdgeData *edge, PointWithID *d, int &p, const Point &x)
{
	if (p == 0)
	{
		p = pos;
		return;
	}
	Point P1 = d[edge[p].adj] - x;
	Point P2 = d[edge[pos].adj] - x;
	int f1 = getFlag(P1);
	int f2 = getFlag(P2);
	if (f1 > f2) p = pos;
	else if (f1 == f2)
	{
		if (Point(0, 0).cross(P1, P2) < 0) p = pos;
	}
}

void kernel(int i, int j, int step, Frag *frag, PointWithID *d, int *pos, int *g, int *edgeCount, EdgeData *edge, Point *convexHull, int *convexHullSize, int *childID)
{
	++i;
	int left = frag[i].l + step * j, right = left + step - 1;
	right = std::min(right, frag[i].r);
	int mid = left + step / 2 - 1;
	if (mid >= right) return;
	if (right - left <= 1)
	{
		int n0 = left;
		convexHull[n0] = d[left];
		childID[n0] = left;
		if (left != right)
		{
			convexHull[++n0] = d[right];
			childID[n0] = right;
		}
		convexHullSize[left] = n0;
		if (right - left == 1)
		{
			Edge xyLine = verticalLine(d[left], d[right]);
			int tmp1 = ins(pos, 0, right, xyLine, edge);
			g[left] = tmp1;
			++edgeCount[left];
			int tmp2 = ins(pos, 0, left, xyLine, edge);
			g[right] = tmp2;
			++edgeCount[right];
			edge[tmp1].eadj = tmp2;
			edge[tmp2].eadj = tmp1;
		}
		return;
	}
	int n1 = convexHullSize[left];
	int n2 = convexHullSize[mid + 1];

	int x, y;
	calculateUpConnectedTangent(left, n1, mid + 1, n2, convexHull, x, y);
	int x_ = x, y_ = y;
	while (x_ < n1 && dblcmp(convexHull[x_].cross(convexHull[x_ + 1], convexHull[y_])) == 0) ++x_;
	while (y_ > mid + 1 && dblcmp(convexHull[x_].cross(convexHull[y_ - 1], convexHull[y_])) == 0) --y_;
	x_ = childID[x_]; y_ = childID[y_];
	for (int i = x + 1; i <= x + n2 - y + 1; ++i)
	{
		convexHull[i] = convexHull[i - x - 1 + y];
		childID[i] = childID[i - x - 1 + y];
	}
	convexHullSize[left] = x + n2 - y + 1;
	x = x_; y = y_;

	
	int x_pre, x_pre2, y_nxt, y_nxt2;
	x_pre = edge[g[x]].previous;
	y_nxt = g[y];
	if (y_nxt != 0 && d[edge[y_nxt].adj].x == d[y].x && d[edge[y_nxt].adj].y > d[y].y)
		y_nxt = edge[y_nxt].next;
	while (1)
	{
		const Point &px = d[x], &py = d[y];
		Edge xyLine = verticalLine(px, py);

		bool endLeft = (x_pre == 0), endRight = (y_nxt == 0);
		if (!endLeft)
		{
			if (dblcmp(px.cross(d[edge[x_pre].adj], py)) <= 0) endLeft = true;
			else {
				while (1)
				{
					x_pre2 = getPre(x_pre, edge);
					if (x_pre2 == 0) break;
					if (dblcmp(px.cross(d[edge[x_pre2].adj], py) <= 0)) break;
					Point O = intersectPoint(xyLine, edge[x_pre].e).second;
					double r2 = dis2(px, O);
					if (r2 < dis2(d[edge[x_pre2].adj], O) + eps) break;
					--edgeCount[x];
					--edgeCount[edge[x_pre].adj];
					del(x_pre, edge, g[x]);
					del(edge[x_pre].eadj, edge, g[edge[x_pre].adj]);
					x_pre = x_pre2;
				}
			}
		}

		if (!endRight)
		{
			if (dblcmp(py.cross(d[edge[y_nxt].adj], px)) >= 0) endRight = true;
			else {
				while (1)
				{
					y_nxt2 = getNxt(y_nxt, edge);
					if (y_nxt2 == 0) break;
					if (dblcmp(py.cross(d[edge[y_nxt2].adj], px) >= 0)) break;
					Point O = intersectPoint(xyLine, edge[y_nxt].e).second;
					double r2 = dis2(py, O);
					if (r2 < dis2(d[edge[y_nxt2].adj], O) + eps) break;
					--edgeCount[y];
					--edgeCount[edge[y_nxt].adj];
					del(y_nxt, edge, g[y]);
					del(edge[y_nxt].eadj, edge, g[edge[y_nxt].adj]);
					y_nxt = y_nxt2;
				}
			}
		}

		int tmp1, tmp2;
		if (x_pre == 0)
		{
			tmp1 = ins(pos, g[x], y, xyLine, edge);
			x_pre = g[x];
		}
		else
			tmp1 = ins(pos, edge[x_pre].next, y, xyLine, edge);
		checkLeader(tmp1, edge, d, g[x], px);
		++edgeCount[x];

		if (y_nxt == 0)
		{
			tmp2 = ins(pos, g[y], x, xyLine, edge);
			y_nxt = g[y];
		}
		else
			tmp2 = ins(pos, y_nxt, x, xyLine, edge);
		checkLeader(tmp2, edge, d, g[y], py);
		++edgeCount[y];
		edge[tmp1].eadj = tmp2;
		edge[tmp2].eadj = tmp1;

		if (endLeft && endRight) break;

		if (endLeft)
		{
			y = edge[y_nxt].adj;
			y_nxt = edge[edge[y_nxt].eadj].next;
		}
		else if (endRight)
		{
			x = edge[x_pre].adj;
			x_pre = edge[edge[x_pre].eadj].previous;
		}
		else
		{
			Point O1 = intersectPoint(xyLine, edge[x_pre].e).second;
			Point O2 = intersectPoint(xyLine, edge[y_nxt].e).second;
			if (dis2(O1, O2) < eps)
			{
				y = edge[y_nxt].adj;
				y_nxt = edge[edge[y_nxt].eadj].next;
				x = edge[x_pre].adj;
				x_pre = edge[edge[x_pre].eadj].previous;
			}
			else if ((O1.y < O2.y) || (O1.y == O2.y && (O1.x < O2.x)))
			{
				y = edge[y_nxt].adj;
				y_nxt = edge[edge[y_nxt].eadj].next;
			}
			else
			{
				x = edge[x_pre].adj;
				x_pre = edge[edge[x_pre].eadj].previous;
			}
		}
	}
}

void calcVoronoi()
{
	g = (int*) malloc((n + 1) * sizeof(int));
	if (g == NULL) { fprintf(stderr, "No Memory"); exit(0); }
	edgeCount = (int*) malloc((n + 1) * sizeof(int));
	if (edgeCount == NULL) { fprintf(stderr, "No Memory"); exit(0); }
	memset(g, 0, (n + 1) * sizeof(int));
	memset(edgeCount, 0, (n + 1) * sizeof(int));
	pos = (int*) malloc(1 * sizeof(int));
	if (pos == NULL) { fprintf(stderr, "No Memory"); exit(0); }
	*pos = 1;
	total = 0;
	for (int i = 1; i <= m; ++i)
	{
		int num = frag[i].r - frag[i].l + 1;
		static const double log2 = log(2);
		int size = 4 * num * (int)(log(num) / log2 + 1.5);
		total += size;
	}
	edge = (EdgeData*) malloc((total + 1) * sizeof(EdgeData));
	if (edge == NULL) { fprintf(stderr, "No Memory"); exit(0); }
	maxcnt = 0;
	for (int i = 1; i <= m; ++i)
	{
		std::sort(d + frag[i].l, d + frag[i].r + 1); //可并行，但意义不大。
		frag[i].r = std::unique(d + frag[i].l, d + frag[i].r + 1) - d - 1;
		maxcnt = std::max(frag[i].r - frag[i].l + 1, maxcnt);
	}
	convexHull = (Point*) malloc((n + 1) * sizeof(Point));
	convexHullSize = (int*) malloc((n + 1) * sizeof(int));
	childID = (int*) malloc((n + 1) * sizeof(int));
	if (convexHull == NULL) { fprintf(stderr, "No Memory"); exit(0); }
	if (convexHullSize == NULL) { fprintf(stderr, "No Memory"); exit(0); }
	if (childID == NULL) { fprintf(stderr, "No Memory"); exit(0); }
	for (int step = 1; step < maxcnt * 2; step <<= 1) 
	{
		for (int i = 0; i < m; ++i)
			if (frag[i + 1].r <= n)
			for (int j = 0; frag[i + 1].l + step * j <= frag[i + 1].r; ++j)
				kernel(i, j, step, frag, d, pos, g, edgeCount, edge, convexHull, convexHullSize, childID);
	}
	free(convexHull);
	free(convexHullSize);
	free(childID);
	Area = (double*) malloc((n + 1) * sizeof(double));
	if (Area == NULL) { fprintf(stderr, "No Memory"); exit(0); }
	
}

const Point frame[] = {
			Point(-minerBorder, -minerBorder),
			Point(minerBorder, -minerBorder),
			Point(minerBorder, minerBorder),
			Point(-minerBorder, minerBorder)
};

int getFrameFlag(const Point &p)
{
	if (p.y == -minerBorder) return 1;
	if (p.y == minerBorder) return 3;
	if (p.x == minerBorder) return 2;
	if (p.x == -minerBorder) return 0;
}

void addPoint(int &top, Point *st, PointWithID *d, const Point &x, EdgeData ed1, EdgeData ed2)
{
	std::pair<bool, Point> tmp = intersectPoint(ed1.e, ed2.e);
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

Point *d2;
void cutLine(int &top, Point *d, const Edge &e, const Point &x)
{
	Point p1, p2;
	getBorderPoint(e, p1, p2);
	if (x.cross(p1, p2) < 0) std::swap(p1, p2);
	int top2 = 0;
	for (int i = 0; i < top; ++i)
	{
		int flag;
		if ((flag = dblcmp(p1.cross(p2, d[i]))) >= 0)
			d2[top2++] = d[i];
		if (flag * dblcmp(p1.cross(p2, d[i + 1])) < 0)
			d2[top2++] = intersectPoint(e, Segment(d[i], d[i + 1])).second;
	}
	d2[top2] = d2[0];
	memcpy(d, d2, (top2 + 1) * sizeof(Point));
	top = top2;
}

Point *st;
void calcAreaOne(int idx, Frag *frag, int *g, EdgeData *edge)
{
	if (frag[idx].l > frag[idx].r || frag[idx].r > n) return;
	Edge e1(frag[idx].a, frag[idx].b - frag[idx].a);
	e1.direction.normalize();
	Edge e2(frag[idx].b, frag[idx].c - frag[idx].b);
	e2.direction.normalize();
	Edge e3(frag[idx].c, frag[idx].a - frag[idx].c);
	e3.direction.normalize();
	/*
	if (idx == 3637)
	{
		e1.print();
		e2.print();
		e3.print();
	}
	*/
	Point x = frag[idx].b - frag[idx].a;
	Point y = frag[idx].c - frag[idx].b;
	Point centerForCutLine = frag[idx].a + (x + y * 0.5) * 0.5;

	//double areaRes = fabs(frag[idx].a.cross(frag[idx].b, frag[idx].c));
	//double totArea = 0;

	for (int i = frag[idx].l; i <= frag[idx].r; ++i)
	{
		int top = 0;
		int cnt = 0;
		if (g[i] != 0) 
		{
			++cnt;
			if (edge[g[i]].next != g[i]) ++cnt;
		}
		if (cnt >= 2)
		{
			/*
	if (idx == 3637)
	{
		d[i].print();
		edge[g[i]].e.print();
		(edge[g[i]].e.center + edge[g[i]].e.direction).print();
	}
	*/
			addPoint(top, st, d, d[i], edge[g[i]], edge[edge[g[i]].next]);
			for (int p = edge[g[i]].next; p != g[i]; p = edge[p].next)
			{
				addPoint(top, st, d, d[i], edge[p], edge[edge[p].next]);
				/*
				if (idx == 3637)
				{
					edge[p].e.print();
					(edge[p].e.center + edge[p].e.direction).print();
				}
				*/
			}
		}
		else {
			top = 4;
			for (int j = 0; j < 4; ++j)
				st[j] = frame[j];
			st[4] = st[0];
			if (cnt == 1)
			{
				cutLine(top, st, edge[g[i]].e, d[i]);
			}
		}
		st[top] = st[0];
		/*
		if (idx == 3637)
		{
			cout << "_-------------------------------_1" << endl;
			for (int j = 0; j < top; ++j)
				st[j].print();
		}
		*/
		cutLine(top, st, e1, centerForCutLine);
		/*
		if (idx == 3637)
		{
		cout << "_-------------------------------_2" << endl;
		for (int j = 0; j < top; ++j)
			st[j].print();
		}
		*/
		cutLine(top, st, e2, centerForCutLine);
		/*
		if (idx == 3637)
		{
		cout << "_-------------------------------_3" << endl;
		for (int j = 0; j < top; ++j)
			st[j].print();
		}
		*/
		cutLine(top, st, e3, centerForCutLine);
		/*
		if (idx == 3637)
		{
		cout << "_-------------------------------_4" << endl;
		for (int j = 0; j < top; ++j)
			st[j].print();
		}
		*/
		double area = 0;
		for (int j = 1; j < top; ++j)
			area += st[0].cross(st[j], st[j + 1]);
		Area[d[i].ID] = area / 2.0;
		//totArea += area;
		/*
		if (idx == exitID)
		{
		printf("%d\n", top);
		for (int j = 0; j < top; ++j)
		{
			st[j].dprint();
			st[j + 1].dprint();
		}
		}*/
		/*
		if (idx == 3637)
		{
			cout << area << endl;
			if (area < 0) exit(0);
			cout << "_-------------------------------_" << endl;
			cout << "_" << endl;
		}
		*/
	}
	/*
	if (0 && !(fabs(areaRes - totArea) < areaRes * 1e-4))
	{
		frag[idx].a.print();
		frag[idx].b.print();
		frag[idx].c.print();
		std::cout << "Points" << std::endl;
		for (int i = frag[idx].l; i <= frag[idx].r; ++i)
		{
			d[i].print();
			if (g[i] == 0)
				std::cout << Area[i] << std::endl;
		}
		printf("%.6lf %.6lf\n", areaRes, totArea);
		puts("Wrong");
	}
	if (idx == exitID)
	exit(0);*/
}

void calcArea()
{
	st = (Point*) malloc((n + 1) * sizeof(Point));
	d2 = (Point*) malloc((n + 1) * sizeof(Point));
	/*
		freopen("d:/Voronoi_CPU.out", "w", stdout);

	int maxcnt = 0;
	int maxID;
	for (int i = 1; i <= m; ++i)
	{
		int cnt = 0;
		for (int j = frag[i].l; j <= n && j <= frag[i].r; ++j)
			++cnt;
		if (maxcnt <cnt)
		{
			maxcnt = cnt;
			maxID = i;
		}
	}
	printf("%d\n", maxcnt);
	{
		for (int j = frag[maxID].l; j <= n && j <= frag[maxID].r; ++j)
			d[j].dprint();
	}*/

	for (int i = 1; i <= m; ++i)
		calcAreaOne(i, frag, g, edge);
	
	free(st);
	free(d2);
}

void freeData()
{
	free(frag);
	free(g);
	free(edge);
	free(d);
	free(Area);
	free(edgeCount);
	free(pos);
}

}

extern "C" {
	void calcArea_GPU(int n, int m, Voronoi_CPU::PointWithID *d, int *g, Voronoi_CPU::EdgeData *edge, Voronoi_CPU::Frag *frag, double *Area, int total, int *edgeCount, int maxcnt);
};

static const float3 m_light_target = make_float3(0, -0.3, 0);

class Matrix4X4
{
public:
	float _m[16];
	Matrix4X4(){};
	Matrix4X4(float f11, float f12, float f13, float f14,
		float f21, float f22, float f23, float f24,
		float f31, float f32, float f33, float f34,
		float f41, float f42, float f43, float f44)
	{
		set(f11, f12, f13, f14,
			f21, f22, f23, f24,
			f31, f32, f33, f34,
			f41, f42, f43, f44);
	}

	void set(float f11, float f12, float f13, float f14,
		float f21, float f22, float f23, float f24,
		float f31, float f32, float f33, float f34,
		float f41, float f42, float f43, float f44)
	{
		_m[0] = f11;		_m[1] = f12;		_m[2] = f13;		_m[3] = f14;
		_m[4] = f21;		_m[5] = f22;		_m[6] = f23;		_m[7] = f24;
		_m[8] = f31;		_m[9] = f32;		_m[10] = f33;		_m[11] = f34;
		_m[12] = f41;		_m[13] = f42;		_m[14] = f43;		_m[15] = f44;
	}

	void makeLookAt(float3 vEye, float3 vCenter,float3 u);
	float3 transform3x3(float3 v);
	float2 transform3x2(float3 v);
};
void Matrix4X4::makeLookAt(float3 vEye, float3 vCenter,float3 u)
{
	float3 n = normalize(vCenter - vEye);
	float3 r = normalize(cross(n, u));
	u = normalize(cross(r, n));
	set(r.x, r.y, r.z, -dot(vEye, r),
		u.x,  u.y,  u.z, -dot(vEye, u),        
		-n.x, -n.y, -n.z, dot(vEye, n),
		0, 0, 0, 1);
}
float3 Matrix4X4::transform3x3(float3 v)
{
	return make_float3( _m[0] * v.x + _m[1] * v.y + _m[2] * v.z + _m[3],
		_m[4] * v.x + _m[5] * v.y + _m[6] * v.z + _m[7],
		_m[8] * v.x + _m[9] * v.y + _m[10] * v.z + _m[11] ) ;
}
float2 Matrix4X4::transform3x2(float3 v)
{
	return make_float2( _m[0] * v.x + _m[1] * v.y + _m[2] * v.z + _m[3],
		_m[4] * v.x + _m[5] * v.y + _m[6] * v.z + _m[7]);
}

/// Finds the smallest power of 2 greater or equal to x.
inline unsigned int pow2roundup(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x+1;
}

inline float max(float a, float b)
{
	return a > b ? a : b;
}

inline RT_HOSTDEVICE int max_component(float3 a)
{
	if(a.x > a.y) {
		if(a.x > a.z) {
			return 0;
		} else {
			return 2;
		}
	} else {
		if(a.y > a.z) {
			return 1;
		} else {
			return 2;
		}
	}
}

float3 sphericalToCartesian( float theta, float phi )
{
	float cos_theta = cosf( theta );
	float sin_theta = sinf( theta );
	float cos_phi = cosf( phi );
	float sin_phi = sinf( phi );
	float3 v;
	v.x = cos_phi * sin_theta;
	v.z = sin_phi * sin_theta;
	v.y = cos_theta;
	return v;
}

enum SplitChoice {
	RoundRobin,
	HighestVariance,
	LongestDim
};

gpc_vertex make_gpc_vertex(float2 t_p)
{
	gpc_vertex temp_vertex;
	temp_vertex.x = t_p.x;
	temp_vertex.y = t_p.y;
	return temp_vertex;
}
void getArea(float2 r_center, float r_radius, float2 t1, float2 t2, float2 t3, float* ret_area)
{
	gpc_polygon sim_circle, sim_triangle;
	gpc_tristrip clip_polygon;

	/// Triangle
	sim_triangle.num_contours = 1;
	sim_triangle.hole = new int(0);
	sim_triangle.contour = new gpc_vertex_list();
	sim_triangle.contour->num_vertices = 3;
	sim_triangle.contour->vertex = new gpc_vertex[sim_triangle.contour->num_vertices];
	sim_triangle.contour->vertex[0] = make_gpc_vertex(t1);
	sim_triangle.contour->vertex[1] = make_gpc_vertex(t2);
	sim_triangle.contour->vertex[2] = make_gpc_vertex(t3);

	/// Circle
	sim_circle.num_contours = 1;
	sim_circle.hole = new int(0);
	sim_circle.contour = new gpc_vertex_list();
	sim_circle.contour->num_vertices = 
		///3;
		///4;
		///8;
		16;
	sim_circle.contour->vertex = new gpc_vertex[sim_circle.contour->num_vertices];
	float m_alpha = 0, m_step = 2 * M_PI / sim_circle.contour->num_vertices;
	for (int i = 0;i < sim_circle.contour->num_vertices;i ++)
	{
		float2 temp_points = make_float2(sin(m_alpha), cos(m_alpha)) * r_radius + r_center;
		sim_circle.contour->vertex[i] = make_gpc_vertex(temp_points);
		m_alpha += m_step;
	}

	/// Clip
	gpc_tristrip_clip(GPC_INT, &sim_circle, &sim_triangle, &clip_polygon);
	get_area(&clip_polygon, ret_area);

	gpc_free_polygon(&sim_circle);
	gpc_free_polygon(&sim_triangle);
	gpc_free_tristrip(&clip_polygon);
}



int OptiXDeviceToCUDADevice( const Context& context, unsigned int optixDeviceIndex )
{
  std::vector<int> devices = context->getEnabledDevices();
  unsigned int numOptixDevices = static_cast<unsigned int>( devices.size() );
  int ordinal;
  if ( optixDeviceIndex < numOptixDevices )
  {
    context->getDeviceAttribute( devices[optixDeviceIndex], RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(ordinal), &ordinal );
    return ordinal;
  }
  return -1;
}
///-----------------------------------------------------------------------------
///
/// Whitted Scene
///
///-----------------------------------------------------------------------------

static char* TestSceneNames[] = {
	"Cornel_Box_Scene",
	"Wedding_Ring_Scene",
	"Mesh_Room_Scene",
	"Conference_Scene",
	"Clock_Scene",
	"Sponza_Scene",
	"Box_Scene",
	"Sibenik_Scene",
	"Torus_Scene"
};
class ProgressivePhotonScene : public SampleScene
{
public:
	ProgressivePhotonScene() : SampleScene()
		, m_frame_number( 0 )
		, m_display_debug_buffer( false )
		, m_print_timings ( false )
		, m_test_scene( Cornel_Box_Scene )
		, m_light_phi( 2.19f )
		, m_light_theta( 1.15f )
		, m_split_choice(LongestDim)
	{}

	/// From SampleScene
	void	initAssistBuffer();
	void	initScene( InitialCameraData& camera_data );
	void	selectScene();
	bool	keyPressed(unsigned char key, int x, int y);
	void	trace( const RayGenCameraData& camera_data );
	void	regenerate_area(RTsize buffer_width, RTsize buffer_height, char* info);
	void    calculateKernelAreaWithVertex();
	void    calculateKernelAreaWithTriangle();
	void	updateKernelArea();
	void	getKernelArea(float3 sp_position, float3 sp_normal, float kernelRadius, 
		int p0, int p1, int p2, std::set<int>& targetTriangleSet,
		float& kernelArea, int cur_depth);
	void	doResize( unsigned int width, unsigned int height );
	Buffer	getOutputBuffer();

	void setTestScene(int testScene) { 
		m_test_scene = testScene; 
		if (m_test_scene == Cornel_Box_Scene)
			m_gather_method = Cornel_Box_Method;
	}
	void setGatherMethod(int gatherMethod) { m_gather_method = gatherMethod; }
	void printTimings()       { m_print_timings = true; }
	void displayDebugBuffer() { m_display_debug_buffer = true; }

	enum GatherMethod{
		Cornel_Box_Method,
		Triangle_Inside_Method,
		Triangle_Vertical_Method,
		Triangle_Extend_Method,
		Triangle_Combine_Method
	};
	enum TestScene{
		Cornel_Box_Scene = 0,
		Wedding_Ring_Scene,
		Small_Room_Scene,
		Conference_Scene,
		Clock_Scene,
		Sponza_Scene,
		Box_Scene,
		Sibenik_Scene,
		Torus_Scene
	};
private:
	void initWeddingRing(InitialCameraData& camera_data);
	void initConference(InitialCameraData& camera_data);
	void initCornelBox(InitialCameraData& camera_data);
	void initSponza(InitialCameraData& camera_data);
	void initSmallRoom(InitialCameraData& camera_data);
	void initClock(InitialCameraData& camera_data);
	void initBox(InitialCameraData& camera_data);
	void initSibenik(InitialCameraData& camera_data);
	void initTorus(InitialCameraData& camera_data);
	void buildGlobalPhotonMap();
	void buildCausticsPhotonMap();
	void setFloatIn(std::vector<PhotonRecord*>& ptrVector, int ttindex, float mt_area);
	void constructVoronoiSubdivision(PhotonRecord** photon_ptrs, int photons_count);
	void loadObjGeometry( const std::string& filename, optix::Aabb& bbox, bool isUnitize);
	void createCornellBoxGeometry();
	GeometryInstance createParallelogram( const float3& anchor,
		const float3& offset1,
		const float3& offset2,
		const float3& color );

	enum OptiXEnterPoint
	{
		EnterPointRayTrace = 0,
		EnterPointCausticsPass,
		EnterPointCausticsGather,
		EnterPointGlobalPass,
		EnterPointGlobalGather,
		EnterPointNum
	};

	enum OptiXRayType
	{
		RayTypeShadowRay = 0,
		RayTypeRayTrace,
		RayTypeCausticsPass,
		RayTypeGlobalPass,
		RayTypeNum
	};

	
	void printFile(char *msg);	
	void printFile(float3 t1, float3 t2);

	int m_gather_method;

	int*		  m_area_index_record;
	float3*		  m_area_normal_record;
	float*		  m_area_record;
	unsigned int  m_frame_number;
	bool          m_display_debug_buffer;
	bool          m_print_timings;
	int           m_test_scene;
	Program       m_pgram_bounding_box;
	Program       m_pgram_intersection;
	Material      m_material;
	Buffer        m_display_buffer;
	Buffer        m_direct_buffer;
	Buffer        m_photons;
	Buffer        m_photon_map;
	Buffer        m_debug_buffer;
	Buffer		  m_sp_triangle_info_buffer;
	Buffer		  m_sp_position_buffer;
	Buffer		  m_sp_normal_buffer;
	Buffer		  m_sp_radius_buffer;
	Buffer		  m_sp_area_buffer;
	Buffer		  m_sp_valid_buffer;
	float         m_light_phi;
	float         m_light_theta;
	unsigned int  m_photon_map_size;
	unsigned int  m_current_valid_photons;
	SplitChoice   m_split_choice;
	PPMLight      m_light;
	int			  m_circle_count;
	bool		  m_finish_gen;
	bool		  m_print_image;
	bool		  m_print_camera;
	int           m_cuda_device;

	const static unsigned int WIDTH;
	const static unsigned int HEIGHT;
	const static unsigned int MAX_PHOTON_COUNT;
	const static unsigned int PHOTON_LAUNCH_WIDTH;
	const static unsigned int PHOTON_LAUNCH_HEIGHT;
	const static unsigned int NUM_PHOTONS;

	PpmObjLoader *loader;

	void initGlobal();
	void initEnterPointGlobalPhotonTrace();
	void initEnterPointCausticsPhotonTrace();
	void initEnterPointRayTrace(InitialCameraData& camera_data);
	void initEnterPointGlobalGather();
	void initEnterPointCausticsGather();
	void initGeometryInstances(InitialCameraData& camera_data);
	
	int Global_Photon_Buffer_Size;
	int Global_Photon_Map_Size;
	int Caustics_Photon_Buffer_Size;
	int Caustics_Photon_Map_Size;
	Buffer m_Global_Photon_Buffer;
	Buffer m_Global_Photon_Map;
	Buffer m_Caustics_Photon_Buffer;
	Buffer m_Caustics_Photon_Count;
	Buffer m_Caustics_Photon_Map;
	string cuda_gather_cu;
	string cuda_ppass_cu;
	string cuda_rtpass_cu;
	string triangle_mesh_cu;
	string projectName;

};
const unsigned int ProgressivePhotonScene::WIDTH  = 800u;
const unsigned int ProgressivePhotonScene::HEIGHT = 600u;
/// const unsigned int ProgressivePhotonScene::WIDTH  = 400u;
/// const unsigned int ProgressivePhotonScene::HEIGHT = 300u;
/// const unsigned int ProgressivePhotonScene::WIDTH  = 1680u;
/// const unsigned int ProgressivePhotonScene::HEIGHT = 974u;
/// const unsigned int ProgressivePhotonScene::WIDTH  = 768u;
/// const unsigned int ProgressivePhotonScene::HEIGHT = 768u;
/// const unsigned int ProgressivePhotonScene::WIDTH  = 256u;
/// const unsigned int ProgressivePhotonScene::HEIGHT = 256u;

const unsigned int ProgressivePhotonScene::MAX_PHOTON_COUNT = 2u;
const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH = 256u;
const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT = 256u;
//const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH = 1024u;
//const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT = 1024u;
//const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH = 512u;
//const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT = 512u;
const unsigned int ProgressivePhotonScene::NUM_PHOTONS = (ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH *
	ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT *
	ProgressivePhotonScene::MAX_PHOTON_COUNT);

void ProgressivePhotonScene::printFile(char *msg)
{	
	char name[256];
	sprintf(name, "%s/%s/%s/grab.txt", sutilSamplesDir(), "progressivePhotonMap", "screengrab");
	printLog(name, msg);
}
void ProgressivePhotonScene::printFile(float3 t1, float3 t2)
{	
	char msg[256];
	sprintf(msg, "%lf %lf %lf %lf %lf %lf\n", t1.x, t1.y, t1.z, t2.x, t2.y, t2.z);
	printFile(msg);
}

void mprintf(float3 &tempfloat3)
{
	std::cerr<< tempfloat3.x << ", " << tempfloat3.y << ", " << tempfloat3.z;
}

// 将light的位置移动微小距离，然后更新光源
void updateLight(PPMLight &light, float3 dis_float3)
{
	light.position += dis_float3;

	light.direction = normalize( m_light_target  - light.position );
	light.anchor = light.position + light.direction * 0.01f;

	float3 m_light_t_normal;
	m_light_t_normal = cross(light.v1, light.direction);
	light.v1 = cross(light.direction, m_light_t_normal);
	light.v2 = cross(light.direction, light.v1);

	std::cerr << "new light anchor: ";
	mprintf(light.anchor);
	std::cerr << std::endl;

	std::cerr << "new light position: ";
	mprintf(light.position);
	std::cerr << std::endl;
}

bool ProgressivePhotonScene::keyPressed(unsigned char key, int x, int y)
{
	float step_size = 0.01f;
	float light_step_size = 0.1f;
	bool light_changed = false;
	switch (key)
	{
	case 'd':
		m_light_phi += step_size;
		if( m_light_phi >  M_PIf * 2.0f ) m_light_phi -= M_PIf * 2.0f;
		light_changed = true;
		break;
	case 'a':
		m_light_phi -= step_size;
		if( m_light_phi <  0.0f ) m_light_phi += M_PIf * 2.0f;
		light_changed = true;
		break;
	case 's':
		std::cerr << "new theta: " << m_light_theta + step_size << " max: " << M_PIf / 2.0f  << std::endl;
		m_light_theta = fminf( m_light_theta + step_size, M_PIf / 2.0f );
		light_changed = true;
		break;
	case 'w':
		std::cerr << "new theta: " << m_light_theta - step_size << " min: 0.0f " << std::endl;
		m_light_theta = fmaxf( m_light_theta - step_size, 0.0f );
		light_changed = true;
		break;
	case '0':
		// 标记照相机，重绘图像
		m_camera_changed = true;
		break;
	case 'p':
		// 输出当前图像到screengrab目录
		m_print_image = true;
		std::cerr << "we print an image" << std::endl;
		break;
	case ']':
		// 输出当前照相机数据
		m_print_camera = true;
		break;
	case '.':
		RTsize buffer_width, buffer_height;
		m_context["rtpass_output_buffer"]->getBuffer()->getSize( buffer_width, buffer_height );
		regenerate_area(buffer_width, buffer_height, "press '.'");
		break;
	// u,i,h,j,k,l改变光源位置
	case 'u':
		updateLight(m_light, make_float3(light_step_size, 0, 0));
		m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
		m_camera_changed = true;
		break;
	case 'i':
		updateLight(m_light, make_float3(-light_step_size, 0, 0));
		m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
		m_camera_changed = true;
		break;
	case 'h':
		updateLight(m_light, make_float3(0, light_step_size, 0));
		m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
		m_camera_changed = true;
		break;
	case 'j':
		updateLight(m_light, make_float3(0, -light_step_size, 0));
		m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
		m_camera_changed = true;
		break;
	case 'k':
		updateLight(m_light, make_float3(0, 0, light_step_size));
		m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
		m_camera_changed = true;
		break;
	case 'l':
		updateLight(m_light, make_float3(0, 0, -light_step_size));
		m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
		m_camera_changed = true;
		break;
	}

	if( light_changed && !m_test_scene ) {
		/// Myron Modify Here! We don't need change light
		/// 		std::cerr << " theta: " << m_light_theta << "  phi: " << m_light_phi << std::endl;
		/// 		m_light.position  = 1000.0f * sphericalToCartesian( m_light_theta, m_light_phi );
		/// 		m_light.direction = normalize( make_float3( 0.0f, 0.0f, 0.0f )  - m_light.position );
		/// 		m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
		/// 		signalCameraChanged(); 
		/// 		return true;
	}

	return false;
}

void ProgressivePhotonScene::selectScene()
{
// 	setTestScene(ProgressivePhotonScene::Cornel_Box_Scene);
//	setTestScene(ProgressivePhotonScene::Sibenik_Scene);
// 	setTestScene(ProgressivePhotonScene::Wedding_Ring_Scene);
//	setTestScene(ProgressivePhotonScene::Conference_Scene);
//	setTestScene(ProgressivePhotonScene::Sponza_Scene);
//	setTestScene(ProgressivePhotonScene::Small_Room_Scene);
// 	setTestScene(ProgressivePhotonScene::Clock_Scene);
	setTestScene(ProgressivePhotonScene::Box_Scene);
//	setTestScene(ProgressivePhotonScene::Torus_Scene);
}
void ProgressivePhotonScene::initAssistBuffer()
{
	m_context["can_count_kernel"]->setFloat( 1.0f );
	/// Target output buffer
	Buffer cornel_primary_edge_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	cornel_primary_edge_buffer->setFormat( RT_FORMAT_FLOAT );
	cornel_primary_edge_buffer->setSize( WIDTH, HEIGHT );
	m_context["primary_edge_buffer"]->set( cornel_primary_edge_buffer );

	Buffer cornel_secondary_edge_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	cornel_secondary_edge_buffer->setFormat( RT_FORMAT_FLOAT );
	cornel_secondary_edge_buffer->setSize( WIDTH, HEIGHT );
	m_context["secondary_edge_buffer"]->set( cornel_secondary_edge_buffer );

	m_sp_triangle_info_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	m_sp_triangle_info_buffer->setFormat( RT_FORMAT_INT4 );
	m_sp_triangle_info_buffer->setSize( WIDTH, HEIGHT );
	m_context["sp_triangle_info_buffer"]->set( m_sp_triangle_info_buffer );

	m_sp_position_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	m_sp_position_buffer->setFormat( RT_FORMAT_FLOAT3 );
	m_sp_position_buffer->setSize( WIDTH, HEIGHT );
	m_context["sp_position_buffer"]->set( m_sp_position_buffer );

	m_sp_normal_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	m_sp_normal_buffer->setFormat( RT_FORMAT_FLOAT3 );
	m_sp_normal_buffer->setSize( WIDTH, HEIGHT );
	m_context["sp_normal_buffer"]->set( m_sp_normal_buffer );	

	m_sp_radius_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	m_sp_radius_buffer->setFormat( RT_FORMAT_FLOAT );
	m_sp_radius_buffer->setSize( WIDTH, HEIGHT );
	m_context["sp_radius_buffer"]->set( m_sp_radius_buffer );	

	m_sp_valid_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	m_sp_valid_buffer->setFormat( RT_FORMAT_INT );
	m_sp_valid_buffer->setSize( WIDTH * HEIGHT * Myron_Valid_Size);
	m_context["sp_valid_buffer"]->set( m_sp_valid_buffer );	

	m_sp_area_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	m_sp_area_buffer->setFormat( RT_FORMAT_FLOAT );
	m_sp_area_buffer->setSize( WIDTH, HEIGHT );
	m_context["sp_area_buffer"]->set( m_sp_area_buffer );	

	m_area_index_record = NULL;
	m_area_normal_record = NULL;
	m_area_record = NULL;
	regenerate_area(WIDTH, HEIGHT, "init scene");
}
void ProgressivePhotonScene::initConference(InitialCameraData& camera_data)
{
	
/// 	camera_data = InitialCameraData( make_float3( -43.465564, 470.632643, -842.851190 ), /// eye
/// 	make_float3( 332.84030, 323.75006, -193.58344 ),      /// lookat
/// 	make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 	35.0f );                              /// vfov


/// 	camera_data = InitialCameraData( make_float3( -0.278133, 0.108563, -0.479883 ), /// eye
/// 		make_float3( 0.0f, 0.0f, 0.0f ),      /// lookat
/// 		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 		35.0f );                              /// vfov
	camera_data = InitialCameraData( make_float3( -0.519556, 0.119797, -0.591875 ), /// eye
		make_float3( 0, 0, 0 ),      /// lookat
		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
		60.0f );                              /// vfov
	///35.0f );                              /// vfov
	
/// 	camera_data = InitialCameraData( make_float3( 0.0f, 0.04f, -0.63f ), /// eye
/// 		make_float3( 0.0f, 0.0f, 0.0f ),      /// lookat
/// 		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 		35.0f );                              /// vfov
	
	/*
	camera_data = InitialCameraData( make_float3( 0.0f, -0.2f, -0.63f ), /// eye
		make_float3( 0.0f, -0.2f, 0.63f ),      /// lookat
		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
		35.0f );                              /// vfov
		*/
	/*
	m_light.is_area_light = 0; 
	m_light.position = make_float3( 0.0f, 0.24f, 0.0f );
	m_light.direction = normalize( make_float3( 0.0f, 0.0f, 0.0f )  - m_light.position );
	m_light.radius    = 0.1f * 0.01745329252f;
	m_light.power     = make_float3( 1.0f, 1.0f, 1.0f );
	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
	*/
	
	m_light.is_area_light = 1; 
	float3 relate_position = make_float3(0.f, 0.f, 0.f);
	m_light.position = make_float3( 0.0f, 0.241f, 0.0f ) + relate_position;
	m_light.anchor = make_float3( 0.0f, 0.2405f, 0.0f ) + relate_position;
	float point_size = 0.08f;
	m_light.v1 = make_float3( 1.0f, 0.f, 0.0f ) * point_size * 2.8;
	m_light.v2 = make_float3( 0.0f, 0.f, 1.0f ) * point_size;


/// 	m_light.position = m_light.position/0.00073911418 + make_float3(332.84030 ,323.75006 ,-193.58344);
/// 	m_light.anchor = m_light.anchor/0.00073911418 + make_float3(332.84030 ,323.75006 ,-193.58344);
/// 	m_light.v1 /= 0.00073911418;
/// 	m_light.v2 /= 0.00073911418;

	m_light.direction = normalize( m_light.anchor  - m_light.position );
	m_light.radius    = 0.1f * 0.01745329252f;
	m_light.power     = make_float3( 1.0f, 1.0f, 1.0f );
	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );


	float default_radius2 = 0.0016f;
	m_context["rtpass_default_radius2"]->setFloat( default_radius2);
	m_context["max_radius2"]->setFloat(default_radius2);

	optix::Aabb aabb;	
	loadObjGeometry( "conference/conference.obj", aabb, true);
}
void ProgressivePhotonScene::initWeddingRing(InitialCameraData& camera_data)
{
	camera_data = InitialCameraData(
		///make_float3( -235.0f, 0.0f, 0.0f ), /// eye
		make_float3( -235.0f, 220.0f, 0.0f ), /// eye
		///make_float3( -235.0f, 22.0f, -235.0f ), /// eye
		make_float3( 0.0f, 0.0f, 0.0f ),      /// lookat
		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
		35.0f );                              /// vfov
	m_light.is_area_light = 0; 
	m_light.position  = 1000.0f * sphericalToCartesian( m_light_theta, m_light_phi );
	///light.position = make_float3( 600.0f, 500.0f, 700.0f );
	m_light.direction = normalize( make_float3( 0.0f, 0.0f, 0.0f )  - m_light.position );
	m_light.radius    = 5.0f *0.01745329252f;
	m_light.power     = make_float3( 0.5e4f, 0.5e4f, 0.5e4f );
	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
	///float default_radius2 = 1.0f;
	///float default_radius2 = 16.0f;
	///float default_radius2 = 4.0f;
	float default_radius2 = 0.81f;
	///float default_radius2 = 0.64f;
	m_context["rtpass_default_radius2"]->setFloat( default_radius2);
	m_context["max_radius2"]->setFloat(default_radius2);
	optix::Aabb aabb;
	loadObjGeometry( "wedding-band.obj", aabb, false);		
}
void ProgressivePhotonScene::initCornelBox(InitialCameraData& camera_data)
{
	createCornellBoxGeometry();
	/// Set up camera
	camera_data = InitialCameraData( make_float3( 278.0f, 273.0f, -850.0f ), /// eye
		make_float3( 278.0f, 273.0f, 0.0f ),    /// lookat
		make_float3( 0.0f, 1.0f,  0.0f ),       /// up
		35.0f );                                /// vfov

	m_light.is_area_light = 1; 
	m_light.anchor = make_float3( 343.0f, 548.6f, 227.0f);
	m_light.v1     = make_float3( 0.0f, 0.0f, 105.0f);
	m_light.v2     = make_float3( -130.0f, 0.0f, 0.0f);
	m_light.direction = normalize(cross( m_light.v1, m_light.v2 ) ); 
	m_light.power  = make_float3( 0.5e6f, 0.4e6f, 0.2e6f );
	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
	float default_radius2 = 900.0f;
	m_context["rtpass_default_radius2"]->setFloat( default_radius2);
	///m_context["rtpass_default_radius2"]->setFloat( 100.0f);
	///m_context["rtpass_default_radius2"]->setFloat( 9.0f);
	///m_context["min_fovy"]->setFloat( tanf(40.0f/180.0f*M_PI/HEIGHT*2) );
	m_context["min_fovy"]->setFloat( tanf(10.0f/180.0f*M_PI/HEIGHT*2) );
	m_context["max_radius2"]->setFloat(default_radius2);
}
void ProgressivePhotonScene::initSponza(InitialCameraData& camera_data)
{
/// 	camera_data = InitialCameraData( make_float3( -0.6f, 0.0f, 0.0f ), /// eye
/// 		make_float3( 0.0f, 0.0f, 0.0f ),      /// lookat
/// 		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 		35.0f );                              /// vfov

/// 	camera_data = InitialCameraData( make_float3( 0.f, 0.25f, 0.0f ), /// eye
/// 		make_float3( -0.6f, 0.25f, 0.0f ),      /// lookat
/// 		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 		35.0f );                              /// vfov

/// 	camera_data = InitialCameraData( make_float3( 0.4f, -0.05f, 0.0f ), /// eye
/// 		make_float3( 0.0f, 0.0f, 0.0f ),      /// lookat
/// 		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 		35.0f );                              /// vfov


	///camera_data = InitialCameraData( make_float3( -0.4f, -0.05f, 0.0f ), /// eye
	///camera_data = InitialCameraData( make_float3( -0.340745f, 0.001279f, -0.306309f ), /// eye
	camera_data = InitialCameraData( make_float3( -0.462603f, 0.204225f, -0.0165347f ), /// eye
		make_float3( 0.f, 0.0f, 0.0f ),      /// lookat
		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
		35.0f );                              /// vfov

/// 	camera_data = InitialCameraData( make_float3( -0.5449f, 0.0006f, -0.0132f ), /// eye
/// 		make_float3( -0.5f, 0.0f, 0.0f ),      /// lookat
/// 		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 		35.0f );                              /// vfov
	
	/*
	m_light.is_area_light = 0; 
	///m_light.position  = 1000.0f * sphericalToCartesian( m_light_theta, m_light_phi );
	m_light.position = make_float3( 0.0f, 0.47f, 0.0f );	
	*/

	m_light.is_area_light = 1; 
	m_light.position = make_float3( 0.0f, 0.47f, 0.0f );
	m_light.anchor = make_float3( 0.0f, 0.45f, 0.0f );
	///float point_size = 0.03f;
	float point_s_size = 2.1f;
	float point_size1 = 0.28f * point_s_size;
	float point_size2 = 0.06f * point_s_size;
	m_light.v1 = make_float3( 1.0f, 0.f, 0.0f ) * point_size1;
	m_light.v2 = make_float3( 0.0f, 0.f, 1.0f ) * point_size2;
	
	m_light.direction = normalize( make_float3( 0.0f, 0.0f, 0.0f )  - m_light.position );
	m_light.radius    = 2.0f * 0.01745329252f;
	///m_light.power     = make_float3( 0.5e4f, 0.5e4f, 0.5e4f );
	m_light.power     = make_float3( 1.0f, 1.0f, 1.0f );
	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );

	float default_radius2 = 0.0016f;
	///float default_radius2 = 0.0001f;
	m_context["rtpass_default_radius2"]->setFloat( default_radius2);
	m_context["max_radius2"]->setFloat(default_radius2);

	optix::Aabb aabb;	
	///loadObjGeometry( "sponza/c-sponza/c-sponza.obj", aabb, true);
	///loadObjGeometry( "sponza/m-sponza/sponza.obj", aabb, true);
	///loadObjGeometry( "sponza/gong-sponza/gong.obj", aabb, true);
	///loadObjGeometry( "sponza/gong-sponza/gong2.obj", aabb, true);

	///loadObjGeometry( "sponza/gong-sponza/gong3.obj", aabb, true);
	///loadObjGeometry( "sponza/gong-sponza/gong4.obj", aabb, true);
	///loadObjGeometry( "sponza/gong-sponza/gong6.obj", aabb, true);
	loadObjGeometry( "sponza/88888888/0000000.obj", aabb, true);
	///loadObjGeometry( "sponza/sponza.obj", aabb, true);	
}
void ProgressivePhotonScene::initSmallRoom(InitialCameraData& camera_data)
{
	camera_data = InitialCameraData( make_float3( 0.380478f, 0.267474f, 0.0291848f ), /// eye
		make_float3( 0.0f, 0.0f, 0.0f ),      /// lookat
		make_float3( 0.0f, 0.0f,  1.0f ),     /// up
		35.0f );                              /// vfov
		
	m_light.is_area_light = 1; 
	m_light.position = make_float3( 0.0f, 0.f, 0.2401f);
	m_light.anchor = make_float3( 0.0f, 0.f, 0.24f);
	float point_size = 0.11f;
	m_light.v1 = make_float3( 1.0f, 0.f, 0.0f ) * point_size;
	m_light.v2 = make_float3( 0.0f, 1.f, 0.0f ) * point_size;
	m_light.direction = normalize( m_light.anchor  - m_light.position );
	m_light.radius    = 0.1f * 0.01745329252f;
	m_light.power     = make_float3( 1.0f, 1.0f, 1.0f );
	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );

	///float default_radius2 = 0.0004f;
	float default_radius2 = 0.0001f;
	m_context["rtpass_default_radius2"]->setFloat( default_radius2);
	m_context["max_radius2"]->setFloat(default_radius2);
	optix::Aabb aabb;	
	loadObjGeometry( "EChess/EChess.obj", aabb, true);

}
void ProgressivePhotonScene::initTorus(InitialCameraData& camera_data)
{
/// 	/// Current Mr Li
/// 	camera_data = InitialCameraData( make_float3( 0.6611862, 0.0602775, 1.07347 ), /// eye
/// 		make_float3(0, -0.35, 0 ),      /// lookat
/// 		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 		30.0f );                              /// vfov
	/// Current Myron
	///camera_data = InitialCameraData( make_float3( 1.00206, 0.142032, 1.5627 ), /// eye
	camera_data = InitialCameraData( make_float3( 0.93709, 0.068026, 1.17512 ), /// eye
		make_float3(0, -0.35, 0 ),      /// lookat
		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
		30.0f );                              /// vfov



	m_light.is_area_light = 0; 
	m_light.position = make_float3( 0.f, 0.f, 0.0f );	

	m_light.is_area_light = 1; 
	float point_size = 0.01f;
	m_light.v1 = make_float3( 1.0f, 0.f, 0.0f ) * point_size;
	m_light.v2 = make_float3( 0.0f, 0.f, 1.0f ) * point_size;

	///m_light.position = make_float3( 1.5, 1.05, -0.2 )/2;			/// Current Mr Li
	m_light.position = make_float3( 0.25, 0.125, -0.2 );			/// Current Myron
	m_light.position = make_float3( 0.25, 0.125, -0.2 );			/// Current Myron2

	m_light.direction = normalize( m_light_target  - m_light.position );
	m_light.anchor = m_light.position + m_light.direction * 0.01f;

	float3 m_light_t_normal;
	m_light_t_normal = cross(m_light.v1, m_light.direction);
	m_light.v1 = cross(m_light.direction, m_light_t_normal);
	m_light.v2 = cross(m_light.direction, m_light.v1);

	m_light.radius    = 2.0f * 0.01745329252f;
	m_light.power     = make_float3( 1.0f, 1.0f, 1.0f );
	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );

	float default_radius2 = 0.004f;
	//float default_radius2 = 0.0004f;
	//float default_radius2 = 0.00004f;
	m_context["rtpass_default_radius2"]->setFloat( default_radius2);
	m_context["max_radius2"]->setFloat(default_radius2);
	optix::Aabb aabb;	
	loadObjGeometry( "torus/torus_maya.obj", aabb, true);
	//loadObjGeometry( "torus/torus.obj", aabb, true);
	///loadObjGeometry( "torus/torus_maya_o.obj", aabb, true);
}
void ProgressivePhotonScene::initSibenik(InitialCameraData& camera_data)
{
/// 		camera_data = InitialCameraData( make_float3( -0.6f, 0.0f, 0.0f ), /// eye
/// 			make_float3( 0.0f, 0.0f, 0.0f ),      /// lookat
/// 			make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 			60.0f );                              /// vfov
	///camera_data = InitialCameraData( make_float3( 0.6f, 0.f, 0.0f ), /// eye
	/*
	camera_data = InitialCameraData( make_float3( -0.4f, -0.35f, 0.3f ), /// eye
	///camera_data = InitialCameraData( make_float3( -0.07f, -0.65f, 0.0f ), /// eye
		make_float3( 0.0f, -0.63f, 0.0f ),      /// lookat
		*/
	
/// 	camera_data = InitialCameraData( make_float3( -0.2f, -0.49f, 0.0f ), /// eye
/// 	///camera_data = InitialCameraData( make_float3( -0.07f, -0.65f, 0.0f ), /// eye
/// 		make_float3( 0.2f, -0.634f, 0.0f ),      /// lookat
/// 		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 		35.0f );                              /// vfov

/// 	camera_data = InitialCameraData( make_float3( 0.9*0.878701, -0.592016, 0.025917 ), /// eye
///	camera_data = InitialCameraData( make_float3( 0.564789, -0.435938, -0.26788 ), /// eye
	camera_data = InitialCameraData( make_float3( -0.676787, -0.599886, 0.130963 ), /// eye
		///camera_data = InitialCameraData( make_float3( -0.07f, -0.65f, 0.0f ), /// eye
		make_float3(-0.184200, -0.576028, 0.074177 ),      /// lookat
		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
		54.0f );                              /// vfov
/// 	camera_data = InitialCameraData( make_float3( 0.736435, -0.214608, 0.121045 ), /// eye
/// 		///camera_data = InitialCameraData( make_float3( -0.07f, -0.65f, 0.0f ), /// eye
/// 		make_float3(-0.184200, -0.576028, 0.074177 ),      /// lookat
/// 		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
/// 		54.0f );                              /// vfov
	
	m_light.is_area_light = 0; 
	///m_light.position  = 1000.0f * sphericalToCartesian( m_light_theta, m_light_phi );
	m_light.position = make_float3( 0.f, 0.f, 0.0f );	

	m_light.is_area_light = 1; 
	float point_size = 0.05f;
	///float point_size = 0.5f;
	m_light.v1 = make_float3( 1.0f, 0.f, 0.0f ) * point_size * 2;
	m_light.v2 = make_float3( 0.0f, 0.f, 1.0f ) * point_size;
	m_light.position = make_float3( -0.184200, -0.192016, 0.025917 );
	m_light.anchor = make_float3( -0.184200, -0.2, 0.025917 );

/// 	m_light.position = make_float3( 0.0, 0.929057, -1.01377 );
/// 	m_light.anchor = m_light.position * 0.90f + make_float3(0.0, -0.5, 0);
	m_light.direction = normalize( m_light.anchor  - m_light.position );

	float3 m_light_t_normal;
	m_light_t_normal = cross(m_light.v1, m_light.direction);
	m_light.v1 = cross(m_light.direction, m_light_t_normal);
	m_light_t_normal = cross(m_light.v2, m_light.direction);
	m_light.v2 = cross(m_light.direction, m_light_t_normal);

	m_light.radius    = 2.0f * 0.01745329252f;
	///m_light.power     = make_float3( 0.5e4f, 0.5e4f, 0.5e4f );
	m_light.power     = make_float3( 1.0f, 1.0f, 1.0f );
	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );

	///float default_radius2 = 0.0016f;
	float default_radius2 = 0.0009f;
	m_context["rtpass_default_radius2"]->setFloat( default_radius2);
	m_context["max_radius2"]->setFloat(default_radius2);
	optix::Aabb aabb;	
	loadObjGeometry( "sibenik/sibenik.obj", aabb, true);
}
void ProgressivePhotonScene::initBox(InitialCameraData& camera_data)
{
	/// 		camera_data = InitialCameraData( make_float3( -0.6f, 0.0f, 0.0f ), /// eye
	/// 			make_float3( 0.0f, 0.0f, 0.0f ),      /// lookat
	/// 			make_float3( 0.0f, 1.0f,  0.0f ),     /// up
	/// 			35.0f );                              /// vfov
	///camera_data = InitialCameraData( make_float3( -0.6f, -0.05f, 0.0f ), /// eye
	///camera_data = InitialCameraData( make_float3( -0.24178f, -0.133496f, 2.43055f ), /// eye
	camera_data = InitialCameraData( make_float3( -1.12526f, -0.306863f, 2.0142f ), /// eye
		make_float3( 0.0f, -0.05f, 0.25f ),      /// lookat
		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
		35.0f );                              /// vfov
	m_light.is_area_light = 1; 
/// 	m_light.position = make_float3( 0.0f, 0.1f, 0.0f );
/// 	m_light.anchor = make_float3( 0.0f, 0.0f, 0.0f );
/// 	m_light.direction = normalize( m_light.anchor  - m_light.position );

	///m_light.anchor = make_float3( -0.347817638160084, 0.4, 0.423288304095876 );
	m_light.anchor = make_float3( 0.323573717848716, 0.4, 0.494796481472676 );
	m_light.direction = make_float3( 0.0f, -1.0f, 0.0f );
	m_light.position = m_light.anchor - m_light.direction * 0.00001f;

	float point_size = 0.02f;
	///float point_size = 0.15f;
	///float point_size = 0.05f;
	m_light.v1 = make_float3( 1.0f, 0.f, 0.0f ) * point_size * 2;
	m_light.v2 = make_float3( 0.0f, 0.f, 1.0f ) * point_size;

	float3 m_light_t_normal;
	m_light_t_normal = cross(m_light.v1, m_light.direction);
	m_light.v1 = cross(m_light.direction, m_light_t_normal);
	m_light_t_normal = cross(m_light.v2, m_light.direction);
	m_light.v2 = cross(m_light.direction, m_light_t_normal);

	m_light.radius    = 2.0f * 0.01745329252f;
	//m_light.power     = make_float3( 0.5e4f, 0.5e4f, 0.5e4f );
	m_light.power     = make_float3( 1.0f, 1.0f, 1.0f );
	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
	float default_radius2 = 0.005f;
	//float default_radius2 = 0.01f;
	m_context["rtpass_default_radius2"]->setFloat( default_radius2);
	m_context["max_radius2"]->setFloat(default_radius2);
	optix::Aabb aabb;	
	loadObjGeometry( "box/box.obj", aabb, true);
}
void ProgressivePhotonScene::initClock(InitialCameraData& camera_data)
{
	/// 		camera_data = InitialCameraData( make_float3( -0.6f, 0.0f, 0.0f ), /// eye
	/// 			make_float3( 0.0f, 0.0f, 0.0f ),      /// lookat
	/// 			make_float3( 0.0f, 1.0f,  0.0f ),     /// up
	/// 			35.0f );                              /// vfov
	camera_data = InitialCameraData( make_float3( 0.6f, -0.05f, 0.0f ), /// eye
		make_float3( 0.0f, 0.0f, 0.0f ),      /// lookat
		make_float3( 0.0f, 1.0f,  0.0f ),     /// up
		35.0f );                              /// vfov
	m_light.is_area_light = 0; 
	///m_light.position  = 1000.0f * sphericalToCartesian( m_light_theta, m_light_phi );
	m_light.position = make_float3( 0.0f, 0.47f, 0.0f );
	m_light.direction = normalize( make_float3( 0.0f, 0.0f, 0.0f )  - m_light.position );
	m_light.radius    = 0.1f * 0.01745329252f;
	///m_light.power     = make_float3( 0.5e4f, 0.5e4f, 0.5e4f );
	m_light.power     = make_float3( 1.0f, 1.0f, 1.0f );
	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
	float default_radius2 = 0.0001f;
	m_context["rtpass_default_radius2"]->setFloat( default_radius2);
	m_context["max_radius2"]->setFloat(default_radius2);
	optix::Aabb aabb;	
	loadObjGeometry( "clocks/clocks.obj", aabb, true);
}

void ProgressivePhotonScene::initGlobal() {

	cuda_gather_cu = "ppm_gather.cu";
	cuda_ppass_cu = "ppm_ppass.cu";
	cuda_rtpass_cu = "ppm_rtpass.cu";
	triangle_mesh_cu = "triangle_mesh.cu";
	projectName = "progressivePhotonMap";

	m_print_image = false;
	m_print_camera = false;
	initAssistBuffer();
	/// There's a performance advantage to using a device that isn't being used as a display.
	/// We'll take a guess and pick the second GPU if the second one has the same compute
	/// capability as the first.
	int deviceId = 0;
	int computeCaps[2];
	if (RTresult code = rtDeviceGetAttribute(0, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps))
		throw Exception::makeException(code, 0);
	for(unsigned int index = 1; index < Context::getDeviceCount(); ++index) {
		int computeCapsB[2];
		if (RTresult code = rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCapsB))
			throw Exception::makeException(code, 0);
		if (computeCaps[0] == computeCapsB[0] && computeCaps[1] == computeCapsB[1]) {
			deviceId = index;
			break;
		}
	}

	m_context->setDevices(&deviceId, &deviceId+1);
	m_context->setRayTypeCount( RayTypeNum );
	m_context->setEntryPointCount( EnterPointNum );
	///m_context->setStackSize( 1280 );
	m_context->setStackSize( 64000 );
	///m_context->setStackSize( 960 );

	m_context["max_depth"]->setUint(8);
	m_context["max_photon_count"]->setUint(MAX_PHOTON_COUNT);
	m_context["scene_epsilon"]->setFloat( 0.0001f );
	m_context["alpha"]->setFloat( 0.7f );
	m_context["total_emitted"]->setFloat( 0.0f );
	m_context["frame_number"]->setFloat( 0.0f );
	m_context["use_debug_buffer"]->setUint( m_display_debug_buffer ? 1 : 0 );
	m_context["eye_lose_angle"]->setFloat( sin(5.0f * 45.0f/HEIGHT * M_PI / 180) );
	///m_context["eye_lose_angle"]->setFloat( sin(2.0f * 45.0f/HEIGHT * M_PI / 180) );

	/// Display buffer
	m_display_buffer = createOutputBuffer(RT_FORMAT_FLOAT4, WIDTH, HEIGHT);
	m_context["output_buffer"]->set( m_display_buffer );

	/// Direct
	m_direct_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
	m_direct_buffer->setFormat(RT_FORMAT_FLOAT3);
	m_direct_buffer->setSize( WIDTH, HEIGHT );
	m_context["direct_buffer"]->set( m_direct_buffer );

	/// Debug output buffer
	m_debug_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT );
	m_debug_buffer->setFormat( RT_FORMAT_FLOAT4 );
	m_debug_buffer->setSize( WIDTH, HEIGHT );
	m_context["debug_buffer"]->set( m_debug_buffer );

	// Set photon tag
	m_context["UsePhoton"]->setUint(1);
	m_context["UseCaustics"]->setUint(1);
	m_context["Progressive"]->setUint(0);
	// Coeff
	m_context["direct_coeff"]->setFloat(1.f);
	m_context["global_coeff"]->setFloat(100.f);
	m_context["caustics_coeff"]->setFloat(0.1f);
	// GlobalRadius2
	m_context["GlobalRadius2"]->setFloat( 10.f );
	// CausticsRadius2
	m_context["CausticsRadius2"]->setFloat( 0.f );
	
	selectScene();
}
void ProgressivePhotonScene::initEnterPointGlobalPhotonTrace() {
	// Global Output Buffer
	Global_Photon_Buffer_Size = NUM_PHOTONS;
	m_context["Global_Photon_Buffer_Size"]->setUint( Global_Photon_Buffer_Size );
	m_context["Global_Cell_Size"]->setUint( MAX_PHOTON_COUNT );
	m_Global_Photon_Buffer = m_context->createBuffer( RT_BUFFER_OUTPUT );
	m_Global_Photon_Buffer->setFormat( RT_FORMAT_USER );
	m_Global_Photon_Buffer->setElementSize( sizeof( PhotonRecord ) );
	m_Global_Photon_Buffer->setSize( Global_Photon_Buffer_Size );
	m_context["Global_Photon_Buffer"]->set( m_Global_Photon_Buffer );

	// Global Map Buffer
	Global_Photon_Map_Size = pow2roundup( NUM_PHOTONS ) - 1;
	m_context["Global_Photon_Map_Size"]->setUint(Global_Photon_Map_Size);
	m_Global_Photon_Map = m_context->createBuffer( RT_BUFFER_INPUT );
	m_Global_Photon_Map->setFormat( RT_FORMAT_USER );
	m_Global_Photon_Map->setElementSize( sizeof( PhotonRecord ) );
	m_Global_Photon_Map->setSize( Global_Photon_Map_Size );
	m_context["Global_Photon_Map"]->set( m_Global_Photon_Map );

	// Generation Program
	std::string ppass_ptx_path = ptxpath( projectName, cuda_ppass_cu);
	Program ray_gen_program = m_context->createProgramFromPTXFile( ppass_ptx_path, "global_ppass_camera" );
	m_context->setRayGenerationProgram( EnterPointGlobalPass, ray_gen_program );

	// Random seed
	Buffer Globalphoton_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT,
		RT_FORMAT_UNSIGNED_INT2,
		PHOTON_LAUNCH_WIDTH,
		PHOTON_LAUNCH_HEIGHT );
	uint2* rnd_seeds = reinterpret_cast<uint2*>( Globalphoton_rnd_seeds->map() );
	for ( unsigned int i = 0; i < PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT; ++i )
		rnd_seeds[i] = random2u();
	Globalphoton_rnd_seeds->unmap();
	m_context["Globalphoton_rnd_seeds"]->set( Globalphoton_rnd_seeds );

}
void ProgressivePhotonScene::initEnterPointCausticsPhotonTrace() {

	// Caustics Output Buffer && Caustics Count Buffer
	Caustics_Photon_Buffer_Size = NUM_PHOTONS;
	m_Caustics_Photon_Buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	m_Caustics_Photon_Buffer->setFormat(RT_FORMAT_USER);
	m_Caustics_Photon_Buffer->setElementSize(sizeof(PhotonRecord));
	m_Caustics_Photon_Buffer->setSize( Caustics_Photon_Buffer_Size );
	m_context["Caustics_Photon_Buffer"]->set(m_Caustics_Photon_Buffer);

	// Caustics Map Buffer
	Caustics_Photon_Map_Size = pow2roundup( NUM_PHOTONS ) - 1;
	m_context["Caustics_Photon_Map_Size"]->setUint(Caustics_Photon_Map_Size);
	m_Caustics_Photon_Map = m_context->createBuffer( RT_BUFFER_INPUT );
	m_Caustics_Photon_Map->setFormat( RT_FORMAT_USER );
	m_Caustics_Photon_Map->setElementSize( sizeof( PhotonRecord ) );
	m_Caustics_Photon_Map->setSize( Caustics_Photon_Map_Size );
	m_context["Caustics_Photon_Map"]->set( m_Caustics_Photon_Map );

	// Generation Program
	std::string ppass_ptx_path = ptxpath( projectName, cuda_ppass_cu);
	Program ray_gen_program = m_context->createProgramFromPTXFile( ppass_ptx_path, "caustics_ppass_camera" );
	m_context->setRayGenerationProgram( EnterPointCausticsPass, ray_gen_program );

	Buffer Causticsphoton_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, 
		RT_FORMAT_UNSIGNED_INT, Caustics_Photon_Buffer_Size );
	uint* seeds = reinterpret_cast<uint*>( Causticsphoton_rnd_seeds->map() );
	for ( unsigned int i = 0; i < Caustics_Photon_Buffer_Size; ++i )
		seeds[i] = random1u();
	Causticsphoton_rnd_seeds->unmap();
	m_context["Causticsphoton_rnd_seeds"]->set( Causticsphoton_rnd_seeds );

	m_context["target_max"]->setFloat(0,0,0);
	m_context["target_min"]->setFloat(0,0,0);
}
void ProgressivePhotonScene::initEnterPointRayTrace(InitialCameraData& camera_data) {
	/// RTPass output buffer
	Buffer output_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT );
	output_buffer->setFormat( RT_FORMAT_USER );
	output_buffer->setElementSize( sizeof( HitRecord ) );
	output_buffer->setSize( WIDTH, HEIGHT );
	m_context["rtpass_output_buffer"]->set( output_buffer );

	/// RTPass ray gen program
	std::string ptx_path = ptxpath( projectName, cuda_rtpass_cu );
	Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "rtpass_camera" );
	m_context->setRayGenerationProgram( EnterPointRayTrace, ray_gen_program );

	/// RTPass exception/miss programs
	Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "rtpass_exception" );
	m_context->setExceptionProgram( EnterPointRayTrace, exception_program );
	m_context["rtpass_bad_color"]->setFloat( 1.0f, 0.0f, 0.0f );
	m_context->setMissProgram( EnterPointRayTrace, m_context->createProgramFromPTXFile( ptx_path, "rtpass_miss" ) );
	m_context["rtpass_bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );

	/// RTPass pixel sample buffers
	Buffer image_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT2, WIDTH, HEIGHT );
	m_context["image_rnd_seeds"]->set( image_rnd_seeds );
	uint2* seeds = reinterpret_cast<uint2*>( image_rnd_seeds->map() );
	for ( unsigned int i = 0; i < WIDTH*HEIGHT; ++i )  
		seeds[i] = random2u();
	image_rnd_seeds->unmap();

	/// Set up camera
	camera_data = InitialCameraData( make_float3( 278.0f, 273.0f, -800.0f ), /// eye
		make_float3( 278.0f, 273.0f, 0.0f ),    /// lookat
		make_float3( 0.0f, 1.0f,  0.0f ),       /// up
		35.0f );                                /// vfov

	/// Declare these so validation will pass
	m_context["rtpass_eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["rtpass_U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["rtpass_V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
	m_context["rtpass_W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

	// Material
}
void ProgressivePhotonScene::initEnterPointGlobalGather() {
	/// Gather phase
	std::string gather_ptx_path = ptxpath( projectName, cuda_gather_cu );
	std::string gather_program_name = "globalDensity";
	if (m_test_scene == Cornel_Box_Scene)
		gather_program_name = gather_program_name.append("_cornel");
	Program gather_program = m_context->createProgramFromPTXFile( gather_ptx_path, gather_program_name );
	m_context->setRayGenerationProgram( EnterPointGlobalGather, gather_program );
	Program exception_program = m_context->createProgramFromPTXFile( gather_ptx_path, "gather_exception" );
	m_context->setExceptionProgram( EnterPointGlobalGather, exception_program );
}
void ProgressivePhotonScene::initEnterPointCausticsGather() {
	/// Gather phase
	std::string gather_ptx_path = ptxpath( projectName, cuda_gather_cu );
	std::string gather_program_name = "causticsDensity";
	if (m_test_scene == Cornel_Box_Scene)
		gather_program_name = gather_program_name.append("_cornel");
	Program gather_program = m_context->createProgramFromPTXFile( gather_ptx_path, gather_program_name );
	m_context->setRayGenerationProgram( EnterPointCausticsGather, gather_program );
	Program exception_program = m_context->createProgramFromPTXFile( gather_ptx_path, "gather_exception" );
	m_context->setExceptionProgram( EnterPointCausticsGather, exception_program );
}
void ProgressivePhotonScene::initGeometryInstances(InitialCameraData& camera_data) {
		/// Populate scene hierarchy
	if( m_test_scene == Wedding_Ring_Scene )
		initWeddingRing(camera_data);
	else if( m_test_scene == Conference_Scene )
		initConference(camera_data);
	else if( m_test_scene == Sponza_Scene )
		initSponza(camera_data);
	else if( m_test_scene == Clock_Scene )
		initClock(camera_data);
	else if (m_test_scene == Cornel_Box_Scene)
		initCornelBox(camera_data);
	else if (m_test_scene == Box_Scene)
		initBox(camera_data);
	else if (m_test_scene == Small_Room_Scene)
		initSmallRoom(camera_data);
	else if (m_test_scene == Sibenik_Scene)
		initSibenik(camera_data);
	else if (m_test_scene == Torus_Scene)
		initTorus(camera_data);

	m_context["ambient_light"]->setFloat( 0.1f, 0.1f, 0.1f);
	std::string full_path = std::string( sutilSamplesDir() ) + "/tutorial/data/CedarCity.hdr";
	///const float3 default_color = make_float3( 0.8f, 0.88f, 0.97f );
	const float3 default_color = make_float3( 0.0f );
	//m_context["envmap"]->setTextureSampler( loadTexture( m_context, full_path, default_color) );
	m_context["envmap"]->setTextureSampler( loadTexture( m_context, "", default_color) );
}

void ProgressivePhotonScene::initScene( InitialCameraData& camera_data )
{
	cout << "Begin to init GAPPM context...\n";

	initGlobal();
	initEnterPointRayTrace(camera_data);
	initEnterPointGlobalPhotonTrace();
	initEnterPointCausticsPhotonTrace();
	initEnterPointGlobalGather();
	initEnterPointCausticsGather();
	initGeometryInstances(camera_data);

	/*
	/// Prepare to run
	m_context->launch(0, 0);

	int m_cuda_device = OptiXDeviceToCUDADevice( m_context, 0 );
	
	if ( m_cuda_device < 0 ) {
		std::cerr << "OptiX device 0 must be a valid CUDA device number.\n";
		exit(1);
	}
    cudaSetDevice(m_cuda_device);
	*/
	m_context->validate();
	m_context->compile();

	cout << "Context init finished\n" ;
}

Buffer ProgressivePhotonScene::getOutputBuffer()
{
	return m_display_buffer;
}

inline uchar4 makeColor( const float3& c )
{
	uchar4 pixel;
	pixel.x = static_cast<unsigned char>( fmaxf( fminf( c.z, 1.0f ), 0.0f ) * 255.99f );
	pixel.y = static_cast<unsigned char>( fmaxf( fminf( c.y, 1.0f ), 0.0f ) * 255.99f );
	pixel.z = static_cast<unsigned char>( fmaxf( fminf( c.x, 1.0f ), 0.0f ) * 255.99f );
	pixel.w = 0; 
	return pixel;
}


bool photonCmpX( PhotonRecord* r1, PhotonRecord* r2 ) { return r1->position.x < r2->position.x; }
bool photonCmpY( PhotonRecord* r1, PhotonRecord* r2 ) { return r1->position.y < r2->position.y; }
bool photonCmpZ( PhotonRecord* r1, PhotonRecord* r2 ) { return r1->position.z < r2->position.z; }


void buildKDTree( PhotonRecord** photons, int start, int end, int depth, PhotonRecord* kd_tree, int current_root,
	SplitChoice split_choice, float3 bbmin, float3 bbmax)
{
	/// If we have zero photons, this is a NULL node
	if( end - start == 0 ) {
		kd_tree[current_root].axis = PPM_NULL;
		kd_tree[current_root].energy = make_float3( 0.0f );
		return;
	}

	/// If we have a single photon
	if( end - start == 1 ) {
		photons[start]->axis = PPM_LEAF;
		kd_tree[current_root] = *(photons[start]);
		return;
	}

	/// Choose axis to split on
	int axis;
	switch(split_choice) {
	case RoundRobin:
		{
			axis = depth%3;
		}
		break;
	case HighestVariance:
		{
			float3 mean  = make_float3( 0.0f ); 
			float3 diff2 = make_float3( 0.0f );
			for(int i = start; i < end; ++i) {
				float3 x     = photons[i]->position;
				float3 delta = x - mean;
				float3 n_inv = make_float3( 1.0f / ( static_cast<float>( i - start ) + 1.0f ) );
				mean = mean + delta * n_inv;
				diff2 += delta*( x - mean );
			}
			float3 n_inv = make_float3( 1.0f / ( static_cast<float>(end-start) - 1.0f ) );
			float3 variance = diff2 * n_inv;
			axis = max_component(variance);
		}
		break;
	case LongestDim:
		{
			float3 diag = bbmax-bbmin;
			axis = max_component(diag);
		}
		break;
	default:
		axis = -1;
		std::cerr << "Unknown SplitChoice " << split_choice << " at "<<__FILE__<<":"<<__LINE__<<"\n";
		exit(2);
		break;
	}

	int median = (start+end) / 2;
	PhotonRecord** start_addr = &(photons[start]);
#if 0
	switch( axis ) {
	case 0:
		std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpX );
		photons[median]->axis = PPM_X;
		break;
	case 1:
		std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpY );
		photons[median]->axis = PPM_Y;
		break;
	case 2:
		std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpZ );
		photons[median]->axis = PPM_Z;
		break;
	}
#else
	switch( axis ) {
	case 0:
		select<PhotonRecord*, 0>( start_addr, 0, end-start-1, median-start );
		photons[median]->axis = PPM_X;
		break;
	case 1:
		select<PhotonRecord*, 1>( start_addr, 0, end-start-1, median-start );
		photons[median]->axis = PPM_Y;
		break;
	case 2:
		select<PhotonRecord*, 2>( start_addr, 0, end-start-1, median-start );
		photons[median]->axis = PPM_Z;
		break;
	}
#endif
	float3 rightMin = bbmin;
	float3 leftMax  = bbmax;
	if(split_choice == LongestDim) {
		float3 midPoint = (*photons[median]).position;
		switch( axis ) {
		case 0:
			rightMin.x = midPoint.x;
			leftMax.x  = midPoint.x;
			break;
		case 1:
			rightMin.y = midPoint.y;
			leftMax.y  = midPoint.y;
			break;
		case 2:
			rightMin.z = midPoint.z;
			leftMax.z  = midPoint.z;
			break;
		}
	}
	PhotonRecord* tempRecord = photons[median];
	kd_tree[current_root] = *tempRecord;
	buildKDTree( photons, start, median, depth+1, kd_tree, 2*current_root+1, split_choice, bbmin,  leftMax );
	buildKDTree( photons, median+1, end, depth+1, kd_tree, 2*current_root+2, split_choice, rightMin, bbmax );
}

bool cmpPhotonRecord (const PhotonRecord * a, const PhotonRecord *b)
{
	return a->pad.z < b->pad.z;
}

/// 在每一个mesh上构建Voronoi图
void ProgressivePhotonScene::constructVoronoiSubdivision(PhotonRecord** photon_ptrs, int photons_count)
{
	double t0, t1;
	double t2 = 0;
	
	if (m_print_timings) std::cerr << "Starting Voronoi Data Transform ... " << std::endl;
	sutilCurrentTime( &t0 );
	/// 建立从mesh的index到其上光子集合的映射表

	PhotonRecord** listMaps = (PhotonRecord**) malloc((photons_count + 1) * sizeof(PhotonRecord*));
	/// 将光子按照其所在的mesh分类

	for (int i = 0;i < photons_count;i ++)
		listMaps[i + 1] = photon_ptrs[i];

	sort(listMaps + 1, listMaps + photons_count + 1, cmpPhotonRecord);

	sutilCurrentTime( &t1 );
	if (m_print_timings) std::cerr << "Voronoi Data Transform finished. " << t1 - t0 << std::endl << std::endl;
	// 为每一个包含光子的mesh进行Voronoi划分
	
	Voronoi_CPU::m = 1;
	for (int i = 2; i <= photons_count; ++i)
		if (listMaps[i]->pad.z != listMaps[i - 1]->pad.z)
			++Voronoi_CPU::m;
	Voronoi_CPU::frag = (Voronoi_CPU::Frag*) malloc((Voronoi_CPU::m + 1) * sizeof(Voronoi_CPU::Frag));
	if (Voronoi_CPU::frag == NULL) { fprintf(stderr, "No Enough Memory!\n"); exit(0); }
	
	Voronoi_CPU::n = photons_count;
	Voronoi_CPU::frag[1].l = 1;
	Voronoi_CPU::m = 1;

	for (int i = 2; i <= photons_count; ++i)
		if (listMaps[i]->pad.z != listMaps[i - 1]->pad.z)
		{
			Voronoi_CPU::frag[Voronoi_CPU::m].r = i - 1;
			++Voronoi_CPU::m;
			Voronoi_CPU::frag[Voronoi_CPU::m].l = i;
		}
	Voronoi_CPU::frag[Voronoi_CPU::m].r = photons_count;
	
	Voronoi_CPU::d = (Voronoi_CPU::PointWithID*) malloc((Voronoi_CPU::n + 1) * sizeof(Voronoi_CPU::PointWithID));
	if (Voronoi_CPU::d == NULL) { fprintf(stderr, "No Enough Memory!\n"); exit(0); }

	int cntN = 0;
	for (int i = 1; i <= Voronoi_CPU::m; ++i)
	{
		int triangleIndex = listMaps[Voronoi_CPU::frag[i].l]->pad.z;
 		PointFloat3 trianglePoints[3];
		unsigned int *t_vindex = loader->model->triangles[triangleIndex].vindices;
		float *apoint = &(loader->model->vertices[t_vindex[0] * 3]);
		float *bpoint = &(loader->model->vertices[t_vindex[1] * 3]);
		float *cpoint = &(loader->model->vertices[t_vindex[2] * 3]);
		trianglePoints[0] = PointFloat3(apoint[0], apoint[1], apoint[2]);
		trianglePoints[1] = PointFloat3(bpoint[0], bpoint[1], bpoint[2]);
		trianglePoints[2] = PointFloat3(cpoint[0], cpoint[1], cpoint[2]);

		PointFloat3 x_ = trianglePoints[1] - trianglePoints[0];
		x_.normalise();
		Voronoi_CPU::frag[i].a = Voronoi_CPU::Point(0.0, 0.0);
		Voronoi_CPU::frag[i].b = Voronoi_CPU::Point(x_.Dot(trianglePoints[1] - trianglePoints[0]), x_.Cross(trianglePoints[1] - trianglePoints[0]).Length()) * Voronoi_CPU::Muls;
		Voronoi_CPU::frag[i].c = Voronoi_CPU::Point(x_.Dot(trianglePoints[2] - trianglePoints[0]), x_.Cross(trianglePoints[2] - trianglePoints[0]).Length()) * Voronoi_CPU::Muls;

		for (int j = Voronoi_CPU::frag[i].l; j <= Voronoi_CPU::frag[i].r; ++j)
		{
			PointFloat3 tmp = PointFloat3(listMaps[j]->position.x, listMaps[j]->position.y, listMaps[j]->position.z);
			++cntN;
			Voronoi_CPU::Point tmp2 = Voronoi_CPU::Point(x_.Dot(tmp - trianglePoints[0]), x_.Cross(tmp - trianglePoints[0]).Length()) * Voronoi_CPU::Muls;
			Voronoi_CPU::d[j].x = tmp2.x;
			Voronoi_CPU::d[j].y = tmp2.y;
			Voronoi_CPU::d[j].ID = j;
		}
	}
	if (m_print_timings) std::cerr << "Starting Voronoi Calculate  ... " << std::endl;
	sutilCurrentTime( &t0 );
	Voronoi_CPU::calcVoronoi();
	sutilCurrentTime( &t1 );
	if (m_print_timings) std::cerr << "Voronoi Calculate finished. " << t1 - t0 << std::endl << std::endl;
	//Voronoi_CPU::calcArea();
	calcArea_GPU(Voronoi_CPU::n, Voronoi_CPU::m, Voronoi_CPU::d, Voronoi_CPU::g, Voronoi_CPU::edge, Voronoi_CPU::frag, Voronoi_CPU::Area, Voronoi_CPU::total, Voronoi_CPU::edgeCount, Voronoi_CPU::maxcnt);
	for (int i = 1; i <= Voronoi_CPU::m; ++i)
	{
		for (int j = Voronoi_CPU::frag[i].l; j <= Voronoi_CPU::frag[i].r; ++j)
		{
			{
				double presentArea = Voronoi_CPU::Area[j] / (Voronoi_CPU::Muls * Voronoi_CPU::Muls);
				listMaps[j]->pad.y = transferFloat * presentArea;
			}
		}
	}
	Voronoi_CPU::freeData();
	free(listMaps);
}

// 在CPU端计算voronoi单元格面积，并创建光子图的KD-tree
void ProgressivePhotonScene::buildGlobalPhotonMap()
{
	double t0, t1;

	if (m_print_timings) std::cerr << "Starting Global photon pass   ... ";

	Buffer Globalphoton_rnd_seeds = m_context["Globalphoton_rnd_seeds"]->getBuffer();
	uint2* seeds = reinterpret_cast<uint2*>( Globalphoton_rnd_seeds->map() );
	for ( unsigned int i = 0; i < PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT; ++i )
		seeds[i] = random2u();
	Globalphoton_rnd_seeds->unmap();

	sutilCurrentTime( &t0 );
	
	m_context->launch( EnterPointGlobalPass,
		static_cast<unsigned int>(PHOTON_LAUNCH_WIDTH),
		static_cast<unsigned int>(PHOTON_LAUNCH_HEIGHT) );

	/// By computing the total number of photons as an unsigned long long we avoid 32 bit
	/// floating point addition errors when the number of photons gets sufficiently large
	/// (the error of adding two floating point numbers when the mantissa bits no longer
	/// overlap).
	m_context["total_emitted"]->setFloat( static_cast<float>((unsigned long long)(m_frame_number+1)*PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT) );
	
	sutilCurrentTime( &t1 );
	if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;


	PhotonRecord* photons_data    = reinterpret_cast<PhotonRecord*>( m_Global_Photon_Buffer->map() );
	PhotonRecord* photon_map_data = reinterpret_cast<PhotonRecord*>( m_Global_Photon_Map->map() );

	for( unsigned int i = 0; i < Global_Photon_Map_Size; ++i ) {
		photon_map_data[i].energy = make_float3( 0.0f );
	}

	/// Push all valid photons to front of list
	unsigned int valid_photons = 0;
	PhotonRecord** temp_photons = new PhotonRecord*[NUM_PHOTONS];
	for( unsigned int i = 0; i < NUM_PHOTONS; ++i ) {
		if( fmaxf( photons_data[i].energy ) > 0.0f ) {
			temp_photons[valid_photons++] = &photons_data[i];
		}
	}
	if ( m_display_debug_buffer ) {
		std::cerr << " ** valid_photon/NUM_PHOTONS =  " 
			<< valid_photons<<"/"<<NUM_PHOTONS
			<<" ("<<valid_photons/static_cast<float>(NUM_PHOTONS)<<")\n";
	}

	/// Make sure we arent at most 1 less than power of 2
	valid_photons = valid_photons >= Global_Photon_Map_Size ? Global_Photon_Map_Size : valid_photons;
	m_current_valid_photons = valid_photons;
	float3 bbmin = make_float3(0.0f);
	float3 bbmax = make_float3(0.0f);
	if( m_split_choice == LongestDim ) {
		bbmin = make_float3(  std::numeric_limits<float>::max() );
		bbmax = make_float3( -std::numeric_limits<float>::max() );
		/// Compute the bounds of the photons
		for(unsigned int i = 0; i < valid_photons; ++i) {
			float3 position = (*temp_photons[i]).position;
			bbmin = fminf(bbmin, position);
			bbmax = fmaxf(bbmax, position);
		}
	}	
	
	if (m_print_timings) std::cerr << "Starting Voronoi Graph build ... " << std::endl;
	sutilCurrentTime( &t0 );
	constructVoronoiSubdivision(temp_photons, valid_photons);
	sutilCurrentTime( &t1 );
	if (m_print_timings) std::cerr << "Voronoi Graph build finished. " << t1 - t0 << std::endl;

	/// Build KD tree 
	if (m_print_timings) std::cerr << "Starting Global kd_tree build ... " << std::endl;
	sutilCurrentTime( &t0 );
	buildKDTree( temp_photons, 0, valid_photons, 0, photon_map_data, 0, m_split_choice, bbmin, bbmax );
	sutilCurrentTime( &t1 );
	if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

	delete[] temp_photons;
	m_Global_Photon_Map->unmap();
	m_Global_Photon_Buffer->unmap();
}

void ProgressivePhotonScene::buildCausticsPhotonMap()
{
	uint photonStart = 0;
	for (int it = 0; it < loader->m_Caustics_Max.size();it ++)
	{
		uint targetSize = Caustics_Photon_Buffer_Size
			/// m_Caustics_Max.size();
			* loader->volumeArray[it] / loader->volumeSum;
		float3 m_Caustics_Max = loader->m_Caustics_Max[it];
		float3 m_Caustics_Min = loader->m_Caustics_Min[it];
		if (targetSize > 0)
		{
			// Target box
			m_context["target_max"]->setFloat(m_Caustics_Max.x, 
				m_Caustics_Max.y, m_Caustics_Max.z);
			m_context["target_min"]->setFloat(m_Caustics_Min.x, 
				m_Caustics_Min.y, m_Caustics_Min.z);
			m_context["PhotonStart"]->setUint(photonStart);
			m_context->launch( EnterPointCausticsPass,  targetSize);
			photonStart += targetSize;
		}
	}
	m_context["total_emitted"]->setFloat( static_cast<float>((unsigned long long)(m_frame_number+1) * photonStart) );

	PhotonRecord* photons_data    = reinterpret_cast<PhotonRecord*>( m_Caustics_Photon_Buffer->map() );
	PhotonRecord* photon_map_data = reinterpret_cast<PhotonRecord*>( m_Caustics_Photon_Map->map() );

	for( unsigned int i = 0; i < Caustics_Photon_Map_Size; ++i ) {
		photon_map_data[i].energy = make_float3( 0.0f );
	}

	/// Push all valid photons to front of list
	unsigned int valid_photons = 0;
	PhotonRecord** temp_photons = new PhotonRecord*[Caustics_Photon_Map_Size];
	for( unsigned int i = 0; i < Caustics_Photon_Map_Size; ++i ) {
		PhotonRecord tmp = photons_data[i];
		if( fmaxf( photons_data[i].energy ) > 0.0f ) {
			//cout << tmp.position.x << " " << tmp.position.y << " " << tmp.position.z << " " << endl;
			temp_photons[valid_photons++] = &photons_data[i];
		}
	}
	if ( m_display_debug_buffer ) {
		std::cerr << " ** valid_photon/NUM_PHOTONS =  " 
			<< valid_photons<<"/"<<Caustics_Photon_Map_Size
			<<" ("<<valid_photons/static_cast<float>(Caustics_Photon_Map_Size)<<")\n";
	}

	/// Make sure we arent at most 1 less than power of 2
	valid_photons = valid_photons >= Caustics_Photon_Map_Size ? Caustics_Photon_Map_Size : valid_photons;
	m_current_valid_photons = valid_photons;
	float3 bbmin = make_float3(0.0f);
	float3 bbmax = make_float3(0.0f);
	if( m_split_choice == LongestDim ) {
		bbmin = make_float3(  std::numeric_limits<float>::max() );
		bbmax = make_float3( -std::numeric_limits<float>::max() );
		/// Compute the bounds of the photons
		for(unsigned int i = 0; i < valid_photons; ++i) {
			float3 position = (*temp_photons[i]).position;
			bbmin = fminf(bbmin, position);
			bbmax = fmaxf(bbmax, position);
		}
	}	
	
	constructVoronoiSubdivision(temp_photons, valid_photons);

	/// Now build KD tree
	buildKDTree( temp_photons, 0, valid_photons, 0, photon_map_data, 0, m_split_choice, bbmin, bbmax );

	delete[] temp_photons;
	m_Caustics_Photon_Map->unmap();
	m_Caustics_Photon_Buffer->unmap();
}

void exchangeValue(float3 ** p0, float3 ** p1)
{
	float3* temp_ptr = *p0;
	*p0 = *p1;
	*p1 = temp_ptr;
}
float3 make_float3_from_float(float* float_v)
{
	return make_float3(float_v[0], float_v[1], float_v[2]);
}

/// Determine whether point P in triangle ABC
bool pointInTriangle(float3 A, float3 B, float3 C, float3 P)
{
	bool tm = 
		(abs(A.x)+abs(B.x)+abs(C.x) > 199.0f && A.y >0&&B.y> 0&&C.y>0  && A.x+B.x+(C.x) < 20.0f)
		&&(abs(A.z)+abs(B.z)+abs(C.z) > 199.0f && A.z+B.z+(C.z) < 20.0f);
	float3 v0 = C - A ;
	float3 v1 = B - A ;
	float3 v2 = P - A ;

	float dot00 = dot(v0, v0) ;
	float dot01 = dot(v0, v1) ;
	float dot02 = dot(v0, v2) ;
	float dot11 = dot(v1, v1) ;
	float dot12 = dot(v1, v2) ;

	float inverDeno = 1 / (dot00 * dot11 - dot01 * dot01) ;

	float u = (dot11 * dot02 - dot01 * dot12) * inverDeno ;
	if (u < 0.0 || u > 1.0) /// if u out of range, return directly
	{
		if (tm)
			int t = 0;
		return false ;
	}

	float v = (dot00 * dot12 - dot01 * dot02) * inverDeno ;
	if (v < 0.0 || v > 1.0) /// if v out of range, return directly
	{
		if (tm)
			int t = 0;
		return false ;
	}

	return u + v <= 1.0 ;
}

/// Determine whether point P in triangle ABC
bool point_in_triangle(float3 A, float3 B, float3 C, float3 P)
{
	float t_area = length( cross(A-B, B-C) );
	float t_area1 = length( cross(A-P, B-P) );
	float t_area2 = length( cross(B-P, C-P) );
	float t_area3 = length( cross(A-P, C-P) );

	if (t_area > t_area1 + t_area2 + t_area3 + 0.01f)
		return false;
	return true;
}

float intersect_length(float3 c_p, float c_r, float3 i_p, float3 o_p,  float3 pf, float *mm_area)
{
	float3 v_o = o_p - i_p;
	if (length(v_o) < 0.000001)
		return 0;
	float3 v_i = c_p - i_p;
	float d_p = dot(v_o, v_i)/length(v_o);

	float one_e = sqrt(c_r*c_r - d_p*d_p);
	float other_e = sqrt(dot(v_i, v_i) - d_p*d_p);
	if (abs(d_p) > 0.000001)
		one_e += d_p/abs(d_p)*other_e;
	*mm_area = one_e * abs(d_p) / 2.0f;

	float3 t_normal = cross(cross(pf - i_p, o_p - i_p), o_p - i_p);
	if (dot(t_normal, pf - i_p) < 0.0001f)
		t_normal = -t_normal;
	if ( dot(c_p - i_p, t_normal) < 0.0001f)
		*mm_area = -*mm_area;
	return one_e;
}

float intersect_clip(float3 c_p, float c_r, float3 p0, float3 p1, float3 pf)
{
	float refer_area = length(cross(p1 - p0, p0 - c_p));
	float dist = refer_area/length(p1 - p0);
	if (dist > c_r)
		return 0;
	else
	{
		float sector_area = c_r * c_r * acosf(dist/c_r) - dist * sqrt(c_r*c_r - dist*dist);
		float3 t_normal = cross(cross(pf - p0, p1 - p0), p1 - p0);
		if (dot(t_normal, pf - p0) < -0.0001f)
			t_normal = -t_normal;
		if ( dot(c_p - p0, t_normal) < -0.00001f)
			sector_area = M_PI * c_r * c_r - sector_area;
		return sector_area;
	}
}
float3 getNormal(float* p0, float* p1, float* p2)
{
	float3 v0 = make_float3_from_float(p0), 
		v1 = make_float3_from_float(p1), 
		v2 = make_float3_from_float(p2);
	const float3 e0 = v1 - v0;
	const float3 e1 = v0 - v2;
	return normalize(cross( e1, e0 ));
}

bool getOverlapArea(float2 sp_position, float kernelRadius, 
	float2 p0, float2 p1, float2 p2, float* m_area)
{
	return true;
}

bool getOverlapArea(float3 sp_position, float3 sp_normal, float kernelRadius, 
	float3 p0, float3 p1, float3 p2, float* m_area)
{
	*m_area = 0;

	/// Project points
	float3 pv0 = p0 + sp_normal * dot(sp_position - p0, sp_normal);
	float3 pv1 = p1 + sp_normal * dot(sp_position - p1, sp_normal);
	float3 pv2 = p2 + sp_normal * dot(sp_position - p2, sp_normal);

	/// Triangle degeneration
	if (length(pv0 - pv1) < 0.001 || length(pv1 - pv2) < 0.001 || length(pv2 - pv0) < 0.001)
		return false;

	float3 *ptr0 = &pv0,*ptr1 = &pv1, *ptr2 = &pv2;
	int pv0_inside = length(pv0 - sp_position) < kernelRadius?1:0, 
		pv1_inside = length(pv1 - sp_position) < kernelRadius?1:0,
		pv2_inside = length(pv2 - sp_position) < kernelRadius?1:0;
	int insideNum = pv0_inside + pv1_inside + pv2_inside;
	float circle_area = M_PI * kernelRadius * kernelRadius;

	if (pointInTriangle(sp_position, pv0, pv1, pv2) == false)
		int mydebug = 0;

	if (insideNum == 0)
	{
		float clip_area = 0;
		clip_area += intersect_clip(sp_position, kernelRadius, pv0, pv1, pv2);
		clip_area += intersect_clip(sp_position, kernelRadius, pv0, pv2, pv1);
		clip_area += intersect_clip(sp_position, kernelRadius, pv1, pv2, pv0);
		if (clip_area < 0.001f)
			*m_area = point_in_triangle(pv0, pv1, pv2, sp_position)?circle_area:0;
		else
		{
			if (circle_area > clip_area)
				*m_area = circle_area - clip_area;
			else
				*m_area = Myron_Green_Mid;
		}
		return true;
	}
	/// change 0,2 and 1,2 and 0,1
	if (pv0_inside == 0 && pv1_inside == 0 && pv2_inside == 1) exchangeValue(&ptr0, &ptr2);
	if (pv0_inside == 0 && pv1_inside == 1 && pv2_inside == 0) exchangeValue(&ptr0, &ptr1);
	if (pv0_inside == 0 && pv1_inside == 1 && pv2_inside == 1) exchangeValue(&ptr0, &ptr2);
	if (pv0_inside == 1 && pv1_inside == 0 && pv2_inside == 1) exchangeValue(&ptr1, &ptr2);

	*m_area = 0.5 * length( cross(pv0 - pv1, pv0 - pv2) );
	if (insideNum == 1)
	{
		if (pointInTriangle(sp_position, pv0, pv1, pv2) == false)
			int mydebug = 0;

		///*m_area = length(*ptr0 - sp_position)/kernelRadius * circle_area;
		float len_1, len_2, tLen_1, tLen_2;
		float traingle_areas1 = 0, traingle_areas2 = 0;
		len_1 = intersect_length(sp_position, kernelRadius, *ptr0, *ptr1, *ptr2, &traingle_areas1);
		len_2 = intersect_length(sp_position, kernelRadius, *ptr0, *ptr2, *ptr2, &traingle_areas2);

		if (len_1 > 2*kernelRadius || len_2 > 2*kernelRadius)
			int mydebug = 0;

		tLen_1 = length(*ptr0 - *ptr1);
		tLen_2 = length(*ptr0 - *ptr2);
		if (tLen_2 < len_2 || tLen_1 < len_1)
			int mydebug = 0;

		float3 v_p1 = (*ptr1 - *ptr0)*len_1/tLen_1;
		float3 v_p2 = (*ptr2 - *ptr0)*len_2/tLen_2;

		float half_edge_length = length( v_p1 - v_p2 ) / 2.0f;
		if (half_edge_length > kernelRadius)
			int mydebug = 0;
		///float traingle_areas = 0.5*length( cross(v_p1, sp_position - *ptr0) ) + 0.5*length( cross(v_p2, sp_position - *ptr0) );
		float traingle_areas = traingle_areas1 + traingle_areas2;
		float shan_areas = kernelRadius*kernelRadius*asinf(half_edge_length/kernelRadius);
		/// Method 1
		float ab_area = kernelRadius*kernelRadius*asinf(half_edge_length/kernelRadius) 
			- half_edge_length * sqrt(kernelRadius*kernelRadius - half_edge_length*half_edge_length);
		*m_area *= len_1*len_2/tLen_1/tLen_2;
		*m_area = *m_area + ab_area;

		/// 		/// Method 2
		/// 		*m_area = traingle_areas + shan_areas;				
		/// 		float cos_Center = dot(v_p1, v_p2)/length(v_p1)/length(v_p2);
		/// 		if (cos_Center > 0.8)
		/// 			int mydebug = 0;

		return true;
	}
	/// 	*m_area = Myron_Green_Mid;
	/// 	return true;
	else if (insideNum == 2)
	{
		float t_area;
		float len_1 = intersect_length(sp_position, kernelRadius, *ptr0, *ptr2, *ptr1, &t_area);
		float len_2 = intersect_length(sp_position, kernelRadius, *ptr1, *ptr2, *ptr1, &t_area);
		*m_area *= (1 - len_1/length(*ptr0 - *ptr2)) * (1 - len_2/length(*ptr1 - *ptr2));
	}
	*m_area = Myron_Green_Mid;
	return true;
}

void ProgressivePhotonScene::getKernelArea(float3 sp_position, float3 sp_normal, float kernelRadius, 
	int p0, int p1, int p2,
	std::set<int>& targetTriangleSet,
	float& kernelArea, int cur_depth)
{
	/// From vertex index to nearby triangle
	std::vector<int>& m_vertex_triangle_vector0 = loader->vertexIndexTablePtr->at(p0);
	std::vector<int>& m_vertex_triangle_vector1 = loader->vertexIndexTablePtr->at(p1);
	std::vector<int>& m_vertex_triangle_vector2 = loader->vertexIndexTablePtr->at(p2);
	std::set<int> tempTriangleSet;
	tempTriangleSet.insert(m_vertex_triangle_vector0.begin(), m_vertex_triangle_vector0.end());
	tempTriangleSet.insert(m_vertex_triangle_vector1.begin(), m_vertex_triangle_vector1.end());
	tempTriangleSet.insert(m_vertex_triangle_vector2.begin(), m_vertex_triangle_vector2.end());

	/// Add Triangle to temp triangle
	std::vector<int3> tempTriangleVector;
	for (std::set<int>::iterator set_iterator = tempTriangleSet.begin();set_iterator != tempTriangleSet.end();set_iterator ++)
	{
		int target_traingle = *set_iterator;
		/// it does exist
		if (targetTriangleSet.find(target_traingle) != targetTriangleSet.end())
			continue;
		targetTriangleSet.insert(target_traingle);
		unsigned int *t_vindex = loader->model->triangles[target_traingle].vindices;
		float3 temp_normal;
		unsigned int *t_nindex = loader->model->triangles[target_traingle].nindices;
		if (t_nindex[0] > 0 && t_nindex[1] > 0 && t_nindex[2] > 0)
		{
			temp_normal = normalize(
				make_float3_from_float(loader->model->normals + t_nindex[0]*3)
				+ make_float3_from_float(loader->model->normals + t_nindex[1]*3)
				+ make_float3_from_float(loader->model->normals + t_nindex[2]*3));
		}
		else temp_normal = getNormal(loader->model->vertices + t_vindex[0]*3,
			loader->model->vertices + t_vindex[1]*3,
			loader->model->vertices + t_vindex[2]*3);

		/// Normal is not correct
		if (dot(temp_normal, sp_normal) < 0.001f)
			continue;

		float m_area = 0;
		/// it not overlap
		if (getOverlapArea(sp_position, sp_normal, kernelRadius, 
			make_float3_from_float(loader->model->vertices + t_vindex[0]*3),
			make_float3_from_float(loader->model->vertices + t_vindex[1]*3),
			make_float3_from_float(loader->model->vertices + t_vindex[2]*3),
			&m_area) == false)
			continue;
		kernelArea += m_area;
		int3 temp_int3 = make_int3(t_vindex[0], t_vindex[1], t_vindex[2]) - make_int3(1);
		tempTriangleVector.push_back(temp_int3);
	}
	if (cur_depth > 1)
		return;
	for (int i = 0;i < tempTriangleVector.size();i ++)
		getKernelArea(sp_position, sp_normal, kernelRadius, 
		tempTriangleVector[i].x, tempTriangleVector[i].y, tempTriangleVector[i].z, targetTriangleSet, kernelArea,
		cur_depth + 1);
}
void ProgressivePhotonScene::calculateKernelAreaWithVertex()
{
	Buffer output_buffer = m_context["rtpass_output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	output_buffer->getSize( buffer_width, buffer_height );
	int4* triangle_infos = reinterpret_cast<int4*>( m_sp_triangle_info_buffer->map() );
	float3* sp_positions = reinterpret_cast<float3*>( m_sp_position_buffer->map() );
	float3* sp_normals = reinterpret_cast<float3*>( m_sp_normal_buffer->map() );
	float* sp_radiuses = reinterpret_cast<float*>( m_sp_radius_buffer->map() );
	float* sp_areas = reinterpret_cast<float*>( m_sp_area_buffer->map() );
	float current_pro = 0.1;
	for( unsigned int j = 0; j < buffer_height; ++j )
	{
		for( unsigned int i = 0; i < buffer_width; ++i )
		{
			int m_launch_index = j*buffer_width+i;
			int3 triangle_info_index = make_int3(triangle_infos[m_launch_index]);
			float3 sp_position = sp_positions[m_launch_index];
			float sp_radiuse = sp_radiuses[m_launch_index];

			/// We count normal
			float3 sp_normal = sp_normals[m_launch_index];/// = normalize( cross( t_vertex0 - t_vertex2, t_vertex1 - t_vertex0 ) );

			std::set<int> targetTriangleSet;
			float kernel_area = 0;
			getKernelArea(sp_position,
				sp_normal,
				sp_radiuse,
				triangle_info_index.x,
				triangle_info_index.y,
				triangle_info_index.z,
				targetTriangleSet,
				kernel_area,
				0
				);
			sp_areas[m_launch_index] = kernel_area;
		}
		float temp_pro = (j*1.0f)/buffer_height;
		if (temp_pro > current_pro)
		{
			std::cout << "\rtotal " << buffer_height << " " << std::setprecision(4) << temp_pro*100 << "% finished";
			current_pro += 0.1;
		}
	}
	m_sp_triangle_info_buffer->unmap();
	m_sp_position_buffer->unmap();
	m_sp_radius_buffer->unmap();
	m_sp_normal_buffer->unmap();
	m_sp_area_buffer->unmap();
}
void ProgressivePhotonScene::updateKernelArea()
{
	///return;
	Buffer output_buffer = m_context["rtpass_output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	output_buffer->getSize( buffer_width, buffer_height );

	int4* vertex_index4_array = reinterpret_cast<int4*>( m_sp_triangle_info_buffer->map() );
	float3* sp_positions = reinterpret_cast<float3*>( m_sp_position_buffer->map() );
	float3* sp_normals = reinterpret_cast<float3*>( m_sp_normal_buffer->map() );
	float* sp_radiuses = reinterpret_cast<float*>( m_sp_radius_buffer->map() );
	int* sp_valid = reinterpret_cast<int*>( m_sp_valid_buffer->map() );

	for (int j = 0;j < buffer_height;++ j)
	{
		unsigned int unsigned_j = j;
		for( unsigned int i = 0; i < buffer_width; ++i )
		{
			int m_launch_index = unsigned_j*buffer_width+i;
			if (vertex_index4_array[m_launch_index].x < 0)
				continue;

			int cur_vertex = vertex_index4_array[m_launch_index].w;
			/// 			if (cur_vertex != m_area_index_record[m_launch_index])
			/// 				sp_radiuses[m_launch_index] = -1;
			/// 			else
			/// 				sp_radiuses[m_launch_index] = 1.0f;

			int2 targetIndex = make_int2(i, unsigned_j);
			int sample_size = 4;
			for (int i_ptr = 0;i_ptr <= sample_size;i_ptr ++)
			{
				int i_p = (i_ptr%2)?(-i_ptr):i_ptr;
				targetIndex.x += i_p;
				if (targetIndex.x < 0 || targetIndex.x >= buffer_width)
					continue;
				for (int j_ptr = 0;j_ptr <= sample_size;j_ptr ++)
				{
					int j_p = (j_ptr%2)?(-j_ptr):j_ptr;
					targetIndex.y += j_p;
					if (targetIndex.y < 0 || targetIndex.y >= buffer_height)
						continue;
					m_launch_index = targetIndex.y*buffer_width + targetIndex.x;
					if (cur_vertex == m_area_index_record[m_launch_index])
					{
						vertex_index4_array[m_launch_index].x = targetIndex.x;
						vertex_index4_array[m_launch_index].y = targetIndex.y;
						j_ptr = i_ptr = 2 * sample_size;
					}
				}
			}
		}
	}

	m_sp_triangle_info_buffer->unmap();
	m_sp_position_buffer->unmap();
	m_sp_radius_buffer->unmap();
	m_sp_normal_buffer->unmap();
	m_sp_valid_buffer->unmap();
}
void ProgressivePhotonScene::calculateKernelAreaWithTriangle()
{
	Buffer output_buffer = m_context["rtpass_output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	output_buffer->getSize( buffer_width, buffer_height );

	int4* vertex_index4_array = reinterpret_cast<int4*>( m_sp_triangle_info_buffer->map() );
	float3* sp_positions = reinterpret_cast<float3*>( m_sp_position_buffer->map() );
	float3* sp_normals = reinterpret_cast<float3*>( m_sp_normal_buffer->map() );
	float* sp_radiuses = reinterpret_cast<float*>( m_sp_radius_buffer->map() );
	float* sp_areas = reinterpret_cast<float*>( m_sp_area_buffer->map() );
	int* sp_valid = reinterpret_cast<int*>( m_sp_valid_buffer->map() );
	float current_pro = 0.1;

	int enter_count = m_circle_count;
	const int total_enter_count = 
		///20;
		enter_count;

	if (m_circle_count < total_enter_count)
		m_finish_gen = true;

	///const int total_enter_count = buffer_height;
	for( int j = m_circle_count; j >= 0; --j )
	{
		unsigned int unsigned_j = j;
		if (m_circle_count <= enter_count - total_enter_count)
			break;
		for( unsigned int i = 0; i < buffer_width; ++i )
		{
			int m_launch_index = unsigned_j*buffer_width+i;
			int4 cur_vertex_index4 = vertex_index4_array[m_launch_index];
			float3 sp_position = sp_positions[m_launch_index];
			float sp_radiuse = sp_radiuses[m_launch_index];
			float stand_area = sp_radiuse * sp_radiuse * M_PI;

			/// We count normal
			float3 sp_normal = sp_normals[m_launch_index];
			float kernelArea = stand_area;
			bool myron_ppm_valid = 1;
			if (cur_vertex_index4.w > 0)
			{
				m_area_index_record[m_launch_index] = cur_vertex_index4.w;
				std::vector<int>& tempTriangleVector = loader->triangleIndexTablePtr->at(cur_vertex_index4.w);

				int cur_offset = 0;
				sp_valid[m_launch_index*Myron_Valid_Size] = -1;

				if ( cur_vertex_index4.x > 0)
					myron_ppm_valid = 0;

				for (std::vector<int>::iterator array_iterator = tempTriangleVector.begin();array_iterator != tempTriangleVector.end() && myron_ppm_valid;array_iterator ++)
				{
					float test_radius = sp_radiuse;
					float3 temp_normal;
					unsigned int *t_vindex = loader->model->triangles[*array_iterator].vindices;
					temp_normal = getNormal(loader->model->vertices + t_vindex[0]*3,
						loader->model->vertices + t_vindex[1]*3,
						loader->model->vertices + t_vindex[2]*3);

					/// Normal is not correct
					if (dot(temp_normal, sp_normal) < 0.0001f)
						continue;

					float m_area = 0;

					/// Method 2
					Matrix4X4 tempMatrix;
					float3 sp_U = cross( sp_normal, make_float3( 0.0f, 1.0f, 0.0f ) );
					if ( fabsf( sp_U.x) < 0.001f && fabsf( sp_U.y ) < 0.001f && fabsf( sp_U.z ) < 0.001f  )
						sp_U = cross( sp_normal, make_float3( 1.0f, 0.0f, 0.0f ) );
					tempMatrix.makeLookAt(sp_position, sp_position + sp_normal, sp_U);
					float3 p3_v1 = make_float3_from_float(loader->model->vertices + t_vindex[0]*3);
					float3 p3_v2 = make_float3_from_float(loader->model->vertices + t_vindex[1]*3);
					float3 p3_v3 = make_float3_from_float(loader->model->vertices + t_vindex[2]*3);
					float real_area = length( cross(p3_v1-p3_v2, p3_v2-p3_v3) );

					float2 p2_v1 = tempMatrix.transform3x2(p3_v1);
					float2 p2_v2 = tempMatrix.transform3x2(p3_v2);
					float2 p2_v3 = tempMatrix.transform3x2(p3_v3);
					float reference_area = length( cross(make_float3(p2_v1-p2_v2, 0), make_float3(p2_v2-p2_v3, 0) ) );

					getArea(make_float2(0), test_radius, p2_v1, p2_v2, p2_v3, &m_area);
					if (m_area > 0.00f)
					{
						kernelArea += m_area;
						sp_valid[m_launch_index*Myron_Valid_Size + cur_offset] = *array_iterator;
						cur_offset ++;
						if (cur_offset >= Myron_Valid_Size)
							break;
						sp_valid[m_launch_index*Myron_Valid_Size + cur_offset] = -1;
					}
				}
				if (kernelArea > stand_area)
				{
					int mydebug = 0;
					kernelArea = stand_area;
				}
				if (kernelArea < 0.1f * stand_area)
				{
					myron_ppm_valid = 0;
					kernelArea = stand_area;
				}
			}
			else
			{
				myron_ppm_valid = 1;
			}
			
			m_area_normal_record[m_launch_index] = sp_normal;
			m_area_record[m_launch_index] = kernelArea;
			vertex_index4_array[m_launch_index].x = myron_ppm_valid?i:-i;
			vertex_index4_array[m_launch_index].y = myron_ppm_valid?j:-j;
			sp_areas[m_launch_index] = kernelArea;
		}
		float temp_pro = 1.0f - (unsigned_j*1.0f)/buffer_height;
		if (temp_pro > current_pro)
		{
			std::cout << "\rtotal " << buffer_height << " " << std::setprecision(4) << temp_pro*100 << "% finished";
			current_pro += 0.1;
		}

		m_circle_count --;
	}
	m_sp_triangle_info_buffer->unmap();
	m_sp_position_buffer->unmap();
	m_sp_radius_buffer->unmap();
	m_sp_normal_buffer->unmap();
	m_sp_area_buffer->unmap();
	m_sp_valid_buffer->unmap();
}
void ProgressivePhotonScene::regenerate_area(RTsize buffer_width, RTsize buffer_height, char* info)
{
	if (m_area_index_record != NULL)
	{
		delete m_area_index_record;
		delete m_area_normal_record;
		delete m_area_record;
	}
	m_area_index_record = new int[buffer_width * buffer_height];
	m_area_normal_record = new float3[buffer_width * buffer_height];
	m_area_record = new float[buffer_width * buffer_height];
	memset(m_area_index_record, 0, buffer_width * buffer_height * sizeof(int));
	memset(m_area_normal_record, 0, buffer_width * buffer_height * sizeof(float3));
	memset(m_area_record, 0, buffer_width * buffer_height * sizeof(float));

	m_circle_count = buffer_height - 1;
	m_finish_gen = false;

	float* sp_areas = reinterpret_cast<float*>( m_sp_area_buffer->map() );
	for( unsigned int j = 0; j < buffer_height; ++j )
	{
		for( unsigned int i = 0; i < buffer_width; ++i )
		{
			int m_launch_index = j*buffer_width+i;
			sp_areas[m_launch_index] = -1.0;
		}
	}
	m_sp_area_buffer->unmap();
	std::cerr << info << " reget kernel area\n";
}


void ProgressivePhotonScene::trace( const RayGenCameraData& camera_data )
{
	double t0, t1;
	Buffer output_buffer = m_context["rtpass_output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	output_buffer->getSize( buffer_width, buffer_height );

	/// Print Images
	if (m_print_image)
	{
		char name1[256], name2[256];
		sprintf(name1, "%s/%s/%s/grab", sutilSamplesDir(), "progressivePhotonMap", "screengrab");
		sprintf(name2, "%s/%s/%s/%s", sutilSamplesDir(), "progressivePhotonMap", "screengrab", TestSceneNames[m_test_scene]);
		grab(buffer_width, buffer_height, name1, name2, m_frame_number);
		m_print_image = false;
	}

	/// Change Camera
	m_frame_number = m_camera_changed ? 0u : m_frame_number+1;
	m_context["frame_number"]->setFloat( static_cast<float>(m_frame_number) );
	if ( m_camera_changed ) 
	{
		m_camera_changed = false;
		m_context["rtpass_eye"]->setFloat( camera_data.eye );
		m_context["rtpass_U"]->setFloat( camera_data.U );
		m_context["rtpass_V"]->setFloat( camera_data.V );
		m_context["rtpass_W"]->setFloat( camera_data.W );
	}

	/// Trace viewing rays
	if (m_print_timings) std::cerr << "Starting RT pass ... ";
//	std::cerr.flush();
	sutilCurrentTime( &t0 );
	m_context->launch( EnterPointRayTrace,
		static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height) );
	sutilCurrentTime( &t1 );
	if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

	/// Trace photons
	buildGlobalPhotonMap();
	//buildCausticsPhotonMap();

	/// Shade view rays by gathering photons
	if (m_print_timings) std::cerr << "Starting gather pass   ... ";
	sutilCurrentTime( &t0 );
	
	m_context->launch( EnterPointGlobalGather,
		static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height) );
		
	/*m_context->launch( EnterPointCausticsGather,
		static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height) );*/
	sutilCurrentTime( &t1 );
	if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;
	
	/// Print Camera
	if (m_print_camera)
	{
		std::cerr << "\n\nthe camera eye: " << camera_data.eye.x << ","
			<< camera_data.eye.y << ","
			<< camera_data.eye.z << std::endl;
		float3 c_look_at = camera_data.eye + camera_data.W * 100.f;
		std::cerr << "the camera lookat: " << c_look_at.x << ","
			<< c_look_at.y << ","
			<< c_look_at.z << std::endl;

		std::cerr << "the light position:" << m_light.position.x << ","
			<< m_light.position.y << ","
			<< m_light.position.z << std::endl;
		std::cerr << "the light anchor:" << m_light.anchor.x << ","
			<< m_light.anchor.y << ","
			<< m_light.anchor.z << std::endl;
		m_print_camera = false;
	}

	/// Debug output
	if( m_display_debug_buffer ) {
		sutilCurrentTime( &t0 );
		float4* debug_data = reinterpret_cast<float4*>( m_debug_buffer->map() );
		Buffer hit_records = m_context["rtpass_output_buffer"]->getBuffer();
		HitRecord* hit_record_data = reinterpret_cast<HitRecord*>( hit_records->map() );
		float4 avg  = make_float4( 0.0f );
		float4 minv = make_float4( std::numeric_limits<float>::max() );
		float4 maxv = make_float4( 0.0f );
		float counter = 0.0f;
		for( unsigned int j = 0; j < buffer_height; ++j ) {
			for( unsigned int i = 0; i < buffer_width; ++i ) {
				if( hit_record_data[j*buffer_width+i].flags & PPM_HIT ) {
					float4 val = debug_data[j*buffer_width+i];
					avg += val;
					minv = fminf(minv, val);
					maxv = fmaxf(maxv, val);
					counter += 1.0f;
				}
			}
		}
		m_debug_buffer->unmap();
		hit_records->unmap();

		avg = avg / counter; 
		sutilCurrentTime( &t1 );
		if (m_print_timings) std::cerr << "State collection time ...           " << t1 - t0 << std::endl;
		std::cerr << "(min, max, average):"
			<< " loop iterations: ( "
			<< minv.x << ", "
			<< maxv.x << ", "
			<< avg.x << " )"
			<< " radius: ( "
			<< minv.y << ", "
			<< maxv.y << ", "
			<< avg.y << " )"
			<< " N: ( "
			<< minv.z << ", "
			<< maxv.z << ", "
			<< avg.z << " )"
			<< " M: ( "
			<< minv.w << ", "
			<< maxv.w << ", "
			<< avg.w << " )";
		std::cerr << ", total_iterations = " << m_frame_number + 1;
		std::cerr << std::endl;
	}
}


void ProgressivePhotonScene::doResize( unsigned int width, unsigned int height )
{
	RTsize oringinalWidth, oringinalHeight;
	m_context["sp_area_buffer"       ]->getBuffer()->getSize(oringinalWidth, oringinalHeight);

	/// display buffer resizing handled in base class
	m_context["rtpass_output_buffer"]->getBuffer()->setSize( width, height );
	m_context["output_buffer"       ]->getBuffer()->setSize( width, height );
	m_context["direct_buffer"]->getBuffer()->setSize( width, height );
	m_context["image_rnd_seeds"     ]->getBuffer()->setSize( width, height );
	m_context["debug_buffer"        ]->getBuffer()->setSize( width, height );
	m_context["primary_edge_buffer"   ]->getBuffer()->setSize( width, height );
	m_context["secondary_edge_buffer"   ]->getBuffer()->setSize( width, height );
	m_context["sp_triangle_info_buffer"   ]->getBuffer()->setSize( width, height );
	m_context["sp_position_buffer"   ]->getBuffer()->setSize( width, height );
	m_context["sp_radius_buffer"   ]->getBuffer()->setSize( width, height );
	m_context["sp_normal_buffer"   ]->getBuffer()->setSize( width, height );
	m_context["sp_area_buffer"   ]->getBuffer()->setSize( width, height );
	m_context["sp_valid_buffer"]->getBuffer()->setSize(width * height * Myron_Valid_Size);	

	m_context["eye_lose_angle"]->setFloat( sin(5.0f * 45.0f/height * M_PI / 180) );

	static bool firstVisit = true;
	if (oringinalHeight != height || oringinalWidth != width || firstVisit)
	{
		firstVisit = false;
		regenerate_area(width, height, "do resize");
	}
	else
	{
		m_camera_changed = false;
	}

	Buffer image_rnd_seeds = m_context["image_rnd_seeds"]->getBuffer();
	uint2* seeds = reinterpret_cast<uint2*>( image_rnd_seeds->map() );
	for ( unsigned int i = 0; i < width*height; ++i )  
		seeds[i] = random2u();
	image_rnd_seeds->unmap();
}

// 创建矩形GeometryInstance
GeometryInstance ProgressivePhotonScene::createParallelogram( const float3& anchor,
	const float3& offset1,
	const float3& offset2,
	const float3& color )
{
	Geometry parallelogram = m_context->createGeometry();
	parallelogram->setPrimitiveCount( 1u );
	parallelogram->setIntersectionProgram( m_pgram_intersection );
	parallelogram->setBoundingBoxProgram( m_pgram_bounding_box );

	float3 normal = normalize( cross( offset1, offset2 ) );
	float d       = dot( normal, anchor );
	float4 plane  = make_float4( normal, d );

	float3 v1 = offset1 / dot( offset1, offset1 );
	float3 v2 = offset2 / dot( offset2, offset2 );

	parallelogram["plane"]->setFloat( plane );
	parallelogram["anchor"]->setFloat( anchor );
	parallelogram["v1"]->setFloat( v1 );
	parallelogram["v2"]->setFloat( v2 );
	parallelogram["v1_l"]->setFloat( length( offset1 ) );
	parallelogram["v2_l"]->setFloat( length( offset2 ) );

	GeometryInstance gi = m_context->createGeometryInstance( parallelogram,
		&m_material,
		&m_material+1 );

	//设置材质
	gi[ "emissive" ]->setFloat( 0.0f, 0.0f, 0.0f );
	gi[ "phong_exp" ]->setFloat( 32.0f );
	gi[ "reflectivity" ]->setFloat( 0.3f, 0.3f, 0.3f );
	gi[ "illum" ]->setInt( 2 );

	gi["ambient_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.2f, 0.2f, 0.2f ) ) );
	gi["diffuse_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.8f, 0.8f, 0.8f ) ) );
	gi["specular_map"]->setTextureSampler( loadTexture( m_context, "", make_float3( 0.0f, 0.0f, 0.0f ) ) );

	gi["Kd"]->setFloat( color );
	gi["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
	gi["use_grid"]->setUint( 0u );
	gi["grid_color"]->setFloat( make_float3( 0.0f ) );
	gi["emitted"]->setFloat( 0.0f, 0.0f, 0.0f );
	return gi;
}


void ProgressivePhotonScene::loadObjGeometry( const std::string& filename, optix::Aabb& bbox, bool isUnitize )
{
	///// Set up material
	//m_material = m_context->createMaterial();
	//m_material->setClosestHitProgram( RayTypeRayTrace, m_context->createProgramFromPTXFile( ptxpath( "progressivePhotonMap", "ppm_rtpass.cu"),
	//	"rtpass_closest_hit") );
	//m_material->setClosestHitProgram( RayTypeGlobalPass,  m_context->createProgramFromPTXFile( ptxpath( "progressivePhotonMap", "ppm_ppass.cu"),
	//	"global_ppass_closest_hit") );
	//m_material->setClosestHitProgram( RayTypeCausticsPass,  m_context->createProgramFromPTXFile( ptxpath( "progressivePhotonMap", "ppm_ppass.cu"),
	//	"caustics_ppass_closest_hit") );
	//m_material->setAnyHitProgram(     RayTypeShadowRay,  m_context->createProgramFromPTXFile( ptxpath( "progressivePhotonMap", "ppm_gather.cu"),
	//	"gather_any_hit") );

	GeometryGroup geometry_group = m_context->createGeometryGroup();
	std::string full_path = std::string( sutilSamplesDir() ) + "/progressivePhotonMap/" + filename;
	loader = new PpmObjLoader( full_path, m_context, geometry_group);
	loader->useUnitization = isUnitize;
	int neighbor_size = 1;
	///if (m_test_scene == Conference_Scene)
	///	neighbor_size = 2;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Myron temp add here to add light
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	std::string ptx_path = ptxpath( "progressivePhotonMap", "parallelogram.cu" );
	m_pgram_bounding_box = m_context->createProgramFromPTXFile( ptx_path, "bounds" );
	m_pgram_intersection = m_context->createProgramFromPTXFile( ptx_path, "intersect" );

/// 	loader->m_light_instance = createParallelogram(m_light.anchor - m_light.v1 - m_light.v2, 
/// 		m_light.v1 * 2, m_light.v2 * 2, make_float3(1.0f));
/// 	loader->m_light_instance["emitted"]->setFloat( make_float3(1.0f) );
/// 	loader->m_light_instance[ "Alpha"  ]->setFloat( 1.0 );

	loader->load(MYRON_PPM, neighbor_size);
	bbox = loader->getSceneBBox();

	m_context["top_object"]->set( geometry_group );
	m_context["top_shadower"]->set( geometry_group );
}


void ProgressivePhotonScene::createCornellBoxGeometry()
{
	/// Set up material
	m_material = m_context->createMaterial();
	m_material->setClosestHitProgram( RayTypeRayTrace, m_context->createProgramFromPTXFile( ptxpath( "progressivePhotonMap", "ppm_rtpass.cu"),
		"rtpass_closest_hit_cornel") );
	m_material->setClosestHitProgram( RayTypeGlobalPass,  m_context->createProgramFromPTXFile( ptxpath( "progressivePhotonMap", "ppm_ppass.cu"),
		"ppass_closest_hit") );
	m_material->setAnyHitProgram(     RayTypeShadowRay,  m_context->createProgramFromPTXFile( ptxpath( "progressivePhotonMap", "ppm_gather.cu"),
		"gather_any_hit") );


	std::string ptx_path = ptxpath( "progressivePhotonMap", "parallelogram.cu" );
	m_pgram_bounding_box = m_context->createProgramFromPTXFile( ptx_path, "bounds" );
	m_pgram_intersection = m_context->createProgramFromPTXFile( ptx_path, "intersect" );


	/// create geometry instances
	std::vector<GeometryInstance> gis;

	const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
	const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
	const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
	const float3 black = make_float3( 0.0f, 0.0f, 0.0f );
	const float3 light = make_float3( 15.0f, 15.0f, 5.0f );

	/// Floor
	gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
		make_float3( 0.0f, 0.0f, 559.2f ),
		make_float3( 556.0f, 0.0f, 0.0f ),
		white ) );

	/// Ceiling
	gis.push_back( createParallelogram( make_float3( 0.0f, 548.8f, 0.0f ),
		make_float3( 556.0f, 0.0f, 0.0f ),
		make_float3( 0.0f, 0.0f, 559.2f ),
		white ) );

	/// Back wall
	gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 559.2f),
		make_float3( 0.0f, 548.8f, 0.0f),
		make_float3( 556.0f, 0.0f, 0.0f),
		white ) );

	/// Right wall
	gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
		make_float3( 0.0f, 548.8f, 0.0f ),
		make_float3( 0.0f, 0.0f, 559.2f ),
		green ) );

	/// Left wall
	gis.push_back( createParallelogram( make_float3( 556.0f, 0.0f, 0.0f ),
		make_float3( 0.0f, 0.0f, 559.2f ),
		make_float3( 0.0f, 548.8f, 0.0f ),
		red ) );

	/// Short block
	gis.push_back( createParallelogram( make_float3( 130.0f, 165.0f, 65.0f),
		make_float3( -48.0f, 0.0f, 160.0f),
		make_float3( 160.0f, 0.0f, 49.0f),
		white ) );
	gis.push_back( createParallelogram( make_float3( 290.0f, 0.0f, 114.0f),
		make_float3( 0.0f, 165.0f, 0.0f),
		make_float3( -50.0f, 0.0f, 158.0f),
		white ) );
	gis.push_back( createParallelogram( make_float3( 130.0f, 0.0f, 65.0f),
		make_float3( 0.0f, 165.0f, 0.0f),
		make_float3( 160.0f, 0.0f, 49.0f),
		white ) );
	gis.push_back( createParallelogram( make_float3( 82.0f, 0.0f, 225.0f),
		make_float3( 0.0f, 165.0f, 0.0f),
		make_float3( 48.0f, 0.0f, -160.0f),
		white ) );
	gis.push_back( createParallelogram( make_float3( 240.0f, 0.0f, 272.0f),
		make_float3( 0.0f, 165.0f, 0.0f),
		make_float3( -158.0f, 0.0f, -47.0f),
		white ) );

	/// Tall block
	gis.push_back( createParallelogram( make_float3( 423.0f, 330.0f, 247.0f),
		make_float3( -158.0f, 0.0f, 49.0f),
		make_float3( 49.0f, 0.0f, 159.0f),
		white ) );
	gis.push_back( createParallelogram( make_float3( 423.0f, 0.0f, 247.0f),
		make_float3( 0.0f, 330.0f, 0.0f),
		make_float3( 49.0f, 0.0f, 159.0f),
		white ) );
	gis.push_back( createParallelogram( make_float3( 472.0f, 0.0f, 406.0f),
		make_float3( 0.0f, 330.0f, 0.0f),
		make_float3( -158.0f, 0.0f, 50.0f),
		white ) );
	gis.push_back( createParallelogram( make_float3( 314.0f, 0.0f, 456.0f),
		make_float3( 0.0f, 330.0f, 0.0f),
		make_float3( -49.0f, 0.0f, -160.0f),
		white ) );
	gis.push_back( createParallelogram( make_float3( 265.0f, 0.0f, 296.0f),
		make_float3( 0.0f, 330.0f, 0.0f),
		make_float3( 158.0f, 0.0f, -49.0f),
		white ) );

	/// Light
	gis.push_back( createParallelogram( make_float3( 343.0f, 548.7f, 227.0f),
		make_float3( 0.0f, 0.0f, 105.0f),
		make_float3( -130.0f, 0.0f, 0.0f),
		black) );
	gis.back()["emitted"]->setFloat( light );


	/// Create geometry group
	GeometryGroup geometry_group = m_context->createGeometryGroup();
	geometry_group->setChildCount( static_cast<unsigned int>( gis.size() ) );
	for ( unsigned int i = 0; i < gis.size(); ++i )
		geometry_group->setChild( i, gis[i] );
	geometry_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

	m_context["top_object"]->set( geometry_group );
	m_context["top_shadower"]->set( geometry_group );
}


///-----------------------------------------------------------------------------
///
/// Main driver
///
///-----------------------------------------------------------------------------


void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
	std::cerr
		<< "Usage  : " << argv0 << " [options]\n"
		<< "App options:\n"
		<< "  -h  | --help                               Print this usage message\n"
		<< "  -c  | --cornell-box                        Display Cornell Box scene\n"
		<< "  -t  | --timeout <sec>                      Seconds before stopping rendering. Set to 0 for no stopping.\n"
#ifndef RELEASE_PUBLIC
		<< "  -pt | --print-timings                      Print timing information\n"
		<< " -ddb | --display-debug-buffer               Display the debug buffer information\n"
#endif
		<< std::endl;
	GLUTDisplay::printUsage();

	std::cerr
		<< "App keystrokes:\n"
		<< "  w Move light up\n"
		<< "  a Move light left\n"
		<< "  s Move light down\n"
		<< "  d Move light right\n"
		<< std::endl;

	if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
	GLUTDisplay::init( argc, argv );

	bool print_timings = true;
	bool display_debug_buffer = false;
	bool cornell_box = false;
	float timeout = -1.0f;

	for ( int i = 1; i < argc; ++i ) {
		std::string arg( argv[i] );
		if ( arg == "--help" || arg == "-h" ) {
			printUsageAndExit( argv[0] );
		} else if ( arg == "--print-timings" || arg == "-pt" ) {
			print_timings = true;
		} else if ( arg == "--display-debug-buffer" || arg == "-ddb" ) {
			display_debug_buffer = true;
		} else if ( arg == "--cornell-box" || arg == "-c" ) {
			cornell_box = true;
		} else if ( arg == "--timeout" || arg == "-t" ) {
			if(++i < argc) {
				timeout = static_cast<float>(atof(argv[i]));
			} else {
				std::cerr << "Missing argument to "<<arg<<"\n";
				printUsageAndExit(argv[0]);
			}
		} else {
			std::cerr << "Unknown option: '" << arg << "'\n";
			printUsageAndExit( argv[0] );
		}
	}

	if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

	try {
		ProgressivePhotonScene scene;
		if (print_timings) scene.printTimings();
		if (display_debug_buffer) scene.displayDebugBuffer();

		scene.setGatherMethod(ProgressivePhotonScene::Triangle_Inside_Method);

		GLUTDisplay::setProgressiveDrawingTimeout(timeout);
		GLUTDisplay::setUseSRGB(true);
		GLUTDisplay::run( "ProgressivePhotonScene", &scene, GLUTDisplay::CDProgressive );
	} catch( Exception& e ){
		sutilReportError( e.getErrorString().c_str() );
		exit(1);
	}

	return 0;
}
