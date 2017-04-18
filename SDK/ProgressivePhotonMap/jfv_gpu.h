#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <assert.h>

using namespace std;
// Initial window dimensions
#define INIT_WINDOW_WIDTH  8

extern "C" {
	//void vonoiOnDevice( int SizeX , int SizeY , const float2 * SiteArray , const int * Ping , int * Pong , int k , int * Mutex);
}

void vonoi( int SizeX , int SizeY , const float2 * SiteArray , const int * Ping , int * Pong , int k , int * Mutex) {
//	vonoiOnDevice( SizeX , SizeY , SiteArray , Ping , Pong , k , Mutex);
}

// Represents a point with (x,y) coordinates
class IntegerPoint {
public:
	int mIndex;
	int x,y;
	IntegerPoint(){}
	IntegerPoint(int findex, int fx, int fy)
		:mIndex(findex), x(fx), y(fy){}
	// Subtract
	IntegerPoint operator - (const IntegerPoint& v) const
	{
		return IntegerPoint(0, x - v.x, y - v.y) ;
	}
	// Dot product
	const float Dot(const IntegerPoint& v) const
	{
		return float(x * v.x + y * v.y);
	}
};

// Represents a point with (x,y) coordinates
struct PointFloat3
{
	float points[3];
	PointFloat3(){}
	PointFloat3(float fx, float fy, float fz)
	{
		points[0] = fx;
		points[1] = fy;
		points[2] = fz;
	}
	PointFloat3 operator - (const PointFloat3& v) const
	{
		return PointFloat3(points[0] - v.points[0], points[1] - v.points[1], points[2] - v.points[2]) ;
	}
	// Dot product
	const float Dot(const PointFloat3& v) const
	{
		return points[0] * v.points[0] + points[1] * v.points[1] + points[2] * v.points[2];
	}
	const float Length() const
	{
		return sqrt(Dot(PointFloat3(points[0], points[1], points[2])));
	}
	// Cross product
	PointFloat3 Cross(const PointFloat3& v) const
	{
		return PointFloat3(
			points[1] * v.points[2] - points[2] * v.points[1],
			points[2] * v.points[0] - points[0] * v.points[2],
			points[0] * v.points[1] - points[1] * v.points[0] ) ;
	}
	void normalise()
	{
		float len = sqrt(Dot(*this));
		points[0] /= len;
		points[1] /= len;
		points[2] /= len;
	}
};
//typedef PointFloat3

enum JFV_ERROR
{
	JFV_NO_ERROR = 0,
	JFV_ERROR_TOO_SMALL_BUFFER = 1,
	JFV_ERROR_NO_SAMPLE_POINTS = 2,
	JFV_ERROR_SAMPLE_POINTS_OUTSIDE = 3,
	JFV_ERROR_NO_BUFFER_MEMORY = 4,
	JFV_ERROR_NO_TRIANGLE_POINTS = 5,
};

class JFV_GPU
{
	// Buffers
	static IntegerPoint* jfvBufferA;
	static IntegerPoint* jfvBufferB;
	
	// We use this boolean to know which buffer we are reading from
	static bool jfvReadingBufferA;

	static IntegerPoint jfv2D2Pixel(int mindex, float fx, float fy, float fxMin, float fxMax, float fyMin, float fyMax);
public:
	static std::map<int , IntegerPoint*> BufferAMap;
	static std::map<int , IntegerPoint*> BufferBMap;
	static IntegerPoint jfvTriangle[3];
	static int jfvTrianglePointNumber;

	static bool jfvIsInTriangle(int, int);
	static void jfvClearBuffers();
	static void jfvMallocBuffers( void );
	static JFV_ERROR jfvExecuteJumpFlooding( int & );
	static void jfvSetTriangle(PointFloat3[3], vector<PointFloat3>, bool insertAssert);
	static IntegerPoint *jfvGetDisplayBuffer(void){
		return jfvReadingBufferA == true ? jfvBufferA : jfvBufferB;
	}
	static void jfvCountArea(float, int);
	// Buffer dimensions
	static int jfvBufferWidth;
	static int jfvBufferHeight;
	static vector<IntegerPoint> jfvSeedSamples;
	static vector<float> jfvSamplesArea;
	static void test();
};
