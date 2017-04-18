#include "jfv_gpu.h"
#include "math.h"
#include <algorithm>
#include <iostream>

using namespace std;

int JFV_GPU::jfvTrianglePointNumber = 0;
int JFV_GPU::jfvBufferWidth = INIT_WINDOW_WIDTH; 
int	JFV_GPU::jfvBufferHeight = INIT_WINDOW_WIDTH;
IntegerPoint* JFV_GPU::jfvBufferA = NULL;
IntegerPoint* JFV_GPU::jfvBufferB = NULL;
bool JFV_GPU::jfvReadingBufferA = true;
vector<IntegerPoint> JFV_GPU::jfvSeedSamples;
vector<float> JFV_GPU::jfvSamplesArea;
IntegerPoint JFV_GPU::jfvTriangle[3];
std::map<int , IntegerPoint*> JFV_GPU::BufferAMap;
std::map<int , IntegerPoint*> JFV_GPU::BufferBMap;

// �ж����ص�x��y�Ƿ�����������
bool JFV_GPU::jfvIsInTriangle(int x, int y)
{
	IntegerPoint P(0, x, y);    
	
	IntegerPoint v0 = jfvTriangle[2] - jfvTriangle[0];
	IntegerPoint v1 = jfvTriangle[1] - jfvTriangle[0];
	IntegerPoint v2 = P - jfvTriangle[0];

	float dot00 = v0.Dot(v0) ;
	float dot01 = v0.Dot(v1) ;
	float dot02 = v0.Dot(v2) ;
	float dot11 = v1.Dot(v1) ;
	float dot12 = v1.Dot(v2) ;

	float inverDeno = 1 / (dot00 * dot11 - dot01 * dot01) ;

	float u = (dot11 * dot02 - dot01 * dot12) * inverDeno ;
	if (u < 0 || u > 1) // if u out of range, return directly
	{
		return false ;
	}

	float v = (dot00 * dot12 - dot01 * dot02) * inverDeno ;
	if (v < 0 || v > 1) // if v out of range, return directly
	{
		return false ;
	}

	return u + v <= 1 ;
}

// �ͷ���Դ
void JFV_GPU::jfvClearBuffers(){
	for(std::map<int, IntegerPoint*>::iterator it = BufferAMap.begin();it != BufferAMap.end(); ++it)
	{
		IntegerPoint* ptrVector = it->second;
		free(ptrVector);
	}
	for(std::map<int, IntegerPoint*>::iterator it = BufferBMap.begin();it != BufferBMap.end(); ++it)
	{
		IntegerPoint* ptrVector = it->second;
		free(ptrVector);
	}
}

// ���༶buffer����Ϊmap�������ظ�ʹ��
void JFV_GPU::jfvMallocBuffers( void ) {		
	if (BufferAMap.find(jfvBufferWidth) == BufferAMap.end())
	{
		BufferAMap[jfvBufferWidth] = (IntegerPoint*)malloc( sizeof( IntegerPoint ) * jfvBufferWidth * jfvBufferWidth );
	}
	if (BufferBMap.find(jfvBufferWidth) == BufferBMap.end())
	{
		BufferBMap[jfvBufferWidth] = (IntegerPoint*)malloc( sizeof( IntegerPoint ) * jfvBufferWidth * jfvBufferWidth );
	}
	jfvBufferA = BufferAMap[jfvBufferWidth];
	jfvBufferB = BufferBMap[jfvBufferWidth];
}
// ����ά������ת��Ϊ��άʱ�����˻�Ϊֱ�ߣ�ѡȡͶ��ķ���
int jfvJudgeNormal(PointFloat3 _trianglePoints[3])
{
	PointFloat3 p01 = _trianglePoints[0] - _trianglePoints[1];
	PointFloat3 p12 = _trianglePoints[1] - _trianglePoints[2];
	
	PointFloat3 crossV = p01.Cross(p12);
	crossV.normalise();

	PointFloat3 dirX(1.0f, 0.0f, 0.0f);
	float dotV = crossV.Dot(dirX);
	// �����������x�ᴹֱ����ô����0
	if (dotV > 0.999999 || dotV < -0.999999)
		return 0;

	PointFloat3 dirY(0.0f, 1.0f, 0.0f);
	dotV = crossV.Dot(dirY);
	// �����������y�ᴹֱ����ô����1
	if (dotV > 0.999999 || dotV < -0.999999)
		return 1;

	// ���򷵻�2
	return 2;
}

// ����ά�ռ両����ת��Ϊbuffer���ص�
IntegerPoint JFV_GPU::jfv2D2Pixel(int mindex, float fx, float fy, float fxMin, float fxMax, float fyMin, float fyMax)
{
	// mindex ��ǰ���ӵ������ţ�fx��fy�Ǹ��������꣬fxMin, fxMax, fyMin, fyMax�ֱ��Ǹ���������ı߽�
	float tX = (fx - fxMin)/(fxMax - fxMin) * jfvBufferWidth;
	float tY = (fy - fyMin)/(fyMax - fyMin) * jfvBufferHeight;
	// תΪint�͵�
	int x = (int)tX;
	int y = (int)tY;
	if (x == jfvBufferWidth) x--;
	if (y == jfvBufferWidth) y--;
	IntegerPoint tempP =  IntegerPoint(mindex, x, y);
	return  tempP;
}

// ��ȡ�����������߽�
void getMaxMin(PointFloat3 _trianglePoints[3], float &fxMax, float &fxMin, int indexA)
{
	if (_trianglePoints[0].points[indexA] < _trianglePoints[1].points[indexA])
	{
		fxMin = _trianglePoints[0].points[indexA];
		fxMax = _trianglePoints[1].points[indexA];
	}
	else
	{
		fxMin = _trianglePoints[1].points[indexA];
		fxMax = _trianglePoints[0].points[indexA];
	}

	if (_trianglePoints[2].points[indexA] < fxMin)
	{
		fxMin = _trianglePoints[2].points[indexA];
	}
	else if (_trianglePoints[2].points[indexA] > fxMax)
	{
		fxMax = _trianglePoints[2].points[indexA];
	}

}

// ΪJFV_GPU���������涥��͹�������
void JFV_GPU::jfvSetTriangle(PointFloat3 _trianglePoints[3], vector<PointFloat3> _samplePoints, bool insertAssert)
{
	// �����Ч������ϵ
	int judgeValue = jfvJudgeNormal(_trianglePoints);
	int indexA = (judgeValue+1)%3;
	int indexB = (judgeValue+2)%3;

	// ��ȡ�����α߽�
	float fxMin, fxMax, fyMin, fyMax;
	getMaxMin(_trianglePoints, fxMax, fxMin, indexA);
	getMaxMin(_trianglePoints, fyMax, fyMin, indexB);

	// ��������ת��Ϊ��������ϵ
	for (unsigned int i = 0;i < 3;i ++)
		jfvTriangle[i] = jfv2D2Pixel(0, _trianglePoints[i].points[indexA], _trianglePoints[i].points[indexB], fxMin, fxMax, fyMin, fyMax);
	jfvTrianglePointNumber = 3;

	// ��������ת��Ϊ��������ϵ
	jfvSeedSamples.clear();
	bool *boolBuffer = (bool*)malloc( sizeof( bool ) * jfvBufferWidth * jfvBufferHeight );
	memset(boolBuffer, 0, sizeof( bool ) * jfvBufferWidth * jfvBufferHeight);

	
	int *colorBuffer = (int*)malloc( sizeof( int ) * jfvBufferWidth * jfvBufferHeight );
	memset(colorBuffer, -1, sizeof( int ) * jfvBufferWidth * jfvBufferHeight);
	// ��ǰbuffer�Ĵ�С�ѱ�������Ѱ��������ظ�㣬ǿ��д��
	if (insertAssert == true)
	{
		for (unsigned int i = 0;i < _samplePoints.size();i ++)
		{
			IntegerPoint tPoint = jfv2D2Pixel(i, _samplePoints[i].points[indexA], _samplePoints[i].points[indexB], fxMin, fxMax, fyMin, fyMax);
			int tmindex = tPoint.y * jfvBufferWidth + tPoint.x;
			// ���֮ǰ����δ��Ⱦɫ����ô����Ⱦɫ
			if (boolBuffer[tmindex] == false)
			{
				jfvSeedSamples.push_back(tPoint);
				boolBuffer[tmindex] = true;
				colorBuffer[tmindex] = i;
			}
			else
			{
				int tm_i, tm_j, mmindex, hasgetNumber = 0;
				// ��0��0��ʼ��Ѱ��δ��Ⱦɫ�����ص���Ϊ�����
				for (tm_j = 0;(tm_j < jfvBufferHeight);tm_j ++)
				{
					for (tm_i = 0;(tm_i < jfvBufferWidth);tm_i ++)
					{
						mmindex = tm_j * jfvBufferWidth + tm_i;
						if (boolBuffer[mmindex] == false)
						{
							hasgetNumber = 1;
							boolBuffer[mmindex] = true;
							break;
						}
					}
					if (hasgetNumber==1)
						break;
				}
				tPoint.x = tm_i; tPoint.y = tm_j;
				// ȷ������û�����
				if (tPoint.x == jfvBufferWidth || tPoint.y == jfvBufferHeight)
					assert(0);
				jfvSeedSamples.push_back(tPoint);
			}
		}
		free(boolBuffer);
		free(colorBuffer);
		return;
	}
	// �����ǰbuffer�Ĵ�С�ɱ䣬����¼���ظ�����
	int a = 0;
	for (unsigned int i = 0;i < _samplePoints.size();i ++)
	{
		IntegerPoint tPoint = jfv2D2Pixel(i, _samplePoints[i].points[indexA], _samplePoints[i].points[indexB], fxMin, fxMax, fyMin, fyMax);
		int tmindex = tPoint.y * jfvBufferWidth + tPoint.x;
		if (boolBuffer[tmindex] == false)
		{
			jfvSeedSamples.push_back(tPoint);
			boolBuffer[tmindex] = true;
			colorBuffer[tmindex] = i;
		}
	}
	free(boolBuffer);
	free(colorBuffer);
}

// Jump Flooding Algorithm
JFV_ERROR JFV_GPU::jfvExecuteJumpFlooding( int & triangleCount ) {

	if (jfvTrianglePointNumber != 3)
		return JFV_ERROR_NO_TRIANGLE_POINTS;
	// No seeds will just give us a black screen :P
	if( jfvSeedSamples.size() < 1 ) {
		printf( "Please create at least 1 seed.\n" );
		return JFV_ERROR_NO_SAMPLE_POINTS;
	}

	//printf( "Executing the Jump Flooding algorithm...\n" );
	std::vector< float2 > SiteVec ;   SiteVec.clear();
    std::vector< int >    SeedVec( jfvBufferWidth * jfvBufferWidth , - 1 ) ;
	std::vector< uchar3 > RandomColorVec ;  RandomColorVec.clear();

	for( unsigned int i = 0; i < jfvSeedSamples.size(); ++i ) {
		IntegerPoint& p = jfvSeedSamples[i];
		int tIndexInBuffer = ( p.y * jfvBufferWidth ) + p.x;
		if (tIndexInBuffer >= jfvBufferHeight * jfvBufferWidth) {
			cout << p.x << " " << p.y << endl;
			return JFV_ERROR_SAMPLE_POINTS_OUTSIDE;
		}
        SiteVec.push_back( make_float2( p.x + 0.5f , p.y + 0.5f ) ) ;
		SeedVec[p.x + p.y * jfvBufferWidth] = i;
        RandomColorVec.push_back( make_uchar3( static_cast< unsigned char >( static_cast< float >( rand() ) / RAND_MAX * 255.0f ) ,
                                               static_cast< unsigned char >( static_cast< float >( rand() ) / RAND_MAX * 255.0f ) ,
                                               static_cast< unsigned char >( static_cast< float >( rand() ) / RAND_MAX * 255.0f ) ) ) ;
	}
	
    //
    size_t SiteSize = jfvSeedSamples.size() * sizeof( float2 ) ;

    float2 * SiteArray = NULL ;
    cudaMalloc( & SiteArray , SiteSize ) ;
    cudaMemcpy( SiteArray , & SiteVec[0] , SiteSize , cudaMemcpyHostToDevice ) ;

    //
    size_t BufferSize = jfvBufferWidth * jfvBufferWidth * sizeof( int ) ;

    int * Ping = NULL , * Pong = NULL ;
    cudaMalloc( & Ping , BufferSize ) , cudaMemcpy( Ping , & SeedVec[0] , BufferSize , cudaMemcpyHostToDevice ) ;
    cudaMalloc( & Pong , BufferSize ) , cudaMemcpy( Pong , Ping , BufferSize , cudaMemcpyDeviceToDevice ) ;

    //
    int * Mutex = NULL ;
    cudaMalloc( & Mutex , sizeof( int ) ) , cudaMemset( Mutex , - 1 , sizeof( int ) ) ;

    //
    //
    cudaDeviceProp CudaDeviceProperty ;
    cudaGetDeviceProperties( & CudaDeviceProperty , 0 ) ;
	

    for ( int k = jfvBufferWidth / 2 ; k > 0 ; k = k >> 1 )
    {
        vonoi( jfvBufferWidth , jfvBufferWidth , SiteArray , Ping , Pong , k , Mutex) ;
        cudaDeviceSynchronize() ;
	
		cudaMemcpy( & SeedVec[0] , Pong , BufferSize , cudaMemcpyDeviceToHost ) ;

        cudaMemcpy( Ping , Pong , BufferSize , cudaMemcpyDeviceToDevice ) ;
        std::swap( Ping , Pong ) ;
    }

    cudaMemcpy( & SeedVec[0] , Pong , BufferSize , cudaMemcpyDeviceToHost ) ;

    std::vector< int >  ClearVec( jfvBufferWidth * jfvBufferWidth , - 1 ) ;
	cudaMemcpy( Ping , & ClearVec[0] , BufferSize , cudaMemcpyHostToDevice ) ;
	cudaMemcpy( Pong , & ClearVec[0] , BufferSize , cudaMemcpyHostToDevice ) ;
    //
    cudaFree( SiteArray ) ;
    cudaFree( Ping ) ;
    cudaFree( Pong ) ;
    cudaFree( Mutex ) ;
	
	
	jfvSamplesArea.clear();
	for (int i = 0; i < jfvSeedSamples.size(); i++)
		jfvSamplesArea.push_back(0);

	for ( int y = 0 ; y < jfvBufferWidth ; ++ y )
	{
		for ( int x = 0 ; x < jfvBufferWidth ; ++x )
		{
			int Seed = SeedVec[x + y * jfvBufferWidth] ;
			if ( Seed != - 1 )
			{
				if (jfvIsInTriangle(x, y))
				{
					jfvSamplesArea[Seed] ++;
					triangleCount ++;
				}
			}
		}
	}

	
	/*FILE * Output = fopen( "1.ppm" , "wb" ) ;
	fprintf( Output , "P6\n%d %d\n255\n" , jfvBufferWidth , jfvBufferWidth ) ;
	std::vector< uchar3 > Pixels( jfvBufferWidth * jfvBufferWidth ) ;
		
	for ( int y = 0 ; y < jfvBufferWidth ; ++ y )
	{
		for ( int x = 0 ; x < jfvBufferWidth ; ++x )
		{
				
			int Seed = SeedVec[x + y * jfvBufferWidth] ;
			if (Seed != -1) {
				Pixels[x + y * jfvBufferWidth] = RandomColorVec[Seed] ;
			}
		}
	}
		
	for( std::vector< float2 >::const_iterator itr = SiteVec.begin() ; itr != SiteVec.end() ; ++ itr )
	{
		const int x = static_cast< int >( floorf( itr->x ) ) ;
		const int y = static_cast< int >( floorf( itr->y ) ) ;
		Pixels[x + y * jfvBufferWidth] = make_uchar3( 255 , 0 , 0 ) ;
	}
	fwrite( & Pixels[0].x , 3 , Pixels.size() , Output ) ;
	fclose( Output ) ;*/

	return JFV_NO_ERROR;
}

// ��¼ÿ�����Ӳ�����Ĳ������
void JFV_GPU::jfvCountArea(float totalArea, int triangleCount)
{
	int sampleSize = jfvSeedSamples.size();
	jfvSamplesArea.clear();
	for (int i = 0;i < sampleSize;i ++)
		jfvSamplesArea.push_back(totalArea/sampleSize);
	return;
	// ���������һ�����ӣ������������������
	if (sampleSize == 1)
	{
		jfvSamplesArea[0] = totalArea;
		return;
	}
	// ��������ΰ��������ش���0������������Ŀ��Ӧ�����
	if (triangleCount > 0)
	{
		for (int i = 0;i < sampleSize;i ++)
		{
			jfvSamplesArea[i] *= (totalArea/triangleCount);
			if (jfvSamplesArea[i] == 0)
			{
				assert(1);
			}
		}
	}
}

//void JFV_GPU::test() {
//	int NumCudaDevice = 0 ;
//    cudaGetDeviceCount( & NumCudaDevice ) ;
//}