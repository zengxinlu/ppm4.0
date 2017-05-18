#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <GLUTDisplay.h>

using namespace optix;
#include <yaml-cpp/yaml.h>
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

static const float3 m_light_target = make_float3(0, 0, 0);

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

static char* TestSceneNames[] = {
	"Cornel_Box_Scene",
	"Wedding_Ring_Scene",
	"Small_Room_Scene",
	"Conference_Scene",
	"Clock_Scene",
	"Sponza_Scene",
	"Box_Scene",
	"Sibenik_Scene",
	"Torus_Scene",
	"EChess_Scene",
	"Diamond_Scene"
};
class ProgressivePhotonScene : public SampleScene
{
public:
	std::string m_model_file;
	std::string m_model;
	ProgressivePhotonScene() : SampleScene()
		, m_frame_number( 0 )
		, m_display_debug_buffer( false )
		, m_print_timings ( false )
		, m_test_scene( Box_Scene )
		, m_light_phi( 2.19f )
		, m_light_theta( 1.15f )
		, m_split_choice(LongestDim)
	{}
	void selectScene(std::string model, std::string modelNum)
	{
		if (model == "box")				setTestScene(ProgressivePhotonScene::Box_Scene);
		if (model == "torus")			setTestScene(ProgressivePhotonScene::Torus_Scene);
		if (model == "cornellbox") 		setTestScene(ProgressivePhotonScene::Cornel_Box_Scene);
		if (model == "sibenik")			setTestScene(ProgressivePhotonScene::Sibenik_Scene);
		if (model == "wedding_ring") 	setTestScene(ProgressivePhotonScene::Wedding_Ring_Scene);
		if (model == "conference")		setTestScene(ProgressivePhotonScene::Conference_Scene);
		if (model == "sponza")			setTestScene(ProgressivePhotonScene::Sponza_Scene);
		if (model == "smallroom")		setTestScene(ProgressivePhotonScene::Small_Room_Scene);
		if (model == "clocks")			setTestScene(ProgressivePhotonScene::Clock_Scene);
		if (model == "echess")			setTestScene(ProgressivePhotonScene::EChess_Scene);
		if (model == "diamond")			setTestScene(ProgressivePhotonScene::Diamond_Scene);
		m_model = model;
		m_model_file = std::string(sutil::samplesDir()) + "/progressivePhotonMap/scenes/" + model + "/" + model + modelNum + ".yaml";
	}	

	/// From SampleScene
	void	initAssistBuffer();
	void	initScene( InitialCameraData& camera_data );
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
	}
	void setGatherMethod(int gatherMethod) { m_gather_method = gatherMethod; }
	void printTimings()       { m_print_timings = true; }
	void displayDebugBuffer() { m_display_debug_buffer = true; }

	void collectionPhotons(std::string filename, int frameNum);
	bool useCollectionPhotons = false;
	bool m_collect_photon = false;
	int  collectPhotonsFrame = 1000;

	enum GatherMethod{
		Cornel_Box_Method,
		Triangle_Inside_Method,
		Triangle_Vertical_Method,
		Triangle_Extend_Method,
		Triangle_Combine_Method
	};
	enum TestScene {
		Cornel_Box_Scene = 0,
		Wedding_Ring_Scene,
		Small_Room_Scene,
		Conference_Scene,
		Clock_Scene,
		Sponza_Scene,
		Box_Scene,
		Sibenik_Scene,
		Torus_Scene,
		EChess_Scene,
		Diamond_Scene,
	};
private:
	void loadScene(InitialCameraData& camera_data);
	void buildGlobalPhotonMap();
	void buildCausticsPhotonMap();
	void setFloatIn(std::vector<PhotonRecord*>& ptrVector, int ttindex, float mt_area);
	void loadObjGeometry( const std::string& filename, optix::Aabb& bbox, bool isUnitize);
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
		EnterPointInitRadius,
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
	PPMLight	  tmpLight;
	int			  m_circle_count;
	bool		  m_finish_gen;
	bool		  m_print_image;
	bool		  m_print_camera;
	int           m_cuda_device;

	const static unsigned int WIDTH;
	const static unsigned int HEIGHT;
	const static unsigned int MAX_PHOTON_COUNT;
	const static unsigned int MAX_PHOTON_DEPTH;
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
	void initEnterPointInitRadius();
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
	string cuda_initRadius_cu;
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

const unsigned int ProgressivePhotonScene::MAX_PHOTON_COUNT = 20u;
const unsigned int ProgressivePhotonScene::MAX_PHOTON_DEPTH = 8u;
//const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH = 256u;
//const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT = 256u;
//const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH = 1024u;
//const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT = 1024u;
const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH = 256u;
const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT = 256u;
const unsigned int ProgressivePhotonScene::NUM_PHOTONS = (ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH *
	ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT *
	ProgressivePhotonScene::MAX_PHOTON_COUNT);

void ProgressivePhotonScene::printFile(char *msg)
{	
	char name[256];
	sprintf(name, "%s/%s/%s/grab.txt", sutil::samplesDir(), "progressivePhotonMap", "screengrab");
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

void photonPrint(PhotonRecord p) {
	printf("%.6f %.6f %.6f\n", p.position.x, p.position.y, p.position.z);
	printf("%.6f %.6f %.6f\n", p.normal.x, p.normal.y, p.normal.z);
	printf("%.6f %.6f %.6f\n", p.ray_dir.x, p.ray_dir.y, p.ray_dir.z);
	printf("%.6f %.6f %.6f\n", p.energy.x, p.energy.y, p.energy.z);
}

void updateLight(PPMLight &light, float3 dis_float3)
{
	light.position += dis_float3;

	light.direction = normalize( m_light_target  - light.position );
	light.anchor = light.position + light.direction * 0.0f;

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
		m_camera_changed = true;
		break;
	case 'p':
		m_print_image = true;
		std::cerr << "we print an image" << std::endl;
		break;
	case ']':
		m_print_camera = true;
		break;
	case '.':
		RTsize buffer_width, buffer_height;
		m_context["rtpass_output_buffer"]->getBuffer()->getSize( buffer_width, buffer_height );
		regenerate_area(buffer_width, buffer_height, "press '.'");
		break;
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

/*
	init model divide 3 step
	1. init camera_data
	2. init light
	3. init radius
*/
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

void ProgressivePhotonScene::loadScene(InitialCameraData& camera_data) {
	std::cerr << m_model_file << std::endl;
	YAML::Node modelConfig = YAML::LoadFile(m_model_file);
	YAML::Node cameraData = modelConfig["camera_data"];
	vector<double> eye = cameraData["eye"].as<vector<double> >();
	vector<double> lookat = cameraData["lookat"].as<vector<double> >();
	vector<double> up = cameraData["up"].as<vector<double> >();
	double vfov = cameraData["vfov"].as<double>();
	camera_data = InitialCameraData( make_float3( eye[0], eye[1], eye[2] ), /// eye
		make_float3( lookat[0], lookat[1], lookat[2] ),      /// lookat
		make_float3( up[0], up[1], up[2] ),     /// up
		vfov );                              /// vfov

	YAML::Node lightData = modelConfig["light_data"];
	m_light.is_area_light = lightData["is_area_light"].as<int>(); 
	if (lightData["direction"].IsNull()) {
		vector<double> target = lightData["target"].as<vector<double> >();
		vector<double> position = lightData["position"].as<vector<double> >();
		float3 tmpTarget = make_float3(target[0], target[1], target[2]);
		m_light.position = make_float3(position[0], position[1], position[2]);
		m_light.direction = normalize(tmpTarget - m_light.position);
		mprintf(m_light.direction);
	} else {
		vector<double> position = lightData["position"].as<vector<double> >();
		vector<double> direction = lightData["direction"].as<vector<double> >();
		m_light.position = make_float3(position[0], position[1], position[2]);
		m_light.direction = make_float3(direction[0], direction[1], direction[2]);
	}
	if (m_light.is_area_light) {
		m_light.anchor = m_light.position + m_light.direction * 0.0f;
		m_light.v1 = make_float3(1.0f, 0.f, 0.0f) * lightData["v1"].as<double>();
		m_light.v2 = make_float3(0.0f, 0.f, 1.0f) * lightData["v2"].as<double>();
		float3 m_light_t_normal;
		m_light_t_normal = cross(m_light.v1, m_light.direction);
		m_light.v1 = cross(m_light.direction, m_light_t_normal);
		m_light_t_normal = cross(m_light.v2, m_light.direction);
		m_light.v2 = cross(m_light.direction, m_light_t_normal);
	}
	m_light.radius = lightData["radius"].as<double>();
	vector<double> power = lightData["power"].as<vector<double> >();
	m_light.power = make_float3(power[0], power[1], power[2]);

	m_context["light"]->setUserData( sizeof(PPMLight), &m_light );

	float default_radius2 = modelConfig["default_radius"].as<double>();
	m_context["rtpass_default_radius2"]->setFloat( default_radius2);
	m_context["max_radius2"]->setFloat(default_radius2);
	vector<float> blend_mothod = modelConfig["blend_mothod"].as<vector<float> >();
	m_context["direct_ratio"]->setFloat(blend_mothod[0]);
	m_context["indirect_ratio"]->setFloat(blend_mothod[1]);
	optix::Aabb aabb;	
	loadObjGeometry(modelConfig["filename"].as<std::string>(), aabb, true);

}
void ProgressivePhotonScene::initGlobal() {

	cuda_initRadius_cu = "ppm_initRadius.cu";
	cuda_gather_cu = "ppm_gather.cu";
	cuda_ppass_cu = "ppm_ppass.cu";
	cuda_rtpass_cu = "ppm_rtpass.cu";
	triangle_mesh_cu = "triangle_mesh.cu";
	projectName = "progressivePhotonMap";

	m_print_image = false;
	m_print_camera = false;
	m_print_timings = false;
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
	m_context->setStackSize( 640 );

	m_context["max_depth"]->setUint(MAX_PHOTON_DEPTH);
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

	Buffer camera_buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT3, WIDTH, HEIGHT);
	m_context["camera_buffer"]->set(camera_buffer);
	uint3* camera_seeds = reinterpret_cast<uint3*>(camera_buffer->map());
	for (unsigned int i = 0; i < WIDTH*HEIGHT; ++i)
	{
		camera_seeds[i].x = 0;
		camera_seeds[i].y = 0;
		camera_seeds[i].z = 0;
	}
	camera_buffer->unmap();

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
	Program gather_program = m_context->createProgramFromPTXFile( gather_ptx_path, gather_program_name );
	m_context->setRayGenerationProgram( EnterPointGlobalGather, gather_program );
	Program exception_program = m_context->createProgramFromPTXFile( gather_ptx_path, "gather_exception" );
	m_context->setExceptionProgram( EnterPointGlobalGather, exception_program );
}
void ProgressivePhotonScene::initEnterPointCausticsGather() {
	/// Gather phase
	std::string gather_ptx_path = ptxpath( projectName, cuda_gather_cu );
	std::string gather_program_name = "causticsDensity";
	Program gather_program = m_context->createProgramFromPTXFile( gather_ptx_path, gather_program_name );
	m_context->setRayGenerationProgram( EnterPointCausticsGather, gather_program );
	Program exception_program = m_context->createProgramFromPTXFile( gather_ptx_path, "gather_exception" );
	m_context->setExceptionProgram( EnterPointCausticsGather, exception_program );
}
void ProgressivePhotonScene::initGeometryInstances(InitialCameraData& camera_data) {
	loadScene(camera_data);

	m_context["ambient_light"]->setFloat( 0.1f, 0.1f, 0.1f);
	std::string full_path = std::string( sutil::samplesDir() ) + "/tutorial/data/CedarCity.hdr";
	///const float3 default_color = make_float3( 0.8f, 0.88f, 0.97f );
	const float3 default_color = make_float3( 0.0f );
	//m_context["envmap"]->setTextureSampler( loadTexture( m_context, full_path, default_color) );
	m_context["envmap"]->setTextureSampler( loadTexture( m_context, "", default_color) );
}
void ProgressivePhotonScene::initEnterPointInitRadius() {
	/// Gather phase
	std::string init_ptx_path = ptxpath(projectName, cuda_initRadius_cu);
	std::string init_program_name = "initRadius";
	Program init_program = m_context->createProgramFromPTXFile(init_ptx_path, init_program_name);
	m_context->setRayGenerationProgram(EnterPointInitRadius, init_program);
	Program exception_program = m_context->createProgramFromPTXFile(init_ptx_path, "init_exception");
	m_context->setExceptionProgram(EnterPointInitRadius, exception_program);
}

void ProgressivePhotonScene::initScene( InitialCameraData& camera_data )
{
	cout << "Begin to init SPPM context...\n";

	initGlobal();
	initEnterPointRayTrace(camera_data);
	initEnterPointGlobalPhotonTrace();
	initEnterPointCausticsPhotonTrace();
	initEnterPointGlobalGather();
	initEnterPointCausticsGather();
	initEnterPointInitRadius();
	initGeometryInstances(camera_data);
	m_context->validate();
	m_context->compile();

	cout << "Context init finished\n" ;

	if (m_collect_photon) {
		std::cerr << collectPhotonsFrame << std::endl;
		collectionPhotons(m_model, collectPhotonsFrame);
	}
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

void ProgressivePhotonScene::buildGlobalPhotonMap()
{
	double t0, t1;

	unsigned int valid_photons = 0;
	PhotonRecord** temp_photons = new PhotonRecord*[NUM_PHOTONS];

	PhotonRecord* photons_data;

	if (!useCollectionPhotons) {

		if (m_print_timings) std::cerr << "Starting Global photon pass   ... ";

		Buffer Globalphoton_rnd_seeds = m_context["Globalphoton_rnd_seeds"]->getBuffer();
		uint2* seeds = reinterpret_cast<uint2*>( Globalphoton_rnd_seeds->map() );
		for ( unsigned int i = 0; i < PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT; ++i )
			seeds[i] = random2u();
		Globalphoton_rnd_seeds->unmap();

		t0 = sutil::currentTime();
	
		m_context->launch( EnterPointGlobalPass,
			static_cast<unsigned int>(PHOTON_LAUNCH_WIDTH),
			static_cast<unsigned int>(PHOTON_LAUNCH_HEIGHT) );

		/// By computing the total number of photons as an unsigned long long we avoid 32 bit
		/// floating point addition errors when the number of photons gets sufficiently large
		/// (the error of adding two floating point numbers when the mantissa bits no longer
		/// overlap).
		photons_data = reinterpret_cast<PhotonRecord*>(m_Global_Photon_Buffer->map());

		t1 = sutil::currentTime();
		if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

		/// Push all valid photons to front of list
		for( unsigned int i = 0; i < NUM_PHOTONS; ++i ) {
			if( fmaxf( photons_data[i].energy ) > 0.0f ) {
				temp_photons[valid_photons++] = &photons_data[i];
			}
		}
		//if ( m_display_debug_buffer ) {
		//	std::cerr << " ** valid_photon/NUM_PHOTONS =  " 
		//		<< valid_photons<<"/"<<NUM_PHOTONS
		//		<<" ("<<valid_photons/static_cast<float>(NUM_PHOTONS)<<")\n";
		//}

	} else {
		photons_data = reinterpret_cast<PhotonRecord*>(m_Global_Photon_Buffer->map());
		char name[256];
		sprintf(name, "%s/%s/%s/%s/%d.txt", sutil::samplesDir(), "../../test", "photonMap", m_model.c_str(), m_frame_number % 10000 + 1);
		//printf("%s\n", name);
		FILE * fin = fopen(name, "rb");
		fread(&valid_photons, 1, sizeof(unsigned int), fin);
		fread(photons_data, valid_photons, sizeof(PhotonRecord), fin);
		for (int i = 0; i < valid_photons; ++i) {
			temp_photons[i] = &photons_data[i];
			//photonPrint(photons_data[i]);
		}
		fclose(fin);
	}

	PhotonRecord* photon_map_data = reinterpret_cast<PhotonRecord*>(m_Global_Photon_Map->map());;

	m_context["total_emitted"]->setFloat(static_cast<float>((unsigned long long)(m_frame_number + 1)*PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT));

	for (unsigned int i = 0; i < Global_Photon_Map_Size; ++i) {
		photon_map_data[i].energy = make_float3(0.0f);
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
	
	/// Build KD tree 
	if (m_print_timings) std::cerr << "Starting Global kd_tree build ... " << std::endl;
	t0 = sutil::currentTime();
	buildKDTree( temp_photons, 0, valid_photons, 0, photon_map_data, 0, m_split_choice, bbmin, bbmax );
	t1 = sutil::currentTime();
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
	double tstart, tend;
	tstart = sutil::currentTime();

	double t0, t1;
	Buffer output_buffer = m_context["rtpass_output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	output_buffer->getSize( buffer_width, buffer_height );

	if ((m_frame_number % 100 == 0 || m_frame_number <= 100) && m_frame_number > 0)
		m_print_image = 1;

	/// Print Images
	if (m_print_image)
	{
		char name1[256], name2[256];
		sprintf(name1, "%s/%s/%s/grab", sutil::samplesDir(), "progressivePhotonMap", "screengrab");
		sprintf(name2, "%s/%s/%s/%s", sutil::samplesDir(), "progressivePhotonMap", "screengrab", TestSceneNames[m_test_scene]);
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
	t0 = sutil::currentTime();

	m_context->launch( EnterPointRayTrace, buffer_width, buffer_height );
	t1 = sutil::currentTime();
	if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

	/// Trace photons
	buildGlobalPhotonMap();
	//buildCausticsPhotonMap();

	if (m_frame_number == 0) {
		t0 = sutil::currentTime();

		m_context->launch(EnterPointInitRadius, buffer_width, buffer_height);

		t1 = sutil::currentTime();

		/*Buffer hit_records = m_context["rtpass_output_buffer"]->getBuffer();
		HitRecord* hit_record_data = reinterpret_cast<HitRecord*>(hit_records->map());

		for (unsigned int j = 0; j < buffer_height; ++j) {
			for (unsigned int i = 0; i < buffer_width; ++i) {
				if (hit_record_data[j*buffer_width + i].flags & PPM_HIT) {
					printf("%d, %d, %.6f\n", j, i, hit_record_data[j*buffer_width + i].radius2);
				}
			}
		}

		hit_records->unmap();*/
	}

	/// Shade view rays by gathering photons
	if (m_print_timings) std::cerr << "Starting gather pass   ... ";
	t0 = sutil::currentTime();
	
	m_context->launch(EnterPointGlobalGather, buffer_width, buffer_height);
		
	/*m_context->launch( EnterPointCausticsGather,
		static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height) );*/
	t1 = sutil::currentTime();
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
		t0 = sutil::currentTime();
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
		t1 = sutil::currentTime();
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

	tend = sutil::currentTime();

	//std::cerr << "Pass :" << m_frame_number << " cost :" << tend - tstart << std::endl;
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
	m_context["camera_buffer"]->getBuffer()->setSize(width, height);
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

	Buffer camera_buffer = m_context["camera_buffer"]->getBuffer();
	uint3* camera_seeds = reinterpret_cast<uint3*>(camera_buffer->map());
	for (unsigned int i = 0; i < width*height; ++i)
	{
		camera_seeds[i].x = 0;
		camera_seeds[i].y = 0;
		camera_seeds[i].z = 0;
	}
	camera_buffer->unmap();

	Buffer image_rnd_seeds = m_context["image_rnd_seeds"]->getBuffer();
	uint2* seeds = reinterpret_cast<uint2*>( image_rnd_seeds->map() );
	for ( unsigned int i = 0; i < width*height; ++i )  
		seeds[i] = random2u();
	image_rnd_seeds->unmap();
}

// GeometryInstance
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

	//���ò���
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
	//std::cerr << filename << std::endl;
	GeometryGroup geometry_group = m_context->createGeometryGroup();
	std::string full_path = std::string( sutil::samplesDir() ) + "/progressivePhotonMap/" + filename;
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


void ProgressivePhotonScene::collectionPhotons(std::string objname, int frameNums) {
	std::cerr << frameNums << std::endl;
	double t0, t1;
	t0 = sutil::currentTime();
	for (int fi = 0; fi < frameNums; ++fi) {
		if (fi % 1000 == 0) {
			std::cerr << fi << std::endl;
		}
		Buffer Globalphoton_rnd_seeds = m_context["Globalphoton_rnd_seeds"]->getBuffer();
		uint2* seeds = reinterpret_cast<uint2*>(Globalphoton_rnd_seeds->map());
		for (unsigned int i = 0; i < PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT; ++i)
			seeds[i] = random2u();
		Globalphoton_rnd_seeds->unmap();

		m_context->launch(EnterPointGlobalPass,
			static_cast<unsigned int>(PHOTON_LAUNCH_WIDTH),
			static_cast<unsigned int>(PHOTON_LAUNCH_HEIGHT));

		PhotonRecord* photons_data = reinterpret_cast<PhotonRecord*>(m_Global_Photon_Buffer->map());

		unsigned int valid_photons = 0;
		PhotonRecord* temp_photons = new PhotonRecord[NUM_PHOTONS]; 
		for (unsigned int i = 0; i < NUM_PHOTONS; ++i) {
			if (fmaxf(photons_data[i].energy) > 0.0f) {
				temp_photons[valid_photons++] = photons_data[i];
			}
		}
		//std::cerr << " ** valid_photon/NUM_PHOTONS =  "
		//	<< valid_photons << "/" << NUM_PHOTONS
		//	<< " (" << valid_photons / static_cast<float>(NUM_PHOTONS) << ")\n";

		char name[256];
		sprintf(name, "%s/%s/%s/%s/%d.txt", sutil::samplesDir(), "progressivePhotonMap", "photonMap", objname.c_str(), fi);
		//printf("%s\n", name);
		FILE * fout = fopen(name, "wb");
		size_t a = fwrite(&valid_photons, sizeof(unsigned int), 1, fout);
		size_t b = fwrite(temp_photons, sizeof(PhotonRecord), valid_photons,  fout);
		//printf("%d %d\n", a, b);
		fclose(fout);
		
		delete[] temp_photons;
		m_Global_Photon_Buffer->unmap();
	}
	t1 = sutil::currentTime();
	if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;
}

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
	GLUTDisplay::init(argc, argv);

	bool print_timings = true;
	bool display_debug_buffer = false;
	bool cornell_box = false;
	float timeout = -1.0f;

	std::string model = "box";
	std::string modelNum = "1";

	for (int i = 1; i < argc; ++i) {
		std::string arg(argv[i]);
		if (arg == "--model") {
			if (++i < argc) {
				std::string arg(argv[i]);
				model = arg;
				if (++i < argc) {
					modelNum = std::string(argv[i]);
				}
				else {
					std::cerr << "Missing argument to " << arg << "\n";
					printUsageAndExit(argv[0]);
				}
			}
			else {
				std::cerr << "Missing argument to " << arg << "\n";
				printUsageAndExit(argv[0]);
			}
		}
	}

	try {
		ProgressivePhotonScene scene;
		if (print_timings) scene.printTimings();
		if (display_debug_buffer) scene.displayDebugBuffer();
		scene.selectScene(model, modelNum);

		//scene.useCollectionPhotons = true;
		//scene.m_collect_photon = true;
		//scene.collectPhotonsFrame = 1010;

		scene.setGatherMethod(ProgressivePhotonScene::Triangle_Inside_Method);
		GLUTDisplay::setProgressiveDrawingTimeout(timeout);
		GLUTDisplay::setUseSRGB(true);
		GLUTDisplay::run( "ProgressivePhotonScene", &scene, GLUTDisplay::CDProgressive );
	} catch( Exception& e ){
		sutil::reportErrorMessage(e.getErrorString().c_str());
		exit(1);
	}
	return 0;
}
