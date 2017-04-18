
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

#pragma once

#include <sutil.h>
#include <vector>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <glm.h>
#include <string>
#include <map>

//-----------------------------------------------------------------------------
// 
//  ObjLoader class declaration 
//
//-----------------------------------------------------------------------------

struct StructTriangleIndex3{
	int x;
	int y;
	int z;
	bool operator<(const StructTriangleIndex3& _Right) const
	{
		if (x != _Right.x)
			return x < _Right.x;
		if (y != _Right.y)
			return y < _Right.y;
		return z < _Right.z;	
	}
};

static __inline__ StructTriangleIndex3 make_STI3(int x, int y, int z)
{
	StructTriangleIndex3 t; t.x = x; t.y = y; t.z = z; return t;
}

class PpmObjLoader
{
private:
	void initialization();
public:
	PpmObjLoader( const std::string& filename,           // Model filename
		optix::Context context,               // Context for RT object creation
		optix::GeometryGroup geometrygroup ); // Empty geom group to hold model
	PpmObjLoader( const std::string& filename,
		optix::Context context,
		optix::GeometryGroup geometrygroup,
		optix::Material material );           // Material override

	void load(int Myron_PPM, int Neighbor_2);
	void unload();
	void buildVertexIndex();
	void buildTriangleIndexTable();
	void depthFirstSearchTriangle(int, std::set<int> &, std::vector<int> &, int);

	optix::Aabb getSceneBBox()const { return m_aabb; }

	static bool isMyFile( const std::string& filename );

	GLMmodel* model;
	
	// Caustics photon
	std::vector<optix::float3> m_Caustics_Max;
	std::vector<optix::float3> m_Caustics_Min;
	std::vector<float> volumeArray;
	float volumeSum;
	/// 从顶点索引到其相邻三角面索引的映射表
	std::vector<std::vector<int>> *vertexIndexTablePtr;
	/// 从三角面索引到其邻域三角面索引的映射表
	std::vector<std::vector<int>> *triangleIndexTablePtr;
	std::map<StructTriangleIndex3, int> *pointsTriangleMap;
	bool useUnitization;
	bool useTriangleTopology;
	optix::GeometryInstance m_light_instance;
private:

	struct MatParams
	{
		std::string name;
		optix::float3 emissive;
		optix::float3 reflectivity;
		float  phong_exp;
		int    illum;
		float  alpha;
		optix::float3 Ka;
		optix::float3 Kd;
		optix::float3 Ks;
		float Kf;
		optix::TextureSampler ambient_map;
		optix::TextureSampler diffuse_map;
		optix::TextureSampler specular_map;
	};
	
	void get_triangle_aabb(GLMmodel* model, optix::int3& vindices, optix::float3& m_max, optix::float3& m_min);
	void createMaterial();
	void createGeometryInstances( GLMmodel* model,
		optix::Program mesh_intersect,
		optix::Program mesh_bbox );
	void loadVertexData( GLMmodel* model );
	void createMaterialParams( GLMmodel* model );
	int loadMaterialParams( optix::GeometryInstance gi, unsigned int index );

	std::string            m_pathname;
	std::string            m_filename;
	optix::Context        m_context;
	optix::GeometryGroup  m_geometrygroup;
	optix::Buffer         m_vbuffer;
	optix::Buffer         m_nbuffer;
	optix::Buffer         m_tbuffer;
	optix::Material       m_material;
	bool                   m_have_default_material;
	int					m_neighbor;
	optix::Aabb      m_aabb;
	std::vector<MatParams> m_material_params;
};



