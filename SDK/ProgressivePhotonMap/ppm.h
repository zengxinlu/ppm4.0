
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

#include <optixu/optixu_math_namespace.h>

#define  transferFloat 1000000.0f

#define MYON_BIGGEST_INTER 2147483647
#define MYRON_PPM 0
#define Myron_Valid_Size 16
#define Myron_Red_High -300.1f
#define Myron_Red_Mid -200.1f
#define Myron_Red_Low -100.1f
#define Myron_Green_High -30.1f
#define Myron_Green_Mid -20.1f
#define Myron_Green_Low -10.1f
#define Myron_Blue_High -3.1f
#define Myron_Blue_Mid -2.1f
#define Myron_Blue_Low -1.1f

#define  PPM_X         ( 1 << 0 )
#define  PPM_Y         ( 1 << 1 )
#define  PPM_Z         ( 1 << 2 )
#define  PPM_LEAF      ( 1 << 3 )
#define  PPM_NULL      ( 1 << 4 )

#define  PPM_IN_SHADOW ( 1 << 5 )
#define  PPM_OVERFLOW  ( 1 << 6 )
#define  PPM_HIT       ( 1 << 7 )

enum RayTypes
{
    rtpass_ray_type = 0,
    ppass_and_gather_ray_type,
    shadow_ray_type,
};

enum HITRECORD_TYPE
{
	HITRECORD_DIFFUSE = 0,
	HITRECORD_SPECULAR,
	HITRECORD_REFRACTION
};

enum OptiXRayType
{
	RayTypeShadowRay = 0,
	RayTypeRayTrace,
	RayTypeCausticsPass,
	RayTypeGlobalPass,
	RayTypeNum
};

struct PPMLight
{
  optix::uint   is_area_light;
  optix::float3 power;

  // For spotlight
  optix::float3 position;
  optix::float3 direction;
  float         radius;

  // Parallelogram
  optix::float3 anchor;
  optix::float3 v1;
  optix::float3 v2;
};

struct HitRecord
{
 // float3 ray_dir;          // rgp

  optix::float3 position;         //
  optix::float3 normal;           // Material shader
  optix::float3 attenuated_Kd;
  optix::uint   flags;

  float         radius2;          // 
  float         photon_count;     // 
  optix::float3 flux;             //
  float         accum_atten;	  //
};


struct PackedHitRecord
{
  optix::float4 a;   // position.x, position.y, position.z, normal.x
  optix::float4 b;   // normal.y,   normal.z,   atten_Kd.x, atten_Kd.y
  optix::float4 c;   // atten_Kd.z, flags,      radius2,    photon_count
  optix::float4 d;   // flux.x,     flux.y,     flux.z,     accum_atten 
};


struct HitPRD
{
  optix::float3 attenuation;
  optix::uint   ray_depth;
};


struct PhotonRecord
{
  optix::float3 position;
  optix::float3 normal;      // Pack this into 4 bytes
  optix::float3 ray_dir;
  optix::float3 energy;
  optix::uint   axis;
  optix::int3 pad;			// padding,    (float)voronoi_area,    triangle index
};


struct PackedPhotonRecord
{
  optix::float4 a;   // position.x, position.y, position.z, normal.x
  optix::float4 b;   // normal.y,   normal.z,   ray_dir.x,  ray_dir.y
  optix::float4 c;   // ray_dir.z,  energy.x,   energy.y,   energy.z
  optix::float4 d;   // axis,       padding,    padding,    padding
};


struct PhotonPRD
{
  optix::float3 energy;
  optix::uint2  sample;
  optix::uint   pm_index;
  optix::uint   num_deposits;
  optix::uint   ray_depth;
  optix::uint   ray_type;
  int last_hitType;
};


struct ShadowPRD
{
  float attenuation;
};

//struct Vec3 { 
//	double x, y, z; /* etc - make sure you have overloaded operator== */ 
//	bool operator == (const Vec3 &a) const {
//		return a.x == x && a.y == y && a.z == z; 
//	}
//};
//
//namespace YAML {
//	template<>
//	struct convert<Vec3> {
//		static Node encode(const Vec3& rhs) {
//			Node node;
//			node.push_back(rhs.x);
//			node.push_back(rhs.y);
//			node.push_back(rhs.z);
//			return node;
//		}
//
//		static bool decode(const Node& node, Vec3& rhs) {
//			if (!node.IsSequence() || node.size() != 3) {
//				return false;
//			}
//
//			rhs.x = node[0].as<double>();
//			rhs.y = node[1].as<double>();
//			rhs.z = node[2].as<double>();
//			return true;
//		}
//	};
//}