
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "ppm.h"
#include "path_tracer.h"
#include "random.h"

using namespace optix;

//
// Scene wide variables
//
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );

//
// Ray generation program
//
rtBuffer<PhotonRecord, 1>        Global_Photon_Buffer;
rtBuffer<uint2, 2>               Globalphoton_rnd_seeds;
rtDeclareVariable(uint,          max_depth, , );
rtDeclareVariable(uint,          max_photon_count, , );
rtDeclareVariable(PPMLight,      light , , );
rtDeclareVariable(float,		refractionFacter , , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );


static __device__ __inline__ float2 rnd_from_uint2( uint2& prev )
{
	return make_float2(rnd(prev.x), rnd(prev.y));
}

static __device__ __inline__ void generateAreaLightPhoton( const PPMLight& light, const float4& d_sample, float3& o, float3& d, float mscale)
{
	// Choose a random position on light
	o = light.anchor + (d_sample.z * 2 - 1) * light.v1 + (d_sample.w * 2 - 1) * light.v2;

	// Choose a random direction from light
	float3 U, V, W;
	createONB( light.direction, U, V, W);
	sampleUnitHemisphere( make_float2(d_sample.x, d_sample.y), U * mscale, V * mscale, W, d );
}

static __device__ __inline__ void generateSpotLightPhoton( const PPMLight& light, const float4& d_sample, float3& o, float3& d)
{
	o = light.position;

	/*
	// Choose random dir by sampling disk of radius light.radius and projecting up to unit hemisphere
	float r = atanf( light.radius) * sqrtf( d_sample.x );
	float theta = 2.0f * M_PIf * d_sample.y;

	float x = r*cosf( theta );
	float y = r*sinf( theta );
	float z = sqrtf( fmaxf( 0.0f, 1.0f - x*x - y*y ) );
	*/

	// Choose random dir by sampling disk of radius light.radius and projecting up to unit hemisphere
	float2 square_sample = make_float2(d_sample.x, d_sample.y); 
	mapToDisk( square_sample );
	square_sample = square_sample * atanf( light.radius );
	float x = square_sample.x;
	float y = square_sample.y;
	float z = sqrtf( fmaxf( 0.0f, 1.0f - x*x - y*y ) );

	// Now transform into light space
	float3 U, V, W;
	createONB(light.direction, U, V, W);
	d =  x*U + y*V + z*W;
}


RT_PROGRAM void global_ppass_camera()
{
	size_t2 size     = Globalphoton_rnd_seeds.size();
	uint    pm_index = (launch_index.y * size.x + launch_index.x) * max_photon_count;
	uint2   seed     = Globalphoton_rnd_seeds[launch_index]; // No need to reset since we dont reuse this seed

	float4 direction_sample = make_float4(
		( static_cast<float>( launch_index.x ) + rnd( seed.x ) ) / static_cast<float>( size.x ),
		( static_cast<float>( launch_index.y ) + rnd( seed.y ) ) / static_cast<float>( size.y ),
		rnd( seed.x ),
		rnd( seed.y )
		);
	float3 ray_origin, ray_direction;

	Globalphoton_rnd_seeds[launch_index] = seed;

	if( light.is_area_light ) {
		generateAreaLightPhoton( light, direction_sample, ray_origin, ray_direction, 1.0 );
	} else {
		generateSpotLightPhoton( light, direction_sample, ray_origin, ray_direction );
	}

	optix::Ray ray(ray_origin, ray_direction, RayTypeGlobalPass, scene_epsilon );

	// Initialize our photons
	for(unsigned int i = 0; i < max_photon_count; ++i) {
		Global_Photon_Buffer[i+pm_index].energy = make_float3(0.0f);
	}

	PhotonPRD prd;
	//  rec.ray_dir = ray_direction; // set in ppass_closest_hit
	prd.energy = light.power;
	prd.sample = seed;
	prd.pm_index = pm_index;
	prd.num_deposits = 0;
	prd.ray_depth = 0;
	prd.last_hitType = HITRECORD_DIFFUSE;
	rtTrace( top_object, ray, prd );
}

//
// Closest hit material
//
rtDeclareVariable(float3,  Ks, , );
rtDeclareVariable(float3,  Kd, , );
rtDeclareVariable(float,  Alpha, , );
rtDeclareVariable(float3,  emitted, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(int4, triangle_info, attribute triangle_info, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PhotonPRD, hit_record, rtPayload, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtTextureSampler<float4, 2> diffuse_map;
rtTextureSampler<float4, 2> specular_map;

rtBuffer<uint2, 2>               image_rnd_seeds;					//随机种子
rtBuffer<float3>	vertex_buffer;     


static __device__ __inline__ float myDot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __device__ __inline__ float3 myCross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static __device__ __inline__ float3 myNormalise(float3 a)
{
	float len = sqrt(myDot(a, a));
	a.x /= len;
	a.y /= len;
	a.z /= len;
	return a;
}

static __device__ __inline__ float2 trans(float3& p0, float3& p1, float3& p2, float3& hp)
{
	float3 x_ = p1 - p0;
	x_ = myNormalise(x_);
	float3 y_ = p2 - p0;
	y_ = myNormalise(myCross(x_, myCross(x_, y_)));

	float3 tmp = hp - p0;
	float2 tmp2 = make_float2(myDot(x_, tmp), myDot(y_, tmp));
	return tmp2;
}

RT_PROGRAM void global_ppass_closest_hit()
{
	// Check if this is a light source
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal     = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	float3 direction = ray.direction;
	float3 hit_point = ray.origin + t_hit*ray.direction;
	float3 new_ray_dir;
	float3 Kd = make_float3( tex2D( diffuse_map,  texcoord.x, texcoord.y) );
	float3 Ks = make_float3( tex2D( specular_map, texcoord.x, texcoord.y) );
	float3 temp_Kd = Kd;
	float3 temp_Ks = Ks;

	float n_dot_l = dot(ffnormal, -ray.direction);
	if( fmaxf( Kd ) > 0.0f && n_dot_l > 0.f) {
		// We hit a diffuse surface; record hit if it has bounced at least once
		if( hit_record.ray_depth >= 0 ) {
			PhotonRecord& rec = Global_Photon_Buffer[hit_record.pm_index + hit_record.num_deposits];
			rec.position = hit_point;
			rec.normal = ffnormal;
			rec.ray_dir = ray.direction;
			rec.energy = hit_record.energy * n_dot_l * Kd;
			rec.triangleIndex = triangle_info.w;

			
			float3 p0 = vertex_buffer[ triangle_info.x ];
			float3 p1 = vertex_buffer[ triangle_info.y ];
			float3 p2 = vertex_buffer[ triangle_info.z ];

			rec.frag_position = trans(p0, p1, p2, hit_point);
			
			rec.tindex.x = triangle_info.w;

			float tempFloat;
			hit_record.num_deposits++;
			if (hit_record.num_deposits >= max_photon_count)
				return;
		}
		hit_record.energy = Kd * n_dot_l * hit_record.energy; 
		float3 U, V, W;
		createONB(ffnormal, U, V, W);
		sampleUnitHemisphere(rnd_from_uint2(hit_record.sample), U, V, W, new_ray_dir);

		hit_record.ray_depth++;
		if ( hit_record.ray_depth >= max_depth) return;
		optix::Ray new_ray( hit_point, new_ray_dir, RayTypeGlobalPass, scene_epsilon );
		rtTrace(top_object, new_ray, hit_record);
	} 
	else
	{
		while (1) {
			if (Alpha < 1)
			{
				PhotonPRD refract_prd = hit_record;
				refract_prd.ray_depth ++;
				if ( refract_prd.ray_depth >= max_depth) break;

				float refraction_facter = refractionFacter;
				float critical_sina = 1/refraction_facter;
				float critical_radian = asinf(critical_sina);

				float max_incidence_radian = M_PIf/2.0, max_emergent_radian = M_PIf * 41.8f/180.0f;
				float top_refacter = 0.96f;
				float bottom_incidence_t = powf(1 - top_refacter, 1/max_incidence_radian);
				float bottom_emergent_t = powf(1 - top_refacter, 1/max_emergent_radian);
				float K_refacter = 1;
		
				float3 R;
				// 折射
				if (refract(R, direction, world_shading_normal, refraction_facter) == true)
				{
					// 入射角
					float incidence_sina = sqrtf( 1.0 - powf( fabsf(dot(direction, world_shading_normal)), 2.0f) );
					float incidence_radian = asinf(incidence_sina);

					// 折射率
					if ( dot(direction, world_shading_normal) < 0)
						K_refacter = 1 - pow(bottom_incidence_t, max_incidence_radian - incidence_radian);
					else
						K_refacter = 1 - pow(bottom_emergent_t, max_emergent_radian - incidence_radian);

					refract_prd.energy *= K_refacter;
					temp_Ks = make_float3(1 - K_refacter);

					optix::Ray refr_ray( hit_point, R, RayTypeGlobalPass, scene_epsilon );
					rtTrace( top_object, refr_ray, refract_prd );
				}
				// 全反射
				else
				{
					refract_prd.energy *= 1.0f;
					if (fmaxf( temp_Ks ) > 0.0f) 
					{
						R = reflect( direction, ffnormal );
						optix::Ray refr_ray( hit_point, R, RayTypeGlobalPass, scene_epsilon );
						rtTrace( top_object, refr_ray, refract_prd );
					}
				}
				break;
			}

			if (fmaxf( temp_Ks ) > 0.0f) 
			{
				float3 Reflect_dir = reflect( direction, ffnormal );
				PhotonPRD reflect_prd = hit_record;
				reflect_prd.energy *= temp_Ks;
				reflect_prd.ray_depth ++;
				if (reflect_prd.ray_depth >= max_depth) break;
				optix::Ray refl_ray( hit_point, Reflect_dir, RayTypeGlobalPass, scene_epsilon );
				rtTrace( top_object, refl_ray, reflect_prd );
			}
			break;
		}
	}
}


rtBuffer<PhotonRecord, 1>		 Caustics_Photon_Buffer;
rtBuffer<uint, 1>			     Causticsphoton_rnd_seeds;
rtDeclareVariable(float3,		 target_max, , );
rtDeclareVariable(float3,		 target_min, , );
rtDeclareVariable(uint,			 PhotonStart, , );
RT_PROGRAM void caustics_ppass_camera() {
	uint   pm_index = launch_index.x + PhotonStart;
	uint   seed     = Causticsphoton_rnd_seeds[pm_index]; // No need to reset since we dont reuse this seed

	float3 direction_sample = make_float3(rnd( seed ), rnd( seed ), rnd( seed ));
	Causticsphoton_rnd_seeds[pm_index] = seed;

	
	float3 ray_origin, ray_direction;
	ray_origin = light.anchor;
	ray_direction = direction_sample*target_max + (make_float3(1.f)-direction_sample)*target_min - ray_origin;
	ray_direction = normalize(ray_direction);

	optix::Ray ray(ray_origin, ray_direction, RayTypeCausticsPass, scene_epsilon );

	Global_Photon_Buffer[pm_index].energy = make_float3(0.0f);

	PhotonPRD prd;
	//  rec.ray_dir = ray_direction; // set in ppass_closest_hit
	prd.energy = light.power;
	prd.sample.x = seed;
	prd.pm_index = pm_index;
	prd.num_deposits = 0;
	prd.ray_depth = 0;
	prd.last_hitType = HITRECORD_DIFFUSE;
	rtTrace( top_object, ray, prd );
}

rtDeclareVariable(float,  RefractionIndex, , );

RT_PROGRAM void caustics_ppass_closest_hit() {
	// Check if this is a light source
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal     = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	float3 direction = ray.direction;
	float3 hit_point = ray.origin + t_hit*ray.direction;
	float3 new_ray_dir;
	float3 Kd = make_float3( tex2D( diffuse_map,  texcoord.x, texcoord.y) );
	float3 Ks = make_float3( tex2D( specular_map, texcoord.x, texcoord.y) );
	float3 temp_Kd = Kd;
	float3 temp_Ks = Ks;

	float n_dot_l = dot(ffnormal, -ray.direction);
	if( fmaxf( Kd ) > 0.0f && n_dot_l > 0.f) {
		// We hit a diffuse surface; record hit if it has bounced at least once
		if( hit_record.ray_depth > 2 ) {		// For Sibnik
		//if( hit_record.ray_depth > 0 ) {		// For other
			PhotonRecord& rec = Caustics_Photon_Buffer[hit_record.pm_index];
			rec.position = hit_point;
			rec.normal = ffnormal;
			rec.ray_dir = ray.direction;
			rec.energy =  Kd * n_dot_l * hit_record.energy;
			float tempFloat;
			hit_record.num_deposits++;
			if (hit_record.num_deposits >= max_photon_count)
				return;
		}
		hit_record.energy = Kd * n_dot_l * hit_record.energy; 
		float3 U, V, W;
		createONB(ffnormal, U, V, W);
		sampleUnitHemisphere(rnd_from_uint2(hit_record.sample), U, V, W, new_ray_dir);
	} 
	else
	{
		while (1)
		{	// if it is fraction 折射
			if (Alpha < 1)
			{
				float refraction_facter = 2.1;
				float critical_sina = 1/refraction_facter;
				float critical_radian = asinf(critical_sina);

				float max_incidence_radian = M_PIf/2.0, max_emergent_radian = M_PIf * 41.8f/180.0f;
				float top_refacter = 0.96f;
				float bottom_incidence_t = powf(1 - top_refacter, 1/max_incidence_radian);
				float bottom_emergent_t = powf(1 - top_refacter, 1/max_emergent_radian);
				float K_refacter = 1;

				// 折射
				if (refract(new_ray_dir, ray.direction, world_shading_normal, refraction_facter) == true)
				{
					// 入射角
					float incidence_sina = sqrtf( 1.0 - powf( fabsf(dot(ray.direction, world_shading_normal)), 2.0f) );
					float incidence_radian = asinf(incidence_sina);

					// 折射率
					if ( dot(ray.direction, world_shading_normal) < 0)
						K_refacter = 1 - pow(bottom_incidence_t, max_incidence_radian - incidence_radian);
					else
						K_refacter = 1 - pow(bottom_emergent_t, max_emergent_radian - incidence_radian);

					hit_record.energy *= K_refacter;
					temp_Ks = make_float3(1 - K_refacter);
				}
				// 全反射
				else
				{
					hit_record.energy *= 1.0f;
					new_ray_dir = reflect( ray.direction, ffnormal );
				}
				break;
			}
			// 反射
			if (fmaxf( temp_Ks ) > 0.0f) 
			{
				new_ray_dir = reflect( ray.direction, ffnormal );
				hit_record.energy *= temp_Ks;
			}
			break;
		}
	}

	hit_record.ray_depth++;
	if ( hit_record.ray_depth >= max_depth )//|| hit_record.ray_depth > max_depth)
		return;

	optix::Ray new_ray( hit_point, new_ray_dir, RayTypeCausticsPass, scene_epsilon );
	rtTrace(top_object, new_ray, hit_record);
}