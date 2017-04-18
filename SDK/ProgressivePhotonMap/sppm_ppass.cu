/************************************************************************/
/* 
Powered by mengyang from GIL Peking University
*/
/************************************************************************/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

//#include "SPPMEntryPoint.h"

#include "path_tracer.h"
#include "ppm.h"
#include "random.h"

using namespace optix;
//
// Scene wide variables
//
rtDeclareVariable(float,         Scene_Epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(float3,		 aabb_max, , ); 
rtDeclareVariable(float3,		 aabb_min, , ); 
rtDeclareVariable(float3,		 target_max, , );
rtDeclareVariable(float3,		 target_min, , );
rtDeclareVariable(uint,			 PhotonStart, , );
rtDeclareVariable(uint,          Global_Photon_Depot, , );


rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtTextureSampler<float4, 2> diffuse_map;
rtTextureSampler<float4, 2> specular_map;

//
// Ray generation program
//
rtBuffer<PhotonRecord, 1>		 Caustics_Photon_Buffer;
rtBuffer<uint, 1>               Caustics_Pass_Seeds;
rtDeclareVariable(PPMLight,      light , , );

rtBuffer<float4, 2>              Frame_Buffer;
rtBuffer<float4, 2>              Output_Buffer;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(float,			 FrameCount, , );

RT_PROGRAM void cppass_camera()
{
	uint    pm_index = launch_index.x + PhotonStart;
	uint   seed     = Caustics_Pass_Seeds[pm_index]; // No need to reset since we dont reuse this seed

	// We only collect Global Photon
	float3 direction_sample = make_float3(rnd( seed ), rnd( seed ), rnd( seed ));
	Caustics_Pass_Seeds[pm_index] = seed;

	// Ray
	float3 ray_origin, ray_direction;
	if(1)//light.lightType == area_light_type) 
	{
		ray_origin = light.anchor;
		ray_direction = direction_sample*target_max + (make_float3(1.f)-direction_sample)*target_min - ray_origin;
		ray_direction = normalize(ray_direction);
	}
	else if (light.lightType == spot_light_type)
	{

		// add small disturbance
		/*
		generateAreaLightPhoton( light, 
			make_float2( (launch_index.x+direction_sample.x)/(float)size.x, (launch_index.y+direction_sample.y)/(float)size.x),
			ray_origin, ray_direction );
		direction_sample = make_float2( rnd( seed.x ) , rnd( seed.y ) )*2.f - 1.f;
		ray_origin += light.v1*direction_sample.x + light.v2*direction_sample.y;
		*/
	}
	// Parallel Light Type
	else if (light.lightType == parallel_light_type)
	{
	}

	optix::Ray ray(ray_origin, ray_direction, RayTypeCausticsPass, Scene_Epsilon );

	// Initialize our photons
	Caustics_Photon_Buffer[pm_index].energy = make_float3(0.0f);

	PhotonPRD prd;
	prd.global_deposits = 0;
	prd.sample.x = seed;
	//  rec.ray_dir = ray_direction; // set in ppass_closest_hit
	prd.energy = light.power;
	prd.pm_index = pm_index;
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
rtDeclareVariable(float,  RefractionIndex, , );
rtDeclareVariable(float3,  emitted, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PhotonPRD, hit_record, rtPayload, );

rtDeclareVariable(float,		 Largest_Dist, , );
RT_PROGRAM void cppass_closest_hit()
{
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal     = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	float3 direction = ray.direction;
	float3 hit_point = ray.origin + t_hit*ray.direction;
	float3 new_ray_dir;

	// Kd and Ks
 	float3 Kd = make_float3( tex2D( diffuse_map,  texcoord.x, texcoord.y) );
 	float3 Ks = make_float3( tex2D( specular_map,  texcoord.x, texcoord.y) );
	// random
	float rnd_arg = rnd(hit_record.sample.x);
	float k_d = fmaxf(Kd), k_s = fmaxf(Ks), k_f = Alpha;
	rnd_arg *= (k_d + k_s + k_f);
	if (rnd_arg < k_d || hit_record.ray_depth > 4)
	{
		if (hit_record.last_hitType != HITRECORD_DIFFUSE)
		{
			// launch_index.x is [0, BUFFER_CLUSTER_SIZE)
			PhotonRecord& rec = Caustics_Photon_Buffer[hit_record.pm_index];
			rec.position = hit_point;
			rec.normal = ffnormal;
			rec.ray_dir = ray.direction;
			rec.energy = fmaxf(hit_record.energy * Kd, make_float3(0.f));
		}
		return;
	}
	else if (rnd_arg < k_d + k_s)
	{
		// Make reflection ray
		hit_record.energy = hit_record.energy * Ks;
		hit_record.last_hitType = HITRECORD_SPECULAR;
		new_ray_dir = reflect( direction, ffnormal );
	}
	else
	{
		hit_record.energy = hit_record.energy * k_f;
		hit_record.last_hitType = HITRECORD_REFRACTION;
		// if inside
		float3 R;
		float ddotn = dot(direction, ffnormal);
		// Get correct normal and Kf
		float3 nerffnormal = (ddotn < 0)?ffnormal:(-ffnormal);
		float nerKf = (ddotn < 0)?RefractionIndex:(1.f/RefractionIndex);
		float cosa1 = -dot(nerffnormal, direction);
		float cosa2 = sqrt( 1 - (1-cosa1*cosa1)/nerKf/nerKf );
		R = direction/nerKf + (cosa1/nerKf - cosa2)*nerffnormal;
		new_ray_dir = normalize(R);
	}
	hit_record.ray_depth ++;
	optix::Ray new_ray( hit_point, new_ray_dir, RayTypeCausticsPass, Scene_Epsilon );
	rtTrace( top_object, new_ray, hit_record );
}
RT_PROGRAM void ppass_exception()
{
}

RT_PROGRAM void cppass_miss()
{
}


rtBuffer<PhotonRecord, 1>		 Global_Photon_Buffer;
rtBuffer<uint, 1>               Global_Pass_Seeds;
RT_PROGRAM void gppass_camera()
{
	uint    pm_index = launch_index.x;
	uint   seed     = Global_Pass_Seeds[pm_index]; // No need to reset since we dont reuse this seed

	// We only collect Global Photon
	float3 direction_sample = make_float3(rnd( seed ), rnd( seed ), rnd( seed ));
	Global_Pass_Seeds[pm_index] = seed;

	// Ray
	float3 ray_origin, ray_direction;
	if(1)//light.lightType == area_light_type) 
	{
		ray_origin = light.position;
		// Look target
		ray_direction = light.anchor + light.v1 * (direction_sample.x * 2.f - 1.f) + 
			light.v2 * (direction_sample.y * 2.f - 1.f);
		ray_direction = normalize(ray_direction - ray_origin);
	}
	else if (light.lightType == spot_light_type)
	{
	}
	// Parallel Light Type
	else if (light.lightType == parallel_light_type)
	{
	}

	optix::Ray ray(ray_origin, ray_direction, RayTypeGlobalPass, Scene_Epsilon );

	// Initialize our photons
	for (int i = 0;i < Global_Photon_Depot;i ++)
		Global_Photon_Buffer[pm_index * Global_Photon_Depot + i].energy = make_float3(0.0f);

	PhotonPRD prd;
	prd.global_deposits = 0;
	prd.sample.x = seed;
	//  rec.ray_dir = ray_direction; // set in ppass_closest_hit
	prd.energy = light.power;
	prd.pm_index = pm_index;
	prd.ray_depth = 0;
	prd.last_hitType = HITRECORD_DIFFUSE;
	rtTrace( top_object, ray, prd );
}
RT_PROGRAM void gppass_closest_hit()
{
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal     = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	float3 direction = ray.direction;
	float3 hit_point = ray.origin + t_hit*ray.direction;
	float3 new_ray_dir;

	// random
	float rnd_arg = rnd(hit_record.sample.x);
	float k_d = fmaxf(Kd), k_s = fmaxf(Ks), k_f = Alpha;
	rnd_arg *= (k_d + k_s + k_f);
	if (rnd_arg < k_d)
	{
		if ( (hit_record.last_hitType == HITRECORD_DIFFUSE) )
		{
			// launch_index.x is [0, BUFFER_CLUSTER_SIZE)
			PhotonRecord& rec = Global_Photon_Buffer[hit_record.pm_index * Global_Photon_Depot
					+ hit_record.global_deposits];
			rec.position = hit_point;
			rec.normal = ffnormal;
			rec.ray_dir = ray.direction;
			rec.energy = fmaxf(hit_record.energy * Kd, make_float3(0.f));
			hit_record.global_deposits ++;
			if (hit_record.global_deposits == Global_Photon_Depot)
				return;
		}
		// Make reflection ray
		hit_record.energy = hit_record.energy * Kd;
		hit_record.last_hitType = HITRECORD_DIFFUSE;
		float3 reflect_ray_dir = reflect( direction, ffnormal );
		float3 U, V, W;
		createONB(ffnormal, U, V, W);
		sampleUnitHemisphere(make_float2(rnd(hit_record.sample.x), rnd(hit_record.sample.x)),
			U, V, W, new_ray_dir);
	}
	else if (rnd_arg < k_d + k_s)
	{
		// Make reflection ray
		hit_record.energy = hit_record.energy * Ks;
		hit_record.last_hitType = HITRECORD_SPECULAR;
		new_ray_dir = reflect( direction, ffnormal );
	}
	else
	{
		hit_record.energy = hit_record.energy * k_f;
		hit_record.last_hitType = HITRECORD_REFRACTION;
		// if inside
		float3 R;
		float ddotn = dot(direction, ffnormal);
		// Get correct normal and Kf
		float3 nerffnormal = (ddotn < 0)?ffnormal:(-ffnormal);
		float nerKf = (ddotn < 0)?RefractionIndex:(1.f/RefractionIndex);
		float cosa1 = -dot(nerffnormal, direction);
		float cosa2 = sqrt( 1 - (1-cosa1*cosa1)/nerKf/nerKf );
		R = direction/nerKf + (cosa1/nerKf - cosa2)*nerffnormal;
		new_ray_dir = normalize(R);
	}
	hit_record.ray_depth ++;
	if ( hit_record.ray_depth < 6 )
	{
		optix::Ray new_ray( hit_point, new_ray_dir, RayTypeGlobalPass, Scene_Epsilon );
		rtTrace( top_object, new_ray, hit_record );
	}
}

RT_PROGRAM void gppass_miss()
{
	float3 ray_origin, ray_direction;
	float2 direction_sample = make_float2(rnd( hit_record.sample.x ), rnd( hit_record.sample.x ));
	if (1)
	{
		ray_origin = light.position;
		// Look target
		ray_direction = light.anchor + light.v1 * (direction_sample.x * 2.f - 1.f) + 
			light.v2 * (direction_sample.y * 2.f - 1.f);
		ray_direction = normalize(ray_direction - ray_origin);
	}
	// Relaunch
	optix::Ray ray(ray_origin, ray_direction, RayTypeGlobalPass, Scene_Epsilon );
	hit_record.ray_depth = 0;
	hit_record.last_hitType = HITRECORD_DIFFUSE;
	rtTrace( top_object, ray, hit_record );
}