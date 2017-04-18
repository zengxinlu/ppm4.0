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

using namespace optix;

//
// Scene wide variables
//
rtDeclareVariable(float,         Scene_Epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint,			 Progressive, , );
rtDeclareVariable(float,			 Largest_Dist, , );
rtDeclareVariable(float,		 CausticsRadius2, , );
rtDeclareVariable(float,		 GlobalRadius2, , );
//
// Ray generation program
//
rtBuffer<uint2, 2>               Ray_Pass_Seeds;
rtBuffer<HitRecord, 1>           GlobalHitRecord_Buffer;
rtBuffer<HitRecord, 1>           CausticsHitRecord_Buffer;

rtBuffer<uint2, 2>				 RayTrace_Pass_Seeds;
rtDeclareVariable(float3,        Rtpass_Eye, , );
rtDeclareVariable(float3,        Rtpass_U, , );
rtDeclareVariable(float3,        Rtpass_V, , );
rtDeclareVariable(float3,        Rtpass_W, , );

rtBuffer<float4, 2>              Frame_Buffer;
rtBuffer<float4, 2>              Output_Buffer;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(float,			 FrameCount, , );

RT_PROGRAM void rtpass_camera()
{
	if (FrameCount < 1.5)
		Output_Buffer[launch_index] = make_float4(0.f);
	float2 screen = make_float2( Frame_Buffer.size() );
	Frame_Buffer[launch_index] = make_float4(0.f);

		
	// Random
	uint2 seed = Ray_Pass_Seeds[launch_index];
	float2 sample = make_float2( rnd( seed.x ) , rnd( seed.y ) );
	Ray_Pass_Seeds[launch_index] = seed;

	float2 d = ( make_float2(launch_index) + sample ) / screen * 2.0f - 1.0f;
	float3 ray_origin = Rtpass_Eye;
	float3 ray_direction = normalize(d.x*Rtpass_U + d.y*Rtpass_V + Rtpass_W);
 
	optix::Ray ray(ray_origin, ray_direction, RayTypeRayTrace, Scene_Epsilon);

	HitPRD prd;
	prd.attenuation = make_float3( 1.0f );
	prd.ray_depth   = 0u; 
	prd.last_hitType = HITRECORD_DIFFUSE;
	prd.pt_index = 0;
	int currentRecord = (launch_index.y * screen.x + launch_index.x)*SPPM_HITRECORD_TYPE_SIZE;

	// Clear HitRecord_Buffer
	if (FrameCount == 1)
	{
		HitRecord t_rec;
		t_rec.photon_count = 0;
		t_rec.indirect_flux = make_float3(0.f);
		t_rec.flags = PPM_NULL;
		// Global
		t_rec.radius2 = GlobalRadius2;
		GlobalHitRecord_Buffer[currentRecord] = t_rec;
		// Caustics
		t_rec.radius2 = CausticsRadius2;
		CausticsHitRecord_Buffer[currentRecord] = t_rec;
	}
	rtTrace( top_object, ray, prd );
}

// 
// Closest hit material
// 
rtDeclareVariable(float,  Alpha, , );
rtDeclareVariable(float,  direct_coeff, , );
rtDeclareVariable(float,  RefractionIndex, , );
rtDeclareVariable(float3,  grid_color, , );
rtDeclareVariable(uint,    use_grid, , );
rtDeclareVariable(float3,  emitted, , );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtTextureSampler<float4, 2> diffuse_map;
rtTextureSampler<float4, 2> specular_map;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 

rtDeclareVariable(HitPRD, hit_prd, rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(PPMLight,      light , , );

RT_PROGRAM void rtpass_closest_hit()
{
	int currentRecord = (launch_index.y * Frame_Buffer.size().x + launch_index.x)*SPPM_HITRECORD_TYPE_SIZE;
	float2 screen = make_float2( Frame_Buffer.size() );
	float3 direction    = ray.direction;
	float3 origin       = ray.origin;
	float3 curAttenuation = hit_prd.attenuation;
	int	   curRayDepth = hit_prd.ray_depth;
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	float3 ffnormal     = faceforward( world_shading_normal, -direction, world_geometric_normal );
	float3 hit_point    = origin + t_hit*direction;
	// add Kd, Ks
	float3 Kd = make_float3( tex2D( diffuse_map,  texcoord.x, texcoord.y) );
	float3 Ks = make_float3( tex2D( specular_map,  texcoord.x, texcoord.y) );
	float m_Kd = fmaxf(Kd), m_Ks = fmaxf(Ks), m_Kf = Alpha;
	// Selector
	uint2 seed = Ray_Pass_Seeds[launch_index];
	float m_SelectorX = rnd( seed.x ) ,m_SelectorY = rnd( seed.y );
	float m_Selector = (m_SelectorX + m_SelectorY)/2.f;
	Ray_Pass_Seeds[launch_index] = seed;
	m_Kd /= (m_Kd + m_Ks + m_Kf);
	m_Ks /= (m_Kd + m_Ks + m_Kf);
	m_Kf /= (m_Kd + m_Ks + m_Kf);
	/*************************************************/
	// If diffuse
	if (m_Kd > 0.f)//0.f && m_Selector < m_Kd)
	{
		// Record hit point
		if (hit_prd.pt_index == 0)
		{		
			// Global
			HitRecord t_rec = GlobalHitRecord_Buffer[currentRecord];
			t_rec.position = hit_point; 
			t_rec.normal = ffnormal;
			t_rec.attenuated_Kd = Kd * curAttenuation;// * m_Kd;
			t_rec.flags = PPM_HIT;
			GlobalHitRecord_Buffer[currentRecord] = t_rec;
			// Caustics
			t_rec = CausticsHitRecord_Buffer[currentRecord];
			t_rec.position = hit_point; 
			t_rec.normal = ffnormal;
			t_rec.attenuated_Kd = Kd * curAttenuation;// * m_Kd;
			t_rec.flags = PPM_HIT;
			CausticsHitRecord_Buffer[currentRecord] = t_rec;
			hit_prd.pt_index = 1;
		}
		float3 direct_flux = make_float3(0.f);
		float sample_num = 4.f;
		for (float i = 0;i < sample_num;i ++)
			for (float j = 0;j < sample_num;j ++)
			{
				// Direct light
				float3 shadow_ray_dir = light.anchor + light.v1 * ( (i + rnd(seed.x))/sample_num * 2.f - 1.f) + 
					light.v2 * ( (j + rnd(seed.y))/sample_num * 2.f - 1.f) - hit_point;
				float dist_to_l = sqrtf(dot(shadow_ray_dir, shadow_ray_dir));
				shadow_ray_dir /= dist_to_l;
				float3 H = shadow_ray_dir - direction;
				float n_dot_l = dot(ffnormal, shadow_ray_dir), lt_l = dot(ffnormal, H);
				// light is on the contrary
				if (n_dot_l > 0.f && lt_l > 0.f && dot(-light.direction, shadow_ray_dir) > 0)
				{
					// Shadow ray
					ShadowPRD prd;
					prd.attenuation = 1.0f;
					optix::Ray shadow_ray( hit_point, shadow_ray_dir, RayTypeShadowRay, Scene_Epsilon, dist_to_l - Scene_Epsilon);
					rtTrace( top_object, shadow_ray, prd );
					float Falloff = 1.f/(dist_to_l*10.f/Largest_Dist+1.f);
					direct_flux += (light.power * (Kd*n_dot_l + Ks*pow(lt_l, 5.f) * Falloff) * curAttenuation)
						* prd.attenuation;
				}
			}
		Frame_Buffer[launch_index] += make_float4(direct_flux)  * direct_coeff / sample_num / sample_num * m_Kd;
	}
	/*************************************************/
	// If specular
	// if (m_Ks > 0.f && m_Selector < m_Kd + m_Ks)
	if (m_Ks > 0.f && curRayDepth < 2)
	{
		// Make reflection ray
		hit_prd.attenuation = curAttenuation * Ks;
		hit_prd.ray_depth = curRayDepth + 1;
		hit_prd.last_hitType = HITRECORD_SPECULAR;
		float3 R = reflect( direction, ffnormal );
		optix::Ray refl_ray( hit_point, R, RayTypeRayTrace, Scene_Epsilon );
		rtTrace( top_object, refl_ray, hit_prd );
	}
	/*************************************************/
	// If frac
	if (m_Kf > 0.f && curRayDepth < 5)
	{
		// Make reflection ray
		hit_prd.attenuation = curAttenuation * Alpha;
		hit_prd.ray_depth = curRayDepth + 1;
		hit_prd.last_hitType = HITRECORD_REFRACTION;
		float3 R;
		float ddotn = dot(direction, ffnormal);
		// Get correct normal and Kf
		float3 nerffnormal = (ddotn < 0)?ffnormal:(-ffnormal);
		float nerKf = (ddotn < 0)?RefractionIndex:(1.f/RefractionIndex);
		float cosa1 = -dot(nerffnormal, direction);
		float cosa2 = sqrt( 1 - (1-cosa1*cosa1)/nerKf/nerKf );
		R = direction/nerKf + (cosa1/nerKf - cosa2)*nerffnormal;
		optix::Ray refl_ray( hit_point, normalize(R), RayTypeRayTrace, Scene_Epsilon );
		rtTrace( top_object, refl_ray, hit_prd );
	}
}

//
// Miss program
//
rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void rtpass_miss()
{
	float theta = atan2f( ray.direction.x, ray.direction.z );
	float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
	float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v     = 0.5f * ( 1.0f + sin(phi) );
	float3 result = make_float3(tex2D(envmap, u, v));
 
	Frame_Buffer[launch_index] += make_float4(result*hit_prd.attenuation);
}

//       
// Stack overflow program
//
rtDeclareVariable(float3, Rtpass_Bad_Color, , );
RT_PROGRAM void rtpass_exception()
{
	float2 screen = make_float2( Frame_Buffer.size() );
	uint currentRecord = (launch_index.y * screen.x + launch_index.x)*SPPM_HITRECORD_TYPE_SIZE;
	for (int i = 0;i < SPPM_HITRECORD_TYPE_SIZE;i ++)
	{
		HitRecord& rec = GlobalHitRecord_Buffer[currentRecord + i];
		rec.flags = PPM_OVERFLOW;
		rec.attenuated_Kd = Rtpass_Bad_Color;
	}
	Frame_Buffer[launch_index] = make_float4(Rtpass_Bad_Color);
}

rtDeclareVariable(ShadowPRD, shadow_prd, rtPayload, );

RT_PROGRAM void rtpass_any_hit()
{
	shadow_prd.attenuation = 0.0f;

	rtTerminateRay();
}
