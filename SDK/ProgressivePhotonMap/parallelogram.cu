
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

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float4, plane, , );
rtDeclareVariable(float3, v1, , );
rtDeclareVariable(float3, v2, , );
rtDeclareVariable(float, v1_l, , );
rtDeclareVariable(float, v2_l, , );
rtDeclareVariable(float, min_fovy, , );
rtDeclareVariable(float, max_radius2, , );
rtDeclareVariable(float3, anchor, , );
rtDeclareVariable(int, lgt_instance, , ) = {0};
rtDeclareVariable(float,         rtpass_default_radius2, , );

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float, primary_edge, attribute primary_edge, ); 
rtDeclareVariable(int4, triangle_info, attribute triangle_info, ); 
rtDeclareVariable(float, secondary_edge, attribute secondary_edge, ); 
rtDeclareVariable(int, lgt_idx, attribute lgt_idx, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(int, is_soft_surface, attribute is_soft_surface, ); 

RT_PROGRAM void intersect(int primIdx)
{
	float3 n = make_float3( plane );
	float dt = dot(ray.direction, n );
	float t = (plane.w - dot(n, ray.origin))/dt;
	if( t > ray.tmin && t < ray.tmax ) {
		float3 p = ray.origin + ray.direction * t;
		float3 vi = p - anchor;
		float a1 = dot(v1, vi);
		if(a1 >= 0 && a1 <= 1){
			float a2 = dot(v2, vi);
			if(a2 >= 0 && a2 <= 1){
				if( rtPotentialIntersection( t ) ) {

					triangle_info = make_int4(1);
					is_soft_surface = 1;
					// Calculate tar_radius by mengyang
					float primary_radius = ( (a1 > 0.5)?(1.0f-a1):a1 )*v1_l;
					float secondary_radius = ( (a2 > 0.5)?(1.0f-a2):a2 )*v2_l;
					if (secondary_radius < primary_radius)
					{
						float temp_radius = secondary_radius;
						secondary_radius = primary_radius;
						primary_radius = temp_radius;
					}
					primary_edge = primary_radius*primary_radius;
					secondary_edge = secondary_radius*secondary_radius;

					if (primary_edge > max_radius2)
						primary_edge = max_radius2;
					if (secondary_edge > max_radius2)
						secondary_edge = max_radius2;

					shading_normal = geometric_normal = n;
					texcoord = make_float3(a1,a2,0);
					lgt_idx = lgt_instance;
					rtReportIntersection( 0 );
				}
			}
		}
	}
}

RT_PROGRAM void bounds (int, float result[6])
{
	// v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
	const float3 tv1  = v1 / dot( v1, v1 );
	const float3 tv2  = v2 / dot( v2, v2 );
	const float3 p00  = anchor;
	const float3 p01  = anchor + tv1;
	const float3 p10  = anchor + tv2;
	const float3 p11  = anchor + tv1 + tv2;
	const float  area = length(cross(tv1, tv2));

	optix::Aabb* aabb = (optix::Aabb*)result;

	if(area > 0.0f && !isinf(area)) {
		aabb->m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ) );
		aabb->m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ) );
	} else {
		aabb->invalidate();
	}
}

