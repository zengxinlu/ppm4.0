
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
#include "helpers.h"
#include "path_tracer.h"
#include "random.h"

using namespace optix;

#define Total_Number 1

//
// Ray generation program
//

rtDeclareVariable(rtObject,      top_object, , );
rtBuffer<float4, 2>              output_buffer;
rtBuffer<float3, 2>              direct_buffer;
rtBuffer<float4, 2>              debug_buffer;
rtBuffer<float, 2>               primary_edge_buffer;
rtBuffer<float, 2>               secondary_edge_buffer;
rtDeclareVariable(float,         can_count_kernel, , );
rtBuffer<int4, 2>                sp_triangle_info_buffer;
rtBuffer<float3, 2>              sp_normal_buffer;
rtBuffer<float, 2>               sp_radius_buffer;
rtBuffer<float, 2>               sp_area_buffer;
rtBuffer<int>					 sp_valid_buffer;

rtDeclareVariable(float, max_radius2, , );
rtBuffer<PackedPhotonRecord, 1>  Global_Photon_Map;
rtBuffer<PackedHitRecord, 2>     rtpass_output_buffer;
rtBuffer<uint2, 2>               image_rnd_seeds;
rtDeclareVariable(float,         rtpass_default_radius2, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(float,         alpha, , );
rtDeclareVariable(float,         total_emitted, , );
rtDeclareVariable(float,         frame_number , , );
rtDeclareVariable(float3,        ambient_light , , );
rtDeclareVariable(uint,          use_debug_buffer, , );
rtDeclareVariable(PPMLight,      light , , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(ShadowPRD, shadow_prd, rtPayload, );


static __device__ __inline__ 
	void accumulatePhoton( const PackedPhotonRecord& photon,
	const float3& rec_normal,
	const float3& rec_atten_Kd,
	uint& num_new_photons, float3& flux_M , float L_K, float3& L_K_M)
{
	float3 photon_energy = make_float3( photon.c.y, photon.c.z, photon.c.w );
	float3 photon_normal = make_float3( photon.a.w, photon.b.x, photon.b.y );
	float p_dot_hit = dot(photon_normal, rec_normal);
	//if (p_dot_hit > 0.99001f) { // Fudge factor for imperfect cornell box geom
	if (p_dot_hit > 0.001f) 
	{ // Fudge factor for imperfect cornell box geom
		float3 photon_ray_dir = make_float3( photon.b.z, photon.b.w, photon.c.x );
		float3 flux = photon_energy * rec_atten_Kd;// * p_dot_hit; // * -dot(photon_ray_dir, rec_normal);
		num_new_photons++;
		flux_M += flux;
		//L_K_M += flux * L_K;
		L_K_M.x += 1.0f;
		L_K_M.y += 1.0f;
	}
	if (p_dot_hit < -0.001f);
	else
	{
		L_K_M.y += 1.0f;//abs(p_dot_hit);
	}
}

#if 0
#define check( condition, color ) \
{ \
	if( !(condition) ) { \
	debug_buffer[index] = make_float4( stack_current, node, Global_Photon_Map_size, 0 ); \
	output_buffer[index] = make_color( color ); \
	return; \
	} \
}
#else
#define check( condition, color )
#endif

#define MAX_DEPTH 20 // one MILLION photons

static __device__ inline float sqr(float a) { return a * a; }

RT_PROGRAM void globalDensity()
{
	PackedHitRecord rec = rtpass_output_buffer[launch_index];
	float3 rec_position = make_float3( rec.a.x, rec.a.y, rec.a.z );
	float3 rec_normal   = make_float3( rec.a.w, rec.b.x, rec.b.y );
	//float3 rec_normal   = sp_normal_buffer[launch_index];
	float3 rec_atten_Kd = make_float3( rec.b.z, rec.b.w, rec.c.x );
	uint   rec_flags    = __float_as_int( rec.c.y );
	float  rec_radius2  = rec.c.z;
	float  rec_photon_count = rec.c.w;
	float3 rec_flux     = make_float3( rec.d.x, rec.d.y, rec.d.z );
	float  rec_accum_atten = rec.d.w;

	// Check if this is hit point lies on an emitter or hit background 
	if( !(rec_flags & PPM_HIT) || rec_flags & PPM_OVERFLOW ) {
		output_buffer[launch_index] = make_float4(rec_atten_Kd);
		//output_buffer[launch_index] = make_float4(0, 0, 0, 0);
		return;
	}
	if (sp_radius_buffer[launch_index] < 0.0f)
	{
		float3 new_flux = rec_flux + make_float3(output_buffer[launch_index]);
		rec.d = make_float4( new_flux ); // set rec.flux
		return;
	}
	
	float real_area = M_PI * rec_radius2;
	int myron_ppm_valid = (sp_triangle_info_buffer[launch_index].x > 0)?1:0;
	uint2 cur_launch_index = make_uint2(sp_triangle_info_buffer[launch_index].x, sp_triangle_info_buffer[launch_index].y);

	int hitTriangleIndex = sp_triangle_info_buffer[launch_index].w;

	if (MYRON_PPM)
	{
		if (myron_ppm_valid)
		{
			//rec_normal = sp_normal_buffer[cur_launch_index];
			real_area = sp_area_buffer[cur_launch_index];
		}
		if (real_area < 0.f)
		{
			output_buffer[launch_index] = make_float4(rec_atten_Kd);
			return;
		}
	}

	unsigned int stack[MAX_DEPTH];
	unsigned int stack_current = 0;
	unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

	push_node( 0 );

	int Global_Photon_Map_size = Global_Photon_Map.size(); // for debugging

	uint num_new_photons = 0u;
	float3 flux_M = make_float3( 0.0f, 0.0f, 0.0f );
	float3 L_K_M = make_float3( 0.0f, 0.0f, 0.0f );
	size_t2 screen_size = output_buffer.size();
	uint loop_iter = 0;
	
	do {

		check( node < Global_Photon_Map_size, make_float3( 1,0,0 ) );
		PackedPhotonRecord& photon = Global_Photon_Map[ node ];

		uint axis = __float_as_int( photon.d.x );
		if( !( axis & PPM_NULL ) ) {

			float3 photon_position = make_float3( photon.a );
			float3 diff = rec_position - photon_position;
			float distance2 = dot(diff, diff);


			if (distance2 <= rec_radius2) {
				int tagindex = __float_as_int( photon.d.w );

				if (!myron_ppm_valid || (!MYRON_PPM))
					tagindex = -1;
				else
				{
					//int index_offset = (launch_index.y * screen_size.x + launch_index.x) * Myron_Valid_Size;
					int index_offset = (cur_launch_index.y * screen_size.x + cur_launch_index.x) * Myron_Valid_Size;

					int valid_count = 0;
					for (valid_count = 0;valid_count < Myron_Valid_Size;valid_count ++)
					{
						if (sp_valid_buffer[index_offset + valid_count] == -1)
							break;
						if (sp_valid_buffer[index_offset + valid_count] == tagindex)
					
						{
							tagindex = -1;
							break;
						}
					}
				}

				float3 pNormal = make_float3(photon.a.w, photon.b.x, photon.b.y);
				if ( dot(pNormal, rec_normal) < 0.001f )
					tagindex = 1;
				//if (tagindex < 0 && hitTriangleIndex == __float_as_int( photon.d.w ))
				if (tagindex < 0)
				{
					{	
						float t = sqrt(distance2/rec_radius2);
						float L_K = t*(-120*t*t + 180*t - 60);
						float3 temp_rec_Kd = rec_atten_Kd;

						// Kernel reduce strategy But there is black cloud!!!!!!!!!!!!!!!
						if (frame_number < Total_Number && MYRON_PPM)
						{
							int mod_number = int(Total_Number + 1 - frame_number);
							float pow_factor = pow(mod_number+1.0f, 0.01f);
							//float3 temp_rec_Kd = rec_atten_Kd * (1-pow(t, 1/pow_factor)) * (pow_factor);
							temp_rec_Kd *= (1-pow(t, 1/pow_factor)) * (1+pow_factor);
						}
						accumulatePhoton(photon, rec_normal, temp_rec_Kd, num_new_photons, flux_M, L_K, L_K_M);
					}

// 					if (tempFLoatArea == 1.0f)
// 					{
// 						mydebug[1] = mydebug[2] = 0;
// 					}
// 					else
// 					{
// 						mydebug[0] = mydebug[2] = 0;
// 					}
				}
			}

			// Recurse
			if( !( axis & PPM_LEAF ) ) {
				float d;
				if      ( axis & PPM_X ) d = diff.x;
				else if ( axis & PPM_Y ) d = diff.y;
				else                      d = diff.z;

				// Calculate the next child selector. 0 is left, 1 is right.
				int selector = d < 0.0f ? 0 : 1;
				if( d*d < rec_radius2 ) {
					check( stack_current+1 < MAX_DEPTH, make_float3( 0,1,0) );
					push_node( (node<<1) + 2 - selector );
				}

				check( stack_current+1 < MAX_DEPTH, make_float3( 0,1,1) );
				node = (node<<1) + 1 + selector;
			} else {
				node = pop_node();
			}
		} else {
			node = pop_node();
		}
		loop_iter++;
	} while ( node );

	// new alpha by mengyang
	float default_alpha = 0.7f, min_radius2 = 100.0f;
	//float default_alpha = 0.7f, min_radius2 = 100.0f;
	float m_alpha = default_alpha;

// 	if (myron_ppm_valid != 1)
// 	{
// 		m_alpha = 0.9f;
// 	}

// #define TempConstNumber 3
// 	if ( ((int)frame_number)%TempConstNumber != (TempConstNumber-1))
// 		m_alpha = 1.0f;

	// Compute new N,R
	float R2 = rec_radius2;
	float N = rec_photon_count;
	float M = static_cast<float>( num_new_photons ) ;
	float new_N = N + m_alpha*M;
	rec.c.w = new_N;  // set rec.photon_count

	float reduction_factor2 = 1.0f;
	float new_R2 = R2; 
	if( M != 0 ) {
		reduction_factor2 = ( N + m_alpha*M ) / ( N + M );
		new_R2 = R2*( reduction_factor2 );
		rec.c.z = new_R2; // set rec.radius2
		
		if (myron_ppm_valid)
			sp_area_buffer[cur_launch_index] = real_area * reduction_factor2;
	}

	// Compute indirectflux
	float3 new_flux;
	float repair_Area;
	
	new_flux = ( rec_flux + flux_M ) * reduction_factor2;
	rec.d = make_float4( new_flux ); // set rec.flux
	repair_Area = real_area;
	//repair_Area = M_PI * rec_radius2;

	float3 indirect_flux = 1.0f / repair_Area * new_flux / total_emitted;
	//float3 indirect_flux = 1.0f / repair_Area * flux_M / total_emitted * (frame_number + 1);
	
	float3 direct_flux = direct_buffer[launch_index]/(frame_number + 1.0f);
	rtpass_output_buffer[launch_index] = rec;
	float3 final_color = 
		//direct_flux + indirect_flux + ambient_light*rec_atten_Kd; 
		//(direct_flux + indirect_flux) * 2.f; 
		direct_flux + indirect_flux * 8.f; 
		//indirect_flux * 15.f;// * 5.f;
		//indirect_flux * 5.f;// * 5.f;
		//direct_flux * 1.f;
		//direct_flux * 30.f;

		//direct_flux*5.0f + indirect_flux * 0.2f; // sponza

		//direct_flux*15.0f + indirect_flux * 1.0f; // conference area reduce
		//direct_flux*0.8f + indirect_flux * 0.5f; // conference area light
		//direct_flux*0.5f + indirect_flux * 1.f; // conference area light
		//direct_flux*1.f + indirect_flux * 0.5f; // small_room
		//(direct_flux*1.f + indirect_flux * 30.0f) * 0.25f;				// box torus sibenik
		//(direct_flux*1.f + indirect_flux * 2.f); // sibnik

		//direct_flux*100.f;
		//direct_flux * 50.f + indirect_flux * 1.0f;
	
	//final_color = light.anchor;
	output_buffer[launch_index] = make_float4(final_color);
	if(use_debug_buffer == 1)
		debug_buffer[launch_index] = make_float4( loop_iter, new_R2, new_N, M );
}

RT_PROGRAM void gather_any_hit()
{
	shadow_prd.attenuation = 0.0f;

	rtTerminateRay();
}


//
// Stack overflow program
//
rtDeclareVariable(float3, rtpass_bad_color, , );
RT_PROGRAM void gather_exception()
{
	output_buffer[launch_index] = make_float4(0.0f, 1.0f, 1.0f, 0.0f);
}


rtBuffer<PackedPhotonRecord, 1>  Caustics_Photon_Map;
RT_PROGRAM void causticsDensity() {
	PackedHitRecord rec = rtpass_output_buffer[launch_index];
	float3 rec_position = make_float3( rec.a.x, rec.a.y, rec.a.z );
	float3 rec_normal   = make_float3( rec.a.w, rec.b.x, rec.b.y );
	float3 rec_atten_Kd = make_float3( rec.b.z, rec.b.w, rec.c.x );
	uint   rec_flags    = __float_as_int( rec.c.y );
	float  rec_radius2  = rec.c.z;
	float  rec_photon_count = rec.c.w;
	float3 rec_flux     = make_float3( rec.d.x, rec.d.y, rec.d.z );
	float  rec_accum_atten = rec.d.w;
	float  myron_count_area = 0, mydebug[3] = {1.0f, 1.0f, 1.0f};

	// Check if this is hit point lies on an emitter or hit background 
	if( !(rec_flags & PPM_HIT) || rec_flags & PPM_OVERFLOW ) {
		output_buffer[launch_index] = make_float4(rec_atten_Kd);
		return;
	}
	if (sp_radius_buffer[launch_index] < 0.0f)
	{
		float3 new_flux = rec_flux + make_float3(output_buffer[launch_index]);
		rec.d = make_float4( new_flux ); // set rec.flux
		return;
	}

	float real_area = M_PI * rec_radius2;
	int myron_ppm_valid = (sp_triangle_info_buffer[launch_index].x > 0)?1:0;
	uint2 cur_launch_index = make_uint2(sp_triangle_info_buffer[launch_index].x, sp_triangle_info_buffer[launch_index].y);

	int hitTriangleIndex = sp_triangle_info_buffer[launch_index].w;

	if (MYRON_PPM)
	{
		if (myron_ppm_valid)
		{
			//rec_normal = sp_normal_buffer[cur_launch_index];
			real_area = sp_area_buffer[cur_launch_index];
		}
		if (real_area < 0.f)
		{
			output_buffer[launch_index] = make_float4(rec_atten_Kd);
			return;
		}
	}

	unsigned int stack[MAX_DEPTH];
	unsigned int stack_current = 0;
	unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

	push_node( 0 );

	int Caustics_Photon_Map_size = Caustics_Photon_Map.size(); // for debugging

	uint num_new_photons = 0u;
	float3 flux_M = make_float3( 0.0f, 0.0f, 0.0f );
	float3 L_K_M = make_float3( 0.0f, 0.0f, 0.0f );
	size_t2 screen_size = output_buffer.size();
	uint loop_iter = 0;
	do {

		check( node < Caustics_Photon_Map_size, make_float3( 1,0,0 ) );
		PackedPhotonRecord& photon = Caustics_Photon_Map[ node ];

		uint axis = __float_as_int( photon.d.x );
		if( !( axis & PPM_NULL ) ) {

			float3 photon_position = make_float3( photon.a );
			float3 diff = rec_position - photon_position;
			float distance2 = dot(diff, diff);


			if (distance2 <= rec_radius2) {
				int tagindex = __float_as_int( photon.d.w );

				if (!myron_ppm_valid || (!MYRON_PPM))
					tagindex = -1;
				else
				{
					//int index_offset = (launch_index.y * screen_size.x + launch_index.x) * Myron_Valid_Size;
					int index_offset = (cur_launch_index.y * screen_size.x + cur_launch_index.x) * Myron_Valid_Size;

					int valid_count = 0;
					for (valid_count = 0;valid_count < Myron_Valid_Size;valid_count ++)
					{
						if (sp_valid_buffer[index_offset + valid_count] == -1)
							break;
						if (sp_valid_buffer[index_offset + valid_count] == tagindex)
					
						{
							tagindex = -1;
							break;
						}
					}
				}

				float3 pNormal = make_float3(photon.a.w, photon.b.x, photon.b.y);
				if ( dot(pNormal, rec_normal) < 0.001f )
					tagindex = 1;
				if (tagindex < 0)
				{
					float tempFLoatArea = __float_as_int( photon.d.z )/transferFloat;
					if (tempFLoatArea > 0.000001f)
					{	
						float t = sqrt(distance2/rec_radius2);
						float L_K = t*(-120*t*t + 180*t - 60);
						float3 temp_rec_Kd = rec_atten_Kd;

						// Kernel reduce strategy But there is black cloud!!!!!!!!!!!!!!!
						if (frame_number < Total_Number && MYRON_PPM)
						{
							int mod_number = int(Total_Number + 1 - frame_number);
							float pow_factor = pow(mod_number+1.0f, 0.01f);
							//float3 temp_rec_Kd = rec_atten_Kd * (1-pow(t, 1/pow_factor)) * (pow_factor);
							temp_rec_Kd *= (1-pow(t, 1/pow_factor)) * (1+pow_factor);
						}
						myron_count_area += tempFLoatArea;
						accumulatePhoton(photon, rec_normal, temp_rec_Kd, num_new_photons, flux_M, L_K, L_K_M);
					}

// 					if (tempFLoatArea == 1.0f)
// 					{
// 						mydebug[1] = mydebug[2] = 0;
// 					}
// 					else
// 					{
// 						mydebug[0] = mydebug[2] = 0;
// 					}
				}
			}

			// Recurse
			if( !( axis & PPM_LEAF ) ) {
				float d;
				if      ( axis & PPM_X ) d = diff.x;
				else if ( axis & PPM_Y ) d = diff.y;
				else                      d = diff.z;

				// Calculate the next child selector. 0 is left, 1 is right.
				int selector = d < 0.0f ? 0 : 1;
				if( d*d < rec_radius2 ) {
					check( stack_current+1 < MAX_DEPTH, make_float3( 0,1,0) );
					push_node( (node<<1) + 2 - selector );
				}

				check( stack_current+1 < MAX_DEPTH, make_float3( 0,1,1) );
				node = (node<<1) + 1 + selector;
			} else {
				node = pop_node();
			}
		} else {
			node = pop_node();
		}
		loop_iter++;
	} while ( node );

	// new alpha by mengyang
	float default_alpha = 0.7f, min_radius2 = 100.0f;
	//float default_alpha = 0.7f, min_radius2 = 100.0f;
	float m_alpha = default_alpha;

// 	if (myron_ppm_valid != 1)
// 	{
// 		m_alpha = 0.9f;
// 	}

// #define TempConstNumber 3
// 	if ( ((int)frame_number)%TempConstNumber != (TempConstNumber-1))
// 		m_alpha = 1.0f;


	// Compute new N,R
	float R2 = rec_radius2;
	float N = rec_photon_count;
	float M = static_cast<float>( num_new_photons ) ;
	float new_N = N + m_alpha*M;
	rec.c.w = new_N;  // set rec.photon_count

	float reduction_factor2 = 1.0f;
	float new_R2 = R2;
	if( M != 0 ) {
		reduction_factor2 = ( N + m_alpha*M ) / ( N + M );
		new_R2 = R2*( reduction_factor2 ); 
		rec.c.z = new_R2; // set rec.radius2
		
		if (myron_ppm_valid)
			sp_area_buffer[cur_launch_index] = real_area * reduction_factor2;
	}

	// Compute indirectflux
	float3 new_flux;
	float repair_Area;

	if (myron_count_area < 0.0000001f)
	{
		if ( num_new_photons > 0)
			mydebug[1] = mydebug[0] = mydebug[2] = 0;
		//mydebug[1] = mydebug[2] = 0;
	}
	else 
	{
		if (myron_count_area < real_area)
		{
			flux_M = flux_M * real_area / myron_count_area;
			//real_area = myron_count_area;
			//mydebug[0] = mydebug[2] = 0;
		}
	}

	new_flux = ( rec_flux + flux_M ) * reduction_factor2;
	rec.d = make_float4( new_flux ); // set rec.flux
	repair_Area = real_area;
	//repair_Area = M_PI * rec_radius2;

	float3 indirect_flux = 1.0f / repair_Area * new_flux / total_emitted;
	//float3 indirect_flux = 1.0f / repair_Area * flux_M / total_emitted * (frame_number + 1);
	
	float3 direct_flux = direct_buffer[launch_index]/(frame_number + 1.0f);
	rtpass_output_buffer[launch_index] = rec;
	float3 final_color = 
		//direct_flux + indirect_flux + ambient_light*rec_atten_Kd; 
		//indirect_flux*15.f;// * 5.f;
		indirect_flux*5.f;// * 5.f;
		//direct_flux * 1.f;
		//direct_flux * 30.f;

		//direct_flux*5.0f + indirect_flux * 0.2f; // sponza

		//direct_flux*15.0f + indirect_flux * 1.0f; // conference area reduce
		//direct_flux*0.8f + indirect_flux * 0.5f; // conference area light
		//direct_flux*0.5f + indirect_flux * 1.f; // conference area light
		//direct_flux*1.f + indirect_flux * 0.5f; // sibnik
		//(direct_flux*1.f + indirect_flux * 30.0f) * 0.25f; // box
		//(direct_flux*1.f + indirect_flux * 2.f); // sibnik

		//direct_flux*100.f;
		//direct_flux * 50.f + indirect_flux * 1.0f;
	final_color = make_float3(final_color.x*mydebug[0], final_color.y*mydebug[1], final_color.z*mydebug[2]);
// 	// 如果不成立
// 	if (myron_ppm_valid != 1)
// 	{
// 		final_color = make_float3(1.0f, 0, 0);
// 	}
// 	else
// 	{
// 		final_color = make_float3(real_area/(M_PI * rec_radius2));
// 	}
	//final_color = light.anchor;
	output_buffer[launch_index] = make_float4(final_color);
	if(use_debug_buffer == 1)
		debug_buffer[launch_index] = make_float4( loop_iter, new_R2, new_N, M );
}

RT_PROGRAM void causticsDensity_cornel() {
}

RT_PROGRAM void globalDensity_cornel()
{
	PackedHitRecord rec = rtpass_output_buffer[launch_index];
	float3 rec_position = make_float3( rec.a.x, rec.a.y, rec.a.z );
	float3 rec_normal   = make_float3( rec.a.w, rec.b.x, rec.b.y );
	//float3 rec_normal   = sp_normal_buffer[launch_index];
	float3 rec_atten_Kd = make_float3( rec.b.z, rec.b.w, rec.c.x );
	uint   rec_flags    = __float_as_int( rec.c.y );
	float  rec_radius2  = rec.c.z;
	float  rec_photon_count = rec.c.w;
	float3 rec_flux     = make_float3( rec.d.x, rec.d.y, rec.d.z );
	float  rec_accum_atten = rec.d.w;

	// Check if this is hit point lies on an emitter or hit background 
	if( !(rec_flags & PPM_HIT) || rec_flags & PPM_OVERFLOW ) {
		output_buffer[launch_index] = make_float4(rec_atten_Kd);
		//output_buffer[launch_index] = make_float4(0, 0, 0, 0);
		return;
	}
	if (sp_radius_buffer[launch_index] < 0.0f)
	{
		float3 new_flux = rec_flux + make_float3(output_buffer[launch_index]);
		rec.d = make_float4( new_flux ); // set rec.flux
		return;
	}
	
	float real_area = M_PI * rec_radius2;
	int myron_ppm_valid = (sp_triangle_info_buffer[launch_index].x > 0)?1:0;
	uint2 cur_launch_index = make_uint2(sp_triangle_info_buffer[launch_index].x, sp_triangle_info_buffer[launch_index].y);

	int hitTriangleIndex = sp_triangle_info_buffer[launch_index].w;

	if (MYRON_PPM)
	{
		if (myron_ppm_valid)
		{
			//rec_normal = sp_normal_buffer[cur_launch_index];
			real_area = sp_area_buffer[cur_launch_index];
		}
		if (real_area < 0.f)
		{
			output_buffer[launch_index] = make_float4(rec_atten_Kd);
			return;
		}
	}

	unsigned int stack[MAX_DEPTH];
	unsigned int stack_current = 0;
	unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

	push_node( 0 );

	int Global_Photon_Map_size = Global_Photon_Map.size(); // for debugging

	uint num_new_photons = 0u;
	float3 flux_M = make_float3( 0.0f, 0.0f, 0.0f );
	float3 L_K_M = make_float3( 0.0f, 0.0f, 0.0f );
	size_t2 screen_size = output_buffer.size();
	uint loop_iter = 0;
	
	do {

		check( node < Global_Photon_Map_size, make_float3( 1,0,0 ) );
		PackedPhotonRecord& photon = Global_Photon_Map[ node ];

		uint axis = __float_as_int( photon.d.x );
		if( !( axis & PPM_NULL ) ) {

			float3 photon_position = make_float3( photon.a );
			float3 diff = rec_position - photon_position;
			float distance2 = dot(diff, diff);


			if (distance2 <= rec_radius2) {
				int tagindex = __float_as_int( photon.d.w );

				if (!myron_ppm_valid || (!MYRON_PPM))
					tagindex = -1;
				else
				{
					//int index_offset = (launch_index.y * screen_size.x + launch_index.x) * Myron_Valid_Size;
					int index_offset = (cur_launch_index.y * screen_size.x + cur_launch_index.x) * Myron_Valid_Size;

					int valid_count = 0;
					for (valid_count = 0;valid_count < Myron_Valid_Size;valid_count ++)
					{
						if (sp_valid_buffer[index_offset + valid_count] == -1)
							break;
						if (sp_valid_buffer[index_offset + valid_count] == tagindex)
					
						{
							tagindex = -1;
							break;
						}
					}
				}

				float3 pNormal = make_float3(photon.a.w, photon.b.x, photon.b.y);
				if ( dot(pNormal, rec_normal) < 0.001f )
					tagindex = 1;
				//if (tagindex < 0 && hitTriangleIndex == __float_as_int( photon.d.w ))
				if (tagindex < 0)
				{
					{	
						float t = sqrt(distance2/rec_radius2);
						float L_K = t*(-120*t*t + 180*t - 60);
						float3 temp_rec_Kd = rec_atten_Kd;

						// Kernel reduce strategy But there is black cloud!!!!!!!!!!!!!!!
						if (frame_number < Total_Number && MYRON_PPM)
						{
							int mod_number = int(Total_Number + 1 - frame_number);
							float pow_factor = pow(mod_number+1.0f, 0.01f);
							//float3 temp_rec_Kd = rec_atten_Kd * (1-pow(t, 1/pow_factor)) * (pow_factor);
							temp_rec_Kd *= (1-pow(t, 1/pow_factor)) * (1+pow_factor);
						}
						accumulatePhoton(photon, rec_normal, temp_rec_Kd, num_new_photons, flux_M, L_K, L_K_M);
					}

// 					if (tempFLoatArea == 1.0f)
// 					{
// 						mydebug[1] = mydebug[2] = 0;
// 					}
// 					else
// 					{
// 						mydebug[0] = mydebug[2] = 0;
// 					}
				}
			}

			// Recurse
			if( !( axis & PPM_LEAF ) ) {
				float d;
				if      ( axis & PPM_X ) d = diff.x;
				else if ( axis & PPM_Y ) d = diff.y;
				else                      d = diff.z;

				// Calculate the next child selector. 0 is left, 1 is right.
				int selector = d < 0.0f ? 0 : 1;
				if( d*d < rec_radius2 ) {
					check( stack_current+1 < MAX_DEPTH, make_float3( 0,1,0) );
					push_node( (node<<1) + 2 - selector );
				}

				check( stack_current+1 < MAX_DEPTH, make_float3( 0,1,1) );
				node = (node<<1) + 1 + selector;
			} else {
				node = pop_node();
			}
		} else {
			node = pop_node();
		}
		loop_iter++;
	} while ( node );

	// new alpha by mengyang
	float default_alpha = 0.7f, min_radius2 = 100.0f;
	//float default_alpha = 0.7f, min_radius2 = 100.0f;
	float m_alpha = default_alpha;

// 	if (myron_ppm_valid != 1)
// 	{
// 		m_alpha = 0.9f;
// 	}

// #define TempConstNumber 3
// 	if ( ((int)frame_number)%TempConstNumber != (TempConstNumber-1))
// 		m_alpha = 1.0f;

	// Compute new N,R
	float R2 = rec_radius2;
	float N = rec_photon_count;
	float M = static_cast<float>( num_new_photons ) ;
	float new_N = N + m_alpha*M;
	rec.c.w = new_N;  // set rec.photon_count

	float reduction_factor2 = 1.0f;
	float new_R2 = R2; 
	if( M != 0 ) {
		reduction_factor2 = ( N + m_alpha*M ) / ( N + M );
		new_R2 = R2*( reduction_factor2 );
		rec.c.z = new_R2; // set rec.radius2
		
		if (myron_ppm_valid)
			sp_area_buffer[cur_launch_index] = real_area * reduction_factor2;
	}

	// Compute indirectflux
	float3 new_flux;
	float repair_Area;
	
	new_flux = ( rec_flux + flux_M ) * reduction_factor2;
	rec.d = make_float4( new_flux ); // set rec.flux
	repair_Area = real_area;
	//repair_Area = M_PI * rec_radius2;

	float3 indirect_flux = 1.0f / repair_Area * new_flux / total_emitted;
	//float3 indirect_flux = 1.0f / repair_Area * flux_M / total_emitted * (frame_number + 1);
	
	float3 direct_flux = direct_buffer[launch_index]/(frame_number + 1.0f);
	rtpass_output_buffer[launch_index] = rec;
	float3 final_color = 
		//direct_flux + indirect_flux + ambient_light*rec_atten_Kd; 
		//indirect_flux*15.f;// * 5.f;
		indirect_flux * 5.f;// * 5.f;
		//direct_flux * 1.f;
		//direct_flux * 30.f;

		//direct_flux*5.0f + indirect_flux * 0.2f; // sponza

		//direct_flux*15.0f + indirect_flux * 1.0f; // conference area reduce
		//direct_flux*0.8f + indirect_flux * 0.5f; // conference area light
		//direct_flux*0.5f + indirect_flux * 1.f; // conference area light
		//direct_flux*1.f + indirect_flux * 0.5f; // sibnik
		//(direct_flux*1.f + indirect_flux * 30.0f) * 0.25f; // box
		//(direct_flux*1.f + indirect_flux * 2.f); // sibnik

		//direct_flux*100.f;
		//direct_flux * 50.f + indirect_flux * 1.0f;
	
	//final_color = light.anchor;
	output_buffer[launch_index] = make_float4(final_color);
	if(use_debug_buffer == 1)
		debug_buffer[launch_index] = make_float4( loop_iter, new_R2, new_N, M );
}
