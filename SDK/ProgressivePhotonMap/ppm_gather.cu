
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
rtBuffer<uint3, 2>               image_rnd_seeds;
rtDeclareVariable(float,         rtpass_default_radius2, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(float,         alpha, , );
rtDeclareVariable(float,         total_emitted, , );
rtDeclareVariable(float, frame_number, , );
rtDeclareVariable(float, direct_ratio, , );
rtDeclareVariable(float, indirect_ratio, , );
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

static __device__ const double chip2_90[] = {100, 2.70554, 4.60517, 6.25139, 7.77944, 9.23636, 10.64464, 12.01704, 13.36157, 14.68366, 15.98718, 17.27501, 18.54935, 19.81193, 21.06414, 22.30713, 23.54183, 24.76904, 25.98942, 27.20357, 28.41198, 29.61509, 30.81328, 32.00690, 33.19624, 34.38159, 35.56317, 36.74122, 37.91592, 39.08747, 40.25602, 41.42174, 42.58475, 43.74518, 44.90316, 46.05879, 47.21217, 48.36341, 49.51258, 50.65977, 51.80506, 52.94851, 54.09020, 55.23019, 56.36854, 57.50530, 58.64054, 59.77429, 60.90661, 62.03754};
static __device__ const double chip2_95[] = {100, 3.84146, 5.99146, 7.81473, 9.48773, 11.07050, 12.59159, 14.06714, 15.50731, 16.91898, 18.30704, 19.67514, 21.02607, 22.36203, 23.68479, 24.99579, 26.29623, 27.58711, 28.86930, 30.14353, 31.41043, 32.67057, 33.92444, 35.17246, 36.41503, 37.65248, 38.88514, 40.11327, 41.33714, 42.55697, 43.77297, 44.98534, 46.19426, 47.39988, 48.60237, 49.80185, 50.99846, 52.19232, 53.38354, 54.57223, 55.75848, 56.94239, 58.12404, 59.30351, 60.48089, 61.65623, 62.82962, 64.00111, 65.17077, 66.33865};
static __device__ const double chip2_99[] = {100, 6.63490, 9.21034, 11.34487, 13.27670, 15.08627, 16.81189, 18.47531, 20.09024, 21.66599, 23.20925, 24.72497, 26.21697, 27.68825, 29.14124, 30.57791, 31.99993, 33.40866, 34.80531, 36.19087, 37.56623, 38.93217, 40.28936, 41.63840, 42.97982, 44.31410, 45.64168, 46.96294, 48.27824, 49.58788, 50.89218, 52.19139, 53.48577, 54.77554, 56.06091, 57.34207, 58.61921, 59.89250, 61.16209, 62.42812, 63.69074, 64.95007, 66.20624, 67.45935, 68.70951, 69.95683, 71.20140, 72.44331, 73.68264, 74.91947};
static __device__ const double chip2_995[] = {100, 7.87944, 10.59663, 12.83816, 14.86026, 16.74960, 18.54758, 20.27774, 21.95495, 23.58935, 25.18818, 26.75685, 28.29952, 29.81947, 31.31935, 32.80132, 34.26719, 35.71847, 37.15645, 38.58226, 39.99685, 41.40106, 42.79565, 44.18128, 45.55851, 46.92789, 48.28988, 49.64492, 50.99338, 52.33562, 53.67196, 55.00270, 56.32811, 57.64845, 58.96393, 60.27477, 61.58118, 62.88334, 64.18141, 65.47557, 66.76596, 68.05273, 69.33600, 70.61590, 71.89255, 73.16606, 74.43654, 75.70407, 76.96877, 78.23071};
static __device__ const double chip2_9995[] = {100, 12.11567, 15.20180, 17.73000, 19.99735, 22.10533, 24.10280, 26.01777, 27.86805, 29.66581, 31.41981, 33.13661, 34.82127, 36.47779, 38.10940, 39.71876, 41.30807, 42.87921, 44.43377, 45.97312, 47.49845, 49.01081, 50.51112, 52.00019, 53.47875, 54.94746, 56.40689, 57.85759, 59.30003, 60.73465, 62.16185, 63.58201, 64.99546, 66.40251, 67.80346, 69.19856, 70.58807, 71.97222, 73.35123, 74.72529, 76.09460, 77.45934, 78.81966, 80.17573, 81.52769, 82.87569, 84.21985, 85.56030, 86.89715, 88.23052};

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

	uint *statistics = &rec.p0.x;
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
						int i = min(static_cast<int>(distance2 * 5 / rec_radius2), 4);
						int j = (static_cast<int>(photon.d.y / M_PI_4) + 8) % 8;
						++statistics[i * 8 + j];
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

float3 new_flux = rec_flux + flux_M;

	int v[5];
	int tot[5];
	bool chip2[5];
	double new_radius2 = rec_radius2;
	for (int i = 0; i < 5; ++i)
	{
		if (i == 0) tot[i] = 0;
		else tot[i] = tot[i - 1];
		for (int j = 0; j < 8; ++j)
			tot[i] += statistics[i * 8 + j];
		if (tot[i] > 20 * (i + 1) * 8)
		{
			v[i] = 0;
			double npi = (double)tot[i] / ((i + 1) * 8);
			for (int j = 0; j <= i; ++j)
				for (int k = 0; k < 8; ++k)
					v[i] += (statistics[j * 8 + k] - npi) * (statistics[j * 8 + k] - npi);
			v[i] /= npi;
			chip2[i] = v[i] <= chip2_99[(i + 1) * 8 - 1];
		}
		else chip2[i] = true;
	}
	
	image_rnd_seeds[launch_index].z = 0;
	if (!chip2[4])
	{
		image_rnd_seeds[launch_index].z = 1;
		for (int i = 3; i >= 0; --i)
			if (chip2[i] || v[i] / chip2_9995[(i + 1) * 8 - 1] <= v[4] / chip2_9995[(4 + 1) * 8 - 1])
			{
				new_radius2 = new_radius2 * ((i + 1.0) / 5.0);
				for (int j = 0; j < 40; ++j)
					statistics[j] = 0;
				break;
			}
		if (new_radius2 == rec_radius2)
		{
			new_radius2 = new_radius2 * ((1.0) / 5.0);
			for (int j = 0; j < 40; ++j)
				statistics[j] = 0;
		}
	}

	float new_Area = M_PI * new_radius2;
	new_flux = new_flux * (new_radius2 / rec_radius2);

	rec.c.z = new_radius2;
	rec.d = make_float4( new_flux );

	float3 indirect_flux = new_flux / total_emitted / new_Area;
	float3 direct_flux = direct_buffer[launch_index] / (frame_number + 1.0f);
	rtpass_output_buffer[launch_index] = rec;

	float3 final_color = direct_flux * direct_ratio + indirect_flux * indirect_ratio;
		
		//direct_flux + indirect_flux + ambient_light*rec_atten_Kd; 
		//(direct_flux + indirect_flux) * 2.f; 
		//direct_flux + indirect_flux * 8.f; 
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
	//	if(use_debug_buffer == 1)
	//	debug_buffer[launch_index] = make_float4( loop_iter, new_R2, new_N, M );
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
// 	// ����������
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