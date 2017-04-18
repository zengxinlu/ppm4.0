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

#define MAX_DEPTH 20

//
// Scene wide variables
//
rtDeclareVariable(rtObject,      top_object, , );

rtBuffer<PackedPhotonRecord, 1>  Caustics_Photon_Map;
rtDeclareVariable(uint,          Caustics_Photon_Map_Size, , );
rtDeclareVariable(float,         caustics_coeff, , );
rtDeclareVariable(float,  CausticsRadius2, , );
rtDeclareVariable(float,  GlobalRadius2, , );

rtBuffer<PackedHitRecord, 1>     GlobalHitRecord_Buffer;
rtBuffer<PackedHitRecord, 1>     CausticsHitRecord_Buffer;

rtDeclareVariable(uint, UsePhoton, , );

rtDeclareVariable(PPMLight,      light , , );

rtBuffer<float4, 2>              Frame_Buffer;
rtBuffer<float4, 2>              Output_Buffer;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(float,			 FrameCount, , );

static __host__ __device__ __inline__ void addFrameBuffer()
{
	float4 tempOutput = Output_Buffer[launch_index];
	tempOutput = tempOutput*(FrameCount-1)/FrameCount + Frame_Buffer[launch_index]/FrameCount;
	Output_Buffer[launch_index] = fminf(tempOutput, make_float4(1.f));	 
}
static __device__ __inline__ 
	void accumulateCausticsPhotonRadius2( const PackedPhotonRecord& photon,
	const float3& rec_normal,
	const float3& rec_atten_Kd,
	uint& num_new_photons, float3& flux_M , float radius_atten)
{
	float3 photon_energy = make_float3( photon.c.y, photon.c.z, photon.c.w );
	float3 photon_normal = make_float3( photon.a.w, photon.b.x, photon.b.y );
	float3 ray_dir = make_float3( photon.b.z, photon.b.w, photon.c.x );
	float p_dot_hit = dot(photon_normal, rec_normal);
	float m_dot_hit = -dot(ray_dir, rec_normal);
 	//if (m_dot_hit > 0.f)
 	//if (p_dot_hit > 0.f) 
	{ // Fudge factor for imperfect cornell box geom
		float3 flux = photon_energy * rec_atten_Kd;// * fabsf(p_dot_hit*m_dot_hit);
		num_new_photons++;
		flux_M += flux;
	}
}
static __device__ __inline__ 
	void accumulateGlobalPhotonRadius2( const PackedPhotonRecord& photon,
	const float3& rec_normal,
	const float3& rec_atten_Kd,
	uint& num_new_photons, float3& flux_M , float radius_atten)
{
	float3 photon_energy = make_float3( photon.c.y, photon.c.z, photon.c.w );
	float3 photon_normal = make_float3( photon.a.w, photon.b.x, photon.b.y );
	float3 ray_dir = make_float3( photon.b.z, photon.b.w, photon.c.x );
	float p_dot_hit = dot(photon_normal, rec_normal);
	float m_dot_hit = -dot(ray_dir, rec_normal);
 	if (m_dot_hit > 0.f)
 	if (p_dot_hit > 0.f) 
	{ // Fudge factor for imperfect cornell box geom
		float3 flux = photon_energy * rec_atten_Kd;// * (p_dot_hit*m_dot_hit) * radius_atten;
		num_new_photons++;
		flux_M += flux;
	}
}
//
// Ray generation program
//
RT_PROGRAM void causticsDensity ()
{
	//for Caustics photon
	int currentRecord = (launch_index.y * Frame_Buffer.size().x + launch_index.x);
	float3 final_color = make_float3(Frame_Buffer[launch_index]);
	while (1)
	{
		PackedHitRecord &rec = CausticsHitRecord_Buffer[currentRecord];
		float3 rec_position = make_float3( rec.a.x, rec.a.y, rec.a.z );
		float3 rec_normal   = make_float3( rec.a.w, rec.b.x, rec.b.y );
		float3 rec_atten_Kd = make_float3( rec.b.z, rec.b.w, rec.c.x );
		uint   rec_flags    = __float_as_int( rec.c.y );
		float  rec_radius2  = rec.c.z;
		float  rec_photon_count = rec.c.w;
		float3 rec_flux = make_float3( rec.d.x, rec.d.y, rec.d.z );
		float  max_radius2 = 0.f;
		// Check if this is hit point lies on an emitter or hit background 
		if( !(rec_flags & PPM_HIT) || rec_flags & PPM_OVERFLOW)
			break;
		unsigned int stack[MAX_DEPTH];
		unsigned int stack_current = 0;
		unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

		push_node( 0 );

		int photon_map_size = Caustics_Photon_Map_Size; // for debugging

		uint num_new_photons = 0u;
		float3 flux_M = make_float3( 0.0f, 0.0f, 0.0f );
		uint loop_iter = 0;
		do {

			PackedPhotonRecord& photon = Caustics_Photon_Map[ node ];

			uint axis = __float_as_int( photon.d.x );
			if( !( axis & PPM_NULL ) ) {

				float3 photon_position = make_float3( photon.a );
				float3 diff = rec_position - photon_position;
				float distance2 = dot(diff, diff);

				if (distance2 < rec_radius2) {
					accumulateCausticsPhotonRadius2(photon, rec_normal, rec_atten_Kd, num_new_photons, flux_M, 
						(rec_radius2-distance2)/rec_radius2);
					if (distance2 > max_radius2)
						max_radius2 = distance2;
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
						push_node( (node<<1) + 2 - selector );
					}

					node = (node<<1) + 1 + selector;
				} else {
					node = pop_node();
				}
			} else {
				node = pop_node();
			}
			loop_iter++;
		} while ( node );
		if (num_new_photons > 0)
		{
			// New photon count
			rec.c.w = rec_photon_count + num_new_photons;
			// New radius2
			float new_radius2 = rec_radius2 * (0.9f*num_new_photons + rec_photon_count)/(num_new_photons + rec_photon_count);
			rec.c.z = new_radius2;
			// New flux
			float3 indirect_flux = (rec_flux + flux_M) * new_radius2 / rec_radius2;
			rec.d = make_float4( indirect_flux );
			// Final color
			final_color += indirect_flux * caustics_coeff / (M_PI * new_radius2) / photon_map_size / FrameCount;
		}
		break;
	}
	Frame_Buffer[launch_index] = fminf(make_float4(final_color), make_float4(1.f));
}
//
// Ray generation program
//
rtBuffer<PackedPhotonRecord, 1>  Global_Photon_Map;
rtDeclareVariable(uint,          Global_Photon_Map_Size, , );
rtDeclareVariable(float,         global_coeff, , );
rtDeclareVariable(float,  Alpha, , );
RT_PROGRAM void globalDensity()
{
	//for Caustics photon
	int currentRecord = (launch_index.y * Frame_Buffer.size().x + launch_index.x);
	float3 final_color = make_float3(Frame_Buffer[launch_index]);
	while (1)
	{
		PackedHitRecord &rec = GlobalHitRecord_Buffer[currentRecord];
		float3 rec_position = make_float3( rec.a.x, rec.a.y, rec.a.z );
		float3 rec_normal   = make_float3( rec.a.w, rec.b.x, rec.b.y );
		float3 rec_atten_Kd = make_float3( rec.b.z, rec.b.w, rec.c.x );
		uint   rec_flags    = __float_as_int( rec.c.y );
		float  rec_radius2  = rec.c.z;
		float  rec_photon_count = rec.c.w;
		float3 rec_flux = make_float3( rec.d.x, rec.d.y, rec.d.z );
		// Check if this is hit point lies on an emitter or hit background 
		if( !(rec_flags & PPM_HIT) || rec_flags & PPM_OVERFLOW)
			break;
		unsigned int stack[MAX_DEPTH];
		unsigned int stack_current = 0;
		unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

		push_node( 0 );

		int photon_map_size = Global_Photon_Map_Size; // for debugging

		uint num_new_photons = 0u;
		float3 flux_M = make_float3( 0.0f, 0.0f, 0.0f );
		uint loop_iter = 0;
		do {

			PackedPhotonRecord& photon = Global_Photon_Map[ node ];

			uint axis = __float_as_int( photon.d.x );
			if( !( axis & PPM_NULL ) ) {

				float3 photon_position = make_float3( photon.a );
				float3 diff = rec_position - photon_position;
				float distance2 = dot(diff, diff);

				if (distance2 < rec_radius2) {
					accumulateGlobalPhotonRadius2(photon, rec_normal, rec_atten_Kd, num_new_photons, flux_M, 
						(rec_radius2-distance2)/rec_radius2);
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
						push_node( (node<<1) + 2 - selector );
					}

					node = (node<<1) + 1 + selector;
				} else {
					node = pop_node();
				}
			} else {
				node = pop_node();
			}
			loop_iter++;
		} while ( node );
		if (num_new_photons > 0)
		{
			//final_color = make_float3(1.f, 0.f, 1.f);
			//break;
			// New photon count
			rec.c.w = rec_photon_count + num_new_photons;
			// New radius2
			float new_radius2 = rec_radius2 * (0.9f*num_new_photons + rec_photon_count)/(num_new_photons + rec_photon_count);
			rec.c.z = new_radius2;
			// New flux
			float3 indirect_flux = (rec_flux + flux_M) * new_radius2 / rec_radius2;
			rec.d = make_float4( indirect_flux );
			// Final color
			final_color += 
				//indirect_flux * CausticsRadius2 / new_radius2 / (rec_photon_count + num_new_photons);
				//indirect_flux * global_coeff / (M_PI * new_radius2) / photon_map_size / FrameCount;
				indirect_flux * global_coeff / (M_PI * rec_radius2) / (rec_photon_count + num_new_photons);
		}
		break;
	}
	Frame_Buffer[launch_index] = fminf(make_float4(final_color), make_float4(1.f));
	// Add to Output
	addFrameBuffer();
}


RT_PROGRAM void gather_exception()
{
	Frame_Buffer[launch_index] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
}