
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

#include "ImageLoader.h"
#include "PPMLoader.h"
#include "HDRLoader.h"
#include "JPGLoader.h"
#include <fstream>


//-----------------------------------------------------------------------------
//  
//  Utility functions 
//
//-----------------------------------------------------------------------------

optix::TextureSampler loadTexture( optix::Context context,
	const std::string& filename,
	const optix::float3& default_color )
{
	bool IsHDR = false;
	bool IsJPG = false;
	size_t len = filename.length();
	if(len >= 3) {
		IsHDR = (filename[len-3] == 'H' || filename[len-3] == 'h') &&
			(filename[len-2] == 'D' || filename[len-2] == 'd') &&
			(filename[len-1] == 'R' || filename[len-1] == 'r');
		IsJPG = (filename[len-3] == 'J' || filename[len-3] == 'j') &&
			(filename[len-2] == 'P' || filename[len-2] == 'p') &&
			(filename[len-1] == 'G' || filename[len-1] == 'g');
	}
	if(IsHDR)
		return loadHDRTexture(context, filename, default_color);
	else if (IsJPG)
		return loadJPGTexture(context, filename, default_color);
	else
		return loadPPMTexture(context, filename, default_color);
}

/*
optix::TextureSampler loadTexture( optix::Context m_Context,
	CxImage *
	const optix::float3& default_color )
{
	// Create tex sampler and populate with default values
	optix::TextureSampler sampler = m_Context->createTextureSampler();
	sampler->setWrapMode( 0, RT_WRAP_REPEAT );
	sampler->setWrapMode( 1, RT_WRAP_REPEAT );
	sampler->setWrapMode( 2, RT_WRAP_REPEAT );
	sampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
	sampler->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
	sampler->setMaxAnisotropy( 1.0f );
	sampler->setMipLevelCount( 1u );
	sampler->setArraySize( 1u );

	if ( false ) {

		// Create buffer with single texel set to default_color
		optix::Buffer buffer = m_Context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1u, 1u );
		float* buffer_data = static_cast<float*>( buffer->map() );
		buffer_data[0] = default_color.x;
		buffer_data[1] = default_color.y;
		buffer_data[2] = default_color.z;
		buffer_data[3] = 1.0f;
		buffer->unmap();

		sampler->setBuffer( 0u, 0u, buffer );
		// Although it would be possible to use nearest filtering here, we chose linear
		// to be consistent with the textures that have been loaded from a file. This
		// allows OptiX to perform some optimizations.
		sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

		return sampler;
	}

	const unsigned int nx = hdr.width();
	const unsigned int ny = hdr.height();

	// Create buffer and populate with HDR data
	optix::Buffer buffer = m_Context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny );
	float* buffer_data = static_cast<float*>( buffer->map() );

	for ( unsigned int i = 0; i < nx; ++i ) {
		for ( unsigned int j = 0; j < ny; ++j ) {

			unsigned int hdr_index = ( (ny-j-1)*nx + i )*4;
			unsigned int buf_index = ( (j     )*nx + i )*4;

			buffer_data[ buf_index + 0 ] = hdr.raster()[ hdr_index + 0 ];
			buffer_data[ buf_index + 1 ] = hdr.raster()[ hdr_index + 1 ];
			buffer_data[ buf_index + 2 ] = hdr.raster()[ hdr_index + 2 ];
			buffer_data[ buf_index + 3 ] = hdr.raster()[ hdr_index + 3 ];
		}
	}

	buffer->unmap();

	sampler->setBuffer( 0u, 0u, buffer );
	sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

	return sampler;
}
*/
