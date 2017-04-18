/*
*/

#pragma once

#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <string>
#include <iosfwd>

//-----------------------------------------------------------------------------
//
// Utility functions
//
//-----------------------------------------------------------------------------

// Creates a TextureSampler object for the given HDR file.  If filename is 
// empty or HDRLoader fails, a 1x1 texture is created with the provided default
// texture color.
SUTILAPI optix::TextureSampler loadJPGTexture( optix::Context context,
	const std::string& jpg_filename,
	const optix::float3& default_color );
SUTILAPI void saveTextureToImage(const std::string& jpg_filename, unsigned char *bytes, unsigned int mWidth, unsigned int mHeight);