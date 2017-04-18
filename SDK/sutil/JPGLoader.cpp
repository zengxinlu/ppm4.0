/*

*/

#include "HDRLoader.h"

#include <math.h>
#include <fstream>
#include <iostream>
#include <xstring>
#include <cstdlib>


#include "CxImage/ximage.h"

optix::TextureSampler loadTextureFromBytes(optix::Context context, void *bytes, int imageWidth, int imageHeight)
{
	// 	tex->LoadTexture2D(image->GetBits(), image->GetWidth(), image->GetHeight(), PF_RGB, PC_RGB, PCT_UNSIGNED_BYTE);
	// 	tex->SetFilterModes(LINEAR, LINEAR);
	// 	tex->SetWrapModes(REPEAT, REPEAT);

	// Create tex sampler and populate with default values
	optix::TextureSampler sampler = context->createTextureSampler();
	sampler->setWrapMode( 0, RT_WRAP_REPEAT );
	sampler->setWrapMode( 1, RT_WRAP_REPEAT );
	sampler->setWrapMode( 2, RT_WRAP_REPEAT );
	sampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
	sampler->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
	sampler->setMaxAnisotropy( 1.0f );
	sampler->setMipLevelCount( 1u );
	sampler->setArraySize( 1u );

	const unsigned int nx = imageWidth;
	const unsigned int ny = imageHeight;

	const char *m_raster = (const char *)bytes;

	// Create buffer and populate with PPM data
	optix::Buffer buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, nx, ny );
	unsigned char* buffer_data = static_cast<unsigned char*>( buffer->map() );
	for ( unsigned int i = 0; i < nx; ++i ) {
		for ( unsigned int j = 0; j < ny; ++j ) {

			unsigned int ppm_index = ( (ny-j-1)*nx + i )*3;
			unsigned int buf_index = ( (j     )*nx + i )*4;

			buffer_data[ buf_index + 0 ] = m_raster[ ppm_index + 0 ];
			buffer_data[ buf_index + 1 ] = m_raster[ ppm_index + 1 ];
			buffer_data[ buf_index + 2 ] = m_raster[ ppm_index + 2 ];
			// 			if (linearize_gamma) {
			// 				buffer_data[ buf_index + 0 ] = srgb2linear[buffer_data[ buf_index + 0 ]];
			// 				buffer_data[ buf_index + 1 ] = srgb2linear[buffer_data[ buf_index + 1 ]];
			// 				buffer_data[ buf_index + 2 ] = srgb2linear[buffer_data[ buf_index + 2 ]];
			// 			}
			buffer_data[ buf_index + 3 ] = 255;
		}
	}
	buffer->unmap();

	sampler->setBuffer( 0u, 0u, buffer );
	sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

	return sampler;
}

optix::TextureSampler loadJPGTexture( optix::Context context,
	const std::string& jpg_filename,
	const optix::float3& default_color )
{
	int mWidth, mHeight;

	char *bytes;

	const char *charStr = jpg_filename.c_str();
// 	int putLen = jpg_filename.length() + 1;
// 	wchar_t *tcharStr = new wchar_t[putLen];
// 	size_t getLen = 0;
// 	mbstowcs_s(&getLen, tcharStr, putLen, charStr, _TRUNCATE);
	CxImage* n_image = new CxImage(charStr, CXIMAGE_FORMAT_JPG);

	mWidth = n_image->GetWidth(), mHeight = n_image->GetHeight();

	bytes = new char[n_image->GetWidth() * n_image->GetHeight() * 3];
	for (int i = 0;i < mHeight;i ++)
		for (int j = 0;j < mWidth;j ++)
		{
			RGBQUAD tempRGB = n_image->GetPixelColor(j, mHeight - 1 - i, false);
			//RGBQUAD tempRGB = n_image->GetPixelColor(j, mHeight - 1 - i, false);
			bytes[(i*mWidth+j)*3] = tempRGB.rgbRed;
			bytes[(i*mWidth+j)*3 + 1] = tempRGB.rgbGreen;
			bytes[(i*mWidth+j)*3 + 2] = tempRGB.rgbBlue;
		}
		n_image->Destroy();

		return loadTextureFromBytes(context, bytes, mWidth, mHeight);
}
void saveTextureToImage(const std::string& image_filename, unsigned char *bytes, unsigned int mWidth, unsigned int mHeight)
{
	std::string bmp_filename = image_filename;
	bmp_filename.append(".bmp");
	wchar_t a;
	const char *charStr = bmp_filename.c_str();
// 	int putLen = bmp_filename.length() + 1;
// 	wchar_t *tcharStr = new wchar_t[putLen];
// 	wchar_t *m;
// 	size_t getLen = 0;
// 	mbstowcs_s(&getLen, tcharStr, putLen, charStr, _TRUNCATE);

	CxImage mImage;

	mImage.Create(mWidth, mHeight, 24);

	for (int i = 0;i < mWidth;i ++)
		for (int j = 0;j < mHeight;j ++)
		{
			RGBQUAD rgbquad;
			rgbquad.rgbRed = bytes[3*(mWidth*j + i)];
			rgbquad.rgbGreen = bytes[3*(mWidth*j + i) + 1];
			rgbquad.rgbBlue = bytes[3*(mWidth*j + i) + 2];

			mImage.SetPixelColor(i, j, rgbquad);
		}

		bool valid = mImage.IsValid();
		if (valid){
			mImage.Save(charStr, CXIMAGE_FORMAT_BMP);
		}
}