#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opengltool.h"
#include "GL/glew.h"
#include <assert.h>

void printLog(char* pTxtFileName, char *msg)
{	
	FILE* pTxtFile = fopen(pTxtFileName, "a");;
    if( pTxtFile == 0 )
        exit(0);
	fprintf(pTxtFile, "%s", msg);
    fclose(pTxtFile);
}

void grabImage()
{
}

#define BMP_Header_Length 54
void grab(int WindowWidth, int WindowHeight, char *mt, char *mm, int tt)
{
if (WindowWidth == 800) WindowWidth -= 8;
if (WindowHeight == 600) WindowHeight -= 8;
    FILE*    pDummyFile;
    FILE*    pWritingFile;
    GLubyte* pPixelData;
    GLubyte  BMP_Header[BMP_Header_Length];
    GLint    i, j;
    GLint    PixelDataLength;

    // 计算像素数据的实际长度
    i = WindowWidth * 3;   // 得到每一行的像素数据长度
	
    while( i%4 != 0 )      // 补充数据，直到i是的倍数
        ++i;               // 本来还有更快的算法，
                           // 但这里仅追求直观，对速度没有太高要求
						   
    PixelDataLength = i * WindowHeight;

    // 分配内存和打开文件
    pPixelData = (GLubyte*)malloc(PixelDataLength);
    if( pPixelData == 0 )
        exit(0);
	
	char temp[256] = "";
	char* lenbmp = ".bmp";
	char* lentxt = ".txt";
	strcpy(temp+strlen(temp), mt);
	strcpy(temp+strlen(temp), lenbmp);
    pDummyFile = fopen(temp, "rb");
	assert(pDummyFile != 0);
	
	memset(temp, 0, sizeof(temp));
	strcpy(temp+strlen(temp), mm);
	sprintf(temp+strlen(temp), "(%08dK)", tt);
	strcpy(temp+strlen(temp), lenbmp);

	pWritingFile = fopen(temp, "wb");
	assert(pWritingFile != 0);

    // 读取像素
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glReadPixels(0, 0, WindowWidth, WindowHeight,
        GL_BGR_EXT, GL_UNSIGNED_BYTE, pPixelData);

	
    // 把dummy.bmp的文件头复制为新文件的文件头
    fread(BMP_Header, sizeof(BMP_Header), 1, pDummyFile);
    fwrite(BMP_Header, sizeof(BMP_Header), 1, pWritingFile);
    fseek(pWritingFile, 0x0012, SEEK_SET);
    i = WindowWidth;
    j = WindowHeight;
    fwrite(&i, sizeof(i), 1, pWritingFile);
    fwrite(&j, sizeof(j), 1, pWritingFile);

    // 写入像素数据
    fseek(pWritingFile, 0, SEEK_END);
    fwrite(pPixelData, PixelDataLength, 1, pWritingFile);


    // 释放内存和关闭文件
    fclose(pDummyFile);
    fclose(pWritingFile);
    free(pPixelData);
}