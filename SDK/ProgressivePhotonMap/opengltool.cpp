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

    // �����������ݵ�ʵ�ʳ���
    i = WindowWidth * 3;   // �õ�ÿһ�е��������ݳ���
	
    while( i%4 != 0 )      // �������ݣ�ֱ��i�ǵı���
        ++i;               // �������и�����㷨��
                           // �������׷��ֱ�ۣ����ٶ�û��̫��Ҫ��
						   
    PixelDataLength = i * WindowHeight;

    // �����ڴ�ʹ��ļ�
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

    // ��ȡ����
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glReadPixels(0, 0, WindowWidth, WindowHeight,
        GL_BGR_EXT, GL_UNSIGNED_BYTE, pPixelData);

	
    // ��dummy.bmp���ļ�ͷ����Ϊ���ļ����ļ�ͷ
    fread(BMP_Header, sizeof(BMP_Header), 1, pDummyFile);
    fwrite(BMP_Header, sizeof(BMP_Header), 1, pWritingFile);
    fseek(pWritingFile, 0x0012, SEEK_SET);
    i = WindowWidth;
    j = WindowHeight;
    fwrite(&i, sizeof(i), 1, pWritingFile);
    fwrite(&j, sizeof(j), 1, pWritingFile);

    // д����������
    fseek(pWritingFile, 0, SEEK_END);
    fwrite(pPixelData, PixelDataLength, 1, pWritingFile);


    // �ͷ��ڴ�͹ر��ļ�
    fclose(pDummyFile);
    fclose(pWritingFile);
    free(pPixelData);
}