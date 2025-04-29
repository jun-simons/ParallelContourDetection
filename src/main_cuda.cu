#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <png.h>
#include <cuda_runtime.h>
#include "clockcycle.h"

typedef struct
{
    int width;
    int height;
    unsigned char *data;
} RGBImage;

typedef struct
{
    int width;
    int height;
    unsigned char *data;
} GrayImage;

RGBImage readPNG(const char *filename)
{
    RGBImage img = {0};
    FILE *fp = fopen(filename, "rb");

    unsigned char header[8];
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
    {
        printf("Error: %s is not a PNG file\n", filename);
        fclose(fp);
        return img;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    png_infop info_ptr = png_create_info_struct(png_ptr);

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("Error during PNG file reading\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return img;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    img.width = png_get_image_width(png_ptr, info_ptr);
    img.height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    if (bit_depth == 16)
    {
        png_set_strip_16(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_palette_to_rgb(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    {
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    }
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
    {
        png_set_tRNS_to_alpha(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    {
        png_set_gray_to_rgb(png_ptr);
    }

    png_read_update_info(png_ptr, info_ptr);

    img.data = (unsigned char *)malloc(img.width * img.height * 3);

    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * img.height);
    for (int y = 0; y < img.height; y++)
    {
        row_pointers[y] = img.data + y * img.width * 3;
    }
    png_read_image(png_ptr, row_pointers);
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

    return img;
}

void writeGrayPNG(const char *filename, const GrayImage *img)
{
    FILE *fp = fopen(filename, "wb");

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("Error during PNG file writing\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return;
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, img->width, img->height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * img->height);
    for (int y = 0; y < img->height; y++)
    {
        row_pointers[y] = img->data + y * img->width;
    }

    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);
    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void rgbToBinary(const RGBImage *img, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    fwrite(&img->width, sizeof(int), 1, fp);
    fwrite(&img->height, sizeof(int), 1, fp);
    fwrite(img->data, sizeof(unsigned char), img->width * img->height * 3, fp);
    fclose(fp);
}

RGBImage binaryToRGB(const char *filename)
{
    RGBImage img = {0};
    FILE *fp = fopen(filename, "rb");

    fread(&img.width, sizeof(int), 1, fp);
    fread(&img.height, sizeof(int), 1, fp);

    img.data = (unsigned char *)malloc(img.width * img.height * 3);

    fread(img.data, sizeof(unsigned char), img.width * img.height * 3, fp);

    fclose(fp);
    return img;
}

GrayImage binaryToGray(const char *filename)
{
    GrayImage img = {0};
    FILE *fp = fopen(filename, "rb");

    fread(&img.width, sizeof(int), 1, fp);
    fread(&img.height, sizeof(int), 1, fp);

    img.data = (unsigned char *)malloc(img.width * img.height);
    fread(img.data, sizeof(unsigned char), img.width * img.height, fp);
    fclose(fp);
    return img;
}

typedef struct
{
    int width;
    int height;
    float *magnitude;
    float *direction;
} GradientImage;

__global__ void rgbToGrayscaleKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        int rgbOffset = (y * width + x) * 3;
        int grayOffset = y * width + x;
        unsigned char r = d_input[rgbOffset];
        unsigned char g = d_input[rgbOffset + 1];
        unsigned char b = d_input[rgbOffset + 2];

        d_output[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

__global__ void gaussianBlurKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        const float kernel[5][5] = {
            {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
            {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
            {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
            {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
            {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}};

        float sum = 0.0f;
        for (int ky = -2; ky <= 2; ky++)
        {
            for (int kx = -2; kx <= 2; kx++)
            {
                int ix = x + kx;
                int iy = y + ky;

                ix = max(0, min(width - 1, ix));
                iy = max(0, min(height - 1, iy));

                int inputOffset = iy * width + ix;
                sum += kernel[ky + 2][kx + 2] * d_input[inputOffset];
            }
        }

        d_output[y * width + x] = (unsigned char)sum;
    }
}

__global__ void sobelXKernel(unsigned char *d_input, float *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        const int kernel[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}};

        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                int ix = x + kx;
                int iy = y + ky;

                ix = max(0, min(width - 1, ix));
                iy = max(0, min(height - 1, iy));

                int inputOffset = iy * width + ix;
                sum += kernel[ky + 1][kx + 1] * d_input[inputOffset];
            }
        }

        d_output[y * width + x] = sum;
    }
}

__global__ void sobelYKernel(unsigned char *d_input, float *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        const int kernel[3][3] = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}};

        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                int ix = x + kx;
                int iy = y + ky;

                ix = max(0, min(width - 1, ix));
                iy = max(0, min(height - 1, iy));

                int inputOffset = iy * width + ix;
                sum += kernel[ky + 1][kx + 1] * d_input[inputOffset];
            }
        }

        d_output[y * width + x] = sum;
    }
}

__global__ void gradientKernel(float *d_gradient_x, float *d_gradient_y, float *d_magnitude, float *d_direction, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        int idx = y * width + x;
        float gx = d_gradient_x[idx];
        float gy = d_gradient_y[idx];

        d_magnitude[idx] = sqrtf(gx * gx + gy * gy);

        d_direction[idx] = atan2f(gy, gx);
    }
}

__global__ void nonMaxSuppressionKernel(float *d_magnitude, float *d_direction, unsigned char *d_output, int width, int height, int startRow, int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (x < width && y < endRow && y >= startRow)
    {
        int idx = y * width + x;
        float mag = d_magnitude[idx];
        float dir = d_direction[idx];

        float dirDegrees = dir * 180.0f / 3.14159265358979323846f;
        if (dirDegrees < 0)
            dirDegrees += 180.0f;

        int angle = ((int)(dirDegrees + 22.5f) % 180) / 45;

        int nx1, ny1, nx2, ny2;

        switch (angle)
        {
        case 0: // 0
            nx1 = -1;
            ny1 = 0;
            nx2 = 1;
            ny2 = 0;
            break;
        case 1: // 45
            nx1 = -1;
            ny1 = -1;
            nx2 = 1;
            ny2 = 1;
            break;
        case 2: // 90
            nx1 = 0;
            ny1 = -1;
            nx2 = 0;
            ny2 = 1;
            break;
        case 3: // 135
            nx1 = -1;
            ny1 = 1;
            nx2 = 1;
            ny2 = -1;
            break;
        default:
            nx1 = 0;
            ny1 = 0;
            nx2 = 0;
            ny2 = 0;
        }

        int x1 = min(width - 1, max(0, x + nx1));
        int y1 = min(height - 1, max(0, y + ny1));
        int x2 = min(width - 1, max(0, x + nx2));
        int y2 = min(height - 1, max(0, y + ny2));

        float mag1 = d_magnitude[y1 * width + x1];
        float mag2 = d_magnitude[y2 * width + x2];

        if (mag >= mag1 && mag >= mag2)
        {
            const float highThreshold = 50.0f;
            if (mag > highThreshold)
            {
                d_output[idx] = 255;
            }
            else
            {
                d_output[idx] = 0;
            }
        }
        else
        {
            d_output[idx] = 0;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("arguments");
        return 1;
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *inputPNG = argv[1];
    int totalImages = atoi(argv[2]);

    RGBImage rgbImg = {0};
    if (rank == 0)
    {
        printf("Rank 0: reading %s\n", inputPNG);
        rgbImg = readPNG(inputPNG);
    }

    int width = rgbImg.width, height = rgbImg.height;
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    size_t rgbBytes = width * height * 3;
    if (rank != 0)
        rgbImg.data = (unsigned char *)malloc(rgbBytes);
    MPI_Bcast(rgbImg.data, rgbBytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int baseJobs = totalImages / size;
    int remainder = totalImages % size;
    int myJobs = baseJobs + (rank < remainder ? 1 : 0);

    if (myJobs == 0)
    {
        MPI_Finalize();
        return 0;
    }

    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    cudaSetDevice(rank % devCount);

    size_t grayBytes = width * height;
    size_t floatBytes = grayBytes * sizeof(float);

    GrayImage grayImg = {width, height, (unsigned char *)malloc(grayBytes)};
    GrayImage blurredImg = {width, height, (unsigned char *)malloc(grayBytes)};
    GrayImage edgesImg = {width, height, (unsigned char *)malloc(grayBytes)};

    GradientImage gradImg;
    gradImg.width = width;
    gradImg.height = height;
    gradImg.magnitude = (float *)malloc(floatBytes);
    gradImg.direction = (float *)malloc(floatBytes);

    unsigned char *d_rgb, *d_gray, *d_blur, *d_edges;
    float *d_gx, *d_gy, *d_mag, *d_dir;
    cudaMalloc(&d_rgb, rgbBytes);
    cudaMalloc(&d_gray, grayBytes);
    cudaMalloc(&d_blur, grayBytes);
    cudaMalloc(&d_edges, grayBytes);
    cudaMalloc(&d_gx, floatBytes);
    cudaMalloc(&d_gy, floatBytes);
    cudaMalloc(&d_mag, floatBytes);
    cudaMalloc(&d_dir, floatBytes);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);

    uint64_t t0 = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        t0 = clock_now();
    }

    for (int job = 0; job < myJobs; ++job)
    {
        cudaMemcpy(d_rgb, rgbImg.data, rgbBytes, cudaMemcpyHostToDevice);

        rgbToGrayscaleKernel<<<gridDim, blockDim>>>(d_rgb, d_gray, width, height, 0, height);
        cudaDeviceSynchronize();
        gaussianBlurKernel<<<gridDim, blockDim>>>(d_gray, d_blur, width, height, 0, height);
        cudaDeviceSynchronize();
        sobelXKernel<<<gridDim, blockDim>>>(d_blur, d_gx, width, height, 0, height);
        cudaDeviceSynchronize();
        sobelYKernel<<<gridDim, blockDim>>>(d_blur, d_gy, width, height, 0, height);
        cudaDeviceSynchronize();
        gradientKernel<<<gridDim, blockDim>>>(d_gx, d_gy, d_mag, d_dir, width, height, 0, height);
        cudaDeviceSynchronize();
        nonMaxSuppressionKernel<<<gridDim, blockDim>>>(d_mag, d_dir, d_edges, width, height, 0, height);
        cudaDeviceSynchronize();

        if (job == 0)
        {
            cudaMemcpy(edgesImg.data, d_edges, grayBytes, cudaMemcpyDeviceToHost);
            char fname[128];
            sprintf(fname, "output_rank%d.png", rank);
            writeGrayPNG(fname, &edgesImg);
            printf("Rank %d: wrote %s\n", rank, fname);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        uint64_t t1 = clock_now();
        printf("Total cycles (all ranks finished): %lu\n", t1 - t0);
    }

    t0 = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        t0 = clock_now();
    }

    MPI_File fh;
    MPI_Status mpistatus;
    const char *binName = "results.bin";

    MPI_File_open(MPI_COMM_WORLD, binName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    if (rank == 0)
    {
        int hdr[3] = {width, height, totalImages};
        MPI_File_write(fh, hdr, 3, MPI_INT, &mpistatus);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int myFirstIdx = rank * baseJobs + (rank < remainder ? rank : remainder);

    MPI_Offset imgBytes = (MPI_Offset)width * height;
    MPI_Offset headerSize = 3 * sizeof(int);

    unsigned char *hostEdge = (unsigned char *)malloc(imgBytes);

    for (int j = 0; j < myJobs; ++j)
    {
        cudaMemcpy(hostEdge, d_edges, imgBytes, cudaMemcpyDeviceToHost);
        MPI_Offset offset = headerSize + (MPI_Offset)(myFirstIdx + j) * imgBytes;
        MPI_File_write_at(fh, offset, hostEdge, imgBytes, MPI_UNSIGNED_CHAR, &mpistatus);
    }
    free(hostEdge);
    MPI_File_close(&fh);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        uint64_t t1 = clock_now();
        printf("Total cycles (all ranks finished): %lu\n", t1 - t0);
    }

    free(grayImg.data);
    free(blurredImg.data);
    free(edgesImg.data);
    free(gradImg.magnitude);
    free(gradImg.direction);
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_edges);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_mag);
    cudaFree(d_dir);
    free(rgbImg.data);

    MPI_Finalize();
    return 0;
}
