#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>

/* compile: gcc -o vecAdd vecAdd.c -lOpenCL */

void handle_error(cl_int status) {
    if (status != CL_SUCCESS) {
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, status);
        exit(EXIT_FAILURE);
    }
}

/* Check compile error on specific device */
void handle_compile_error(cl_int status, cl_program program, cl_device_id device_id) {
    if (status == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char *log;
        cl_int status;

        // Get log size
        status = clGetProgramBuildInfo(program, device_id, 
                                CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        handle_error(status);

        log = (char *) malloc(log_size + 1);
        // Get log 
        status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        handle_error(status);
        printf("log size: %d\n", &log_size);
        log[log_size] = '\0';
        printf("log size: %d\n", &log_size); 
        printf("Compile error: %s\n", log);

        free(log);

        exit(0);
    }
}

char *get_kernel_source(const char *file_name, size_t *size) {
    size_t length;
    FILE *fp;
    char *kernel_source; 

    fp = fopen(file_name, "r");
    if (fp == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(fp, 0, SEEK_END);
    length = (size_t)ftell(fp);
    rewind(fp); 
    
    kernel_source = (char *)malloc(length + 1);
    fread(kernel_source, 1, length, fp);
    kernel_source[length] = '\0';
    *size = length;

    fclose(fp);

    return kernel_source;
}

int main(int argc, char* argv[]) {
    cl_int status;

    cl_platform_id platform_id;
    cl_device_id device_id;

    cl_context context;
    cl_command_queue queue;
    char *kernel_source;
    size_t source_size;
    cl_program program;
    cl_kernel kernel;

    status = clGetPlatformIDs(1, &platform_id, NULL);
    handle_error(status);

    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    handle_error(status);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
    handle_error(status);

    queue = clCreateCommandQueue(context, device_id, 0, &status);
    handle_error(status);

    kernel_source = get_kernel_source("vecAddKernel.cl", &source_size);
    program = clCreateProgramWithSource(context, 1, (const char**) &kernel_source, 
                                        &source_size, &status);
    handle_error(status);

    status = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
    handle_compile_error(status, program, device_id);

    kernel = clCreateKernel(program, "add_vec", &status);
    handle_error(status);


    size_t vec_size = 16384;
    int *A = (int*)malloc(sizeof(int) * vec_size);
    int *B = (int*)malloc(sizeof(int) * vec_size);
    int *C = (int*)malloc(sizeof(int) * vec_size);

    int i;

    for (i=0; i<vec_size; i++) {
        A[i] = i;
        B[i] = i;
    }

    /* Create Memory Buffer */
    cl_mem buf_A, buf_B, buf_C;

    buf_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * vec_size, NULL, &status);
    handle_error(status);

    buf_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * vec_size, NULL, &status);
    handle_error(status);

    buf_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * vec_size, NULL, &status);
    handle_error(status);

    /* Enqueue Write Buffer */
    status = clEnqueueWriteBuffer(queue, buf_A, CL_FALSE, 0, sizeof(int) * vec_size, A, 0, NULL, NULL);
    handle_error(status);

    status = clEnqueueWriteBuffer(queue, buf_B, CL_FALSE, 0, sizeof(int) * vec_size, B, 0, NULL, NULL);
    handle_error(status);

    /* Set kernel Argument */
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_A);
    handle_error(status);

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_B);
    handle_error(status);

    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_C);
    handle_error(status);

    /* Execute Kernel */
    size_t global_size = vec_size;
    size_t local_size = 256;

    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, 
                                    &global_size, &local_size, 0, NULL, NULL);
    handle_error(status);

    /* Enqueue ReadBuffer */
    status = clEnqueueReadBuffer(queue, buf_C, CL_TRUE, 0, sizeof(int) * vec_size, C,
                                0, NULL, NULL);
    handle_error(status);

    // Verification
    for (i=0; i<vec_size; i++) {
        if (C[i] != A[i] + B[i]) {
            printf("Verification Failed!\n");
            break;
        }
    }

    if (i == vec_size) {
        printf("Verification Success!\n");
    }

    clReleaseMemObject(buf_A);
    clReleaseMemObject(buf_B);
    clReleaseMemObject(buf_C);

    free(A);
    free(B);
    free(C);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
