#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>

/* compile: gcc -o {file name} {file name}.c -lOpenCL */

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

        // Get log size
        status = clGetProgramBuildInfo(program, device_id, 
                                CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        handle_error(status);

        log = (char *) malloc(log_size + 1);
        // Get log 
        status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        handle_error(status);
        log[log_size] = '\0';

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
 
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
