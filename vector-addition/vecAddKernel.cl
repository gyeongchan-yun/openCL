__kernel void add_vec(__global const int *A, 
                      __global const int *B,
                      __global int *C) {
        
        int i = get_global_id(0);
        C[i] = A[i] + B[i];
    }
