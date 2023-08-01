/* This is machine problem 1, part 2, force evaluation
 *
 * The problem is to take two sets of charged particles, 
 * where each particle has a position and a charge associated with itself,
 * and calculate the force between specific pairs of particles. 
 * An index array holds the information which particle in set B should be
 * paired with which particle in set A.
 * SUBMISSION GUIDELINES:
 * You should submit two files, called mp1-part2-solution-kernel.cu and mp1-part2-solution-host.cu
 * which contain your version of the force_eval and host_charged_particles functions.
 */


#include <stdlib.h>
#include <stdio.h>

#include "mp1-util.h"
#define EPSILON 0.00001f

// amount of floating point numbers between answer and computed value 
// for the answer to be taken correctly. 2's complement magick.
const int maxUlps = 1000;

event_pair timer;
  
float4 force_calc(float4 A, float4 B) 
{
  float x = B.x - A.x;
  float y = B.y - A.y;
  float z = B.z - A.z;
  float rsq = x*x + y*y + z*z;
  // avoid divide by zero
  if(rsq < EPSILON)
  {
    rsq += EPSILON;
  }
  float r = sqrt(rsq);
  float f = A.w * B.w / rsq;
  float inv_r = 1.0f / r;
  float4 fv = make_float4(x*inv_r,y*inv_r,z*inv_r,f);
  return fv;
}
 
void host_force_eval(float4 *set_A, float4 *set_B, int * indices, float4 *force_vectors, int array_length)
{
  for(int i=0;i<array_length;i++)
  {
    if(indices[i] < array_length && indices[i] >= 0)
    {
      force_vectors[i] = force_calc(set_A[i],set_B[indices[i]]);
    }
    else
    {
      force_vectors[i] = make_float4(0.0,0.0,0.0,0.0);
    }
  }
}


__global__ void force_eval(float4 *set_A, float4 *set_B, int * indices, float4 *force_vectors, int array_length)
{
  const unsigned int block_id = blockIdx.x;
  const unsigned int thread_id = threadIdx.x;

  // sizeof(float4) = 16 bytes, 4 * fp32
  // share memory: we need to store set A, indices, and force results
  // compute force in-place, and due to the memory pattern for set B, we wioll not copy set B into share memory
  // for set A: 512 * 16 = 256 * 32
  // for force results: 512 * 16 = 256 * 32 --- del
  // for indices: 512 * 4 = 64 * 32
  // in total: 576 * 32 --(24)-> 600 *32 = 18.5 * 1024 --- del
  // in total: 320 * 32 --(16)-> 336 *32 = 10.5 * 1024
  __shared__ __align__(16 * 1024) char share_mem[336 *32];
  float4 * A_share = reinterpret_cast<float4 *>(share_mem);
  // float4 * force_share = reinterpret_cast<float4 *>(share_mem + (256+8) * 32); // del
  int * indices_share = reinterpret_cast<int *>(share_mem + (256+8) * 32);

  float4 zero_float4 = make_float4(0.0,0.0,0.0,0.0);
  float4 B_reg = zero_float4;

  A_share[thread_id] = __ldg(set_A + 512 * block_id + thread_id);
  indices_share[thread_id] = __ldg(indices + 512 * block_id + thread_id);
  // __syncthreads();
  if (indices_share[thread_id] < array_length && indices_share[thread_id] >= 0) {
    B_reg = __ldg(set_B + indices_share[thread_id]); // TODO: better memory access pattern for B
  
    // __syncthreads();
    float x = B_reg.x - A_share[thread_id].x;
    float y = B_reg.y - A_share[thread_id].y;
    float z = B_reg.z - A_share[thread_id].z;
    float rsq = x*x + y*y + z*z + EPSILON;
    float r = sqrt(rsq);
    float f = A_share[thread_id].w * B_reg.w / rsq;
    float inv_r = 1.0f / r;

    force_vectors[512 * block_id + thread_id] = make_float4(x*inv_r,y*inv_r,z*inv_r,f);
  } else {
    force_vectors[512 * block_id + thread_id] = zero_float4;
  }
  // __syncthreads();
  // force_vectors[512 * block_id + thread_id] = force_share[thread_id];
  return;
}



void host_charged_particles(float4 *h_set_A, float4 *h_set_B, int *h_indices, float4 *h_force_vectors, int num_elements)
{ 
  float4 *device_set_A = 0;
  float4 *device_set_B = 0;
  int *device_indices = 0;
  float4 *device_force_vectors = 0;
  // compute the size of the arrays in bytes
  int num_bytes = num_elements * sizeof(float4);
  // cudaMalloc device arrays
  cudaMalloc((void**)&device_set_A, num_bytes);
  cudaMalloc((void**)&device_set_B, num_bytes);
  cudaMalloc((void**)&device_indices, num_elements * sizeof(int));
  cudaMalloc((void**)&device_force_vectors, num_bytes);

  // if either memory allocation failed, report an error message
  if(device_set_A == 0 || device_set_B == 0 || device_indices == 0 || device_force_vectors == 0) {
    printf("couldn't allocate memory\n");
    return;
  }

  cudaMemcpy(device_set_A, h_set_A, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_set_B, h_set_B, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_indices, h_indices, num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(device_force_vectors, 0, num_bytes);

  int block_size = 512;
  int grid_size = (num_elements + block_size - 1) / block_size;
  
  start_timer(&timer);
  // launch kernel
  force_eval<<<grid_size, block_size>>>(device_set_A, device_set_B, device_indices, device_force_vectors, num_elements);
  check_launch("gpu force eval");
  stop_timer(&timer,"gpu force eval");
  
  cudaMemcpy(h_force_vectors, device_force_vectors, num_bytes, cudaMemcpyDeviceToHost);
  return;
}


int main(void)
{
  // create arrays of 4M elements
  int num_elements =  1 << 22;

  // pointers to host & device arrays
  float4 *h_set_A = 0;
  float4 *h_set_B = 0;
  int *h_indices = 0;
  float4 *h_force_vectors = 0;
  float4 *h_force_vectors_checker = 0;
  
   // initialize
  srand(time(NULL)); 
  
  // malloc host array
  h_set_A = (float4*)malloc(num_elements * sizeof(float4));
  h_set_B = (float4*)malloc(num_elements * sizeof(float4));
  h_indices = (int*)malloc(num_elements * sizeof(int));
  h_force_vectors = (float4*)malloc(num_elements * sizeof(float4));
  h_force_vectors_checker = (float4*)malloc(num_elements * sizeof(float4));
  
  // if either memory allocation failed, report an error message
  if(h_set_A == 0 || h_set_B == 0 || h_force_vectors == 0 || h_indices == 0 || h_force_vectors_checker == 0)
  {
    printf("couldn't allocate memory\n");
    exit(1);
  }

  // generate random input
  for(int i=0;i< num_elements;i++)
  {
    h_set_A[i] = make_float4(rand(),rand(),rand(),rand()); 
    h_set_B[i] = make_float4(rand(),rand(),rand(),rand());

    // some indices will be invalid
    h_indices[i] = rand() % (num_elements + 2);
  }
  
  start_timer(&timer);
  // generate reference output
  host_force_eval(h_set_A, h_set_B, h_indices, h_force_vectors_checker, num_elements);
  
  check_launch("host force eval");
  stop_timer(&timer,"host force eval");
  
  // the results of the calculation need to end up in h_force_vectors;
  host_charged_particles(h_set_A, h_set_B, h_indices, h_force_vectors, num_elements);
  
  // check CUDA output versus reference output
  int error = 0;
  
  for(int i=0;i<num_elements;i++)
  {
    float4 v = h_force_vectors[i];
    float4 vc = h_force_vectors_checker[i];

    if( !AlmostEqual2sComplement(v.x,vc.x,maxUlps) ||
    	!AlmostEqual2sComplement(v.y,vc.y,maxUlps) ||
    	!AlmostEqual2sComplement(v.z,vc.z,maxUlps) ||
    	!AlmostEqual2sComplement(v.w,vc.w,maxUlps)) 
    { 
      error = 1;
    }
  }
  printf("\n");
  
  if(error)
  {
    printf("Output of CUDA version and normal version didn't match! \n");
  }
  else
  {
    printf("Worked! CUDA and reference output match. \n");
  }
 
  // deallocate memory
  free(h_set_A);
  free(h_set_B);
  free(h_indices);
  free(h_force_vectors);
  free(h_force_vectors_checker);
}

