#ifdef USECUDA

#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse_v2.h"

using namespace std;

#ifdef SINGLE
  typedef float REAL;
#else
  typedef double REAL;
#endif

extern int NN, NNp, sparseM_NNZ, sparseG_NNZ, *sparseMrowStarts, *sparseMcol, BCnVelNodes, zeroPressureNode, Z_chol_L_NZMAX, timeN, monPoint;
extern double dt, timeT, t_ini, tolerance, *sparseAvalue, *sparseKvalue, *UnpHalf_prev, *Pn, *R1, *R11, *R12, *R13, *R2, *Un, *UnpHalf, *KtimesAcc_prev, *Acc_prev, *Acc, *MdOrigInv, *Unp1, *Pnp1, *Pdot;
//extern double *dummy3NN, *dummyNNp;
extern char dummyUserInput;


extern double *K_d, *A_d, *G1_d, *G2_d, *G3_d;
extern double *MdInv_d, *Un_d, *Pn_d, *Pnp1_d, *Pnp1_prev_d;
extern double *MdOrigInv_d;
extern double *R1_d, *R2_d, *R3_d;
extern double *UnpHalf_d, *Pdot_d, *Acc_d, *Unp1_d, *Unp1_prev_d;
extern double *UnpHalf_prev_d;
extern double *Acc_prev_d;
extern double *KtimesAcc_prev_d;
extern int *Mcol_d, *Mrow_d, *MrowStarts_d, *Gcol_d, *Grow_d, *GrowStarts_d;
extern int *BCvelNodes_d;
extern int *Z_sym_pinv_d, *Z_chol_Lp_d, *Z_chol_Li_d;
extern double *Z_chol_Lx_d;

extern cusparseHandle_t   handle;
extern cusparseMatDescr_t descr;
extern cublasHandle_t     handleCUBLAS;
extern cudaError_t        cudaStatus;

extern cusparseSolveAnalysisInfo_t analysisInfo1, analysisInfo2;
extern double getHighResolutionTime(int, double);





//========================================================================
__global__ void getMonitorData(int monPoint, int NN, double *Un_d, double *Pn_d, double *monitorData_d)
//========================================================================
{
   monitorData_d[0] = Un_d[monPoint];
   monitorData_d[1] = Un_d[monPoint + NN];
   monitorData_d[2] = Un_d[monPoint + 2*NN];
   monitorData_d[3] = Pn_d[monPoint];
}




//========================================================================
__global__ void combineThreeVectors(int N, double *A, double *A1, double *A2, double *A3)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   //for (int i=0; i<N; i++) {
   //   printf("%d   %g   %g   %g\n", i, A1[i], A2[i], A3[i]);
   //}

   while (tid < N) {
      A[tid]       = A1[tid];
      A[tid + N]   = A2[tid];
      A[tid + 2*N] = A3[tid];

      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void applyVelBC(int Nbc, int N, int *velBCdata, double *A)
//========================================================================
{
   // Change R1 for known velocity BCs
   int node;

   for (int i = 0; i < Nbc; i++) {
      node = velBCdata[i];   // Node at which this velocity BC is specified.
     
      // Change R1 for the given u and v velocities.
      A[node]       = 0.0;   // This is not velocity, but velocity difference between 2 iterations.
      A[node + N]   = 0.0;   // This is not velocity, but velocity difference between 2 iterations.
      A[node + 2*N] = 0.0;   // This is not velocity, but velocity difference between 2 iterations.
   }
}





//========================================================================
__global__ void applyPresBC(int zeroPressureNode, double *R2_d)
//========================================================================
{
   // Change R2 for known zero presssures at the outlets
   if (zeroPressureNode > 0) {  // If node is negative it means we do not set pressure to zero at any node.
      R2_d[zeroPressureNode] = 0.0;  // This is not the RHS for pressure, but pressure difference between 2 iterations.
   }
}





//========================================================================
__global__ void calculate_UnpHalf(int N, double dt, double *UnpHalf_d, double *Un_d, double *R1_d, double *MdInv_d)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      UnpHalf_d[tid] = Un_d[tid] + dt * R1_d[tid] * MdInv_d[tid];

      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void calculate_step2dummyV1(int N, double d, double *A, double *B)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      A[tid] = d * B[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void calculate_step2dummyV2(int N, double d, double *A, double *B, double *C, double *D)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      A[tid] = d * B[tid] - C[tid] * D[tid];
      //A[tid] = C[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}




/*
//========================================================================
__global__ void copyVector(int N, double *A, double *B)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      A[tid] = B[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}
*/




//========================================================================
__global__ void subtractVectors(int N, double *A, double *B, double *C)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      A[tid] = B[tid] - C[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void permuteVector(int N, int *p, double *b, double *x)
//========================================================================
{
   // Equates b to x, but using the permutation vector b, i.e. x(p(k)) = b(k)

   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      x[p[tid]] = b[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void permuteVectorBack(int N, int *p, double *b, double *x)
//========================================================================
{
   // Equates b to x, but using the permutation vector b, i.e. x(k) = b(p(k))

   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      x[tid] = b[p[tid]];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void calculate_Pnp1(int N, double dt, double *Pnp1_d, double *Pn_d, double *Pdot_d)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      Pnp1_d[tid] = Pn_d[tid] + dt * Pdot_d[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void calculate_Unp1(int N, double dt, double *Unp1_d, double *UnpHalf_d, double *Acc_d)                                                   // TODO: This function is the same as the previous one. 
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      Unp1_d[tid] = UnpHalf_d[tid] + dt * Acc_d[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void calculate_R3(int N, double dt, double *A, double *B)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      A[tid] = A[tid] - dt * B[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void multiplyVectors(int N, double *A, double *B, double *C)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      A[tid] = B[tid] * C[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
void step1GPUpart(void)
//========================================================================
{
   // CUSPARSE Reference: docs.nvidia.com/cuda/cusparse/index.html#appendix-b-cusparse-library-c---example


   // Calculate the RHS vector of step 1.
   // R1 = - K * UnpHalf_prev - A * UnpHalf_prev - G * Pn;

   int NNZM = sparseM_NNZ / 3;
   int NNZG = sparseG_NNZ / 3;

   cudaStatus = cudaMemcpy(A_d,            sparseAvalue,  NNZM * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error101: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(UnpHalf_prev_d, Un,            3*NN * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error102: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Pn_d,           Pn,            NNp  * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error103: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   double alpha = -1.000000000000000;
   double beta = 0.000000000000000;
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, UnpHalf_prev_d, &beta, R1_d);                // Part of (- K * UnpHalf_prev)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, UnpHalf_prev_d + NN, &beta, R1_d + NN);      // Part of (- K * UnpHalf_prev)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, UnpHalf_prev_d + 2*NN, &beta, R1_d + 2*NN);  // Part of (- K * UnpHalf_prev)

   beta = 1.000000000000000;
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, A_d, MrowStarts_d, Mcol_d, UnpHalf_prev_d, &beta, R1_d);                // Part of (- A * UnpHalf_prev)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, A_d, MrowStarts_d, Mcol_d, UnpHalf_prev_d + NN, &beta, R1_d + NN);      // Part of (- A * UnpHalf_prev)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, A_d, MrowStarts_d, Mcol_d, UnpHalf_prev_d + 2*NN, &beta, R1_d + 2*NN);  // Part of (- A * UnpHalf_prev)

   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G1_d, GrowStarts_d, Gcol_d, Pn_d, &beta, R1_d);                        // Part of (- G1 * Pn)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G2_d, GrowStarts_d, Gcol_d, Pn_d, &beta, R1_d + NN);                   // Part of (- G1 * Pn)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G3_d, GrowStarts_d, Gcol_d, Pn_d, &beta, R1_d + 2*NN);                 // Part of (- G1 * Pn)

   applyVelBC<<<1,1>>>(BCnVelNodes, NN, BCvelNodes_d, R1_d);

   // Calculate UnpHalf
   calculate_UnpHalf<<<256,256>>>(3*NN, dt, UnpHalf_d, Un_d, R1_d, MdInv_d);
   
   cudaStatus = cudaMemcpy(UnpHalf, UnpHalf_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);    if(cudaStatus != cudaSuccess) { printf("Error104: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

}  // End of function step1GPUpart()







//========================================================================
void step2GPU(int iter)
//========================================================================
{
   // Executes step 2 of the method to determine pressure of the new time step.

   // Calculate the RHS vector of step 2.
   // This is 1/(dt*dt) times of the residual defined in Blasco's paper.
   // R2 = Gt * (UnpHalf / (dt*dt) - MdOrigInv * K * Acc_prev)
   

   double *dummy_d;             // This will store (UnpHalf / (dt*dt) - MdOrigInv * K * Acc_prev) array.
   cudaStatus = cudaMalloc((void**)&dummy_d, 3*NN * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error105: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   
   double oneOverdt2 = 1.0000000000000000 / (dt*dt);

   if (iter == 1) {
      calculate_step2dummyV1<<<256,256>>>(3*NN, oneOverdt2, dummy_d, UnpHalf_d);
   } else {
      cudaStatus = cudaMemcpy(KtimesAcc_prev_d, KtimesAcc_prev, 3*NN * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error107: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

      calculate_step2dummyV2<<<256,256>>>(3*NN, oneOverdt2, dummy_d, UnpHalf_d, MdOrigInv_d, KtimesAcc_prev_d);
   }

   // Multiply Gt with the previously calculated dummy arrays.
   int NNZG = sparseG_NNZ / 3;

   double alpha = 1.000000000000000;
   double beta = 0.000000000000000;
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G1_d, GrowStarts_d, Gcol_d, dummy_d,        &beta, R2_d);  // 1st part of [Gt] * {dummy}
   beta = 1.000000000000000;
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G2_d, GrowStarts_d, Gcol_d, dummy_d + NN,   &beta, R2_d);  // 2nd part of [Gt] * {dummy}
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G3_d, GrowStarts_d, Gcol_d, dummy_d + 2*NN, &beta, R2_d);  // 3rd part of [Gt] * {dummy}

   applyPresBC<<<1,1>>>(zeroPressureNode, R2_d);

   cudaStatus = cudaMemcpy(R2, R2_d, NNp * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error113: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   
   // Solve for Pdot using Cholesky factorization obtained in step 0.
   // Reference: Timothy Davis' book, page 136

   cublasDcopy(handleCUBLAS, NNp, R2_d, 1, Pdot_d, 1);    // Copy R2 to Pdot

   double *x_d;
   cudaStatus = cudaMalloc((void**)&x_d, NNp * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error114: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }     // Dummy vector to store intermdiate value of the 1st triangular solver.
   
   permuteVector<<<256,256>>>(NNp, Z_sym_pinv_d, R2_d, x_d);    // x_d(Z_sym_pinv_d(k)) = R2_d(k)


   double Start, wallClockTime;   

   Start = getHighResolutionTime(1, 1.0);

   cusparseSolveAnalysisInfo_t analysisInfo;
   cusparseCreateSolveAnalysisInfo(&analysisInfo);

   cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
   cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_UPPER);
   cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
   
   
   // For some unknown reason calling the analysis steps here again and again before the solve steps is much faster than calling them once.
   cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE, NNp, Z_chol_L_NZMAX, descr, Z_chol_Lx_d, Z_chol_Lp_d, Z_chol_Li_d, analysisInfo);   // TODO : Do the analsis part only once.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("1st analysis in step2  took  %8.3f seconds.\n", wallClockTime);
   
   Start = getHighResolutionTime(1, 1.0);
   double *y_d;
   alpha = 1.0;
   cudaStatus = cudaMalloc((void**)&y_d, NNp * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error115: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }     // Dummy vector to store the result of the 1st triangular solver.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("Memcpy in step2        took  %8.3f seconds.\n", wallClockTime);

   Start = getHighResolutionTime(1, 1.0);
   cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, NNp, &alpha, descr, Z_chol_Lx_d, Z_chol_Lp_d, Z_chol_Li_d, analysisInfo, x_d, y_d);
   wallClockTime = getHighResolutionTime(2, Start);
   printf("1st solve in step2     took  %8.3f seconds.\n", wallClockTime);


   cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
   cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_UPPER);
   cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
   
   cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NNp, Z_chol_L_NZMAX, descr, Z_chol_Lx_d, Z_chol_Lp_d, Z_chol_Li_d, analysisInfo);
   cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NNp, &alpha, descr, Z_chol_Lx_d, Z_chol_Lp_d, Z_chol_Li_d, analysisInfo, y_d, x_d);


   permuteVectorBack<<<256,256>>>(NNp, Z_sym_pinv_d, x_d, Pdot_d);    // Pdot_d(k) = x(Z_sym_pinv_d(k))

   cudaStatus = cudaMemcpy(Pdot, Pdot_d, NNp * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error116: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }                                                                           // TODO: Is this necesssary?

   calculate_Pnp1<<<256,256>>>(NNp, dt, Pnp1_d, Pn_d, Pdot_d);    // Pnp1 = Pn + dt * Pdot                                                          // TODO: Use CUBLAS function
   
   cudaStatus = cudaMemcpy(Pnp1, Pnp1_d, NNp * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error117: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   
   // Set matrix type back to general
   cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

   // Delete dummy variables from the GPU memory
   cudaFree(x_d);
   cudaFree(y_d);
   cudaFree(dummy_d);
   
}  // End of function step2GPU()





//========================================================================
void step3GPU(int iter)
//========================================================================
{
   // Executes step 3 of the method to determine the velocity of the new time step.

   // Calculate the RHS vector of step 3.
   // R3 = - dt * (G * Pdot + K * Acc_prev)

   int NNZG = sparseG_NNZ / 3;

   double alpha = -dt;
   double beta = 0.000000000000000;                                                                                                                  // TODO: Use a single Rvel instead of (R1 and R3)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G1_d, GrowStarts_d, Gcol_d, Pdot_d, &beta, R3_d);             // This contributes to (- dt * G1 * Pdot)  part of R3
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G2_d, GrowStarts_d, Gcol_d, Pdot_d, &beta, R3_d + NN);        // This contributes to (- dt * G2 * Pdot)  part of R3
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G3_d, GrowStarts_d, Gcol_d, Pdot_d, &beta, R3_d + 2*NN);      // This contributes to (- dt * G3 * Pdot)  part of R3

   // Subtract dt * KtimesAcc_prev from R3 if iter is not 1.
   if (iter != 1) {
      calculate_R3<<<256,256>>>(3*NN, dt, R3_d, KtimesAcc_prev_d);
   }

   applyVelBC<<<1,1>>>(BCnVelNodes, NN, BCvelNodes_d, R3_d);

   // Calculate Acc (Acc = R3 * MdInv)
   multiplyVectors<<<256,256>>>(3*NN, Acc_d, R3_d, MdInv_d);                                                                                         // TODO: Use CUBLAS function
   cudaStatus = cudaMemcpy(Acc, Acc_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error118: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   
   // Calculate Unp1 (Unp1 = UnpHalf + dt * Acc)
   calculate_Unp1<<<256,256>>>(3*NN, dt, Unp1_d, UnpHalf_d, Acc_d);                                                                                  // TODO: Use CUBLAS function

   cudaStatus = cudaMemcpy(Unp1, Unp1_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error119: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   
}  // End of function step3GPU()





//========================================================================
void calculate_KtimesAcc_prevGPU(void)
//========================================================================
{
   int NNZM = sparseM_NNZ / 3;

   double alpha = 1.0;
   double beta = 0.0;
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, Acc_prev_d,        &beta, KtimesAcc_prev_d);         //  1st part of [K] * {Acc_prev}
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, Acc_prev_d + NN,   &beta, KtimesAcc_prev_d + NN);    //  2nd part of [K] * {Acc_prev}
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, Acc_prev_d + 2*NN, &beta, KtimesAcc_prev_d + 2*NN);  //  3rd part of [K] * {Acc_prev}

   cudaStatus = cudaMemcpy(KtimesAcc_prev, KtimesAcc_prev_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error120: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

}  // End of function calculate_KtimesAcc_prevGPU()





//========================================================================
bool checkConvergenceGPU(void)
//========================================================================
{
   double norm1, norm2, normalizedNorm1, normalizedNorm2;

   // Calculate normalized norm for velocity
   cublasDnrm2(handleCUBLAS, 3*NN, Unp1_d, 1, &norm1);    // norm1 = sqrt(sum(Unp1(i)*Unp1(i)))

   double *dummy_d;
   cudaStatus = cudaMalloc((void**)&dummy_d, 3*NN * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error121: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }   // dummy_d will store (Unp1_d - Unp1_prev_d)

   subtractVectors<<<256,256>>>(3*NN, dummy_d, Unp1_d, Unp1_prev_d);
   cublasDnrm2(handleCUBLAS, 3*NN, dummy_d, 1, &norm2);   // norm2 = sqrt(sum((Unp1(i)-Unp_prev(i))*(Unp1(i)-Unp_prev(i))))
   normalizedNorm1 = norm2 / norm1;   // Normalized norm for velocity

   cudaFree(dummy_d);


   // Calculate normalized norm for pressure
   cublasDnrm2(handleCUBLAS, NNp, Pnp1_d, 1, &norm1);    // norm1 = sqrt(sum(Pnp1(i)*Pnp1(i)))

   cudaStatus = cudaMalloc((void**)&dummy_d, NNp * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error122: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }    // dummy_d will now store (Pnp1_d - Pnp1_prev_d)

   subtractVectors<<<256,256>>>(NNp, Pnp1_d, Pnp1_prev_d, dummy_d);
   cublasDnrm2(handleCUBLAS, NNp, dummy_d, 1, &norm2);   // norm2 = sqrt(sum((Pnp1(i)-Pnp_prev(i))*(Pnp1(i)-Pnp_prev(i))))
   normalizedNorm2 = norm2 / norm1;   // Normalized norm for pressure

   cudaFree(dummy_d);
   

   // Check convergence and get ready for the next iteration
   if (normalizedNorm1 < tolerance && normalizedNorm2 < tolerance) {
      return 1;
   } else {
      // Get ready for the next iteration
      cudaStatus = cudaMemcpy(UnpHalf_prev_d, UnpHalf_d, 3*NN * sizeof(double), cudaMemcpyDeviceToDevice);   if(cudaStatus != cudaSuccess) { printf("Error123: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
      cudaStatus = cudaMemcpy(Unp1_prev_d,    Unp1_d,    3*NN * sizeof(double), cudaMemcpyDeviceToDevice);   if(cudaStatus != cudaSuccess) { printf("Error124: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
      cudaStatus = cudaMemcpy(Acc_prev_d,     Acc_d,     3*NN * sizeof(double), cudaMemcpyDeviceToDevice);   if(cudaStatus != cudaSuccess) { printf("Error125: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
      cudaStatus = cudaMemcpy(Pnp1_prev_d,    Pnp1_d,    NNp  * sizeof(double), cudaMemcpyDeviceToDevice);   if(cudaStatus != cudaSuccess) { printf("Error126: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

      return 0;      
   }

}  // End of function checkConvergence()





//========================================================================
void printMonitorDataGPU(int iter)
//========================================================================
{
   //double *monitorData_d;
   //cudaMalloc((void**)&monitorData_d, 4 * sizeof(double));

   //getMonitorData<<<1,1>>>(monPoint, NN, Un_d, Pn_d, monitorData_d);

   //double *monitorData;
   //monitorData = new double[4];
   cudaStatus = cudaMemcpy(Un, Un_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error127: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Pn, Pn_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error128: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   printf("\n%6d  %6d  %10.5f  %12.5f  %12.5f  %12.5f  %12.5f\n", timeN, iter, timeT, Un[monPoint], Un[monPoint + NN], Un[monPoint + 2*NN], Pn[monPoint]);
}  // End of function printMonitorDataGPU()





//========================================================================
void choleskyAnalysisGPU()
//========================================================================
{
   // Perform the analysis part of the Cholesky factorization only once
   // before the time loop.
 
   cusparseCreateSolveAnalysisInfo(&analysisInfo1);
   cusparseCreateSolveAnalysisInfo(&analysisInfo2);
   
   cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
   cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_UPPER);
   cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
   
   //cusparseStatus_t status;
   
   /*status = */cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE,     NNp, Z_chol_L_NZMAX, descr, Z_chol_Lx_d, Z_chol_Lp_d, Z_chol_Li_d, analysisInfo1);  // Analysis of the 1st triangular solver
   
   /*
   switch(status)
    {
        case CUSPARSE_STATUS_SUCCESS:          cout << "\n\n1CUSPARSE_STATUS_SUCCESS\n\n";
        case CUSPARSE_STATUS_NOT_INITIALIZED:  cout << "1CUSPARSE_STATUS_NOT_INITIALIZED\n\n";
        case CUSPARSE_STATUS_ALLOC_FAILED:     cout << "1CUSPARSE_STATUS_ALLOC_FAILED\n\n";
        case CUSPARSE_STATUS_INVALID_VALUE:    cout << "1CUSPARSE_STATUS_INVALID_VALUE\n\n"; 
        case CUSPARSE_STATUS_ARCH_MISMATCH:    cout << "1CUSPARSE_STATUS_ARCH_MISMATCH\n\n"; 
        case CUSPARSE_STATUS_MAPPING_ERROR:    cout << "1CUSPARSE_STATUS_MAPPING_ERROR\n\n";
        case CUSPARSE_STATUS_EXECUTION_FAILED: cout << "1CUSPARSE_STATUS_EXECUTION_FAILED\n\n"; 
        case CUSPARSE_STATUS_INTERNAL_ERROR:   cout << "1CUSPARSE_STATUS_INTERNAL_ERROR\n\n"; 
    }
    
   status = cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NNp, Z_chol_L_NZMAX, descr, Z_chol_Lx_d, Z_chol_Lp_d, Z_chol_Li_d, analysisInfo2);  // Analysis of the 2nd triangular solver

   switch(status)
    {
        case CUSPARSE_STATUS_SUCCESS:          cout << "\n\n1CUSPARSE_STATUS_SUCCESS\n\n";
        case CUSPARSE_STATUS_NOT_INITIALIZED:  cout << "1CUSPARSE_STATUS_NOT_INITIALIZED\n\n";
        case CUSPARSE_STATUS_ALLOC_FAILED:     cout << "1CUSPARSE_STATUS_ALLOC_FAILED\n\n";
        case CUSPARSE_STATUS_INVALID_VALUE:    cout << "1CUSPARSE_STATUS_INVALID_VALUE\n\n"; 
        case CUSPARSE_STATUS_ARCH_MISMATCH:    cout << "1CUSPARSE_STATUS_ARCH_MISMATCH\n\n"; 
        case CUSPARSE_STATUS_MAPPING_ERROR:    cout << "1CUSPARSE_STATUS_MAPPING_ERROR\n\n";
        case CUSPARSE_STATUS_EXECUTION_FAILED: cout << "1CUSPARSE_STATUS_EXECUTION_FAILED\n\n"; 
        case CUSPARSE_STATUS_INTERNAL_ERROR:   cout << "1CUSPARSE_STATUS_INTERNAL_ERROR\n\n";
    }
    */
   
}  // End of function choleskyAnalysisGPU()












/*
   
    Reference CUSPARSE Code

    cudaError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4,cudaStat5,cudaStat6;
    int *    cooRowIndexHostPtr=0;
    int *    cooColIndexHostPtr=0;    
    double * cooValHostPtr=0;
    int *    cooRowIndex=0;
    int *    cooColIndex=0;    
    double * cooVal=0;
    int *    xIndHostPtr=0;
    double * xValHostPtr=0;
    double * yHostPtr=0;
    int *    xInd=0;
    double * xVal=0;
    double * y=0;  
    int *    csrRowPtr=0;
    double * zHostPtr=0; 
    double * z=0; 
    int      n, nnz, nnz_vector;
    double dzero =0.0;
    double dtwo  =2.0;
    double dthree=3.0;
    double dfive =5.0;




    // Create the following sparse test matrix in COO format
    //   |1.0     2.0 3.0|
    //   |    4.0        |
    //   |5.0     6.0 7.0|
    //   |    8.0     9.0|

    n=4;
    nnz=9; 
    
    cooRowIndexHostPtr = new int[nnz];
    cooColIndexHostPtr = new int[nnz];
    cooValHostPtr      = new double[nnz];
    
    if ((!cooRowIndexHostPtr) || (!cooColIndexHostPtr) || (!cooValHostPtr)){
       CLEANUP("Host malloc failed (matrix)");
       return 1;
    }

    cooRowIndexHostPtr[0]=0; cooColIndexHostPtr[0]=0; cooValHostPtr[0]=1.0;  
    cooRowIndexHostPtr[1]=0; cooColIndexHostPtr[1]=2; cooValHostPtr[1]=2.0;  
    cooRowIndexHostPtr[2]=0; cooColIndexHostPtr[2]=3; cooValHostPtr[2]=3.0;  
    cooRowIndexHostPtr[3]=1; cooColIndexHostPtr[3]=1; cooValHostPtr[3]=4.0;  
    cooRowIndexHostPtr[4]=2; cooColIndexHostPtr[4]=0; cooValHostPtr[4]=5.0;  
    cooRowIndexHostPtr[5]=2; cooColIndexHostPtr[5]=2; cooValHostPtr[5]=6.0;
    cooRowIndexHostPtr[6]=2; cooColIndexHostPtr[6]=3; cooValHostPtr[6]=7.0;  
    cooRowIndexHostPtr[7]=3; cooColIndexHostPtr[7]=1; cooValHostPtr[7]=8.0;  
    cooRowIndexHostPtr[8]=3; cooColIndexHostPtr[8]=3; cooValHostPtr[8]=9.0; 

    // Create a sparse and dense vector
    //   xVal= [100.0 200.0 400.0]   (sparse)
    //   xInd= [0     1     3    ]
    //   y   = [10.0 20.0 30.0 40.0 | 50.0 60.0 70.0 80.0] (dense)
    nnz_vector = 3;

    xIndHostPtr = (int *)   malloc(nnz_vector*sizeof(xIndHostPtr[0])); 
    xValHostPtr = (double *)malloc(nnz_vector*sizeof(xValHostPtr[0])); 
    yHostPtr    = (double *)malloc(2*n       *sizeof(yHostPtr[0]));
    zHostPtr    = (double *)malloc(2*(n+1)   *sizeof(zHostPtr[0]));
    if((!xIndHostPtr) || (!xValHostPtr) || (!yHostPtr) || (!zHostPtr)){
        CLEANUP("Host malloc failed (vectors)");
        return 1; 
    }
    yHostPtr[0] = 10.0;  xIndHostPtr[0]=0; xValHostPtr[0]=100.0; 
    yHostPtr[1] = 20.0;  xIndHostPtr[1]=1; xValHostPtr[1]=200.0;  
    yHostPtr[2] = 30.0;
    yHostPtr[3] = 40.0;  xIndHostPtr[2]=3; xValHostPtr[2]=400.0;  
    yHostPtr[4] = 50.0;
    yHostPtr[5] = 60.0;
    yHostPtr[6] = 70.0;
    yHostPtr[7] = 80.0;


    // Allocate GPU memory and copy the matrix and vectors into it
    cudaStat1 = cudaMalloc((void**)&cooRowIndex,nnz*sizeof(cooRowIndex[0])); 
    cudaStat2 = cudaMalloc((void**)&cooColIndex,nnz*sizeof(cooColIndex[0]));
    cudaStat3 = cudaMalloc((void**)&cooVal,     nnz*sizeof(cooVal[0])); 
    cudaStat4 = cudaMalloc((void**)&y,          2*n*sizeof(y[0]));   
    cudaStat5 = cudaMalloc((void**)&xInd,nnz_vector*sizeof(xInd[0])); 
    cudaStat6 = cudaMalloc((void**)&xVal,nnz_vector*sizeof(xVal[0]));

    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
        (cudaStat4 != cudaSuccess) ||
        (cudaStat5 != cudaSuccess) ||
        (cudaStat6 != cudaSuccess)) {
        CLEANUP("Device malloc failed");
        return 1; 
    }    
    cudaStat1 = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr, (size_t)(nnz*sizeof(cooRowIndex[0])), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cooColIndex, cooColIndexHostPtr, (size_t)(nnz*sizeof(cooColIndex[0])), cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(cooVal,      cooValHostPtr,      (size_t)(nnz*sizeof(cooVal[0])),      cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(y,           yHostPtr,           (size_t)(2*n*sizeof(y[0])),           cudaMemcpyHostToDevice);
    cudaStat5 = cudaMemcpy(xInd,        xIndHostPtr,        (size_t)(nnz_vector*sizeof(xInd[0])), cudaMemcpyHostToDevice);
    cudaStat6 = cudaMemcpy(xVal,        xValHostPtr,        (size_t)(nnz_vector*sizeof(xVal[0])), cudaMemcpyHostToDevice);
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
        (cudaStat4 != cudaSuccess) ||
        (cudaStat5 != cudaSuccess) ||
        (cudaStat6 != cudaSuccess)) {
        CLEANUP("Memcpy from Host to Device failed");
        return 1;
    }

    // Exercise conversion routines (convert matrix from COO 2 CSR format)
    cudaStat1 = cudaMalloc((void**)&csrRowPtr,(n+1)*sizeof(csrRowPtr[0]));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed (csrRowPtr)");
        return 1;
    }
    status= cusparseXcoo2csr(handle,cooRowIndex,nnz,n,
                             csrRowPtr,CUSPARSE_INDEX_BASE_ZERO); 
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Conversion from COO to CSR format failed");
        return 1;
    }  
    //csrRowPtr = [0 3 4 7 9]


    // exercise Level 1 routines (scatter vector elements)
    status= cusparseDsctr(handle, nnz_vector, xVal, xInd, 
                          &y[n], CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Scatter from sparse to dense vector failed");
        return 1;
    }  
    //y = [10 20 30 40 | 100 200 70 400]



    // Exercise Level 2 routines (csrmv)
    status= cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
                           &dtwo, descr, cooVal, csrRowPtr, cooColIndex, 
                           &y[0], &dthree, &y[n]);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix-vector multiplication failed");
        return 1;
    }    
    //y = [10 20 30 40 | 680 760 1230 2240]
    cudaMemcpy(yHostPtr, y, (size_t)(2*n*sizeof(y[0])), cudaMemcpyDeviceToHost);
    
    //printf("Intermediate results:\n");
    //for (int j=0; j<2; j++){
    //    for (int i=0; i<n; i++){        
    //        printf("yHostPtr[%d,%d]=%f\n",i,j,yHostPtr[i+n*j]);
    //    }
    //}
    

    // exercise Level 3 routines (csrmm)
    cudaStat1 = cudaMalloc((void**)&z, 2*(n+1)*sizeof(z[0]));   
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed (z)");
        return 1;
    }
    cudaStat1 = cudaMemset((void *)z,0, 2*(n+1)*sizeof(z[0]));    
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memset on Device failed");
        return 1;
    }
    status= cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, 2, n, 
                           nnz, &dfive, descr, cooVal, csrRowPtr, cooColIndex, 
                           y, n, &dzero, z, n+1);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix-matrix multiplication failed");
        return 1;
    }  

    // print final results (z)
    cudaStat1 = cudaMemcpy(zHostPtr, z, 
                           (size_t)(2*(n+1)*sizeof(z[0])), 
                           cudaMemcpyDeviceToHost);
    if (cudaStat1 != cudaSuccess)  {
        CLEANUP("Memcpy from Device to Host failed");
        return 1;
    } 
    //z = [950 400 2550 2600 0 | 49300 15200 132300 131200 0]
    
    //printf("Final results:\n");
    //for (int j=0; j<2; j++){
    //    for (int i=0; i<n+1; i++){
    //        printf("z[%d,%d]=%f\n",i,j,zHostPtr[i+(n+1)*j]);
    //    }
    //}
    

    // destroy matrix descriptor
    status = cusparseDestroyMatDescr(descr); 
    descr = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor destruction failed");
        return 1;
    }    

    // destroy handle
    status = cusparseDestroy(handle);
    handle = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library release of resources failed");
        return 1;
    }   

    // check the results
    // Notice that CLEANUP() contains a call to cusparseDestroy(handle)
    if ((zHostPtr[0] != 950.0)    || 
        (zHostPtr[1] != 400.0)    || 
        (zHostPtr[2] != 2550.0)   || 
        (zHostPtr[3] != 2600.0)   || 
        (zHostPtr[4] != 0.0)      || 
        (zHostPtr[5] != 49300.0)  || 
        (zHostPtr[6] != 15200.0)  || 
        (zHostPtr[7] != 132300.0) || 
        (zHostPtr[8] != 131200.0) || 
        (zHostPtr[9] != 0.0)      ||
        (yHostPtr[0] != 10.0)     || 
        (yHostPtr[1] != 20.0)     || 
        (yHostPtr[2] != 30.0)     || 
        (yHostPtr[3] != 40.0)     || 
        (yHostPtr[4] != 680.0)    || 
        (yHostPtr[5] != 760.0)    || 
        (yHostPtr[6] != 1230.0)   || 
        (yHostPtr[7] != 2240.0)){ 
        CLEANUP("example test FAILED");
        return 1;
    }
    else{
        CLEANUP("example test PASSED");
        return 0;
    }
*/

#endif //USECUDA
