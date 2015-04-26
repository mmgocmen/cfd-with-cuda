// TODO: After the run is finished the executable terminates with the following  error.                                                               // TODO
// terminate called after throwing an instance of 'thrust::system::system_error'
//   what():  unload of CUDA runtime failed
// Aborted (core dumped)

// Everything in this file is used only if compiled with USEDUDA defined

#ifdef USECUDA

#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse_v2.h"
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/diagonal.h>
#include <cusp/precond/ainv.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>
#include <cusp/transpose.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>


#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>


int NBLOCKS = 128;
int NTHREADS = 1024;

#define NGP_SIZE (8)
#define NENv_SIZE (27)
#define DIM_SIZE (3)


using namespace std;
using namespace thrust;

extern int NN, NNp, sparseM_NNZ, sparseG_NNZ, *sparseMrowStarts, *sparseGrowStarts, *sparseMrow, *sparseMcol, BCnVelNodes, zeroPressureNode, timeN, monPoint;
extern int ** BCvelNodes;
extern double dt, timeT, t_ini, tolerance, *sparseAvalue, *sparseKvalue, *UnpHalf_prev, *Pn, *R1, *R11, *R12, *R13, *R2, *Un, *UnpHalf, *KtimesAcc_prev, *Acc_prev, *Acc, *MdInv, *MdOrigInv, *Unp1, *Pnp1, *Pdot;
extern double convergenceCriteria, maxAcc;
extern double wallClockTimeCurrentTimeStep;
extern char dummyUserInput;
extern string whichProblem;

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

extern cusparseHandle_t   handle;
extern cusparseMatDescr_t descr;
extern cublasHandle_t     handleCUBLAS;
extern cudaError_t        cudaStatus;

extern cusparseSolveAnalysisInfo_t analysisInfo1, analysisInfo2;

extern int *sparseGrow, *sparseGcol;
extern double *sparseG1value, *sparseG2value, *sparseG3value;

extern size_t freeGPUmemory, totalGPUmemory;   // To measure total and free GPU memory
cusp::csr_matrix<int, double, cusp::device_memory> Z_CUSP_CSR_d;

extern bool PRINT_TIMES;

extern double getHighResolutionTime(int, double);
extern void createTecplot();


extern int NE, NGP, NENv;
extern int *NmeshColors, *meshColors, *elementsOfColor;
extern int nActiveColors;
extern int *LtoGvel_1d;
extern int *sparseMapM_1d;
extern double *Sv_1d;
extern double *gDSv_1d, *GQfactor_1d;

extern int *NmeshColors_d, *meshColors_d, *elementsOfColor_d;
extern int *LtoGvel_1d_d;
extern int *sparseMapM_1d_d;
extern double *Sv_1d_d;
extern double *gDSv_1d_d, *GQfactor_1d_d;

struct weighted_absolute_difference // used at steady state convergence check 
{
  double oneOverdt;

  weighted_absolute_difference(const double oneOverdt)
    : oneOverdt(oneOverdt)
  {}

  __host__ __device__
  double operator()(thrust::tuple<double,double> t)
  {
    double thrust_Unp1_d = thrust::get<0>(t);
    double thrust_Un_d   = thrust::get<1>(t);
    double accDummy;
    
    accDummy = thrust_Unp1_d - thrust_Un_d;

    return fabs(accDummy);
  }
};

// double *UnGPU;




//========================================================================
void selectCUDAdevice()
//========================================================================
{
   // Print information about available CUDA devices set the CUDA device to be used
   cudaDeviceProp prop;
   int nDevices;
      
   cout << "Available CUDA devices are" << endl;
   cudaGetDeviceCount(&nDevices);
   
   for (int i = 0; i < nDevices; i++) {
     cudaGetDeviceProperties(&prop, i);
     printf("  %d: %s\n", i, prop.name);
   }

   if (nDevices == 1) {                                                                                                                            // TODO : This will not work on every machine.
      cudaSetDevice(0);
      cout << "\nDevice " << 0 << " is selected.\n";
   } else {
	  cudaSetDevice(nDevices - 1);
	  //cudaSetDevice(0);
	  cout << "\nDevice " << nDevices - 1 << " is selected.\n";
   } 
      
   cudaMemGetInfo(&freeGPUmemory, &totalGPUmemory);
   cout << "  Total GPU memory = " << totalGPUmemory << endl;
   cout << "  Free  GPU memory = " << freeGPUmemory << endl << endl;

}  // End of function selectCUDAdevice()





//========================================================================
void initializeAndAllocateGPU()
//========================================================================
{
   // Do the necessary memory allocations for the GPU. Apply the initial
   // condition or read the restart file.

   handle = 0;
   descr  = 0;
   
   // Initialize cusparse library
   cusparseCreate(&handle);
   cublasCreate(&handleCUBLAS);

   // Create and setup matrix descriptor
   cusparseCreateMatDescr(&descr);

   cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);


   int NNZM = sparseM_NNZ / 3;
   int NNZG = sparseG_NNZ / 3;

   cudaStatus = cudaMalloc((void**)&K_d,              NNZM   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error01: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   //cudaStatus = cudaMalloc((void**)&A_d,              NNZM   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error02: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&G1_d,             NNZG   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error03: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&G2_d,             NNZG   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error04: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&G3_d,             NNZG   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error05: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&UnpHalf_prev_d,   3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error06: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&KtimesAcc_prev_d, 3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error07: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Acc_d,            3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error08: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Acc_prev_d,       3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error09: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Pn_d,             NNp    * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error10: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Pnp1_d,           NNp    * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error11: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Pnp1_prev_d,      NNp    * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error12: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Pdot_d,           NNp    * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error13: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   
   cudaStatus = cudaMalloc((void**)&Mrow_d,           NNZM   * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error14: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Mcol_d,           NNZM   * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error15: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&MrowStarts_d,     (NN+1) * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error16: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Grow_d,           NNZG   * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error17: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Gcol_d,           NNZG   * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error18: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&GrowStarts_d,     (NN+1) * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error19: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   cudaStatus = cudaMalloc((void**)&BCvelNodes_d,     BCnVelNodes * sizeof(int)); if(cudaStatus != cudaSuccess) { printf("Error20: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   cudaStatus = cudaMalloc((void**)&MdOrigInv_d,      3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error21: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&MdInv_d,          3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error22: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Un_d,             3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error23: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Unp1_d,           3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error24: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Unp1_prev_d,      3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error25: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&UnpHalf_d,        3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error26: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&R1_d,             3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error27: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&R2_d,             NNp    * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error28: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&R3_d,             3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error29: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   cudaStatus = cudaMemcpy(MrowStarts_d, sparseMrowStarts, (NN+1) * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error30: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(K_d,          sparseKvalue,     NNZM   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error31: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Mrow_d,       sparseMrow,       NNZM   * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error32: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Mcol_d,       sparseMcol,       NNZM   * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error33: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(G1_d,         sparseG1value,    NNZG   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error34: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(G2_d,         sparseG2value,    NNZG   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error35: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(G3_d,         sparseG3value,    NNZG   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error36: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   
   cudaStatus = cudaMemcpy(Grow_d,       sparseGrow,       NNZG   * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error37: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Gcol_d,       sparseGcol,       NNZG   * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error38: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(GrowStarts_d, sparseGrowStarts, (NN+1) * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error39: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   cudaStatus = cudaMemcpy(MdInv_d,       MdInv,           3*NN   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error40: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(MdOrigInv_d,   MdOrigInv,       3*NN   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error41: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
    

   int LARGE = 30;
   cudaStatus = cudaMalloc((void**)&NmeshColors_d,      LARGE           * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error42: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   //cudaStatus = cudaMalloc((void**)&meshColors_d,       NE              * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error43: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&elementsOfColor_d,  NE              * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error43: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   //cudaStatus = cudaMalloc((void**)&sparseMapM_1d_d,    NE*NENv*NENv    * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error44: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&LtoGvel_1d_d,       NE*NENv*3       * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error45: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }      
   cudaStatus = cudaMalloc((void**)&Sv_1d_d,            NGP*NENv        * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error46: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&gDSv_1d_d,          NE*NGP*NENv*3   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error47: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&GQfactor_1d_d,      NE*NGP          * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error48: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   cudaStatus = cudaMemcpy(NmeshColors_d,     NmeshColors,     LARGE           * sizeof(int),      cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error49: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   //cudaStatus = cudaMemcpy(meshColors_d,      meshColors,      NE              * sizeof(int),      cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error50: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(elementsOfColor_d, elementsOfColor, NE              * sizeof(int),      cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error50: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   //cudaStatus = cudaMemcpy(sparseMapM_1d_d,   sparseMapM_1d,   NE*NENv*NENv    * sizeof(int),      cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error51: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(LtoGvel_1d_d,      LtoGvel_1d,      NE*NENv*3       * sizeof(int),      cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error52: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Sv_1d_d,           Sv_1d,           NGP*NENv        * sizeof(double),   cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error53: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(gDSv_1d_d,         gDSv_1d,         NE*NGP*NENv*3   * sizeof(double),   cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error54: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(GQfactor_1d_d,     GQfactor_1d,     NE*NGP          * sizeof(double),   cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error55: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }


   // Extract the 1st column of BCvelNodes and send it to the device.
   int *dummy;
   dummy = new int[BCnVelNodes];
   for(int i = 0; i < BCnVelNodes; i++) {
      dummy[i] = BCvelNodes[i][0];
   }
   cudaStatus = cudaMemcpy(BCvelNodes_d, dummy,  BCnVelNodes * sizeof(int), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error42: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   delete[] dummy;


   // Send Un to the GPU
   cudaStatus = cudaMemcpy(Un_d, Un, 3*NN * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error43: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Pn_d, Pn, NNp  * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error44: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   // Initialize Pdot
   cudaMemset(Pdot_d, 0, NNp*sizeof(double));

   cudaMemGetInfo(&freeGPUmemory, &totalGPUmemory);
   cout << endl;
   cout << "After initializeAndAllocateGPU() function, free GPU memory = " << freeGPUmemory << endl;

}  // End of function initializeAndAllocateGPU()





//========================================================================
void calculateZ_CUSP()
//========================================================================
{
   // Uses CUSP to calculates [Z] = Gt * MdInvOrig * G.
                                                                                                                                                      // TODO: Try to minimize host-device memory transfers
                                                                                                                                                      // TODO: Use "views" property of CUSP to minimize device memory usage.
   int NNZG = sparseG_NNZ/3;

   // Define G1, G2, G3 as COO matrices on the HOST
   cusp::coo_matrix<int, double, cusp::host_memory> G1_CUSP(NN, NNp, NNZG);
   cusp::coo_matrix<int, double, cusp::host_memory> G2_CUSP(NN, NNp, NNZG);
   cusp::coo_matrix<int, double, cusp::host_memory> G3_CUSP(NN, NNp, NNZG);

   // Copy COO vectors of G1, G2 and G3 to CUSP matrices
   thrust::copy(sparseGrow,    sparseGrow + NNZG,    G1_CUSP.row_indices.begin());
   thrust::copy(sparseGcol,    sparseGcol + NNZG,    G1_CUSP.column_indices.begin());
   thrust::copy(sparseG1value, sparseG1value + NNZG, G1_CUSP.values.begin());

   thrust::copy(sparseGrow,    sparseGrow + NNZG,    G2_CUSP.row_indices.begin());
   thrust::copy(sparseGcol,    sparseGcol +  NNZG,   G2_CUSP.column_indices.begin());
   thrust::copy(sparseG2value, sparseG2value + NNZG, G2_CUSP.values.begin());

   thrust::copy(sparseGrow,    sparseGrow + NNZG,    G3_CUSP.row_indices.begin());
   thrust::copy(sparseGcol,    sparseGcol +  NNZG,   G3_CUSP.column_indices.begin());
   thrust::copy(sparseG3value, sparseG3value + NNZG, G3_CUSP.values.begin());

   // Define traspose of G matrices on the HOST.
   cusp::coo_matrix<int, double, cusp::host_memory> G1t_CUSP(NNp, NN, NNZG);
   cusp::coo_matrix<int, double, cusp::host_memory> G2t_CUSP(NNp, NN, NNZG);
   cusp::coo_matrix<int, double, cusp::host_memory> G3t_CUSP(NNp, NN, NNZG);

   cusp::transpose(G1_CUSP, G1t_CUSP);
   cusp::transpose(G2_CUSP, G2t_CUSP);
   cusp::transpose(G3_CUSP, G3t_CUSP);


   // Multiply G1, G2 and G3 with MdOrigInv diagonal matrix.
   double *G1mod, *G2mod, *G3mod;  // These are values of G matrices multiplied with the diagonal MdOrigInv matrix.
   G1mod = new double[NNZG];
   G2mod = new double[NNZG];
   G3mod = new double[NNZG];

   for (int i = 0; i < NNZG; i++) {
      G1mod[i] = sparseG1value[i] * MdOrigInv[sparseGrow[i]];
      G2mod[i] = sparseG2value[i] * MdOrigInv[sparseGrow[i]];
      G3mod[i] = sparseG3value[i] * MdOrigInv[sparseGrow[i]];
   }


   // Copy these modified G values to device
   thrust::copy(G1mod, G1mod + NNZG, G1_CUSP.values.begin());
   thrust::copy(G2mod, G2mod + NNZG, G2_CUSP.values.begin());
   thrust::copy(G3mod, G3mod + NNZG, G3_CUSP.values.begin());


   // Multiply Gt * Gmod matrices one by one. First store the results to a dummy matrix
   // and them add them to Z_CUSP_COO
   cusp::coo_matrix<int, double, cusp::host_memory> dummy;
   cusp::coo_matrix<int, double, cusp::host_memory> Z_CUSP_COO;

   cusp::multiply(G1t_CUSP, G1_CUSP, dummy);
   Z_CUSP_COO = dummy;

   cusp::multiply(G2t_CUSP, G2_CUSP, dummy);
   cusp::add(Z_CUSP_COO, dummy, Z_CUSP_COO);
   
   cusp::multiply(G3t_CUSP, G3_CUSP, dummy);
   cusp::add(Z_CUSP_COO, dummy, Z_CUSP_COO);

   // Convert Z_CUSP_COO into CSR format
   cusp::csr_matrix<int, double, cusp::host_memory> Z_CUSP_CSR;
   Z_CUSP_CSR = Z_CUSP_COO;

   
   // Modify Z_CUSP_CSR for known pressures.
   int LARGE = 1000;                                                                                                                                 // TODO: How important is this LARGE value in solution accuracy and convergence rate of CG?
   if (zeroPressureNode > 0) {  // If node is negative it means we do not set pressure to zero at any node.
      // Multiply Z[node][node] by LARGE
      for (int j = Z_CUSP_CSR.row_offsets[zeroPressureNode]; j < Z_CUSP_CSR.row_offsets[zeroPressureNode + 1]; j++) {  // Go through row "zerpPressureNode" of [Z].
         if (Z_CUSP_CSR.column_indices[j] == zeroPressureNode) {   // Determine the position of the diagonal entry in column "zeroPressureNode"
            Z_CUSP_CSR.values[j] = Z_CUSP_CSR.values[j] * LARGE;
            break;
         }
      }
   }

   Z_CUSP_CSR_d = Z_CUSP_CSR;
   
   // CONTROL
   //cusp::print(Z_CUSP_CSR);
   cout << endl << " NNZ of Z_CUSP_CSR = " << Z_CUSP_CSR.num_entries << endl;

   cudaMemGetInfo(&freeGPUmemory, &totalGPUmemory);
   cout << endl;
   cout << "At the end of calculateZ_CUSP() function, free GPU memory = " << freeGPUmemory << endl;


   
   /*
   // Write Z_CUSP_CSR matrix to a file for further use by the MKL_CG solver in a different run.
   ZcsrFile = fopen((whichProblem + ".zCSR").c_str(), "wb");

   int *rowOffsets, *colIndices;
   double *values;
   
   rowOffsets = new int[NNp+1];
   colIndices = new int[Z_CUSP_CSR.num_entries];
   values = new double[Z_CUSP_CSR.num_entries];
   
   for (int i = 0; i < NNp + 1; i++) {
      rowOffsets[i] = Z_CUSP_CSR.row_offsets[i];
   }
   
   for (int i = 0; i < Z_CUSP_CSR.num_entries; i++) {
      colIndices[i] = Z_CUSP_CSR.column_indices[i];
      values[i] = Z_CUSP_CSR.values[i];
   }

   fwrite(&Z_CUSP_CSR.num_entries, sizeof(int),    size_t(1),                      ZcsrFile);
   fwrite(rowOffsets,              sizeof(int),    size_t(NNp+1),                  ZcsrFile);
   fwrite(colIndices,              sizeof(int),    size_t(Z_CUSP_CSR.num_entries), ZcsrFile);
   fwrite(values,                  sizeof(double), size_t(Z_CUSP_CSR.num_entries), ZcsrFile);
   
   fclose(ZcsrFile);

   delete[] rowOffsets;
   delete[] colIndices;
   delete[] values;
   */

}  // End of function calculateZ_CUSP()





//========================================================================
void CUSP_CG_solver()
//========================================================================
{
   // Solve the system of step 2 [Z]{Pdot}={R2} using CG.

   // Allocate right hand side vector RHS_d and solution vector soln_d in device memory.   
   thrust::device_ptr<double> wrapped_R2_d(R2_d);
   thrust::device_ptr<double> wrapped_Pdot_d(Pdot_d);
   
   typedef typename cusp::array1d_view< thrust::device_ptr<double> > DeviceValueArrayView;   
   
   DeviceValueArrayView RHS_d (wrapped_R2_d, wrapped_R2_d + NNp);
   DeviceValueArrayView soln_d (wrapped_Pdot_d, wrapped_Pdot_d + NNp);  
   
   // Set monitor
   cusp::default_monitor<double> monitor(RHS_d, 1000, 1e-6);
   //cusp::verbose_monitor<double> monitor(RHS_d, 1000, 1e-12);

   // Set preconditioner
   cusp::precond::diagonal<double, cusp::device_memory> M(Z_CUSP_CSR_d);
   //cusp::identity_operator<double, cusp::device_memory> M(NNp, NNp);
   //cusp::precond::scaled_bridson_ainv<double, cusp::device_memory> M(Z_CUSP_CSR_d, .1);
   //cusp::precond::aggregation::smoothed_aggregation<int, double, cusp::device_memory> M(Z_CUSP_CSR_d);

   cusp::krylov::cg(Z_CUSP_CSR_d, soln_d, RHS_d, monitor, M);
   
   if (PRINT_TIMES) cout << "CUSP CG solver made " << monitor.iteration_count() << " iterations. Residual norm is " << monitor.residual_norm() << endl;

   // CONTROL
   //for(int i = 0; i < NNp; i++) {
   //   cout << Pdot[i] << endl;
   //}

}  // End of function CUSP_CG_solver()





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
__global__ void applyVelBC(int Nbc, int N, int *velBCdata, double *A)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
   // Change R1 for known velocity BCs
   int node;

   while (tid < Nbc) {
      node = velBCdata[tid];   // Node at which this velocity BC is specified.
     
      // Change R1 for the given u and v velocities.
      A[node]       = 0.0;   // This is not velocity, but velocity difference between 2 iterations.
      A[node + N]   = 0.0;   // This is not velocity, but velocity difference between 2 iterations.
      A[node + 2*N] = 0.0;   // This is not velocity, but velocity difference between 2 iterations.
      
      tid += blockDim.x * gridDim.x;      
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
      
      tid += blockDim.x * gridDim.x;
   }
}





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
__global__ void calcAndAssembleMatrixA(int NE, int NENv, int NGP, int NN, int offsetElements,
                                       int *elementsOfColor,
                                       int *LtoGvel,
                                       double *U, double *U_prev,
                                       double *Sv, double *gDSv, double *GQfactor,
                                       double *R1)
//========================================================================
{

   __shared__ double s_u[NENv_SIZE];
   __shared__ double s_v[NENv_SIZE];
   __shared__ double s_w[NENv_SIZE];
   
   __shared__ double s_u_prev[NENv_SIZE];
   __shared__ double s_v_prev[NENv_SIZE];
   __shared__ double s_w_prev[NENv_SIZE];
   
   __shared__ double s_u_GQ[NGP_SIZE];
   __shared__ double s_v_GQ[NGP_SIZE];
   __shared__ double s_w_GQ[NGP_SIZE];
   
   __shared__ double s_Sv[NGP_SIZE*NENv_SIZE];
   __shared__ double s_gDSv[NGP_SIZE*NENv_SIZE*DIM_SIZE];
   
   __shared__ double s_Ae[NENv_SIZE*NENv_SIZE];   
   
   __shared__ double s_R1e[3*NENv_SIZE];
   
   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   const int ebid = elementsOfColor[offsetElements + bid];

   // Extract elemental u, v and w velocity values from the global
   // solution array of the previous iteration.
   if (tid < NENv) {
      const int iLtoGu = LtoGvel[tid + ebid*NENv*3];
      const int iLtoGv = LtoGvel[tid + ebid*NENv*3 + NENv];
      const int iLtoGw = LtoGvel[tid + ebid*NENv*3 + NENv*2];       
      s_u[tid] = U[iLtoGu];
      s_v[tid] = U[iLtoGv];
      s_w[tid] = U[iLtoGw];
      s_u_prev[tid] = U_prev[iLtoGu];
      s_v_prev[tid] = U_prev[iLtoGv];
      s_w_prev[tid] = U_prev[iLtoGw];   
   }
   
   //// CONTROL
   //if (ebid == 0) {
      //if (tid < NENv) {
            //printf("s_u[%d] = %f \n", tid, s_u[tid]);
            //printf("s_v[%d] = %f \n", tid, s_v[tid]);
            //printf("s_w[%d] = %f \n", tid, s_w[tid]);
      //}
   //}
   ////===========================================   
   
   // Copy shape functions to shared memory 
   if (tid < NENv) {
      for (int k = 0; k < NGP; k++) {
         const int iSv = NENv*k+tid;
         s_Sv[iSv] = Sv[iSv];
      }
   }
   
   //// CONTROL
   //if (ebid == 0) {
      //if (tid < NENv) {
         //for (int k = 0; k < NGP; k++) {
            //const int iSv = NENv*k+tid;
            //printf("s_sV[%d] = %f \n", iSv, s_Sv[iSv]);
         //}
      //}
   //}
   ////===========================================
   
   // Copy elements gDSv to shared memory
   if (tid < NENv) {
      for (int k = 0; k < NGP; k++) {
         for (int i = 0; i < 3; i++) { 
            const int iGDSv1 = ebid*NENv*NGP*3+k*NENv*3+tid*3+i;
            const int iGDSv2 = k*NENv*3+tid*3+i;
            s_gDSv[iGDSv2] = gDSv[iGDSv1];
         }
      }  
   }
   
   //// CONTROL
   //if (ebid == 0) {
      //if (tid < NENv) {
         //for (int k = 0; k < NGP; k++) {
            //for (int i = 0; i < 3; i++) { 
               //const int iGDSv2 = k*NENv*3+tid*3+i;
               //printf("s_gDSv[%d] = %f \n", iGDSv2, s_gDSv[iGDSv2]);
            //}
         //}  
      //}
   //}
   ////===========================================
   
   // Initialize elemental stiffness matrix Ae
   if (tid < NENv) {
      for (int i = 0; i < NENv; i++) {
         const int iAe = i*NENv+tid;
         s_Ae[iAe] = 0.00000;
      }  
   }      
   
   // Initialize elemental stiffness matrix R1e
   if (tid < NENv) {
      s_R1e[tid]        = 0.00000;
      s_R1e[tid+NENv]   = 0.00000;
      s_R1e[tid+2*NENv] = 0.00000;
   }
   

	if (tid < NGP) {
	// Above calculated u0 and v0 values are at the nodes. However in GQ
	// integration we need them at GQ points.
	
	   // Initialize velocities at GQ points
       s_u_GQ[tid] = 0.00000;
       s_v_GQ[tid] = 0.00000;
       s_w_GQ[tid] = 0.00000;
       
	   for (int i = 0; i < NENv; i++) {
	      const int iSv = tid*NENv+i;
	      s_u_GQ[tid] += s_Sv[iSv] * s_u[i];
	      s_v_GQ[tid] += s_Sv[iSv] * s_v[i];
	      s_w_GQ[tid] += s_Sv[iSv] * s_w[i];
	   }
	}
   
   //// CONTROL - NOT COMPLETE
   //if (ebid == 0) {
      //if (tid < NGP) {
         
         //// Above calculated u0 and v0 values are at the nodes. However in GQ
         //// integration we need them at GQ points.
         //for (int j = 0; j < NENv; j++) {       
            //for (int i = 0; i < NENv; i++) {
               //const int iSv  = tid*NENv+i;
               //const int iuGQ = tid*NENv+j;
               //s_u_GQ[iuGQ] += s_Sv[iSv] * s_u[i];
               //s_v_GQ[iuGQ] += s_Sv[iSv] * s_v[i];
               //s_w_GQ[iuGQ] += s_Sv[iSv] * s_w[i];
            //}
         //}
      //}
   //}
   ////===========================================  
   
     
   // Calculate elemental stiffnes matrix   
   if (tid < NENv) {
      
      for (int k = 0; k < NGP; k++) {
         
         const double GQfactorThread = GQfactor[ebid*NGP+k];
             
               
         for (int i = 0; i < NENv; i++) {
            const int iGDSv = k*NENv*3+tid*3;
            const int iAe = i*NENv+tid;
            const int iSv = k*NENv+i;
            
            s_Ae[iAe] += (s_u_GQ[k] * s_gDSv[iGDSv] +
                          s_v_GQ[k] * s_gDSv[iGDSv+1] + 
                          s_w_GQ[k] * s_gDSv[iGDSv+2]) * s_Sv[iSv] * GQfactorThread;
         
         }
      }
      
   }
   

   // Calculate elemental R1 (right hand side vector), R1e
   if (tid < NENv) {
      for (int i = 0; i < NENv; i++) {
         const int iAe = tid*NENv+i;
         s_R1e[tid]        += s_Ae[iAe] * s_u_prev[i];    
         s_R1e[tid+NENv]   += s_Ae[iAe] * s_v_prev[i];     
         s_R1e[tid+2*NENv] += s_Ae[iAe] * s_w_prev[i];     
      }
   }
   
   if (tid < NENv) { 
      const int iLtoGu = LtoGvel[tid + ebid*NENv*3];
      const int iLtoGv = LtoGvel[tid + ebid*NENv*3 + NENv];
      const int iLtoGw = LtoGvel[tid + ebid*NENv*3 + NENv*2];
      R1[iLtoGu] -= s_R1e[tid];    
      R1[iLtoGv] -= s_R1e[tid+NENv];     
      R1[iLtoGw] -= s_R1e[tid+2*NENv]; 
   }   
   
   
   //if (ebid == 0) {
      //if (tid < NENv) {
         //for (int i = 0; i < NENv; i++) {
            //const int iAe = i*NENv+tid;
            //Ae_control[iAe] = s_Ae[iAe];
         //}  
      //}
   //}
   
}  // End of function calcAndAssembleMatrixA()  





//========================================================================
void calculateMatrixAGPU()
//========================================================================
{
   // Calculate Ae and multiply them with Ue.

   //int offset_gDSv_1d_d;
   //int offset_GQfactor_1d_d;
   //int offset_SparseMapM_1d_d;
   int offsetElements = 0;
   int nBlocksColor;
   int nThreadsPerBlock = 32;
   
   //cudaMemset(A_d, 0, sparseM_NNZ/3*sizeof(double));
   cudaMemset(R1_d, 0, 3*NN*sizeof(double));
   
   for (int color = 0; color < nActiveColors; color++) {
      
      nBlocksColor = NmeshColors[color];
      
      //cout << endl;
      //cout << "offset_" << NmeshColors[color] << " = " << offsetElements << endl;
      
      calcAndAssembleMatrixA<<<nBlocksColor,nThreadsPerBlock>>>(NE, NENv, NGP, NN, offsetElements,
                                                                elementsOfColor_d,
                                                                LtoGvel_1d_d,  
                                                                Un_d, UnpHalf_prev_d,
                                                                Sv_1d_d, gDSv_1d_d, GQfactor_1d_d,
                                                                R1_d);
                                                                
      offsetElements += NmeshColors[color];
      
   }
   
}  // End of function calculateMatrixAGPU()





//========================================================================
void step1GPUpart(int iter)
//========================================================================
{
   // CUSPARSE Reference: docs.nvidia.com/cuda/cusparse/index.html#appendix-b-cusparse-library-c---example


   // Calculate the RHS vector of step 1.
   // R1 = - K * UnpHalf_prev - A * UnpHalf_prev - G * Pn;

   int NNZM = sparseM_NNZ / 3;
   int NNZG = sparseG_NNZ / 3;

   double alpha = -1.000000000000000;
   double beta = 1.000000000000000;
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, UnpHalf_prev_d, &beta, R1_d);                // Part of (- K * UnpHalf_prev)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, UnpHalf_prev_d + NN, &beta, R1_d + NN);      // Part of (- K * UnpHalf_prev)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, UnpHalf_prev_d + 2*NN, &beta, R1_d + 2*NN);  // Part of (- K * UnpHalf_prev)

   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G1_d, GrowStarts_d, Gcol_d, Pn_d, &beta, R1_d);                        // Part of (- G1 * Pn)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G2_d, GrowStarts_d, Gcol_d, Pn_d, &beta, R1_d + NN);                   // Part of (- G1 * Pn)
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G3_d, GrowStarts_d, Gcol_d, Pn_d, &beta, R1_d + 2*NN);                 // Part of (- G1 * Pn)

   applyVelBC<<<NBLOCKS,NTHREADS>>>(BCnVelNodes, NN, BCvelNodes_d, R1_d);

   // Calculate UnpHalf
   calculate_UnpHalf<<<NBLOCKS,NTHREADS>>>(3*NN, dt, UnpHalf_d, Un_d, R1_d, MdInv_d);

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
   cudaStatus = cudaMalloc((void**)&dummy_d, 3*NN * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error102: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   
   double oneOverdt2 = 1.0000000000000000 / (dt*dt);

   if (iter == 1) {
      calculate_step2dummyV1<<<NBLOCKS,NTHREADS>>>(3*NN, oneOverdt2, dummy_d, UnpHalf_d);
   } else {
      calculate_step2dummyV2<<<NBLOCKS,NTHREADS>>>(3*NN, oneOverdt2, dummy_d, UnpHalf_d, MdOrigInv_d, KtimesAcc_prev_d);
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

   
   // Use CUSP's CG solver to solve [Z] {Pdot}= {R2} system

   CUSP_CG_solver();  // Calculate Pdot

   calculate_Pnp1<<<NBLOCKS,NTHREADS>>>(NNp, dt, Pnp1_d, Pn_d, Pdot_d);        // Pnp1 = Pn + dt * Pdot                                                       // TODO: Use CUBLAS function
   
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
      calculate_R3<<<NBLOCKS,NTHREADS>>>(3*NN, dt, R3_d, KtimesAcc_prev_d);
   }

   applyVelBC<<<NBLOCKS,NTHREADS>>>(BCnVelNodes, NN, BCvelNodes_d, R3_d);

   // Calculate Acc (Acc = R3 * MdInv)
   multiplyVectors<<<NBLOCKS,NTHREADS>>>(3*NN, Acc_d, R3_d, MdInv_d);                                                                                         // TODO: Use CUBLAS function
   
   // Calculate Unp1 (Unp1 = UnpHalf + dt * Acc)
   calculate_Unp1<<<NBLOCKS,NTHREADS>>>(3*NN, dt, Unp1_d, UnpHalf_d, Acc_d);                                                                                  // TODO: Use CUBLAS function

   //cudaStatus = cudaMemcpy(Unp1, Unp1_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error107: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   
}  // End of function step3GPU()





//========================================================================
void calculate_KtimesAcc_prevGPU(void)
//========================================================================
{
   int NNZM = sparseM_NNZ / 3;

   double alpha = 1.000000000000000;
   double beta = 0.000000000000000;
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, Acc_prev_d,        &beta, KtimesAcc_prev_d);         //  1st part of [K] * {Acc_prev}
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, Acc_prev_d + NN,   &beta, KtimesAcc_prev_d + NN);    //  2nd part of [K] * {Acc_prev}
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, K_d, MrowStarts_d, Mcol_d, Acc_prev_d + 2*NN, &beta, KtimesAcc_prev_d + 2*NN);  //  3rd part of [K] * {Acc_prev}
}  // End of function calculate_KtimesAcc_prevGPU()





//========================================================================
bool checkConvergenceGPU(void)
//========================================================================
{
   double norm1, norm2, normalizedNorm1, normalizedNorm2;

   // Calculate normalized norm for velocity
   cublasDnrm2(handleCUBLAS, 3*NN, Unp1_d, 1, &norm1);    // norm1 = sqrt(sum(Unp1(i)*Unp1(i)))

   double *dummy_d;
   cudaStatus = cudaMalloc((void**)&dummy_d, 3*NN * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error108: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }   // dummy_d will store (Unp1_d - Unp1_prev_d)

   subtractVectors<<<NBLOCKS,NTHREADS>>>(3*NN, dummy_d, Unp1_d, Unp1_prev_d);
   cublasDnrm2(handleCUBLAS, 3*NN, dummy_d, 1, &norm2);   // norm2 = sqrt(sum((Unp1(i)-Unp_prev(i))*(Unp1(i)-Unp_prev(i))))
   normalizedNorm1 = norm2 / norm1;                       // Normalized norm for velocity

   cudaFree(dummy_d);


   // Calculate normalized norm for pressure
   cublasDnrm2(handleCUBLAS, NNp, Pnp1_d, 1, &norm1);    // norm1 = sqrt(sum(Pnp1(i)*Pnp1(i)))

   cudaStatus = cudaMalloc((void**)&dummy_d, NNp * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error109: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }    // dummy_d will now store (Pnp1_d - Pnp1_prev_d)

   subtractVectors<<<NBLOCKS,NTHREADS>>>(NNp, dummy_d, Pnp1_d, Pnp1_prev_d);
   cublasDnrm2(handleCUBLAS, NNp, dummy_d, 1, &norm2);   // norm2 = sqrt(sum((Pnp1(i)-Pnp_prev(i))*(Pnp1(i)-Pnp_prev(i))))
   normalizedNorm2 = norm2 / norm1;                      // Normalized norm for pressure

   cudaFree(dummy_d);
   

   // Check convergence and get ready for the next iteration
   if (normalizedNorm1 < tolerance && normalizedNorm2 < tolerance) {
      return 1;
   } else {
      return 0;      
   }

}  // End of function checkConvergenceGPU()





//========================================================================
void printMonitorDataGPU(int iter)
//========================================================================
{
                                                                                                                                                     // TODO: Avoid the following device-to-host copies and print monitor data using device variables.
   //double *monitorData_d;
   //cudaMalloc((void**)&monitorData_d, 4 * sizeof(double));

   //getMonitorData<<<1,1>>>(monPoint, NN, Un_d, Pn_d, monitorData_d);

   //double *monitorData;
   //monitorData = new double[4];
   cudaStatus = cudaMemcpy(Un, Un_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error110: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Pn, Pn_d, NNp  * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error111: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   printf("%6d  %6d  %10.5f  %12.5f  %12.5f  %12.5f  %12.5f  %12.5f  %12.5f\n", 
          timeN, iter, timeT, Un[monPoint], Un[monPoint + NN], Un[monPoint + 2*NN], Pn[monPoint],  wallClockTimeCurrentTimeStep, maxAcc);
}  // End of function printMonitorDataGPU()





//========================================================================
bool checkConvergenceInTimeGPU(void)
//========================================================================
{
   double oneOverdt = 1.0000000000000000 / dt;
   
   // wrap raw pointer with a device_ptr 
   thrust::device_ptr<double> thrust_Un_dbeg(Un_d);
   thrust::device_ptr<double> thrust_Un_dend = thrust_Un_dbeg + 3*NN;
     
   thrust::device_ptr<double> thrust_Unp1_dbeg(Unp1_d);
   thrust::device_ptr<double> thrust_Unp1_dend = thrust_Unp1_dbeg + 3*NN;
   
   // do the reduction
   maxAcc = transform_reduce(make_zip_iterator(make_tuple(thrust_Unp1_dbeg, thrust_Un_dbeg)),
                             make_zip_iterator(make_tuple(thrust_Unp1_dend, thrust_Un_dend)),
                             weighted_absolute_difference(oneOverdt),
                             -1.f,
                             maximum<double>());
   

   // Check convergence
   if (maxAcc > convergenceCriteria) {
      return 0;
   } else {
      return 1;      
   }

}  // End of function checkConvergenceInTimeGPU()


#endif //USECUDA
