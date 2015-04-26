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
#include <cusp/coo_matrix.h>   // NON
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/gmres.h>
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

#ifdef SINGLE
  typedef float REAL;
#else
  typedef double REAL;
#endif

extern int NN, NNp, nBC, BCnVelNodes, zeroPressureNode, timeN, monPoint;
extern double dt, timeT, t_ini, tolerance, **BCstr;
extern double convergenceCriteria, maxAcc;
extern double wallClockTimeCurrentTimeStep;
extern int ** BCvelNodes;

extern int sparseM_NNZ, *sparseMrowStarts, *sparseMcol;
extern int sparseG_NNZ, *sparseGrowStarts, *sparseGcol;
extern int sparseZ_NNZ, *sparseZrowStarts, *sparseZcol;
extern double *sparseG1value, *sparseG2value, *sparseG3value;
extern double *sparseMvalue, *sparseKvalue, *sparseZvalue; 
extern double *Uk, *Uk_prev, *Pk, *Pk_prev, *Pk_prevprev, *Pk_diff; 
extern double *R1, *R11, *R12, *R13, *R2;
extern char dummyUserInput;
extern string whichProblem;

extern double *M_d, *K_d, *A_d, *G1_d, *G2_d, *G3_d, *Z_d;
extern double *Uk_d, *Uk_dir_d, *Uk_prev_d, *Pk_d, *Pk_prev_d, *Pk_prevprev_d, *Pdiff_d;
extern double *R1_d, *R2_d;
extern int *Mcol_d, *MrowStarts_d, *Gcol_d, *GrowStarts_d;
extern int *Zcol_d, *ZrowStarts_d;
extern int *BCvelNodes_d, *BCvelNodesWhichBC_d;
extern double *BCstr_1d_d;

extern cusparseHandle_t   handle;
extern cusparseMatDescr_t descr;
extern cublasHandle_t     handleCUBLAS;
extern cudaError_t        cudaStatus;

extern cusparseSolveAnalysisInfo_t analysisInfo1, analysisInfo2;

extern size_t freeGPUmemory, totalGPUmemory;   // To measure total and free GPU memory

extern bool PRINT_TIMES;

extern double getHighResolutionTime(int, double);
extern void createTecplot();


extern int NE, NGP, NENv;
extern int *NmeshColors, *meshColors, *elementsOfColor;
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

   cudaStatus = cudaMalloc((void**)&M_d,              NNZM        * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error01: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&K_d,              NNZM        * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error01: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&A_d,              NNZM        * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error01: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Z_d,              sparseZ_NNZ * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error01: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   cudaStatus = cudaMalloc((void**)&G1_d,             NNZG   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error03: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&G2_d,             NNZG   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error04: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&G3_d,             NNZG   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error05: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Pk_d,             NNp    * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error10: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Pk_prev_d,        NNp    * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error11: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Pk_prevprev_d,    NNp    * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error12: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Pdiff_d,          NNp    * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error13: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   
   cudaStatus = cudaMalloc((void**)&Mcol_d,           NNZM     * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error15: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&MrowStarts_d,     (NN+1)   * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error16: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Gcol_d,           NNZG     * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error18: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&GrowStarts_d,     (NN+1)   * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error19: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Zcol_d,           sparseZ_NNZ * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error19: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&ZrowStarts_d,     (NNp+1)     * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error19: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }


   cudaStatus = cudaMalloc((void**)&BCvelNodes_d,        BCnVelNodes * sizeof(int));    if(cudaStatus != cudaSuccess) { printf("Error20: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&BCvelNodesWhichBC_d, BCnVelNodes * sizeof(int));    if(cudaStatus != cudaSuccess) { printf("Error20: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&BCstr_1d_d,          3*nBC       * sizeof(double)); if(cudaStatus != cudaSuccess) { printf("Error20: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }


   cudaStatus = cudaMalloc((void**)&Uk_d,             3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error23: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Uk_dir_d,           NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error23: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&Uk_prev_d,        3*NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error24: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&R1_d,               NN   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error27: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&R2_d,             NNp    * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error28: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }


   cudaStatus = cudaMemcpy(MrowStarts_d, sparseMrowStarts, (NN+1) * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error30: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Mcol_d,       sparseMcol,       NNZM   * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error30: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(K_d,          sparseKvalue,     NNZM   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error31: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(M_d,          sparseMvalue,     NNZM   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error31: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   cudaStatus = cudaMemcpy(Gcol_d,       sparseGcol,       NNZG   * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error38: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(GrowStarts_d, sparseGrowStarts, (NN+1) * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error39: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(G1_d,         sparseG1value,    NNZG   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error34: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(G2_d,         sparseG2value,    NNZG   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error35: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(G3_d,         sparseG3value,    NNZG   * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error36: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; } 

   cudaStatus = cudaMemcpy(ZrowStarts_d, sparseZrowStarts, (NNp+1) * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error30: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Zcol_d,       sparseZcol,       sparseZ_NNZ  * sizeof(int),    cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error30: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Z_d,          sparseZvalue,     sparseZ_NNZ  * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error31: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   int LARGE = 30;
   cudaStatus = cudaMalloc((void**)&NmeshColors_d,      LARGE           * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error42: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   //cudaStatus = cudaMalloc((void**)&meshColors_d,       NE              * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error43: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&elementsOfColor_d,  NE              * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error43: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&sparseMapM_1d_d,    NE*NENv*NENv    * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error44: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&LtoGvel_1d_d,       NE*NENv*3       * sizeof(int));      if(cudaStatus != cudaSuccess) { printf("Error45: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }      
   cudaStatus = cudaMalloc((void**)&Sv_1d_d,            NGP*NENv        * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error46: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&gDSv_1d_d,          NE*NGP*NENv*3   * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error47: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMalloc((void**)&GQfactor_1d_d,      NE*NGP          * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error48: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   cudaStatus = cudaMemcpy(NmeshColors_d,     NmeshColors,     LARGE           * sizeof(int),      cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error49: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   //cudaStatus = cudaMemcpy(meshColors_d,      meshColors,      NE              * sizeof(int),      cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error50: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(elementsOfColor_d, elementsOfColor, NE              * sizeof(int),      cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error50: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(sparseMapM_1d_d,   sparseMapM_1d,   NE*NENv*NENv    * sizeof(int),      cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error51: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
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

   // Extract the 2nd column of BCvelNodes and send it to the device.
   int *dummy2;
   dummy2 = new int[BCnVelNodes];
   for(int i = 0; i < BCnVelNodes; i++) {
      dummy2[i] = BCvelNodes[i][1];
   }
   cudaStatus = cudaMemcpy(BCvelNodesWhichBC_d, dummy2,  BCnVelNodes * sizeof(int), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error42: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   delete[] dummy2;

   // Copy BCstr values to BCstr_1d_d
   double *dummy3;
   dummy3 = new double[3*nBC];
   for(int i = 0; i < nBC; i++) {
      dummy3[3*i]   = BCstr[i][0];
      dummy3[3*i+1] = BCstr[i][1];
      dummy3[3*i+2] = BCstr[i][2];
   }
   cudaStatus = cudaMemcpy(BCstr_1d_d, dummy3, 3*nBC * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error42: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   delete[] dummy3;
   
   // Send Uk and Pk to the GPU
   cudaStatus = cudaMemcpy(Uk_prev_d, Uk_prev, 3*NN * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error43: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Pk_prev_d, Pk_prev, NNp  * sizeof(double), cudaMemcpyHostToDevice);   if(cudaStatus != cudaSuccess) { printf("Error44: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaMemset(Pk_prevprev_d, 0, NNp*sizeof(double));   
   
   cudaMemGetInfo(&freeGPUmemory, &totalGPUmemory);
   cout << endl;
   cout << "After initializeAndAllocateGPU() function, free GPU memory = " << freeGPUmemory << endl;

}  // End of function initializeAndAllocateGPU()





//========================================================================
void CUSP_CG_solver()
//========================================================================
{
   // Solve the system of step 2 [Z]{Pdiff}={R2} using CG.

   cudaMemset(Pdiff_d, 0, NNp*sizeof(double));
   
   // Wrap Z in order to use at cusp fuctions

   // *NOTE* raw pointers must be wrapped with thrust::device_ptr!
   thrust::device_ptr<int> wrapped_ZrowStarts_d(ZrowStarts_d);
   thrust::device_ptr<int> wrapped_Zcol_d(Zcol_d);
   thrust::device_ptr<double> wrapped_Z_d(Z_d);
   thrust::device_ptr<double> wrapped_R2_d(R2_d);
   thrust::device_ptr<double> wrapped_Pdiff_d(Pdiff_d); 
   // use array1d_view to wrap the individual arrays
   typedef typename cusp::array1d_view< thrust::device_ptr<int> > DeviceIndexArrayView;
   typedef typename cusp::array1d_view< thrust::device_ptr<double> > DeviceValueArrayView;
 
   DeviceIndexArrayView row_offsetsZ (wrapped_ZrowStarts_d, wrapped_ZrowStarts_d + (NNp+1));
   DeviceIndexArrayView column_indicesZ (wrapped_Zcol_d, wrapped_Zcol_d + sparseZ_NNZ);
   DeviceValueArrayView valuesZ (wrapped_Z_d, wrapped_Z_d + sparseZ_NNZ);
   DeviceValueArrayView RHS_d (wrapped_R2_d, wrapped_R2_d + NNp);
   DeviceValueArrayView soln_d (wrapped_Pdiff_d, wrapped_Pdiff_d + NNp);   
   
   // combine the three array1d_views into a csr_matrix_view
   typedef cusp::csr_matrix_view<DeviceIndexArrayView,
                                 DeviceIndexArrayView,
                                 DeviceValueArrayView> DeviceView;
                                 
   DeviceView Z(NNp, NNp, sparseZ_NNZ, row_offsetsZ, column_indicesZ, valuesZ);   
   
   //CONTROL
   //double *sparseR2valueGPU; 
   //sparseR2valueGPU = new double[NNp];
   //cudaStatus = cudaMemcpy(sparseR2valueGPU, R2_d, NNp * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error100.1: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   //cout << "i   R2_gpu" << endl;
   //for (int i=0; i < NNp; i++) {
      //cout << i << "  " << sparseR2valueGPU[i] << endl;     
   //}    
   
   // Set monitor
   cusp::default_monitor<double> monitor(RHS_d, 1000, 1e-6);
   //cusp::verbose_monitor<real2> monitor(b, solverIterMax, solverTol);

   // Set preconditioner
   cusp::precond::diagonal<double, cusp::device_memory> M(Z);
   //cusp::identity_operator<double, cusp::device_memory> M(NNp, NNp);
   //cusp::precond::scaled_bridson_ainv<double, cusp::device_memory> M(Z_CUSP_CSR_d, .1);
   //cusp::precond::aggregation::smoothed_aggregation<int, double, cusp::device_memory> M(Z_CUSP_CSR_d);

   cusp::krylov::cg(Z, soln_d, RHS_d, monitor, M);
   
   if (PRINT_TIMES) cout << "CUSP CG solver made " << monitor.iteration_count() << " iterations. Residual norm is " << monitor.residual_norm() << endl;

   // CONTROL
   //for(int i = 0; i < NNp; i++) {
   //   cout << Pdot[i] << endl;
   //}

}  // End of function CUSP_CG_solver()





//-----------------------------------------------------------------------------
void CUSP_GMRES_solver()
//-----------------------------------------------------------------------------
{
   // Solve the system of step 1 [A]{U^k+1}={R1} using GMRES.

   int NNZM = sparseM_NNZ / 3;
   cudaMemset(Uk_dir_d, 0, NN*sizeof(double)); // Initialize U

   // Wrap A in order to use at cusp fuctions

   // *NOTE* raw pointers must be wrapped with thrust::device_ptr!
   thrust::device_ptr<int> wrapped_ArowStarts_d(MrowStarts_d);
   thrust::device_ptr<int> wrapped_Acol_d(Mcol_d);
   thrust::device_ptr<double> wrapped_A_d(A_d);
   thrust::device_ptr<double> wrapped_R1_d(R1_d);
   thrust::device_ptr<double> wrapped_Uk_d(Uk_dir_d);
   // use array1d_view to wrap the individual arrays
   typedef typename cusp::array1d_view< thrust::device_ptr<int> > DeviceIndexArrayView;
   typedef typename cusp::array1d_view< thrust::device_ptr<double> > DeviceValueArrayView;
 
   DeviceIndexArrayView row_offsetsA (wrapped_ArowStarts_d, wrapped_ArowStarts_d + (NN+1));
   DeviceIndexArrayView column_indicesA (wrapped_Acol_d, wrapped_Acol_d + NNZM);
   DeviceValueArrayView valuesA (wrapped_A_d, wrapped_A_d + NNZM);
   DeviceValueArrayView RHS1_d (wrapped_R1_d, wrapped_R1_d + NN);
   DeviceValueArrayView soln1_d (wrapped_Uk_d, wrapped_Uk_d + NN);    
   
   // combine the three array1d_views into a csr_matrix_view
   typedef cusp::csr_matrix_view<DeviceIndexArrayView,
                                 DeviceIndexArrayView,
                                 DeviceValueArrayView> DeviceView;
                                 
   DeviceView A(NN, NN, NNZM, row_offsetsA, column_indicesA, valuesA);   

   // Set stopping criteria:
   //cusp::verbose_monitor<double> monitor(b, solverIterMax, solverTol);
   cusp::default_monitor<double> monitor(RHS1_d, 1000, 1e-6);

   // Set preconditioner (identity)
   //cusp::identity_operator<double, cusp::device_memory> M(A_CUSP_CSR_d.num_rows, A_CUSP_CSR_d.num_rows);
   cusp::precond::diagonal<double, cusp::device_memory> M(A);
   //cusp::precond::aggregation::smoothed_aggregation<int, double, cusp::device_memory> M(A);
   
   //int restart = 100;
   // cout << "Iterative solution is started." << endl;
   //cusp::krylov::gmres(A, soln1_d, RHS1_d, restart, monitor);
   cusp::krylov::bicgstab(A, soln1_d, RHS1_d, monitor, M);
   // cout << "Iterative solution is finished." << endl;



   // report solver results
   if (monitor.converged())
   {
       std::cout << "Solver converged to " << monitor.relative_tolerance() << " relative tolerance";
       std::cout << " after " << monitor.iteration_count() << " iterations";
       std::cout << endl;       
   }
   else
   {
       std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
       std::cout << " to " << monitor.relative_tolerance() << " relative tolerance ";
       std::cout << endl;       
   }

}  // End of function CUSP_GMRES_solver()





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
__global__ void applyVelBC_RHS(int Nbc, int dirMom, int *velBCdata, int *whichBCdata, double *BCvalue, double *R)
//========================================================================
{
   
   int tid = threadIdx.x + blockIdx.x * blockDim.x;   
   // Change R1 for known velocity BCs
   int node, whichBC;

   while (tid < Nbc) {
      node = velBCdata[tid];          // Node at which this velocity BC is specified.
      whichBC = whichBCdata[tid];     // Number of the specified BC
      
      // Change R1 for the given u, v and w velocities.
      R[node] = BCvalue[whichBC*3 + dirMom];    // This is velocity.
      
      tid += blockDim.x * gridDim.x;      
   }
}





//========================================================================
__global__ void applyVelBC_LHS(int N, int Nbc, int *velBCdata, int NNZM,
                               int *rowStartsM, int *colM, double *valueA)
//========================================================================
{
   
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   int node, j;

   while (tid < Nbc) {   
   // Change stiffness matrix A for known velocity BCs
      node = velBCdata[tid];          // Node at which this velocity BC is specified.
      for (j = rowStartsM[node]; j < rowStartsM[node+1]; j++) {
         valueA[j] = 0.0;
         if(colM[j] == node) {                                      
            valueA[j] = 1.0;   
         }         
      }
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
__global__ void calculate_dummyPdiff(int N, double *Pprev, double *Pprevprev, double *Pdifference)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      Pdifference[tid] = 2*Pprev[tid] - Pprevprev[tid];

      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void calculate_MplusK(int NNZM, double *A, double *M, double *K)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < NNZM) {
      A[tid] = M[tid] + K[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void calculate_Pk(int N, double *Pk_d, double *Pk_prev_d, double *Pdiff_d)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      Pk_d[tid] = Pk_prev_d[tid] + Pdiff_d[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void copy_Uk(int N, int shift, double *UkDir, double *UkAll)
//========================================================================
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;

   while (tid < N) {

      UkAll[tid+shift] = UkDir[tid];
      
      tid += blockDim.x * gridDim.x;
   }
}





//========================================================================
__global__ void calcAndAssembleMatrixA(int NE, int NENv, int NGP, int NN, int offsetElements,
                                       int *elementsOfColor,
                                       int *LtoGvel, int *sparseMapM,
                                       double *U, double *Sv, double *gDSv, double *GQfactor,
                                       int NNZM, double *A)
//========================================================================
{

   __shared__ double s_u[NENv_SIZE];
   __shared__ double s_v[NENv_SIZE];
   __shared__ double s_w[NENv_SIZE];
   
   __shared__ double s_u_GQ[NGP_SIZE];
   __shared__ double s_v_GQ[NGP_SIZE];
   __shared__ double s_w_GQ[NGP_SIZE];
   
   __shared__ double s_Sv[NGP_SIZE*NENv_SIZE];
   __shared__ double s_gDSv[NGP_SIZE*NENv_SIZE*DIM_SIZE];
   
   __shared__ double s_Ae[NENv_SIZE*NENv_SIZE];
   
   __shared__ int s_sparseMapM[NENv_SIZE*NENv_SIZE];
   
   int tid = threadIdx.x;
   int bid = blockIdx.x;
   int ebid = elementsOfColor[offsetElements + bid];

   // Extract elemental u, v and w velocity values from the global
   // solution array of the previous iteration.
   if (tid < NENv) {
      const int iLtoGu = LtoGvel[tid + ebid*NENv*3];
      const int iLtoGv = LtoGvel[tid + ebid*NENv*3 + NENv];
      const int iLtoGw = LtoGvel[tid + ebid*NENv*3 + NENv*2];
      s_u[tid] = U[iLtoGu];
      s_v[tid] = U[iLtoGv];
      s_w[tid] = U[iLtoGw];
   }
   
   
   // Copy shape functions to shared memory 
   if (tid < NENv) {
      for (int k = 0; k < NGP; k++) {
         const int iSv = NENv*k+tid;
         s_Sv[iSv] = Sv[iSv];
      }
   }
   
   
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
     
   
   // Copy elemental local to global map to shared memory
   if (tid < NENv) {
      for (int i = 0; i < NENv; i++) {
         const int iMap1 = ebid*NENv*NENv+i*NENv+tid;
         const int iMap2 = i*NENv+tid;
         s_sparseMapM[iMap2] = sparseMapM[iMap1];
      }  
   }
     
   
   // Initialize elemental stiffness matrix Ae
   if (tid < NENv) {
      for (int i = 0; i < NENv; i++) {
         const int iAe = i*NENv+tid;
         s_Ae[iAe] = 0.00000;
      }  
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
   

   // Assemble stiffness matrix A
   if (tid < NENv) {
      for (int i = 0; i < NENv; i++) {
         const int iAe = i*NENv+tid;
         const int iA  = s_sparseMapM[iAe];
         A[iA] += s_Ae[iAe];
      }  
   }
   
}   





//========================================================================
void calculateMatrixAGPU()
//========================================================================
{
   // Calculates advectice stiffness matrix(B(U)) and adds it to LHS matrix of step one
   int NNZM = sparseM_NNZ / 3;   
   
   int offsetElements = 0;
   
   int nBlocksColor;
   int nThreadsPerBlock = 32;
      
   for (int color = 0; color < 8; color++) {
      
      nBlocksColor = NmeshColors[color];
      
      //cout << endl;
      //cout << "offset_" << NmeshColors[color] << " = " << offsetElements << endl;

      calcAndAssembleMatrixA<<<nBlocksColor,nThreadsPerBlock>>>(NE, NENv, NGP, NN, offsetElements,
                                                                elementsOfColor_d,
                                                                LtoGvel_1d_d, sparseMapM_1d_d,  
                                                                Uk_prev_d, Sv_1d_d, gDSv_1d_d, GQfactor_1d_d,
                                                                NNZM, A_d);
                                                                
      offsetElements += NmeshColors[color];
      
   }
   
}  // End of function step2GPU()





//========================================================================
void step1GPU()
//========================================================================
{
   // CUSPARSE Reference: docs.nvidia.com/cuda/cusparse/index.html#appendix-b-cusparse-library-c---example

   double Start, wallClockTime;
   
   // Calculate the LHS matrix of step 1.
   // A = [1/dt*M] + [vK] + [A(U)]

   int NNZM = sparseM_NNZ / 3;
   int NNZG = sparseG_NNZ / 3;
   
   Start = getHighResolutionTime(1, 1.0); 
   
   calculate_MplusK<<<NBLOCKS,NTHREADS>>>(NNZM, A_d, M_d, K_d); // Calculates 1/dtM + vK   
   
   //CONTROL
   //double *sparseAvalueGPU; 
   //sparseAvalueGPU = new double[sparseM_NNZ];
   //cudaStatus = cudaMemcpy(sparseAvalueGPU, A_d, sparseM_NNZ * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error100.1: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   //cout << "i   A_gpu" << endl;
   //for (int i=100; i < 150; i++) {
      //cout << i << "  " << sparseAvalueGPU[i] << endl;     
   //}     
   
   cudaThreadSynchronize();   
   
   calculateMatrixAGPU(); // Calculates B(stiffness matrix of advective term) and adds to LHS system matrix

   cudaThreadSynchronize();
   
   applyVelBC_LHS<<<NBLOCKS,NTHREADS>>>(NN, BCnVelNodes, BCvelNodes_d, NNZM,
                                        MrowStarts_d, Mcol_d, A_d);
   
   cudaThreadSynchronize();
   
   wallClockTime = getHighResolutionTime(2, Start);

   if (PRINT_TIMES) printf("calculate LHS system matrix() took %6.3f seconds.\n", wallClockTime);
   
   
   // Calculate the RHS vector and the LHS matrix of step 1.
   // R1 = [1/dt*M]*U^k - G*(2*P^k-P^(k-1))
   
   double *dummyPdiff_d;   // Stores 2*P^k-P^(k-1)
   cudaStatus = cudaMalloc((void**)&dummyPdiff_d, NNp * sizeof(double));   if(cudaStatus != cudaSuccess) { printf("Error102: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   calculate_dummyPdiff<<<NBLOCKS,NTHREADS>>>(NNp, Pk_prev_d, Pk_prevprev_d, dummyPdiff_d);
       
       
   for (int direction = 0; direction < 3; direction++) {

      Start = getHighResolutionTime(1, 1.0); 

      double alpha = 1.000000000000000;
      double beta = 0.000000000000000;
      
      switch (direction) {
         case 0:
            cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, M_d, MrowStarts_d, Mcol_d, Uk_prev_d, &beta, R1_d);        // Part of ((1/dt)M * Uk_prev) x-dir
            break;
         case 1:
            cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, M_d, MrowStarts_d, Mcol_d, Uk_prev_d + NN, &beta, R1_d);   // Part of ((1/dt)M * Uk_prev) y-dir
            break;
         case 2:
            cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, NNZM, &alpha, descr, M_d, MrowStarts_d, Mcol_d, Uk_prev_d + 2*NN, &beta, R1_d); // Part of ((1/dt)M * Uk_prev) z-dir
            break;
      }

      ////CONTROL
      //double *sparseR1valueGPU; 
      //sparseR1valueGPU = new double[3*NN];
      //cudaStatus = cudaMemcpy(sparseR1valueGPU, R1_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error100.1: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
      //cout << "i   R1_gpu" << endl;
      //for (int i=100; i < 150; i++) {
         //cout << i << "  " << sparseR1valueGPU[i] << endl << endl;     
      //}

      alpha = -1.000000000000000;
      beta = 1.000000000000000;
      
      switch (direction) {
         case 0:
            cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G1_d, GrowStarts_d, Gcol_d, dummyPdiff_d, &beta, R1_d); // Part of (- G1 * Pdiff) x-dir
            break;
         case 1:
            cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G2_d, GrowStarts_d, Gcol_d, dummyPdiff_d, &beta, R1_d); // Part of (- G2 * Pdiff) y-dir
            break;
         case 2:
            cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G3_d, GrowStarts_d, Gcol_d, dummyPdiff_d, &beta, R1_d); // Part of (- G3 * Pdiff) z-dir
            break;
      }      

      applyVelBC_RHS<<<NBLOCKS,NTHREADS>>>(BCnVelNodes, direction, BCvelNodes_d, BCvelNodesWhichBC_d, BCstr_1d_d, R1_d);
      
      cudaThreadSynchronize();
      
      wallClockTime = getHighResolutionTime(2, Start);
      if (PRINT_TIMES) printf("step1 RHS_%d() took %6.3f seconds.\n", direction, wallClockTime);
         
         
      Start = getHighResolutionTime(1, 1.0);
         
      CUSP_GMRES_solver();  // Calculate U^k
      
      int shiftDir = direction*NN;
      copy_Uk<<<NBLOCKS,NTHREADS>>>(NN, shiftDir, Uk_dir_d, Uk_d); // copy the velocities in one direction to global velocity array
      
      cudaThreadSynchronize();
            
      wallClockTime = getHighResolutionTime(2, Start);
      if (PRINT_TIMES) printf("CUSP_BiCGStab_solver_%d() took %6.3f seconds.\n", direction, wallClockTime);       
      
      //CONTROL
      //double *sparseUvalueGPU; 
      //sparseUvalueGPU = new double[3*NN];
      //cudaStatus = cudaMemcpy(sparseUvalueGPU, Uk_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error100.1: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
      //cout << "i   Uk_gpu" << endl;
      //for (int i=500; i < 729; i++) {
         //cout << i << "  " << sparseUvalueGPU[i] << endl;    
      //} 
      //delete[] sparseUvalueGPU;
   }
   
   cudaFree(dummyPdiff_d);  
      
    
   
}  // End of function step1GPU()





//========================================================================
void step2GPU()
//========================================================================
{
   // Executes step 2 of the method to determine pressure of the new time step.

   // Calculate the RHS vector of step 2.
   // R2 = -1/dt * Gt * U^k+1
      
   double oneOverdt = -1.0000000000000000 / (dt);

   int NNZG = sparseG_NNZ / 3;

   double alpha = oneOverdt;
   double beta = 0.000000000000000;
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G1_d, GrowStarts_d, Gcol_d, Uk_d,        &beta, R2_d);  // 1st part of -1/dt * [Gt] * {Uk}
   beta = 1.000000000000000;
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G2_d, GrowStarts_d, Gcol_d, Uk_d + NN,   &beta, R2_d);  // 2nd part of -1/dt * [Gt] * {Uk}
   cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, NN, NNp, NNZG, &alpha, descr, G3_d, GrowStarts_d, Gcol_d, Uk_d + 2*NN, &beta, R2_d);  // 3rd part of -1/dt * [Gt] * {uk}

   applyPresBC<<<1,1>>>(zeroPressureNode, R2_d);

//double *R2_dummy;
//R2_dummy = new double[NNp];
//cudaStatus = cudaMemcpy(R2_dummy, R2_d, NNp * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error100.1: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
//cout << "i   R2_gpu" << endl;
//for (int i=0; i < 125; i++) {
   //cout << i << "  " << R2_dummy[i] << endl;     
//}
//delete [] R2_dummy;

   
   // Use CUSP's CG solver to solve [Z] {Pdiff}= {R2} system

   CUSP_CG_solver();  // Calculate Pdot

   calculate_Pk<<<NBLOCKS,NTHREADS>>>(NNp, Pk_d, Pk_prev_d, Pdiff_d);        // Pk = Pk_prev + Pdiff                                                       // TODO: Use CUBLAS function

   //CONTROL
   //double *sparsePvalueGPU; 
   //sparsePvalueGPU = new double[NNp];
   //cudaStatus = cudaMemcpy(sparsePvalueGPU, Pk_d, NNp * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error100.1: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   //cout << "i   Pk_gpu" << endl;
   //for (int i=0; i < NNp; i++) {
      //cout << i << "  " << sparsePvalueGPU[i] << endl;    
   //} 
   //delete[] sparsePvalueGPU;
   
}  // End of function step2GPU()





//========================================================================
void printMonitorDataGPU()
//========================================================================
{
                                                                                                                                                     // TODO: Avoid the following device-to-host copies and print monitor data using device variables.
   //double *monitorData_d;
   //cudaMalloc((void**)&monitorData_d, 4 * sizeof(double));

   //getMonitorData<<<1,1>>>(monPoint, NN, Un_d, Pn_d, monitorData_d);

   //double *monitorData;
   //monitorData = new double[4];
   cudaStatus = cudaMemcpy(Uk, Uk_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error110: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
   cudaStatus = cudaMemcpy(Pk, Pk_d, NNp  * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error111: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }

   printf("%6d  %10.5f  %12.5f  %12.5f  %12.5f  %12.5f  %12.5f  %12.5f\n", 
          timeN, timeT, Uk[monPoint], Uk[monPoint + NN], Uk[monPoint + 2*NN], Pk[monPoint], wallClockTimeCurrentTimeStep, maxAcc);   

}  // End of function printMonitorDataGPU()





//========================================================================
bool checkConvergenceInTimeGPU(void)
//========================================================================
{
   double oneOverdt = 1.0000000000000000 / dt;
   
   // wrap raw pointer with a device_ptr 
   thrust::device_ptr<double> thrust_Un_dbeg(Uk_prev_d);
   thrust::device_ptr<double> thrust_Un_dend = thrust_Un_dbeg + 3*NN;
     
   thrust::device_ptr<double> thrust_Unp1_dbeg(Uk_d);
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
