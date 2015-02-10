#include <stdio.h>
#include <iostream>
#include <cusparse.h>
#include <cublas.h>

using namespace std;

#ifdef SINGLE
  typedef float real2;
#else
  typedef double real2;
#endif

extern int   *rowStarts, *col, NN, solverIterMax, solverIter, bigNumber;
extern double solverTol, solverNorm;
extern real2 *u, *val, *F;
time_t start, end;




//-------------------------------------------------------------------------
void CUSPARSEsolver()
//-------------------------------------------------------------------------
{
   int *d_col, *d_row;
   real2 a, b, r0, r1;
   real2 *d_val, *d_x;
   real2 *d_r, *d_p, *d_Ax;

   int i, k;
   real2 *val_real2, *F_real2;
   
   //-------------------------------------------------------------------------------
   // Converting val and F values from double to real2 
   val_real2 = new real2[rowStarts[NN]];
   for(i=0; i<rowStarts[NN]; i++) {
      val_real2[i] = real2(val[i]);
   }

   F_real2 = new real2[NN];
   for(i=0; i<NN; i++) {
      F_real2[i] = real2(F[i])*bigNumber;
   }
   //------------------------------------------------------------------------------- 

   cusparseHandle_t handle = 0;
   cusparseStatus_t status;
   status = cusparseCreate(&handle);
   if (status != CUSPARSE_STATUS_SUCCESS) {
      fprintf( stderr, "!!!! CUSPARSE initialization error\n" );
   }
   cusparseMatDescr_t descr = 0;
   status = cusparseCreateMatDescr(&descr); 
   if (status != CUSPARSE_STATUS_SUCCESS) {
      fprintf( stderr, "!!!! CUSPARSE cusparseCreateMatDescr error\n" );
   } 

   cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
   
   for(i=0; i<NN; i++) {
      u[i] = 0.0;
   }

   cudaMalloc((void**)&d_col, (rowStarts[NN])*sizeof(int));
   cudaMalloc((void**)&d_row, (NN+1)*sizeof(int));
   cudaMalloc((void**)&d_val, (rowStarts[NN])*sizeof(real2));
   cudaMalloc((void**)&d_x,  NN*sizeof(real2));  
   cudaMalloc((void**)&d_r, NN*sizeof(real2));
   cudaMalloc((void**)&d_p, NN*sizeof(real2));
   cudaMalloc((void**)&d_Ax, NN*sizeof(real2));

   cudaMemcpy(d_col, col, (rowStarts[NN])*sizeof(int), cudaMemcpyHostToDevice);  
   cudaMemcpy(d_row, rowStarts, (NN+1)*sizeof(int), cudaMemcpyHostToDevice);  
   cudaMemcpy(d_val, val_real2, (rowStarts[NN])*sizeof(real2), cudaMemcpyHostToDevice); 
   cudaMemcpy(d_x, u, NN*sizeof(real2), cudaMemcpyHostToDevice);
   cudaMemcpy(d_r, F_real2, NN*sizeof(real2), cudaMemcpyHostToDevice);

   #ifdef SINGLE
      cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, 1.0, descr, d_val, d_row, d_col, d_x, 0.0, d_Ax);
      cublasSaxpy(NN, -1.0, d_Ax, 1, d_r, 1);
      r1 = cublasSdot(NN, d_r, 1, d_r, 1);
   #else
      cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, 1.0, descr, d_val, d_row, d_col, d_x, 0.0, d_Ax);
      cublasDaxpy(NN, -1.0, d_Ax, 1, d_r, 1);
      r1 = cublasDdot(NN, d_r, 1, d_r, 1);
   #endif
   r0=0;
   
   k = 1;
   while (r1 > solverTol*solverTol && k <= solverIterMax) {
      if (k > 1) {
         b = r1 / r0;
         #ifdef SINGLE
            cublasSscal(NN, b, d_p, 1);
            cublasSaxpy(NN, 1.0, d_r, 1, d_p, 1);
         #else
            cublasDscal(NN, b, d_p, 1);
            cublasDaxpy(NN, 1.0, d_r, 1, d_p, 1);
         #endif
      } else {
         #ifdef SINGLE
            cublasScopy(NN, d_r, 1, d_p, 1);
         #else
            cublasDcopy(NN, d_r, 1, d_p, 1);
         #endif
      }

      #ifdef SINGLE
         cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, 1.0, descr, d_val, d_row, d_col, d_p, 0.0, d_Ax);
         a = r1 / cublasSdot(NN, d_p, 1, d_Ax, 1);
         cublasSaxpy(NN, a, d_p, 1, d_x, 1);
         cublasSaxpy(NN, -a, d_Ax, 1, d_r, 1);

         r0 = r1;
         r1 = cublasSdot(NN, d_r, 1, d_r, 1);
      #else
         cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, 1.0, descr, d_val, d_row, d_col, d_p, 0.0, d_Ax);
         a = r1 / cublasDdot(NN, d_p, 1, d_Ax, 1);
         cublasDaxpy(NN, a, d_p, 1, d_x, 1);
         cublasDaxpy(NN, -a, d_Ax, 1, d_r, 1);

         r0 = r1;
         r1 = cublasDdot(NN, d_r, 1, d_r, 1);
      #endif

      cudaThreadSynchronize();
      k++;
   }
   
   //-------------------------------------------------------------------------------
   // Writes CG solution answers
   cudaMemcpy(u, d_x, (NN)*sizeof(real2), cudaMemcpyDeviceToHost);
   cout << endl;
   //cout << endl;
   //for(i=0; i<NN; i++) {
   //   printf("%f \n", u[i]);
   //}
   //cout << endl;
   cout <<"number of iterations: "<< k << endl;
   //-------------------------------------------------------------------------------

   cusparseDestroy(handle);

   cudaFree(d_col);
   cudaFree(d_row);
   cudaFree(d_val);
   cudaFree(d_x);
   cudaFree(d_r);
   cudaFree(d_p);
   cudaFree(d_Ax);
}

