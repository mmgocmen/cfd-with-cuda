#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/bicg.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/gmres.h>
#include <cusp/transpose.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>
#include <cusp/blas.h>

#include <stdio.h>
#include <cusparse.h>
#include <cublas.h>

#include <sys/time.h>

using namespace std;

#ifdef SINGLE
  typedef float real2;
#else
  typedef double real2;
#endif

extern int *rowStartsSmall, *colSmall, NN, NNZ, solverIterMax;
extern double solverTol;
extern real2 *uDiagonal, *vDiagonal, *wDiagonal, *u, *v, *w;
extern real2 *Cx, *Cy, *Cz;
extern real2 *F, *pPrime;
extern int *rowStartsDiagonal, *colDiagonal;

double getHighResolutionTime();

//-----------------------------------------------------------------------------
void CUSP_pC_CUDA_CR()
//-----------------------------------------------------------------------------
{

   double Start6, End6, Start7, End7;
   
   Start6 = getHighResolutionTime();         
   //---------------------------------------------- 
   //calculate arrays for x dimension
   //---------------------------------------------- 

   Start7 = getHighResolutionTime();   
   //---------------------------------------------- 
   // Copy C_x from host to device
   // Allocate stifness matrix C_x in CSR format
   cusp::csr_matrix<int, real2, cusp::device_memory> CCx(NN, NN, NNZ);
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,CCx.row_offsets.begin());
   thrust::copy(colSmall,colSmall +  NNZ,CCx.column_indices.begin());
   thrust::copy(Cx,Cx + NNZ,CCx.values.begin());
   //---------------------------------------------- 
   
   //---------------------------------------------- 
   // transpose(C_x)
   cusp::csr_matrix<int, real2, cusp::device_memory> CxT;
   cusp::transpose(CCx, CxT);
   //---------------------------------------------- 
   End7 = getHighResolutionTime();    
   printf("         Time for transpose(C_x)             = %-.4g seconds.\n", End7 - Start7);       

   Start7 = getHighResolutionTime();     
   //---------------------------------------------- 
   // Copy K_u^(-1) from host to device 
   // Allocate stifness matrix K_u^(-1) in CSR format   
   cusp::csr_matrix<int, real2, cusp::device_memory> uDiagonal_CUSP(NN, NN, NN);
   thrust::copy(rowStartsDiagonal,rowStartsDiagonal + NN + 1,uDiagonal_CUSP.row_offsets.begin());
   thrust::copy(colDiagonal,colDiagonal +  NN,uDiagonal_CUSP.column_indices.begin());
   thrust::copy(uDiagonal,uDiagonal + NN,uDiagonal_CUSP.values.begin()); 
   //----------------------------------------------     

   //---------------------------------------------- 
   // Copy velocities from host to device memory
   cusp::array1d<real2, cusp::device_memory> u_CUSP(NN);   
   thrust::copy(u, u + NN, u_CUSP.begin()); 
   //----------------------------------------------
   
   //----------------------------------------------  
   // RHS of the equation [4a]  
   // transpose(C_x)*u 
   // \______________/ 
   //        F1             
   cusp::array1d<real2, cusp::device_memory> F1(NN);
   cusp::multiply(CxT, u_CUSP, F1);   
   cusp::array1d<real2, cusp::device_memory> Fsum(NN);
   cusp::blas::fill(Fsum,0.0);   
   cusp::blas::axpy(F1,Fsum,-1); 
   //---------------------------------------------- 
   
   {
      // create temporary empty matrix to delete array
      cusp::array1d<real2, cusp::device_memory> tmp(1);
      F1.swap(tmp);
   }    
   {
      // create temporary empty matrix to delete array
      cusp::array1d<real2, cusp::device_memory> tmp(1);
      u_CUSP.swap(tmp);
   }  
   End7 = getHighResolutionTime();    
   printf("         Time for sum [transpose(C_x)*u]     = %-.4g seconds.\n", End7 - Start7);    

   Start7 = getHighResolutionTime();       
   //---------------------------------------------- 
   // LHS of the equation [4a]
   // transpose(C_x)*(diagonal(K_u)^-1
   cusp::csr_matrix<int, real2, cusp::device_memory> CxTdia;
   cusp::multiply(CxT, uDiagonal_CUSP, CxTdia);
   //----------------------------------------------    
   
   {
      // create temporary empty matrix to delete array
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      CxT.swap(tmp);
   } 
   {
      // create temporary empty matrix
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      uDiagonal_CUSP.swap(tmp);
   }   
   End7 = getHighResolutionTime();   
   printf("         Time for [transpose(C_x)] * K_u     = %-.4g seconds.\n", End7 - Start7);  
   
   Start7 = getHighResolutionTime();    
   //----------------------------------------------   
   // LHS of the equation [4a]
   // [transpose(C_x)*(diagonal(K_u)^-1]*C_x 
   // \________________________________/
   //          from above (CxTdia)   
   cusp::csr_matrix<int, real2, cusp::device_memory> valx;   
   cusp::multiply(CxTdia, CCx, valx);   
   //----------------------------------------------    
   End7 = getHighResolutionTime();   
   printf("         Time for [trans(C_x)*K_u] * C_x     = %-.4g seconds.\n", End7 - Start7);     
   End6 = getHighResolutionTime();   
   printf("      Time for calc pC arrays for x dim   = %-.4g seconds.\n", End6 - Start6); 
   
   
   Start6 = getHighResolutionTime();      
   //---------------------------------------------- 
   //calculate arrays for y dimension
   //---------------------------------------------- 

   Start7 = getHighResolutionTime();   
   //---------------------------------------------- 
   // Copy C_y from host to device
   // Allocate stifness matrix C_y in CSR format
   cusp::csr_matrix<int, real2, cusp::device_memory> CCy(NN, NN, NNZ);
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,CCy.row_offsets.begin());
   thrust::copy(colSmall,colSmall +  NNZ,CCy.column_indices.begin());
   thrust::copy(Cy,Cy + NNZ,CCy.values.begin());
   //---------------------------------------------- 
   
   //---------------------------------------------- 
   // transpose(C_y)
   cusp::csr_matrix<int, real2, cusp::device_memory> CyT;
   cusp::transpose(CCy, CyT);
   //----------------------------------------------   
   End7 = getHighResolutionTime();    
   printf("         Time for transpose(C_y)             = %-.4g seconds.\n", End7 - Start7);   
      
   Start7 = getHighResolutionTime();         
   //---------------------------------------------- 
   // Copy K_v^(-1) from host to device 
   // Allocate stifness matrix K_v^(-1) in CSR format   
   cusp::csr_matrix<int, real2, cusp::device_memory> vDiagonal_CUSP(NN, NN, NN);
   thrust::copy(rowStartsDiagonal,rowStartsDiagonal + NN + 1,vDiagonal_CUSP.row_offsets.begin());
   thrust::copy(colDiagonal,colDiagonal +  NN,vDiagonal_CUSP.column_indices.begin());
   thrust::copy(vDiagonal,vDiagonal + NN,vDiagonal_CUSP.values.begin()); 
   //----------------------------------------------     

   //---------------------------------------------- 
   // Copy velocities from host to device memory
   cusp::array1d<real2, cusp::device_memory> v_CUSP(NN);   
   thrust::copy(v, v + NN, v_CUSP.begin()); 
   //----------------------------------------------
   
   //----------------------------------------------  
   // RHS of the equation [4a]  
   // transpose(C_y)*u 
   // \______________/ 
   //        F2             
   cusp::array1d<real2, cusp::device_memory> F2(NN);
   cusp::multiply(CyT, v_CUSP, F2);     
   cusp::blas::axpy(F2,Fsum,-1); 
   //---------------------------------------------- 
   
   {
      // create temporary empty matrix to delete array
      cusp::array1d<real2, cusp::device_memory> tmp(1);
      F2.swap(tmp);
   }    
   {
      // create temporary empty matrix to delete array
      cusp::array1d<real2, cusp::device_memory> tmp(1);
      v_CUSP.swap(tmp);
   }       
   End7 = getHighResolutionTime();    
   printf("         Time for sum [transpose(C_y)*v]     = %-.4g seconds.\n", End7 - Start7);
   
   Start7 = getHighResolutionTime();      
   //---------------------------------------------- 
   // LHS of the equation [4a]
   // transpose(C_y)*(diagonal(K_v)^-1
   cusp::csr_matrix<int, real2, cusp::device_memory> CyTdia;
   cusp::multiply(CyT, vDiagonal_CUSP, CyTdia);
   //----------------------------------------------    
   
   {
      // create temporary empty matrix to delete array
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      CyT.swap(tmp);
   } 
   {
      // create temporary empty matrix
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      vDiagonal_CUSP.swap(tmp);
   }
   End7 = getHighResolutionTime();   
   printf("         Time for [transpose(C_y)] * K_v     = %-.4g seconds.\n", End7 - Start7);   

   Start7 = getHighResolutionTime();   
   //----------------------------------------------   
   // LHS of the equation [4a]
   // [transpose(C_y)*(diagonal(K_v)^-1]*C_y 
   // \________________________________/
   //          from above (CyTdia)   
   cusp::csr_matrix<int, real2, cusp::device_memory> valy;   
   cusp::multiply(CyTdia, CCy, valy);   
   // summing x, y components
   // [transpose(C_x)*(diagonal(K_u)^-1]*C_x + [transpose(C_y)*(diagonal(K_v)^-1]*C_y
   cusp::blas::axpy(valy.values,valx.values,1);
   //----------------------------------------------  
   
   {
      // create temporary empty matrix
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      valy.swap(tmp);
   }  
   End7 = getHighResolutionTime();   
   printf("         Time for [trans(C_y)*K_v] * C_y     = %-.4g seconds.\n", End7 - Start7);     
   End6 = getHighResolutionTime();   
   printf("      Time for calc pC arrays for y dim   = %-.4g seconds.\n", End6 - Start6);    
   

   Start6 = getHighResolutionTime();    
   //---------------------------------------------- 
   //calculate arrays for z dimension
   //---------------------------------------------- 
   
   Start7 = getHighResolutionTime();   
   //---------------------------------------------- 
   // Copy C_z from host to device
   // Allocate stifness matrix C_y in CSR format
   cusp::csr_matrix<int, real2, cusp::device_memory> CCz(NN, NN, NNZ);
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,CCz.row_offsets.begin());
   thrust::copy(colSmall,colSmall +  NNZ,CCz.column_indices.begin());
   thrust::copy(Cz,Cz + NNZ,CCz.values.begin());
   //---------------------------------------------- 
   
   //---------------------------------------------- 
   // transpose(C_z)
   cusp::csr_matrix<int, real2, cusp::device_memory> CzT;
   cusp::transpose(CCz, CzT);
   //----------------------------------------------
   End7 = getHighResolutionTime();    
   printf("         Time for transpose(C_z)             = %-.4g seconds.\n", End7 - Start7);   
      
   Start7 = getHighResolutionTime();      
   //---------------------------------------------- 
   // Copy K_w^(-1) from host to device 
   // Allocate stifness matrix K_w^(-1) in CSR format   
   cusp::csr_matrix<int, real2, cusp::device_memory> wDiagonal_CUSP(NN, NN, NN);
   thrust::copy(rowStartsDiagonal,rowStartsDiagonal + NN + 1,wDiagonal_CUSP.row_offsets.begin());
   thrust::copy(colDiagonal,colDiagonal +  NN,wDiagonal_CUSP.column_indices.begin());
   thrust::copy(wDiagonal,wDiagonal + NN,wDiagonal_CUSP.values.begin()); 
   //----------------------------------------------     

   //---------------------------------------------- 
   // Copy velocities from host to device memory
   cusp::array1d<real2, cusp::device_memory> w_CUSP(NN);   
   thrust::copy(w, w + NN, w_CUSP.begin()); 
   //----------------------------------------------
   
   //----------------------------------------------  
   // RHS of the equation [4a]  
   // transpose(C_z)*u 
   // \______________/ 
   //        F2             
   cusp::array1d<real2, cusp::device_memory> F3(NN);
   cusp::multiply(CzT, w_CUSP, F3);     
   cusp::blas::axpy(F3,Fsum,-1); 
   //---------------------------------------------- 
   
   {
      // create temporary empty matrix to delete array
      cusp::array1d<real2, cusp::device_memory> tmp(1);
      F3.swap(tmp);
   }    
   {
      // create temporary empty matrix to delete array
      cusp::array1d<real2, cusp::device_memory> tmp(1);
      w_CUSP.swap(tmp);
   }
   End7 = getHighResolutionTime();    
   printf("         Time for sum [transpose(C_z)*w]     = %-.4g seconds.\n", End7 - Start7);   

   Start7 = getHighResolutionTime();   
   //---------------------------------------------- 
   // LHS of the equation [4a]
   // transpose(C_z)*(diagonal(K_w)^-1
   cusp::csr_matrix<int, real2, cusp::device_memory> CzTdia;
   cusp::multiply(CzT, wDiagonal_CUSP, CzTdia);
   //----------------------------------------------    
   
   {
      // create temporary empty matrix to delete array
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      CzT.swap(tmp);
   } 
   {
      // create temporary empty matrix to delete array
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      wDiagonal_CUSP.swap(tmp);
   }   
   End7 = getHighResolutionTime();   
   printf("         Time for [transpose(C_z)] * K_w     = %-.4g seconds.\n", End7 - Start7);   

   Start7 = getHighResolutionTime();   
   //----------------------------------------------   
   // LHS of the equation [4a]
   // [transpose(C_z)*(diagonal(K_w)^-1]*C_z 
   // \________________________________/
   //          from above (CzTdia)   
   cusp::csr_matrix<int, real2, cusp::device_memory> valz;   
   cusp::multiply(CzTdia, CCz, valz);   
   // summing x, y, z components
   // [transpose(C_x)*(diagonal(K_u)^-1]*C_x + [transpose(C_y)*(diagonal(K_v)^-1]*C_y + [transpose(C_z)*(diagonal(K_w)^-1]*C_z
   cusp::blas::axpy(valz.values,valx.values,1);
   //----------------------------------------------  
   
   {
      // create temporary empty matrix
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      valz.swap(tmp);
   }  
   End7 = getHighResolutionTime();   
   printf("         Time for [trans(C_z)*K_w] * C_z     = %-.4g seconds.\n", End7 - Start7);   
   End6 = getHighResolutionTime();   
   printf("      Time for calc pC arrays for z dim   = %-.4g seconds.\n", End6 - Start6); 
   
   
   Start6 = getHighResolutionTime();       
   // Copy resulting LHS and RHS vectors from device memory to host memory
   int *row_deltaP, *col_deltaP;
   real2 *val_deltaP, *F_deltaP;
   
   row_deltaP = new int[NN+1];
   col_deltaP = new int[valx.row_offsets[NN]];
   val_deltaP = new real2[valx.row_offsets[NN]];

   thrust::copy(valx.row_offsets.begin(), valx.row_offsets.end(), row_deltaP);
   thrust::copy(valx.column_indices.begin(), valx.column_indices.end(), col_deltaP);
   thrust::copy(valx.values.begin(), valx.values.end(), val_deltaP);
   {
      // create temporary empty matrix to delete array
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      valx.swap(tmp);
   }  
   
   F_deltaP = new real2[NN];
   thrust::copy(Fsum.begin(), Fsum.end(), F_deltaP);  
   {
      // create temporary empty matrix to delete array
      cusp::array1d<real2, cusp::device_memory> tmp(1);
      Fsum.swap(tmp);
   }    
   
   
   //----------------------------------------------
   //-------------CONJUGATE RESIDUAL---------------
   
   real2 a, b, r0, r1, residual;
   int k;
   int *d_col, *d_row;
   real2 *d_val, *d_x, *d_r, *d_p, *d_Ax, *d_Ar;  
   
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
   
   cudaMalloc((void**)&d_col, row_deltaP[NN]*sizeof(int)) ;
   cudaMalloc((void**)&d_row, (NN+1)*sizeof(int)) ;
   cudaMalloc((void**)&d_val, row_deltaP[NN]*sizeof(real2)) ;
   cudaMalloc((void**)&d_x, NN*sizeof(real2)) ;  
   cudaMalloc((void**)&d_r, NN*sizeof(real2)) ;
   cudaMalloc((void**)&d_p, NN*sizeof(real2)) ;
   cudaMalloc((void**)&d_Ax, NN*sizeof(real2)) ;
   cudaMalloc((void**)&d_Ar, NN*sizeof(real2)) ;  

   cudaMemcpy(d_col, col_deltaP, row_deltaP[NN]*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_row, row_deltaP, (NN+1)*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_val, val_deltaP, row_deltaP[NN]*sizeof(real2), cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, pPrime, NN*sizeof(real2), cudaMemcpyHostToDevice);
   cudaMemcpy(d_r, F_deltaP, NN*sizeof(real2), cudaMemcpyHostToDevice);
   
   delete[] col_deltaP;
   delete[] row_deltaP;
   delete[] val_deltaP;
   delete[] F_deltaP;
   End6 = getHighResolutionTime();   
   printf("      Time for init variables for CR      = %-.4g seconds.\n", End6 - Start6); 

   Start6 = getHighResolutionTime();    
   cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, 1.0, descr, d_val, d_row, d_col, d_x, 0.0, d_Ax);
   // y = alpha * op(A) * x + beta * y
   // cusparseDcsrmv(handle, cusparseOperation_t transA, m, n, alpha, descrA, *csrValA, *csrRowPtrA, *csrColIndA, *x, beta, *y )
   // descrA = matrix property of A
   // d_Ax = 1.0 * A * d_x + 0.0 * d_Ax
   
   cublasDaxpy(NN, -1.0, d_Ax, 1, d_r, 1);
   // cublasDaxpy(int n, *alpha, *x, incx, *y, incy)
   // y[j] = alpha * x[k] + y[j]
   // d_r = -1.0 * d_Ax + d_r

   cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, 1.0 , descr, d_val, d_row, d_col, d_r, 0.0, d_Ar);
   // y = alpha * op(A) * x + beta * y
   // cusparseDcsrmv(handle, cusparseOperation_t transA, m, n, alpha, descrA, *csrValA, *csrRowPtrA, *csrColIndA, *x, beta, *y )
   // descrA = matrix property of A  
   // d_Ar = 1.0 * A * d_r + 0.0 * d_Ar
      
   r1 = cublasDdot(NN, d_r, 1, d_Ar, 1);
   // result = cublasDdot(int n, *x, incx, *y, incy)
   // result = total(x[k] × y[j]) 
   // r1 = total(d_r[i] * d_Ar[i])
   
   residual = cublasDdot(NN, d_r, 1, d_r, 1);
   // result = cublasDdot(int n, *x, incx, *y, incy)
   // result = total(x[k] × y[j]) 
   // residual = total(d_r[i] * d_r[i])
   
   k = 1;
   while (residual > solverTol*solverTol && k <= solverIterMax) {
      if (k > 1) {
         b = r1 / r0;
 
         cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, b, descr, d_val, d_row, d_col, d_p, 0.0, d_Ax);
         // y = alpha * op(A) * x + beta * y
         // cusparseDcsrmv(handle, cusparseOperation_t transA, m, n, alpha, descrA, *csrValA, *csrRowPtrA, *csrColIndA, *x, beta, *y )
         // descrA = matrix property of A  
         // d_Ax = b * A * d_p + 0.0 * d_Ax 
         
         cublasDaxpy(NN, 1.0, d_Ar, 1, d_Ax, 1);
         // cublasDaxpy(int n, *alpha, *x, incx, *y, incy)
         // y[j] = alpha * x[k] + y[j]
         // d_Ax = 1.0 * d_Ar + d_Ax           
         
         cublasDscal(NN, b, d_p, 1);
         // cublasDscal(int n, *alpha, *x, incx)
         // x[j] = alpha * x[j]
         // d_p = b * d_p         
         
         cublasDaxpy(NN, 1.0, d_r, 1, d_p, 1);
         // cublasDaxpy(int n, *alpha, *x, incx, *y, incy)
         // y[j] = alpha * x[k] + y[j]
         // d_p = 1.0 * d_r + d_p           
           
      } else {
         cublasDcopy(NN, d_r, 1, d_p, 1);
         // d_p = d_r
         
         cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, 1.0, descr, d_val, d_row, d_col, d_p, 0.0, d_Ax);
         // y = alpha * op(A) * x + beta * y
         // cusparseDcsrmv(handle, cusparseOperation_t transA, m, n, alpha, descrA, *csrValA, *csrRowPtrA, *csrColIndA, *x, beta, *y )
         // descrA = matrix property of A  
         // d_Ax = 1.0 * A * d_p + 0.0 * d_Ax         
      }
      
      a = r1 / cublasDdot(NN, d_Ax, 1, d_Ax, 1);
      // a = r1 / total(d_Ax[i] * d_Ax[i]) 
      
      cublasDaxpy(NN, a, d_p, 1, d_x, 1);
      // cublasDaxpy(int n, *alpha, *x, incx, *y, incy)
      // y[j] = alpha * x[k] + y[j]
      // d_x = a * d_p + d_x   
      
      cublasDaxpy(NN, -a, d_Ax, 1, d_r, 1);
      // cublasDaxpy(int n, *alpha, *x, incx, *y, incy)
      // y[j] = alpha * x[k] + y[j]
      // d_r = -a * d_Ax + d_r      

      r0 = r1;
      
      cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, NN, NN, 1.0 , descr, d_val, d_row, d_col, d_r, 0.0, d_Ar);
      // y = alpha * op(A) * x + beta * y
      // cusparseDcsrmv(handle, cusparseOperation_t transA, m, n, alpha, descrA, *csrValA, *csrRowPtrA, *csrColIndA, *x, beta, *y )
      // descrA = matrix property of A  
      // d_Ar = 1.0 * A * d_r + 0.0 * d_Ar
      
      r1 = cublasDdot(NN, d_r, 1, d_Ar, 1);
      // result = cublasDdot(int n, *x, incx, *y, incy)
      // result = total(x[k] × y[j]) 
      // r1 = total(d_r[i] * d_Ar[i])  
      
      cudaThreadSynchronize();
      k++;
      
      residual = cublasDdot(NN, d_r, 1, d_r, 1);
      // result = cublasDdot(int n, *x, incx, *y, incy)
      // result = total(x[k] × y[j]) 
      // residual = total(d_r[i] * d_r[i])      
   }

   cudaMemcpy(pPrime, d_x, NN*sizeof(real2), cudaMemcpyDeviceToHost);
   End6 = getHighResolutionTime();   
   printf("      Time for CR calculations            = %-.4g seconds.\n", End6 - Start6);      
   
   cusparseDestroy(handle);
   cudaFree(d_col);
   cudaFree(d_row);
   cudaFree(d_val);
   cudaFree(d_x);
   cudaFree(d_r);
   cudaFree(d_p);
   cudaFree(d_Ax);   
   cudaFree(d_Ar);
   
   if (k > solverIterMax) {
      std::cout << "      Solver reached iteration limit " << k-1 << " before converging";      
      std::cout << " to " <<  solverTol ;
      std::cout << ", final residual is " << sqrt(residual) << endl;
   }
   else {
      std::cout << "      Solver converged to " << sqrt(residual) << " relative tolerance";
      std::cout << " after " << k-1 << " iterations" << endl;
   }

}  // End of function CUSP_pC_CUDA_CR()
