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
void CUSP_pC_CUSP_CG()
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
   
   //----------------------------------------------
   //-------------CONJUGATE GRADIENT---------------

   Start6 = getHighResolutionTime();     
   //----------------------------------------------
   // Solve pressure correction equation [4a] with CUSP's CG

   cusp::array1d<real2, cusp::device_memory> x(NN);

   // Copy previous solution to device memory
   thrust::copy(pPrime, pPrime + NN, x.begin());
   
   // Set stopping criteria:
   //cusp::verbose_monitor<real2> monitor(b, solverIterMax, solverTol);
   cusp::default_monitor<real2> monitor(Fsum, solverIterMax, solverTol);

   // Set preconditioner (identity)
   cusp::identity_operator<real2, cusp::device_memory> M(valx.num_rows, valx.num_rows);

   // Solve the linear system A * x = Fsum with the Conjugate Gradient method
   // cusp::krylov::bicgstab(A, x, Fsum, monitor, M);
   //int restart = 40;
   // cout << "Iterative solution is started." << endl;
   cusp::krylov::cg(valx, x, Fsum, monitor, M);
   // cout << "Iterative solution is finished." << endl;

   // Copy x from device back to u on host 
   thrust::copy(x.begin(), x.end(), pPrime);
   
   End6 = getHighResolutionTime();   
   printf("      Time for CG calculations            = %-.4g seconds.\n", End6 - Start6);     
   
   // report solver results
   if (monitor.converged())
   {
       std::cout << "      Solver converged to " << monitor.relative_tolerance() << " relative tolerance";
       std::cout << " after " << monitor.iteration_count() << " iterations" << endl;
   }
   else
   {
       std::cout << "      Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
       std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " << endl;
   }


}  // End of function CUSP_pressureCorrection()
