#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/precond/diagonal.h>
#include <cusp/relaxation/jacobi.h>
#include <cusp/relaxation/polynomial.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/cr.h>
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
extern real2 *K_u_diagonal, *K_v_diagonal, *K_w_diagonal, *u, *v, *w;
extern real2 *Cx, *Cy, *Cz;
extern real2 *CxT, *CyT, *CzT;
extern real2 *delta_p;
extern int *rowStartsDiagonal, *colDiagonal;
extern real2 *F_deltaP, *val_deltaP;
extern int *row_deltaP, *col_deltaP;
extern int iter;

void applyBC_deltaP();
double getHighResolutionTime();

//-----------------------------------------------------------------------------
void CUSP_pC_CUSP_CR()
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
   cusp::csr_matrix<int, real2, cusp::device_memory> Cx_CUSP(NN, NN, NNZ);
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,Cx_CUSP.row_offsets.begin());
   thrust::copy(colSmall,colSmall +  NNZ,Cx_CUSP.column_indices.begin());
   thrust::copy(Cx,Cx + NNZ,Cx_CUSP.values.begin());
   //---------------------------------------------- 
   
   //---------------------------------------------- 
   // Copy transpose(C_x) from host to device
   // Allocate stifness matrix transpose(C_x) in CSR format
   cusp::csr_matrix<int, real2, cusp::device_memory> CxT_CUSP(NN, NN, NNZ);
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,CxT_CUSP.row_offsets.begin());
   thrust::copy(colSmall,colSmall +  NNZ,CxT_CUSP.column_indices.begin());
   thrust::copy(CxT,CxT + NNZ,CxT_CUSP.values.begin());
   //---------------------------------------------- 
   End7 = getHighResolutionTime();    
   printf("         Time for copy Cx and CxT            = %-.4g seconds.\n", End7 - Start7);       

   Start7 = getHighResolutionTime();     
   //---------------------------------------------- 
   // Copy K_u^(-1) from host to device 
   // Allocate stifness matrix K_u^(-1) in CSR format   
   cusp::csr_matrix<int, real2, cusp::device_memory> K_u_diagonal_CUSP(NN, NN, NN);
   thrust::copy(rowStartsDiagonal,rowStartsDiagonal + NN + 1,K_u_diagonal_CUSP.row_offsets.begin());
   thrust::copy(colDiagonal,colDiagonal +  NN,K_u_diagonal_CUSP.column_indices.begin());
   thrust::copy(K_u_diagonal,K_u_diagonal + NN,K_u_diagonal_CUSP.values.begin()); 
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
   cusp::multiply(CxT_CUSP, u_CUSP, F1);   
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
   cusp::multiply(CxT_CUSP, K_u_diagonal_CUSP, CxTdia);
   //----------------------------------------------    
   
   {
      // create temporary empty matrix to delete array
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      CxT_CUSP.swap(tmp);
   } 
   {
      // create temporary empty matrix
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      K_u_diagonal_CUSP.swap(tmp);
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
   cusp::multiply(CxTdia, Cx_CUSP, valx);   
   // cout << "NNZ K pressure correction = " << valx.row_offsets[NN] << endl;
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
   cusp::csr_matrix<int, real2, cusp::device_memory> Cy_CUSP(NN, NN, NNZ);
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,Cy_CUSP.row_offsets.begin());
   thrust::copy(colSmall,colSmall +  NNZ,Cy_CUSP.column_indices.begin());
   thrust::copy(Cy,Cy + NNZ,Cy_CUSP.values.begin());
   //---------------------------------------------- 
   
   //---------------------------------------------- 
   // Copy transpose(C_y) from host to device
   // Allocate stifness matrix transpose(C_y) in CSR format
   cusp::csr_matrix<int, real2, cusp::device_memory> CyT_CUSP(NN, NN, NNZ);
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,CyT_CUSP.row_offsets.begin());
   thrust::copy(colSmall,colSmall +  NNZ,CyT_CUSP.column_indices.begin());
   thrust::copy(CyT,CyT + NNZ,CyT_CUSP.values.begin());
   //---------------------------------------------- 
   End7 = getHighResolutionTime();    
   printf("         Time for copy Cy and CyT            = %-.4g seconds.\n", End7 - Start7);    
      
   Start7 = getHighResolutionTime();         
   //---------------------------------------------- 
   // Copy K_v^(-1) from host to device 
   // Allocate stifness matrix K_v^(-1) in CSR format   
   cusp::csr_matrix<int, real2, cusp::device_memory> K_v_diagonal_CUSP(NN, NN, NN);
   thrust::copy(rowStartsDiagonal,rowStartsDiagonal + NN + 1,K_v_diagonal_CUSP.row_offsets.begin());
   thrust::copy(colDiagonal,colDiagonal +  NN,K_v_diagonal_CUSP.column_indices.begin());
   thrust::copy(K_v_diagonal,K_v_diagonal + NN,K_v_diagonal_CUSP.values.begin()); 
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
   cusp::multiply(CyT_CUSP, v_CUSP, F2);     
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
   cusp::multiply(CyT_CUSP, K_v_diagonal_CUSP, CyTdia);
   //----------------------------------------------    
   
   {
      // create temporary empty matrix to delete array
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      CyT_CUSP.swap(tmp);
   } 
   {
      // create temporary empty matrix
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      K_v_diagonal_CUSP.swap(tmp);
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
   cusp::multiply(CyTdia, Cy_CUSP, valy);   
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
   cusp::csr_matrix<int, real2, cusp::device_memory> Cz_CUSP(NN, NN, NNZ);
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,Cz_CUSP.row_offsets.begin());
   thrust::copy(colSmall,colSmall +  NNZ,Cz_CUSP.column_indices.begin());
   thrust::copy(Cz,Cz + NNZ,Cz_CUSP.values.begin());
   //---------------------------------------------- 
   
   //---------------------------------------------- 
   // Copy transpose(C_z) from host to device
   // Allocate stifness matrix transpose(C_z) in CSR format
   cusp::csr_matrix<int, real2, cusp::device_memory> CzT_CUSP(NN, NN, NNZ);
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,CzT_CUSP.row_offsets.begin());
   thrust::copy(colSmall,colSmall +  NNZ,CzT_CUSP.column_indices.begin());
   thrust::copy(CzT,CzT + NNZ,CzT_CUSP.values.begin());
   //---------------------------------------------- 
   End7 = getHighResolutionTime();    
   printf("         Time for copy Cz and CzT            = %-.4g seconds.\n", End7 - Start7);    
      
   Start7 = getHighResolutionTime();      
   //---------------------------------------------- 
   // Copy K_w^(-1) from host to device 
   // Allocate stifness matrix K_w^(-1) in CSR format   
   cusp::csr_matrix<int, real2, cusp::device_memory> K_w_diagonal_CUSP(NN, NN, NN);
   thrust::copy(rowStartsDiagonal,rowStartsDiagonal + NN + 1,K_w_diagonal_CUSP.row_offsets.begin());
   thrust::copy(colDiagonal,colDiagonal +  NN,K_w_diagonal_CUSP.column_indices.begin());
   thrust::copy(K_w_diagonal,K_w_diagonal + NN,K_w_diagonal_CUSP.values.begin()); 
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
   cusp::multiply(CzT_CUSP, w_CUSP, F3);     
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
   cusp::multiply(CzT_CUSP, K_w_diagonal_CUSP, CzTdia);
   //----------------------------------------------    
   
   {
      // create temporary empty matrix to delete array
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      CzT_CUSP.swap(tmp);
   } 
   {
      // create temporary empty matrix to delete array
      cusp::csr_matrix<int,real2,cusp::device_memory> tmp(1,1,1);
      K_w_diagonal_CUSP.swap(tmp);
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
   cusp::multiply(CzTdia, Cz_CUSP, valz);   
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

   if (iter==1){   
      val_deltaP = new real2[valx.row_offsets[NN]];
      F_deltaP = new real2[NN];
      row_deltaP = new int[NN+1];
      col_deltaP = new int[valx.row_offsets[NN]];      
   }

   thrust::copy(valx.row_offsets.begin(), valx.row_offsets.end(), row_deltaP);
   thrust::copy(valx.column_indices.begin(), valx.column_indices.end(), col_deltaP);
   thrust::copy(valx.values.begin(), valx.values.end(), val_deltaP);
   
   thrust::copy(Fsum.begin(), Fsum.end(), F_deltaP);  
   {
      // create temporary empty matrix to delete array
      cusp::array1d<real2, cusp::device_memory> tmp(1);
      Fsum.swap(tmp);
   }
   applyBC_deltaP();

   thrust::copy(val_deltaP,val_deltaP + NNZ,valx.values.begin());   
   
   cusp::array1d<real2, cusp::device_memory> F(NN);
   thrust::copy(F_deltaP,F_deltaP + NN,F.begin());
   
   End6 = getHighResolutionTime();   
   printf("      Time for init variables for CR      = %-.4g seconds.\n", End6 - Start6);
   
   if (iter!=1) {   
      //----------------------------------------------
      //-------------CONJUGATE GRADIENT---------------

      Start6 = getHighResolutionTime();     
      //----------------------------------------------
      // Solve pressure correction equation [4a] with CUSP's CG

      cusp::array1d<real2, cusp::device_memory> x(NN);

      // Copy previous solution to device memory
      thrust::copy(delta_p, delta_p + NN, x.begin());
      
      // Set stopping criteria:
      //cusp::verbose_monitor<real2> monitor(b, solverIterMax, solverTol);
      cusp::default_monitor<real2> monitor(F, solverIterMax, solverTol);

      // Set preconditioner 
      // 1) identity
      //cusp::identity_operator<real2, cusp::device_memory> M(valx.num_rows, valx.num_rows);    
      // 2) smoothed aggregation preconditioner and jacobi smoother
      //cusp::precond::aggregation::smoothed_aggregation<int, real2, cusp::device_memory> M(valx);    
      // 3) smoothed aggregation preconditioner and polynomial smoother
	   //typedef cusp::relaxation::polynomial<int,cusp::device_memory> Smoother;      
      //cusp::precond::aggregation::smoothed_aggregation<int, real2, cusp::device_memory, Smoother> M(valx);    
      // 4) diagonal preconditioner
      cusp::precond::diagonal<real2, cusp::device_memory> M(valx);      
      
      // Solve the linear system A * x = Fsum with the Conjugate Gradient method
      // cusp::krylov::bicgstab(A, x, Fsum, monitor, M);
      //int restart = 40;
      // cout << "Iterative solution is started." << endl;
      cusp::krylov::cr(valx, x, F, monitor, M);
      // cout << "Iterative solution is finished." << endl;

      // Copy x from device back to u on host 
      thrust::copy(x.begin(), x.end(), delta_p);
      
      End6 = getHighResolutionTime();   
      printf("      Time for CR calculations            = %-.4g seconds.\n", End6 - Start6);     
      
      // report solver results
      if (monitor.converged())
      {
         std::cout << "      Solver converged to " << monitor.relative_tolerance() << " relative tolerance";
         std::cout << " after " << monitor.iteration_count() << " iterations";
         std::cout << " (" << monitor.residual_norm() << " final residual)" << endl;   
      }
      else
      {
         std::cout << "      Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
         std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " ;
         std::cout << " (" << monitor.residual_norm() << " final residual)" << endl;  
      }
   }

}  // End of function CUSP_pressureCorrection()
