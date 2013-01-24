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

//-----------------------------------------------------------------------------
void CUSP_pC_CUSP_CG()
//-----------------------------------------------------------------------------
{
   //OPERATIONS FOR THE LHS OF THE EQUATION [4a]
   
   //---------------------------------------------- 
   // Copy C_x, C_y, C_z from host to device 
   // Allocate stifness matrix C_x, C_y, C_z in CSR format
   cusp::csr_matrix<int, real2, cusp::device_memory> CCx(NN, NN, NNZ);
   cusp::csr_matrix<int, real2, cusp::device_memory> CCy(NN, NN, NNZ);
   cusp::csr_matrix<int, real2, cusp::device_memory> CCz(NN, NN, NNZ);   
   // Copy CSR row pointers to device memory
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,CCx.row_offsets.begin());
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,CCy.row_offsets.begin());
   thrust::copy(rowStartsSmall,rowStartsSmall + NN + 1,CCz.row_offsets.begin());   
   // Copy CSR column indices to device memory
   thrust::copy(colSmall,colSmall +  NNZ,CCx.column_indices.begin());
   thrust::copy(colSmall,colSmall +  NNZ,CCy.column_indices.begin());
   thrust::copy(colSmall,colSmall +  NNZ,CCz.column_indices.begin());
   // Copy CSR values to device memory
   thrust::copy(Cx,Cx + NNZ,CCx.values.begin()); 
   thrust::copy(Cy,Cy + NNZ,CCy.values.begin()); 
   thrust::copy(Cz,Cz + NNZ,CCz.values.begin()); 
   //---------------------------------------------- 
      
   //---------------------------------------------- 
   // Copy K_u^(-1), K_v^(-1), K_w^(-1) from host to device 
   // Allocate stifness matrix K_u^(-1), K_v^(-1), K_w^(-1) in CSR format   
   cusp::csr_matrix<int, real2, cusp::device_memory> uDiagonal_CUSP(NN, NN, NN);
   cusp::csr_matrix<int, real2, cusp::device_memory> vDiagonal_CUSP(NN, NN, NN);
   cusp::csr_matrix<int, real2, cusp::device_memory> wDiagonal_CUSP(NN, NN, NN);
   // Copy CSR row pointers to device memory
   thrust::copy(rowStartsDiagonal,rowStartsDiagonal + NN + 1,uDiagonal_CUSP.row_offsets.begin());
   thrust::copy(rowStartsDiagonal,rowStartsDiagonal + NN + 1,vDiagonal_CUSP.row_offsets.begin());
   thrust::copy(rowStartsDiagonal,rowStartsDiagonal + NN + 1,wDiagonal_CUSP.row_offsets.begin());   
   // Copy CSR column indices to device memory
   thrust::copy(colDiagonal,colDiagonal +  NN,uDiagonal_CUSP.column_indices.begin());
   thrust::copy(colDiagonal,colDiagonal +  NN,vDiagonal_CUSP.column_indices.begin());
   thrust::copy(colDiagonal,colDiagonal +  NN,wDiagonal_CUSP.column_indices.begin());
   // Copy CSR values to device memory
   thrust::copy(uDiagonal,uDiagonal + NN,uDiagonal_CUSP.values.begin()); 
   thrust::copy(vDiagonal,vDiagonal + NN,vDiagonal_CUSP.values.begin()); 
   thrust::copy(wDiagonal,wDiagonal + NN,wDiagonal_CUSP.values.begin()); 
   //----------------------------------------------     

   //---------------------------------------------- 
   // LHS of the equation [4a]
   // transpose(C_x), transpose(C_y), transpose(C_z) 
   cusp::csr_matrix<int, real2, cusp::device_memory> CxT;
   cusp::csr_matrix<int, real2, cusp::device_memory> CyT;
   cusp::csr_matrix<int, real2, cusp::device_memory> CzT;
   cusp::transpose(CCx, CxT);
   cusp::transpose(CCy, CyT);
   cusp::transpose(CCz, CzT);
   //----------------------------------------------      

   //---------------------------------------------- 
   // LHS of the equation [4a]
   // transpose(C_x)*(diagonal(K_u)^-1, transpose(C_y)*(diagonal(K_v)^-1, transpose(C_z)*(diagonal(K_w)^-1
   cusp::csr_matrix<int, real2, cusp::device_memory> CxTdia;
   cusp::csr_matrix<int, real2, cusp::device_memory> CyTdia;
   cusp::csr_matrix<int, real2, cusp::device_memory> CzTdia;
   cusp::multiply(CxT, uDiagonal_CUSP, CxTdia);
   cusp::multiply(CyT, vDiagonal_CUSP, CyTdia);
   cusp::multiply(CzT, wDiagonal_CUSP, CzTdia);
   //----------------------------------------------    

   //----------------------------------------------   
   // LHS of the equation [4a]
   // [transpose(C_x)*(diagonal(K_u)^-1]*C_x 
   // \________________________________/
   //          from above (CxTdia)   
   cusp::csr_matrix<int, real2, cusp::device_memory> valx;
   cusp::csr_matrix<int, real2, cusp::device_memory> valy;   
   cusp::csr_matrix<int, real2, cusp::device_memory> valz;   
   cusp::multiply(CxTdia, CCx, valx);
   cusp::multiply(CyTdia, CCy, valy);
   cusp::multiply(CzTdia, CCz, valz);
   //----------------------------------------------    
   
   //----------------------------------------------   
   // LHS of the equation [4a]  
   // summing x, y, z components
   // [transpose(C_x)*(diagonal(K_u)^-1]*C_x + [transpose(C_y)*(diagonal(K_v)^-1]*C_y + [transpose(C_z)*(diagonal(K_w)^-1]*C_z
   cusp::blas::axpy(valz.values,valy.values,1);
   cusp::blas::axpy(valy.values,valx.values,1);
   //---------------------------------------------- 
   
   
   //OPERATIONS FOR THE RHS OF THE EQUATION [4a]
   
   // Copy velocities from host to device memory
   cusp::array1d<real2, cusp::device_memory> u_CUSP(NN);
   cusp::array1d<real2, cusp::device_memory> v_CUSP(NN);  
   cusp::array1d<real2, cusp::device_memory> w_CUSP(NN);
   thrust::copy(u, u + NN, u_CUSP.begin());   
   thrust::copy(v, v + NN, v_CUSP.begin());
   thrust::copy(w, w + NN, w_CUSP.begin());
   //----------------------------------------------  
   // RHS of the equation [4a]  
   // -transpose(C_x)*u - transpose(C_y)*v - transpose(C_z)*w
   //  \______________/   \______________/   \______________/
   //        - F1       -         F2       -        F3     
   cusp::array1d<real2, cusp::device_memory> F1(NN);
   cusp::array1d<real2, cusp::device_memory> F2(NN);
   cusp::array1d<real2, cusp::device_memory> F3(NN);
   cusp::array1d<real2, cusp::device_memory> Fsum(NN);
   cusp::multiply(CxT, u_CUSP, F1);
   cusp::multiply(CyT, v_CUSP, F2);
   cusp::multiply(CzT, w_CUSP, F3);

   cusp::blas::fill(Fsum,0.0);
   cusp::blas::axpy(F1,Fsum,-1); 
   cusp::blas::axpy(F2,Fsum,-1); 
   cusp::blas::axpy(F3,Fsum,-1); 
   //----------------------------------------------

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
   
   // report solver results
   if (monitor.converged())
   {
       std::cout << "Solver converged to " << monitor.relative_tolerance() << " relative tolerance";
       std::cout << " after " << monitor.iteration_count() << " iterations" << endl;
   }
   else
   {
       std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
       std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " << endl;
   }


}  // End of function CUSP_pressureCorrection()
