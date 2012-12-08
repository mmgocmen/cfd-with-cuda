#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
//#include <cusp/krylov/gmres.h>
#include <cusp/precond/diagonal.h>
#include <cusp/precond/ainv.h>
#include <cusp/precond/smoothed_aggregation.h>

using namespace std;

#ifdef SINGLE
  typedef float real2;
#else
  typedef double real2;
#endif

extern int   *rowStarts, *col, NN, NNZ, solverIterMax, solverIter;
extern double solverTol, solverNorm;
extern real2 *u, *val, *F;




//-----------------------------------------------------------------------------
void CUSPsolver()
//-----------------------------------------------------------------------------
{
   // Solve system of linear equations using an iterative solver of CUSP on a GPU.

   //cout << endl << "Start of CUSPsolver() function." << endl;
   //cout << "NN = " << NN << endl;
   //cout << "NNZ  = " << NNZ << endl;

   // Allocate stifness matrix [A] in CSR format and right hand side vector {b}
   // and solution vector {x} in device memory.
   cusp::csr_matrix<int, real2, cusp::device_memory> A(NN, NN, NNZ);
   cusp::array1d<real2, cusp::device_memory> b(NN);
   cusp::array1d<real2, cusp::device_memory> x(NN);

   // Copy CSR row pointers to device memory
   thrust::copy(rowStarts, rowStarts + NN + 1, A.row_offsets.begin());

   // Copy CSR column indices to device memory
   thrust::copy(col, col +  NNZ, A.column_indices.begin());

   // Copy CSR values to device memory
   thrust::copy(val, val + NNZ, A.values.begin()); 

   // Copy right hand side vector to device memory
   thrust::copy(F, F + NN, b.begin());

   // Copy previous solution to device memory
   thrust::copy(u, u + NN, x.begin());
   
   //cusp::verbose_monitor<real2> monitor(b, solverIterMax, solverTol);
   cusp::default_monitor<real2> monitor(b, solverIterMax, solverTol);

   // Set preconditioner
   // cusp::identity_operator<real2, cusp::device_memory> M(A.num_rows, A.num_rows);
   //cusp::precond::diagonal<real2, cusp::device_memory> M(A);
   //cusp::precond::scaled_bridson_ainv<real2, cusp::device_memory> M(A, .1);
   //cusp::precond::smoothed_aggregation<int, real2, cusp::device_memory> M(A);

   cusp::krylov::cg(A, x, b, monitor);
   //cusp::krylov::cg(A, x, b, monitor);


   // Copy x from device back to u on host 
   thrust::copy(x.begin(), x.end(), u);
   
   solverIter = monitor.iteration_count();
   solverNorm = monitor.residual_norm();

   cout << "number of iterations: " << solverIter << endl;

   // ----------------------CONTROL------------------------
   // Print the solution to check
   // for(int i=0; i<=NN; i++) {
   //    cout << "u[" << i << "] = " << u[i] << endl;
   // }
   // ----------------------CONTROL------------------------

   //cout << "End of CUSPsolver() function." << endl;

}  // End of function CUSPSolver()


