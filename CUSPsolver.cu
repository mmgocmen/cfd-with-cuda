#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/bicg.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/gmres.h>

using namespace std;

#ifdef SINGLE
  typedef float real2;
#else
  typedef double real2;
#endif

extern int   *rowStarts, *col, Ndof, NNZ, solverIterMax;
extern double solverTol;
extern real2 *u, *val, *F;


//-----------------------------------------------------------------------------
void CUSPsolver()
//-----------------------------------------------------------------------------
{
   // Solve system of linear equations using an iterative solver of CUSP on a GPU.

   //cout << endl << "Start of CUSPsolver() function." << endl;
   //cout << "Ndof = " << Ndof << endl;
   //cout << "NNZ  = " << NNZ << endl;

   // Allocate stifness matrix [A] in CSR format and right hand side vector {b}
   // and solution vector {x} in device memory.
   cusp::csr_matrix<int, real2, cusp::device_memory> A(Ndof, Ndof, NNZ);
   cusp::array1d<real2, cusp::device_memory> b(Ndof);
   cusp::array1d<real2, cusp::device_memory> x(Ndof);

   // Copy CSR row pointers to device memory
   thrust::copy(rowStarts, rowStarts + Ndof + 1, A.row_offsets.begin());

   // Copy CSR column indices to device memory
   thrust::copy(col, col +  NNZ, A.column_indices.begin());

   // Copy CSR values to device memory
   thrust::copy(val, val + NNZ, A.values.begin()); 

   // Copy right hand side vector to device memory
   thrust::copy(F, F + Ndof, b.begin());

   // Copy previous solution to device memory
   thrust::copy(u, u + Ndof, x.begin());
   
   // Set stopping criteria:
   //cusp::verbose_monitor<real2> monitor(b, solverIterMax, solverTol);
   cusp::default_monitor<real2> monitor(b, solverIterMax, solverTol);

   // Set preconditioner (identity)
   cusp::identity_operator<real2, cusp::device_memory> M(A.num_rows, A.num_rows);

   // Solve the linear system A * x = b with the Conjugate Gradient method
   // cusp::krylov::bicgstab(A, x, b, monitor, M);
   int restart = 50;
   // cout << "Iterative solution is started." << endl;
   cusp::krylov::gmres(A, x, b, restart, monitor);
   // cout << "Iterative solution is finished." << endl;

   // Copy x from device back to u on host 
   thrust::copy(x.begin(), x.end(), u);

   // ----------------------CONTROL------------------------
   // Print the solution to check
   // for(int i=0; i<=Ndof; i++) {
   //    cout << "u[" << i << "] = " << u[i] << endl;
   // }
   // ----------------------CONTROL------------------------

   //cout << "End of CUSPsolver() function." << endl;

}  // End of function CUSPSolver()


