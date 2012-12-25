#include <iostream>

#include "mkl.h"
#include "mkl_pardiso.h"
#include "mkl_types.h"

using namespace std;

#ifdef SINGLE
  typedef float real2;
#else
  typedef double real2;
#endif

extern int *rowStartsSmall, *colSmall, NN, NNZ;
extern real2 *velVector, *val, *F;

//-----------------------------------------------------------------------------
void pardisoSolver_momentum()
//-----------------------------------------------------------------------------
{
   // Solve system of linear equations using the Pardiso solver available in
   // Intel's MKL library.

   MKL_INT mtype = 11;      // Real unsymmetric matrix
   MKL_INT nrhs = 1;        // Number of right hand sides
   void *pt[64];

   MKL_INT iparm[64];
   MKL_INT maxfct, mnum, phase, error, msglvl;

   MKL_INT i, j;
   double ddum;             // Double dummy 
   MKL_INT idum;            // Integer dummy

   for (i = 0; i < 64; i++) {
      iparm[i] = 0;
   }

   iparm[0] = 1;            // No solver default
   iparm[1] = 2;            // Fill-in reordering from METIS
   iparm[3] = 0;            // No iterative-direct algorithm
   iparm[4] = 0;            // No user fill-in reducing permutation
   iparm[5] = 0;            // Write solution into x
   iparm[7] = 2;            // Max numbers of iterative refinement steps
   iparm[9] = 13;           // Perturb the pivot elements with 1E-13
   iparm[10] = 1;           // Use nonsymmetric permutation and scaling MPS
   iparm[12] = 0;           // Maximum weighted matching algorithm is switched-on (default for non-symmetric)
   iparm[13] = 0;           // Output: Number of perturbed pivots
   iparm[17] = -1;          // Output: Number of nonzeros in the factor LU
   iparm[18] = -1;          // Output: Mflops for LU factorization
   iparm[19] = 0;           // Output: Numbers of CG Iterations
   iparm[34] = 1;

   maxfct = 1;              // Maximum number of numerical factorizations
   mnum = 1;                // Which factorization to use
   msglvl = 0;              // Do not print statistical information
   error = 0;               // Initialize error flag

   for (i = 0; i < 64; i++) {
      pt[i] = 0;
   }

   // Set phase to linear system solution
   phase = 13;

   // Set the number of cores to be used for parallel execution
   mkl_set_num_threads(16);

   PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
            &NN, val, rowStartsSmall, colSmall, &idum,
            &nrhs, iparm, &msglvl, F, velVector, &error);

   if (error != 0) {
      cout << "\nERROR during solution:" << error << "\n\n";
      exit (3);
   }

   //printf ("\nSolution of the system is: \n");
   //for (j = 0; j < n; j++) {
   //   cout << "x [" << j << "] = " << x[j] << endl;
   //}

   // Release internal memory
   phase = -1;
   PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
            &NN, &ddum, rowStartsSmall, colSmall, &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);

}  // End of function pardisoSolver()


