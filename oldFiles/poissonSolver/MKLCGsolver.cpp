#include <stdio.h>
#include <iostream>
#include "mkl_rci.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_service.h"

using namespace std;

#ifdef SINGLE
  typedef float real2;
#else
  typedef double real2;
#endif

extern int   *rowStarts, *col, NN, NNZ, solverIterMax, solverIter;
extern real2 *val, *u, *F;
extern double solverTol, solverNorm;




//-------------------------------------------------------------------------
void MKLCGsolver()
//-------------------------------------------------------------------------
{
   mkl_set_num_threads(32);

   MKL_INT n, rci_request, itercount, i;
   n = NN;
 
   MKL_INT ipar[128];
   double dpar[128], tmp[4*n];
   char tr = 'u';


   // MKL solvers need only the upper triangle of the symmetric matrices.
   // So let's find them.
   // This is using 1-based indexing.

   int   NNZupper, *rowStartsUpper, *colUpper;
   double *valUpper;

   NNZupper = NN + (NNZ-NN)/2;
   valUpper = new real2[NNZupper];
   colUpper = new int[NNZupper];
   rowStartsUpper = new int[NN+1];

   rowStartsUpper[0] = 1;

   int counter = 0;  // Counter for the nonzeros on the upper half

   for(int r=0; r<NN; r++) {  // Row loop
      for(int c=rowStarts[r]; c<rowStarts[r+1]; c++) {  // Nonzero column loop
         if(col[c]>=r) {  // These are on the upper half
            colUpper[counter] = col[c]+1;
            valUpper[counter] = val[c];
            counter++;
         }
      }
      rowStartsUpper[r+1] = counter+1;
   }



   dcg_init (&n, u, F, &rci_request, ipar, dpar, tmp);
   if (rci_request != 0) {
      printf("This example FAILED as the solver has returned the ERROR ");
      printf ("code %d", rci_request);
      MKL_FreeBuffers();
      return;
   }

   ipar[5] = solverIterMax;
   ipar[8] = 1;
   ipar[9] = 0;
   dpar[0] = solverTol;

   dcg_check (&n, u, F, &rci_request, ipar, dpar, tmp);
   if (rci_request != 0) {
      printf("This example FAILED as the solver has returned the ERROR ");
      printf ("code %d", rci_request);
      MKL_FreeBuffers();
      return;
   }

   rci:dcg (&n, u, F, &rci_request, ipar, dpar, tmp);
   if (rci_request == 0) {   // The solution is found with the required precision
      dcg_get (&n, u, F, &rci_request, ipar, dpar, tmp, &solverIter);
      cout << "number of iterations: " << solverIter << endl;
      MKL_FreeBuffers();
      return;
   } else if (rci_request == 1) { // Compute the vector A*tmp[0] and put the result in vector tmp[n]
      mkl_dcsrsymv (&tr, &n, valUpper, rowStartsUpper, colUpper, tmp, &tmp[n]);
      goto rci;
   } else {  // If rci_request=anything else, then dcg subroutine failed
      printf("This example FAILED as the solver has returned the ERROR ");
      printf ("code %d", rci_request);
      MKL_FreeBuffers();
      return;
   }

   delete valUpper;
   delete colUpper;
   delete rowStartsUpper;

}

