#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cula_sparse.h>

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
void CULAsolver()
//-----------------------------------------------------------------------------
{
   culaStatus status;
   culaIterativeConfig config;
   culaIterativeResult result;

   culaCgOptions solverOptions;
   culaJacobiOptions jacobiOptions;
   culaBlockjacobiOptions blockJacobiOptions;
   culaIlu0Options ilu0Options;
   culaEmptyOptions emptyOptions;

   status = culaSparseInitialize();



   char buf[256];
   int exitVal = EXIT_FAILURE;

   if( !status ) { // success

   } else {

      culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
      printf("%s\n", buf);

      if( status == culaInsufficientComputeCapability )
         exitVal = EXIT_SUCCESS;

      if( status != culaDataError ) // For data errors, we can learn more by continuing
      {
         culaSparseShutdown();
         exit(exitVal);
      }
   }

   culaIterativeConfigInit(&config);
   config.tolerance = solverTol;
   config.maxIterations = solverIterMax;
   config.indexing = 0;

   culaCgOptionsInit(&solverOptions);

   culaEmptyOptionsInit(&emptyOptions);
   culaJacobiOptionsInit(&jacobiOptions);
   culaBlockjacobiOptionsInit(&blockJacobiOptions);
   culaIlu0OptionsInit(&ilu0Options);
   
   // Select a precondtioner and unccoment the corresponding line below. the 1st one corresponds to no preconditioner.
   status = culaDcsrCg(&config, &solverOptions, &emptyOptions, NN, NNZ, val, col, rowStarts, u, F, &result);
   //status = culaDcsrCgIlu0(&config, &solverOptions, &ilu0Options, NN, NNZ, val, col, rowStarts, u, F, &result);
   //status = culaDcsrCGJacobi(&config, &solverOptions, &jacobiOptions, NN, NNZ, val, col, rowStarts, u, F, &result);
   //status = culaDcsrCgBlockjacobi(&config, &solverOptions, &blockJacobiOptions, NN, NNZ, val, col, rowStarts, u, F, &result);

   solverIter = result.iterations;
   solverNorm = result.residual.relative;

   culaSparseShutdown();

   cout <<"number of iterations: " << solverIter << endl;

}  // End of function CULASolver()


