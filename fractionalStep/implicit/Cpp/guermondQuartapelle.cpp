
/*******************************************************************
                  3D Unsteady Navier-Stokes Solver
********************************************************************

          This code is a part of the CFD-with-CUDA project
               http://code.google.com/p/cfd-with-cuda

                          Dr. Cuneyt Sert
                Department of Mechanical Engineering
                  Middle East Technical University
                           Ankara, Turkey
                   http://www.metu.edu.tr/~csert


********************************************************************
                            FORMULATION
********************************************************************

  Fractional Step Method described in Blasco, Codina and Huerta's
  1998 paper is used. Hexahedral elements with 9-node bilinear
  pressure and 27-node biquadratic velocity approximation are
  supported.


********************************************************************
                       THIRD PARTY LIBRARIES
********************************************************************
  CPU version: CSparse library of Timothy Davis is used in the
               creation of the [Z] matrix (especially necessary for
               the sparse matrix-matrix multiplication).
  
               In step 2, Intel MKL's Conjugate Gradient solver is
               used. N_MKL_THREADS variable sets the number of
               parallel threads used by MKL.

  GPU version: It uses NVIDIA's CUDA Toolkit. CUSP library is used
               to calculate the [Z] matrix and Conjugate Gradient
               solution of step 2.


********************************************************************
                      PREPROCESSOR DIRECTIVES
********************************************************************

  WINDOWS:    Timing functions are different on Windows and Linux.
              Define WINDOWS to compile on a Windows machine.

  USECUDA:    NVIDIA's CUDA library is used to perform certain
              tasks on a graphics card (GPU).


********************************************************************
                       INPUT and OUTPUT FILES
********************************************************************

  INP:     Input file that provides the solution parameters, mesh
           and boundary conditions. Its location and name is given
           in the ProblemName.txt file. It includes only corner
           nodes of the elements, not mid-edge, mid-face or
           mid-element nodes.
       
  DAT:     Output file with velocity components and pressure to be
           visualized using the Tecplot software.
  

********************************************************************
                            SCREEN OUTPUT
********************************************************************

   During a run velocity and pressure values of a selected monitor
   point are written on the screen. Monitor point coordinates are
   read from the input file.
   
   Also time taken in each step of the fractional step solver is
   written on the screen. To stop this set PRINT_TIMES flag to
   zero.

********************************************************************/




#define _CRT_SECURE_NO_DEPRECATE    // This is necessary to avoid fopen() warning of MSVC.
int N_MKL_THREADS = 8;              // Number of Intel MKL threads
int N_OPENMP_THREADS = 8;           // Number of openMP threads
bool PRINT_TIMES = 1;               // Set to 1 to see the time taken by each step of the solver on the screen


#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <vector>
#include "mkl_types.h"
#include "mkl_rci.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_service.h"
#include <limits>
#include <omp.h>

//#include "mkl_pardiso.h"

#include <paralution.hpp>

using namespace paralution;

//extern "C" void mkl_freebuffers();

#ifdef USECUDA
   #include <cuda_runtime.h>
   #include "cublas_v2.h"
   #include "cusparse_v2.h"
#endif


#ifdef WINDOWS
   #include <time.h>
#else
   #include <sys/time.h>
#endif

using namespace std;

#ifdef SINGLE              // Many major parameters can automatically be defined as                                                                  TODO: Use "real" throughout the code.
  typedef float real;      // float by using -DSINGLE during compilcation. Default
#else                      // behavior is to use double precision.
  typedef double real;
#endif


//========================================================================
// Global Variables
//========================================================================

ifstream inpFile;            // Input file with INP extension.
ifstream problemNameFile;    // Input file name is read from this ProblemName.txt file.
ofstream datFile;            // Output file with DAT extension.
ofstream checkAFile; // Output file with DAT extension.

string whichProblem;

int eType;         // Element type. 1: 3D Hexahedron, 2: 3D Tetrahedron
int NE;            // Number of elements
int NN;            // Number of all nodes
int NCN;           // Number of corner nodes (no mid-edge, mid-face or mid-element nodes)
int NNp;           // Number of pressure nodes
int NENv;          // Number of velocity nodes on an element
int NENp;          // Number of pressure nodes on an element
int NGP;           // Number of Gauss Quadrature points

double dt;         // Time step
double t_ini;      // Initial time
double t_final;    // Final time
int timeN;         // Discrete time level
double timeT;      // Actual time

int    maxIter;    // Maximum iteration number performed inside each time step
double tolerance;  // Tolerance for the iterations performed in each time step
double convergenceCriteria;   // Convergence criteria for steady state problems

bool   isRestart;  // Switch that defines if solver continues from a previous
                   // solution or starts from the initial condition
double density;    // Density of the material
double viscosity;  // Viscosity of the material                                                                                                      TODO: Kinematic or dynamic?
double fx, fy, fz; // Body force components

double alpha;      //                                                                                                                                TODO: Read from the input file, but not used

int    NEC;        // Number of element corners
int    NEF;        // Number of element faces
int    NEE;        // Number of element edges

double **coord;    // Coordinates (x, y, z) of mesh nodes. Initial size is [NE*NENv][3]. Later reduces to [NN][3]

int **LtoGnode;    // Local to global node mapping of velocity nodes (size:NExNENv)
int **LtoGvel;     // Local to global mapping of velocity unknowns (size:NEx3*NENv)
int **LtoGpres;    // Local to global mapping of pressure unknowns (size:NExNENp)

int **elemNeighbors;    // Neighbors of each element
int *NelemNeighbors;    // Number of neighbors of each element

int *meshColors;        // Colors of each mesh 
int *NmeshColors;       // Number of elements at each color
int *elementsOfColor;   // Elements at each color in a sorted way
int nActiveColors;      // Number of different colors in mesh 

int nBC;                // Number of different boundary conditions.
double *BCtype;         // Type of each BC. 1: Specified velocity
double **BCstr;         // Specified velocity values for each BC.                                                                                         TODO: These should actually be strings.
int BCnVelFaces;        // Number of element faces where velocity BC is specified.
int BCnVelNodes;        // Number of nodes where velocity BC is specified.
int BCnOutFaces;        // Number of element faces where outflow BC is specified.
int **BCvelFaces;       // Stores element number, face number and BC number for each face where velocity BC is specified.
int **BCvelNodes;       // Stores node number and BC number for each node where velocity BC is specified.
int **BCoutFaces;       // Stores element number, face number and BC number for each face where outflow BC is specified.
int zeroPressureNode;   // Node where pressure is set to zero. A negative value is ignored.

double monPointCoord[3];  // Coordinates of the monitor point.
int monPoint;             // Node that is being monitored.

int **elemsOfVelNodes;    // List of elements that are connected to velocity nodes
int **elemsOfPresNodes;   // List of elements that are connected to pressure nodes
int *NelemOfVelNodes;     // Number of elements connnected to velocity nodes
int *NelemOfPresNodes;    // Number of elements connnected to pressure nodes


int sparseM_NNZ;          // Counts nonzero entries in i) a single sub-mass matrix and ii) full Mass matrix.
double *sparseMvalue;     // Nonzero values of the global mass matrix. Actually the values of only the upper-left sub mass matrix are stored.
int *sparseMcol;          // Nonzero columns of M, K and A matrices. Info is kept only for the upper-left sub mass matrix.
int *sparseMrow;          // Nonzero rows of M, K and A matrices. Info is kept only for the upper-left sub mass matrix.
int *sparseMrowStarts;    // Row start indices of M, K and A matrices (for CSR storage). Info is kept only for the upper-left sub mass matrix.
int *sparseMrowStartsMod; // A modified version of the above array, used by MKL

double *sparseKvalue;     // Nonzero values of the global K matrix. Actually the values of only the upper-left sub stiffness matrix are stored. [K] has the same sparsity structure as [M].
double *sparseAvalue;     // Nonzero values of the system matrix arises from momentum equation. 

int sparseG_NNZ;          // Counts nonzero entries in i) sub-G matrix and ii) full G matrix.
double *sparseG1value;    // Nonzero values of the 1st part of the global G matrix.
double *sparseG2value;    // Nonzero values of the 2nd part of the global G matrix.
double *sparseG3value;    // Nonzero values of the 3rd part of the global G matrix.
int *sparseGcol;          // Nonzero columns of only one sub G matrix.
int *sparseGrow;          // Nonzero rows of only one sub G matrix.
int *sparseGrowStarts;    // Row start indices of G matrix (for CSR storage).
int *sparseGrowStartsMod; // A modified version of the above array, used by MKL

int sparseZ_NNZ;          // Counts nonzero entries in i) a single sub-mass matrix and ii) full Mass matrix.
double *sparseZvalue;     // Nonzero values of the global mass matrix. Actually the values of only the upper-left sub mass matrix are stored.
int *sparseZcol;          // Nonzero columns of Z matrix.
int *sparseZrow;          // Nonzero rows of Z matrix.
int *sparseZrowStarts;    // Row start indices of Z matrix (for CSR storage).
int *sparseZrowStartsMod; // A modified version of the above array, used by MKL

int ***sparseMapM;        // Maps each element's local M, K, A entries to the global ones that are stored in sparse format.
int ***sparseMapG;        // Maps each element's local G entries to the global ones that are stored in sparse format.
int ***sparseMapZGPU;     // Maps each element's local Z entries to the global ones that are stored in sparse format.
int **sparseMapZ;         // Maps each element's local Z entries to the global ones that are stored in sparse format.
int **sparseMapZupper;    // Determines which elemental values will be added to sparse Z (upper triangular).
int *sparseMapZupperCount;// Number of values of elemental stiffness matrice contributions to global stiffness matrice Z.


double **GQpoint, *GQweight; // GQ points and weights.

double **Sp;              // Shape functions for pressure evaluated at GQ points. (size:NENpxNGP)
double ***dSp;            // Derivatives of shape functions for pressure wrt to ksi, eta & zeta evaluated at GQ points. (size:NENpxNGP)
double **Sv;              // Shape functions for velocity evaluated at GQ points. (size:NENvxNGP) 
double ***dSv;            // Derivatives of shape functions for velocity wrt to ksi, eta & zeta evaluated at GQ points. (size:NENvxNGP)

double **detJacob;        // Determinant of the Jacobian matrix evaluated at a certain (ksi, eta, zeta)
double ****gDSp;          // Derivatives of shape functions for pressure wrt x, y & z at GQ points. (size:3xNENvxNGP)
double ****gDSv;          // Derivatives of shape functions for velocity wrt x, y & z at GQ points. (size:3xNENvxNGP)

double *Uk;               // x, y and z velocity components of time step k+1. (U^k+1 of the reference paper).
double *Uk_prev;          // x, y and z velocity components of time step k. (U^k of the reference paper).
double *Uk_dir;           // Keeps each component of velocity.

double *Pk;               // Pressure of time step k+1 (P^k+1 of the reference paper).
double *Pk_prev;          // P^k of the reference paper.
double *Pk_prevprev;      // P^k-1 of the reference paper.
double *Pdiff;            // P^k+1-P^k

double *R1;               // RHS vector of intermediate velocity calculation.
double *R2;               // RHS vector of pressure calculation.

double *gDSv_1d, *GQfactor_1d, *Sv_1d;
int    *sparseMapM_1d;
int    *LtoGvel_1d;

int direction;   // Determines the direction worked on
double StartCurrentTimeStep, wallClockTimeCurrentTimeStep;
bool checkAccConvergence;
double maxAcc;
char dummyUserInput;      // Used for debugging  


// CUDA DEVICE variables. Their names end with "_d".
#ifdef USECUDA
   double *M_d, *K_d, *G1_d, *G2_d, *G3_d, *A_d, *Z_d;
   int    *Mcol_d, *MrowStarts_d, *Gcol_d, *GrowStarts_d;
   int    *Zcol_d, *ZrowStarts_d;   
   double *Uk_d, *Uk_dir_d, *Uk_prev_d;
   double *Pk_d, *Pk_prev_d, *Pk_prevprev_d, *Pdiff_d; 
   double *R1_d, *R2_d;
   int    *BCvelNodes_d, *BCvelNodesWhichBC_d;
   double *BCstr_1d_d;   
   
   int *NmeshColors_d, *meshColors_d, *elementsOfColor_d;
   int *LtoGvel_1d_d;
   int *sparseMapM_1d_d;
   double *Sv_1d_d;
   double *gDSv_1d_d, *GQfactor_1d_d;

   cusparseHandle_t            handle;
   cusparseMatDescr_t          descr;
   cublasHandle_t              handleCUBLAS;
   cudaError_t                 cudaStatus;
   cusparseSolveAnalysisInfo_t analysisInfo1, analysisInfo2;
   
   size_t freeGPUmemory, totalGPUmemory;   // To measure total and free GPU memory
#endif





//========================================================================
// Functions
//========================================================================
void readInputFile();
double getHighResolutionTime(int, double);
void findElemNeighbors();
void setupMeshColoring();
void setupNonCornerNodes();
void setupLtoGdof();
void determineVelBCnodes();
void findElemsOfPresNodes();
void findElemsOfVelNodes();
void findMonitorPoint();
void setupSparseM();
void setupSparseG();
void setupSparseZ();
void calculateZ();
void setupGQ();
void calcShape();
void calcJacob();
void initializeAndAllocate();
void readRestartFile();
void createTecplot();
void timeLoop();
void step0();
void calculateMatrixA();
void step1();
void step2();
void paralution_BiCGStab_solver();
//void MKL_pardisoSolver();
void MKL_GMRES_solver();
void MKL_CG_solver();
void applyBC_initial();
void applyBC_Step1(int);
void applyBC_Step2(int);
void waitForUser(string);

// Functions that are used when USECUDA option is defined.
#ifdef USECUDA
   void setupSparseZGPU();
   void calculateZGPU();
   void selectCUDAdevice();
   void initializeAndAllocateGPU();
   void calculateMatrixAGPU();   
   void step1GPU();
   void step2GPU();
   bool checkConvergenceGPU();
   bool checkConvergenceInTimeGPU();   
   void printMonitorDataGPU();
 #endif




//========================================================================
int main()

{
   cout << "\n\n*********************************************************";
   cout <<   "\n*    3D Unsteady Incompressible Navier-Stokes Solver    *";
   cout <<   "\n*        Formulation of Blasco, Codina & Huerta         *";
   cout <<   "\n*             Part of CFD-with-CUDA project             *";
   cout <<   "\n*                    Dr. Cuneyt Sert                    *";
   cout <<   "\n*        http://code.google.com/p/cfd-with-cuda         *";
   cout <<   "\n*********************************************************\n\n";

   waitForUser("Just started. Enter a character... ");


   #ifdef USECUDA
      selectCUDAdevice();
   #endif
   

   // Set the thread number for MKL parallelization
   mkl_set_dynamic(0);
   mkl_set_num_threads(N_MKL_THREADS);
   
   // Set the thread number for openMP parallelization   
   omp_set_num_threads(N_OPENMP_THREADS);   
   
   init_paralution();
   
   set_omp_threads_paralution(N_OPENMP_THREADS);

   info_paralution();


   
   double Start, Start1, wallClockTime;   // Used for run time measurement.

   Start1 = getHighResolutionTime(1, 1.0);

   Start = getHighResolutionTime(1, 1.0);
   readInputFile();                       // Read the input file.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("readInputFile()        took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   Start = getHighResolutionTime(1, 1.0);
   findElemsOfPresNodes();                // Finds elements that are connected to each pressure node.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("findElemsOfPresNodes() took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   Start = getHighResolutionTime(1, 1.0);
   findElemNeighbors();                   // Finds neighbors of all elements.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("findElemNeighbors()    took  %8.3f seconds.\n", wallClockTime);
   
   Start = getHighResolutionTime(1, 1.0);
   setupMeshColoring();
   wallClockTime = getHighResolutionTime(2, Start);
   printf("setupMeshColoring()    took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   Start = getHighResolutionTime(1, 1.0);
   setupNonCornerNodes();                 // Find non-corner nodes, add them to LtoGnode and calculate their coordinates.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("setupNonCornerNodes()  took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   Start = getHighResolutionTime(1, 1.0);
   setupLtoGdof();                        // Creates LtoGvel and LtoGpres using LtoGnode.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("setupLtoGdof()         took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   Start = getHighResolutionTime(1, 1.0);
   determineVelBCnodes();                 // Converts face-based velocity BC data into a node-based format.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("determineVelBCnodes()  took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   Start = getHighResolutionTime(1, 1.0);
   findElemsOfVelNodes();                 // Finds elements that are connected to each velocity node.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("findElemsOfVelNodes()  took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   Start = getHighResolutionTime(1, 1.0);
   findMonitorPoint();                    // Finds the node that is closest to the monitor point coordinates.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("findMonitorPoint()     took  %8.3f seconds.\n", wallClockTime);

   Start = getHighResolutionTime(1, 1.0);
   setupSparseM();                        // Finds the sparsity pattern of the Mass matrix.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("setupSparseM()         took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   Start = getHighResolutionTime(1, 1.0);
   setupSparseG();                        // Finds the sparsity pattern of the G matrix.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("setupSparseG()         took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   #ifdef USECUDA
      Start = getHighResolutionTime(1, 1.0);
      setupSparseZGPU();                  // Finds the sparsity pattern of the Z matrix.
      wallClockTime = getHighResolutionTime(2, Start);
      printf("setupSparseZ()         took  %8.3f seconds.\n", wallClockTime);

      waitForUser("Enter a character... ");
   #else
      Start = getHighResolutionTime(1, 1.0);
      setupSparseZ();                     // Finds the sparsity pattern of the Z matrix.
      wallClockTime = getHighResolutionTime(2, Start);
      printf("setupSparseZ()         took  %8.3f seconds.\n", wallClockTime);

      waitForUser("Enter a character... ");
   #endif   
   
   Start = getHighResolutionTime(1, 1.0);
   setupGQ();                             // Sets up GQ points and weights.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("setupGQ()              took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   Start = getHighResolutionTime(1, 1.0);
   calcShape();                           // Calculates shape functions and their derivatives at GQ points.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("calcShape()            took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   Start = getHighResolutionTime(1, 1.0);
   calcJacob();                           // Calculates the determinant of the Jacobian and global shape function derivatives at each GQ point.
   wallClockTime = getHighResolutionTime(2, Start);
   printf("calcJacob()            took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   timeLoop();                            // Main solution loop.
   
   stop_paralution();
   
   wallClockTime = getHighResolutionTime(2, Start1);
   printf("\nTotal run took %10.3f seconds.\n", wallClockTime);
   
   cout << endl << "The program is terminated successfully.\n\n\n";

   waitForUser("Enter a character... ");

   createTecplot();

   return 0;

} // End of function main()





//========================================================================
void readInputFile()
//========================================================================
{
   // Read the input file with INP extension.
   
   string dummy, dummy2, dummy4, dummy5;
   int intDummy;

   problemNameFile.open(string("ProblemName.txt").c_str(), ios::in);   // This ProblemName.txt file includes the name of the input file.
   problemNameFile >> whichProblem;   // This is used to construct input file's name.
   problemNameFile.close();
   
   inpFile.open((whichProblem + ".inp").c_str(), ios::in);
     
   inpFile.ignore(256, '\n');   // Read and ignore the line
   inpFile.ignore(256, '\n');   // Read and ignore the line

   inpFile.ignore(256, ':');    inpFile >> eType;       inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> NE;          inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> NCN;         inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> NENv;        inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> NENp;        inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> NGP;         inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> alpha;       inpFile.ignore(256, '\n');                                                                   // TODO: alpha is not used
   inpFile.ignore(256, ':');    inpFile >> dt;          inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> t_ini;       inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> t_final;     inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> maxIter;     inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> tolerance;   inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> convergenceCriteria;   inpFile.ignore(256, '\n');   
   inpFile.ignore(256, ':');    inpFile >> isRestart;   inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> density;     inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> viscosity;   inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> fx;          inpFile.ignore(256, '\n');
   inpFile.ignore(256, ':');    inpFile >> fy;          inpFile.ignore(256, '\n');                                                                   // TODO: Also read fz



   // Read corner node coordinates
   coord = new double*[NE*NENv];   // At this point we'll read the coordinates of only NCN corner nodes.
                                   // Later we'll add non-corner nodes to it. At this point we do NOT
                                   // know the total number of nodes. Therefore we use a large enough
                                   // number of NE*NENv. Later the size will be reduced to NN.
   for (int i=0; i<NE*NENv; i++) {
      coord[i] = new double[3];
   }

   inpFile.ignore(256, '\n');   // Read and ignore the line
   inpFile.ignore(256, '\n');   // Read and ignore the line

   for (int i=0; i<NCN; i++){
      inpFile >> intDummy >> coord[i][0] >> coord[i][1] >> coord[i][2];
      inpFile.ignore(256, '\n');
   }
   

   if (eType == 1) {  // Hexahedral element
     NEC = 8;             // Number of element corners
     NEF = 6;             // Number of element faces
     NEE = 12;            // Number of element edges
   } else {           // Tetrahedral element
     NEC = 4;
     NEF = 4;
     NEE = 6;
   }




   // Read corner nodes of each element, i.e. LtoGnode
   LtoGnode = new int*[NE];
   for (int i=0; i<NE; i++) {
      LtoGnode[i] = new int[NENv];
   }


   for (int i=0; i<NE; i++) {
      for (int j=0; j<NENv; j++) {
         LtoGnode[i][j] = -1;     //Initialize to -1
      }
   }


   inpFile.ignore(256, '\n'); // Read and ignore the line
   inpFile.ignore(256, '\n'); // Read and ignore the line 

   for (int e = 0; e < NE; e++){
      inpFile >> intDummy;
      for (int i = 0; i < NEC; i++){
         inpFile >> LtoGnode[e][i];
         LtoGnode[e][i] = LtoGnode[e][i] - 1;                              // MATLAB -> C++ index switch 
      }
      inpFile.ignore(256, '\n'); // Read and ignore the line 
   }
 

   // Read number of different BC types and details of each BC
   inpFile.ignore(256, '\n'); // Read and ignore the line
   inpFile.ignore(256, '\n'); // Read and ignore the line
   
   inpFile.ignore(256, ':');    inpFile >> nBC;       inpFile.ignore(256, '\n');
   
   // Allocate BCtype and BCstr
   BCtype = new double[nBC];

   BCstr = new double*[nBC];
   for (int i=0; i<nBC; i++) {
      BCstr[i] = new double[3];
   }
   
   for (int i = 0; i<nBC; i++){
      inpFile.ignore(256, ':');
      inpFile >> BCtype[i];

      inpFile >> BCstr[i][0];
      inpFile >> dummy;

      inpFile >> BCstr[i][1];
      inpFile >> dummy;

      inpFile >> BCstr[i][2];
      inpFile.ignore(256, '\n'); // Ignore the rest of the line
   }
   
   inpFile.ignore(256, '\n'); // Read and ignore the line 
   inpFile.ignore(256, ':');     inpFile >> BCnVelFaces;
   inpFile.ignore(256, '\n'); // Ignore the rest of the line
   
   inpFile.ignore(256, ':');     inpFile >> BCnOutFaces;
   inpFile.ignore(256, '\n'); // Ignore the rest of the line



   // Read velocity BCs
   inpFile.ignore(256, '\n'); // Read and ignore the line
   inpFile.ignore(256, '\n'); // Read and ignore the line
      
   if (BCnVelFaces != 0){
      BCvelFaces = new int*[BCnVelFaces];
      for (int i = 0; i < BCnVelFaces; i++){
         BCvelFaces[i] = new int[3];
      }
      
      for (int i = 0; i < BCnVelFaces; i++){
         inpFile >> BCvelFaces[i][0] >> BCvelFaces[i][1] >> BCvelFaces[i][2];
         BCvelFaces[i][0] = BCvelFaces[i][0] - 1;                              // MATLAB -> C++ index switch
         BCvelFaces[i][1] = BCvelFaces[i][1] - 1;                              // MATLAB -> C++ index switch
         BCvelFaces[i][2] = BCvelFaces[i][2] - 1;                              // MATLAB -> C++ index switch
         inpFile.ignore(256, '\n'); // Ignore the rest of the line
      }
   }

  // Read outflow BCs
   inpFile.ignore(256, '\n'); // Read and ignore the line  
   inpFile.ignore(256, '\n'); // Read and ignore the line
   
   if (BCnOutFaces != 0){
      BCoutFaces = new int*[BCnOutFaces];
      for (int i = 0; i < BCnOutFaces; i++){
         BCoutFaces[i] = new int[3];
      }
      for (int i = 0; i < BCnOutFaces; i++){
         inpFile >> BCoutFaces[i][0] >> BCoutFaces[i][1] >> BCoutFaces[i][2];
         BCoutFaces[i][0] = BCoutFaces[i][0] - 1;                              // MATLAB -> C++ index switch
         BCoutFaces[i][1] = BCoutFaces[i][1] - 1;                              // MATLAB -> C++ index switch
         BCoutFaces[i][2] = BCoutFaces[i][2] - 1;                              // MATLAB -> C++ index switch
         inpFile.ignore(256, '\n'); // Ignore the rest of the line
      }
   }
   
   
   
   // Read the node where pressure is taken to be zero
   inpFile.ignore(256, '\n'); // Read and ignore the line
   inpFile.ignore(256, '\n'); // Read and ignore the line
   inpFile >> zeroPressureNode;
   zeroPressureNode = zeroPressureNode - 1;                                // MATLAB -> C++ index switch
   inpFile.ignore(256, '\n'); // Ignore the rest of the line
   
   
   
   // Read monitor point coordinates
   inpFile.ignore(256, '\n'); // Read and ignore the line
   inpFile.ignore(256, '\n'); // Read and ignore the line
   inpFile >> monPointCoord[0] >> monPointCoord[1] >> monPointCoord[2];
   
   
   inpFile.close();
   
   
   // Determine NNp, number of pressure nodes
   if (NENp == 1) {   // Only 1 pressure node at the element center. Not tested at all.                                                              TODO: Either test this element and fully support it or remove details about it.
      NNp = NE;
   } else {
      NNp = NCN;      // Pressure are stored at element corners.
   }

} // End of function readInputFile()





//========================================================================
void findElemsOfPresNodes()
//========================================================================
{
   // Determines elements connected to pressure nodes. It is stored in a
   // matrix of size NNpx10, where 10 is a number, estimated to be larger
   // than the maximum number of elements connected to a pressure node.

   // Also an array (NelemOfPresNodes) store the actual number of elements
   // connected to each pressure node.

   // It is assumed that pressure nodes are at element corners.

   int LARGE = 10;  // It is assumed that not more than 10 elements are
                    // connected to a pressure node.
                                                                                                                                                     // TODO: Define this somewhere else, which will be easy to notice.
                                                                                                                                                     // TODO: Make sure that this is not violated.
   int node;

   elemsOfPresNodes = new int*[NNp];
   for (int i=0; i<NNp; i++) {
      elemsOfPresNodes[i] = new int[LARGE];
   }

   NelemOfPresNodes = new int[NNp];

   for (int i = 0; i < NNp; i++) {
      NelemOfPresNodes[i] = 0;      // Initialize to zero
   }

   // Form elemsOfPresNodes using LtoGnode of each element
   for (int e = 0; e < NE; e++) {
      for (int i = 0; i < NENp; i++) {
         node = LtoGnode[e][i];   // It is assumed that pressure nodes are at element corners.
         elemsOfPresNodes[node][NelemOfPresNodes[node]] = e;
         NelemOfPresNodes[node] = NelemOfPresNodes[node] + 1;
      }
   }

   //  CONTROL
   //for (int i=0; i<NNp; i++) {
   //   cout << i << ":  " << NelemOfPresNodes[i] << endl;
   //}
   
   //for (int i=0; i<NNp; i++) {
   //   cout << i << ":  " ;
   //   for (int j=0; j<NelemOfPresNodes[i]; j++) {
   //      cout << elemsOfPresNodes[i][j] << "  ";
   //   }
   //   cout << endl;
   //}

}  // End of function findElemsOfPresNodes()





//========================================================================
void findElemNeighbors()
//========================================================================
{
   // Determines neighboring element/face for each face of each element.

   int node, elem;
   int LARGE = 26;                                                                                                                                   // TODO: 26 is the maximum number of elements for hexahedral elements. For tetrahedra's it'll be different
   bool inList;
   
   NelemNeighbors = new int[NE];
   elemNeighbors = new int*[NE];
   for (int i = 0; i < NE; i++) {
      elemNeighbors[i] = new int[LARGE];
   }

   for (int e = 0; e < NE; e++) {
      for (int i = 0; i < LARGE; i++) {
         elemNeighbors[e][i] = -1;   // Initialize
      }
      NelemNeighbors[e] = 0;         // Initialize

      // Determine all elements around this element from elemsOfPresNodes
      for (int i = 0; i < NEC; i++) {
         node = LtoGnode[e][i];
         for (int j = 0; j < NelemOfPresNodes[node]; j++) {
            elem = elemsOfPresNodes[node][j];
            if (elem == e) {
               continue;
            }
            // Check if elem is already in allNeighbors list or not
            inList = 0;
            for (int k = 0; k < NelemNeighbors[e]; k++) {
               if (elem == elemNeighbors[e][k]) {
                  inList = 1;
                  break;
               }
            }
            if (inList == 0) {
               elemNeighbors[e][NelemNeighbors[e]] = elem;
               NelemNeighbors[e] = NelemNeighbors[e] + 1;
            }
         }
      }
   }  // End of element loop


   // CONTROL
   //for (int e = 0; e < NE; e++) {
   //   cout << e << ": " << NelemNeighbors[e] << ": ";
   //   for (int i = 0; i < NelemNeighbors[e]; i++) {
   //      cout << elemNeighbors[e][i] << ", ";
   //   }
   //   cout << endl;
   //}
   

}  // End of function findElemNeighbors()





//========================================================================
void setupMeshColoring()

{
   // Determines mesh coloring in order to prevent race condition at 
   // the assembly of [A] at GPU.

   int node, elem;
   int LARGE = 30;   // Number of colors (It should be 8 for structured hexahedral meshes)                                                                                                                                  // TODO: 26 is the maximum number of elements for hexahedral elements. For tetrahedra's it'll be different
   bool check;
   int candidateColor; 
   
   meshColors      = new int[NE];
   NmeshColors     = new int[LARGE];
   elementsOfColor = new int[NE];

   for (int e = 0; e < NE; e++) {
      meshColors[e] = -1;
   }
   
   for (int i = 0; i < LARGE; i++) {
      NmeshColors[i] = 0;
   }
   
   for (int e = 0; e < NE; e++) {
      
      candidateColor = 0;
      
      for (int i = 0; i < NelemNeighbors[e]; i++) {
         elem = elemNeighbors[e][i];
         if (candidateColor == meshColors[elem]) {
            candidateColor = candidateColor + 1;
            i = 0;
         }
      }
      
      meshColors[e] = candidateColor;
      NmeshColors[candidateColor] = NmeshColors[candidateColor] + 1;
      
   }  // End of element loop

   int count = 0;
   nActiveColors = 0;
   for (int i = 0; i < LARGE; i++) {
      if (NmeshColors[i] > 0) {
         cout << "color" << i << " has" << NmeshColors[i] << endl; 
         nActiveColors = nActiveColors + 1;
         for (int e = 0; e < NE; e++) {
            if (meshColors[e] == i) {
               elementsOfColor[count] = e;
               count = count + 1;
            }
         }
      }
   }
   

   cout << "Number of active colors = " << nActiveColors << endl;

   // CONTROL
   //for (int e = 0; e < NE; e++) {
      //cout << e << ": " << meshColors[e] << endl;
   //}
   //cout << endl;
   
   //for (int e = 0; e < NE; e++) {
      //cout << e << ": " << elementsOfColor[e] << endl;
   //}
   //cout << endl;
      
   //for (int i = 0; i < LARGE; i++) {
      //cout << i << ": " << NmeshColors[i] << endl;
   //}
   //cout << endl;

//int imin = std::numeric_limits<int>::min(); // minimum value
//int imax = std::numeric_limits<int>::max();   

//long lmin = std::numeric_limits<long>::min(); // minimum value
//long lmax = std::numeric_limits<long>::max();  

//long long llmin = std::numeric_limits<long long>::min(); // minimum value
//long long llmax = std::numeric_limits<long long>::max();
//cout << endl;
//cout << "imin = " << imin << endl;
//cout << "imax = " << imax << endl;
//cout << "lmin = " << lmin << endl;
//cout << "lmax = " << lmax << endl;
//cout << "llmin = " << llmin << endl;
//cout << "llmax = " << llmax << endl;
//cout << sizeof(int) << endl;
//cout << sizeof(short) << endl;
//cout << sizeof(long) << endl;      
   

}  // End of function setupMeshColoring()





//========================================================================
void setupNonCornerNodes()
//========================================================================
{
   // Calculates coordinates of non corner nodes and adds them to LtoGnode.

   if (NENv == NENp) {   // Don't do anything if NENv == NENp
      return;
   }

   double SMALL = 1e-10;   // Value used for coordinate equality check
   int nodeCount = NCN;    // This will be incremented as new mid-edge and mid-face nodes are added.
   int matchFound, ne, n1, n2, n3, n4, n5, n6, n7, n8;
   
   double midEdgeCoord[3];
   double midFaceCoord[3];
   double midElemCoord[3];


   int NneighborMEN;  // Number of mid-edge nodes of neighbors of an element
   int *neighborMEN;  // List of mid-edge nodes of neighbors of an element
   neighborMEN = new int [26*NEE];   // Maximum number of mid-edge nodes of neighbors of an element is NEF*NEE                                       // TODO: 26 is valid for a hexahedral element

   //Go through all edges of the element and check whether there are any new mid-edge nodes or not.
   for (int e = 0; e < NE; e++) {
   
      // Determine all mid-edge nodes of neighboring elements
      NneighborMEN = 0;           // Initialize
      for (int i = 0; i < 26*NEE; i++) {
         neighborMEN[i] = -1;     // Initialize
      }
      for (int e2 = 0; e2 < NelemNeighbors[e]; e2++) {  // Loop over element e's neighbors
         ne = elemNeighbors[e][e2];
         for (int i = NEC; i < NEC + NEE; i++) {
            if (LtoGnode[ne][i] != -1) {   // Check whether there is a previously found neighboring mid-edge node
               neighborMEN[NneighborMEN] = LtoGnode[ne][i];
               NneighborMEN++;
            }
         }
      }

      // CONTROL
      //cout << "NneighborMEN = " << NneighborMEN << endl;
      //cout << "e = " << e << ":  ";
      //for (int i = 0; i < NneighborMEN; i++) {
      //   cout << neighborMEN[i] << ",  ";
      //}
      //cout << "\n\n\n";
      
      for (int ed = 0; ed < NEE; ed++) {
         // Determine corner nodes of edge ed
         if (eType == 1) {   // Hexahedral element
            switch (ed) {
            case 0:
              n1 = LtoGnode[e][0];
              n2 = LtoGnode[e][1];
              break;
            case 1:
              n1 = LtoGnode[e][1];
              n2 = LtoGnode[e][2];
              break;
            case 2:
              n1 = LtoGnode[e][2];
              n2 = LtoGnode[e][3];
              break;
            case 3:
              n1 = LtoGnode[e][3];
              n2 = LtoGnode[e][0];
              break;
            case 4:
              n1 = LtoGnode[e][0];
              n2 = LtoGnode[e][4];
              break;
            case 5:
              n1 = LtoGnode[e][1];
              n2 = LtoGnode[e][5];
              break;
            case 6:
              n1 = LtoGnode[e][2];
              n2 = LtoGnode[e][6];
              break;
            case 7:
              n1 = LtoGnode[e][3];
              n2 = LtoGnode[e][7];
              break;
            case 8:
              n1 = LtoGnode[e][4];
              n2 = LtoGnode[e][5];
              break;
            case 9:
              n1 = LtoGnode[e][5];
              n2 = LtoGnode[e][6];
              break;
            case 10:
              n1 = LtoGnode[e][6];
              n2 = LtoGnode[e][7];
              break;
            case 11:
              n1 = LtoGnode[e][7];
              n2 = LtoGnode[e][4];
              break;
            }
         } else if (eType == 2) {   // Tetrahedral element
           printf("\n\n\nERROR: Tetrahedral elements are not implemented in function setupMidFaceNodes() yet!!!\n\n\n");
         }

         midEdgeCoord[0] = 0.5 * (coord[n1][0] + coord[n2][0]);
         midEdgeCoord[1] = 0.5 * (coord[n1][1] + coord[n2][1]);
         midEdgeCoord[2] = 0.5 * (coord[n1][2] + coord[n2][2]);

         matchFound = 0;
         
         // Search if this new mid-edge node coordinate was already found previously.
         int midEdgeNode;
         for (int i = 0; i < NneighborMEN; i++) {
            midEdgeNode = neighborMEN[i];   // Neighboring mid-edge node
            if (abs(midEdgeCoord[0] - coord[midEdgeNode][0]) < SMALL) {
               if (abs(midEdgeCoord[1] - coord[midEdgeNode][1]) < SMALL) {
                  if (abs(midEdgeCoord[2] - coord[midEdgeNode][2]) < SMALL) {   // Match found, this is not a new node.
                     LtoGnode[e][ed+NEC] = midEdgeNode;
                     matchFound = 1;
                     break;
                  }
               }
            }
         }
         
         /*
         // Search if this new coordinate was already found previously.
         for (int i = NCN; i<nodeCount; i++) {
            if (abs(midEdgeCoord[0] - coord[i][0]) < SMALL) {
               if (abs(midEdgeCoord[1] - coord[i][1]) < SMALL) {
                  if (abs(midEdgeCoord[2] - coord[i][2]) < SMALL) {   // Match found, this is not a new node.
                     LtoGnode[e][ed+NEC] = i;
                     matchFound = 1;
                     break;
                  }
               }
            }
         }
         */

         if (matchFound == 0) {   // No match found, this is a new node.
            LtoGnode[e][ed+NEC] = nodeCount;
            coord[nodeCount][0] = midEdgeCoord[0];
            coord[nodeCount][1] = midEdgeCoord[1];
            coord[nodeCount][2] = midEdgeCoord[2];
            nodeCount = nodeCount + 1;
         }
      }  // End of ed (edge) loop
   }  // End of e (element) loop

   delete[] neighborMEN;

   int lastNode = nodeCount;

   int NneighborMFN;  // Number of mid-face nodes of neighbors of an element
   int *neighborMFN;  // List of mid-face nodes of neighbors of an element
   neighborMFN = new int [26*NEF];   // Maximum number of mid-face nodes of neighbors of an element is 26*NEF                                        // TODO: 26 is valid for a hexahedral element

   // Go through all faces of the element and check whether there are any new mid-face nodes or not.
   for (int e = 0; e < NE; e++) {

      // Determine all mid-face nodes of neighboring elements
      NneighborMFN = 0;           // Initialize
      for (int i = 0; i < NEF*NEF; i++) {
         neighborMFN[i] = -1;     // Initialize
      }
      for (int e2 = 0; e2 < NelemNeighbors[e]; e2++) {  // Loop over element e's neighbors
         ne = elemNeighbors[e][e2];
         for (int i = NEC+NEE; i < NENv; i++) {
            if (LtoGnode[ne][i] != -1) {   // Check whether there is a previously found neighboring mid-face node
               neighborMFN[NneighborMFN] = LtoGnode[ne][i];
               NneighborMFN++;
            }
         }
      }

      /* CONTROL
      cout << "NneighborMFN = " << NneighborMFN << endl;
      cout << "e = " << e << ":  ";
      for (int i = 0; i < NneighborMFN; i++) {
         cout << neighborMFN[i] << ",  ";
      }
      cout << "\n\n\n";
      */

      for (int f = 0; f < NEF; f++) {
         // Determine corner nodes of face f
         if (eType == 1) {   // Hexahedral element
            switch (f) {
            case 0:
               n1 = LtoGnode[e][0];
               n2 = LtoGnode[e][1];
               n3 = LtoGnode[e][2];
               n4 = LtoGnode[e][3];
               break;
            case 1:
               n1 = LtoGnode[e][0];
               n2 = LtoGnode[e][1];
               n3 = LtoGnode[e][4];
               n4 = LtoGnode[e][5];
               break;
            case 2:
               n1 = LtoGnode[e][1];
               n2 = LtoGnode[e][2];
               n3 = LtoGnode[e][5];
               n4 = LtoGnode[e][6];
               break;
            case 3:
               n1 = LtoGnode[e][2];
               n2 = LtoGnode[e][3];
               n3 = LtoGnode[e][6];
               n4 = LtoGnode[e][7];
               break;
            case 4:
               n1 = LtoGnode[e][0];
               n2 = LtoGnode[e][3];
               n3 = LtoGnode[e][4];
               n4 = LtoGnode[e][7];
               break;
            case 5:
               n1 = LtoGnode[e][4];
               n2 = LtoGnode[e][5];
               n3 = LtoGnode[e][6];
               n4 = LtoGnode[e][7];
               break;
            }
           
            midFaceCoord[0] = 0.25 * (coord[n1][0] + coord[n2][0] + coord[n3][0] + coord[n4][0]);
            midFaceCoord[1] = 0.25 * (coord[n1][1] + coord[n2][1] + coord[n3][1] + coord[n4][1]);
            midFaceCoord[2] = 0.25 * (coord[n1][2] + coord[n2][2] + coord[n3][2] + coord[n4][2]);
         
         } else if (eType == 2) {   // Tetrahedral element
            printf("\n\n\nERROR: Tetrahedral elements are not implemented in function setupMidFaceNodes() yet!!!\n\n\n");
         }

         matchFound = 0;

         // Search if this new mid-face node coordinate was already found previously.
         int midFaceNode;
         for (int i = 0; i < NneighborMFN; i++) {
            midFaceNode = neighborMFN[i];   // Neighboring mid-face node
            if (abs(midFaceCoord[0] - coord[midFaceNode][0]) < SMALL) {
               if (abs(midFaceCoord[1] - coord[midFaceNode][1]) < SMALL) {
                  if (abs(midFaceCoord[2] - coord[midFaceNode][2]) < SMALL) {   // Match found, this is not a new node.
                     LtoGnode[e][f+NEC+NEE] = midFaceNode;
                     matchFound = 1;
                     break;
                  }
               }
            }
         }

         /*
         // Search if this new coordinate was already found previously.
         for (int i = lastNode; i<nodeCount; i++) {
            if (abs(midFaceCoord[0] - coord[i][0]) < SMALL) {
               if (abs(midFaceCoord[1] - coord[i][1]) < SMALL) {
                  if (abs(midFaceCoord[2] - coord[i][2]) < SMALL) {   // Match found, this is not a new node.
                     LtoGnode[e][f+NEC+NEE] = i;
                     matchFound = 1;
                     break;
                  }
               }
            }
         }
         */

         if (matchFound == 0) {   // No match found, this is a new node.
            LtoGnode[e][f+NEC+NEE] = nodeCount;
            coord[nodeCount][0] = midFaceCoord[0];
            coord[nodeCount][1] = midFaceCoord[1];
            coord[nodeCount][2] = midFaceCoord[2];
            nodeCount = nodeCount + 1;
         }
      }  // End of f (face) loop
   }  // End of e (element) loop

   delete[] neighborMFN;

   // Add the mid-element node as a new node.
   for (int e = 0; e < NE; e++) {
      if (eType == 1) {  // Hexahedral element
         n1 = LtoGnode[e][0];
         n2 = LtoGnode[e][1];
         n3 = LtoGnode[e][2];
         n4 = LtoGnode[e][3];
         n5 = LtoGnode[e][4];
         n6 = LtoGnode[e][5];
         n7 = LtoGnode[e][6];
         n8 = LtoGnode[e][7];
         
         midElemCoord[0] = 0.125 * (coord[n1][0] + coord[n2][0] + coord[n3][0] + coord[n4][0] + coord[n5][0] + coord[n6][0] + coord[n7][0] + coord[n8][0]);
         midElemCoord[1] = 0.125 * (coord[n1][1] + coord[n2][1] + coord[n3][1] + coord[n4][1] + coord[n5][1] + coord[n6][1] + coord[n7][1] + coord[n8][1]);
         midElemCoord[2] = 0.125 * (coord[n1][2] + coord[n2][2] + coord[n3][2] + coord[n4][2] + coord[n5][2] + coord[n6][2] + coord[n7][2] + coord[n8][2]);
         
         LtoGnode[e][NEC+NEE+NEF] = nodeCount;
         coord[nodeCount][0] = midElemCoord[0];
         coord[nodeCount][1] = midElemCoord[1];
         coord[nodeCount][2] = midElemCoord[2];
         nodeCount = nodeCount + 1;
      } else if (eType == 2) {   // Tetrahedral element
         printf("\n\n\nERROR: Tetrahedral elements are not implemented in function setupMidFaceNodes() yet!!!\n\n\n");
      }
  
   }  // End of e loop


   // From now on use NN instead of nodeCount
   NN = nodeCount;

   //for (int e=0; e<NE; e++) {
   //   for(int i=0; i<NENv; i++) {
   //      cout << e << "  " << i << "  " << LtoGnode[e][i] << endl;
   //   } 
   //}




   // Decrease the size of coord by copying it and reallocating it with the correct size.
   double **copyCoord;
   copyCoord = new double*[NN];
   for (int i=0; i<NN; i++) {
      copyCoord[i] = new double[3];
   }

   for (int i=0; i<NN; i++) {
      for (int j=0; j<3; j++) {
         copyCoord[i][j] = coord[i][j];
      }
   }

   for (int i = 0; i<NE*NENv; i++) {
      delete[] coord[i];
   }
   delete[] coord;


   // Reallocate coord with the correct size NN.
   coord = new double*[NN];
   for (int i=0; i<NN; i++) {
      coord[i] = new double[3];
   }

   // Copy coordCopy back to coord.
   for (int i=0; i<NN; i++) {
      for (int j=0; j<3; j++) {
         coord[i][j] = copyCoord[i][j];
      }
   }

   // Delete copyCoord
   for (int i = 0; i<NN; i++) {
      delete[] copyCoord[i];
   }
   delete[] copyCoord;


   delete[] NelemNeighbors;

   for (int i = 0; i<NE; i++) {
      delete[] elemNeighbors[i];
   }
   delete[] elemNeighbors;

}  // End of function setupNonCornerNodes()





//========================================================================
void setupLtoGdof()
//========================================================================
{
   // Sets up LtoGvel and LtoGpres for each element using LtoGnode. As an
   // example LtoGvel uses the following local unknown ordering for a
   // quadrilateral element with NENv = 27
   //
   // u0, u1, u2, ..., u25, u26, v0, v1, v2, ..., v25, v26, w0, w1, w2, ..., w25, w26

   LtoGvel = new int*[NE];
   for (int i=0; i<NE; i++) {
      LtoGvel[i] = new int[3*NENv];                                                                                                                  // TODO : Actually 3*NENv size is unnecessary. Just NENv is enough. In that case just LtoGnode is enough, there is not need for LtoGvel.
   }

   LtoGpres = new int*[NE];
   for (int i=0; i<NE; i++) {
      LtoGpres[i] = new int[NENp];
   }

   int velCounter, presCounter;

   for (int e = 0; e < NE; e++) {
      velCounter = 0;   // Velocity unknown counter
      presCounter = 0;  // Pressure unknown counter
  
      // u velocity unknowns
      for (int i = 0; i<NENv; i++) {
         LtoGvel[e][velCounter] = LtoGnode[e][i];
         velCounter = velCounter + 1;
      }
  
      // v velocity unknowns
      for (int i = 0; i<NENv; i++) {
         LtoGvel[e][velCounter] = NN + LtoGnode[e][i];
         velCounter = velCounter + 1;
      }

      // w velocity unknowns
      for (int i = 0; i<NENv; i++) {
         LtoGvel[e][velCounter] = 2*NN + LtoGnode[e][i];
         velCounter = velCounter + 1;
      }

      // pressure unknowns
      // Note that first pressure unknown is numbered as 0, but not 3*NN.
      for (int i = 0; i < NENp; i++) {
         LtoGpres[e][presCounter] = LtoGnode[e][i];                                                                                                  // TODO: Aren't LtoGpres and part of LtoGnode the same for NENp=8 hexa elements?
         presCounter = presCounter + 1;
      }
   }
   
   
   LtoGvel_1d = new int[NE*NENv*3];
   int count = 0;
   
   for (int e = 0; e < NE; e++) {
      for (int i = 0; i < NENv*3; i++) {
         LtoGvel_1d[count] = LtoGvel[e][i];
         count = count + 1;
      }
   }
   

   //  CONTROL
   //for (int e=0; e<NE; e++) {
      //for(int i=0; i<3*NENv; i++) {
         //cout << e << "  " << i << "  " << LtoGvel[e][i] << " | " << LtoGvel_1d[e*NENv*3+i] << endl;
      //}
      //cout << endl;
      ////for(int i=0; i<NENp; i++) {
         ////cout << e << "  " << i << "  " << LtoGpres[e][i] << endl;
      ////}
      ////cout << endl;
   //}
   

}  // End of function setupLtoGdof()





//========================================================================
void determineVelBCnodes()
//========================================================================
{
   // Element faces where velocity BCs are specified were read from the input
   // file. Now let's determine the actual nodes where these BCs are specified.

   int e, f, n1, n2, n3, n4, n5, whichBC;

   double* velBCinfo;   // Dummy variable to store which velocity BC is specified at a node.

   velBCinfo = new double[NN];

   for (int i = 0; i < NN; i++) {
      velBCinfo[i] = -1;    // Initialize to -1.
   }

   for (int i = 0; i < BCnVelFaces; i++) {
      e       = BCvelFaces[i][0];   // Element where velocity BC is specified.
      f       = BCvelFaces[i][1];   // Face where velocity BC is specified.
      whichBC = BCvelFaces[i][2];   // Number of specified BC.
  
      // Consider corner nodes of the face
      if (eType == 1) {   // Hexahedral element
         switch (f) {
         case 0:
            n1 = LtoGnode[e][0];
            n2 = LtoGnode[e][1];
            n3 = LtoGnode[e][2];
            n4 = LtoGnode[e][3];
            break;  
         case 1:
            n1 = LtoGnode[e][0];
            n2 = LtoGnode[e][1];
            n3 = LtoGnode[e][4];
            n4 = LtoGnode[e][5];
            break;
         case 2:
            n1 = LtoGnode[e][1];
            n2 = LtoGnode[e][2];
            n3 = LtoGnode[e][5];
            n4 = LtoGnode[e][6];
            break;
         case 3:
            n1 = LtoGnode[e][2];
            n2 = LtoGnode[e][3];
            n3 = LtoGnode[e][6];
            n4 = LtoGnode[e][7];
            break;
         case 4:
            n1 = LtoGnode[e][0];
            n2 = LtoGnode[e][3];
            n3 = LtoGnode[e][4];
            n4 = LtoGnode[e][7];
            break;
         case 5:
            n1 = LtoGnode[e][4];
            n2 = LtoGnode[e][5];
            n3 = LtoGnode[e][6];
            n4 = LtoGnode[e][7];
            break;
         }

         velBCinfo[n1] = whichBC;
         velBCinfo[n2] = whichBC;
         velBCinfo[n3] = whichBC;
         velBCinfo[n4] = whichBC;

      } else if (eType == 2) {   // Tetrahedral element
        printf("\n\n\nERROR: Tetrahedral elements are not implemented in function determineVelBCnodes() yet!!!\n\n\n");
      }
  
  
      // Consider mid-edge and mid-face nodes if there are any.
      if (NENp != NENv) {
         if (eType == 1) {  // Hexahedral element
            switch (f) {
            case 0:
               n1 = LtoGnode[e][8];
               n2 = LtoGnode[e][9];
               n3 = LtoGnode[e][10];
               n4 = LtoGnode[e][11];
               n5 = LtoGnode[e][20];
               break;
            case 1:
               n1 = LtoGnode[e][8];
               n2 = LtoGnode[e][12];
               n3 = LtoGnode[e][13];
               n4 = LtoGnode[e][16];
               n5 = LtoGnode[e][21];
               break;
            case 2:
               n1 = LtoGnode[e][9];
               n2 = LtoGnode[e][13];
               n3 = LtoGnode[e][14];
               n4 = LtoGnode[e][17];
               n5 = LtoGnode[e][22];
               break;
            case 3:
               n1 = LtoGnode[e][10];
               n2 = LtoGnode[e][14];
               n3 = LtoGnode[e][15];
               n4 = LtoGnode[e][18];
               n5 = LtoGnode[e][23];
               break;
            case 4:
               n1 = LtoGnode[e][11];
               n2 = LtoGnode[e][12];
               n3 = LtoGnode[e][15];
               n4 = LtoGnode[e][19];
               n5 = LtoGnode[e][24];
               break;
            case 5:
               n1 = LtoGnode[e][16];
               n2 = LtoGnode[e][17];
               n3 = LtoGnode[e][18];
               n4 = LtoGnode[e][19];
               n5 = LtoGnode[e][25];
               break;
            }

            velBCinfo[n1] = whichBC;
            velBCinfo[n2] = whichBC;
            velBCinfo[n3] = whichBC;
            velBCinfo[n4] = whichBC;
            velBCinfo[n5] = whichBC;

         } else if (eType == 2) {   // Tetrahedral element
            printf("\n\n\nERROR: Tetrahedral elements are not implemented in function determineVelBCnodes() yet!!!\n\n\n");
         }
      }
   }  // End of BCvelFaces loop


   // Count the number of velocity BC nodes
   BCnVelNodes = 0;
   for (int i = 0; i < NN; i++) {
      if (velBCinfo[i] != -1) {
         BCnVelNodes = BCnVelNodes + 1;
      }
   }

   // Store velBCinfo variable as BCvelNodes
   BCvelNodes = new int*[BCnVelNodes];

   for (int i = 0; i < BCnVelNodes; i++) {
      BCvelNodes[i] = new int[2];
   }

   int counter = 0;
   for (int i = 0; i < NN; i++) {
      if (velBCinfo[i] != -1) {
         BCvelNodes[counter][0] = i;
         BCvelNodes[counter][1] = int(velBCinfo[i]);
         counter = counter + 1;
      }
   }

   delete[] velBCinfo;
   
   for (int i = 0; i<BCnVelFaces; i++) {
      delete[] BCvelFaces[i];
   }
   delete[] BCvelFaces;


   //  CONTROL
   //for (int i=0; i<BCnVelNodes; i++) {
   //   cout << i << "  " << "  " << BCvelNodes[i][0] << "  " << BCvelNodes[i][1] << endl;
   //}
    
}  // End of function determineVelBCnodes()





//========================================================================
void findElemsOfVelNodes()
//========================================================================
{
   // Determines elements connected to velocity nodes (elemsOfVelNodes).
   // It is necessary for sparse storage. It is stored in a matrix of size
   // NNx10, where 10 is a number, estimated to be larger than the maximum 
   // number of elements connected to a velocity node.

   // Also an array (NelemOfVelNodes) stores the actual number of elements
   // connected to each velocity node.

   int LARGE = 10;  // It is assumed that not more than 10 elements are
                    // connected to a velocity node.
                                                                                                                                                     // TODO: Define this somewhere else, which will be easy to notice.
                                                                                                                                                     // TODO: Make sure that this is not violated.
   int node;

   elemsOfVelNodes = new int*[NN];
   for (int i=0; i<NN; i++) {
      elemsOfVelNodes[i] = new int[LARGE];
   }

   NelemOfVelNodes  = new int[NN];

   for (int i = 0; i < NN; i++) {
      NelemOfVelNodes[i] = 0;       // Initialize to zero
   }

   // Form elemsOfVelNodes using LtoGvel of each element
   for (int e = 0; e < NE; e++) {
      for (int i = 0; i < NENv; i++) {
         node = LtoGvel[e][i];
         elemsOfVelNodes[node][NelemOfVelNodes[node]] = e;
         NelemOfVelNodes[node] = NelemOfVelNodes[node] + 1;
      }
   }

   //  CONTROL
   /*
   for (int i=0; i<NN; i++) {
      cout << i << "  " << NelemOfVelNodes[i] << endl;
   }

   cout << endl;

   for (int i=0; i<NNp; i++) {
      cout << i << "  " << NelemOfPresNodes[i] << endl;
   }
   */

}  // End of function findElemsOfVelNodes()





//========================================================================
void findMonitorPoint()
//========================================================================
{
   // Find the point that is closest to the monitor point coordinates read
   // from the input file.

   double distance = 1e6;   // Initialize to a large value

   double dx, dy, dz;

   for (int i = 0; i < NCN; i++) {
      dx = coord[i][0] - monPointCoord[0];
      dy = coord[i][1] - monPointCoord[1];
      dz = coord[i][2] - monPointCoord[2];
      
      if (sqrt(dx*dx + dy*dy + dz*dz) < distance) {
         distance = sqrt(dx*dx + dy*dy + dz*dz);
         monPoint = i;
      }
   }

   //  CONTROL
   // cout << monPoint << endl;

}  // End of function findMonitorPoint()





//========================================================================
void setupSparseM()
//========================================================================
{
   // Sets up row and column arrays of the global mass matrix. It is only used
   // in lumped form. Its size is 3NNx3NN. It is consisted of three identical
   // parts; upper-left, middle and lower right. Data for only upper-left part
   // will be stored and the others will be generated as required.


   // Work only with the upper-left part of [M].

   // In each row, find the columns with nonzero entries.

   int colCount;
   int LARGE;   // Maximum number of elements connected to a velocity node.

   // Determine the maximum number of elements connected to a velocity node.
   LARGE = 0;   // Initialize to a low value
   for (int i = 0; i < NN; i++) {
      if (NelemOfVelNodes[i] > LARGE) {
         LARGE = NelemOfVelNodes[i];
      }
   }

   // CONTROL
   // cout << endl << "LARGE = " << LARGE << endl;

   int sparseM_NNZ_onePart = 0;  // Counts nonzero entries in only 1 sub-mass matrix.

   int *NNZcolInARow;      // Number of nonzero columns in each row. This is nothing but the list of nodes that are in communication with each node.
   int **NZcolsInARow;     // Nonzero columns in each row.

   NNZcolInARow = new int[NN];

   NZcolsInARow = new int*[NN];
   for (int i=0; i<NN; i++) {
      NZcolsInARow[i] = new int[LARGE*NENv];                                                                                                         // TODO: This array may take too much memory in 3D. Instead this information can be stored in a CSR type arrangement.
   }

   int *isColNZ;      // A flag array to store whether the column is zero or not.
                      // Stores similar information as NZcolsInARow, but makes counting nonzeros easier.
   isColNZ = new int[NN];

   for (int i = 0; i < NN; i++) {
      isColNZ[i] = 0;
   }

   for (int r = 0; r < NN; r++) {   // Loop over all rows
      colCount = 0;
  
      for (int i = 0; i < NelemOfVelNodes[r]; i++) {   // NelemOfVelNodes[r] is the number of elements connected to node r
         int e = elemsOfVelNodes[r][i];    // This element contributes to row r.
         for (int j = 0; j < NENv; j++) {
            if (isColNZ[LtoGvel[e][j]] == 0) {   // 0 means this column had no previous non zero contribution.
               isColNZ[LtoGvel[e][j]] = 1;    // 1 means this column is non zero.
               NZcolsInARow[r][colCount] = LtoGvel[e][j];
               colCount = colCount + 1;
            }
         }
      }
  
      NNZcolInARow[r] = colCount;
      sparseM_NNZ_onePart = sparseM_NNZ_onePart + colCount;

      // Get ready for the next row. Set non-zero values of isColNZ to zero.
      for (int i = 0; i < NNZcolInARow[r]; i++) {
         isColNZ[ NZcolsInARow[r][i] ]  = 0;
      }
   }

   delete[] isColNZ;

   // Entries in each row of NZcolsInARow are not sorted. Let's sort them out.
   vector<int> toBeSorted;
   int nnz;
   for (int r = 0; r < NN; r++) {
      nnz = NNZcolInARow[r];   // Number of nonzeros in row r
      toBeSorted.resize(nnz);  // Will store nonzeros of row r for sorting.
      for (int i = 0; i < nnz; i++) {
         toBeSorted[i] = NZcolsInARow[r][i];   // Nonzero columns of row r (unsorted)
      }
      std::sort(toBeSorted.begin(), toBeSorted.end());     // Sorted version.
      
      // Copy sorted array back to NZcolsInARow
      for (int i = 0; i < nnz; i++) {
         NZcolsInARow[r][i] = toBeSorted[i];
      }
      
      toBeSorted.clear();  // Delete the entries in toBeSorted
   }

   /* CONTROL
   for (int i = 0; i < NN; i++) {
      for (int j = 0; j < NNZcolInARow[i]; j++) {
         cout << NZcolsInARow[i][j] << "  ";
      }
      cout << endl;
   }
   */

   waitForUser("OK1. Enter a character... ");

   // Allocate memory for 3 vectors of sparseM. Thinking about the whole
   // mass matrix, let's define the sizes properly by using three times of the
   // calculated NNZ.
   sparseMcol   = new int[sparseM_NNZ_onePart];
   sparseMrow   = new int[sparseM_NNZ_onePart];
   sparseMvalue = new double[sparseM_NNZ_onePart];

   waitForUser("OK2,3,4. Enter a character... ");
   
   // Fill in soln.sparseM.col and soln.sparseM.row arrays
   // This is done in a row-by-row way.
   int NNZcounter = 0;
   for (int r = 0; r < NN; r++) {    // Loop over all rows
      for (int i = 0; i < NNZcolInARow[r]; i++) {
         sparseMrow[NNZcounter] = r;
         sparseMcol[NNZcounter] = NZcolsInARow[r][i];
         NNZcounter = NNZcounter + 1;
      }
   }

   sparseM_NNZ = 3 * sparseM_NNZ_onePart;   // Triple the number of nonzeros.
   
   // Sparse storage of the K matrix is the same as M. Only extra
   // value arrays are necessary.
   sparseKvalue = new double[sparseM_NNZ/3];    // Only store the nonzeros of the upper-left sub matrix.
   sparseAvalue = new double[sparseM_NNZ/3];    // Stores the system that arises from momentum equation.
   
   waitForUser("OK5,6. Enter a character... ");

   // For CSR storage necessary for MKL and CUSP, we need row starts array too.
   sparseMrowStarts = new int[NN+1];
   sparseMrowStarts[0] = 0;
   for (int i = 1; i <= NN; i++) {
      sparseMrowStarts[i] = sparseMrowStarts[i-1] + NNZcolInARow[i-1];
   }
   
   // MKL also needs the following modified version of sparseMrowStarts
   sparseMrowStartsMod = new int[NN];
   for (int i = 0; i < NN; i++) {
      sparseMrowStartsMod[i] = sparseMrowStarts[i+1];
   }
   
   // CONTROL
   //cout << sparseMrowStarts[NN+1]  << "   "  << sparseM_NNZ/3 << endl;
   //for (int i = 0; i < NN+1; i++) {
   //   cout << sparseMrowStarts[i] << endl;
   //}

   // Determine local-to-sparse mapping, i.e. find the location of the entries
   // of elemental sub-mass matrices in sparse storage. This will be used in
   // the assembly process.

   // First determine the nonzero entry number at the beginning of each row.
   // rowStarts[NN] is equal to NNZ+1.                                                                                                               // TODO: Is NNZ+1 correct for C++?
   int *rowStarts;
   rowStarts = new int[NN+1];

   waitForUser("OK7. Enter a character... ");

   rowStarts[0] = 0;   // First row starts with the zeroth entry
   for (int i = 1; i < NN+1; i++) {
      rowStarts[i] = rowStarts[i-1] + NNZcolInARow[i-1];
   }

   // CONTROL
   //for (int i = 0; i < NN+1; i++) {
   //   cout << rowStarts[i] << endl;
   //}

   sparseMapM = new int **[NE];
   for (int i = 0; i < NE; i++) {
      sparseMapM[i] = new int *[NENv];
      for (int j = 0; j < NENv; j++) {
         sparseMapM[i][j] = new int[NENv];
      }
   }

   waitForUser("OK8. Enter a character... ");

   int r, c;

   for (int e = 0; e < NE; e++) {
      for (int i = 0; i < NENv; i++) {
         r = LtoGvel[e][i];

         int *col;    // Nonzeros of row r
         col = new int[rowStarts[r+1] - rowStarts[r]];
         
         for (int j = 0; j < NNZcolInARow[r]; j++) {
            col[j] = sparseMcol[j + rowStarts[r]];
         }

         for (int j = 0; j < NENv; j++) {
            // Find the location in the col array
            int jj = LtoGvel[e][j];
            for (c = 0; c < NNZcolInARow[r]; c++) {
               if (jj == col[c]) {
                  break;
               }
            }
            sparseMapM[e][i][j] = c + rowStarts[r];
         }
         delete[] col;
      }
   }
   
   
   sparseMapM_1d = new int[NE*NENv*NENv];
      
   int count = 0;

   for (int e = 0; e < NE; e++) {
      for (int i = 0; i < NENv; i++) {
         for (int j = 0; j < NENv; j++) {
            sparseMapM_1d[count] = sparseMapM[e][i][j];
            count = count + 1;
         }
      }
   }
   
   
   // CONTROL
   //for (int e = 0; e < NE; e++) {
      //for (int i = 0; i < NENv; i++) {
         //for (int j = 0; j < NENv; j++) {
            //cout << e << "  " << i << "  " << j << "  " <<  sparseMapM[e][i][j] << " | " <<
             //sparseMapM_1d[e*NENv*NENv + i*NENv + j] << endl;
         //}
      //}
      //cout << endl;
   //}
   

   for (int i = 0; i<NN; i++) {
      delete[] NZcolsInARow[i];
   }
   delete[] NZcolsInARow;

   delete[] NNZcolInARow;
   delete[] rowStarts;

}  // End of function setupSparseM()





//========================================================================
void setupSparseG()
//========================================================================
{
   // Sets up row and column arrays of the global G matrix.

   // First work only with the upper part of the matrix and then extend it
   // for the lower part, which is identical to the upper part.


   // In each row, find the columns with nonzero entries.

   int colCount;
   int LARGE;   // Maximum number of elements connected to a pressure node.

   // Determine the maximum number of elements connected to a pressure node.
   LARGE = 0;   // Initialize to a low value
   for (int i = 0; i < NNp; i++) {
      if (NelemOfPresNodes[i] > LARGE) {
         LARGE = NelemOfPresNodes[i];
      }
   }

   // CONTROL
   // cout << endl << "LARGE = " << LARGE << endl;

   int sparseG_NNZ_onePart = 0;  // Counts nonzero entries in sub-G matrix.

   int *NNZcolInARow;      // Number of nonzero columns in each row. This is nothing but the list of nodes that are in communication with each node.
   int **NZcolsInARow;     // Nonzero columns in each row.

   NNZcolInARow = new int[NN];

   NZcolsInARow = new int*[NN];
   for (int i=0; i<NN; i++) {
      NZcolsInARow[i] = new int[LARGE*NENp];                                                                                                         // TODO: This array may take too much memory in 3D. Instead this information can be stored in a CSR type arrangement.
   }

   int *isColNZ;      // A flag array to store whether the column is zero or not.
                      // Stores similar information as NZcolsInARow, but makes counting nonzeros easier.
   isColNZ = new int[NNp];

   for (int i = 0; i < NNp; i++) {
      isColNZ[i] = 0;
   }
   
   for (int r = 0; r < NN; r++) {   // Loop over all rows
      colCount = 0;
  
      for (int i = 0; i < NelemOfVelNodes[r]; i++) {   // NelemOfVelNodes[r] is the number of elements connected to node r
         int e = elemsOfVelNodes[r][i];    // This element contributes to row r.
         for (int j = 0; j < NENp; j++) {
            if (isColNZ[LtoGpres[e][j]] == 0) {   // 0 means this column had no previous non zero contribution.
               isColNZ[LtoGpres[e][j]] = 1;    // 1 means this column is non zero.
               NZcolsInARow[r][colCount] = LtoGpres[e][j];
               colCount = colCount + 1;
            }
         }
      }
  
      NNZcolInARow[r] = colCount;
      sparseG_NNZ_onePart = sparseG_NNZ_onePart + colCount;

      // Get ready for the next row. Set non-zero values of isColNZ to zero.
      for (int i = 0; i < NNZcolInARow[r]; i++) {
         isColNZ[ NZcolsInARow[r][i] ]  = 0;
      }
   }

   delete[] isColNZ;

   // Entries in each row of NZcolsInARow are not sorted. Let's sort them out.
   vector<int> toBeSorted;
   int nnz;
   for (int r = 0; r < NN; r++) {
      nnz = NNZcolInARow[r];   // Number of nonzeros in row r
      toBeSorted.resize(nnz);  // Will store nonzeros of row r for sorting.
      for (int i = 0; i < nnz; i++) {
         toBeSorted[i] = NZcolsInARow[r][i];   // Nonzero columns of row r (unsorted)
      }
      std::sort(toBeSorted.begin(), toBeSorted.end());     // Sorted version.
      
      // Copy sorted array back to NZcolsInARow
      for (int i = 0; i < nnz; i++) {
         NZcolsInARow[r][i] = toBeSorted[i];
      }
      
      toBeSorted.clear();  // Delete the entries in toBeSorted
   }

   /* CONTROL
   for (int i = 0; i < NN; i++) {
      for (int j = 0; j < NNZcolInARow[i]; j++) {
         cout << NZcolsInARow[i][j] << "  ";
      }
      cout << endl;
   }
   */


   // Allocate memory for 3 vectors of sparseG. G matrix consiss of three sub matrices.
   // They all have the same sparsity structure with different value vectors.
   sparseGcol   = new int[sparseG_NNZ_onePart];
   sparseGrow   = new int[sparseG_NNZ_onePart];
   sparseG1value = new double[sparseG_NNZ_onePart];
   sparseG2value = new double[sparseG_NNZ_onePart];
   sparseG3value = new double[sparseG_NNZ_onePart];


   // Fill in soln.sparseG.col and soln.sparseG.row arrays
   // This is done in a row-by-row way.
   int NNZcounter = 0;
   for (int r = 0; r < NN; r++) {    // Loop over all rows
      for (int i = 0; i < NNZcolInARow[r]; i++) {
         sparseGrow[NNZcounter] = r;
         sparseGcol[NNZcounter] = NZcolsInARow[r][i];
         NNZcounter = NNZcounter + 1;
      }
   }

   sparseG_NNZ = 3 * sparseG_NNZ_onePart;   // Triple the number of nonzeros.
   


   // For CSR storage necessary for MKL and CUSP, we need row starts array too.                                                                      // TODO: This is repeated below.
   sparseGrowStarts = new int[NN+1];
   sparseGrowStarts[0] = 0;
   for (int i = 1; i <= NN; i++) {
      sparseGrowStarts[i] = sparseGrowStarts[i-1] + NNZcolInARow[i-1];
   }

   // MKL also needs the following modified version of sparseGrowStarts
   sparseGrowStartsMod = new int[NN];
   for (int i = 0; i < NN; i++) {
      sparseGrowStartsMod[i] = sparseGrowStarts[i+1];
   };

   // CONTROL
   //cout << sparseGrowStarts[NN+1]  << "   "  << sparseG_NNZ/3 << endl;
   //for (int i = 0; i < NN+1; i++) {
   //   cout << sparseGrowStarts[i] << endl;
   //}
   //cout << "\n\n\n\n";


   // Determine local-to-sparse mapping, i.e. find the location of the entries
   // of elemental sub-G matrices in sparse storage. This will be used in
   // the assembly process.

   // First determine the nonzero entry number at the beginning of each row.
   // rowStarts[NN] is equal to NNZ+1.                                                                                                               // TODO: Is NNZ+1 correct for C++?
   int *rowStarts;
   rowStarts = new int[NN+1];
   rowStarts[0] = 0;   // First row starts with the zeroth entry
   for (int i = 1; i < NN+1; i++) {
      rowStarts[i] = rowStarts[i-1] + NNZcolInARow[i-1];
      //  CONTROL
      //cout << "i = " << i << "   rowStarts[i] = " << rowStarts[i] << endl;
   }


   sparseMapG = new int **[NE];
   for (int i = 0; i < NE; i++) {
      sparseMapG[i] = new int *[NENv];
      for (int j = 0; j < NENv; j++) {
         sparseMapG[i][j] = new int[NENp];
      }
   }

   int r, c;

   for (int e = 0; e < NE; e++) {
      for (int i = 0; i < NENv; i++) {
         r = LtoGvel[e][i];

         int *col;    // Nonzeros of row r
         col = new int[rowStarts[r+1] - rowStarts[r]];
         
         for (int j = 0; j < NNZcolInARow[r]; j++) {
            col[j] = sparseGcol[j + rowStarts[r]];
         }

         for (int j = 0; j < NENp; j++) {
            // Find the location in the col array
            int jj = LtoGpres[e][j];
            for (c = 0; c < NNZcolInARow[r]; c++){
               if (jj == col[c]) {
                  break;
               }
            }
            sparseMapG[e][i][j] = c + rowStarts[r];
          }
         delete[] col;
       }
   }

   /* CONTROL
   for (int e = 0; e < NE; e++) {
      for (int i = 0; i < NENv; i++) {
         for (int j = 0; j < NENp; j++) {
            cout << sparseMapG[e][i][j] << endl;
         }
      }
      cout << endl;
   }
   */


   for (int i = 0; i<NN; i++) {
      delete[] NZcolsInARow[i];
   }
   delete[] NZcolsInARow;

   delete[] NNZcolInARow;

   delete[] rowStarts;

}  // End of function setupSparseG()





//========================================================================
void setupSparseZ()
//========================================================================
{
   // Sets up row and column arrays of the Laplacian matrix for pressure. Its size is NNpxNNp.

   // In each row, find the columns with nonzero entries.

   int colCount;
   int LARGE;   // Maximum number of elements connected to a pressure node.

   // Determine the maximum number of elements connected to a pressure node.
   LARGE = 0;   // Initialize to a low value
   for (int i = 0; i < NNp; i++) {
      if (NelemOfPresNodes[i] > LARGE) {
         LARGE = NelemOfPresNodes[i];
      }
   }

   // CONTROL
   // cout << endl << "LARGE = " << LARGE << endl;

   sparseZ_NNZ = 0;  // Counts nonzero entries in Z matrix.

   int *NNZcolInARow;      // Number of nonzero columns in each row. This is nothing but the list of nodes that are in communication with each node.
   int **NZcolsInARow;     // Nonzero columns in each row.

   NNZcolInARow = new int[NNp];

   NZcolsInARow = new int*[NNp];
   for (int i=0; i<NNp; i++) {
      NZcolsInARow[i] = new int[LARGE*NENp];                                                                                                         // TODO: This array may take too much memory in 3D. Instead this information can be stored in a CSR type arrangement.
   }

   int *isColNZ;      // A flag array to store whether the column is zero or not.
                      // Stores similar information as NZcolsInARow, but makes counting nonzeros easier.
   isColNZ = new int[NNp];

   for (int i = 0; i < NNp; i++) {
      isColNZ[i] = 0;
   }
   
   for (int r = 0; r < NNp; r++) {   // Loop over all rows
      colCount = 0;
  
      for (int i = 0; i < NelemOfPresNodes[r]; i++) {   // NelemOfVelNodes[r] is the number of elements connected to node r
         int e = elemsOfPresNodes[r][i];    // This element contributes to row r.
         for (int j = 0; j < NENp; j++) {
            if (LtoGpres[e][j] > (r-1)) {
               if (isColNZ[LtoGpres[e][j]] == 0) {   // 0 means this column had no previous non zero contribution.
                  isColNZ[LtoGpres[e][j]] = 1;    // 1 means this column is non zero.
                  NZcolsInARow[r][colCount] = LtoGpres[e][j];
                  colCount = colCount + 1;
               }
            }
         }
      }
  
      NNZcolInARow[r] = colCount;
      sparseZ_NNZ = sparseZ_NNZ + colCount;

      // Get ready for the next row. Set non-zero values of isColNZ to zero.
      for (int i = 0; i < NNZcolInARow[r]; i++) {
         isColNZ[ NZcolsInARow[r][i] ]  = 0;
      }
   }

   delete[] isColNZ;

   // Entries in each row of NZcolsInARow are not sorted. Let's sort them out.
   vector<int> toBeSorted;
   int nnz;
   for (int r = 0; r < NNp; r++) {
      nnz = NNZcolInARow[r];   // Number of nonzeros in row r
      toBeSorted.resize(nnz);  // Will store nonzeros of row r for sorting.
      for (int i = 0; i < nnz; i++) {
         toBeSorted[i] = NZcolsInARow[r][i];   // Nonzero columns of row r (unsorted)
      }
      std::sort(toBeSorted.begin(), toBeSorted.end());     // Sorted version.
      
      // Copy sorted array back to NZcolsInARow
      for (int i = 0; i < nnz; i++) {
         NZcolsInARow[r][i] = toBeSorted[i];
      }
      
      toBeSorted.clear();  // Delete the entries in toBeSorted
   }

   /* CONTROL
   for (int i = 0; i < NNp; i++) {
      for (int j = 0; j < NNZcolInARow[i]; j++) {
         cout << NZcolsInARow[i][j] << "  ";
      }
      cout << endl;
   }
   */


   // Allocate memory for 3 vectors of sparseZ.
   sparseZcol   = new int[sparseZ_NNZ];
   sparseZrow   = new int[sparseZ_NNZ];
   sparseZvalue = new double[sparseZ_NNZ];
   
   
   // Fill in soln.sparseZ.col and soln.sparseZ.row arrays
   // This is done in a row-by-row way.
   int NNZcounter = 0;
   for (int r = 0; r < NNp; r++) {    // Loop over all rows
      for (int i = 0; i < NNZcolInARow[r]; i++) {
         sparseZrow[NNZcounter] = r;
         sparseZcol[NNZcounter] = NZcolsInARow[r][i];
         NNZcounter = NNZcounter + 1;
      }
   }


   // For CSR storage necessary for MKL and CUSP, we need row starts array too.                                                                      // TODO: This is repeated below.
   sparseZrowStarts = new int[NNp+1];
   sparseZrowStarts[0] = 0;
   for (int i = 1; i <= NNp; i++) {
      sparseZrowStarts[i] = sparseZrowStarts[i-1] + NNZcolInARow[i-1];
   }

   // MKL also needs the following modified version of sparseZrowStarts
   sparseZrowStartsMod = new int[NNp];
   for (int i = 0; i < NNp; i++) {
      sparseZrowStartsMod[i] = sparseZrowStarts[i+1];
   }

   //CONTROL
   //cout << endl;
   //cout << sparseZrowStarts[NNp]  << "   "  << sparseZ_NNZ << endl;
   //for (int i = 0; i < NNp; i++) {
      //cout << sparseZrowStarts[i] << endl;
      //for (int j = sparseZrowStarts[i]; j < sparseZrowStarts[i+1]; j++) {
         //cout << sparseZcol[j] << "  " ;
      //}
      //cout << endl;
   //}
   //cout << "\n\n\n\n";


   // Determine local-to-sparse mapping, i.e. find the location of the entries
   // of elemental sub-Z matrices in sparse storage. This will be used in
   // the assembly process.

   // First determine the nonzero entry number at the beginning of each row.
   // rowStarts[NNp] is equal to NNZ+1.                                                                                                               // TODO: Is NNZ+1 correct for C++?
   int *rowStarts;
   rowStarts = new int[NNp+1];
   rowStarts[0] = 0;   // First row starts with the zeroth entry
   for (int i = 1; i < NNp+1; i++) {
      rowStarts[i] = rowStarts[i-1] + NNZcolInARow[i-1];
      //  CONTROL
      //cout << "i = " << i << "   rowStarts[i] = " << rowStarts[i] << endl;
   }


   sparseMapZ = new int *[NE];
   for (int i = 0; i < NE; i++) {
      sparseMapZ[i] = new int [NENp*NENp];
   }
   
   sparseMapZupper = new int *[NE];
   for (int i = 0; i < NE; i++) {
      sparseMapZupper[i] = new int [NENp*NENp];
   }
   
   sparseMapZupperCount = new int [NE];

   
   int r, c, counterUpper, counter;

   for (int e = 0; e < NE; e++) {
      counterUpper = 0;
      counter = 0;
      for (int i = 0; i < NENp; i++) {
         r = LtoGpres[e][i];

         int *col;    // Nonzeros of row r
         col = new int[rowStarts[r+1] - rowStarts[r]];
         
         for (int j = 0; j < NNZcolInARow[r]; j++) {
            col[j] = sparseZcol[j + rowStarts[r]];
         }
         for (int j = 0; j < NENp; j++) {
            if (LtoGpres[e][j] > r-1) {
               // Find the location in the col array
               int jj = LtoGpres[e][j];
               for (c = 0; c < NNZcolInARow[r]; c++){
                  if (jj == col[c]) {
                     break;
                  }
               }
               sparseMapZ[e][counterUpper] = c + rowStarts[r];
               sparseMapZupper[e][counterUpper] = counter;
               counterUpper = counterUpper + 1;
            }
            counter = counter + 1;
         }
         delete[] col;
       }
       
       sparseMapZupperCount[e] = counterUpper;
       
   }
   
   // MKL's CG solver needs 1 based CSR vectors
   
   for (int i = 0; i <= NNp; i++) {
      sparseZrowStarts[i] = sparseZrowStarts[i] + 1;
   }
   
   for (int i = 0; i < sparseZ_NNZ; i++) {
      sparseZcol[i] = sparseZcol[i] + 1;
   }
   
   
   //CONTROL
   //cout << endl;
   //cout << endl;
   //for (int e = 0; e < 3; e++) {
      //cout << " e = " << e << endl;
      //for (int i = 0; i < NENp; i++) {
         //for (int j = 0; j < NENp; j++) {
            //cout << sparseMapZ[e][i][j] << endl;
         //}
      //}
      //cout << endl;
   //}
     
   for (int i = 0; i<NNp; i++) {
      delete[] NZcolsInARow[i];
   }
   delete[] NZcolsInARow;

   delete[] NNZcolInARow;

   delete[] rowStarts;
   
   for (int i = 0; i<NN; i++) {
      delete[] elemsOfVelNodes[i];
   }
   delete[] elemsOfVelNodes;


   for (int i = 0; i<NNp; i++) {
      delete[] elemsOfPresNodes[i];
   }
   delete[] elemsOfPresNodes;


   delete[] NelemOfVelNodes;
   delete[] NelemOfPresNodes;   

}  // End of function setupSparseZ()





//========================================================================
void setupSparseZGPU()
//========================================================================
{
   // Sets up row and column arrays of the Laplacian matrix for pressure. Its size is NNpxNNp.

   // In each row, find the columns with nonzero entries.

   int colCount;
   int LARGE;   // Maximum number of elements connected to a pressure node.

   // Determine the maximum number of elements connected to a pressure node.
   LARGE = 0;   // Initialize to a low value
   for (int i = 0; i < NNp; i++) {
      if (NelemOfPresNodes[i] > LARGE) {
         LARGE = NelemOfPresNodes[i];
      }
   }

   // CONTROL
   // cout << endl << "LARGE = " << LARGE << endl;

   sparseZ_NNZ = 0;  // Counts nonzero entries in Z matrix.

   int *NNZcolInARow;      // Number of nonzero columns in each row. This is nothing but the list of nodes that are in communication with each node.
   int **NZcolsInARow;     // Nonzero columns in each row.

   NNZcolInARow = new int[NNp];

   NZcolsInARow = new int*[NNp];
   for (int i=0; i<NNp; i++) {
      NZcolsInARow[i] = new int[LARGE*NENp];                                                                                                         // TODO: This array may take too much memory in 3D. Instead this information can be stored in a CSR type arrangement.
   }

   int *isColNZ;      // A flag array to store whether the column is zero or not.
                      // Stores similar information as NZcolsInARow, but makes counting nonzeros easier.
   isColNZ = new int[NNp];

   for (int i = 0; i < NNp; i++) {
      isColNZ[i] = 0;
   }
   
   for (int r = 0; r < NNp; r++) {   // Loop over all rows
      colCount = 0;
  
      for (int i = 0; i < NelemOfPresNodes[r]; i++) {   // NelemOfVelNodes[r] is the number of elements connected to node r
         int e = elemsOfPresNodes[r][i];    // This element contributes to row r.
         for (int j = 0; j < NENp; j++) {
            if (isColNZ[LtoGpres[e][j]] == 0) {   // 0 means this column had no previous non zero contribution.
               isColNZ[LtoGpres[e][j]] = 1;    // 1 means this column is non zero.
               NZcolsInARow[r][colCount] = LtoGpres[e][j];
               colCount = colCount + 1;
            }
         }
      }
  
      NNZcolInARow[r] = colCount;
      sparseZ_NNZ = sparseZ_NNZ + colCount;

      // Get ready for the next row. Set non-zero values of isColNZ to zero.
      for (int i = 0; i < NNZcolInARow[r]; i++) {
         isColNZ[ NZcolsInARow[r][i] ]  = 0;
      }
   }

   delete[] isColNZ;

   // Entries in each row of NZcolsInARow are not sorted. Let's sort them out.
   vector<int> toBeSorted;
   int nnz;
   for (int r = 0; r < NNp; r++) {
      nnz = NNZcolInARow[r];   // Number of nonzeros in row r
      toBeSorted.resize(nnz);  // Will store nonzeros of row r for sorting.
      for (int i = 0; i < nnz; i++) {
         toBeSorted[i] = NZcolsInARow[r][i];   // Nonzero columns of row r (unsorted)
      }
      std::sort(toBeSorted.begin(), toBeSorted.end());     // Sorted version.
      
      // Copy sorted array back to NZcolsInARow
      for (int i = 0; i < nnz; i++) {
         NZcolsInARow[r][i] = toBeSorted[i];
      }
      
      toBeSorted.clear();  // Delete the entries in toBeSorted
   }

   /* CONTROL
   for (int i = 0; i < NNp; i++) {
      for (int j = 0; j < NNZcolInARow[i]; j++) {
         cout << NZcolsInARow[i][j] << "  ";
      }
      cout << endl;
   }
   */


   // Allocate memory for 3 vectors of sparseZ.
   sparseZcol   = new int[sparseZ_NNZ];
   sparseZrow   = new int[sparseZ_NNZ];
   sparseZvalue = new double[sparseZ_NNZ];
   
   
   // Fill in soln.sparseZ.col and soln.sparseZ.row arrays
   // This is done in a row-by-row way.
   int NNZcounter = 0;
   for (int r = 0; r < NNp; r++) {    // Loop over all rows
      for (int i = 0; i < NNZcolInARow[r]; i++) {
         sparseZrow[NNZcounter] = r;
         sparseZcol[NNZcounter] = NZcolsInARow[r][i];
         NNZcounter = NNZcounter + 1;
      }
   }


   // For CSR storage necessary for MKL and CUSP, we need row starts array too.                                                                      // TODO: This is repeated below.
   sparseZrowStarts = new int[NNp+1];
   sparseZrowStarts[0] = 0;
   for (int i = 1; i <= NNp; i++) {
      sparseZrowStarts[i] = sparseZrowStarts[i-1] + NNZcolInARow[i-1];
   }

   // MKL also needs the following modified version of sparseZrowStarts
   sparseZrowStartsMod = new int[NNp];
   for (int i = 0; i < NNp; i++) {
      sparseZrowStartsMod[i] = sparseZrowStarts[i+1];
   };

   //CONTROL
   //cout << endl;
   //cout << sparseZrowStarts[NNp]  << "   "  << sparseZ_NNZ << endl;
   //for (int i = 0; i < NNp; i++) {
      //cout << sparseZrowStarts[i] << endl;
      //for (int j = sparseZrowStarts[i]; j < sparseZrowStarts[i+1]; j++) {
         //cout << sparseZcol[j] << "  " ;
      //}
      //cout << endl;
   //}
   //cout << "\n\n\n\n";


   // Determine local-to-sparse mapping, i.e. find the location of the entries
   // of elemental sub-Z matrices in sparse storage. This will be used in
   // the assembly process.

   // First determine the nonzero entry number at the beginning of each row.
   // rowStarts[NNp] is equal to NNZ+1.                                                                                                               // TODO: Is NNZ+1 correct for C++?
   int *rowStarts;
   rowStarts = new int[NNp+1];
   rowStarts[0] = 0;   // First row starts with the zeroth entry
   for (int i = 1; i < NNp+1; i++) {
      rowStarts[i] = rowStarts[i-1] + NNZcolInARow[i-1];
      //  CONTROL
      //cout << "i = " << i << "   rowStarts[i] = " << rowStarts[i] << endl;
   }


   sparseMapZGPU = new int **[NE];
   for (int i = 0; i < NE; i++) {
      sparseMapZGPU[i] = new int *[NENp];
      for (int j = 0; j < NENp; j++) {
         sparseMapZGPU[i][j] = new int[NENp];
      }
   }

   
   int r, c;

   for (int e = 0; e < NE; e++) {
      for (int i = 0; i < NENp; i++) {
         r = LtoGpres[e][i];

         int *col;    // Nonzeros of row r
         col = new int[rowStarts[r+1] - rowStarts[r]];
         
         for (int j = 0; j < NNZcolInARow[r]; j++) {
            col[j] = sparseZcol[j + rowStarts[r]];
         }

         for (int j = 0; j < NENp; j++) {
            // Find the location in the col array
            int jj = LtoGpres[e][j];
            for (c = 0; c < NNZcolInARow[r]; c++) {
               if (jj == col[c]) {
                  break;
               }
            }
            sparseMapZGPU[e][i][j] = c + rowStarts[r];
         }
         delete[] col;
      }
   }
   
   
   //CONTROL
   //cout << endl;
   //cout << endl;
   //for (int e = 0; e < 3; e++) {
      //cout << " e = " << e << endl;
      //for (int i = 0; i < NENp; i++) {
         //for (int j = 0; j < NENp; j++) {
            //cout << sparseMapZ[e][i][j] << endl;
         //}
      //}
      //cout << endl;
   //}
     
   for (int i = 0; i<NNp; i++) {
      delete[] NZcolsInARow[i];
   }
   delete[] NZcolsInARow;

   delete[] NNZcolInARow;

   delete[] rowStarts;
   
   for (int i = 0; i<NN; i++) {
      delete[] elemsOfVelNodes[i];
   }
   delete[] elemsOfVelNodes;


   for (int i = 0; i<NNp; i++) {
      delete[] elemsOfPresNodes[i];
   }
   delete[] elemsOfPresNodes;


   delete[] NelemOfVelNodes;
   delete[] NelemOfPresNodes;   

}  // End of function setupSparseZGPU()





//========================================================================
void setupGQ()
//========================================================================
{
   GQpoint = new double*[NGP];
   for (int i=0; i<NGP; i++) {
      GQpoint[i] = new double[3];
   }

   GQweight = new double[NGP];
   
   if (eType == 1) {         // Hexahedral element
      if (NGP == 1)  {          // 1 point quadrature
         GQpoint[0][0] = 0.0;  GQpoint[0][1] = 0.0;  GQpoint[0][2] = 0.0;
         GQweight[0] = 4.0;                                                                                                                                    // TODO: Is this correct?
      } else if (NGP == 8)  {   // 8 point quadrature
       GQpoint[0][0] = -sqrt(1./3);   GQpoint[0][1] = -sqrt(1./3);   GQpoint[0][2] = -sqrt(1./3);
       GQpoint[1][0] = sqrt(1./3);    GQpoint[1][1] = -sqrt(1./3);   GQpoint[1][2] = -sqrt(1./3);
       GQpoint[2][0] = -sqrt(1./3);   GQpoint[2][1] = sqrt(1./3);    GQpoint[2][2] = -sqrt(1./3);
       GQpoint[3][0] = sqrt(1./3);    GQpoint[3][1] = sqrt(1./3);    GQpoint[3][2] = -sqrt(1./3);
       GQpoint[4][0] = -sqrt(1./3);   GQpoint[4][1] = -sqrt(1./3);   GQpoint[4][2] = sqrt(1./3);
       GQpoint[5][0] = sqrt(1./3);    GQpoint[5][1] = -sqrt(1./3);   GQpoint[5][2] = sqrt(1./3);
       GQpoint[6][0] = -sqrt(1./3);   GQpoint[6][1] = sqrt(1./3);    GQpoint[6][2] = sqrt(1./3);
       GQpoint[7][0] = sqrt(1./3);    GQpoint[7][1] = sqrt(1./3);    GQpoint[7][2] = sqrt(1./3);
       GQweight[0] = 1.0;
       GQweight[1] = 1.0;
       GQweight[2] = 1.0;
       GQweight[3] = 1.0;
       GQweight[4] = 1.0;
       GQweight[5] = 1.0;
       GQweight[6] = 1.0;
       GQweight[7] = 1.0;
     } else if (NGP == 27) {    // 27 point quadrature
 
     // TODO : ...
 
     }
     
   } else if (eType == 2) {  // Tetrahedral element  
     
     // TODO : ...
     
   }
}  // End of function setupGQ()





//========================================================================
void calcShape()
//========================================================================
{
   // Calculates the values of the shape functions and their derivatives with
   // respect to ksi and eta at GQ points.

   // Sv, Sp   : Shape functions for velocity and pressure approximation.
   // dSv, dSp : ksi and eta derivatives of Sv and Sp.

   double ksi, eta, zeta;

   Sv = new double*[NGP];
   for (int i=0; i<NGP; i++) {
      Sv[i] = new double[NENv];
   }

   Sp = new double*[NGP];
   for (int i=0; i<NGP; i++) {
      Sp[i] = new double[NENp];
   }

   dSv = new double**[3];
   for (int i=0; i<3; i++) {
      dSv[i] = new double*[NENv];
      for (int j=0; j<NENv; j++) {
         dSv[i][j] = new double[NGP];
      }
   }

   dSp = new double**[3];
   for (int i=0; i<3; i++) {
      dSp[i] = new double*[NENp];
      for (int j=0; j<NENp; j++) {
         dSp[i][j] = new double[NGP];
      }
   }

   if (eType == 1) {  // Hexahedral element
     
      if (NENp == 8) {
         for (int k = 0; k < NGP; k++) {
            ksi  = GQpoint[k][0];
            eta  = GQpoint[k][1];
            zeta = GQpoint[k][2];
         
            Sp[k][0] = 0.125*(1-ksi)*(1-eta)*(1-zeta);
            Sp[k][1] = 0.125*(1+ksi)*(1-eta)*(1-zeta);
            Sp[k][2] = 0.125*(1+ksi)*(1+eta)*(1-zeta);
            Sp[k][3] = 0.125*(1-ksi)*(1+eta)*(1-zeta);
            Sp[k][4] = 0.125*(1-ksi)*(1-eta)*(1+zeta);
            Sp[k][5] = 0.125*(1+ksi)*(1-eta)*(1+zeta);
            Sp[k][6] = 0.125*(1+ksi)*(1+eta)*(1+zeta);
            Sp[k][7] = 0.125*(1-ksi)*(1+eta)*(1+zeta);
            
            // ksi derivatives of Sp
            dSp[0][0][k] = -0.125*(1-eta)*(1-zeta);
            dSp[0][1][k] =  0.125*(1-eta)*(1-zeta);
            dSp[0][2][k] =  0.125*(1+eta)*(1-zeta);
            dSp[0][3][k] = -0.125*(1+eta)*(1-zeta);
            dSp[0][4][k] = -0.125*(1-eta)*(1+zeta);
            dSp[0][5][k] =  0.125*(1-eta)*(1+zeta);
            dSp[0][6][k] =  0.125*(1+eta)*(1+zeta);
            dSp[0][7][k] = -0.125*(1+eta)*(1+zeta);
            
            // eta derivatives of Sp
            dSp[1][0][k] = -0.125*(1-ksi)*(1-zeta);
            dSp[1][1][k] = -0.125*(1+ksi)*(1-zeta);
            dSp[1][2][k] =  0.125*(1+ksi)*(1-zeta);
            dSp[1][3][k] =  0.125*(1-ksi)*(1-zeta);
            dSp[1][4][k] = -0.125*(1-ksi)*(1+zeta);
            dSp[1][5][k] = -0.125*(1+ksi)*(1+zeta);
            dSp[1][6][k] =  0.125*(1+ksi)*(1+zeta);
            dSp[1][7][k] =  0.125*(1-ksi)*(1+zeta);
         
            // zeta derivatives of Sp
            dSp[2][0][k] = -0.125*(1-ksi)*(1-eta);
            dSp[2][1][k] = -0.125*(1+ksi)*(1-eta);
            dSp[2][2][k] = -0.125*(1+ksi)*(1+eta);
            dSp[2][3][k] = -0.125*(1-ksi)*(1+eta);
            dSp[2][4][k] =  0.125*(1-ksi)*(1-eta);
            dSp[2][5][k] =  0.125*(1+ksi)*(1-eta);
            dSp[2][6][k] =  0.125*(1+ksi)*(1+eta);
            dSp[2][7][k] =  0.125*(1-ksi)*(1+eta);
         }
      } else {
         printf("\n\n\n ERROR: Only NENp = 8 is supported for hexahedral elements.\n\n\n");
      }
     
      if (NENv == 8) {
         //Sv = Sp;
         //dSv = dSp;
      } else if (NENv == 27) {
         for (int k = 0; k < NGP; k++) {
            ksi  = GQpoint[k][0];
            eta  = GQpoint[k][1];
            zeta = GQpoint[k][2];
         
            Sv[k][0] = 0.125 * (ksi*ksi - ksi) * (eta*eta - eta) * (zeta*zeta - zeta);
            Sv[k][1] = 0.125 * (ksi*ksi + ksi) * (eta*eta - eta) * (zeta*zeta - zeta);
            Sv[k][2] = 0.125 * (ksi*ksi + ksi) * (eta*eta + eta) * (zeta*zeta - zeta);
            Sv[k][3] = 0.125 * (ksi*ksi - ksi) * (eta*eta + eta) * (zeta*zeta - zeta);
            Sv[k][4] = 0.125 * (ksi*ksi - ksi) * (eta*eta - eta) * (zeta*zeta + zeta);
            Sv[k][5] = 0.125 * (ksi*ksi + ksi) * (eta*eta - eta) * (zeta*zeta + zeta);
            Sv[k][6] = 0.125 * (ksi*ksi + ksi) * (eta*eta + eta) * (zeta*zeta + zeta);
            Sv[k][7] = 0.125 * (ksi*ksi - ksi) * (eta*eta + eta) * (zeta*zeta + zeta);
         
            Sv[k][8]  = 0.25 * (1 - ksi*ksi) * (eta*eta - eta) * (zeta*zeta - zeta);
            Sv[k][9]  = 0.25 * (ksi*ksi + ksi) * (1 - eta*eta) * (zeta*zeta - zeta);
            Sv[k][10] = 0.25 * (1 - ksi*ksi) * (eta*eta + eta) * (zeta*zeta - zeta);
            Sv[k][11] = 0.25 * (ksi*ksi - ksi) * (1 - eta*eta) * (zeta*zeta - zeta);
         
            Sv[k][12] = 0.25 * (ksi*ksi - ksi) * (eta*eta - eta) * (1 - zeta*zeta);
            Sv[k][13] = 0.25 * (ksi*ksi + ksi) * (eta*eta - eta) * (1 - zeta*zeta);
            Sv[k][14] = 0.25 * (ksi*ksi + ksi) * (eta*eta + eta) * (1 - zeta*zeta);
            Sv[k][15] = 0.25 * (ksi*ksi - ksi) * (eta*eta + eta) * (1 - zeta*zeta);
         
            Sv[k][16] = 0.25 * (1 - ksi*ksi) * (eta*eta - eta) * (zeta*zeta + zeta);
            Sv[k][17] = 0.25 * (ksi*ksi + ksi) * (1 - eta*eta) * (zeta*zeta + zeta);
            Sv[k][18] = 0.25 * (1 - ksi*ksi) * (eta*eta + eta) * (zeta*zeta + zeta);
            Sv[k][19] = 0.25 * (ksi*ksi - ksi) * (1 - eta*eta) * (zeta*zeta + zeta);
         
            Sv[k][20] = 0.5 * (1 - ksi*ksi) * (1 - eta*eta) * (zeta*zeta - zeta);
            Sv[k][21] = 0.5 * (1 - ksi*ksi) * (eta*eta - eta) * (1 - zeta*zeta);
            Sv[k][22] = 0.5 * (ksi*ksi + ksi) * (1 - eta*eta) * (1 - zeta*zeta);
            Sv[k][23] = 0.5 * (1 - ksi*ksi) * (eta*eta + eta) * (1 - zeta*zeta);
            Sv[k][24] = 0.5 * (ksi*ksi - ksi) * (1 - eta*eta) * (1 - zeta*zeta);
            Sv[k][25] = 0.5 * (1 - ksi*ksi) * (1 - eta*eta) * (zeta*zeta + zeta);

            Sv[k][26] = (1 - ksi*ksi) * (1 - eta*eta) * (1 - zeta*zeta);

            // ksi derivatives of Sv
            dSv[0][0][k] = 0.125 * (2*ksi - 1) * (eta*eta - eta) * (zeta*zeta - zeta);
            dSv[0][1][k] = 0.125 * (2*ksi + 1) * (eta*eta - eta) * (zeta*zeta - zeta);
            dSv[0][2][k] = 0.125 * (2*ksi + 1) * (eta*eta + eta) * (zeta*zeta - zeta);
            dSv[0][3][k] = 0.125 * (2*ksi - 1) * (eta*eta + eta) * (zeta*zeta - zeta);
            dSv[0][4][k] = 0.125 * (2*ksi - 1) * (eta*eta - eta) * (zeta*zeta + zeta);
            dSv[0][5][k] = 0.125 * (2*ksi + 1) * (eta*eta - eta) * (zeta*zeta + zeta);
            dSv[0][6][k] = 0.125 * (2*ksi + 1) * (eta*eta + eta) * (zeta*zeta + zeta);
            dSv[0][7][k] = 0.125 * (2*ksi - 1) * (eta*eta + eta) * (zeta*zeta + zeta);
         
            dSv[0][8][k]  = 0.25 * (- 2*ksi) * (eta*eta - eta) * (zeta*zeta - zeta);
            dSv[0][9][k]  = 0.25 * (2*ksi + 1) * (1 - eta*eta) * (zeta*zeta - zeta);
            dSv[0][10][k] = 0.25 * (- 2*ksi) * (eta*eta + eta) * (zeta*zeta - zeta);
            dSv[0][11][k] = 0.25 * (2*ksi - 1) * (1 - eta*eta) * (zeta*zeta - zeta);
         
            dSv[0][12][k] = 0.25 * (2*ksi - 1) * (eta*eta - eta) * (1 - zeta*zeta);
            dSv[0][13][k] = 0.25 * (2*ksi + 1) * (eta*eta - eta) * (1 - zeta*zeta);
            dSv[0][14][k] = 0.25 * (2*ksi + 1) * (eta*eta + eta) * (1 - zeta*zeta);
            dSv[0][15][k] = 0.25 * (2*ksi - 1) * (eta*eta + eta) * (1 - zeta*zeta);
         
            dSv[0][16][k] = 0.25 * (- 2*ksi) * (eta*eta - eta) * (zeta*zeta + zeta);
            dSv[0][17][k] = 0.25 * (2*ksi + 1) * (1 - eta*eta) * (zeta*zeta + zeta);
            dSv[0][18][k] = 0.25 * (- 2*ksi) * (eta*eta + eta) * (zeta*zeta + zeta);
            dSv[0][19][k] = 0.25 * (2*ksi - 1) * (1 - eta*eta) * (zeta*zeta + zeta);
         
            dSv[0][20][k] = 0.5 * (- 2*ksi) * (1 - eta*eta) * (zeta*zeta - zeta);
            dSv[0][21][k] = 0.5 * (- 2*ksi) * (eta*eta - eta) * (1 - zeta*zeta);
            dSv[0][22][k] = 0.5 * (2*ksi + 1) * (1 - eta*eta) * (1 - zeta*zeta);
            dSv[0][23][k] = 0.5 * (- 2*ksi) * (eta*eta + eta) * (1 - zeta*zeta);
            dSv[0][24][k] = 0.5 * (2*ksi - 1) * (1 - eta*eta) * (1 - zeta*zeta);
            dSv[0][25][k] = 0.5 * (- 2*ksi) * (1 - eta*eta) * (zeta*zeta + zeta);

            dSv[0][26][k] = (- 2*ksi) * (1 - eta*eta) * (1 - zeta*zeta);
         
         
            // eta derivatives of Sv
            dSv[1][0][k] = 0.125 * (ksi*ksi - ksi) * (2*eta - 1) * (zeta*zeta - zeta);
            dSv[1][1][k] = 0.125 * (ksi*ksi + ksi) * (2*eta - 1) * (zeta*zeta - zeta);
            dSv[1][2][k] = 0.125 * (ksi*ksi + ksi) * (2*eta + 1) * (zeta*zeta - zeta);
            dSv[1][3][k] = 0.125 * (ksi*ksi - ksi) * (2*eta + 1) * (zeta*zeta - zeta);
            dSv[1][4][k] = 0.125 * (ksi*ksi - ksi) * (2*eta - 1) * (zeta*zeta + zeta);  
            dSv[1][5][k] = 0.125 * (ksi*ksi + ksi) * (2*eta - 1) * (zeta*zeta + zeta);
            dSv[1][6][k] = 0.125 * (ksi*ksi + ksi) * (2*eta + 1) * (zeta*zeta + zeta);
            dSv[1][7][k] = 0.125 * (ksi*ksi - ksi) * (2*eta + 1) * (zeta*zeta + zeta);
         
            dSv[1][8][k]  = 0.25 * (1 - ksi*ksi) * (2*eta - 1) * (zeta*zeta - zeta);
            dSv[1][9][k]  = 0.25 * (ksi*ksi + ksi) * (- 2*eta) * (zeta*zeta - zeta);
            dSv[1][10][k] = 0.25 * (1 - ksi*ksi) * (2*eta + 1) * (zeta*zeta - zeta);
            dSv[1][11][k] = 0.25 * (ksi*ksi - ksi) * (- 2*eta) * (zeta*zeta - zeta);
         
            dSv[1][12][k] = 0.25 * (ksi*ksi - ksi) * (2*eta - 1) * (1 - zeta*zeta);
            dSv[1][13][k] = 0.25 * (ksi*ksi + ksi) * (2*eta - 1) * (1 - zeta*zeta);
            dSv[1][14][k] = 0.25 * (ksi*ksi + ksi) * (2*eta + 1) * (1 - zeta*zeta);
            dSv[1][15][k] = 0.25 * (ksi*ksi - ksi) * (2*eta + 1) * (1 - zeta*zeta);
         
            dSv[1][16][k] = 0.25 * (1 - ksi*ksi) * (2*eta - 1) * (zeta*zeta + zeta);
            dSv[1][17][k] = 0.25 * (ksi*ksi + ksi) * (- 2*eta) * (zeta*zeta + zeta);
            dSv[1][18][k] = 0.25 * (1 - ksi*ksi) * (2*eta + 1) * (zeta*zeta + zeta);
            dSv[1][19][k] = 0.25 * (ksi*ksi - ksi) * (- 2*eta) * (zeta*zeta + zeta);
         
            dSv[1][20][k] = 0.5 * (1 - ksi*ksi) * (- 2*eta) * (zeta*zeta - zeta);
            dSv[1][21][k] = 0.5 * (1 - ksi*ksi) * (2*eta - 1) * (1 - zeta*zeta);
            dSv[1][22][k] = 0.5 * (ksi*ksi + ksi) * (- 2*eta) * (1 - zeta*zeta);
            dSv[1][23][k] = 0.5 * (1 - ksi*ksi) * (2*eta + 1) * (1 - zeta*zeta);
            dSv[1][24][k] = 0.5 * (ksi*ksi - ksi) * (- 2*eta) * (1 - zeta*zeta);
            dSv[1][25][k] = 0.5 * (1 - ksi*ksi) * (- 2*eta) * (zeta*zeta + zeta);

            dSv[1][26][k] = (1 - ksi*ksi) * (- 2*eta) * (1 - zeta*zeta);
        
         
            // zeta derivatives of Sv
            dSv[2][0][k] = 0.125 * (ksi*ksi - ksi) * (eta*eta - eta) * (2*zeta - 1);
            dSv[2][1][k] = 0.125 * (ksi*ksi + ksi) * (eta*eta - eta) * (2*zeta - 1);
            dSv[2][2][k] = 0.125 * (ksi*ksi + ksi) * (eta*eta + eta) * (2*zeta - 1);
            dSv[2][3][k] = 0.125 * (ksi*ksi - ksi) * (eta*eta + eta) * (2*zeta - 1);
            dSv[2][4][k] = 0.125 * (ksi*ksi - ksi) * (eta*eta - eta) * (2*zeta + 1);
            dSv[2][5][k] = 0.125 * (ksi*ksi + ksi) * (eta*eta - eta) * (2*zeta + 1);
            dSv[2][6][k] = 0.125 * (ksi*ksi + ksi) * (eta*eta + eta) * (2*zeta + 1);
            dSv[2][7][k] = 0.125 * (ksi*ksi - ksi) * (eta*eta + eta) * (2*zeta + 1);
         
            dSv[2][8][k]  = 0.25 * (1 - ksi*ksi) * (eta*eta - eta) * (2*zeta - 1);
            dSv[2][9][k]  = 0.25 * (ksi*ksi + ksi) * (1 - eta*eta) * (2*zeta - 1);
            dSv[2][10][k] = 0.25 * (1 - ksi*ksi) * (eta*eta + eta) * (2*zeta - 1);
            dSv[2][11][k] = 0.25 * (ksi*ksi - ksi) * (1 - eta*eta) * (2*zeta - 1);
         
            dSv[2][12][k] = 0.25 * (ksi*ksi - ksi) * (eta*eta - eta) * (- 2*zeta);
            dSv[2][13][k] = 0.25 * (ksi*ksi + ksi) * (eta*eta - eta) * (- 2*zeta);
            dSv[2][14][k] = 0.25 * (ksi*ksi + ksi) * (eta*eta + eta) * (- 2*zeta);
            dSv[2][15][k] = 0.25 * (ksi*ksi - ksi) * (eta*eta + eta) * (- 2*zeta);
         
            dSv[2][16][k] = 0.25 * (1 - ksi*ksi) * (eta*eta - eta) * (2*zeta + 1);
            dSv[2][17][k] = 0.25 * (ksi*ksi + ksi) * (1 - eta*eta) * (2*zeta + 1);
            dSv[2][18][k] = 0.25 * (1 - ksi*ksi) * (eta*eta + eta) * (2*zeta + 1);
            dSv[2][19][k] = 0.25 * (ksi*ksi - ksi) * (1 - eta*eta) * (2*zeta + 1);
         
            dSv[2][20][k] = 0.5 * (1 - ksi*ksi) * (1 - eta*eta) * (2*zeta - 1);
            dSv[2][21][k] = 0.5 * (1 - ksi*ksi) * (eta*eta - eta) * (- 2*zeta);
            dSv[2][22][k] = 0.5 * (ksi*ksi + ksi) * (1 - eta*eta) * (- 2*zeta);
            dSv[2][23][k] = 0.5 * (1 - ksi*ksi) * (eta*eta + eta) * (- 2*zeta);
            dSv[2][24][k] = 0.5 * (ksi*ksi - ksi) * (1 - eta*eta) * (- 2*zeta);
            dSv[2][25][k] = 0.5 * (1 - ksi*ksi) * (1 - eta*eta) * (2*zeta + 1);

            dSv[2][26][k] = (1 - ksi*ksi) * (1 - eta*eta) * (- 2*zeta);
         }
      } else {
         printf("\n\n\n ERROR: Only NENv = 8 and 27 are supported for hexahedral elements.\n\n\n");
      }
     
   } else if (eType == 2) { // Tetrahedral element
     
      // TODO : ...
   
   }  // End of eType


   Sv_1d = new double[NGP*NENv];
   
   int count = 0;
   
   for (int k = 0; k < NGP; k++) {
      for (int i = 0; i < NENv; i++) {
         Sv_1d[count] = Sv[k][i];
         count = count + 1;
      }
   }
   
   // CONTROL
   //for (int k = 0; k < NGP; k++) {
      //for (int j = 0; j < NENv; j++) {
         //cout << k << "  "  << j << "  " << Sv[k][j] << " | " << Sv_1d[k*NENv+j] << endl;
      //}
   //}   

   /* CONTROL
   for (int i = 0; i < 3; i++){
     for (int j = 0; j < NENp; j++) {
         for (int k = 0; k < NGP; k++) {
            cout << i << "  " << j << "  "  << k << "  " << dSp[i][j][k] << endl;
         }
      }
   }
   */

}  // End of function calcShape()





//------------------------------------------------------------------------------
void calcJacob()
//------------------------------------------------------------------------------
{
   // Calculates Jacobian matrix and its determinant for each element. Shape
   // functions for corner nodes (pressure nodes) are used for Jacobian
   // calculation.
   // Also calculates and stores the derivatives of velocity shape functions
   // wrt x and y at GQ points for each element.

   int iG; 
   double **e_coord;
   double **Jacob, **invJacob;
   
   e_coord = new double*[NEC];
   
   for (int i = 0; i < NEC; i++) {
      e_coord[i] = new double[3];
   }

   Jacob = new double*[3];
   invJacob = new double*[3];
   for (int i = 0; i < 3; i++) {
      Jacob[i] = new double[3];
      invJacob[i] = new double[3];
   }
   
   detJacob = new double*[NE];
   for (int i = 0; i < NE; i++) {
      detJacob[i] = new double[NGP];
   }
   
   gDSp = new double***[NE];

   for (int i = 0; i < NE; i++) {
      gDSp[i] = new double**[NGP];
      for(int j = 0; j < NGP; j++) {
         gDSp[i][j] = new double*[NENp];     
         for(int k = 0; k < NENp; k++) {
            gDSp[i][j][k] = new double[3];
         }
      }	
   }

   gDSv = new double***[NE];

   for (int i = 0; i < NE; i++) {
      gDSv[i] = new double**[NGP];
      for(int j = 0; j < NGP; j++) {
         gDSv[i][j] = new double*[NENv];
         for(int k = 0; k < NENv; k++) {
            gDSv[i][j][k] = new double[3];
         }
      }
   }

   for (int e = 0; e < NE; e++){
      // Find e_coord, coordinates for NEC corners of element e.
      for (int i = 0; i < NEC; i++){
         iG = LtoGnode[e][i];
         e_coord[i][0] = coord[iG][0]; 
         e_coord[i][1] = coord[iG][1];
         e_coord[i][2] = coord[iG][2];
      }
   
      // For each GQ point calculate 3x3 Jacobian matrix, its inverse and its
      // determinant. Also calculate derivatives of shape functions wrt global
      // coordinates x, y & z. These are the derivatives that we'll use in
      // evaluating integrals of elemental systems.

      double sum;

      for (int k = 0; k < NGP; k++) {
         for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
               sum = 0;
               for (int m = 0; m < NENp; m++) {
                  sum = sum + dSp[i][m][k] * e_coord[m][j];
               }
               Jacob[i][j] = sum;
            }
         }
         
         invJacob[0][0] =   Jacob[1][1]*Jacob[2][2] - Jacob[2][1]*Jacob[1][2];
         invJacob[0][1] = -(Jacob[0][1]*Jacob[2][2] - Jacob[0][2]*Jacob[2][1]);
         invJacob[0][2] =   Jacob[0][1]*Jacob[1][2] - Jacob[1][1]*Jacob[0][2];
         invJacob[1][0] = -(Jacob[1][0]*Jacob[2][2] - Jacob[1][2]*Jacob[2][0]);
         invJacob[1][1] =   Jacob[2][2]*Jacob[0][0] - Jacob[2][0]*Jacob[0][2];
         invJacob[1][2] = -(Jacob[1][2]*Jacob[0][0] - Jacob[1][0]*Jacob[0][2]);
         invJacob[2][0] =   Jacob[1][0]*Jacob[2][1] - Jacob[2][0]*Jacob[1][1];
         invJacob[2][1] = -(Jacob[2][1]*Jacob[0][0] - Jacob[2][0]*Jacob[0][1]);
         invJacob[2][2] =   Jacob[1][1]*Jacob[0][0] - Jacob[1][0]*Jacob[0][1];

         detJacob[e][k] = Jacob[0][0]*(Jacob[1][1]*Jacob[2][2] - Jacob[2][1]*Jacob[1][2]) +
                          Jacob[0][1]*(Jacob[1][2]*Jacob[2][0] - Jacob[1][0]*Jacob[2][2]) +
                          Jacob[0][2]*(Jacob[1][0]*Jacob[2][1] - Jacob[1][1]*Jacob[2][0]);
         
         for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
               invJacob[i][j] = invJacob[i][j] / detJacob[e][k];
            }    
         }
         
         for (int i = 0; i < 3; i++){
            for (int j = 0; j < NENp; j++) {
               sum = 0;
               for (int m = 0; m < 3; m++) { 
                  sum = sum + invJacob[i][m] * dSp[m][j][k];
               }
               gDSp[e][k][j][i] = sum;
            }
         }

         for (int i = 0; i < 3; i++){
            for (int j = 0; j < NENv; j++) {
               sum = 0;
               for (int m = 0; m < 3; m++) { 
                  sum = sum + invJacob[i][m] * dSv[m][j][k];
               }
               gDSv[e][k][j][i] = sum;
            }
         }

      /* CONTROL
      cout << e << "  " << k << "  " << detJacob[e][k] << endl;

      for (int i = 0; i < 3; i++){
         for (int j = 0; j < 3; j++){
            cout << e << "  " << i << "  " << j << "  " << invJacob[i][j] << endl;
         }
      }
      */

      }   // End of GQ loop

   }   // End of element loop

   
   /*  CONTROL
   for (int e = 0; e < 1; e++){
      for (int i = 0; i < NGP; i++){
         for (int j = 0; j < NENv; j++) {
            for (int k = 0; k < 3; k++) {
               cout << e << "  " << i << "  " << j << "  "  << k << "  " << gDSv[e][i][j][k] << endl;
            }
         }
      }
   }
   */
   
   GQfactor_1d = new double[NE*NGP];
   
   int count = 0;
   
   for (int e = 0; e < NE; e++) {
      for (int k = 0; k < NGP; k++) {
         GQfactor_1d[count] = detJacob[e][k] * GQweight[k];
         count = count + 1;
      }
   }
   
   // CONTROL
   //for (int e = 0; e < NE; e++) {
      //for (int k = 0; k < NGP; k++) {
         //cout << e << "  " << k << "  " << detJacob[e][k] << "  " << GQweight[k] << " | " << GQfactor_1d[e*NGP + k] << endl;
      //}
   //}
   
   
   gDSv_1d = new double[NE*NGP*NENv*3];
   
   count = 0;
   
   for (int e = 0; e < NE; e++) {
      for (int k = 0; k < NGP; k++) {
         for (int j = 0; j < NENv; j++) {
            for (int i = 0; i < 3; i++) {
               gDSv_1d[count] = gDSv[e][k][j][i];
               count = count + 1;
            }
         }
      }
   }   
   
   // CONTROL
   //cout << endl;
   //cout << " GDSV " << endl;
   //for (int e = 0; e < 2; e++){
      //for (int i = 0; i < NGP; i++){
         //for (int j = 0; j < NENv; j++) {
            //for (int k = 0; k < 3; k++) {
               //cout << e << "  " << i << "  " << j << "  " << k << "  " << gDSv[e][i][j][k] << " | " << 
                  //gDSv_1d[e*NGP*NENv*3+i*NENv*3+j*3+k] << endl;
            //}
         //}
      //}
      //cout << endl;
   //}   

   // Deallocate unnecessary variables

   for (int i = 0; i < 3; i++) {
      delete[] Jacob[i];
   }
   delete[] Jacob;

   for (int i = 0; i < 3; i++) {
      delete[] invJacob[i];
   }
   delete[] invJacob;
   
   for (int i = 0; i < NENp; i++) {
      delete[] e_coord[i];
   }
   delete[] e_coord;

                                                                                                                                                     // TODO : Deallocate dSv and dSv.
} // End of function calcJacob()





//========================================================================
void initializeAndAllocate()

{
   // Do the necessary memory allocations. Apply the initial condition or read
   // the restart file.

   Uk      = new double[3*NN];     // x, y and z velocity components of time step k+1.
   Uk_prev = new double[3*NN];     // x, y and z velocity components of time step k.
   Uk_dir  = new double[NN];       // Keeps each component of velocity.

   Pk          = new double[NNp];         // Pressure of time step n.
   Pk_prev     = new double[NNp];         // U_i+1^n+1 of the reference paper.
   Pk_prevprev = new double[NNp];         // p_i+1^n+1 of the reference paper.
   Pdiff       = new double[NNp];         // Pdot_i+1^n+1 of the reference paper.

   R1 = new double[NN];               // RHS vector of intermediate velocity calculation.
   R2 = new double[NNp];              // RHS vector of pressure calculation.

   // Initialize all these variables to zero
   for (int i = 0; i < 3*NN; i++) {
      Uk[i]            = 0.0;
      Uk_prev[i]       = 0.0;
   }

   for (int i = 0; i < NN; i++) {
      R1[i]     = 0.0;
      Uk_dir[i] = 0.0;
   }

   for (int i = 0; i < NNp; i++) {
      Pk[i]          = 0.0;
      Pk_prev[i]     = 0.0;
      Pk_prevprev[i] = 0.0;
      Pdiff[i]       = 0.0;
      R2[i]          = 0.0;
   }


   // Read the restart file if isRestart is equal to 1. If not, apply the
   // specified BCs.
   if (isRestart == 1) {
      readRestartFile();
   } else {
      applyBC_initial();
   }

   createTecplot();
   
   
   // Initialize discrete time level and time.
   timeN = 0;
   timeT = t_ini;

}  // End of function initializeAndAllocate()





//========================================================================
void timeLoop()

{
   // Main time loop of the solution.

   double Start, wallClockTime;
   
   double oneOverdt = 1.0000000000000000 / dt;
   double dummyAcc;


   // Initialize the solution using the specified initial condition and do
   // memory allocations.
   initializeAndAllocate();

   // Calculate certain matrices and their inverses only once before the time loop.
   Start = getHighResolutionTime(1, 1.0);
   step0();
   wallClockTime = getHighResolutionTime(2, Start);
   printf("step0()                took  %8.3f seconds.\n", wallClockTime);

   waitForUser("Enter a character... ");

   #ifdef USECUDA
      initializeAndAllocateGPU();
   #endif

   cout << endl;
   cout << " NN = " << NN << endl;
   cout << " NNp = " << NNp << endl;
   cout << " sparseM_NNZ = " << sparseM_NNZ << endl;
   cout << " sparseG_NNZ = " << sparseG_NNZ << endl;
   #ifndef USECUDA
      cout << " NNZ of upper part of Z = " << sparseZrowStarts[NNp] << endl;
   #endif
   
   printf("\n\nMonitoring node is %d, with coordinates [%f, %f, %f]\n\n\n",
           monPoint, coord[monPoint][0], coord[monPoint][1], coord[monPoint][2]);

   printf("Time step  Time       u_monitor     v_monitor     w_monitor     p_monitor     TimeSpend      maxAcc \n");
   printf("----------------------------------------------------------------------------------------------------\n");

   while (timeT < t_final) {  // Time loop
      StartCurrentTimeStep = getHighResolutionTime(1, 1.0);   
      timeN = timeN + 1;
      timeT = timeT + dt;

      // Calculate intermediate velocity.
      Start = getHighResolutionTime(1, 1.0);
      #ifdef USECUDA
         step1GPU();
      #else   
         step1();
      #endif
      wallClockTime = getHighResolutionTime(2, Start);
      if (PRINT_TIMES) printf("step1() took %6.3f seconds.\n", wallClockTime);

      waitForUser("Enter a character... ");

      // Calculate pressure of the new time step
      Start = getHighResolutionTime(1, 1.0);
      #ifdef USECUDA
         step2GPU();
         cudaThreadSynchronize();
      #else
         step2();
      #endif
      wallClockTime = getHighResolutionTime(2, Start);
      if (PRINT_TIMES) printf("step2() took %6.3f seconds.\n", wallClockTime);

      waitForUser("Enter a character... ");
      
      
      Start = getHighResolutionTime(1, 1.0);     
      // Check if solution reachs steady state
      #ifdef USECUDA
         checkAccConvergence = checkConvergenceInTimeGPU();
         cudaThreadSynchronize();
      #else    
         checkAccConvergence = 1;
         
         maxAcc = 0.00000;
         for (int i = 0; i < 3*NN; i++) {
            dummyAcc = (Uk[i] - Uk_prev[i]) * oneOverdt;
            if (abs(dummyAcc) > maxAcc) {
               maxAcc = dummyAcc;
            }
         }
         
         if (maxAcc > convergenceCriteria) {
            checkAccConvergence = 0;
         }         
      #endif
      
      wallClockTime = getHighResolutionTime(2, Start);
      if (PRINT_TIMES) printf("checkConvergenceSteadyState() took %6.3f seconds.\n", wallClockTime);
     
     
      // Get ready for the next time step
      #ifdef USECUDA
         cudaStatus = cudaMemcpy(Uk_prev_d, Uk_d, 3*NN * sizeof(double), cudaMemcpyDeviceToDevice);   if(cudaStatus != cudaSuccess) { printf("Error52: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
         cudaStatus = cudaMemcpy(Pk_prevprev_d, Pk_prev_d, NNp  * sizeof(double), cudaMemcpyDeviceToDevice);   if(cudaStatus != cudaSuccess) { printf("Error53: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
         cudaStatus = cudaMemcpy(Pk_prev_d,     Pk_d,      NNp  * sizeof(double), cudaMemcpyDeviceToDevice);   if(cudaStatus != cudaSuccess) { printf("Error53: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
         cudaThreadSynchronize();
      #else
         for (int i = 0; i < 3*NN; i++) {
            Uk_prev[i] = Uk[i];
         }

         for (int i = 0; i < NNp; i++) {
            Pk_prevprev[i] = Pk_prev[i];
            Pk_prev[i]       = Pk[i];
         }
      #endif
      
      
      wallClockTimeCurrentTimeStep = getHighResolutionTime(2, StartCurrentTimeStep);      
      
      // Print monitor point data                                                                                                                    // TODO: If convergence is not achieved iter is written as iterMax+1, which is not correct.
      #ifdef USECUDA
         printMonitorDataGPU();
         cudaThreadSynchronize();
      #else
         printf("%6d  %10.5f  %12.5f  %12.5f  %12.5f  %12.5f %12.5f %12.5f\n",
                timeN, timeT, Uk[monPoint],
                Uk[NN+monPoint], Uk[2*NN+monPoint], Pk[monPoint], wallClockTimeCurrentTimeStep, maxAcc);
      #endif      
      
      
      if (checkAccConvergence == 1) {
         #ifdef USECUDA
            cudaStatus = cudaMemcpy(Uk, Uk_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error54: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
            cudaStatus = cudaMemcpy(Pk, Pk_d, NNp  * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error55: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
            cudaThreadSynchronize();
         #endif
         createTecplot();
         break;    
      }
      else {
            if (timeN % 1000 == 0 || abs(timeT - t_final) < 1e-10) {
               #ifdef USECUDA
                  cudaStatus = cudaMemcpy(Uk, Uk_d, 3*NN * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error54: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
                  cudaStatus = cudaMemcpy(Pk, Pk_d, NNp  * sizeof(double), cudaMemcpyDeviceToHost);   if(cudaStatus != cudaSuccess) { printf("Error55: %s\n", cudaGetErrorString(cudaStatus)); cin >> dummyUserInput; }
                  cudaThreadSynchronize();
               #endif
               createTecplot();
            }
      }
      
      
   }  // End of while loop for time

}  // End of function timeLoop()





//========================================================================
void step0()

{
   // Calculates [M], [G] and [K] only once, before the time loop.

   int nnzM = sparseM_NNZ / 3;
   int nnzM2 = 2 * nnzM;
   int nnzM3 = 3 * nnzM;

   int nnzG = sparseG_NNZ / 3;
   int nnzG2 = 2 * nnzG;

   double **Me_11, **Ke_11, **Ge_1, **Ge_2, **Ge_3;
   double GQfactor;
   double inverseDensity;
   double inverseDt;
   
   inverseDensity = 1.0 / density;
   inverseDt = 1.0 / dt;

   for (int i = 0; i < sparseM_NNZ/3; i++){
      sparseMvalue[i] = 0.0;
   }

   for (int i = 0; i < sparseM_NNZ/3; i++){
	   sparseKvalue[i] = 0.0;
   }

   for (int i = 0; i < sparseG_NNZ/3; i++){
      sparseG1value[i] = 0.0;
      sparseG2value[i] = 0.0;
      sparseG3value[i] = 0.0;
   }

   Me_11 = new double*[NENv];
   Ke_11 = new double*[NENv];
   for (int i = 0; i < NENv; i++) {
      Me_11[i] = new double[NENv];
      Ke_11[i] = new double[NENv];
   }

   Ge_1 = new double*[NENv];
   Ge_2 = new double*[NENv];
   Ge_3 = new double*[NENv];
   for (int i = 0; i < NENv; i++) {
      Ge_1[i] = new double[NENp];
      Ge_2[i] = new double[NENp];
      Ge_3[i] = new double[NENp];
   }

   // Calculate Me, Ke and Ge, and assemble them into M, K and G
   for (int e = 0; e < NE; e++) {

      for (int i = 0; i < NENv; i++) {
         for (int j = 0; j < NENv; j++) {
            Me_11[i][j] = 0.0;
            Ke_11[i][j] = 0.0;
         }
         for (int j = 0; j < NENp; j++) {
            Ge_1[i][j] = 0.0;
            Ge_2[i][j] = 0.0;
            Ge_3[i][j] = 0.0;
         }
      }

      for (int k = 0; k < NGP; k++) {   // Gauss Quadrature loop
         GQfactor = detJacob[e][k] * GQweight[k];
       
         for (int i = 0; i < NENv; i++) {
            for (int j = 0; j < NENv; j++) {
               Me_11[i][j] = Me_11[i][j] + inverseDt * Sv[k][i] * Sv[k][j] * GQfactor;

               Ke_11[i][j] = Ke_11[i][j] + viscosity * (gDSv[e][k][i][0] * gDSv[e][k][j][0] +
                                                        gDSv[e][k][i][1] * gDSv[e][k][j][1] +
                                                        gDSv[e][k][i][2] * gDSv[e][k][j][2]) * GQfactor;
            }
         }

         for (int i = 0; i < NENv; i++) {
            for (int j = 0; j < NENp; j++) {
               Ge_1[i][j] = Ge_1[i][j] - inverseDensity * Sp[k][j] * gDSv[e][k][i][0] * GQfactor;
               Ge_2[i][j] = Ge_2[i][j] - inverseDensity * Sp[k][j] * gDSv[e][k][i][1] * GQfactor;
               Ge_3[i][j] = Ge_3[i][j] - inverseDensity * Sp[k][j] * gDSv[e][k][i][2] * GQfactor;
            }
         }
       
      } // GQ loop
     
     
      // Assemble Me and Ke into sparse M and K.
      for (int i = 0; i < NENv; i++) {
         for (int j = 0; j < NENv; j++) {
            sparseMvalue[sparseMapM[e][i][j]] += Me_11[i][j];   // Assemble upper left sub-matrix of M
            sparseKvalue[sparseMapM[e][i][j]] += Ke_11[i][j];   // Assemble upper left sub-matrix of K
         }
      }
     
      // Assemble Ge into sparse G.
      for (int i = 0; i < NENv; i++) {
         for (int j = 0; j < NENp; j++) {
            sparseG1value[sparseMapG[e][i][j]] += Ge_1[i][j];   // Assemble upper part of G
            sparseG2value[sparseMapG[e][i][j]] += Ge_2[i][j];   // Assemble middle of G
            sparseG3value[sparseMapG[e][i][j]] += Ge_3[i][j];   // Assemble lower part of G
         }
      }
     
   }  // Element loop

   

   for (int i = 0; i < NENv; i++) {
      delete[] Me_11[i];
      delete[] Ke_11[i];
      delete[] Ge_1[i];
      delete[] Ge_2[i];
      delete[] Ge_3[i];
   }
   delete[] Me_11;
   delete[] Ke_11;
   delete[] Ge_1;
   delete[] Ge_2;
   delete[] Ge_3; 

   //  CONTROL
//   for (int i = 0; i < sparseM_NNZ/3; i++){
//      cout << i+1 << "  " << sparseMrow[i]+1 << "  " << sparseMcol[i]+1 << "  " << sparseMvalue[i] << endl;
//   }
//   for (int i = 0; i < sparseM_NNZ/3; i++){
//      cout << i+1 << "  " << sparseKrow[i]+1 << "  " << sparseKcol[i]+1 << "  " << sparseKvalue[i] << endl;
//   }
//   for (int i = 0; i < sparseG_NNZ/3; i++){
//      cout << i+1 << "  " << sparseGrow[i]+1 << "  " << sparseGcol[i]+1 << "  " << sparseG1value[i] << "  " << sparseG2value[i] << "  " << sparseG3value[i] << endl;
//   }
   
   waitForUser("OK000. Enter a character... ");
   
   #ifdef USECUDA
      calculateZGPU();
   #else
      calculateZ();
   #endif   
   


   // Apply pressure BCs to [Z]
   applyBC_Step2(1);      
      
}  // End of function step0()





//========================================================================
void calculateZ()
//========================================================================
{
   
   // Calculates [Z] (K_hat at the reference paper).

   double *Ze_11, *ZeUpper_11;
   double GQfactor;
   int counter;
   
   for (int i = 0; i < sparseZ_NNZ; i++){
      sparseZvalue[i] = 0.0;
   }

   Ze_11  = new double[NENp*NENp];
   ZeUpper_11  = new double[NENp*NENp];

   // Calculate Ze assemble them into Z 
   for (int e = 0; e < NE; e++) {

      for (int i = 0; i < NENp*NENp; i++) {
         Ze_11[i]  = 0.0;
         ZeUpper_11[i]  = 0.0;
      }
      
      for (int k = 0; k < NGP; k++) {   // Gauss Quadrature loop
         GQfactor = detJacob[e][k] * GQweight[k];
         counter = 0;
         for (int i = 0; i < NENp; i++) {
            for (int j = 0; j < NENp; j++) {
               Ze_11[counter] = Ze_11[counter] - (gDSp[e][k][i][0] * gDSp[e][k][j][0] +
                                                  gDSp[e][k][i][1] * gDSp[e][k][j][1] +
                                                  gDSp[e][k][i][2] * gDSp[e][k][j][2]) * GQfactor;
               counter = counter + 1;
            }
         }
       
      } // GQ loop
      
      // Assemble Ze into sparse Z.
      for (int i = 0; i < sparseMapZupperCount[e]; i++) {
         ZeUpper_11[i] = Ze_11[sparseMapZupper[e][i]];      // Only the elemental values (Ze) at upper triangular part of the 
                                                            // Z should be added to the sparse Z.
         sparseZvalue[sparseMapZ[e][i]] += ZeUpper_11[i];   // Assemble Z         
      }
     
   }  // Element loop

   delete[] Ze_11;
   delete[] ZeUpper_11;
   
   
   ////CONTROL
   //cout << "Z MATRIX" << endl;   
   //cout << endl;
   //cout << "z_nnz" << sparseZrowStarts[NNp]  << "   "  << sparseZ_NNZ << endl;
   //for (int i = 0; i < NNp; i++) {
      //cout << "sparseZrowStarts[" << i << "] = " << sparseZrowStarts[i] << endl;
      //for (int j = sparseZrowStarts[i]; j < sparseZrowStarts[i+1]; j++) {
         //cout << sparseZvalue[j] << "  " ;
      //}
      //cout << endl;
   //}
   //cout << "\n\n\n\n";
   
   //cout << "Z MATRIX" << endl;
   //int check = 0;
   //double ZEROZERO = 0.00000000; 
   //std::cout.precision(3);
   //for (int i = 0; i < NNp; i++) {
      //for (int j = 0; j < NNp; j++) {
         //check = 0;
         //for (int k = sparseZrowStarts[i]; k < sparseZrowStarts[i+1]; k++) {
            //if (j == sparseZcol[k]) {
               //check = 1;
               //cout << scientific << sparseZvalue[j] << " " ;
               //break;
            //}
         //}
         
         //if (check == 0) {
            //cout << scientific << ZEROZERO << " " ;
         //}    
      
      //}
      
      //cout << endl;
      
   //}

 
}  // End of function calculateZ()





//========================================================================
void calculateZGPU()
//========================================================================
{
   
   // Calculates [Z] (K_hat at the reference paper).
   // Z is a symmetric matrix but
   // CUSP needs full matrix so there is a fuction called calculateZGPU.

   double **Ze_11;
   double GQfactor;

   for (int i = 0; i < sparseZ_NNZ; i++){
      sparseZvalue[i] = 0.0;
   }

   Ze_11 = new double*[NENp];
   for (int i = 0; i < NENp; i++) {
      Ze_11[i] = new double[NENp];
   }

   // Calculate Ze assemble them into Z 
   for (int e = 0; e < NE; e++) {

      for (int i = 0; i < NENp; i++) {
         for (int j = 0; j < NENp; j++) {
            Ze_11[i][j] = 0.0;
         }
      }
      
      for (int k = 0; k < NGP; k++) {   // Gauss Quadrature loop
         GQfactor = detJacob[e][k] * GQweight[k];
         
         for (int i = 0; i < NENp; i++) {
            for (int j = 0; j < NENp; j++) {
               Ze_11[i][j] = Ze_11[i][j] - (gDSp[e][k][i][0] * gDSp[e][k][j][0] +
                                            gDSp[e][k][i][1] * gDSp[e][k][j][1] +
                                            gDSp[e][k][i][2] * gDSp[e][k][j][2]) * GQfactor;
            }
         }
       
      } // GQ loop
      
      // Assemble Ze into sparse Z.
      for (int i = 0; i < NENp; i++) {
         for (int j = 0; j < NENp; j++) {
            sparseZvalue[sparseMapZGPU[e][i][j]] += Ze_11[i][j];   // Assemble Z
         }
      }
     
   }  // Element loop

   for (int i = 0; i < NENp; i++) {
      delete[] Ze_11[i];
   }
   delete[] Ze_11;
   
   //CONTROL
   // cout << "Z MATRIX" << endl;   
   // cout << endl;
   // cout << "z_nnz" << sparseZrowStarts[NNp]  << "   "  << sparseZ_NNZ << endl;
   // for (int i = 0; i < NNp; i++) {
      // cout << "sparseZrowStarts[" << i << "] = " << sparseZrowStarts[i] << endl;
      // for (int j = sparseZrowStarts[i]; j < sparseZrowStarts[i+1]; j++) {
         // cout << sparseZvalue[j] << "  " ;
      // }
      // cout << endl;
   // }
   // cout << "\n\n\n\n";
   
   //cout << "Z MATRIX" << endl;
   //int check = 0;
   //double ZEROZERO = 0.00000000; 
   //std::cout.precision(3);
   //for (int i = 0; i < NNp; i++) {
      //for (int j = 0; j < NNp; j++) {
         //check = 0;
         //for (int k = sparseZrowStarts[i]; k < sparseZrowStarts[i+1]; k++) {
            //if (j == sparseZcol[k]) {
               //check = 1;
               //cout << scientific << sparseZvalue[j] << " " ;
               //break;
            //}
         //}
         
         //if (check == 0) {
            //cout << scientific << ZEROZERO << " " ;
         //}    
      
      //}
      
      //cout << endl;
      
   //}

 
}  // End of function calculateZGPU()





//========================================================================
void calculateMatrixA()

{
   // Calculate Ae and assemble into A. This is called only for the first
   // iteration of each time step.

   int nnzM = sparseM_NNZ / 3;
   int nnzM2 = 2 * nnzM;
   
   double *u0_nodal, *v0_nodal, *w0_nodal;
   double u0, v0, w0;
   double Du0, Dv0, Dw0;
   
   double **Ae_11;
   double GQfactor;
   
   int offsetElements = 0;   
   
   // Calculate Ae and assemble it into A   
   for (int color = 0; color < nActiveColors; color++) {
      
      //cout << endl;
      //cout << "offset_" << NmeshColors[color] << " = " << offsetElements << endl;
      #pragma omp parallel private(Ae_11, u0, v0, w0, Du0, Dv0, Dw0, u0_nodal, v0_nodal, w0_nodal, GQfactor) shared(NmeshColors, Sv, NENv, NGP, offsetElements)
      {   
         
         u0_nodal = new double[NENv];
         v0_nodal = new double[NENv];
         w0_nodal = new double[NENv];
         
         Ae_11 = new double*[NENv];
         for (int i = 0; i < NENv; i++) {
            Ae_11[i] = new double[NENv];
         }                
         
         #pragma omp for 
            for (int eCount = 0; eCount < NmeshColors[color]; eCount++) {
               
               int e = elementsOfColor[offsetElements + eCount]; // Element that particular thread works on 
               
               for (int i = 0; i < NENv; i++) {
                  for (int j = 0; j < NENv; j++) {
                     Ae_11[i][j] = 0.0;
                  }
               }         
               
               // Extract elemental u, v and w velocity values from the global solution
               // solution array of the previous iteration.
               int iG;
               for (int i = 0; i<NENv; i++) {
                  iG = LtoGvel[e][i];
                  u0_nodal[i] = Uk_prev[iG];
           
                  iG = LtoGvel[e][i + NENv];
                  v0_nodal[i] = Uk_prev[iG];
               
                  iG = LtoGvel[e][i + 2*NENv];
                  w0_nodal[i] = Uk_prev[iG];
               }

               for (int k = 0; k < NGP; k++) {   // Gauss Quadrature loop
                  //GQfactor = GQfactor_1d[e*NGP+k];
                  GQfactor = detJacob[e][k] * GQweight[k];
            
                  // Above calculated u0 and v0 values are at the nodes. However in GQ
                  // integration we need them at GQ points. Let's calculate them using
                  // interpolation based on shape functions.
                  u0 = 0.0;
                  v0 = 0.0;
                  w0 = 0.0;
                  Du0 = 0.0;
                  Dv0 = 0.0;
                  Dw0 = 0.0;                  
                  
                  for (int i = 0; i<NENv; i++) {
                     u0 = u0 + Sv[k][i] * u0_nodal[i];
                     v0 = v0 + Sv[k][i] * v0_nodal[i];
                     w0 = w0 + Sv[k][i] * w0_nodal[i];
                     Du0 = Du0 + dSv[0][i][k] * u0_nodal[i];
                     Dv0 = Dv0 + dSv[1][i][k] * v0_nodal[i];
                     Dw0 = Dw0 + dSv[2][i][k] * w0_nodal[i];                     
                  }
                
                  for (int i = 0; i < NENv; i++) {
                     for (int j = 0; j < NENv; j++) {
                        Ae_11[i][j] = Ae_11[i][j] + (u0 * gDSv[e][k][j][0] + v0 * gDSv[e][k][j][1] + w0 * gDSv[e][k][j][2]) * Sv[k][i] * GQfactor 
                                                  + 0.0 * (Du0 + Dv0 + Dw0) * Sv[k][i] * Sv[k][j] * GQfactor;
                     }
                  }       
               } // GQ loop
               
               // Assemble Ae into sparse A.
               for (int i = 0; i < NENv; i++) {
                  for (int j = 0; j < NENv; j++) {
                     sparseAvalue[sparseMapM[e][i][j]] += Ae_11[i][j];   // Assemble upper left sub-matrix of A
                  }
               }               
              
            } // End of element loop, end of #pragma for
            
         for (int i = 0; i < NENv; i++) {
            delete[] Ae_11[i];
         }
         delete[] Ae_11; 
         
         delete[] u0_nodal;
         delete[] v0_nodal;
         delete[] w0_nodal;                   
            
      } // End of #pragma parallel
                                                                 
      offsetElements += NmeshColors[color];
      
   }

   //  CONTROL
   //for (int i = 0; i < sparseM_NNZ/3; i++){
   //   cout << i+1 << "  " << sparseMrow[i]+1 << "  " << sparseMcol[i]+1 << "  " << sparseAvalue[i] << endl;
   //}
   
} // End of function calculateMatrixA()





//========================================================================
void step1()

{
   // Executes step 1 of the method to determine the intermediate velocity.
   
   double Start, wallClockTime;

   Start = getHighResolutionTime(1, 1.0);
   
   // Calculate the LHS matrix of step 1.
   // A = [1/dt*M] + [vK] + [A(U)]

   int NNZM = sparseM_NNZ / 3;
   int NNZG = sparseG_NNZ / 3;
   
   for (int i = 0; i < NNZM; i++){
      sparseAvalue[i] = sparseMvalue[i] + sparseKvalue[i];
   }   
   
   // Calculate B(stiffness matrix of advective term) and adds to LHS system matrix
   calculateMatrixA();   
   
   // Modify sparseA for velocity BCs
   applyBC_Step1(1);   
   
   wallClockTime = getHighResolutionTime(2, Start);

   if (PRINT_TIMES) printf("calculate LHS system matrix() took %6.3f seconds.\n", wallClockTime);
   
   
   // Calculate the RHS vector and the LHS matrix of step 1.
   // R1 = [1/dt*M]*U^k - G*(2*P^k-P^(k-1))
   
   // Calculate Pdiff = 2*Pprev-Pprevprev
   double *dummyPdiff;
   dummyPdiff = new double[NNp];      
              
   for (int i = 0; i < NNp; i++) {
      dummyPdiff[i] = 2*Pk_prev[i] - Pk_prevprev[i];
   }


   char transa, matdescra[6];
   double alpha, beta;
   int m = NN;
   int k = NNp;

   transa = 'n';

   matdescra[0] = 'g';
   matdescra[1] = 'u';
   matdescra[2] = 'n';
   matdescra[3] = 'c';

   double *Uk_prev1, *Uk_prev2, *Uk_prev3;
   
   Uk_prev1 = new double[NN];
   Uk_prev2 = new double[NN];
   Uk_prev3 = new double[NN];
   
   for (int i = 0; i < NN; i++) {
      Uk_prev1[i] = Uk_prev[i];
      Uk_prev2[i] = Uk_prev[i + NN];
      Uk_prev3[i] = Uk_prev[i + 2*NN];
   }

   for (direction = 0; direction < 3; direction++) {

      Start = getHighResolutionTime(1, 1.0); 
   
      alpha = 1.000000000000000;
      beta = 0.000000000000000;
      
      switch (direction) {
         case 0:
            mkl_dcsrmv(&transa, &m, &m, &alpha, matdescra, sparseMvalue, sparseMcol, sparseMrowStarts, sparseMrowStartsMod, Uk_prev1, &beta, R1);   // This contributes to (M * Uk_prev)  part of R1
            break;
         case 1:
            mkl_dcsrmv(&transa, &m, &m, &alpha, matdescra, sparseMvalue, sparseMcol, sparseMrowStarts, sparseMrowStartsMod, Uk_prev2, &beta, R1);   // This contributes to (M * Uk_prev)  part of R2
            break;
         case 2:
            mkl_dcsrmv(&transa, &m, &m, &alpha, matdescra, sparseMvalue, sparseMcol, sparseMrowStarts, sparseMrowStartsMod, Uk_prev3, &beta, R1);   // This contributes to (M * Uk_prev)  part of R3
            break;
      }
      
      
      alpha = -1.000000000000000;
      beta = 1.000000000000000;
      
      switch (direction) {
         case 0:
            mkl_dcsrmv(&transa, &m, &k, &alpha, matdescra, sparseG1value, sparseGcol, sparseGrowStarts, sparseGrowStartsMod, dummyPdiff, &beta, R1);   // This contributes to (G * Pdiff)  part of R1
            break;
         case 1:
            mkl_dcsrmv(&transa, &m, &k, &alpha, matdescra, sparseG2value, sparseGcol, sparseGrowStarts, sparseGrowStartsMod, dummyPdiff, &beta, R1);   // This contributes to (G * Pdiff)  part of R2
            break;
         case 2:
            mkl_dcsrmv(&transa, &m, &k, &alpha, matdescra, sparseG3value, sparseGcol, sparseGrowStarts, sparseGrowStartsMod, dummyPdiff, &beta, R1);   // This contributes to (G * Pdiff)  part of R3
            break;
      }    
            
      // Modify R1 for velocity BCs
      applyBC_Step1(2);
      
      wallClockTime = getHighResolutionTime(2, Start);
      if (PRINT_TIMES) printf("step1 RHS_%d() took %6.3f seconds.\n", direction, wallClockTime);
         
         
      Start = getHighResolutionTime(1, 1.0);
      
      // Paralution BiCGStab Solver
      paralution_BiCGStab_solver();
      
//CONTROL
//for (int i = 90; i < 100; i++) {
  //cout << i << "   " << Uk_dir[i] << endl;
//}
      
      int shiftDir = direction*NN;
      
      for (int i = 0; i < NN; i++) {
         Uk[i+shiftDir] = Uk_dir[i];
      }  
      
      wallClockTime = getHighResolutionTime(2, Start);
      if (PRINT_TIMES) printf("Paralution_BiCGStab_solver_%d() took %6.3f seconds.\n", direction, wallClockTime);       
      
   }

   // CONTROL
   //for (int i = 0; i < NN; i++) {
   //   cout << i << "   " << R11[i] << endl;
   //}
   

   
   ////CONTROL
   //cout << endl;
   //for (int i = 0; i < NNp; i++) {
      //cout << "Pk_dummy[" << i << "] = " << Pk_dummy[i] << endl;
   //}

   // CONTROL
   //for (int i = 0; i < 3*NN; i++) {
   //   cout << i << "   " << R1[i] << endl;
   //}

   // CONTROL
   //for (int i = 0; i < 3*NN; i++) {
   //   cout << i << "   " << R1[i] << endl;
   //}

   delete[] Uk_prev1;
   delete[] Uk_prev2;
   delete[] Uk_prev3;
   
   delete[] dummyPdiff;
   
   
//// CONTROL A
//checkAFile.open(string("checkAFile.dat").c_str(), ios::out);

//checkAFile << "A Matrix" << endl;
//checkAFile << sparseM_NNZ << endl;
//checkAFile.precision(16);
//for (int i = 0; i < 3*NN ; i++){
//for (int j = sparseArowStarts[i]; j < sparseArowStarts[i+1]; j++) {
//checkAFile << i << " " << sparseAcol[j] << " " << sparseAvalue[j] << endl;
//}
//}
//checkAFile << 3*NN << endl;
//for (int i = 0; i < 3*NN; i++) {
//checkAFile << R1[i] << endl;
//}

//checkAFile.close();
   
}  // End of function step1()





//========================================================================
void step2()

{
   // Executes step 2 of the method to determine pressure of the new time step.

   // Calculate the RHS vector of step 2.
   // R2 = -1/dt(Gt * Uk)
   
   char transa, matdescra[6];
   double alpha, beta;
   int m = NN;
   int k = NNp;

   matdescra[0] = 'g';
   matdescra[1] = 'u';
   matdescra[2] = 'n';  
   matdescra[3] = 'c';

   alpha = -1.0/dt; // -1/dt
   transa = 't';    // Multiply using Gt, not G
      
   double *Uk1, *Uk2, *Uk3;
   
   Uk1 = new double[NN];
   Uk2 = new double[NN];
   Uk3 = new double[NN];
   
   for (int i = 0; i < NN; i++) {
      Uk1[i] = Uk[i];
      Uk2[i] = Uk[i + NN];
      Uk3[i] = Uk[i + 2*NN];
   }

   beta = 0.0;
   mkl_dcsrmv(&transa, &m, &k, &alpha, matdescra, sparseG1value, sparseGcol, sparseGrowStarts, sparseGrowStartsMod, Uk1, &beta, R2);   // This contributes to (Gt * u) which is R2
   beta = 1.0;   // Add the results of the following Matrix-vector multiplication to R2
   mkl_dcsrmv(&transa, &m, &k, &alpha, matdescra, sparseG2value, sparseGcol, sparseGrowStarts, sparseGrowStartsMod, Uk2, &beta, R2);   // This contributes to (Gt * v) which is R2
   mkl_dcsrmv(&transa, &m, &k, &alpha, matdescra, sparseG3value, sparseGcol, sparseGrowStarts, sparseGrowStartsMod, Uk3, &beta, R2);   // This contributes to (Gt * w) which is R2

   
   // Apply BCs for step2. Modify R2 for pressure BCs.
   applyBC_Step2(2);

   // CONTROL
   //cout << "zeroPressureNode = " << zeroPressureNode << endl;
   
   //cout << "Z MATRIX" << endl;   
   //cout << endl;
   //cout << "z_nnz" << sparseZrowStarts[NNp]  << "   "  << sparseZ_NNZ << endl;
   //for (int i = 0; i < NNp; i++) {
      //cout << "sparseZrowStarts[" << i << "] = " << sparseZrowStarts[i] << endl;
      //for (int j = sparseZrowStarts[i]-1; j < sparseZrowStarts[i+1]-1; j++) {
         //cout << sparseZvalue[j] << "  " ;
      //}
      //cout << endl;
   //}
   //cout << "\n\n\n\n";
   
   //for (int i = 0; i < 125; i++) {
      //cout << "R2 [" << i << "] = " << R2[i] << endl;
   //}
   
   // Solve for Pdiff using MKL's CG solver.
   MKL_CG_solver();


   // CONTROL
   //for (int i=0; i<100; i++) {
     //cout << Pdiff[i] << endl;
   //}


   // Calculate Pnp1
   for (int i = 0; i < NNp; i++) {
      Pk[i] = Pk_prev[i] + Pdiff[i];
   }
   
   // CONTROL
   //for (int i=0; i<NNp; i++) {
   //   cout << Pnp1[i] << endl;
   //}

   delete[] Uk1;
   delete[] Uk2;
   delete[] Uk3;

}  // End of function step2()





//========================================================================
void paralution_BiCGStab_solver()

{
   // Solve system of linear equations using the Paralution's BiCGStab solver
   
   int NNZM = sparseM_NNZ / 3;   
     
   LocalMatrix<double> A_p;
   LocalVector<double> R1_p;
   LocalVector<double> Uk_dir_p;      
   
   A_p.SetDataPtrCSR(&sparseMrowStarts, &sparseMcol, &sparseAvalue, "BiCGStab LHS", NNZM, NN, NN) ;      

   R1_p.SetDataPtr(&R1, "BiCGStab RHS", NN);   

   Uk_dir_p.SetDataPtr(&Uk_dir, "BiCGStab result", NN);   
   Uk_dir_p.Zeros();
   
   
   double tick, tack;
   
   tick = paralution_time();
   
   // Linear Solver
   BiCGStab<LocalMatrix<double>, LocalVector<double>, double > ls;
   
   //abs_tol | rel_tol | div_tol | max_iter
   ls.Init(1e-15, 1e-6, 1e+8, 1000);

   // Preconditioner
   Jacobi<LocalMatrix<double>, LocalVector<double>, double > p;
   
   ls.SetOperator(A_p);
   ls.SetPreconditioner(p);
   
   ls.Build();

   ls.Solve(R1_p, &Uk_dir_p);
   
   tack = paralution_time();
   cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << endl;
   
   A_p.LeaveDataPtrCSR(&sparseMrowStarts, &sparseMcol, &sparseAvalue);   // Get back data from paralution
   R1_p.LeaveDataPtr(&R1);                                               // Get back data from paralution
   Uk_dir_p.LeaveDataPtr(&Uk_dir);                                       // Get back data from paralution
   
   ls.Clear();
   
}  // End of function paralution_BiCGStab_solver()





////-----------------------------------------------------------------------------
//void MKL_pardisoSolver()
////-----------------------------------------------------------------------------
//{
   //// Solve system of linear equations using the Pardiso solver available in
   //// Intel's MKL library.

   //int NN3 = 3*NN;
   
   //MKL_INT mtype = 11;      // Real unsymmetric matrix
   //MKL_INT nrhs = 1;        // Number of right hand sides
   //void *pt[64];

   //MKL_INT iparm[64];
   //MKL_INT maxfct, mnum, phase, error, msglvl;

   //MKL_INT i, j;
   //double ddum;             // Double dummy 
   //MKL_INT idum;            // Integer dummy

   //for (i = 0; i < 64; i++) {
      //iparm[i] = 0;
   //}

   //iparm[0] = 1;            // No solver default
   //iparm[1] = 2;            // Fill-in reordering from METIS
   //iparm[3] = 0;            // No iterative-direct algorithm
   //iparm[4] = 0;            // No user fill-in reducing permutation
   //iparm[5] = 0;            // Write solution into x
   //iparm[7] = 2;            // Max numbers of iterative refinement steps
   //iparm[9] = 13;           // Perturb the pivot elements with 1E-13
   //iparm[10] = 1;           // Use nonsymmetric permutation and scaling MPS
   //iparm[12] = 1;           // Maximum weighted matching algorithm is switched-on (default for non-symmetric)
   //iparm[13] = 0;           // Output: Number of perturbed pivots
   //iparm[17] = -1;          // Output: Number of nonzeros in the factor LU
   //iparm[18] = -1;          // Output: Mflops for LU factorization
   //iparm[19] = 0;           // Output: Numbers of CG Iterations
   //iparm[27] = 0;           // Double precision
   //iparm[34] = 1;           // zero-based indexing         

   //maxfct = 1;              // Maximum number of numerical factorizations
   //mnum = 1;                // Which factorization to use
   //msglvl = 0;              // Do not print statistical information
   //error = 0;               // Initialize error flag

   //for (i = 0; i < 64; i++) {
      //pt[i] = 0;
   //}

   //// Set phase to linear system solution
   //phase = 13;

   //// Set the number of cores to be used for parallel execution
   ////mkl_set_num_threads(16);

   //PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
            //&NN3, sparseAvalue, sparseMrowStarts, sparseMcol, &idum,
            //&nrhs, iparm, &msglvl, R1, Uk, &error);

   //if (error != 0) {
      //cout << "\nERROR during solution:" << error << "\n\n";
      //exit (3);
   //}

   ////printf ("\nSolution of the system is: \n");
   ////for (j = NN-NNp; j < NN; j++) {
      ////cout << "Uk [" << j << "] = " << Uk[j] << endl;
   ////}
   

   //// Release internal memory
   //phase = -1;
   //PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
            //&NN3, &ddum, sparseMrowStarts, sparseMcol, &idum, &nrhs,
            //iparm, &msglvl, &ddum, &ddum, &error);

//}  // End of function pardisoSolver()





//========================================================================
void MKL_GMRES_solver()
//========================================================================
{
   // Solve system of linear equations using the Pardiso solver available in
   // Intel's MKL library.
   
   //Rearrange CSR vectors for mkl's GMRES (make 0 based to 1 based) 
   
   for (int j = 0; j < 3*NN+1; j++){
      sparseMrowStarts[j] = sparseMrowStarts[j] + 1;
   } 

   for (int j = 0; j < sparseM_NNZ; j++){
      sparseMcol[j] = sparseMcol[j] + 1;
   }

   // Initialize the solution to zero
   for (int i = 0; i < 3*NN; i++) {
      Uk[i] = 0.0;
   }   
   
	/*---------------------------------------------------------------------------
	/* Some additional variables to use with the RCI (P)FGMRES solver
	/*---------------------------------------------------------------------------*/
  MKL_INT ipar[128];
  double dpar[128];  
  double *tmp;
  
  ipar[14] = 100;  //  the number of the non-restarted FGMRES iterations default=150
  int size_tmp = (2*ipar[14]+1) * 3*NN + (ipar[14] * (ipar[14]+9))/2 + 1; 
  tmp = new double[size_tmp];  

  MKL_INT itercount, expected_itercount = 5;
  MKL_INT RCI_request, i, ivar;
  double dvar;
  char cvar;

  //printf ("--------------------------------------------------\n");
  //printf ("The SIMPLEST example of usage of RCI FGMRES solver\n");
  //printf ("to solve a non-symmetric indefinite non-degenerate\n");
  //printf ("       algebraic system of linear equations\n");
  //printf ("--------------------------------------------------\n\n");

  ivar = 3*NN;
  cvar = 'N';

	/*---------------------------------------------------------------------------
	/* Initialize the solver
	/*---------------------------------------------------------------------------*/
  dfgmres_init (&ivar, Uk, R1, &RCI_request, ipar, dpar, tmp);
  if (RCI_request != 0) {
   printf("\nThis example FAILED as the solver has returned the ERROR ");
   printf ("code %d", RCI_request);  
  }
	/*---------------------------------------------------------------------------
	/* Set the desired parameters:
	/* LOGICAL parameters:
	/* do residual stopping test
	/* do not request for the user defined stopping test
	/* do the check of the norm of the next generated vector automatically
	/* DOUBLE PRECISION parameters
	/* set the relative tolerance to 1.0D-3 instead of default value 1.0D-6
	/*---------------------------------------------------------------------------*/
  ipar[4] = 1000;  //  the maximum number of iterrations default=150
  ipar[7] = 1;  
  ipar[8] = 1;
  ipar[9] = 0;
  ipar[11] = 1;
  dpar[0] = 1.0E-8;
	/*---------------------------------------------------------------------------
	/* Check the correctness and consistency of the newly set parameters
	/*---------------------------------------------------------------------------*/
  dfgmres_check (&ivar, Uk, R1, &RCI_request, ipar, dpar, tmp);
  if (RCI_request != 0) {
   printf("\nThis example FAILED as the solver has returned the ERROR ");
   printf ("code %d", RCI_request);  
  }  
	/*---------------------------------------------------------------------------
	/* Compute the solution by RCI (P)FGMRES solver without preconditioning
	/* Reverse Communication starts here
	/*---------------------------------------------------------------------------*/
ONE:dfgmres (&ivar, Uk, R1, &RCI_request, ipar, dpar, tmp);
	/*---------------------------------------------------------------------------
	/* If RCI_request=0, then the solution was found with the required precision
	/*---------------------------------------------------------------------------*/
   
// cout << "RCI_request = " << RCI_request << endl;
  if (RCI_request == 0)
    goto COMPLETE;    
	/*---------------------------------------------------------------------------
	/* If RCI_request=1, then compute the vector A*tmp[ipar[21]-1]
	/* and put the result in vector tmp[ipar[22]-1]
	/*---------------------------------------------------------------------------
	/* NOTE that ipar[21] and ipar[22] contain FORTRAN style addresses,
	/* therefore, in C code it is required to subtract 1 from them to get C style
	/* addresses
	/*---------------------------------------------------------------------------*/
  if (RCI_request == 1)
    {
      mkl_dcsrgemv (&cvar, &ivar, sparseAvalue, sparseMrowStarts, sparseMcol, &tmp[ipar[21] - 1],
		    &tmp[ipar[22] - 1]);
      goto ONE;
    }    
	/*---------------------------------------------------------------------------
	/* If RCI_request=anything else, then dfgmres subroutine failed
	/* to compute the solution vector: computed_solution[N]
	/*---------------------------------------------------------------------------*/
  else
    {
      printf("\nThis example FAILED as the solver has returned the ERROR ");
      printf ("code %d", RCI_request);
    }
	/*---------------------------------------------------------------------------
	/* Reverse Communication ends here
	/* Get the current iteration number and the FGMRES solution (DO NOT FORGET to
	/* call dfgmres_get routine as computed_solution is still containing
	/* the initial guess!)
	/*---------------------------------------------------------------------------*/
 
COMPLETE:dfgmres_get (&ivar, Uk, R1, &RCI_request, ipar, dpar, tmp,
	       &itercount);

  /*
     /*---------------------------------------------------------------------------
     /* Print solution vector: computed_solution[N] and the number of iterations:
     itercount
     /*--------------------------------------------------------------------------- */
  if (PRINT_TIMES) cout << "MKL_GMRES converged after " << itercount << " iterations." << endl;     

  // printf ("\n The following solution has been obtained: \n");
  // for (i = 0; i < N; i++)
    // {
      // printf ("computed_solution[%d]=", i);
      // printf ("%e\n", x[i]);
    // }
  
   for (int j = 0; j < 3*NN+1; j++){
      sparseMrowStarts[j] = sparseMrowStarts[j] - 1;
   }

   for (int j = 0; j < sparseM_NNZ; j++){
      sparseMcol[j] = sparseMcol[j] - 1;
   }
   
   //for (int i=500; i < 729; i++) {
      //cout << i << "  " << Uk[i] << endl;    
   //}    
   
   MKL_Free_Buffers();
   delete[] tmp;

}  // End of function GMRESsolver()





//========================================================================
void MKL_CG_solver()

{
   // Solve the system of step 2 [Z]{Pdot}={R2} using CG.
   // Reference: CG and PCG examples coming with Intel MKL.

   MKL_INT n, rci_request;
   n = NNp;

   MKL_INT ipar[128];
   double dpar[128], *tmp;
   tmp = new double[4*n];
   char tr = 'u';
   char matdes[3];
   double one = 1.000000000000000;

   matdes[0] = 'd';
   matdes[1] = 'l';
   matdes[2] = 'n';

   // Initialize the solution to zero
   for (int i = 0; i < NNp; i++) {
      Pdiff[i] = 0.0;
   }

   dcg_init (&n, Pdiff, R2, &rci_request, ipar, dpar, tmp);
   if (rci_request != 0) {
      printf("FAILURE in MKL CG solver. ERROR code %d\n\n", rci_request);
      MKL_Free_Buffers();
      return;
   }

   ipar[4] = 1000;    // Max. iteration number. Default is max(150,n)
   ipar[7] = 1;       // Perform iteration number based stopping check. Default is 1.
   ipar[8] = 1;       // Perform residual based stopping check. Default is 0.
   ipar[9] = 0;       // Do not perform user specified stopping check. Default is 1.
   //ipar[10] = 1;      // Perform Jacobi Preconditioner   
   dpar[0] = 1e-6;    // Relative tolerance. Default is 1e-6.

   int solverIter;

   dcg_check (&n, Pdiff, R2, &rci_request, ipar, dpar, tmp);
   if (rci_request != 0) {
      printf("FAILURE in MKL CG solver. ERROR code %d\n\n", rci_request);
      MKL_Free_Buffers();
      return;
   }

   rci:dcg (&n, Pdiff, R2, &rci_request, ipar, dpar, tmp);
   if (rci_request == 0) {   // The solution is found with the required precision
      dcg_get (&n, Pdiff, R2, &rci_request, ipar, dpar, tmp, &solverIter);
      if (PRINT_TIMES) cout << "MKL_CG converged after " << solverIter << " iterations." << endl;
      MKL_Free_Buffers();
      goto out;
   } else if (rci_request == 1) { // Compute the vector A*tmp[0] and put the result in vector tmp[n]
      mkl_dcsrsymv (&tr, &n, sparseZvalue, sparseZrowStarts, sparseZcol, tmp, &tmp[n]);
      goto rci;
   } else if (rci_request == 3) { // Apply the preconditioner matrix C_inverse on vector tmp[2*n] and put the result in vector tmp[3*n]
      mkl_dcsrsv (&matdes[2], &n, &one, matdes, sparseZvalue, sparseZcol, sparseZrowStarts, &sparseZrowStarts[1], &tmp[2*n], &tmp[3*n]);
      goto rci;
   } else {  // If rci_request=anything else, then dcg subroutine failed
      printf("FAILURE in MKL CG solver. ERROR code %d\n\n", rci_request);
      MKL_Free_Buffers();
      return;
   }

   out:
   // CONTROL
   //for(int i = 0; i < NNp; i++) {
      //cout << "Pdiff[" << i << "] = " << Pdiff[i] << endl;
   //}

   delete[] tmp;

}  // End of function MKL_CG_solver()





//========================================================================
void applyBC_initial()
//========================================================================
{
   // Apply the specified BCs before the solution starts.
   //double x, y, z;
   int node, whichBC;

    int counter = 0;
    double x, y, z, velocity;

   // Apply velocity BCs
   for (int i = 0; i < BCnVelNodes; i++) {
      node = BCvelNodes[i][0];     // Node at which this velocity BC is specified.
      whichBC = BCvelNodes[i][1];  // Number of the specified BC

      //x = coord[node][0];           // May be necessary for BC.str evaluation
      //y = coord[node][1];
      //z = coord[node][2];
     
      // Change Un with the given u, v and w velocities.
      Uk_prev[node]        = BCstr[whichBC][0];                                                                                                           // TODO : Actually BCstr should be strings and here we need a function parser.
      Uk_prev[node + NN]   = BCstr[whichBC][1];
      Uk_prev[node + 2*NN] = BCstr[whichBC][2];

      // Below is for the fully-developed inlet of the bending square duct problem
      
      //if (whichBC == 0) {
         //counter++;

         //x = coord[node][0];
         //y = coord[node][1];
         //z = coord[node][2];

         //velocity = 2.25 * (4*y - 4*y*y) * (4*z - 4*z*z);   // Average u is 1.0

         ////printf("%d : %f   %f   %f   %f\n", counter, x, y, z, velocity);

         //Uk_prev[node]        = velocity;
         //Uk_prev[node + NN]   = 0.0;
         //Uk_prev[node + 2*NN] = 0.0;
      //}
      
   }

   // CONTROL
   //for (int i = 0; i < 3*NN; i++) {
   //   cout << Un[i] << endl;
   //}
}  // End of function applyBC_initial()





//========================================================================
void applyBC_Step1(int flag)

{
   // When flag=1, modify system matrix (step 1) for velocity BCs. When flag=2, modify the right
   // hand side vector of step 1 (R1) for velocity BCs.

   int node, whichBC;

   if (flag == 1) {
      for (int i = 0; i < BCnVelNodes; i++) {
         node = BCvelNodes[i][0];   // Node at which this velocity BC specified.
         
         for (int j = sparseMrowStarts[node]; j < sparseMrowStarts[node+1]; j++) {
            sparseAvalue[j] = 0.0;
            if(sparseMcol[j] == node) {                                      
               sparseAvalue[j] = 1.0;    
            }         
         }
      }

   } else if (flag == 2) {
      for (int i = 0; i < BCnVelNodes; i++) {
         node = BCvelNodes[i][0];   // Node at which this velocity BC is specified.
         whichBC = BCvelNodes[i][1];  // Number of the specified BC         
     
         // Change R for the given u, v and w velocities.
         R1[node] = BCstr[whichBC][direction];  // This is velocity.
      }
   }
}  // End of function applyBC_Step1()





//========================================================================
void applyBC_Step2(int flag)

{
   // When flag=1, modify Z for pressure BCs. When flag=2, modify the right
   // hand side vector of step 2 (R2) for pressure BCs.

   // WARNING : In step 2 pressure differences between 2 iterations is
   // calculated. Therefore when specifying pressure BCs a value of zero is
   // specified instead of the original pressure value.

   // In order not to break down the symmetry of [Z], we use the "LARGE number"
   // trick.

   double LARGE = 1000;                                                                                                                               // TODO: Implement EBCs without the use of LARGE.

   int node = zeroPressureNode;     // Node at which pressure is set to zero.

   if (flag == 1) {
      //if (node > 0) {  // If node is negative it means we do not set pressure to zero at any node.
         //// Multiply Z[node][node] by LARGE
         //for (int j = sparseZrowStarts[node]; j < sparseZrowStarts[node+1]; j++) {  // Go through column "node" of [L].
            //if (sparseZcol[j] == node) {   // Determine the position of the diagonal entry in column "node"
               //sparseZvalue[j] = sparseZvalue[j] * LARGE;
               //break;
            //}
         //}
      //}
      if (node > 0) {  // If node is negative it means we do not set pressure to zero at any node.
         // Multiply Z[node][node] by LARGE
         #ifdef USECUDA
            for (int j = sparseZrowStarts[node]; j < sparseZrowStarts[node+1]; j++) {  // Go through column "node" of [L].
               if (sparseZcol[j] == node) {   // Determine the position of the diagonal entry in column "node"
                  sparseZvalue[j] = sparseZvalue[j] * LARGE;
                  break;
               }
            }
         #else
            for (int j = (sparseZrowStarts[node]-1); j < (sparseZrowStarts[node+1]-1); j++) {  // Go through column "node" of [L].
               if ( (sparseZcol[j] - 1) == node) {   // Determine the position of the diagonal entry in column "node"
                  sparseZvalue[j] = sparseZvalue[j] * LARGE;
                  break;
               }
            }
         #endif
      }
   } else if (flag == 2) {
      if (node > 0) {  // If node is negative it means we do not set pressure to zero at any node.
         R2[node] = 0.0;  // This is not the RHS for pressure, but pressure difference between 2 iterations.
      }
   }
}  // End of function applyBC_Step2()





//========================================================================
void readRestartFile()
//========================================================================
{
   // Reads the restart file, which is a Tecplot DAT file

   double dummy1, dummy2, dummy3, dummy4;
   
   ifstream restartFile;
 
   restartFile.open((whichProblem + "_restart.dat").c_str(), ios::in);
     
   restartFile.ignore(256, '\n');   // Read and ignore the line
   restartFile.ignore(256, '\n');   // Read and ignore the line
   restartFile.ignore(256, '\n');   // Read and ignore the line

   // Read u, v, w and p values
   for (int i = 0; i<NCN; i++) {
      restartFile >> dummy1 >> dummy2 >> dummy3 >> Uk_prev[i] >> Uk_prev[NN + i] >> Uk_prev[2*NN + i] >> Pk_prev[i];
      restartFile.ignore(256, '\n');   // Ignore the rest of the line
   }
   
   for (int i = NCN; i<NN; i++) {
      restartFile >> dummy1 >> dummy2 >> dummy3 >> Uk_prev[i] >> Uk_prev[NN + i] >> Uk_prev[2*NN + i] >> dummy4;
      restartFile.ignore(256, '\n');   // Ignore the rest of the line
   }
   
   restartFile.close();
   
} // End of function readRestartFile()





//========================================================================
void createTecplot()
//========================================================================
{
   // This file is used if NENp and NENv are different. In this case each
   // hexahedral element is divided into eight sub-elements and Tecplot file
   // is created as if there are 8*NE hexahedral elements. Missing mid-edge,
   // mid-face and mid-element pressure values are evaluated by linear
   // interpolation.

   // Call the simple version of this function if NENp and NENv are the same.

   int node, n1, n2, n3, n4, n5, n6, n7, n8;

   // Write the calculated unknowns to a Tecplot file
   datFile.open((whichProblem + ".dat").c_str(), ios::out);

   datFile << "TITLE = " << whichProblem << endl;
   datFile << "VARIABLES = x,  y,  z,  u, v, w, p" << endl;
   
   if (eType == 1) {
      datFile << "ZONE N=" <<  NN  << ", E=" << 8*NE << ", F=FEPOINT, ET=BRICK" << endl;
      // New Tecplot 360 documentation has the following format but the above seems to be working also
      // ZONE NODES=..., ELEMENTS=..., DATAPACKING=POINT, ZONETYPE=FEBRICK
   } else {
      printf("\n\n\nERROR: Tetrahedral elements are not implemented in function createTecplot() yet!!!\n\n\n");
   }

   // Separate Un into uNode, vNode and wNode variables
   double *uNode, *vNode, *wNode;
   uNode = new double[NN];
   vNode = new double[NN];
   wNode = new double[NN];

   for (int i = 0; i < NN; i++) {
      uNode[i] = Uk[i];
      vNode[i] = Uk[i+NN];
      wNode[i] = Uk[i+2*NN];
   }

   // Copy pressure solution into pNode array, but the size of pNode is NN,
   // because it will also store pressure values at mid-egde, mid-face and
   // mid-element nodes.
   double *pNode;
   pNode = new double[NN];

   for (int i = 0; i < NNp; i++) {
      pNode[i] = Pk[i];
   }

   // Interpolate pressure at non-corner nodes.
   for (int e = 0; e < NE; e++) {
      // Calculate mid-edge pressures as averages of the corner pressures.
      for (int ed = 0; ed < NEE; ed++) {
         // Determine corner nodes of edge ed
         if (eType == 1) {   // Hexahedral element
            switch (ed) {
            case 0:
              n1 = LtoGnode[e][0];
              n2 = LtoGnode[e][1];
              break;
            case 1:
              n1 = LtoGnode[e][1];
              n2 = LtoGnode[e][2];
              break;
            case 2:
              n1 = LtoGnode[e][2];
              n2 = LtoGnode[e][3];
              break;
            case 3:
              n1 = LtoGnode[e][3];
              n2 = LtoGnode[e][0];
              break;
            case 4:
              n1 = LtoGnode[e][0];
              n2 = LtoGnode[e][4];
              break;
            case 5:
              n1 = LtoGnode[e][1];
              n2 = LtoGnode[e][5];
              break;
            case 6:
              n1 = LtoGnode[e][2];
              n2 = LtoGnode[e][6];
              break;
            case 7:
              n1 = LtoGnode[e][3];
              n2 = LtoGnode[e][7];
              break;
            case 8:
              n1 = LtoGnode[e][4];
              n2 = LtoGnode[e][5];
              break;
            case 9:
              n1 = LtoGnode[e][5];
              n2 = LtoGnode[e][6];
              break;
            case 10:
              n1 = LtoGnode[e][6];
              n2 = LtoGnode[e][7];
              break;
            case 11:
              n1 = LtoGnode[e][7];
              n2 = LtoGnode[e][4];
              break;
            }  // End of ed switch
         } else if (eType == 2) {   // Tetrahedral element
           printf("\n\n\nERROR: Tetrahedral elements are not implemented in function createTecplot() yet!!!\n\n\n");
         }
         
         node = LtoGnode[e][ed+NEC];

         pNode[node] = 0.5 * (pNode[n1] + pNode[n2]);

      }  // End of ed (edge) loop


      // Calculate mid-face pressures as averages of the corner pressures.
      for (int f = 0; f < NEF; f++) {
  
         // Determine corner nodes of face f
         if (eType == 1) {   // Hexahedral element
            switch (f) {
            case 0:
               n1 = LtoGnode[e][0];
               n2 = LtoGnode[e][1];
               n3 = LtoGnode[e][2];
               n4 = LtoGnode[e][3];
               break;
            case 1:
               n1 = LtoGnode[e][0];
               n2 = LtoGnode[e][1];
               n3 = LtoGnode[e][4];
               n4 = LtoGnode[e][5];
               break;
            case 2:
               n1 = LtoGnode[e][1];
               n2 = LtoGnode[e][2];
               n3 = LtoGnode[e][5];
               n4 = LtoGnode[e][6];
               break;
            case 3:
               n1 = LtoGnode[e][2];
               n2 = LtoGnode[e][3];
               n3 = LtoGnode[e][6];
               n4 = LtoGnode[e][7];
               break;
            case 4:
               n1 = LtoGnode[e][0];
               n2 = LtoGnode[e][3];
               n3 = LtoGnode[e][4];
               n4 = LtoGnode[e][7];
               break;
            case 5:
               n1 = LtoGnode[e][4];
               n2 = LtoGnode[e][5];
               n3 = LtoGnode[e][6];
               n4 = LtoGnode[e][7];
               break;
            }

            node = LtoGnode[e][f+NEC+NEE];
            pNode[node] = 0.25 * (pNode[n1] + pNode[n2] + pNode[n3] + pNode[n4]);
         
         } else if (eType == 2) {   // Tetrahedral element
            printf("\n\n\nERROR: Tetrahedral elements are not implemented in function createTecplot() yet!!!\n\n\n");
         }
      }  // End of f (face) loop


      // Find add the mid-element node pressures.
      if (eType == 1) {   // Hexahedral element
         n1 = LtoGnode[e][0];
         n2 = LtoGnode[e][1];
         n3 = LtoGnode[e][2];
         n4 = LtoGnode[e][3];
         n5 = LtoGnode[e][4];
         n6 = LtoGnode[e][5];
         n7 = LtoGnode[e][6];
         n8 = LtoGnode[e][7];
    
         node = LtoGnode[e][NEC+NEE+NEF];
    
         pNode[node] = 0.125 * (pNode[n1] + pNode[n2] + pNode[n3] + pNode[n4] + pNode[n5] + pNode[n6] + pNode[n7] + pNode[n8]);

      } else if (eType == 2) {   // Tetrahedral element
         printf("\n\n\nERROR: Tetrahedral elements are not implemented in function createTecplot() yet!!!\n\n\n");
      }
   }  // End of element loop


   // Print the coordinates and the calculated velocity and pressure values
   double x, y, z;
   for (int i = 0; i < NN; i++) {
      x = coord[i][0];
      y = coord[i][1];
      z = coord[i][2];
      datFile.precision(11);
      datFile << scientific << x << " " << y << " " << z << " " << uNode[i] << " " << vNode[i] << " " << wNode[i] << " " << pNode[i] << endl;
   }


   // Print the connectivity list. We will divide hexahedral elements into 8
   // and divide tetrahedral elements into ... elements.                                                                                             TODO
   if (eType == 1) {   // Hexahedral elements
      for (int e = 0; e < NE; e++) {
         // 1st sub-element of element e
         datFile << LtoGnode[e][0] + 1 << " " << LtoGnode[e][8] + 1 << " " << LtoGnode[e][20] + 1 << " " << LtoGnode[e][11] + 1 << " " << LtoGnode[e][12] + 1 << " " << LtoGnode[e][21] + 1 << " " << LtoGnode[e][26] + 1 << " " << LtoGnode[e][24] + 1 << endl;
         // 2nd sub-element of element e
         datFile << LtoGnode[e][8] + 1 << " " << LtoGnode[e][1] + 1 << " " << LtoGnode[e][9] + 1 << " " << LtoGnode[e][20] + 1 << " " << LtoGnode[e][21] + 1 << " " << LtoGnode[e][13] + 1 << " " << LtoGnode[e][22] + 1 << " " << LtoGnode[e][26] + 1 << endl;
         // 3rd sub-element of element e
         datFile << LtoGnode[e][11] + 1 << " " << LtoGnode[e][20] + 1 << " " << LtoGnode[e][10] + 1 << " " << LtoGnode[e][3] + 1 << " " << LtoGnode[e][24] + 1 << " " << LtoGnode[e][26] + 1 << " " << LtoGnode[e][23] + 1 << " " << LtoGnode[e][15] + 1 << endl;
         // 4th sub-element of element e
         datFile << LtoGnode[e][20] + 1 << " " << LtoGnode[e][9] + 1 << " " << LtoGnode[e][2] + 1 << " " << LtoGnode[e][10] + 1 << " " << LtoGnode[e][26] + 1 << " " << LtoGnode[e][22] + 1 << " " << LtoGnode[e][14] + 1 << " " << LtoGnode[e][23] + 1 << endl;
         // 5th sub-element of element e
         datFile << LtoGnode[e][12] + 1 << " " << LtoGnode[e][21] + 1 << " " << LtoGnode[e][26] + 1 << " " << LtoGnode[e][24] + 1 << " " << LtoGnode[e][4] + 1 << " " << LtoGnode[e][16] + 1 << " " << LtoGnode[e][25] + 1 << " " << LtoGnode[e][19] + 1 << endl;
         // 6th sub-element of element e
         datFile << LtoGnode[e][21] + 1 << " " << LtoGnode[e][13] + 1 << " " << LtoGnode[e][22] + 1 << " " << LtoGnode[e][26] + 1 << " " << LtoGnode[e][16] + 1 << " " << LtoGnode[e][5] + 1 << " " << LtoGnode[e][17] + 1 << " " << LtoGnode[e][25] + 1 << endl;
         // 7th sub-element of element e
         datFile << LtoGnode[e][24] + 1 << " " << LtoGnode[e][26] + 1 << " " << LtoGnode[e][23] + 1 << " " << LtoGnode[e][15] + 1 << " " << LtoGnode[e][19] + 1 << " " << LtoGnode[e][25] + 1 << " " << LtoGnode[e][18] + 1 << " " << LtoGnode[e][7] + 1 << endl;
         // 8th sub-element of element e
         datFile << LtoGnode[e][26] + 1 << " " << LtoGnode[e][22] + 1 << " " << LtoGnode[e][14] + 1 << " " << LtoGnode[e][23] + 1 << " " << LtoGnode[e][25] + 1 << " " << LtoGnode[e][17] + 1 << " " << LtoGnode[e][6] + 1 << " " << LtoGnode[e][18] + 1 << endl;
      }
   } else if (eType == 2) {  // Tetrahedral elements
      printf("\n\n\nERROR: Tetrahedral elements are not implemented in function createTecplot() yet!!!\n\n\n");
   }

   delete[] uNode;
   delete[] vNode;
   delete[] wNode;
   delete[] pNode;

   datFile.close();

}  // End of function createTecplot()





//-----------------------------------------------------------------------------
double getHighResolutionTime(int flag, double start)
//-----------------------------------------------------------------------------
{
   // If flag is 1 return the current time (start value is not used).
   // If flag is 2 return the current time minus the start value, i.e. return an elapsed time.

   #ifdef WINDOWS
      // Windows
      if (flag == 1) {
         return double(clock());    // On Windows, clock() returns wall clock time.
                                    // On Linux it returns CPU time.
      } else {
         return ( double(clock()) - start ) / CLOCKS_PER_SEC;
      }
      
   #else
      // Linux
      struct timeval tod;

      gettimeofday(&tod, NULL);  // Measures wall clock time
      double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);

      if (flag == 1) {
         return time_seconds;
      } else {
         return (time_seconds - start);
      }
   #endif // WINDOWS

} // End of function getHighResolutionTime()





//-----------------------------------------------------------------------------
void waitForUser(string str)
//-----------------------------------------------------------------------------
{
   // Used for checking memory usage. Prints the input string to the screen and
   // waits for the user to enter a character.

   char dummyUserInput;
   //cout << str;
   //cin >> dummyUserInput;
}




























/*

//------------------------------------------------------------------------------
void readRestartFile()
//------------------------------------------------------------------------------
{
   // Reads the restart file, which is a Tecplot DAT file

   double dummy;
   ifstream restartFile;
   
   restartFile.open((whichProblem + restartExtension).c_str(), ios::in);
     
   restartFile.ignore(256, '\n');   // Read and ignore the line
   restartFile.ignore(256, '\n');   // Read and ignore the line
   restartFile.ignore(256, '\n');   // Read and ignore the line

   // Read u, v, w and p values
   for (int i = 0; i<NN; i++) {
      restartFile >> dummy >> dummy >> dummy >> u[i] >> v[i] >> w[i] >> p[i];
      restartFile.ignore(256, '\n');   // Ignore the rest of the line
   }

   restartFile.close();
   
} // End of function readRestartFile()

*/





/* CSparse Exercise

   csi *Ai, *Aj;
   double *Ax;

   Ai = new csi[4];
   Aj = new csi[4];
   Ax = new double[10];

   Ai[0] = 0;   Aj[0] = 0;   Ax[0] = 4.5;
   Ai[1] = 1;   Aj[1] = 0;   Ax[1] = 3.1;
   Ai[2] = 3;   Aj[2] = 0;   Ax[2] = 3.5;
   Ai[3] = 1;   Aj[3] = 1;   Ax[3] = 2.9;
   Ai[4] = 2;   Aj[4] = 1;   Ax[4] = 1.7;
   Ai[5] = 3;   Aj[5] = 1;   Ax[5] = 0.4;
   Ai[6] = 0;   Aj[6] = 2;   Ax[6] = 3.2;
   Ai[7] = 2;   Aj[7] = 2;   Ax[7] = 3.0;
   Ai[8] = 1;   Aj[8] = 3;   Ax[8] = 0.9;
   Ai[9] = 3;   Aj[9] = 3;   Ax[9] = 1.0;

   csi m = 4;
   csi n = 4;
   csi nzmax = 10;
   csi values = 1;
   csi triplet = 1;
 
   cs *A_cs;
   A_cs = cs_spalloc(m, n, nzmax, values, triplet);

   A_cs->i = Ai;
   A_cs->p = Aj;
   A_cs->x = Ax;
   A_cs->nz = 10;

   cs *B_cs;
   B_cs = cs_compress(A_cs);
   cs_print(A_cs, 0);
   cs_print(B_cs, 0);

   //cs_spfree(A_cs);
   //cs_spfree(B_cs);
*/





/* CUSP Exercise

   int *rowIndices;
   int *colIndices;
   float *values;

   int N = 3;
   int NNZ = 3;
   rowIndices = new int[N];
   colIndices = new int[N];
   values = new float[NNZ];

   // initialize matrix entries on host
   rowIndices[0] = 0; colIndices[0] = 2; values[0] = 10;
   rowIndices[1] = 1; colIndices[1] = 0; values[1] = 20;
   rowIndices[2] = 2; colIndices[2] = 0; values[2] = 30;

  // A now represents the following matrix
  //    [ 0  0  10]
  //    [20  0   0]
  //    [30  0   0]

  cusp::coo_matrix<int,float,cusp::host_memory> A(N,N,NNZ);
  thrust::copy(rowIndices, rowIndices + N, A.row_indices.begin());
  thrust::copy(colIndices, colIndices + N, A.column_indices.begin());
  thrust::copy(values, values + NNZ, A.values.begin());

  cusp::coo_matrix<int,float,cusp::device_memory> B(N,N,NNZ);
  thrust::copy(rowIndices, rowIndices + N, B.row_indices.begin());
  thrust::copy(colIndices, colIndices + N, B.column_indices.begin());
  thrust::copy(values, values + NNZ, B.values.begin());

  cusp::csr_matrix<int,float,cusp::device_memory> Acsr(A);
  cusp::csr_matrix<int,float,cusp::device_memory> Bcsr(B);

  cusp::coo_matrix<int,float,cusp::device_memory> C;
  cusp::multiply(Acsr, Bcsr, C);
  cusp::print(C);
*/





/* MKL Sparse Matrix Vector Multiplication Test (Extracted from MKL's cspblas_dcs.c example code)

#define M 5
#define NNZ 13

double values[NNZ] = {1.0, -1.0, -3.0, -2.0, 5.0, 4.0, 6.0, 4.0, -4.0, 2.0, 7.0, 8.0, -5.0};
int columns[NNZ]   = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
int rowIndex[M+1]  = {0, 3,  5,  8,  11, 13};
double x_vec[M]	 = {1.0, 1.0, 1.0, 1.0, 1.0};
double y_vec[M]	 = {0.0, 0.0, 0.0, 0.0, 0.0};

char transa, matdescra[6];
double alpha = 1.0, beta = 0.0;
int m, k;
int pointerB[M], pointerE[M];

transa = 'n';
   
matdescra[0] = 'g';
matdescra[1] = 'u';
matdescra[2] = 'n';
matdescra[3] = 'c';
   
m = M;
   
for (int i = 0; i < m; i++) {
   pointerB[i] = rowIndex[i];
   pointerE[i] = rowIndex[i+1];
};
   
mkl_dcsrmv(&transa, &m, &m, &alpha, matdescra, values, columns, pointerB, pointerE, x_vec, &beta, y_vec);

for (int i = 0; i < m; i++) {
   printf("%7.1f\n", y_vec[i]);
}

*/





/*
// CSparse transpose test
cs *AA;
cs *AAt;
int AA_NNZ = 10;
int *AArow;
AArow = new int[AA_NNZ];
int *AAcolstarts;
AAcolstarts = new int[5];
double *AAvalue;
AAvalue = new double[AA_NNZ];
AArow[0] = 0;
AArow[1] = 1;
AArow[2] = 3;
AArow[3] = 1;
AArow[4] = 2;
AArow[5] = 3;
AArow[6] = 0;
AArow[7] = 2;
AArow[8] = 1;
AArow[9] = 3;
AAvalue[0] = 4.5;
AAvalue[1] = 3.2;
AAvalue[2] = 3.1;
AAvalue[3] = 2.9;
AAvalue[4] = 0.9;
AAvalue[5] = 1.7;
AAvalue[6] = 3.0;
AAvalue[7] = 3.5;
AAvalue[8] = 0.4;
AAvalue[9] = 1.0;
AAcolstarts[0] = 0;
AAcolstarts[1] = 3;
AAcolstarts[2] = 6;
AAcolstarts[3] = 8;
AAcolstarts[4] = 10;

AA = cs_spalloc(4, 4, AA_NNZ, 1, 0);
AA->i = AArow;
AA->p = AAcolstarts;
AA->x = AAvalue;
//AA->nz = AA_NNZ;  // cs_print fails when you put this

cs_print(AA,0);

//G_cs_CSC = cs_compress(G_cs);             // Convert G from triplet format into compressed column format.

AAt = cs_transpose(AA, 1);    // Determine the transpose of G in compressed column format.

cs_print(AAt,0);
*/

