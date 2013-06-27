/*****************************************************************
*        This code is a part of the CFD-with-CUDA project        *
*             http://code.google.com/p/cfd-with-cuda             *
*                                                                *
*              Dr. Cuneyt Sert and Mahmut M. Gocmen              *
*                                                                *
*              Department of Mechanical Engineering              *
*                Middle East Technical University                *
*                         Ankara, Turkey                         *
*            http://www.me.metu.edu.tr/people/cuneyt             *
*                                                                *
*                                                                *
* The following definitions can be made during compilation       *
*                                                                *
* -DSIGLE     : Major variables are defined as single precision  *
* -DCG_CUDA   : Pressure correction equation solver selection    *
* -DCR_CUDA   :    "                                             *
* -DCG_CUSP   :    "                                             *
* -DCR_CUSP   :    "                                             *
* -DGMRES_CUSP: Momentum equation solver selection               *
* -DBiCG_CUSP :    "                                             *
*****************************************************************/

#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <algorithm>

#include <cusparse.h>
#include <cublas.h>

using namespace std;

#ifdef SINGLE              // Many major parameters can automatically defined to be
  typedef float real;      // float by using -DSINGLE during compilcation. Default
#else                      // bahavior is to use double precision.
  typedef double real;
#endif



//------------------------------------------------------------------------------
// Global Variables
//------------------------------------------------------------------------------

ifstream meshfile;          // Input file with INP extension.
ifstream problemFile;       // Input file name is read from this ProblemFile.txt file.
ofstream outputFile;        // Output file with DAT extension.
ofstream outputControl;     // Used for debugging.
ofstream outputPostProcess; // Output file for examine results.

string problemNameFile, whichProblem;
string problemName      = "ProblemName.txt";
string controlFile      = "Control_Output.txt";
string inputExtension   = ".inp";
string outputExtension;
string outputExtensionPostProcess;
string restartExtension = "_restart.dat";


int eType;   // Element type. 3: 3D Hexahedron, 4: 3D Tetrahedron
int NE;      // Number of elements
int NN;      // Number of both corner and mid-face nodes, if there are any 
int NGP;     // Number of GQ points.
int NEU;     // Number of elemental unknowns, 3*NENv + NENp
int Ndof;    // Total number of unknowns in the problem, 4*NN

int NCN;               // Number of corner nodes of the mesh.
int NENv;              // Number of velocity nodes on an element.
int NENp;              // Number of pressure nodes on an element.
int nonlinearIterMax;  // Maximum number of nonlinear iterations.
int solverIterMax;     // Maximum number of iterations of the iterative solvers.
int nDATiter;          // Period of saving results.

double density;        // Density of the material.
double viscosity;      // Viscosity of the material.
double fx, fy;         // 
double nonlinearTol;   // Nonlinear iteration tolerance.
double solverTol;      // Iterative solver tolerance.

bool isRestart;        // Switch that defines if solver continues from a previous solution(results)
                       // or starts from beginning.
int **LtoG;            // Local to global node mapping. (size:NExNENv)
int **velNodes;        // Nodes that are defined as velocity boundary.
int **pressureNodes;   // Nodes that are defined as pressure boundary.

double **monitorPoints; // Coordinates of points that are wanted to monitor.
int *monitorNodes;      // Mesh nodes that are closest to the coordinates of points that are wanted to monitor.
double **coord;         // Coordinates (x, y, z) of the mesh nodes. (size:NNx3)
int nBC;                // Number of different boundary conditions.
int nVelNodes;          // Number of mesh nodes where velocity is specified.
int nPressureNodes;     // Number of mesh nodes where pressure is specified.
int nMonitorPoints;     // Number of monitor points.

double *BCtype;         // Type of boundary condition.
double **BCstrings;     // Values of boundary condition. 

double axyFunc, fxyFunc; 
double **GQpoint, *GQweight; // GQ points and weights.

double **Sp;     // Shape functions for pressure evaluated at GQ points. (size:NENpxNGP)
double ***DSp;   // Derivatives of shape functions for pressure wrt to ksi, eta & zeta evaluated at GQ points. (size:NENpxNGP)
double **Sv;     // Shape functions for velocity evaluated at GQ points. (size:NENvxNGP) 
double ***DSv;   // Derivatives of shape functions for velocity wrt to ksi, eta & zeta evaluated at GQ points. (size:NENvxNGP)

double **detJacob; // Determinant of the Jacobian matrix evaluated at a certain (ksi, eta, zeta)
double ****gDSp;   // Derivatives of shape functions for pressure wrt x, y & z at GQ points. (size:3xNENvxNGP)
double ****gDSv;   // Derivatives of shape functions for velocity wrt x, y & z at GQ points. (size:3xNENvxNGP)

int **GtoL;           // Stores information of nodes present in which elements. (size:NNx?)
int *rowStartsSmall;  // Rows of the nonzeros of the stiffness matrix. (size:NNZ) 
int *colSmall;        // Columns of the nonzeros of the stiffness matrix. (size:NNZ)
int **KeKMapSmall;    // Store the location of each entry of elemental Ke in sparsely stored global
                      // stiffness matrix. (size:NENvxNENv)
int ***KeKMap;        // Store the location of each entry of elemental Ke in sparsely stored global
                      // stiffness matrix for all elements. (size: NExNENvxNENv)           
int NNZ;              // Number of nonzero entries in global stiffness matrix.

int *rowStartsDiagonal; // Rows of the nonzeros of the diagonal matrix. (size:NNZ) 
int *colDiagonal;       // Columns of the nonzeros of the diagonal matrix. (size:NNZ)
int NNZ_diagonal;       // Number of nonzero entries in diagonal matrix.

double **Ke_1, **Ke_2, **Ke_3;              // Elemental stiffness matrices. (size:NN)
                                            // Ke_1 is related with K_u_diagonal.
                                            // Ke_2 is related with K_v_diagonal.
                                            // Ke_3 is related with K_w_diagonal.
double **Ke_1_add, **Ke_2_add, **Ke_3_add;  // For constructing elemental stiffness matrices. (size:NENvxNENv)
double *uNodal, *vNodal, *wNodal;           // Nodal unknowns for an element. (size:NENv)
double u0, v0, w0;                          // Unknowns at GQ points. 
double *Du0, *Dv0, *Dw0;                    // Derivatives of the unknowns at GQ points. (size:NENv)

real *F;                          // Global force vectors for momentum equations. (size:NN)
real *u, *v, *w;                  // Global unknown vectors for velocities. (size:NN)
real *u_temp, *v_temp, *w_temp;   // Temporary unknown vectors for velocities. (size:NN)
real *delta_p, *p;                // Global unknown vectors for pressures. (size:NN)

real *K_u_diagonal, *K_v_diagonal, *K_w_diagonal; // Diagonal members of the stiffness matrices (K_uu, K_vv, K_ww). (size:NN)
real *tempDiagonal;                               // Temporary array for constructing K_u_diagonal etc. (size:NN)
real *K_1;                                        // Global stiffness matrix
                                                  // K_1 will be used for K_uu, K_vv, K_ww (size:NNZ)

double **Ke_uv, **Ke_uw;                          // Elemental stiffness matrices for constructing. (size:NENvxNENv)
double **Ke_vu, **Ke_vw;                          // K_uv, K_uw, K_vu, K_vw, K_wu, K_wv
double **Ke_wu, **Ke_wv;                          // 
double **Ke_uv_add, **Ke_uw_add, **Ke_vw_add;     // For constructing elemental stiffness matrices. (size:NENvxNENv)

real *K_uv, *K_uw;                                // Global stiffness matrices for
real *K_vu, *K_vw;                                // K_uv, K_uw, K_vu, K_vw, K_wu, K_wv (size:NNZ)
real *K_wu, *K_wv;                                // 

double **Cx_elemental, **Cy_elemental, **Cz_elemental;              // Elemental stiffness matrices for constructing Cx, Cy, Cz. (size:NENvxNENp)
double **CxT_elemental, **CyT_elemental, **CzT_elemental;           // Elemental stiffness matrices for constructing CxT, CyT, CzT. (size:NENvxNENp)
double **Cx_elemental_add, **Cy_elemental_add, **Cz_elemental_add;  // For constructing elemental stiffness matrices. (size:NENvxNENp)

real *Cx, *Cy, *Cz;        // Pressure gradient operators. (size:NNZ)
real *CxT, *CyT, *CzT;     // Velocity divergence operators, transposes of Cx, Cy and Cz. (size:NNZ)

real *val_deltaP;   // Values of the nonzeros of the pressure correction equations LHS.
real *F_deltaP;     // Stores the values of pressure correction equations RHS.
int *row_deltaP;    // Rows of the nonzeros of the pressure correction equations LHS.
int *col_deltaP;    // Columns of the nonzeros of the pressure correction equations LHS.

int iter;                // Counts number of iterations
int phase;               // Defines on which dimension calculations takes place 
int STEP;                // Shows on which step calculations takes place
                         // On reference paper(Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian)
                         // nonlinear iterations are divied to three steps 
int vectorOperationNo;   // Defines which vector operations will be done 
double alpha[4];         // Relaxation factors
                         // alpha[0]: Relaxation factor for u velocities
                         // alpha[1]: Relaxation factor for v velocities
                         // alpha[2]: Relaxation factor for w velocities
                         // alpha[3]: Relaxation factor for pressures

// Variables for CUDA operations
int *d_col, *d_row;                  // Variables are needed for GPU calculations(vector operations).
real *d_val, *d_x, *d_r, *d_rTemp;   // Variables are needed for GPU calculations(vector operations).




//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------
void readInput();
void readRestartFile();
void gaussQuad();
void calcShape();
void calcJacobian();
void initGlobalSysVariables();
void calcKeKMap();
void calcPressureGradientOp();
void assemble_pressureGradientOp(int e);
void assemble_pressureGradientOp_map(int e);
void calcFixedK();
void assemble_fixedK(int e); 
void assemble_fixedK_map(int e); 
void calcGlobalSys_p();
void calcGlobalSys_mom();
void assemble_mom(int e);
void assemble_mom_map(int e);
void applyBC();
void applyBC_p();
void applyBC_deltaP();
void vectorProduct();
void solve();
void postProcess();
void writeTecplotFile();
void compressedSparseRowStorage();
double getHighResolutionTime();

// Pressure correction equation solvers
#ifdef CG_CUDA
   extern void CUSP_pC_CUDA_CG();
#endif
#ifdef CR_CUDA
   extern void CUSP_pC_CUDA_CR();
#endif
#ifdef CG_CUSP
   extern void CUSP_pC_CUSP_CG();
#endif
#ifdef CR_CUSP
   extern void CUSP_pC_CUSP_CR();
#endif

// Momentum equation solvers
#ifdef GMRES_CUSP
   extern void CUSP_GMRES();
#endif
#ifdef BiCG_CUSP
   extern void CUSP_BiCG();
#endif





//------------------------------------------------------------------------------
int main()
//------------------------------------------------------------------------------
{
   cout << "\n********************************************************";
   cout << "\n*   Navier-Stokes 3D - Part of CFD with CUDA project   *";
   cout << "\n*                  Segregated Solver                   *";
   cout << "\n*        http://code.google.com/p/cfd-with-cuda        *";
   cout << "\n********************************************************\n\n";

   cout << "The program is started." << endl ;
   
   double Start, End, Start1, End1;   // Used for run time measurement.

   Start1 = getHighResolutionTime();   

   Start = getHighResolutionTime();
   readInput();                       // Read the input file.
   End = getHighResolutionTime();
   printf("Time for Input    = %-.4g seconds.\n", End - Start);
   
   Start = getHighResolutionTime();
   compressedSparseRowStorage();      // Setup variables for CSR storage.
   End = getHighResolutionTime();
   printf("Time for CSR      = %-.4g seconds.\n", End - Start);

   Start = getHighResolutionTime();   
   gaussQuad();                       // Setup variables for Gauss Quadrature integration.
   End = getHighResolutionTime();
   printf("Time for GQ       = %-.4g seconds.\n", End - Start);

   Start = getHighResolutionTime();
   calcShape();                       // Calculate shape functions at GQ points.
   End = getHighResolutionTime();
   printf("Time for Shape    = %-.4g seconds.\n", End - Start);
   
   Start = getHighResolutionTime();
   calcJacobian();                    // Calculate element Jacobians and shape function derivatives at GQ points.
   End = getHighResolutionTime();
   printf("Time for Jacobian = %-.4g seconds.\n", End - Start);

   Start = getHighResolutionTime();   
   calcKeKMap();                      // Calculate elemental to global stiffness matrix's mapping data for all elements. 
   End = getHighResolutionTime();
   printf("Time for KeKMap   = %-.4g seconds.\n", End - Start);
   
   Start = getHighResolutionTime();
   initGlobalSysVariables();          // Initialize global system variables before solution.
   End = getHighResolutionTime();
   printf("Time for InitVar  = %-.4g seconds.\n", End - Start);
   
   Start = getHighResolutionTime();
   calcPressureGradientOp();          // Calculate presurre gradient operators (Cx, Cy, Cz, CxT, CyT, CzT)
   End = getHighResolutionTime();
   printf("Time for C vecs   = %-.4g seconds.\n", End - Start);
   
   Start = getHighResolutionTime();
   calcFixedK();                      // Calculates K_uv, K_uw, K_vu, K_vw, K_wu, K_wv
   End = getHighResolutionTime();
   printf("Time for K vecs   = %-.4g seconds.\n", End - Start);   
   
   Start = getHighResolutionTime();
   solve();                           // Do nonlinear iterations 
   End = getHighResolutionTime();
   printf("Time for Iter's   = %-.4g seconds.\n", End - Start);   

   Start = getHighResolutionTime();
   postProcess();                     // Write calculated unknows to a file. 
   End = getHighResolutionTime();
   printf("Time for Post Pr. = %-.4g seconds.\n", End - Start); 
   
   End1 = getHighResolutionTime();

   printf("Total Time        = %-.4g seconds.\n", End1 - Start1);

   cout << endl << "The program is terminated successfully.\nPress a key to close this window...\n";

   return 0;

} // End of function main()




//------------------------------------------------------------------------------
void readInput()
//------------------------------------------------------------------------------
{
   // Read the input file with INP extension.
   
   string dummy, dummy2, dummy4, dummy5;
   int dummy3, i, j;

   problemFile.open(problemName.c_str(), ios::in);   // This is file called problemName.txt
   problemFile >> whichProblem;                      // This is used to construct input file's name.
   problemFile.close();
   
   meshfile.open((whichProblem + inputExtension).c_str(), ios::in);
     
   meshfile.ignore(256, '\n'); // Read and ignore the line
   meshfile.ignore(256, '\n'); // Read and ignore the line
   
   meshfile >> dummy >> dummy2 >> eType;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> NE;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> NCN;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> NN;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> NENv;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> NENp;   
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> NGP;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> nonlinearIterMax;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> nonlinearTol;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> solverIterMax;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> solverTol;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> alpha[0] >>  alpha[1] >> alpha[2] >> alpha[3];
   meshfile.ignore(256, '\n'); // Ignore the rest of the line  
   
   meshfile >> dummy >> dummy2 >> nDATiter;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line   
   
   meshfile >> dummy >> dummy2 >> isRestart;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line   
   
   meshfile >> dummy >> dummy2 >> density;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> viscosity;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> fx;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   meshfile >> dummy >> dummy2 >> fy;
   
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile.ignore(256, '\n'); // Read and ignore the line
   meshfile.ignore(256, '\n'); // Read and ignore the line  
   
   NEU = 3*NENv + NENp;    // Number of elemental unknowns.
   Ndof = 3*NN + NCN;      // Total number of unknowns in the problem (disregarding the BCs).
                           // Assumes that velocities and pressures are stored at all mesh nodes.

   // Read node coordinates
   coord = new double*[NN];
   
   if (eType == 2 || eType == 1) {   // These are 2D element types. Not used in this 3D code
      for (i = 0; i < NN; i++) {
         coord[i] = new double[2];
      }
   	
      for (i=0; i<NN; i++){
         meshfile >> dummy3 >> coord[i][0] >> coord[i][1] ;
         meshfile.ignore(256, '\n'); // Ignore the rest of the line    
      }  
   } else {                          // 3D elements
      for (i = 0; i < NN; i++) {
         coord[i] = new double[3];
      }
   	
      for (i=0; i<NN; i++){
         meshfile >> dummy3 >> coord[i][0] >> coord[i][1] >> coord[i][2] ;
         meshfile.ignore(256, '\n'); // Ignore the rest of the line    
      }  
   }
   
   meshfile.ignore(256, '\n'); // Read and ignore the line
   meshfile.ignore(256, '\n'); // Read and ignore the line 


   // Read element connectivity, i.e. LtoG
   LtoG = new int*[NE];

   for (i=0; i<NE; i++) {
      LtoG[i] = new int[NENv];
   }

   for (int e = 0; e<NE; e++){
      meshfile >> dummy3;
      for (i = 0; i<NENv; i++){
         meshfile >> LtoG[e][i];
      }
      meshfile.ignore(256, '\n'); // Read and ignore the line 
   } 

   meshfile.ignore(256, '\n'); // Read and ignore the line
   meshfile.ignore(256, '\n'); // Read and ignore the line 


   // Read number of different BC types and details of each BC
   meshfile >> dummy >> dummy2 >> nBC;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
      
   BCtype = new double[nBC];
   BCstrings = new double*[nBC];
   for (i=0; i<nBC; i++) {
      BCstrings[i] = new double[3];
   }   
    
   for (i = 0; i<nBC-1; i++){
      meshfile >> dummy >> BCtype[i] >> dummy2 >> dummy3 >> BCstrings[i][0] >> dummy4 >> BCstrings[i][1] >> dummy5 >> BCstrings[i][2];
      meshfile.ignore(256, '\n'); // Ignore the rest of the line
   }
   meshfile >> dummy >> BCtype[2]  >> dummy >> BCstrings[i][0];
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   // Read EBC data 
    
   meshfile.ignore(256, '\n'); // Read and ignore the line 
   meshfile >> dummy >> dummy2 >> nVelNodes;

   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile >> dummy >> dummy2 >> nPressureNodes;

   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
      
   if (nVelNodes!=0){
      velNodes = new int*[nVelNodes];
      for (i = 0; i < nVelNodes; i++){
         velNodes[i] = new int[2];
      }
      for (i = 0; i < nVelNodes; i++){
         meshfile >> velNodes[i][0] >> velNodes[i][1];   
         meshfile.ignore(256, '\n'); // Ignore the rest of the line
      }
   }

   meshfile.ignore(256, '\n'); // Ignore the rest of the line  
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   
   if (nPressureNodes!=0){
      pressureNodes = new int*[nPressureNodes];
      for (i = 0; i < nPressureNodes; i++){
         pressureNodes[i] = new int[2];
      }
      for (i = 0; i < nPressureNodes; i++){
         meshfile >> pressureNodes[i][0] >> pressureNodes[i][1];   
         meshfile.ignore(256, '\n'); // Ignore the rest of the line
      }
   }
   
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile >> dummy >> dummy2 >> nMonitorPoints;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile.ignore(256, '\n'); // Ignore the rest of the line


   // Read monitor point part and find the nodes that are closest to the monitored coordinates. 
   double distance, minDistance;
   
   if (nMonitorPoints!=0){
      monitorNodes = new int[nMonitorPoints];
      monitorPoints = new double*[nMonitorPoints];
      for (i = 0; i < nMonitorPoints; i++){
         monitorPoints[i] = new double[3];
      }
      for (i = 0; i < nMonitorPoints; i++){
         meshfile >> dummy >> monitorPoints[i][0] >> monitorPoints[i][1] >> monitorPoints[i][2];
         meshfile.ignore(256, '\n'); // Ignore the rest of the line
         minDistance = 1000.0;
         for (j=0; j<NN; j++){
            distance = sqrt((coord[j][0]-monitorPoints[i][0])*(coord[j][0]-monitorPoints[i][0])+
                                   (coord[j][1]-monitorPoints[i][1])*(coord[j][1]-monitorPoints[i][1])+
                                   (coord[j][2]-monitorPoints[i][2])*(coord[j][2]-monitorPoints[i][2]));
            if (distance < minDistance) {
               minDistance = distance;
               monitorNodes[i] = j;
            }
         }
      }
   }
    
   meshfile.close();

} // End of function readInput()




//------------------------------------------------------------------------------
void compressedSparseRowStorage()
//------------------------------------------------------------------------------
{
   // Creates rowStarts and col vectors of CSR storage scheme. Allocates memory
   // for the val vector of CSR storage scheme.
   
   int i, j, k, m, x, y, valGtoL, check, temp, *checkCol, noOfColGtoL, *GtoLCounter;

   GtoL = new int*[NN];         // Stores the elements that are connected to each node.
   noOfColGtoL = 50;            // It is assumed that max 50 elements are connected to a node.
   GtoLCounter = new int[NN];   // Counts the elements that are connected to each node.
   
   for (i=0; i<NN; i++) {     
      GtoL[i] = new int[noOfColGtoL];   
   }

   // Initialize GtoL and GtoLcounter to -1 and 0.
   // Because we don't know a node presents in how many elements, we fill GtoL with "-1".
   // After filling GtoL with element numbers, "-1"s will be stop criteria.
   for(i=0; i<NN; i++) {
      for(j=0; j<noOfColGtoL; j++) {
         GtoL[i][j] = -1;
      }
      GtoLCounter[i] = 0;
   }
   
   // Determine GtoL and GtoLcounter.
   for (i=0; i<NE; i++) {
      for(j=0; j<NENv; j++) {
         GtoL[ LtoG[i][j] ][ GtoLCounter[LtoG[i][j]] ] = i;
         GtoLCounter[ LtoG[i][j] ] += 1;
      }
   }
   
   delete[] GtoLCounter;


   // Find size of the col vector. Create rowStarts & rowStartsSmall
                                    // Stores the number of nonzeros at each rows of [K]
   rowStartsSmall = new int[NN+1];  // rowStarts for a small part of K(only for "u" velocity in another words 1/16 of the K(if NENv==NENp)) 
   checkCol = new int[1000];        // For checking the non zero column overlaps.
                                    // Stores non-zero column number for rows (must be a large enough value)
   rowStartsSmall[0] = 0;

   for(i=0; i<NN; i++) {
      NNZ = 0;
      
      if(GtoL[i][0] == -1) {
         NNZ = 1;
      } else {
      
         for(k=0; k<1000; k++) {    // Prepare checkCol for new row
            checkCol[k] = -1;
         }
      
         for(j=0; j<noOfColGtoL; j++) {
            valGtoL = GtoL[i][j];
            if(valGtoL != -1) {
               for(x=0; x<NENp; x++) {
                  check = 1;         // For checking if column overlap occurs or not
                  for(y=0; y<NNZ; y++) {
                     if(checkCol[y] == (LtoG[valGtoL][x])) {   // This column was already created
                        check = 0;
                     }
                  }
                  if (check) {
                     checkCol[NNZ]=(LtoG[valGtoL][x]);         // Adding new non zero number to checkCol
                     NNZ++;
                  }
               }
            }
         }

      }     
      rowStartsSmall[i+1] = NNZ + rowStartsSmall[i];
   } // End node loop
   
   delete[] checkCol;

   // Create col & colSmall

   colSmall = new int[rowStartsSmall[NN]];  // col for a part of K(only for "u" velocity in another words 1/16 of the K(if NENv==NENp))  TODO: 1/16 hexahedral elemanlar icin sadece.

   for(i=0; i<NN; i++) {
      NNZ = 0;

      if(GtoL[i][0] == -1) {
         colSmall[rowStartsSmall[i]] = i;
      } else {
      
         for(j=0; j<noOfColGtoL; j++) {
            valGtoL = GtoL[i][j];
            if(valGtoL != -1) {
               for(x=0; x<NENp; x++) {
                  check = 1;
                  for(y=0; y<NNZ; y++) {
                     if(colSmall[rowStartsSmall[i]+y] == (LtoG[valGtoL][x])) {   // For checking if column overlap occurs or not
                        check = 0;
                     }
                  }
                  if (check) {
                     colSmall[rowStartsSmall[i]+NNZ] = (LtoG[valGtoL][x]);
                     NNZ++;
                  }
               }
            }   
         }
         
         for(k=1; k<NNZ; k++) {           // Sorting the column vector values
            for(m=1; m<NNZ; m++) {        // For each row from smaller to bigger
               if(colSmall[rowStartsSmall[i]+m] < colSmall[rowStartsSmall[i]+m-1]) {
                  temp = colSmall[rowStartsSmall[i]+m];
                  colSmall[rowStartsSmall[i]+m] = colSmall[rowStartsSmall[i]+m-1];
                  colSmall[rowStartsSmall[i]+m-1] = temp;
               }
            }   
         }

      }      
   }  

   NNZ = rowStartsSmall[NN];
 
   // Create CSR vectors for diagonal matrix
   // TODO: will be unnecessary after rearranging the matrix-vector operations at GPU 
   rowStartsDiagonal = new int[NN+1];
   for(i=0; i<=NN; i++) {
      rowStartsDiagonal[i] = i;     
   }
   
   colDiagonal = new int[rowStartsDiagonal[NN]];
   for(i=0; i<NN; i++) {
      colDiagonal[i] = i;
   }
   
   for (i = 0; i<NN; i++) {
      delete[] GtoL[i];
   }
   delete[] GtoL;
  
} // End of function compressedSparseRowStorage()




//------------------------------------------------------------------------------
void gaussQuad()
//------------------------------------------------------------------------------
{
   // Generates the NGP-point Gauss quadrature points and weights.

   if (eType == 3) {      // 3D Hexahedron element
      
      // Allocate arrays for Gauss quadrature points and weights
      GQpoint = new double*[NGP];
      for (int i=0; i<NGP; i++) {
         GQpoint[i] = new double[3];
      }
      GQweight = new double[NGP];

      GQpoint[0][0] = -sqrt(1.0/3);   GQpoint[0][1] = -sqrt(1.0/3);   GQpoint[0][2] = -sqrt(1.0/3);
      GQpoint[1][0] = sqrt(1.0/3);    GQpoint[1][1] = -sqrt(1.0/3);   GQpoint[1][2] = -sqrt(1.0/3);
      GQpoint[2][0] = sqrt(1.0/3);    GQpoint[2][1] = sqrt(1.0/3);    GQpoint[2][2] = -sqrt(1.0/3);
      GQpoint[3][0] = -sqrt(1.0/3);   GQpoint[3][1] = sqrt(1.0/3);    GQpoint[3][2] = -sqrt(1.0/3);
      GQpoint[4][0] = -sqrt(1.0/3);   GQpoint[4][1] = -sqrt(1.0/3);   GQpoint[4][2] = sqrt(1.0/3);
      GQpoint[5][0] = sqrt(1.0/3);    GQpoint[5][1] = -sqrt(1.0/3);   GQpoint[5][2] = sqrt(1.0/3);
      GQpoint[6][0] = sqrt(1.0/3);    GQpoint[6][1] = sqrt(1.0/3);    GQpoint[6][2] = sqrt(1.0/3);
      GQpoint[7][0] = -sqrt(1.0/3);   GQpoint[7][1] = sqrt(1.0/3);    GQpoint[7][2] = sqrt(1.0/3);
      GQweight[0] = 1.0;
      GQweight[1] = 1.0;
      GQweight[2] = 1.0;
      GQweight[3] = 1.0;
      GQweight[4] = 1.0;
      GQweight[5] = 1.0;
      GQweight[6] = 1.0;
      GQweight[7] = 1.0;
       
   } else if (eType == 4) {      // 3D Tetrahedron element

      // Allocate arrays for Gauss quadrature points and weights
      GQpoint = new double*[NGP];
      for (int i=0; i<NGP; i++) {
         GQpoint[i] = new double[3];
      }
      GQweight = new double[NGP];
   
      GQpoint[0][0] = 0.58541020;   GQpoint[0][1] = 0.13819660;   GQpoint[0][2] = 0.13819660;
      GQpoint[1][0] = 0.13819660;   GQpoint[1][1] = 0.58541020;   GQpoint[1][2] = 0.13819660;
      GQpoint[2][0] = 0.13819660;   GQpoint[2][1] = 0.13819660;   GQpoint[2][2] = 0.58541020;
      GQpoint[3][0] = 0.13819660;   GQpoint[3][1] = 0.13819660;   GQpoint[3][2] = 0.13819660;
      GQweight[0] = 1.0/24;
      GQweight[1] = 1.0/24;
      GQweight[2] = 1.0/24;
      GQweight[3] = 1.0/24;
   }

} // End of function gaussQuad()




//------------------------------------------------------------------------------
void calcShape()
//------------------------------------------------------------------------------
{
   // Calculates the values of the shape functions and their derivatives with
   // respect to ksi, eta and zeta at GQ points.

   double ksi, eta, zeta;
   
   if (eType == 3) {     // 3D Hexahedron element

      if (NENp == 8) {
         Sp = new double*[8];
         for (int i=0; i<NGP; i++) {
            Sp[i] = new double[NGP];
         }
 
         DSp = new double **[3];
         for (int i=0; i<3; i++) {
            DSp[i] = new double *[8];
            for (int j=0; j<8; j++) {
               DSp[i][j] = new double[NGP];
            }
         }
 
         for (int k = 0; k<NGP; k++) {
            ksi  = GQpoint[k][0];
            eta  = GQpoint[k][1];
            zeta = GQpoint[k][2];
            
            Sp[0][k] = 0.125*(1-ksi)*(1-eta)*(1-zeta);
            Sp[1][k] = 0.125*(1+ksi)*(1-eta)*(1-zeta);
            Sp[2][k] = 0.125*(1+ksi)*(1+eta)*(1-zeta);
            Sp[3][k] = 0.125*(1-ksi)*(1+eta)*(1-zeta);
            Sp[4][k] = 0.125*(1-ksi)*(1-eta)*(1+zeta);
            Sp[5][k] = 0.125*(1+ksi)*(1-eta)*(1+zeta);
            Sp[6][k] = 0.125*(1+ksi)*(1+eta)*(1+zeta);
            Sp[7][k] = 0.125*(1-ksi)*(1+eta)*(1+zeta);

            DSp[0][0][k] = -0.125*(1-eta)*(1-zeta);  // ksi derivative of the 1st shape function at k-th GQ point.
            DSp[1][0][k] = -0.125*(1-ksi)*(1-zeta);  // eta derivative of the 1st shape function at k-th GQ point.
            DSp[2][0][k] = -0.125*(1-ksi)*(1-eta);   // zeta derivative of the 1st shape function at k-th GQ point.
            DSp[0][1][k] =  0.125*(1-eta)*(1-zeta);
            DSp[1][1][k] = -0.125*(1+ksi)*(1-zeta);
            DSp[2][1][k] = -0.125*(1+ksi)*(1-eta);
            DSp[0][2][k] =  0.125*(1+eta)*(1-zeta);
            DSp[1][2][k] =  0.125*(1+ksi)*(1-zeta);
            DSp[2][2][k] = -0.125*(1+ksi)*(1+eta);
            DSp[0][3][k] = -0.125*(1+eta)*(1-zeta);
            DSp[1][3][k] =  0.125*(1-ksi)*(1-zeta);
            DSp[2][3][k] = -0.125*(1-ksi)*(1+eta);
            DSp[0][4][k] = -0.125*(1-eta)*(1+zeta);
            DSp[1][4][k] = -0.125*(1-ksi)*(1+zeta);
            DSp[2][4][k] =  0.125*(1-ksi)*(1-eta);
            DSp[0][5][k] =  0.125*(1-eta)*(1+zeta);
            DSp[1][5][k] = -0.125*(1+ksi)*(1+zeta);
            DSp[2][5][k] =  0.125*(1+ksi)*(1-eta);
            DSp[0][6][k] =  0.125*(1+eta)*(1+zeta);
            DSp[1][6][k] =  0.125*(1+ksi)*(1+zeta);
            DSp[2][6][k] =  0.125*(1+ksi)*(1+eta);
            DSp[0][7][k] = -0.125*(1+eta)*(1+zeta);
            DSp[1][7][k] =  0.125*(1-ksi)*(1+zeta);
            DSp[2][7][k] =  0.125*(1-ksi)*(1+eta);
         }
      }
      
      if (NENv == 8) {
         Sv = new double*[8];
         for (int i=0; i<NGP; i++) {
            Sv[i] = new double[NGP];
         }
      
         DSv = new double **[3];
         for (int i=0; i<3; i++) {
            DSv[i] = new double *[8];
            for (int j=0; j<8; j++) {
               DSv[i][j] = new double[NGP];
            }
         }

         for (int i=0; i<NGP; i++) {
            Sv[i] = Sp[i];
         }

         for (int i=0; i<3; i++) {
            for (int j=0; j<8; j++) {
               DSv[i][j] = DSp[i][j];
            }
         }
      }
   
   } else if (eType == 4) {     // 3D Tetrahedron element

      if (NENp == 4) {
         Sp = new double*[4];
         for (int i=0; i<NGP; i++) {
            Sp[i] = new double[NGP];
         }

         DSp = new double **[3];
         for (int i=0; i<3; i++) {
            DSp[i] = new double *[4];
            for (int j=0; j<4; j++) {
               DSp[i][j] = new double[NGP];
            }
         }

         for (int k = 0; k<NGP; k++) {
            ksi = GQpoint[k][0];
            eta = GQpoint[k][1];
            zeta = GQpoint[k][2];

            Sp[0][k] = 1-ksi-eta-zeta;
            Sp[1][k] = ksi;
            Sp[2][k] = eta;
            Sp[3][k] = zeta;

            DSp[0][0][k] = -1;   // ksi derivative of the 1st shape function at k-th GQ point.
            DSp[1][0][k] = -1;   // eta derivative of the 1st shape function at k-th GQ point.
            DSp[2][0][k] = -1;   // zeta derivative of the 1st shape function at k-th GQ point.
            DSp[0][1][k] = 1;
            DSp[1][1][k] = 0;
            DSp[2][1][k] = 0;
            DSp[0][2][k] = 0;
            DSp[1][2][k] = 1;
            DSp[2][2][k] = 0;
            DSp[0][3][k] = 0;
            DSp[1][3][k] = 0;
            DSp[2][3][k] = 1;
         }
      }   

      if (NENv == 4) {
         Sv = new double*[4];
         for (int i=0; i<NGP; i++) {
            Sv[i] = new double[NGP];
         }

         DSv = new double **[3];
         for (int i=0; i<3; i++) {
            DSv[i] = new double *[4];
            for (int j=0; j<4; j++) {
               DSv[i][j] = new double[NGP];
            }
         }

         for (int i=0; i<NGP; i++) {
            Sv[i] = Sp[i];
         }

         for (int i=0; i<3; i++) {
            for (int j=0; j<4; j++) {
               DSv[i][j] = DSp[i][j];
            }
         }
      }

   }  // Endif eType

   for (int i = 0; i<NGP; i++) {
      delete[] GQpoint[i];
   }
   delete[] GQpoint;
      
} // End of function calcShape()




//------------------------------------------------------------------------------
void calcJacobian()
//------------------------------------------------------------------------------
{
   // Evaluates and stores determinant of the Jacobian matrix of each element at
   // all GQ points and derivatives of shape functions wrt global coordinates
   // x, y and z.

   int e, i, j, k, m, iG; 
   double **e_coord;
   double **Jacob, **invJacob;
   double temp;
   
   if (eType == 3 || eType == 4) {
   
      e_coord = new double*[NENp];
   
      for (i=0; i<NENp; i++) {
         e_coord[i] = new double[3];
      }
   
      Jacob = new double*[3];
      invJacob = new double*[3];
      for (i=0; i<3; i++) {
         Jacob[i] = new double[3];
         invJacob[i] = new double[3];
      }
      
      detJacob = new double*[NE];
      for (i=0; i<NE; i++) {
         detJacob[i] = new double[NGP];
      }
      
      gDSp = new double***[NE];

      for (i=0; i<NE; i++) {
         gDSp[i] = new double**[3];
         for(j=0; j<3; j++) {
            gDSp[i][j] = new double*[NENp];     
            for(k=0; k<NENp; k++) {
               gDSp[i][j][k] = new double[NGP];
            }
         }	
      }
      
      gDSv = new double***[NE];

      for (i=0; i<NE; i++) {
         gDSv[i] = new double**[3];
         for(j=0; j<3; j++) {
            gDSv[i][j] = new double*[NENv];
            for(k=0; k<NENv; k++) {
               gDSv[i][j][k] = new double[NGP];
            }
         }	
      }      
   
      for (e = 0; e<NE; e++){
         // To calculate Jacobian matrix of an element we need e_coord matrix of
         // size NENx3. Each row of it stores x, y & z coordinates of the nodes of
         // an element.
         for (i = 0; i<NENp; i++){
            iG = LtoG[e][i];
            e_coord[i][0] = coord[iG][0]; 
            e_coord[i][1] = coord[iG][1];
            e_coord[i][2] = coord[iG][2];
         }
      
         // For each GQ point calculate 3x3 Jacobian matrix, its inverse and its
         // determinant. Also calculate derivatives of shape functions wrt global
         // coordinates x, y & z. These are the derivatives that we'll use in
         // evaluating K and F integrals. 
   	
         for (k = 0; k<NGP; k++) {
            for (i = 0; i<3; i++) {
               for (j = 0; j<3; j++) {
                  temp = 0;
                  for (m = 0; m<NENp; m++) {
                     temp = DSp[i][m][k] * e_coord[m][j]+temp;
                  }
                  Jacob[i][j] = temp;
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
            
            for (i = 0; i<3; i++){
               for (j = 0; j<3; j++){
                  invJacob[i][j] = invJacob[i][j] / detJacob[e][k];
               }    
            }
         
            for (i = 0; i<3; i++){
               for (j = 0; j<NENp; j++) {
                  temp = 0;
                  for (m = 0; m<3; m++) { 
                     temp = invJacob[i][m] * DSp[m][j][k] + temp;
                  }
                  gDSp[e][i][j][k] = temp;
                  gDSv[e][i][j][k] = gDSp[e][i][j][k];
               }
            }
         }   // End GQ loop
      }   // End element loop
   }   // Endif eType

   // Deallocate unnecessary variables

   for (int i = 0; i<3; i++) {
      delete[] Jacob[i];
   }
   delete[] Jacob;

   for (int i = 0; i<3; i++) {
      delete[] invJacob[i];
   }
   delete[] invJacob;
   
   for (int i = 0; i<NENp; i++) {
      delete[] e_coord[i];
   }
   delete[] e_coord;
   
} // End of function calcJacobian()




//------------------------------------------------------------------------------
void calcKeKMap()
//------------------------------------------------------------------------------
{
   // This function calculates the KeKMap.
   // KeKMap keeps elemental to global stiffness matrix's mapping data for all elements.


   //-----------Ke to K map selection----------------------------
   //KeKMap : costs memory([NE][NENv][NENv]*4byte), runs faster. (default)
   //KeKMapSmall : negligible memory, runs slower. Steps to use;
   //               (1) Comment out KeKMapUSE parts
   //               (2) Uncomment KeKMapSmallUSE parts
   //------------------------------------------------------------   
   
   //-----KeKMapUSE-----
   int *eLtoG, loc, colCounter, k, e, i, j;   
   
   KeKMap = new int**[NE];                //Keeps elemental to global stiffness matrix's mapping data
   for(e=0; e<NE; e++) {                  //It costs some memory([NE][NENv][NENv]*4byte) but makes assembly function runs faster.
      KeKMap[e] = new int*[NENv];
      for(j=0; j<NENv; j++) {
         KeKMap[e][j] = new int[NENv];
      }
   }
   
   eLtoG = new int[NENv];           // elemental LtoG data    
   
   for(e=0; e<NE; e++) {

      for(k=0; k<NENv; k++) {
         eLtoG[k] = (LtoG[e][k]);      // Takes node data from LtoG
      } 

      for(i=0; i<NENv; i++) {
         for(j=0; j<NENv; j++) {
            colCounter=0;
            for(loc=rowStartsSmall[eLtoG[i]]; loc<rowStartsSmall[eLtoG[i]+1]; loc++) {  // loc is the location of the col vector(col[x], loc=x) 
               if(colSmall[loc] == eLtoG[j]) {                                         // Selection process of the KeKMapSmall data from the col vector
                  KeKMap[e][i][j] = colCounter; 
                  break;
               }
               colCounter++;
            }
         }
      }   
   }   // End of element loop 
   
   delete[] eLtoG;
   //-----KeKMapUSE-----

} // End of function calcKeKMap()




//------------------------------------------------------------------------------
void initGlobalSysVariables()
//------------------------------------------------------------------------------
{
   // Allocate memory for stiffness matrices and force vectors.
   
   int i;
   
   K_1 = new real[NNZ];   // K_1 will be used for K_uu, K_vv, K_ww
   
   K_uv = new real[NNZ];
   K_uw = new real[NNZ];
   K_vu = new real[NNZ];
   K_vw = new real[NNZ];
   K_wu = new real[NNZ];
   K_wv = new real[NNZ];
   
   Cx = new real[NNZ];
   Cy = new real[NNZ];
   Cz = new real[NNZ];

   CxT = new real[NNZ];   // Transpose of Cx
   CyT = new real[NNZ];   // Transpose of Cy
   CzT = new real[NNZ];   // Transpose of Cz

   for (i=0; i<NNZ; i++) {
      K_1[i] = 0.0;
      K_uv[i] = 0.0;
      K_uw[i] = 0.0;
      K_vu[i] = 0.0;
      K_vw[i] = 0.0;
      K_wu[i] = 0.0;
      K_wv[i] = 0.0;
      Cx[i] = 0.0;
      Cy[i] = 0.0;
      Cz[i] = 0.0;
      CxT[i] = 0.0;
      CyT[i] = 0.0;
      CzT[i] = 0.0;
   }
   
   Cx_elemental = new double*[NENv];
   Cy_elemental = new double*[NENv];
   Cz_elemental = new double*[NENv];
   CxT_elemental = new double*[NENv];
   CyT_elemental = new double*[NENv];
   CzT_elemental = new double*[NENv];

   Cx_elemental_add = new double*[NENv];
   Cy_elemental_add = new double*[NENv];
   Cz_elemental_add = new double*[NENv];
   
   for (i=0; i<NENv; i++) {
      Cx_elemental[i] = new double[NENv];
      Cy_elemental[i] = new double[NENv];
      Cz_elemental[i] = new double[NENv];
      CxT_elemental[i] = new double[NENv];
      CyT_elemental[i] = new double[NENv];
      CzT_elemental[i] = new double[NENv];

      Cx_elemental_add[i] = new double[NENv];
      Cy_elemental_add[i] = new double[NENv];
      Cz_elemental_add[i] = new double[NENv];
   }  
   
   
   Ke_1 = new double*[NENv];
   Ke_2 = new double*[NENv];
   Ke_3 = new double*[NENv];
   
   Ke_1_add = new double*[NENv];
   Ke_2_add = new double*[NENv];
   Ke_3_add = new double*[NENv];   

   for (i=0; i<NENv; i++) {
      Ke_1[i] = new double[NENv];
      Ke_2[i] = new double[NENv];
      Ke_3[i] = new double[NENv];      

      Ke_1_add[i] = new double[NENv];
      Ke_2_add[i] = new double[NENv];
      Ke_3_add[i] = new double[NENv];      
   }
   
   
   Ke_uv = new double*[NENv];
   Ke_uw = new double*[NENv];
   Ke_vu = new double*[NENv];
   Ke_vw = new double*[NENv];
   Ke_wu = new double*[NENv];
   Ke_wv = new double*[NENv];
   
   Ke_uv_add = new double*[NENv];
   Ke_uw_add = new double*[NENv];
   Ke_vw_add = new double*[NENv];   
   
   for (i=0; i<NENv; i++) {
      Ke_uv[i] = new double[NENv];
      Ke_uw[i] = new double[NENv];
      Ke_vu[i] = new double[NENv];
      Ke_vw[i] = new double[NENv];
      Ke_wu[i] = new double[NENv];
      Ke_wv[i] = new double[NENv];
      
      Ke_uv_add[i] = new double[NENv];
      Ke_uw_add[i] = new double[NENv];
      Ke_vw_add[i] = new double[NENv];         
   }
   
   uNodal = new double[NENv];
   vNodal = new double[NENv];
   wNodal = new double[NENv];

   Du0 = new double[3];
   Dv0 = new double[3];
   Dw0 = new double[3];   
   
   u = new real[NN];
   v = new real[NN];
   w = new real[NN];
   u_temp = new real[NN];
   v_temp = new real[NN];
   w_temp = new real[NN];   
   p = new real[NN];  
   delta_p = new real[NN];

   K_u_diagonal = new real[NN];
   K_v_diagonal = new real[NN];
   K_w_diagonal = new real[NN]; 
   tempDiagonal = new real[NN];  
   
   // Initial guesses for unknowns 
   for (i=0; i<NN; i++) {
      u[i] = 0.0;
      v[i] = 0.0;
      w[i] = 0.0;
      u_temp[i] = 0.0;
      v_temp[i] = 0.0;
      w_temp[i] = 0.0;      
      p[i] = 0.0;
      delta_p[i] = 0.0;
      K_u_diagonal[i] = 0.0;
      K_v_diagonal[i] = 0.0;
      K_w_diagonal[i] = 0.0;   
      tempDiagonal[i] = 0.0;      
   } 
   
   F = new real[NN];   // RHS vector for momentum equations
                       // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4e], [4f])  

   for (i=0; i<NN; i++) {
      F[i] = 0;
   }
   
   
   if(isRestart) {
      readRestartFile();
   }
} // End of function initGlobalSysVariables()




//------------------------------------------------------------------------------
void calcFixedK()
//------------------------------------------------------------------------------
{
   // Calculates Cx_elemental, Cy_elemental, Cz_elemental one by one for each
   // element and assembles them into the global Cx, Cy, Cz and CxT, CyT, CzT.
   int e, i, j, k, m, n, node;   
   
   for (e = 0; e<NE; e++) {
      // Intitialize Ke and Fe to zero.

      for (i=0; i<NENv; i++) {
         for (j=0; j<NENp; j++) {
            Ke_uv[i][j] = 0.0; 
            Ke_uw[i][j] = 0.0; 
            Ke_vu[i][j] = 0.0; 
            Ke_vw[i][j] = 0.0; 
            Ke_wu[i][j] = 0.0; 
            Ke_wv[i][j] = 0.0;               
         }      
      }  
      
      for (k = 0; k<NGP; k++) {   // Gauss quadrature loop
         
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENp; j++) {
               Ke_uv_add[i][j] = 0.0;
               Ke_uw_add[i][j] = 0.0;
               Ke_vw_add[i][j] = 0.0;
            }         
         }

         for (i=0; i<NENv; i++) {
            for (j=0; j<NENp; j++) {
               Ke_uv_add[i][j] = Ke_uv_add[i][j] + viscosity * gDSv[e][1][i][k] * gDSv[e][0][j][k];
               Ke_uw_add[i][j] = Ke_uw_add[i][j] + viscosity * gDSv[e][2][i][k] * gDSv[e][0][j][k]; 
               Ke_vw_add[i][j] = Ke_vw_add[i][j] + viscosity * gDSv[e][2][i][k] * gDSv[e][1][j][k];
            }         
         }
         
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENp; j++) {               
               Ke_uv[i][j] += Ke_uv_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_uw[i][j] += Ke_uw_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_vw[i][j] += Ke_vw_add[i][j] * detJacob[e][k] * GQweight[k];               
            }    
         }

      }   // End GQ loop
      
      for (i=0; i<NENv; i++) {
         for (j=0; j<NENp; j++) {
            Ke_vu[j][i] = Ke_uv[i][j] ;
            Ke_wu[j][i] = Ke_uw[i][j] ;
            Ke_wv[j][i] = Ke_vw[i][j] ;            
         }
      }
      
      //-----KeKMapSmallUSE-----
      //assemble_fixedK(e);
      //-----KeKMapSmallUSE-----      
      
      //-----KeKMapUSE-----
      assemble_fixedK_map(e); 
      //-----KeKMapUSE-----
   }   // End element loop
   
} // End of function calcFixedK()




//------------------------------------------------------------------------------
void assemble_fixedK(int e)
//------------------------------------------------------------------------------
{
   // Inserts Ke_uv, Ke_uw, Ke_vu, Ke_vw, Ke_wu, Ke_wv into proper locations of
   // the global K_uv, K_uw, K_vu, K_vw, K_wu, K_wv.

   int i, j;

   // Create KeKMapSmall, which stores the mapping between the entries of Ke
   // and val vector of CSR.

   int *eLtoG, loc, colCounter, k;
    
   eLtoG = new int[NENv];           // elemental LtoG data
   for(k=0; k<NENv; k++) {
      eLtoG[k] = (LtoG[e][k]);      // Takes node data from LtoG
   } 

   KeKMapSmall = new int*[NENv];
   for(j=0; j<NENv; j++) {
      KeKMapSmall[j] = new int[NENv];
   }

   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         colCounter=0;
         for(loc=rowStartsSmall[eLtoG[i]]; loc<rowStartsSmall[eLtoG[i]+1]; loc++) {  // loc is the location of the col vector(col[x], loc=x) 
            if(colSmall[loc] == eLtoG[j]) {                                         // Selection process of the KeKMapSmall data from the col vector
               KeKMapSmall[i][j] = colCounter; 
               break;
            }
            colCounter++;
         }
      }
   }
   
   // Creating K_uv, K_uw, K_vu, K_vw, K_wu, K_wv value vectors for sparse storage
   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         K_uv[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_uv[i][j] ;
         K_uw[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_uw[i][j] ;  
         K_vu[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_vu[i][j] ;  
         K_vw[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_vw[i][j] ;  
         K_wu[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_wu[i][j] ;  
         K_wv[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_wv[i][j] ;         
      }
   }

   
   for (int i = 0; i<NENv; i++) {
      delete[] KeKMapSmall[i];
   }
   delete[] KeKMapSmall;

   delete[] eLtoG;

} // End of function assemble_fixedK()




//------------------------------------------------------------------------------
void assemble_fixedK_map(int e)
//------------------------------------------------------------------------------
{
   // Inserts Ke_uv, Ke_uw, Ke_vu, Ke_vw, Ke_wu, Ke_wv into proper locations of
   // the global K_uv, K_uw, K_vu, K_vw, K_wu, K_wv.
   int i, j;

   // Creating K_uv, K_uw, K_vu, K_vw, K_wu, K_wv value vectors for sparse storage
   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         K_uv[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += Ke_uv[i][j] ;
         K_uw[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += Ke_uw[i][j] ;  
         K_vu[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += Ke_vu[i][j] ;  
         K_vw[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += Ke_vw[i][j] ;  
         K_wu[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += Ke_wu[i][j] ;  
         K_wv[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += Ke_wv[i][j] ;         
      }
   }
   
} // End of function assemble_fixedK()




//------------------------------------------------------------------------------
void calcPressureGradientOp()
//------------------------------------------------------------------------------
{
   // Calculates Cx_elemental, Cy_elemental, Cz_elemental one by one for each
   // element and assembles them into the global Cx, Cy, Cz and CxT, CyT, CzT.
   int e, i, j, k, m, n, node;
  
   for (e = 0; e<NE; e++) {
      // Intitialize Ke and Fe to zero.

      for (i=0; i<NENv; i++) {
         for (j=0; j<NENp; j++) {
            Cx_elemental[i][j] = 0;
            Cy_elemental[i][j] = 0;
            Cz_elemental[i][j] = 0;
            CxT_elemental[i][j] = 0;
            CyT_elemental[i][j] = 0;
            CzT_elemental[i][j] = 0;
         }
      }
      
      for (k = 0; k<NGP; k++) {   // Gauss quadrature loop
         
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENp; j++) {
               Cx_elemental_add[i][j] = 0;
               Cy_elemental_add[i][j] = 0;
               Cz_elemental_add[i][j] = 0;
            }
         }

         for (i=0; i<NENv; i++) {
            for (j=0; j<NENp; j++) {
               Cx_elemental_add[i][j] = Cx_elemental_add[i][j] + gDSv[e][0][i][k] * Sp[j][k];
               Cy_elemental_add[i][j] = Cy_elemental_add[i][j] + gDSv[e][1][i][k] * Sp[j][k];
               Cz_elemental_add[i][j] = Cz_elemental_add[i][j] + gDSv[e][2][i][k] * Sp[j][k];
            }
         }
         
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENp; j++) {
               Cx_elemental[i][j] += Cx_elemental_add[i][j] * detJacob[e][k] * GQweight[k];
               Cy_elemental[i][j] += Cy_elemental_add[i][j] * detJacob[e][k] * GQweight[k];
               Cz_elemental[i][j] += Cz_elemental_add[i][j] * detJacob[e][k] * GQweight[k];
            }
         }

      }   // End GQ loop
      
      for (i=0; i<NENv; i++) {
         for (j=0; j<NENp; j++) {
            CxT_elemental[j][i] = Cx_elemental[i][j];
            CyT_elemental[j][i] = Cy_elemental[i][j];
            CzT_elemental[j][i] = Cz_elemental[i][j];
         }
      }
      
      //-----KeKMapSmallUSE-----
      //assemble_pressureGradientOp(e);
      //-----KeKMapSmallUSE-----
      
      //-----KeKMapUSE-----
      assemble_pressureGradientOp_map(e);
      //-----KeKMapUSE-----
   }   // End element loop

} // End of function calcPressureGradientOp()




//------------------------------------------------------------------------------
void assemble_pressureGradientOp(int e)
//------------------------------------------------------------------------------
{
   // Inserts Cx_elemental, Cy_elemental, Cz_elemental, CxT_elemental, CyT_elemental,
   // CzT_elemental into proper locations of the global Cx, Cy, Cz and CxT, CyT, CzT.

   int i, j;

   // Create KeKMapSmall, which stores the mapping between the entries of Ke
   // and val vector of CSR.

   int *eLtoG, loc, colCounter, k;
    
   eLtoG = new int[NENv];           // Elemental LtoG data
   for(k=0; k<NENv; k++) {
      eLtoG[k] = (LtoG[e][k]);      // Takes node data from LtoG
   }

   KeKMapSmall = new int*[NENv];
   for(j=0; j<NENv; j++) {
      KeKMapSmall[j] = new int[NENv];
   }

   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         colCounter=0;
         for(loc=rowStartsSmall[eLtoG[i]]; loc<rowStartsSmall[eLtoG[i]+1]; loc++) {  // loc is the location of the col vector(col[x], loc=x)
            if(colSmall[loc] == eLtoG[j]) {                                          // Selection process of the KeKMapSmall data from the col vector
               KeKMapSmall[i][j] = colCounter;
               break;
            }
            colCounter++;
         }
      }
   }
   
   // Creating Cx, Cy, Cz and CxT, CyT, CzT value vectors for sparse storage
   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         Cx[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Cx_elemental[i][j] ;
         Cy[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Cy_elemental[i][j] ;
         Cz[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Cz_elemental[i][j] ;
         CxT[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += CxT_elemental[i][j] ;
         CyT[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += CyT_elemental[i][j] ;
         CzT[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += CzT_elemental[i][j] ;
      }
   }

   
   for (int i = 0; i<NENv; i++) {
      delete[] KeKMapSmall[i];
   }
   delete[] KeKMapSmall;

   delete[] eLtoG;
   
} // End of function assemble_pressureGradientOp()




//------------------------------------------------------------------------------
void assemble_pressureGradientOp_map(int e)
//------------------------------------------------------------------------------
{
   // Inserts Cx_elemental, Cy_elemental, Cz_elemental, CxT_elemental, CyT_elemental,
   // CzT_elemental into proper locations of the global Cx, Cy, Cz and CxT, CyT, CzT.
   int i, j;

   // Creating Cx, Cy, Cz and CxT, CyT, CzT value vectors for sparse storage
   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         Cx[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += Cx_elemental[i][j] ;
         Cy[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += Cy_elemental[i][j] ;
         Cz[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += Cz_elemental[i][j] ;
         CxT[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += CxT_elemental[i][j] ;
         CyT[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += CyT_elemental[i][j] ;
         CzT[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += CzT_elemental[i][j] ;
      }
   }
} // End of function assemble_pressureGradientOp_map()




//------------------------------------------------------------------------------
void calcGlobalSys_p()
//------------------------------------------------------------------------------
{
   // Calculates Ke and Fe one by one for each element and assembles them into
   // the global K and F.

   int e, i, j, k, m, n, node;  
   
   for(i=0; i<rowStartsDiagonal[NN]; i++) {
      K_u_diagonal[i] = 0;
      K_v_diagonal[i] = 0;  
      K_w_diagonal[i] = 0;
   }

   // Calculating the elemental stiffness matrix(Ke) and force vector(Fe)

   for (e = 0; e<NE; e++) {
      // Intitialize Ke and Fe to zero.

      for (i=0; i<NENv; i++) {
         for (j=0; j<NENv; j++) {
            Ke_1[i][j] = 0;
            Ke_2[i][j] = 0;
            Ke_3[i][j] = 0; 
         }      
      }  
      
      for (i=0; i<NENv; i++) {
         uNodal[i] = u[LtoG[e][i]]; 
         vNodal[i] = v[LtoG[e][i]];
         wNodal[i] = w[LtoG[e][i]];
      }
      
      for (k = 0; k<NGP; k++) {   // Gauss quadrature loop
            
         u0 = 0;
         v0 = 0;
         w0 = 0;

         for (i=0; i<3; i++) {
            Du0[i] = 0;
            Dv0[i] = 0;
            Dw0[i] = 0;
         }
         
         for (i=0; i<NENv; i++) {
            u0 = u0 + Sp[i][k] * uNodal[i];
            v0 = v0 + Sp[i][k] * vNodal[i];
            w0 = w0 + Sp[i][k] * wNodal[i];            
         }
         
         for (i=0; i<3; i++) {
            for (j=0; j<NENv; j++) {
               Du0[i] = Du0[i] + gDSp[e][i][j][k] * uNodal[j];
               Dv0[i] = Dv0[i] + gDSp[e][i][j][k] * vNodal[j];
               Dw0[i] = Dw0[i] + gDSp[e][i][j][k] * wNodal[j];
            }
         }
         
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {
               Ke_1_add[i][j] = 0;
               Ke_2_add[i][j] = 0;
               Ke_3_add[i][j] = 0;               
            }       
         }

         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {
               Ke_1_add[i][j] = Ke_1_add[i][j] + viscosity * (
                     2 * gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                     1 * gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                     1 * gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                     density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);
               Ke_2_add[i][j] = Ke_2_add[i][j] + viscosity * (
                     1 * gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                     2 * gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                     1 * gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                     density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);
               Ke_3_add[i][j] = Ke_3_add[i][j] + viscosity * (
                     1 * gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                     1 * gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                     2 * gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                     density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);                           
            }        
         }
         
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {            
               Ke_1[i][j] += Ke_1_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_2[i][j] += Ke_2_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_3[i][j] += Ke_3_add[i][j] * detJacob[e][k] * GQweight[k];               
            }   
         }

      }   // End GQ loop 

      // Create diagonal matrices

      for (i=0; i<NENv; i++) {
         K_u_diagonal[LtoG[e][i]] = K_u_diagonal[LtoG[e][i]] + Ke_1[i][i];
         K_v_diagonal[LtoG[e][i]] = K_v_diagonal[LtoG[e][i]] + Ke_2[i][i];
         K_w_diagonal[LtoG[e][i]] = K_w_diagonal[LtoG[e][i]] + Ke_3[i][i];         
      }

   }   // End element loop
 
} // End of function calcGlobalSys_p()




//------------------------------------------------------------------------------
void calcGlobalSys_mom()
//------------------------------------------------------------------------------
{
   // Calculates Ke's one by one for each element and assembles them into
   // the global K's.

   int e, i, j, k, m, n, node, valSwitch;
   double Tau;
   real factor[3];
   
   switch (phase) {
      case 0:
         factor[0] = 2.0;
         factor[1] = 1.0;
         factor[2] = 1.0;   
      break;
      case 1:
         factor[0] = 1.0;
         factor[1] = 2.0;
         factor[2] = 1.0;           
      break;
      case 2:
         factor[0] = 1.0;
         factor[1] = 1.0;
         factor[2] = 2.0;     
      break;
   }

   for(i=0; i<NNZ; i++) {
      K_1[i] = 0;
   }
   
   for(i=0; i<rowStartsDiagonal[NN]; i++) {
      tempDiagonal[i] = 0;
   }

   // Calculating the elemental stiffness matrix(Ke) and force vector(Fe)

   for (e = 0; e<NE; e++) {
      // Intitialize Ke and Fe to zero.

      for (i=0; i<NENv; i++) {
         for (j=0; j<NENv; j++) {
            Ke_1[i][j] = 0;
         }   
      }  
      
      for (i=0; i<NENv; i++) {
         uNodal[i] = u[LtoG[e][i]]; 
         vNodal[i] = v[LtoG[e][i]];
         wNodal[i] = w[LtoG[e][i]];
      }
      
      for (k = 0; k<NGP; k++) {   // Gauss quadrature loop
            
         u0 = 0;
         v0 = 0;
         w0 = 0;

         for (i=0; i<3; i++) {
            Du0[i] = 0;
            Dv0[i] = 0;
            Dw0[i] = 0;
         }
         
         for (i=0; i<NENv; i++) {
            u0 = u0 + Sp[i][k] * uNodal[i];
            v0 = v0 + Sp[i][k] * vNodal[i];
            w0 = w0 + Sp[i][k] * wNodal[i];            
         }
         
         for (i=0; i<3; i++) {
            for (j=0; j<NENv; j++) {
               Du0[i] = Du0[i] + gDSp[e][i][j][k] * uNodal[j];
               Dv0[i] = Dv0[i] + gDSp[e][i][j][k] * vNodal[j];
               Dw0[i] = Dw0[i] + gDSp[e][i][j][k] * wNodal[j];
            }
         }
         
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {
               Ke_1_add[i][j] = 0;
            }   
         }

         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {
               Ke_1_add[i][j] = Ke_1_add[i][j] + viscosity * (
                     factor[0] * gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                     factor[1] * gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                     factor[2] * gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                     density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);      
            }      
         }
         
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {            
               Ke_1[i][j] += Ke_1_add[i][j] * detJacob[e][k] * GQweight[k];
            } 
         }

      }   // End GQ loop  

      // Create diagonal matrices

      for (i=0; i<NENv; i++) {
         tempDiagonal[LtoG[e][i]] = tempDiagonal[LtoG[e][i]] + Ke_1[i][i];
         Ke_1[i][i] += (alpha[0]/(1.0-alpha[0]))*Ke_1[i][i];
      }      
      
      //-----KeKMapSmallUSE-----  
      //assemble_mom(e);  // Send Ke_1  for creating K_1
      //-----KeKMapSmallUSE-----  
      
      //-----KeKMapUSE-----  
      assemble_mom_map(e);  // Send Ke_1  for creating K_1
      //-----KeKMapUSE-----       
      
   }   // End element loop
 
} // End of function calcGlobalSys_mom()




//------------------------------------------------------------------------------
void assemble_mom(int e)
//------------------------------------------------------------------------------
{
   // Inserts Ke's into proper locations of K's.

   int i, j;

   // Create KeKMapSmall, which stores the mapping between the entries of Ke
   // and val vector of CSR.

   int *eLtoG, loc, colCounter, k;
    
   eLtoG = new int[NENv];           // elemental LtoG data
   for(k=0; k<NENv; k++) {
      eLtoG[k] = (LtoG[e][k]);      // Takes node data from LtoG
   } 

   KeKMapSmall = new int*[NENv];
   for(j=0; j<NENv; j++) {
      KeKMapSmall[j] = new int[NENv];
   }

   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         colCounter=0;
         for(loc=rowStartsSmall[eLtoG[i]]; loc<rowStartsSmall[eLtoG[i]+1]; loc++) {  // loc is the location of the col vector(col[x], loc=x) 
            if(colSmall[loc] == eLtoG[j]) {                                         // Selection process of the KeKMapSmall data from the col vector
               KeKMapSmall[i][j] = colCounter; 
               break;
            }
            colCounter++;
         }
      }
   }
   
   // Creating K's value vectors
   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         K_1[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_1[i][j] ;  
      }
   }

   
   for (int i = 0; i<NENv; i++) {
      delete[] KeKMapSmall[i];
   }
   delete[] KeKMapSmall;

   delete[] eLtoG;
   
} // End of function assemble_mom()




//------------------------------------------------------------------------------
void assemble_mom_map(int e)
//------------------------------------------------------------------------------
{
   // Inserts Ke's into proper locations of K's.
   int i, j;
   
   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         K_1[ rowStartsSmall[LtoG[e][i]] + KeKMap[e][i][j] ] += Ke_1[i][j] ; 
      }
   }
   
} // End of function assemble_mom_map()




//------------------------------------------------------------------------------
void applyBC()
//------------------------------------------------------------------------------
{
   // For EBCs reduction is not applied. Instead K and F are modified as
   // explained in class, which requires modification of both [K] and {F}.
   // SV values specified for NBCs are added to {F}.
   // At this function modification for velocity EBC's takes place.    

   int i, j, whichBC, node;
   double x, y, z; 
   int loc, colCounter;
   
   // Modify CSR vectors for velocity and wall BCs
   for (i = 0; i<nVelNodes; i++) {
      node = velNodes[i][0];         // Node at which this EBC is specified
   	
      x = coord[node][0];            // May be necessary for BCstring evaluation
      y = coord[node][1];
      z = coord[node][2];

      colCounter = 0;
      for(loc=rowStartsSmall[node]; loc<rowStartsSmall[node+1]; loc++) {  // loc is the location of the col vector(col[x], loc=x)
         if(colSmall[loc] == node) {                                      // Selection process of the KeKMapSmall data from the col vector.
            break;
         }
         colCounter++;
      }

      whichBC = velNodes[i][1]-1;      // Number of the specified BC 
      
      for (j=rowStartsSmall[node]; j<rowStartsSmall[node+1]; j++) {
         K_1[j] = 0.0;
      }
      K_1[ rowStartsSmall[node] + colCounter ] = 1;

      switch (phase) {
         case 0:
            F[node] = BCstrings[whichBC][0];    // Specified value of the PV 
            u[node] = BCstrings[whichBC][0];
            break;
         case 1:
            F[node] = BCstrings[whichBC][1];    // Specified value of the PV  
            v[node] = BCstrings[whichBC][1];
            break;
         case 2:
            F[node] = BCstrings[whichBC][2];    // Specified value of the PV 
            w[node] = BCstrings[whichBC][2];
            break;
      }   
   }   

} // End of function applyBC()




//------------------------------------------------------------------------------
void applyBC_p()
//------------------------------------------------------------------------------
{
   // For EBCs reduction is not applied. Instead K and F are modified.
   // Which requires modification of both [K] and {F}.
   // At this function modification for pressure EBC's takes place. 
   
   int i, j, whichBC, node ;
   double x, y, z;
   int loc, colCounter;

   // Modify CSR vectors for pressure BCs
   for (i = 0; i<nPressureNodes; i++) {
      node = pressureNodes[i][0];         // Node at which this EBC is specified
   	
      x = coord[node][0];                 // May be necessary for BCstring evaluation
      y = coord[node][1];
      z = coord[node][2];

      colCounter = 0;
      for(loc=rowStartsSmall[node]; loc<rowStartsSmall[node+1]; loc++) {  // loc is the location of the col vector(col[x], loc=x)
         if(colSmall[loc] == node) {                                      // Selection process of the KeKMapSmall data from the col vector.
            break;
         }
         colCounter++;
      }
      
      whichBC = pressureNodes[i][1]-1;      // Number of the specified BC   	
      
      // for (j=rowStartsSmall[node]; j<rowStartsSmall[node+1]; j++) {
         // val_f[j] = 0.0;
      // }
      // val_f[ rowStartsSmall[node] + colCounter ] = 1;
      
      p[node] = BCstrings[whichBC][0];    // Specified value of the PV          
   } 

} // End of function applyBC_p()




//------------------------------------------------------------------------------
void applyBC_deltaP()
//------------------------------------------------------------------------------
{
   // At this function modification for pressure EBC's takes place for solving 
   // pressure correction equation.
   // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4a])
   int i, j, whichBC, node ;
   double x, y, z;
   int loc, colCounter;

   // Modify CSR vectors for pressure BCs
   for (i = 0; i<nPressureNodes; i++) {
      node = pressureNodes[i][0];         // Node at which this EBC is specified
   	
      x = coord[node][0];                 // May be necessary for BCstring evaluation
      y = coord[node][1];
      z = coord[node][2];

      colCounter = 0;
      for(loc=row_deltaP[node]; loc<row_deltaP[node+1]; loc++) {   // loc is the location of the col vector(col[x], loc=x)
         if(col_deltaP[loc] == node) {                                   // Selection process of the KeKMapSmall data from the col vector.
            break;
         }
         colCounter++;
      }
      
      whichBC = pressureNodes[i][1]-1;      // Number of the specified BC   	
      
      for (j=row_deltaP[node]; j<row_deltaP[node+1]; j++) {
         val_deltaP[j] = 0.0;
      }
      val_deltaP[ row_deltaP[node] + colCounter ] = 1;
      
      F_deltaP[node] = 0.0;    // Specified value of the PV          
   } 

} // End of function applyBC_deltaP()



//------------------------------------------------------------------------------
void vectorProduct()
//------------------------------------------------------------------------------
{
   // This function makes calculations for some matrix-vector operations on GPU
   // Some operations at RHS of the mass-adjust velocity equations and
   // some operations at RHS of the momentum equations.
   // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4b], [4c] [4e], [4f])   
   
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
   
   cudaMalloc((void**)&d_col, (NNZ)*sizeof(int));
   cudaMalloc((void**)&d_row, (NN+1)*sizeof(int));
   cudaMalloc((void**)&d_val, (NNZ)*sizeof(real));
   cudaMalloc((void**)&d_x, NN*sizeof(real));  
   cudaMalloc((void**)&d_r, NN*sizeof(real));

   cudaMemcpy(d_col, colSmall, (NNZ)*sizeof(int), cudaMemcpyHostToDevice);  
   cudaMemcpy(d_row, rowStartsSmall, (NN+1)*sizeof(int), cudaMemcpyHostToDevice);  
   
   switch (vectorOperationNo) {
   
   case 1:
      // Calculate part of a RHS of the momentum equations
      // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4e], [4f])
      // f_x = - K_uv*v - K_uw*w   
      // f_y = - K_vu*u - K_vw*w
      // f_z = - K_wu*u - K_wv*v
      cudaMalloc((void**)&d_rTemp, NN*sizeof(real));  
   
      switch (phase) {
         case 0:
            cudaMemcpy(d_val, K_uv, (NNZ)*sizeof(real), cudaMemcpyHostToDevice);          
            cudaMemcpy(d_x, v, NN*sizeof(real), cudaMemcpyHostToDevice); 
            break;
         case 1:
            cudaMemcpy(d_val, K_vu, (NNZ)*sizeof(real), cudaMemcpyHostToDevice);             
            cudaMemcpy(d_x, u, NN*sizeof(real), cudaMemcpyHostToDevice);
            break;
         case 2:
            cudaMemcpy(d_val, K_wu, (NNZ)*sizeof(real), cudaMemcpyHostToDevice);  
            cudaMemcpy(d_x, u, NN*sizeof(real), cudaMemcpyHostToDevice);
            break;
      }
      
      #ifdef SINGLE
         cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,NN,NN,1.0,descr,d_val,d_row,d_col,d_x,0.0,d_rTemp);
      #else
         cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,NN,NN,1.0,descr,d_val,d_row,d_col,d_x,0.0,d_rTemp);
      #endif
      
      switch (phase) {
         case 0:
            cudaMemcpy(d_val, K_uw, (NNZ)*sizeof(real), cudaMemcpyHostToDevice);           
            cudaMemcpy(d_x, w, NN*sizeof(real), cudaMemcpyHostToDevice); 
            break;
         case 1:
            cudaMemcpy(d_val, K_vw, (NNZ)*sizeof(real), cudaMemcpyHostToDevice);  
            cudaMemcpy(d_x, w, NN*sizeof(real), cudaMemcpyHostToDevice);
            break;
         case 2:
            cudaMemcpy(d_val, K_wv, (NNZ)*sizeof(real), cudaMemcpyHostToDevice);  
            cudaMemcpy(d_x, v, NN*sizeof(real), cudaMemcpyHostToDevice);
            break;
      }
      
      #ifdef SINGLE
         cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,NN,NN,1.0,descr,d_val,d_row,d_col,d_x,0.0,d_r);
      #else
         cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,NN,NN,1.0,descr,d_val,d_row,d_col,d_x,0.0,d_r);
      #endif      
      
      #ifdef SINGLE
         cublasSaxpy(NN,1.0,d_r,1,d_rTemp,1);
      #else
         cublasDaxpy(NN,1.0,d_r,1,d_rTemp,1);
      #endif   
      
      // Calculate part of a RHS of the momentum equations
      // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4e], [4f])
      // C_x * p^(i+1), C_y * p^(i+1), C_z * p^(i+1)
      switch (phase) {
         case 0:
            cudaMemcpy(d_val, Cx, (NNZ)*sizeof(real), cudaMemcpyHostToDevice); 
            break;
         case 1:
            cudaMemcpy(d_val, Cy, (NNZ)*sizeof(real), cudaMemcpyHostToDevice); 
            break;
         case 2:
            cudaMemcpy(d_val, Cz, (NNZ)*sizeof(real), cudaMemcpyHostToDevice); 
            break;
      }    
      cudaMemcpy(d_x, p, NN*sizeof(real), cudaMemcpyHostToDevice);
      
      #ifdef SINGLE
         cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,NN,NN,1.0,descr,d_val,d_row,d_col,d_x,0.0,d_r);
      #else
         cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,NN,NN,1.0,descr,d_val,d_row,d_col,d_x,0.0,d_r);
      #endif
      
      #ifdef SINGLE
         cublasSaxpy(NN,-1.0,d_rTemp,1,d_r,1);
      #else
         cublasDaxpy(NN,-1.0,d_rTemp,1,d_r,1);
      #endif  

      cudaMemcpy(F,d_r,(NN)*sizeof(real),cudaMemcpyDeviceToHost);   
      cudaFree(d_rTemp);      
      break;
      
   case 2: 
      // Calculating part of a RHS of the velocity correction
      // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4b], [4c])      
      // C_x * deltaP^(i+1/2), C_y * deltaP^(i+1/2), C_z * deltaP^(i+1/2)
      switch (phase) {
         case 0:
            cudaMemcpy(d_val, Cx, (NNZ)*sizeof(real), cudaMemcpyHostToDevice); 
            break;
         case 1:
            cudaMemcpy(d_val, Cy, (NNZ)*sizeof(real), cudaMemcpyHostToDevice); 
            break;
         case 2:
            cudaMemcpy(d_val, Cz, (NNZ)*sizeof(real), cudaMemcpyHostToDevice); 
            break;
      }

      cudaMemcpy(d_x, delta_p, NN*sizeof(real), cudaMemcpyHostToDevice);
      
      #ifdef SINGLE
         cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,NN,NN,1.0,descr,d_val,d_row,d_col,d_x,0.0,d_r);
      #else
         cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,NN,NN,1.0,descr,d_val,d_row,d_col,d_x,0.0,d_r);
      #endif      
      cudaMemcpy(F,d_r,(NN)*sizeof(real),cudaMemcpyDeviceToHost);
      break;
   }      
   
   cudaFree(d_col);
   cudaFree(d_row);
   cudaFree(d_val);
   cudaFree(d_x);
   cudaFree(d_r);

} // End of function vectorProduct()




//------------------------------------------------------------------------------
void solve()
//------------------------------------------------------------------------------
{
   // This function is for nonlinear iterations
   // Overall structure of the function;
   // while(iteration number < maximum non-linear iterations)
   //   calculate STEP 1 (solve SCPE for pressure correction, get delta_p)
   //   calculate STEP 2 (mass-adjust velocity field and increment pressure, get u^(i+1/2), v^(i+1/2), w^(i+1/2) & p^(i+1))
   //   calculate STEP 3 (solve x, y, z momentum equations, get u^(i+1), v^(i+1), w^(i+1))
   //   check convergence
   //   print monitor points and other info
   // end
   //
   // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4a],)     
   
   
   
   int i, j;
   real temp;
   real change, maxChange;
   double Start2, End2, Start3, End3, Start4, End4, Start5, End5;  
   
   cout << endl << "SOLVING CYCLE STARTS...";
   cout << endl << "============================================" << endl;   
   
   for (iter = 1; iter < nonlinearIterMax; iter++) {
      Start5 = getHighResolutionTime();   
      cout << endl << "ITERATION NO = " << iter << endl;
      
      // -----------------------S T A R T   O F   S T E P  1------------------------------------
      // (1) solve SCPE for pressure correction delta(p)
      // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4a], [4b], [4c], [4d], [4e], [4f])   
      Start2 = getHighResolutionTime();      
      Start3 = getHighResolutionTime();     
      
      applyBC_p();
      applyBC();  
      
      End3 = getHighResolutionTime();
      printf("   Time for both applyBC's             = %-.4g seconds.\n", End3 - Start3);          
      Start3 = getHighResolutionTime();   
      
      calcGlobalSys_p();
      
      End3 = getHighResolutionTime();
      printf("   Time for calcGlobalSys for all      = %-.4g seconds.\n", End3 - Start3);   
      
      Start3 = getHighResolutionTime();         

      for (i=0; i<NN; i++) {
         K_u_diagonal[i] = 1.0/K_u_diagonal[i];
         K_v_diagonal[i] = 1.0/K_v_diagonal[i];
         K_w_diagonal[i] = 1.0/K_w_diagonal[i];
      }
      
      End3 = getHighResolutionTime();
      printf("   Time for taking inv of dia(K)       = %-.4g seconds.\n", End3 - Start3);   
      
      Start3 = getHighResolutionTime();         
            
      applyBC_p();
      applyBC();      
      End3 = getHighResolutionTime();
      printf("   Time for both applyBC's             = %-.4g seconds.\n", End3 - Start3);    
      
      Start3 = getHighResolutionTime();           
      #ifdef CG_CUDA
         CUSP_pC_CUDA_CG();
      #endif
      #ifdef CR_CUDA
         CUSP_pC_CUDA_CR();
      #endif
      #ifdef CG_CUSP
         CUSP_pC_CUSP_CG();
      #endif
      #ifdef CR_CUSP
         CUSP_pC_CUSP_CR();
      #endif      
      End3 = getHighResolutionTime();
      printf("   Time for CUSP op's + CR solver      = %-.4g seconds.\n", End3 - Start3);      
      
      End2 = getHighResolutionTime();
      printf("Total time for STEP 1         = %-.4g seconds.\n", End2 - Start2);           
      cout << "STEP 1 is okay: delta(p)^(i+1/2) is calculated." << endl;
      // delta(p)^(i+1/2) is calculated
      // -------------------------E N D   O F   S T E P  1--------------------------------------
      
      
      
      // -----------------------S T A R T   O F   S T E P  2------------------------------------    
      // (2) mass-adjust velocity field and increment pressure via [4b], [4c], [4d].
      // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4b], [4c], [4d]) 
      
      Start2 = getHighResolutionTime();      
      for (phase=0; phase<3; phase++) {    // Defines on which dimension(x, y, z) calculations takes place
         vectorOperationNo = 2;
         vectorProduct();   // Calculates C_x * deltaP^(i+1/2), C_y * deltaP^(i+1/2), C_z * deltaP^(i+1/2) at GPU according to phase.
                            // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4b], [4c])
                            
         switch (phase) {   // Defines on which dimension(x, y, z) calculations takes place
            case 0:
               for (i=0; i<NN; i++) {
                  u[i] += K_u_diagonal[i]*F[i];   // Calculate u^(i+1/2)
               }
               break;
            case 1:
               for (i=0; i<NN; i++) {
                  v[i] += K_v_diagonal[i]*F[i];   // Calculate v^(i+1/2)
               } 
               break;
            case 2:
               for (i=0; i<NN; i++) {
                  w[i] += K_w_diagonal[i]*F[i];   // Calculate w^(i+1/2)
               }  
               break;
         }
         applyBC();
      }   // End of phase loop
      
      for (i=0; i<NN; i++) {
         p[i] = p[i] + (1.0-alpha[3]) * delta_p[i];   // Calculate p^(i+1)
      }
      
      End2 = getHighResolutionTime();
      printf("Total time for STEP 2         = %-.4g seconds.\n", End2 - Start2);         
      cout << "STEP 2 is okay: u^(i+1/2), v^(i+1/2), w^(i+1/2) & p^(i+1) are calculated." << endl;        
      // u^(i+1/2), v^(i+1/2), w^(i+1/2) & p^(i+1) are calculated      
      // -------------------------E N D   O F   S T E P  2--------------------------------------     

      
      
      // -----------------------S T A R T   O F   S T E P  3------------------------------------ 
      // Solve x, y and z momentum equations([4e], [4f]) for u, v, w
      // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4e], [4f])       
      
      Start2 = getHighResolutionTime();
      
      for (phase=0; phase<3; phase++) {   // Defines on which dimension(x, y, z) calculations takes place
         Start4 = getHighResolutionTime(); 
         Start3 = getHighResolutionTime();  
         calcGlobalSys_mom();
         
         switch (phase) {
            case 0:
               End3 = getHighResolutionTime();                  
               printf("      Time for calcGlobalSys for x        = %-.4g seconds.\n", End3 - Start3); 
               break;
            case 1:
               End3 = getHighResolutionTime();                  
               printf("      Time for calcGlobalSys for y        = %-.4g seconds.\n", End3 - Start3); 
               break;
            case 2: 
               End3 = getHighResolutionTime();                  
               printf("      Time for calcGlobalSys for z        = %-.4g seconds.\n", End3 - Start3); 
               break;
         }
         
         Start3 = getHighResolutionTime();
         applyBC_p();         
         applyBC();
         End3 = getHighResolutionTime();
         printf("      Time for both applyBC's             = %-.4g seconds.\n", End3 - Start3);   

         Start3 = getHighResolutionTime();           
         vectorOperationNo = 1;
         vectorProduct();   // Calculates f_u + C_x * p (@GPU) or f_v + C_y * p (@GPU) or f_w + C_z * p (@GPU) according to phase.
                            // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4e], [4f])     
         End3 = getHighResolutionTime();
         switch (phase) {   // Defines on which dimension(x, y, z) calculations takes place
            case 0:                 
               printf("      Time for f_u + C_x * p (@GPU)       = %-.4g seconds.\n", End3 - Start3); 
               break;
            case 1:                 
               printf("      Time for f_v + C_y * p (@GPU)       = %-.4g seconds.\n", End3 - Start3);     
               break;
            case 2:                
               printf("      Time for f_w + C_z * p (@GPU)       = %-.4g seconds.\n", End3 - Start3);     
               break;
         }
         
         Start3 = getHighResolutionTime();
         switch (phase) {   // Defines on which dimension(x, y, z) calculations takes place
            case 0:
               for (i=0; i<NN; i++) {
                  F[i]= (alpha[0]/(1.0-alpha[0]))*tempDiagonal[i]*u[i] + F[i]; // Calculates final values for RHS of the x-momentum equation
               }                                                               // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4e])    
               End3 = getHighResolutionTime();
               printf("      Time for K_u(dia) * u + [C_x*p]     = %-.4g seconds.\n", End3 - Start3);                  
               break;
            case 1:
               for (i=0; i<NN; i++) {
                  F[i]= (alpha[1]/(1.0-alpha[1]))*tempDiagonal[i]*v[i] + F[i]; // Calculates final values for RHS of the y-momentum equation
               }                                                               // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4f])  
               End3 = getHighResolutionTime();
               printf("      Time for K_v(dia) * v + [C_y*p]     = %-.4g seconds.\n", End3 - Start3);   
               break;
            case 2:
               for (i=0; i<NN; i++) {
                  F[i]= (alpha[2]/(1.0-alpha[2]))*tempDiagonal[i]*w[i] + F[i]; // Calculates final values for RHS of the x-momentum equation
               }                                                               // (Segregated Finite Element Algorithms for the Numerical Solution of Large-Scale Incompressible Flow Problems, Vahe Horoutunian, [4f*])               
               End3 = getHighResolutionTime();                                 // *Paper derives equations for two dimensions.
               printf("      Time for K_w(dia) * w + [C_z*p]     = %-.4g seconds.\n", End3 - Start3);  
               break;
         }           

         Start3 = getHighResolutionTime();   
         applyBC();
         End3 = getHighResolutionTime();
         printf("      Time for both applyBC's             = %-.4g seconds.\n      ", End3 - Start3);   
         
         Start3 = getHighResolutionTime();     
         #ifdef GMRES_CUSP
            CUSP_GMRES(); // Non-sym, positive def
         #endif
         #ifdef BiCG_CUSP
            CUSP_BiCG();  // Non-sym, positive def 
         #endif
         End3 = getHighResolutionTime();
         printf("\n      Time for momentum eq solver         = %-.4g seconds.", End3 - Start3);     
         switch (phase) {   // Defines on which dimension(x, y, z) calculations takes place
            case 0:
               cout << endl << "   x-momentum is solved." << endl; 
               End4 = getHighResolutionTime();
               printf("   Total time for solving x-momentum   = %-.4g seconds.\n", End4 - Start4);    
               break;
            case 1:  
               cout << endl << "   y-momentum is solved." << endl; 
               End4 = getHighResolutionTime();
               printf("   Total time for solving y-momentum   = %-.4g seconds.\n", End4 - Start4);                   
               break;
            case 2: 
               cout << endl << "   z-momentum is solved." << endl;
               End4 = getHighResolutionTime();
               printf("   Total time for solving z-momentum   = %-.4g seconds.\n", End4 - Start4);                   
               break;
         }
      }   // End of phase loop
      for (i=0; i<NN; i++) {
         u[i] = u_temp[i];
         v[i] = v_temp[i];
         w[i] = w_temp[i];         
      }   
      
      End2 = getHighResolutionTime();
      printf("Total time for STEP 3         = %-.4g seconds.\n", End2 - Start2);         
      cout << "STEP 3 is okay: u^(i+1), v^(i+1), w^(i+1) are calculated." << endl;
      // Momentum equations are solved. u^(i+1), v^(i+1), w^(i+1) are calculated.
      // -------------------------E N D   O F   S T E P  3--------------------------------------
      
      
      
      // Calculates maximum error/change for checking convergence.
      Start2 = getHighResolutionTime();   
      maxChange = abs(delta_p[0]);
      
      for (i=1; i<NN; i++) {
         change = abs(delta_p[i]);
         if (change > maxChange) {
            maxChange = change;
         }
      }
      
      End2 = getHighResolutionTime();
      
      printf("Total time for calc maxChange = %-.4g seconds.\n", End2 - Start2);  
      
      End5 = getHighResolutionTime();      
      
      cout <<  " Iter |   Time(sec)   |  Max. Change  |    Mon u    |    Mon v    |    Mon w    |    Mon p  " << endl;
      cout <<  "============================================================================================" << endl;        
      printf("%5d %10.4g %19.5e", iter, End5 - Start5, maxChange);

      if (nMonitorPoints > 0) {
         printf("%11d %14.4e %13.4e %13.4e %13.4e\n", monitorNodes[0],
                                                      u[monitorNodes[0]],
                                                      u[monitorNodes[0] + NN],
                                                      u[monitorNodes[0] + NN*2],
                                                      u[monitorNodes[0] + NN*3]);
         for (i=1; i<nMonitorPoints; i++) {
            printf("%59d %14.4e %13.4e %13.4e %13.4e\n", monitorNodes[i],
                                                         u[monitorNodes[i]],
                                                         u[monitorNodes[i] + NN],
                                                         u[monitorNodes[i] + NN*2],
                                                         u[monitorNodes[i] + NN*3]);
         } 
      }
      cout << endl;
      
      if (maxChange < nonlinearTol && iter > 1) {
         for (phase=0; phase<3; phase++) {
            applyBC();
         }
         applyBC_p();
         break;
      }
      
      // Write Tecplot file
      if(iter % nDATiter == 0 || iter == nonlinearIterMax) {
         writeTecplotFile();
         // cout << "A DAT file is created for Tecplot." << endl;
      }      
        
   }   // End of nonlinearIter loop
   
   
   // Giving info about convergence
   if (iter > nonlinearIterMax) { 
      cout << endl << "Solution did not converge in " << nonlinearIterMax << " iterations." << endl; 
   }
   else {
      cout << endl << "Convergence is achieved at " << iter << " iterations." << endl; 
      writeTecplotFile();      
   }

}  // End of function solve()




//------------------------------------------------------------------------------
void postProcess()
//------------------------------------------------------------------------------
{
   // Write the calculated unknowns to a file named ProblemName.out.
   
   ostringstream dummy;
   dummy << iter;
   outputExtensionPostProcess = "_" + dummy.str() + ".out";

   outputPostProcess.open((whichProblem + outputExtensionPostProcess).c_str(), ios::out);   

   int i;
   // Print the calculated values
   for (i = 0; i<NN; i++) {
      outputPostProcess.precision(5);
      outputPostProcess << scientific  << "\t" << i << " " << u[i] << " " << v[i] << " " << w[i] << " " << p[i] << endl;
   }
   
   outputPostProcess.close();

} // End of function postProcess()




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




//------------------------------------------------------------------------------
void writeTecplotFile()
//------------------------------------------------------------------------------
{
   // Write the calculated unknowns to a Tecplot file
   double x, y, z;
   int i, e;

   ostringstream dummy;
   dummy << iter;
   outputExtension = "_" + dummy.str() + ".dat";

   outputFile.open((whichProblem + outputExtension).c_str(), ios::out);
   
   outputFile << "TITLE = " << whichProblem << endl;
   outputFile << "VARIABLES = X,  Y,  Z,  U, V, W, P" << endl;
   
   if (eType == 1) {
      outputFile << "ZONE N=" <<  NN  << " E=" << NE << " F=FEPOINT ET=QUADRILATERAL" << endl;
      }        
   else if (eType == 3) {
      outputFile << "ZONE NODES=" <<  NN  << ", ELEMENTS=" << NE << ", DATAPACKING=POINT, ZONETYPE=FEBRICK" << endl;   
      }
      else { 
      outputFile << "ZONE NODES=" <<  NN  << ", ELEMENTS=" << NE << ", DATAPACKING=POINT, ZONETYPE=FETETRAHEDRON" << endl;   
      }

   // Print the coordinates and the calculated values
   for (i = 0; i<NN; i++) {
      x = coord[i][0];
      y = coord[i][1];
      z = coord[i][2];  
      outputFile.precision(5);
      outputFile << scientific  << "\t" << x << " "  << y << " "  << z << " " << u[i] << " " << v[i] << " " << w[i] << " " << p[i] << endl;
   }

   // Print the connectivity list
   for (e = 0; e<NE; e++) {
      outputFile << fixed << "\t";
      for (i = 0; i<NENv; i++) {
         // outputFile.precision(5);
         outputFile << LtoG[e][i]+1 << " " ;
      }
      outputFile<< endl;
   }

   outputFile.close();
} // End of function writeTecplotFile()




//-----------------------------------------------------------------------------
double getHighResolutionTime(void)
//-----------------------------------------------------------------------------
{
   // Works on Linux

   struct timeval tod;

   gettimeofday(&tod, NULL);
   double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
   return time_seconds;
} // End of function getHighResolutionTime()

