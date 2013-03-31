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

#ifdef SINGLE
  typedef float real;
#else
  typedef double real;
#endif

ifstream meshfile;
ifstream problemFile;
ofstream outputFile;
ofstream outputControl;

string problemNameFile, whichProblem;
string problemName     = "ProblemName.txt";
string controlFile     = "Control_Output.txt";
string inputExtension  = ".inp";
string outputExtension;
string restartExtension = "_restart.dat";

int name, eType, NE, NN, NGP, NEU, Ndof;
int NCN, NENv, NENp, nonlinearIterMax, solverIterMax, nDATiter;
double density, viscosity, fx, fy, nonlinearTol, solverTol;
bool   isRestart;
int **LtoG, **velNodes, **pressureNodes;
double **monitorPoints;
int *monitorNodes;
double **coord;
int nBC, nVelNodes, nPressureNodes, nMonitorPoints;
double *BCtype, **BCstrings;
double axyFunc, fxyFunc;
double **GQpoint, *GQweight;
double **Sp, ***DSp, **Sv, ***DSv;
double **detJacob, ****gDSp, ****gDSv;

int **GtoL, *rowStartsSmall, *colSmall, **KeKMapSmall, NNZ;
int *rowStartsDiagonal, *colDiagonal, NNZ_diagonal;
int *KtoKdiaMap;

double **Ke_11, **Ke_12, **Ke_13, **Ke_14;
double **Ke_11_add, **Ke_12_add, **Ke_13_add, **Ke_14_add;
double *uNodal, *vNodal, *wNodal;
double u0, v0, w0; 
double *Du0, *Dv0, *Dw0;
real *F, *u, *v, *w, *pPrime, *p, *velVector;
real *uDiagonal, *vDiagonal, *wDiagonal, *tempDiagonal;
real *Cx, *Cy, *Cz;
real *val, *val_f, *K_12, *K_13;
real *val_deltaP, *F_deltaP;
int *row_deltaP, *col_deltaP;

int iter;
int phase, STEP, vectorOperationNo;
double alpha [4];

//Variables for CUDA operations
int *d_col, *d_row;
real *d_val, *d_x, *d_r, *d_rTemp;

void readInput();
void readRestartFile();
void gaussQuad();
void calcShape();
void calcJacobian();
void initGlobalSysVariables();
void calcGlobalSys_p();
void assemble_p(int e, double **Ke_11, double **Ke_14);
void calcGlobalSys_mom();
void assemble_mom(int e, double **Ke_11, double **Ke_12, double **Ke_13, double **Ke_14);
void applyBC();
void applyBC_p();
void applyBC_deltaP();
void vectorProduct();
void solve();
void postProcess();
void writeTecplotFile();
void compressedSparseRowStorage();
double getHighResolutionTime();

//pressure correction equation solvers
#ifdef CG_CUDA
   extern void CUSP_pC_CUDA_CG();
#endif
#ifdef CR_CUDA
   extern void CUSP_pC_CUDA_CR();
#endif
#ifdef CG_CUSP
   extern void CUSP_pC_CUSP_CG();
#endif

//momentum equation solvers
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
   cout << "\n*       http://code.google.com/p/cfd-with-cuda         *";
   cout << "\n********************************************************\n\n";

   cout << "The program is started." << endl ;
   
   double Start, End, Start1, End1;

   Start1 = getHighResolutionTime();   

   Start = getHighResolutionTime();
   readInput();             
   End = getHighResolutionTime();
   printf("Time for Input    = %-.4g seconds.\n", End - Start);
   
   Start = getHighResolutionTime();
   compressedSparseRowStorage(); 
   End = getHighResolutionTime();
   printf("Time for CSR      = %-.4g seconds.\n", End - Start);

   Start = getHighResolutionTime();   
   gaussQuad();
   End = getHighResolutionTime();
   printf("Time for GQ       = %-.4g seconds.\n", End - Start);

   Start = getHighResolutionTime();
   calcShape();
   End = getHighResolutionTime();
   printf("Time for Shape    = %-.4g seconds.\n", End - Start);
   
   Start = getHighResolutionTime();
   calcJacobian();
   End = getHighResolutionTime();
   printf("Time for Jacobian = %-.4g seconds.\n", End - Start);

   Start = getHighResolutionTime();
   initGlobalSysVariables();
   End = getHighResolutionTime();
   printf("Time for InitVar  = %-.4g seconds.\n", End - Start);
   
   Start = getHighResolutionTime();
   solve();
   End = getHighResolutionTime();
   printf("Time for Iter's   = %-.4g seconds.\n", End - Start);   

   End1 = getHighResolutionTime();

   printf("Total Time        = %-.4g seconds.\n", End1 - Start1);

   cout << endl << "The program is terminated successfully.\nPress a key to close this window...\n";

   return 0;

} // End of function main()




//------------------------------------------------------------------------------
void readInput()
//------------------------------------------------------------------------------
{

   string dummy, dummy2, dummy4, dummy5;
   int dummy3, i, j;

   problemFile.open(problemName.c_str(), ios::in);   // Read problem name
   problemFile >> whichProblem;                                  
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
   
   NEU = 3*NENv + NENp;
   Ndof = 3*NN + NCN;
   
   // Read node coordinates
   coord = new double*[NN];
   
   if (eType == 2 || eType == 1) {
      for (i = 0; i < NN; i++) {
         coord[i] = new double[2];
      }
   	
      for (i=0; i<NN; i++){
         meshfile >> dummy3 >> coord[i][0] >> coord[i][1] ;
         meshfile.ignore(256, '\n'); // Ignore the rest of the line    
      }  
   } else {
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
   GtoLCounter = new int[NN];
   
   for (i=0; i<NN; i++) {     
      GtoL[i] = new int[noOfColGtoL];   
   }

   for(i=0; i<NN; i++) {
      for(j=0; j<noOfColGtoL; j++) {
         GtoL[i][j] = -1;      // For the nodes that didn't connect 4 nodes.  TODO: Not clear.
      }
      GtoLCounter[i] = 0;
   }
   
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
                     if(checkCol[y] == (LtoG[valGtoL][x])) {   // This column was created
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
   }
   
   delete[] checkCol;

   // Create col & colSmall

   colSmall = new int[rowStartsSmall[NN]];  // col for a part of K(only for "u" velocity in another words 1/16 of the K(if NENv==NENp)) 

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

   // Allocate and initialize the val vectors
   val = new real[rowStartsSmall[NN]];  
   
   val_f = new real[rowStartsSmall[NN]];
   
   K_12 = new real[rowStartsSmall[NN]]; 
   
   K_13 = new real[rowStartsSmall[NN]]; 

   // Create CSR vectors for diagonal matrix
   rowStartsDiagonal = new int[NN+1];
   for(i=0; i<=NN; i++) {
      rowStartsDiagonal[i] = i;
   }
   
   colDiagonal = new int[rowStartsDiagonal[NN]];
   for(i=0; i<NN; i++) {
      colDiagonal[i] = i;
   }  
   
   // Create K to Kdia map 
   KtoKdiaMap = new int[NN];
   for(i=0; i<NN; i++) {
      for(j=rowStartsSmall[i]; j<rowStartsSmall[i+1]; j++) {
         if(colSmall[j]==i){
            KtoKdiaMap[i] = j;
            break;
         }
      }
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
         }
      }
   }

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
   
}  // End of function calcJacobian()



//------------------------------------------------------------------------------
void initGlobalSysVariables()
//------------------------------------------------------------------------------
{
   // Allocate memory for stiffness matrices and force vectors
   
   int i;
   
   F = new real[NN];

   for (i=0; i<NN; i++) {
      F[i] = 0;
   }
   
   Ke_11 = new double*[NENv];
   Ke_12 = new double*[NENv];
   Ke_13 = new double*[NENv];
   Ke_14 = new double*[NENv];
   
   Ke_11_add = new double*[NENv];
   Ke_12_add = new double*[NENv];
   Ke_13_add = new double*[NENv];   
   Ke_14_add = new double*[NENv];

   for (i=0; i<NENv; i++) {
      Ke_11[i] = new double[NENv];
      Ke_12[i] = new double[NENv];
      Ke_13[i] = new double[NENv];      
      Ke_14[i] = new double[NENp];

      Ke_11_add[i] = new double[NENv];
      Ke_12_add[i] = new double[NENv];
      Ke_13_add[i] = new double[NENv];      
      Ke_14_add[i] = new double[NENp];
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
   p = new real[NN];  
   pPrime = new real[NN];
   velVector = new real[NN];

   uDiagonal = new real[NN];
   vDiagonal = new real[NN];
   wDiagonal = new real[NN]; 
   tempDiagonal = new real[NN];  
   
   // Initial guesses for unknowns 
   for (i=0; i<NN; i++) {
      u[i] = 0.0;
      v[i] = 0.0;
      w[i] = 0.0;
      p[i] = 0.0;
      pPrime[i] = 0.0;
      velVector[i] = 0.0; 
      uDiagonal[i] = 0.0;
      vDiagonal[i] = 0.0;
      wDiagonal[i] = 0.0;   
      tempDiagonal[i] = 0.0;      
   }  

   Cx = new real[rowStartsSmall[NN]];
   Cy = new real[rowStartsSmall[NN]];
   Cz = new real[rowStartsSmall[NN]];
   
   for (i=0; i<rowStartsSmall[NN]; i++) {
      Cx[i] = 0.0;
      Cy[i] = 0.0;
      Cz[i] = 0.0;
   }
   
   for(i=0; i<rowStartsSmall[NN]; i++) {
      val[i] = 0;
      val_f[i] = 0;
      K_12[i] = 0;
      K_13[i] = 0; 
   }    
   
   if(isRestart) {
      readRestartFile();
   }
} 




//------------------------------------------------------------------------------
void calcGlobalSys_p()
//------------------------------------------------------------------------------
{
   // Calculates Ke and Fe one by one for each element and assembles them into
   // the global K and F.

   int e, i, j, k, m, n, node, valSwitch;
   double Tau;
   int factor[3];
   
   switch (phase) {
      case 0:
         factor[0] = 2;
         factor[1] = 1;
         factor[2] = 1;
      break;
      case 1:
         factor[0] = 1;
         factor[1] = 2;
         factor[2] = 1;      
      break;
      case 2:
         factor[0] = 1;
         factor[1] = 1;
         factor[2] = 2;      
      break;
   }

   for(i=0; i<rowStartsSmall[NN]; i++) {
      val[i] = 0;
      val_f[i] = 0;      
   }
   
   for(i=0; i<rowStartsDiagonal[NN]; i++) {
      tempDiagonal[i] = 0;
   }

   // Calculating the elemental stiffness matrix(Ke) and force vector(Fe)

   for (e = 0; e<NE; e++) {
      // Intitialize Ke and Fe to zero.

      for (i=0; i<NENv; i++) {
         for (j=0; j<NENv; j++) {
            Ke_11[i][j] = 0;
         }
         for (j=0; j<NENp; j++) {
            Ke_14[i][j] = 0;
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
               Ke_11_add[i][j] = 0;
            }
            for (j=0; j<NENp; j++) {
               Ke_14_add[i][j] = 0;
            }         
         }

         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {
               Ke_11_add[i][j] = Ke_11_add[i][j] + viscosity * (
                     factor[0] * gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                     factor[1] * gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                     factor[2] * gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                     density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);
                     
            }
            for (j=0; j<NENp; j++) {
               Ke_14_add[i][j] = Ke_14_add[i][j] + gDSv[e][phase][i][k] * Sp[j][k];
            }         
         }
         
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {            
               Ke_11[i][j] += Ke_11_add[i][j] * detJacob[e][k] * GQweight[k];
            }
            for (j=0; j<NENp; j++) {               
               Ke_14[i][j] += Ke_14_add[i][j] * detJacob[e][k] * GQweight[k];
            }    
         }

      }   // End GQ loop 

      // Create diagonal matrices

      for (i=0; i<NENv; i++) {
         tempDiagonal[LtoG[e][i]] = tempDiagonal[LtoG[e][i]] + Ke_11[i][i];
         Ke_11[i][i] += (alpha[0]/(1-alpha[0]))*Ke_11[i][i];
      }

      assemble_p(e, Ke_11, Ke_14);  // Send Ke_11 and Ke_14 for creating val and val_f (Cx, Cy, Cz) vectors   

   }   // End element loop
 
} // End of function calcGlobalSys()




//------------------------------------------------------------------------------
void assemble_p(int e, double **Ke_11, double **Ke_14)
//------------------------------------------------------------------------------
{
   // Inserts Fe into proper locations of F.

   int i, j;

   // Create KeKMapSmall, which stores the mapping between the entries of Ke
   // and val vector of CSR.

   // TODO: Why do we calculate KeKmapSmall in each iteration again and again? Isn't it costly?
   //       Is it feasible to calculate it for each element only once and store?

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
   
   // Creating val vector
   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         val[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_11[i][j] ;
         
         val_f[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_14[i][j] ;      
      }
   }

   
   for (int i = 0; i<NENv; i++) {
      delete[] KeKMapSmall[i];
   }
   delete[] KeKMapSmall;

   delete[] eLtoG;
   
} // End of function assemble()




//------------------------------------------------------------------------------
void calcGlobalSys_mom()
//------------------------------------------------------------------------------
{
   // Calculates Ke and Fe one by one for each element and assembles them into
   // the global K and F.

   int e, i, j, k, m, n, node, valSwitch;
   double Tau;
   real factor[3];
   int factor2[2];
   
   switch (phase) {
      case 0:
         factor[0] = 2.0;
         factor[1] = 1.0;
         factor[2] = 1.0;   
         factor2[0] = 1;
         factor2[1] = 2;
      break;
      case 1:
         factor[0] = 1.0;
         factor[1] = 2.0;
         factor[2] = 1.0;    
         factor2[0] = 0;
         factor2[1] = 2;         
      break;
      case 2:
         factor[0] = 1.0;
         factor[1] = 1.0;
         factor[2] = 2.0; 
         factor2[0] = 0;
         factor2[1] = 1;       
      break;
   }

   for(i=0; i<rowStartsSmall[NN]; i++) {
      val[i] = 0;
      val_f[i] = 0;      
      K_12[i] = 0;      
      K_13[i] = 0;
   }
   
   for(i=0; i<rowStartsDiagonal[NN]; i++) {
      tempDiagonal[i] = 0;
   }

   // Calculating the elemental stiffness matrix(Ke) and force vector(Fe)

   for (e = 0; e<NE; e++) {
      // Intitialize Ke and Fe to zero.

      for (i=0; i<NENv; i++) {
         for (j=0; j<NENv; j++) {
            Ke_11[i][j] = 0;
         }
         for (j=0; j<NENv; j++) {
            Ke_12[i][j] = 0;
         }
         for (j=0; j<NENp; j++) {
            Ke_13[i][j] = 0;
         }           
         for (j=0; j<NENp; j++) {
            Ke_14[i][j] = 0;
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
               Ke_11_add[i][j] = 0;
            }
            for (j=0; j<NENv; j++) {
               Ke_12_add[i][j] = 0;
            }
            for (j=0; j<NENv; j++) {
               Ke_13_add[i][j] = 0;
            }
            for (j=0; j<NENp; j++) {
               Ke_14_add[i][j] = 0;
            }         
         }

         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {
               Ke_11_add[i][j] = Ke_11_add[i][j] + viscosity * (
                     factor[0] * gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                     factor[1] * gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                     factor[2] * gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                     density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);
                     
               Ke_12_add[i][j] = Ke_12_add[i][j] + viscosity * gDSv[e][factor2[0]][i][k] * gDSv[e][phase][j][k];
               
               Ke_13_add[i][j] = Ke_13_add[i][j] + viscosity * gDSv[e][factor2[1]][i][k] * gDSv[e][phase][j][k];        
            }
            for (j=0; j<NENp; j++) {
               Ke_14_add[i][j] = Ke_14_add[i][j] + gDSv[e][phase][i][k] * Sp[j][k];
            }         
         }
         
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {            
               Ke_11[i][j] += Ke_11_add[i][j] * detJacob[e][k] * GQweight[k];
            }
            for (j=0; j<NENv; j++) {            
               Ke_12[i][j] += Ke_12_add[i][j] * detJacob[e][k] * GQweight[k];
            }
            for (j=0; j<NENv; j++) {            
               Ke_13[i][j] += Ke_13_add[i][j] * detJacob[e][k] * GQweight[k];
            }            
            for (j=0; j<NENp; j++) {               
               Ke_14[i][j] += Ke_14_add[i][j] * detJacob[e][k] * GQweight[k];
            }    
         }

      }   // End GQ loop  

      // Create diagonal matrices

      for (i=0; i<NENv; i++) {
         tempDiagonal[LtoG[e][i]] = tempDiagonal[LtoG[e][i]] + Ke_11[i][i];
         Ke_11[i][i] += (alpha[0]/(1.0-alpha[0]))*Ke_11[i][i];
      }      
      
      assemble_mom(e, Ke_11, Ke_12, Ke_13, Ke_14);  // Send Ke_11 and Ke_14 for creating val and val_f (Cx, Cy, Cz) vectors   

   }   // End element loop
 
} // End of function calcGlobalSys()




//------------------------------------------------------------------------------
void assemble_mom(int e, double **Ke_11, double **Ke_12, double **Ke_13, double **Ke_14)
//------------------------------------------------------------------------------
{
   // Inserts Fe into proper locations of F.

   int i, j;

   // Create KeKMapSmall, which stores the mapping between the entries of Ke
   // and val vector of CSR.

   // TODO: Why do we calculate KeKmapSmall in each iteration again and again? Isn't it costly?
   //       Is it feasible to calculate it for each element only once and store?

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
   
   // Creating val vector
   for(i=0; i<NENv; i++) {
      for(j=0; j<NENv; j++) {
         val[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_11[i][j] ;
         K_12[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_12[i][j] ;
         K_13[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_13[i][j] ;
         val_f[ rowStartsSmall[LtoG[e][i]] + KeKMapSmall[i][j] ] += Ke_14[i][j] ;      
      }
   }

   
   for (int i = 0; i<NENv; i++) {
      delete[] KeKMapSmall[i];
   }
   delete[] KeKMapSmall;

   delete[] eLtoG;
   
} // End of function assemble()




//------------------------------------------------------------------------------
void applyBC()
//------------------------------------------------------------------------------
{
   // For EBCs reduction is not applied. Instead K and F are modified as
   // explained in class, which requires modification of both [K] and {F}.
   // SV values specified for NBCs are added to {F}.

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
      for(loc=rowStartsSmall[node]; loc<rowStartsSmall[node+1]; loc++) {   // loc is the location of the col vector(col[x], loc=x)
         if(colSmall[loc] == node) {                                   // Selection process of the KeKMapSmall data from the col vector.
            break;
         }
         colCounter++;
      }

      whichBC = velNodes[i][1]-1;      // Number of the specified BC 
      
      for (j=rowStartsSmall[node]; j<rowStartsSmall[node+1]; j++) {
         val[j] = 0.0;
      }
      val[ rowStartsSmall[node] + colCounter ] = 1;

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

} // End of function ApplyBC()




//------------------------------------------------------------------------------
void applyBC_p()
//------------------------------------------------------------------------------
{
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
      for(loc=rowStartsSmall[node]; loc<rowStartsSmall[node+1]; loc++) {   // loc is the location of the col vector(col[x], loc=x)
         if(colSmall[loc] == node) {                                   // Selection process of the KeKMapSmall data from the col vector.
            break;
         }
         colCounter++;
      }
      
      whichBC = pressureNodes[i][1]-1;      // Number of the specified BC   	
      
      for (j=rowStartsSmall[node]; j<rowStartsSmall[node+1]; j++) {
         val_f[j] = 0.0;
      }
      val_f[ rowStartsSmall[node] + colCounter ] = 1;
      
      p[node] = BCstrings[whichBC][0];    // Specified value of the PV          
   } 

} // End of function ApplyBC()




//------------------------------------------------------------------------------
void applyBC_deltaP()
//------------------------------------------------------------------------------
{
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

} // End of function ApplyBC()



//------------------------------------------------------------------------------
void vectorProduct()
//------------------------------------------------------------------------------
{
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
   
   cudaMalloc((void**)&d_col, (rowStartsSmall[NN])*sizeof(int));
   cudaMalloc((void**)&d_row, (NN+1)*sizeof(int));
   cudaMalloc((void**)&d_val, (rowStartsSmall[NN])*sizeof(real));
   cudaMalloc((void**)&d_x, NN*sizeof(real));  
   cudaMalloc((void**)&d_r, NN*sizeof(real));

   cudaMemcpy(d_col, colSmall, (rowStartsSmall[NN])*sizeof(int), cudaMemcpyHostToDevice);  
   cudaMemcpy(d_row, rowStartsSmall, (NN+1)*sizeof(int), cudaMemcpyHostToDevice);  
   
   switch (vectorOperationNo) {
   
   case 1:
      // Calculate part of a RHS of the momentum equations ([4e], [4f])
      // f_x = - K_uv*v - K_uw*w   
      // f_y = - K_vu*u - K_vw*w
      // f_z = - K_wu*u - K_wv*v
      cudaMalloc((void**)&d_rTemp, NN*sizeof(real));  

      cudaMemcpy(d_val, K_12, (rowStartsSmall[NN])*sizeof(real), cudaMemcpyHostToDevice);    
      switch (phase) {
         case 0:
            cudaMemcpy(d_x, v, NN*sizeof(real), cudaMemcpyHostToDevice); 
            break;
         case 1:
            cudaMemcpy(d_x, u, NN*sizeof(real), cudaMemcpyHostToDevice);
            break;
         case 2:
            cudaMemcpy(d_x, u, NN*sizeof(real), cudaMemcpyHostToDevice);
            break;
      }
      
      #ifdef SINGLE
         cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,NN,NN,1.0,descr,d_val,d_row,d_col,d_x,0.0,d_rTemp);
      #else
         cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,NN,NN,1.0,descr,d_val,d_row,d_col,d_x,0.0,d_rTemp);
      #endif
      
      cudaMemcpy(d_val, K_13, (rowStartsSmall[NN])*sizeof(real), cudaMemcpyHostToDevice);
      switch (phase) {
         case 0:
            cudaMemcpy(d_x, w, NN*sizeof(real), cudaMemcpyHostToDevice); 
            break;
         case 1:
            cudaMemcpy(d_x, w, NN*sizeof(real), cudaMemcpyHostToDevice);
            break;
         case 2:
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
      
      // Calculate part of a RHS of the momentum equations ([4e], [4f])
      // C_x * p^(i+1), C_y * p^(i+1), C_z * p^(i+1)
      cudaMemcpy(d_val, val_f, (rowStartsSmall[NN])*sizeof(real), cudaMemcpyHostToDevice);    
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
      // Calculating part of a RHS of the velocity correction ([4b], [4c])
      // C_x * deltaP^(i+1/2), C_y * deltaP^(i+1/2), C_z * deltaP^(i+1/2)
      switch (phase) {
         case 0:
            cudaMemcpy(d_val, Cx, (rowStartsSmall[NN])*sizeof(real), cudaMemcpyHostToDevice); 
            break;
         case 1:
            cudaMemcpy(d_val, Cy, (rowStartsSmall[NN])*sizeof(real), cudaMemcpyHostToDevice); 
            break;
         case 2:
            cudaMemcpy(d_val, Cz, (rowStartsSmall[NN])*sizeof(real), cudaMemcpyHostToDevice); 
            break;
      }

      cudaMemcpy(d_x, pPrime, NN*sizeof(real), cudaMemcpyHostToDevice);
      
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
}




//------------------------------------------------------------------------------
void solve()
//------------------------------------------------------------------------------
{
   int i, j;
   real temp;
   real change, maxChange;
   double Start2, End2, Start3, End3, Start4, End4, Start5, End5;  
   
   cout << endl << "SOLVING CYCLE STARTS...";
   cout << endl << "============================================" << endl;   
   
   for (iter = 1; iter < nonlinearIterMax; iter++) {
      Start5 = getHighResolutionTime();   
      cout << endl << "ITERATION NO = " << iter << endl;
      // (1) solve SCPE for pressure correction delta(p) (equations: [4a])
      Start2 = getHighResolutionTime();      
      Start3 = getHighResolutionTime();     
      
      applyBC_p();
      applyBC();  
      
      End3 = getHighResolutionTime();
      printf("   Time for both applyBC's             = %-.4g seconds.\n", End3 - Start3);          
      Start3 = getHighResolutionTime();   
      for (phase=0; phase<3; phase++) {
         calcGlobalSys_p();
         switch (phase) {
            case 0:
               for(i=0; i<rowStartsDiagonal[NN]; i++) {
                  uDiagonal[i] = tempDiagonal[i];
               }
               for (i=0; i<NNZ; i++) {
                  Cx[i] = val_f[i];
               }
               break;
            case 1:
               for(i=0; i<rowStartsDiagonal[NN]; i++) {
                  vDiagonal[i] = tempDiagonal[i];
               }
               for (i=0; i<NNZ; i++) {
                  Cy[i] = val_f[i];
               }     
               break;
            case 2:
               for(i=0; i<rowStartsDiagonal[NN]; i++) {
                  wDiagonal[i] = tempDiagonal[i];
               }
               for (i=0; i<NNZ; i++) {
                  Cz[i] = val_f[i];
               }     
               break;
         }
      }

      for (i=0; i<NN; i++) {
         uDiagonal[i] = 1.0/uDiagonal[i];
         vDiagonal[i] = 1.0/vDiagonal[i];
         wDiagonal[i] = 1.0/wDiagonal[i];
      }
      
      End3 = getHighResolutionTime();
      printf("   Time for calcGlobalSys for all      = %-.4g seconds.\n", End3 - Start3);   
      
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
      End3 = getHighResolutionTime();
      printf("   Time for CUSP op's + CR solver      = %-.4g seconds.\n", End3 - Start3);      
      
      End2 = getHighResolutionTime();
      printf("Total time for STEP 1         = %-.4g seconds.\n", End2 - Start2);           
      cout << "STEP 1 is okay: delta(p)^(i+1/2) is calculated." << endl;
      // delta(p)^(i+1/2) is calculated
      //-----------------------------------------
      
      //-----------------------------------------      
      // (2) mass-adjust velocity field and increment pressure via (equations: [4b, 4c, 4d])
      Start2 = getHighResolutionTime();      
      for (phase=0; phase<3; phase++) {      
         vectorOperationNo = 2;
         vectorProduct();
         switch (phase) {
            case 0:
               for (i=0; i<NN; i++) {
                  u[i] += uDiagonal[i]*F[i];
               }
               break;
            case 1:
               for (i=0; i<NN; i++) {
                  v[i] += vDiagonal[i]*F[i];
               } 
               break;
            case 2:
               for (i=0; i<NN; i++) {
                  w[i] += wDiagonal[i]*F[i];
               }  
               break;
         }
         applyBC();
      }
      
      for (i=0; i<NN; i++) {
         p[i] = p[i] + (1.0-alpha[3]) * pPrime[i];
      }
      
      End2 = getHighResolutionTime();
      printf("Total time for STEP 2         = %-.4g seconds.\n", End2 - Start2);         
      cout << "STEP 2 is okay: u^(i+1/2), v^(i+1/2), w^(i+1/2) & p^(i) are calculated." << endl;        
      // u^(i+1/2), v^(i+1/2), w^(i+1/2) & p^(i) are calculated      
      //-----------------------------------------      

      //----------------------------------------- 
      // Solve x, y and z momentum equations([4e], [4f]) for u, v, w
      Start2 = getHighResolutionTime();
      real *u_temp, *v_temp, *w_temp;
      u_temp = new real[NN];
      v_temp = new real[NN];
      w_temp = new real[NN];
      
      for (phase=0; phase<3; phase++) {
         Start4 = getHighResolutionTime(); 
         Start3 = getHighResolutionTime();  
         calcGlobalSys_mom();
         
         switch (phase) {
            case 0:
               for (i=0; i<NNZ; i++) {
                  Cx[i] = val_f[i];
               }
               End3 = getHighResolutionTime();                  
               printf("      Time for calcGlobalSys for x        = %-.4g seconds.\n", End3 - Start3); 
               break;
            case 1:
               for (i=0; i<NNZ; i++) {
                  Cy[i] = val_f[i];
               }     
               End3 = getHighResolutionTime();                  
               printf("      Time for calcGlobalSys for y        = %-.4g seconds.\n", End3 - Start3); 
               break;
            case 2:
               for (i=0; i<NNZ; i++) {
                  Cz[i] = val_f[i];
               }     
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
         vectorProduct();
         End3 = getHighResolutionTime();
         switch (phase) {
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
         switch (phase) {
            case 0:
               for (i=0; i<NN; i++) {
                  F[i]= (alpha[0]/(1.0-alpha[0]))*tempDiagonal[i]*u[i] + F[i];
               }
               End3 = getHighResolutionTime();
               printf("      Time for K_u(dia) * u + [C_x*p]     = %-.4g seconds.\n", End3 - Start3);                  
               break;
            case 1:
               for (i=0; i<NN; i++) {
                  F[i]= (alpha[1]/(1.0-alpha[1]))*tempDiagonal[i]*v[i] + F[i];
               }  
               End3 = getHighResolutionTime();
               printf("      Time for K_v(dia) * v + [C_y*p]     = %-.4g seconds.\n", End3 - Start3);   
               break;
            case 2:
               for (i=0; i<NN; i++) {
                  F[i]= (alpha[2]/(1.0-alpha[2]))*tempDiagonal[i]*w[i] + F[i]; 
               }   
               End3 = getHighResolutionTime();
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
         switch (phase) {
            case 0:
               for (i=0; i<NN; i++) {
                  u_temp[i] = velVector[i];
               }
               cout << endl << "   x-momentum is solved." << endl; 
               End4 = getHighResolutionTime();
               printf("   Total time for solving x-momentum   = %-.4g seconds.\n", End4 - Start4);    
               break;
            case 1:
               for (i=0; i<NN; i++) {
                  v_temp[i] = velVector[i];                
               }   
               cout << endl << "   y-momentum is solved." << endl; 
               End4 = getHighResolutionTime();
               printf("   Total time for solving y-momentum   = %-.4g seconds.\n", End4 - Start4);                   
               break;
            case 2:
               for (i=0; i<NN; i++) {
                  w_temp[i] = velVector[i];                  
               }   
               cout << endl << "   z-momentum is solved." << endl;
               End4 = getHighResolutionTime();
               printf("   Total time for solving z-momentum   = %-.4g seconds.\n", End4 - Start4);                   
               break;
         }
      }
      for (i=0; i<NN; i++) {
         u[i] = u_temp[i];
         v[i] = v_temp[i];
         w[i] = w_temp[i];         
      }
      
      delete[] u_temp;
      delete[] v_temp;
      delete[] w_temp;      
      
      End2 = getHighResolutionTime();
      printf("Total time for STEP 3         = %-.4g seconds.\n", End2 - Start2);         
      cout << "STEP 3 is okay: u^(i+1), v^(i+1), w^(i+1) are calculated." << endl;
      // Momentum equations are solved. u^(i+1), v^(i+1), w^(i+1) are calculated.
      //-----------------------------------------
      
      Start2 = getHighResolutionTime();   
      maxChange = abs(pPrime[0]);
      
      for (i=1; i<NN; i++) {
         change = abs(pPrime[i]);
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
        
   }
   
   
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
   // Write the calculated unknowns on the screen.
   // Actually it is a good idea to write the results to an output file named
   // ProblemName.out too.

   printf("\nCalculated unknowns are \n\n");
   printf(" Node      x       y       z         u         v         w       p \n");
   printf("========================================================================\n");
   if (eType == 3 || eType == 4) {
      for (int i = 0; i<NN; i++) { 
         printf("%-5d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n", i, coord[i][0], coord[i][1], coord[i][2], u[i], v[i], w[i], p[i]);
      }
   }
   else { 
      for (int i = 0; i<NN; i++) { 
         printf("%-5d %18.8f %18.8f %20.8f\n", i, coord[i][0], coord[i][1], u[i]);
      }
   }   

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
