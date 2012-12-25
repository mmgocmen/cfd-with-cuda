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
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <algorithm>

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

int    name, eType, NE, NN, NGP, NEU, Ndof;
int    NCN, NENv, NENp, nonlinearIterMax, solverIter, solverIterMax, nDATiter;
double density, viscosity, fx, fy, nonlinearTol, solverTol, solverNorm;
bool   isRestart;
int    **LtoG, **velNodes, **pressureNodes;
double **monitorPoints;
int    *monitorNodes;
double **coord;
int    nBC, nVelNodes, nPressureNodes, nMonitorPoints;
double *elem_he;
double *BCtype, **BCstrings;
double axyFunc, fxyFunc;
double **GQpoint, *GQweight;
double **Sp, ***DSp, **Sv, ***DSv;
double **detJacob, ****gDSp, ****gDSv;
int    iter;
real   *F, *u, *uOld;       // Can be float or double.

double *Fe, **Ke;
double *Fe_1, *Fe_2, *Fe_3, *Fe_4, *Fe_1_add, *Fe_2_add, *Fe_3_add, *Fe_4_add;
double **Ke_11, **Ke_12, **Ke_13, **Ke_14, **Ke_21, **Ke_22, **Ke_23, **Ke_24, **Ke_31, **Ke_32, **Ke_33, **Ke_34, **Ke_41, **Ke_42, **Ke_43, **Ke_44;
double **Ke_11_add, **Ke_12_add, **Ke_13_add, **Ke_14_add, **Ke_21_add, **Ke_22_add, **Ke_23_add, **Ke_24_add;
double **Ke_31_add, **Ke_32_add, **Ke_33_add, **Ke_34_add, **Ke_41_add, **Ke_42_add, **Ke_43_add, **Ke_44_add;
double *uNodal, *vNodal, *wNodal;
double u0, v0, w0;
double *Du0, *Dv0, *Dw0;

int    **GtoL, *rowStarts, *rowStartsSmall, *colSmall, *col, **KeKMapSmall, NNZ;
real   *val;       // Can be float or double.

void readInput();
void readRestartFile();
void calcElemSize();
void gaussQuad();
void calcShape();
void calcJacobian();
void initGlobalSysVariables();
void calcGlobalSys();
void assemble(int e, double **Ke, double *Fe);
void applyBC();
void solve();
void postProcess();
void writeTecplotFile();
void compressedSparseRowStorage();

#ifdef CUSP
   extern int CUSPsolver(); 
#endif

#ifdef MKLPARDISO
   extern int pardisoSolver(); 
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

   time_t start, end;
   time (&start);     // Start measuring execution time.

   readInput();                   cout << "Input file is read." << endl ;
   calcElemSize();                cout << "Element sizes are calculated for GLS stabilization." << endl;
   compressedSparseRowStorage();  cout << "CSR vectors are created." << endl ;
   gaussQuad();
   calcShape();
   calcJacobian();
   initGlobalSysVariables();
   solve();
   time (&end);      // Stop measuring execution time.

   cout << endl << "Elapsed wall clock time is " << difftime (end,start) << " seconds." << endl;
   cout << endl << "The program is terminated successfully.\nPress a key to close this window...";

   cin.get();
   return 0;

} // End of function main()




//------------------------------------------------------------------------------
void readInput()
//------------------------------------------------------------------------------
{
   // Reads the input file

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
      BCstrings[i] = new double[3];    // TODO : Later these should be strings that can have x, y and z in them.
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
void calcElemSize()
//------------------------------------------------------------------------------
{
   // Calculates the diameter of the circumcircle around the tetrahedral and
   // hexahedral elements. It is used for GLS stabilization.
   
   int i, j, k;
   double maxDist;
   elem_he = new double[NE];

   if (eType == 3) {    // 3D Hexahedron element
      double *distance = new double[4];
      double **points = new double*[8];
      for(i=0; i<8; i++) {
         points[i] = new double[3];
      }      
      
      for (i=0; i < NE; i++) {

         for (j=0; j<8; j++) {
            for (k=0; k<3; k++) {
               points[j][k] = coord[LtoG[i][j]][k];
            }
         }

         distance[0] = sqrt((points[0][0]-points[6][0])*(points[0][0]-points[6][0])+
                            (points[0][1]-points[6][1])*(points[0][1]-points[6][1])+
                            (points[0][2]-points[6][2])*(points[0][2]-points[6][2]));
                                           
         distance[1] = sqrt((points[1][0]-points[7][0])*(points[1][0]-points[7][0])+
                            (points[1][1]-points[7][1])*(points[1][1]-points[7][1])+
                            (points[1][2]-points[7][2])*(points[1][2]-points[7][2]));         
                                           
         distance[2] = sqrt((points[2][0]-points[4][0])*(points[2][0]-points[4][0])+
                            (points[2][1]-points[4][1])*(points[2][1]-points[4][1])+
                            (points[2][2]-points[4][2])*(points[2][2]-points[4][2]));       

         distance[3] = sqrt((points[3][0]-points[5][0])*(points[3][0]-points[5][0])+
                            (points[3][1]-points[5][1])*(points[3][1]-points[5][1])+
                            (points[3][2]-points[5][2])*(points[3][2]-points[5][2]));       
                                           
         maxDist = distance[0];
         
         for (j=1; j<4; j++) {
            if (distance[j] > maxDist) {
               maxDist = distance[j];
            }
         }
         
         elem_he[i] = maxDist;
         
      }
      
   }
   else if (eType == 4) {    // 3D Tetrahedron element
      double *distance = new double[6];
      double **points = new double*[4];
      for(i=0; i<4; i++) {
         points[i] = new double[3];
      }
      
      for (i=0; i < NE; i++) {
         
         for (j=0; j<4; j++) {
            for (k=0; k<3; k++) {
               points[j][k] = coord[LtoG[i][j]][k];
            }
         }
         
         distance[0] = sqrt((points[0][0]-points[1][0])*(points[0][0]-points[1][0])+
                            (points[0][1]-points[1][1])*(points[0][1]-points[1][1])+
                            (points[0][2]-points[1][2])*(points[0][2]-points[1][2]));
                                           
         distance[1] = sqrt((points[0][0]-points[2][0])*(points[0][0]-points[2][0])+
                            (points[0][1]-points[2][1])*(points[0][1]-points[2][1])+
                            (points[0][2]-points[2][2])*(points[0][2]-points[2][2]));         
                                           
         distance[2] = sqrt((points[0][0]-points[3][0])*(points[0][0]-points[3][0])+
                            (points[0][1]-points[3][1])*(points[0][1]-points[3][1])+
                            (points[0][2]-points[3][2])*(points[0][2]-points[3][2]));       

         distance[3] = sqrt((points[1][0]-points[2][0])*(points[1][0]-points[2][0])+
                            (points[1][1]-points[2][1])*(points[1][1]-points[2][1])+
                            (points[1][2]-points[2][2])*(points[1][2]-points[2][2]));       

         distance[4] = sqrt((points[1][0]-points[3][0])*(points[1][0]-points[3][0])+
                            (points[1][1]-points[3][1])*(points[1][1]-points[3][1])+
                            (points[1][2]-points[3][2])*(points[1][2]-points[3][2]));                

         distance[5] = sqrt((points[2][0]-points[3][0])*(points[2][0]-points[3][0])+
                            (points[2][1]-points[3][1])*(points[2][1]-points[3][1])+
                            (points[2][2]-points[3][2])*(points[2][2]-points[3][2]));                                                    
         
         maxDist = distance[0];
         
         for (j=0; j<6; j++) {
            if (distance[j] > maxDist) {
               maxDist = distance[j];
            }
         }
         elem_he[i] = maxDist;      
      }
   }  // Endif eType 
}




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

   rowStarts = new int[Ndof+1];     // Stores the number of nonzeros at each rows of [K]
   rowStartsSmall = new int[NN+1];  // rowStarts for a small part of K(only for "u" velocity in another words 1/16 of the K(if NENv==NENp)) 
   checkCol = new int[1000];        // For checking the non zero column overlaps.
                                    // Stores non-zero column number for rows (must be a large enough value)
   rowStarts[0] = 0;

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
      rowStarts[i+1] = NNZ + rowStarts[i];
   }

   // Create rowStarts from rowStarsSmall
   for (i=0; i<=NN; i++) {
      rowStartsSmall[i] = rowStarts[i];
   }
 
   for (i=1; i<=NN; i++) {
      rowStarts[i] = rowStarts[i]*4;
   }
   
   for (i=1; i<=NN*3; i++) {
      rowStarts[NN+i] = rowStarts[NN] + rowStarts[i];
   }
   
   delete[] checkCol;


   // Create col & colSmall

   col = new int[rowStarts[Ndof]];       // Stores which non zero columns at which row data  TODO: Unclear comment.
   colSmall = new int[rowStarts[NN]/4];  // col for a part of K(only for "u" velocity in another words 1/16 of the K(if NENv==NENp)) 

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
   
   // Create col from colSmall
   m=0;
   for (i=0; i<NN; i++) {
      for (k=0; k<(4*NN); k=k+NN) {
         for (j=rowStartsSmall[i]; j<rowStartsSmall[i+1]; j++) {
            col[m]= colSmall[j] + k;
            m++;
         }
      }  
   }   
   
   for (i=0; i<rowStarts[NN]*3; i++) {
      col[i+rowStarts[NN]]=col[i];
   }


   ////----------------------CONTROL---------------------------------------

   //cout << endl;
   //for (i=0; i<5; i++) { 
   //   for (j=rowStartsSmall[i]; j<rowStartsSmall[i+1]; j++) {
   //      cout << colSmall[j] << " " ;   
   //   }
   //   cout << endl;
   //}
   //cout << endl;
   //cout << endl;
   //for (i=0; i<5; i++) { 
   //   for (j=rowStarts[i]; j<rowStarts[i+1]; j++) {
   //      cout << col[j] << " " ;   
   //   }
   //   cout << endl;
   //}

   ////----------------------CONTROL---------------------------------------


   NNZ = rowStarts[Ndof];

   // Allocate and initialize the val vector
   val = new real[rowStarts[Ndof]];

   for(i=0; i<rowStarts[Ndof]; i++) {
      val[i] = 0;
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
   // Allocate the memory to store elemental stiffness matrices and force vectors
   
   int i;
   
   uOld = new real[Ndof];   // Values of the unknowns from the previous nonlinear iteration
   u = new real[Ndof];

   if(isRestart) {
      readRestartFile();
   } else {
      for (i=0; i<Ndof; i++) {
         u[i] = 0.0;
      }
   }

   F = new real[Ndof];

   // TODO: Shouldn't all the following variables be real, but not double?
   
   Fe = new double[NEU];
   Ke = new double*[NEU];
   for (i=0; i<NEU; i++) {
      Ke[i] = new double[NEU];
   }
   
   Fe_1 = new double[NENv];
   Fe_2 = new double[NENv];
   Fe_3 = new double[NENv];
   Fe_4 = new double[NENp];
   
   Fe_1_add = new double[NENv];
   Fe_2_add = new double[NENv];
   Fe_3_add = new double[NENv];
   Fe_4_add = new double[NENp];
   
   Ke_11 = new double*[NENv];
   Ke_12 = new double*[NENv];
   Ke_13 = new double*[NENv];
   Ke_14 = new double*[NENv];
   Ke_21 = new double*[NENv];
   Ke_22 = new double*[NENv];
   Ke_23 = new double*[NENv];
   Ke_24 = new double*[NENv];
   Ke_31 = new double*[NENv];
   Ke_32 = new double*[NENv];
   Ke_33 = new double*[NENv];
   Ke_34 = new double*[NENv];
   Ke_41 = new double*[NENv];
   Ke_42 = new double*[NENv];
   Ke_43 = new double*[NENv];
   Ke_44 = new double*[NENp];
   
   Ke_11_add = new double*[NENv];
   Ke_12_add = new double*[NENv];
   Ke_13_add = new double*[NENv];
   Ke_14_add = new double*[NENv];
   Ke_21_add = new double*[NENv];
   Ke_22_add = new double*[NENv];
   Ke_23_add = new double*[NENv];
   Ke_24_add = new double*[NENv];
   Ke_31_add = new double*[NENv];
   Ke_32_add = new double*[NENv];
   Ke_33_add = new double*[NENv];
   Ke_34_add = new double*[NENv];
   Ke_41_add = new double*[NENp];
   Ke_42_add = new double*[NENp];
   Ke_43_add = new double*[NENp];
   Ke_44_add = new double*[NENp];

   for (i=0; i<NENv; i++) {
      Ke_11[i] = new double[NENv];
      Ke_12[i] = new double[NENv];
      Ke_13[i] = new double[NENv];
      Ke_14[i] = new double[NENp];
      Ke_21[i] = new double[NENv];
      Ke_22[i] = new double[NENv];
      Ke_23[i] = new double[NENv];
      Ke_24[i] = new double[NENp];
      Ke_31[i] = new double[NENv];
      Ke_32[i] = new double[NENp];
      Ke_33[i] = new double[NENv];
      Ke_34[i] = new double[NENp];
      
      Ke_11_add[i] = new double[NENv];
      Ke_12_add[i] = new double[NENv];
      Ke_13_add[i] = new double[NENv];
      Ke_14_add[i] = new double[NENp];
      Ke_21_add[i] = new double[NENv];
      Ke_22_add[i] = new double[NENv];
      Ke_23_add[i] = new double[NENv];
      Ke_24_add[i] = new double[NENp];
      Ke_31_add[i] = new double[NENv];
      Ke_32_add[i] = new double[NENp];
      Ke_33_add[i] = new double[NENv];
      Ke_34_add[i] = new double[NENp];
   }

   for (i=0; i<NENp; i++) {
      Ke_41[i] = new double[NENv];
      Ke_42[i] = new double[NENv];
      Ke_43[i] = new double[NENv];
      Ke_44[i] = new double[NENp];
      Ke_41_add[i] = new double[NENp];
      Ke_42_add[i] = new double[NENp];
      Ke_43_add[i] = new double[NENp];
      Ke_44_add[i] = new double[NENp];
   }
   
   //are for easiness when calculating elemental stiffness matrix[Ke]
   //keeping velocity values in small arrays instead of using full array
   uNodal = new double[NENv];    
   vNodal = new double[NENv];
   wNodal = new double[NENv];

   //keeping rate of change of velocities in x, y, z directions.
   Du0 = new double[3];    
   Dv0 = new double[3];
   Dw0 = new double[3];

}  // End of function initGlobalSysVariables()




//------------------------------------------------------------------------------
void calcGlobalSys()
//------------------------------------------------------------------------------
{
   // Calculates Ke and Fe one by one for each element and assembles them into
   // the global K (stored in CSR format) and F.

   int e, i, j, k, m, n, node;
   double Tau;
   //   double x, y, axy, fxy;

   // Initialize the arrays

   for (i=0; i<Ndof; i++) {
      F[i] = 0;
   }

   for(i=0; i<rowStarts[Ndof]; i++) {
      val[i] = 0;
   }
   
   // Calculating the elemental stiffness matrix(Ke) and force vector(Fe)

   for (e = 0; e<NE; e++) {

      // Intitialize Ke and Fe to zero.
      for (i=0; i<NEU; i++) {
         Fe[i] = 0;
         for (j=0; j<NEU; j++) {
            Ke[i][j] = 0;
         }
      }

      for (i=0; i<NENv; i++) {
         Fe_1[i] = 0;
         Fe_2[i] = 0;
         Fe_3[i] = 0;
         Fe_4[i] = 0;
         for (j=0; j<NENv; j++) {
            Ke_11[i][j] = 0;
            Ke_12[i][j] = 0;
            Ke_13[i][j] = 0;
            Ke_21[i][j] = 0;
            Ke_22[i][j] = 0;
            Ke_23[i][j] = 0;
            Ke_31[i][j] = 0;
            Ke_32[i][j] = 0;
            Ke_33[i][j] = 0;
         }
         for (j=0; j<NENp; j++) {
            Ke_14[i][j] = 0;
            Ke_24[i][j] = 0;
            Ke_34[i][j] = 0;
         }         
      }   
      for (i=0; i<NENp; i++) {
         for (j=0; j<NENp; j++) {
            Ke_41[i][j] = 0;
            Ke_42[i][j] = 0;
            Ke_43[i][j] = 0;
            Ke_44[i][j] = 0;
         }
      }
      
      for (i=0; i<NENv; i++) {
         uNodal[i] = u[LtoG[e][i]]; 
         vNodal[i] = u[LtoG[e][i]+NN];
         wNodal[i] = u[LtoG[e][i]+2*NN];
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
            Fe_1_add[i] = 0;
            Fe_2_add[i] = 0;
            Fe_3_add[i] = 0;
            Fe_4_add[i] = 0;
            for (j=0; j<NENv; j++) {
               Ke_11_add[i][j] = 0;
               Ke_12_add[i][j] = 0;
               Ke_13_add[i][j] = 0;
               Ke_21_add[i][j] = 0;
               Ke_22_add[i][j] = 0;
               Ke_23_add[i][j] = 0;
               Ke_31_add[i][j] = 0;
               Ke_32_add[i][j] = 0;
               Ke_33_add[i][j] = 0;
            }
            for (j=0; j<NENp; j++) {
               Ke_14_add[i][j] = 0;
               Ke_24_add[i][j] = 0;
               Ke_34_add[i][j] = 0;
            }         
         }
         for (i=0; i<NENp; i++) {
            for (j=0; j<NENp; j++) {
               Ke_41_add[i][j] = 0;
               Ke_42_add[i][j] = 0;
               Ke_43_add[i][j] = 0;
               Ke_44_add[i][j] = 0;
            }
         }

         if (iter<100000) { // uses Picard iteration for initial values 
         
            for (i=0; i<NENv; i++) {
               for (j=0; j<NENv; j++) {
                  Ke_11_add[i][j] = Ke_11_add[i][j] + viscosity * (
                        2 * gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                        gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                        gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                        density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);
                        
                  Ke_12_add[i][j] = Ke_12_add[i][j] + viscosity * gDSv[e][1][i][k] * gDSv[e][0][j][k];
                  
                  Ke_13_add[i][j] = Ke_13_add[i][j] + viscosity * gDSv[e][2][i][k] * gDSv[e][0][j][k];
                  
                  Ke_22_add[i][j] = Ke_22_add[i][j] + viscosity * (
                        gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                        2 * gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                        gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                        density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);
                        
                  Ke_23_add[i][j] = Ke_23_add[i][j] + viscosity * gDSv[e][2][i][k] * gDSv[e][1][j][k];
                  
                  Ke_33_add[i][j] = Ke_33_add[i][j] + viscosity * (
                        gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                        gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                        2 * gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                        density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);
                        
               }
               for (j=0; j<NENp; j++) {
                  Ke_14_add[i][j] = Ke_14_add[i][j] - gDSv[e][0][i][k] * Sp[j][k];
                  Ke_24_add[i][j] = Ke_24_add[i][j] - gDSv[e][1][i][k] * Sp[j][k];
                  Ke_34_add[i][j] = Ke_34_add[i][j] - gDSv[e][2][i][k] * Sp[j][k];
               }         
            }
            
            for (i=0; i<NENv; i++) {
               for (j=0; j<NENp; j++) {
                  Ke_41_add[j][i] = Ke_14_add[i][j] ;
                  Ke_42_add[j][i] = Ke_24_add[i][j] ;
                  Ke_43_add[j][i] = Ke_34_add[i][j] ;
               }
            }

         }
         
         else {

            for (i=0; i<NENv; i++) {
               Fe_1_add[i] = Fe_1_add[i] + density * Sv[i][k] * (u0 * Du0[0] + v0 * Du0[1] + w0 * Du0[2]);
               Fe_2_add[i] = Fe_2_add[i] + density * Sv[i][k] * (u0 * Dv0[0] + v0 * Dv0[1] + w0 * Dv0[2]); 
               Fe_3_add[i] = Fe_3_add[i] + density * Sv[i][k] * (u0 * Dw0[0] + v0 * Dw0[1] + w0 * Dw0[2]);
               for (j=0; j<NENv; j++) {
                  Ke_11_add[i][j] = Ke_11_add[i][j] + viscosity * (
                        2 * gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                        gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                        gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                        density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]) +
                        density * Sv[i][k] * Sv[j][k] * Du0[0] ;                    
                  
                        
                  Ke_12_add[i][j] = Ke_12_add[i][j] + viscosity * gDSv[e][1][i][k] * gDSv[e][0][j][k] +
                        density * Sv[i][k] * Sv[j][k] * Du0[1] ;
                  
                  Ke_13_add[i][j] = Ke_13_add[i][j] + viscosity * gDSv[e][2][i][k] * gDSv[e][0][j][k] +
                        density * Sv[i][k] * Sv[j][k] * Du0[2] ;
                        
                  Ke_21_add[i][j] = Ke_21_add[i][j] + viscosity * gDSv[e][1][i][k] * gDSv[e][0][j][k] +
                        density * Sv[i][k] * Sv[j][k] * Dv0[0] ;     
                  
                  Ke_22_add[i][j] = Ke_22_add[i][j] + viscosity * (
                        gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                        2 * gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                        gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                        density * Sv[i][k]  * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k])+
                        density * Sv[i][k] * Sv[j][k] * Dv0[1] ;
                                     
                        
                  Ke_23_add[i][j] = Ke_23_add[i][j] + viscosity * gDSv[e][2][i][k] * gDSv[e][1][j][k] +
                        density * Sv[i][k] * Sv[j][k] * Dv0[2] ;
                  
                  Ke_31_add[i][j] = Ke_31_add[i][j] + viscosity * gDSv[e][2][i][k] * gDSv[e][0][j][k] +
                        density * Sv[i][k] * Sv[j][k] * Dw0[0] ;
                        
                  Ke_32_add[i][j] = Ke_32_add[i][j] + viscosity * gDSv[e][2][i][k] * gDSv[e][1][j][k] +
                        density * Sv[i][k] * Sv[j][k] * Dw0[1] ;      
                  
                  Ke_33_add[i][j] = Ke_33_add[i][j] + viscosity * (
                        gDSv[e][0][i][k] * gDSv[e][0][j][k] + 
                        gDSv[e][1][i][k] * gDSv[e][1][j][k] +
                        2 * gDSv[e][2][i][k] * gDSv[e][2][j][k]) +
                        density * Sv[i][k] * (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]) +
                        density * Sv[i][k] * Sv[j][k] * Dw0[2] ;                               
               }
               for (j=0; j<NENp; j++) {
                  Ke_14_add[i][j] = Ke_14_add[i][j] - gDSv[e][0][i][k] * Sp[j][k];
                  Ke_24_add[i][j] = Ke_24_add[i][j] - gDSv[e][1][i][k] * Sp[j][k];
                  Ke_34_add[i][j] = Ke_34_add[i][j] - gDSv[e][2][i][k] * Sp[j][k];
               }         
            }
            
            for (i=0; i<NENv; i++) {
               for (j=0; j<NENp; j++) {
                  Ke_41_add[j][i] = Ke_14_add[i][j] ;
                  Ke_42_add[j][i] = Ke_24_add[i][j] ;
                  Ke_43_add[j][i] = Ke_34_add[i][j] ;
               }
            }

         }

         // Apply GLS stabilization for linear elements with NENv = NENp
         Tau = (1.0/12.0)*elem_he[e]*elem_he[e] / viscosity; // GLS parameter
         // Tau = pow( pow( (2*pow(u0*u0+v0*v0+w0*w0 , 0.5))/elem_he[e], 2) + pow((4*viscosity)/(elem_he[e]*elem_he[e]) , 2) , -0.5);  // GLS parameter TODO new GLS parameter
         // source: "A Review of Petrov-Galerkin Stabilization Approaches and an Extension to Meshfree Methods" [Thomas-Peter Fries, Hermann G. Matthies]

         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {
              Ke_11_add[i][j] = Ke_11_add[i][j] + Tau * density * density *
                             (u0 * gDSv[e][0][i][k] + v0 * gDSv[e][1][i][k] + w0 * gDSv[e][2][i][k]) *
                             (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);                             
             
              Ke_22_add[i][j] = Ke_22_add[i][j] + Tau * density * density * 
                             (u0 * gDSv[e][0][i][k] + v0 * gDSv[e][1][i][k] + w0 * gDSv[e][2][i][k]) *
                             (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);   
                             
              Ke_33_add[i][j] = Ke_33_add[i][j] + Tau * density * density * 
                             (u0 * gDSv[e][0][i][k] + v0 * gDSv[e][1][i][k] + w0 * gDSv[e][2][i][k]) *
                             (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]);                                
            } 
         } 

         for (i=0; i<NENv; i++) {
            for (j=0; j<NENp; j++) {
              Ke_14_add[i][j] = Ke_14_add[i][j] + Tau * density * 
                             (u0 * gDSv[e][0][i][k] + v0 * gDSv[e][1][i][k] + w0 * gDSv[e][2][i][k]) * gDSp[e][0][j][k]; 
             
              Ke_24_add[i][j] = Ke_24_add[i][j] + Tau * density * 
                             (u0 * gDSv[e][0][i][k] + v0 * gDSv[e][1][i][k] + w0 * gDSv[e][2][i][k]) * gDSp[e][1][j][k]; 
                             
              Ke_34_add[i][j] = Ke_34_add[i][j] + Tau * density * 
                             (u0 * gDSv[e][0][i][k] + v0 * gDSv[e][1][i][k] + w0 * gDSv[e][2][i][k]) * gDSp[e][2][j][k];                              
            } 
         } 

         for (i=0; i<NENp; i++) {
            for (j=0; j<NENv; j++) {
              Ke_41_add[i][j] = Ke_41_add[i][j] - Tau * density * 
                             (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]) * gDSp[e][0][i][k]; 
             
              Ke_42_add[i][j] = Ke_42_add[i][j] - Tau * density * 
                             (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]) * gDSp[e][1][i][k]; 
                             
              Ke_43_add[i][j] = Ke_43_add[i][j] - Tau * density * 
                             (u0 * gDSv[e][0][j][k] + v0 * gDSv[e][1][j][k] + w0 * gDSv[e][2][j][k]) * gDSp[e][2][i][k];        
            } 
         } 
         
         for (i=0; i<NENp; i++) {
            for (j=0; j<NENp; j++) {
               Ke_44_add[i][j] = Ke_44_add[i][j] - Tau *
                              ( gDSv[e][0][i][k] * gDSv[e][0][j][k] + gDSv[e][1][i][k] * gDSv[e][1][j][k] + gDSv[e][2][i][k] * gDSv[e][2][j][k] );
            }
         }
         
         //-------------------------------------------------------------------------   

            
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {            
               Ke_11[i][j] += Ke_11_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_12[i][j] += Ke_12_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_13[i][j] += Ke_13_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_21[i][j] += Ke_21_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_22[i][j] += Ke_22_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_23[i][j] += Ke_23_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_31[i][j] += Ke_31_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_32[i][j] += Ke_32_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_33[i][j] += Ke_33_add[i][j] * detJacob[e][k] * GQweight[k];
            }
            for (j=0; j<NENp; j++) {               
               Ke_14[i][j] += Ke_14_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_24[i][j] += Ke_24_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_34[i][j] += Ke_34_add[i][j] * detJacob[e][k] * GQweight[k];
            }    
            Fe_1[i] +=  Fe_1_add[i] * detJacob[e][k] * GQweight[k];
            Fe_2[i] +=  Fe_2_add[i] * detJacob[e][k] * GQweight[k];
            Fe_3[i] +=  Fe_3_add[i] * detJacob[e][k] * GQweight[k];
         }
         for (i=0; i<NENv; i++) {
            for (j=0; j<NENv; j++) {
               Ke_41[i][j] += Ke_41_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_42[i][j] += Ke_42_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_43[i][j] += Ke_43_add[i][j] * detJacob[e][k] * GQweight[k];
               Ke_44[i][j] += Ke_44_add[i][j] * detJacob[e][k] * GQweight[k];
            }
         }

      }   // End GQ loop  

      // Assembly of Fe

      i=0;
      for (j=0; j<NENv; j++) {
         Fe[i]=Fe_1[j];
         i++;
      }
      for (j=0; j<NENv; j++) {
         Fe[i]=Fe_2[j];
         i++;
      }
      for (j=0; j<NENv; j++) {
         Fe[i]=Fe_3[j];
         i++;
      }
      for (j=0; j<NENp; j++) {
         Fe[i]=Fe_4[j];
         i++;
      }         

      // Assembly of Ke

      i=0;
      j=0;    
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENv; n++) {       
            Ke[i][j] =  Ke_11[m][n];
            j++;
         }
         i++;
      }
      
      i=i-NENv;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENv; n++) {       
            Ke[i][j+NENv] =  Ke_12[m][n];
            j++;
         }
         i++;
      }

      i=i-NENv;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENv; n++) {       
            Ke[i][j+2*NENv] =  Ke_13[m][n];
            j++;
         }
         i++;
      }
      
      i=i-NENv;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENp; n++) {       
            Ke[i][j+3*NENv] =  Ke_14[m][n];
            j++;
         }
         i++;
      }
      
      if (iter<100000) {
         for (m=0; m<NENv; m++) {
            j=0;
            for (n=0; n<NENv; n++) {  
               Ke[i][j] =  Ke_12[n][m];
               j++;
            }
            i++;
         }
      }
      else {
            for (m=0; m<NENv; m++) {
               j=0;
               for (n=0; n<NENv; n++) {  
                  Ke[i][j] =  Ke_21[m][n];
                  j++;
               }
            i++;
            }
      }

      i=i-NENv;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENv; n++) {       
            Ke[i][j+NENv] =  Ke_22[m][n];
            j++;
         }
         i++;
      }

      i=i-NENv;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENv; n++) {       
            Ke[i][j+2*NENv] =  Ke_23[m][n];
            j++;
         }
         i++;
      }

      i=i-NENv;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENp; n++) {       
            Ke[i][j+3*NENv] =  Ke_24[m][n];
            j++;
         }
         i++;
      }

      if (iter<100000) {
         for (m=0; m<NENv; m++) {
            j=0;
            for (n=0; n<NENv; n++) {       
               Ke[i][j] =  Ke_13[n][m];
               j++;
            }
            i++;
         }   
      }
      else {
         for (m=0; m<NENv; m++) {
            j=0;
            for (n=0; n<NENv; n++) {       
               Ke[i][j] =  Ke_31[m][n];
               j++;
            }
            i++;
         }  
      }

      if (iter<100000) {
         i=i-NENv;
         for (m=0; m<NENv; m++) {
            j=0;
            for (n=0; n<NENv; n++) {       
               Ke[i][j+NENv] =  Ke_23[n][m];
               j++;
            }
            i++;
         }  
      }
      else {      
         i=i-NENv;
         for (m=0; m<NENv; m++) {
            j=0;
            for (n=0; n<NENv; n++) {       
               Ke[i][j+NENv] =  Ke_32[m][n];
               j++;
            }
            i++;
         } 
      }

      i=i-NENv;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENv; n++) {       
            Ke[i][j+2*NENv] =  Ke_33[m][n];
            j++;
         }
         i++;
      }

      i=i-NENv;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENp; n++) {       
            Ke[i][j+3*NENv] =  Ke_34[m][n];
            j++;
         }
         i++;
      }

      j=0;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENp; n++) {       
            Ke[i][j] =  Ke_41[m][n];
            j++;
         }
         i++;
      }
      
      i=i-NENv;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENp; n++) {       
            Ke[i][j+NENv] =  Ke_42[m][n];
            j++;
         }
         i++;
      } 

      i=i-NENv;
      for (m=0; m<NENv; m++) {
         j=0;
         for (n=0; n<NENp; n++) {       
            Ke[i][j+2*NENv] =  Ke_43[m][n];
            j++;
         }
         i++;
      } 

      i=i-NENv;
      for (m=0; m<NENp; m++) {
         j=0;
         for (n=0; n<NENp; n++) {       
            Ke[i][j+3*NENv] = Ke_44[m][n];
            j++;
         }
         i++;
      }

      assemble(e, Ke, Fe);  // sending Ke & Fe for assembly

   }  // End of element loop
 
} // End of function calcGlobalSys()




//------------------------------------------------------------------------------
void assemble(int e, double **Ke, double *Fe)
//------------------------------------------------------------------------------
{
   // Inserts Fe into proper locations of F by the help of LtoG array. Also
   // inserts Ke into the val vector of CSR storage using KeKmapSmall.

   // TODO: Does this assembly work for the case of NENp not being equal to NENv?
   //       If not, put a WARNING here.

   int i, j, iG, jG; 
   
   for (i = 0; i<NEU; i++) {
      iG = LtoG[e][i%NENv] + NN*(i/NENv);
      F[iG] = F[iG] + Fe[i];
   }   

   // Create KeKMapSmall, which stores the mapping between the entries of Ke
   // and val vector of CSR.

   // TODO: Why do we calculate KeKmapSmall in each iteration again and again? Isn't it costly?
   //       Is it feasible to calculate it for each element only once and store?

   int shiftRow, shiftCol;
   int *eLtoG, p, colCounter, k;
    
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
         for(p=rowStartsSmall[eLtoG[i]]; p<rowStartsSmall[eLtoG[i]+1]; p++) {  // p is the location of the col vector(col[x], p=x) 
            if(colSmall[p] == eLtoG[j]) {                                         // Selection process of the KeKMapSmall data from the col vector
               KeKMapSmall[i][j] = colCounter; 
               break;
            }
            colCounter++;
         }
      }
   }

   // Insert Ke into the val vector of CSR.
   for(shiftRow=1; shiftRow<=4; shiftRow++) {      // 4 is the number of unknowns(u,v,w,p)
      for(shiftCol=1; shiftCol<=4; shiftCol++) {   // 4 is the number of unknowns(u,v,w,p)
         for(i=0; i<NENv; i++) {
            for(j=0; j<NENv; j++) {
               val[rowStarts[LtoG[e][i] + NN * (shiftRow-1)] + ((rowStartsSmall[LtoG[e][i]+1] -    // TODO: What is going on here? Explain on paper.
                   rowStartsSmall[LtoG[e][i]]) * (shiftCol-1)) + KeKMapSmall[i][j]] +=
                   Ke[ i + ((shiftRow-1) * NENv) ][ j + ((shiftCol-1) * NENv) ] ;
            }
         }
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
   // For EBCs reduction is not applied. Instead val vector of CSR and F are
   // modified. TODO: Modified how? What do we do?

   int i, j, whichBC, node;
   //double x, y, z;

   int p, colCounter;

   // Modify CSR vectors for velocity BCs
   for (i = 0; i<nVelNodes; i++) {
      node = velNodes[i][0];         // Node at which this EBC is specified

      //x = coord[node][0];          // May be necessary for BCstring evaluation. But for the time being BCstrings are not strings but constant values.
      //y = coord[node][1];
      //z = coord[node][2];

      colCounter = 0;
      for(p=rowStartsSmall[node]; p<rowStartsSmall[node+1]; p++) {   // p is the location of the col vector(col[x], p=x)
         if(colSmall[p] == node) {                                   // Selection process of the KeKMapSmall data from the col vector.
            break;
         }
         colCounter++;
      }

      whichBC = velNodes[i][1]-1;         // Number of the specified BC

      // Modify val and F for the specified u velocity.
      for (j=rowStarts[node]; j<rowStarts[node+1]; j++) {
         val[j] = 0.0;                    // Set non-diagonal entries to 0.
      }
      val[ rowStarts[node] + colCounter ] = 1;     // Set the diagonal entry to 1.

      F[node] = BCstrings[whichBC][0];    // Specified u velocity value.

      // Modify val and F for the specified v velocity.
      for (j=rowStarts[node+NN]; j<rowStarts[node+1+NN]; j++) {
         val[j] = 0.0;
      }
      val[ rowStarts[node + NN] + (rowStartsSmall[node+1]- rowStartsSmall[node]) + colCounter ] = 1;

      F[node + NN] = BCstrings[whichBC][1];    // Specified v velocity value.

      // Modify val and F for the specified w velocity.
      for (j=rowStarts[node+2*NN]; j<rowStarts[node+1+2*NN]; j++) {
         val[j] = 0.0;
      }
      val[ rowStarts[node+2*NN] + ((rowStartsSmall[node+1]- rowStartsSmall[node]) * 2) + colCounter ] = 1;

      F[node + NN*2] = BCstrings[whichBC][2];    // Specified w velocity value.
   }   


   // Modify val and F for pressure BCs
   for (i = 0; i<nPressureNodes; i++) {
      node = pressureNodes[i][0];         // Node at which pressure BC is specified
   	
      // x = coord[node][0];              // May be necessary for BCstring evaluation. But for the time being BCstrings are not strings but constant values.
      // y = coord[node][1];
      // z = coord[node][2];

      colCounter = 0;
      for(p=rowStartsSmall[node]; p<rowStartsSmall[node+1]; p++) {   // p is the location of the col vector(col[x], p=x) 
         if(colSmall[p] == node) {                                   // Selection process of the KeKMapSmall data from the col vector.
            break; 
         }
         colCounter++;
      }
      
      whichBC = pressureNodes[i][1] - 1;         // Number of the specified BC   	
      
      for (j=rowStarts[node + 3*NN]; j<rowStarts[node+1 + 3*NN]; j++) {
         val[j] = 0.0;                           // Set non-digonal entries to 0.
      }
      val[ rowStarts[node+3*NN] + ((rowStartsSmall[node+1]- rowStartsSmall[node]) * 3) + colCounter ] = 1;     // Set the digonal entry to 1.
      
      F[node + NN*3] = BCstrings[whichBC][0];    // Specified pressure value
   }

} // End of function applyBC()




//------------------------------------------------------------------------------
void solve()
//------------------------------------------------------------------------------
{
   // Creates and solves the global system of equations in a nonlinear iteration
   // loop using either CUDA on GPU or Pardiso on CPU.

   bool err;
   int i, j;
   //time_t start, end;

   //----------------CONTROL-----------------------------

   // Creates an output file, named "Control_Output" for K and F
   // outputControl.open(controlFile.c_str(), ios::out);
   // outputControl << NN << endl;
   // outputControl << endl;
   // outputControl.precision(4);
   // for(i=0; i<4*NN; i++) {
      // for(j=0; j<4*NN; j++) {
         // outputControl << fixed << "\t" << K[i][j]  ;   
      // }
      // outputControl << fixed << "\t" << F[i] << endl;
   // }
   // outputControl.close();

   //----------------CONTROL-----------------------------

   
   cout << endl << " Iter |  Max. Change  | Solver Iter | Solver Norm | Mon Node |    Mon u    |    Mon v    |    Mon w    |    Mon p";
   cout << endl << "=====================================================================================================================" << endl;

   // Linearization loop starts here.

   for (iter=1; iter<=nonlinearIterMax; iter++) {
      //cout << "Calculating the global system ... ";
      //time (&start);
      calcGlobalSys();
      //time (&end);
      //cout << "Done. Elapsed wall clock time is " << difftime (end,start) << " seconds." << endl;

      //cout << "Applying the BCs ... ";
      //time (&start);
      applyBC();
      //time (&end);
      //cout << "Done. Elapsed wall clock time is " << difftime (end,start) << " seconds." << endl;

      #ifdef CUSP
         //cout << endl << "CUSPsolver() function is started ... " << endl;
         //time (&start);
         CUSPsolver();
         //time (&end);
         //cout << "Done. Elapsed wall clock time is " << difftime (end,start) << " seconds." << endl;
      #endif

      #ifdef MKLPARDISO
         //cout << endl << "pardisoSolver() function is started ... " << endl;
         //time (&start);
         pardisoSolver();
         //time (&end);
         //cout << "Done. Elapsed wall clock time is " << difftime (end,start) << " seconds." << endl;
      #endif


      //----------------CONTROL-----------------------------

      // Printing velocity values after each iteration
   
      //if (eType == 3) {
      //   for (int i = 0; i<NN; i++) { 
      //      printf("%-5d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n", i, coord[i][0], coord[i][1], coord[i][2], u[i], u[i+NN], u[i+NN*2], u[i+NN*3]);
      //   }
      //}

      //----------------CONTROL-----------------------------


      // Calculate maximum change in the unknowns with respect to the values of the previous iteration.
      double change, maxChange;

      maxChange = abs(u[0] - uOld[0]);

      for (i=1; i<Ndof; i++) {
         change = abs(u[i] - uOld[i]);
         
         if (change > maxChange) {
            maxChange = change;
         }
      }

      printf("%5d %14.5e %11d %15.3e", iter, maxChange, solverIter, solverNorm);

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
  
      if (maxChange < nonlinearTol) {
         break;
      }
       
      for (i=0; i<Ndof; i++) {
         uOld[i] = u[i];
      }

      // Write Tecplot file
      if(iter % nDATiter == 0 || iter == nonlinearIterMax) {
         writeTecplotFile();
         // cout << "A DAT file is created for Tecplot." << endl;
      }

   } // End of iter loop
   
   
   // Give info about convergence
   if (iter > nonlinearIterMax) { 
      cout << endl << "Solution did not converge in " << nonlinearIterMax << " iterations." << endl; 
   }
   else {
      cout << endl << "Convergence is achieved at " << iter << " iterations." << endl;
      writeTecplotFile();
   }   
   
   
   // Deleting the unnecessary arrays for future
   delete[] F;

}  // End of function solve()




//------------------------------------------------------------------------------
void postProcess()
//------------------------------------------------------------------------------
{
   // Write the calculated unknowns on the screen. Used only for CONTROL purposes.


   printf("\nCalculated unknowns are \n\n");
   printf(" Node      x       y       z         u         v         w       p \n");
   printf("========================================================================\n");
   if (eType == 3 || eType == 4) {
      for (int i = 0; i<NN; i++) { 
         printf("%-5d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n", i, coord[i][0],
                coord[i][1], coord[i][2], u[i], u[i+NN], u[i+NN*2], u[i+NN*3]);
      }
   } else { 
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
      restartFile >> dummy >> dummy >> dummy >> u[i] >> u[i+NN] >> u[i+NN*2] >> u[i+NN*3];
      restartFile.ignore(256, '\n');   // Ignore the rest of the line
   }

   restartFile.close();

   // Set uOld to the values read from the restart file.
   for (int i = 0; i<NN; i++) {
      uOld[i] = u[i];
   }

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
      outputFile << scientific  << "\t" << x << " "  << y << " "  << z << " " << u[i] << " " << u[i+NN] << " " << u[i+NN*2] << " " << u[i+NN*3] << endl;
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

