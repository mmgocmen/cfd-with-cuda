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

using namespace std;

#define _USE_MATH_DEFINES     // Required for pi = 3.14... constant 

#ifdef SINGLE
  typedef float real;
#else
  typedef double real;
#endif

ifstream meshfile;
ifstream problemFile;
ofstream outputFile;

string problemNameFile, whichProblem;         // Change this line to specify the name of the problem as seen in input and output files.
string problemName = "ProblemName.txt";
string inputExtension  = ".inp";              // Change this line to specify the extension of input file (with dot).
string outputExtension  = ".dat";             // Change this line to specify the extension of output file (with dot).

int name, eType, NE, NEN, NN, NGP;
int **LtoG, **EBCnodes;
double **coord;
double **NBCfaces;                            // TODO : Why is this double?
int nBC, nEBCnodes, nNBCfaces;
double *BCtype, *BCstrings;
double axyFunc, fxyFunc;
double **GQpoint, *GQweight;
double **S, ***DS;
double **Jacob, **detJacob, **invJacob, ****gDS;

int bigNumber;

int **GtoL, *rowStarts, *col, ***KeKmap, **KeKMapSmall, NNZ; 
real **K, *F, *u;
real *val;

int *rowStartsUpper, *colUpper, NNZupper;     // For symmetric CSR required by MKL 
real *valUpper;                               // For symmetric CSR required by MKL

int solverIterMax, solverIter;
double solverTol, solverNorm;

void readInput();
void gaussQuad();
void calcShape();
void calcJacobian();
void calcGlobalSys();
void assemble(int e, real **Ke, real *Fe);
void applyBC();
void solve();
void postProcess();
void gaussElimination(int N, real **K, real *F, real *u, bool& err);
void writeTecplotFile();
void compressedSparseRowStorage();
double getHighResolutionTime();

#ifdef CUSP
   extern void CUSPsolver();
#endif

#ifdef CUSPARSE
   extern void CUSPARSEsolver();
#endif

#ifdef MKLCG
   extern void MKLCGsolver();
#endif

#ifdef CULA
   extern void CULAsolver();
#endif



//-------------------------------------------------------------
int main()
//-------------------------------------------------------------
{
   double Start, End, Start1, End1;

   Start1 = getHighResolutionTime();


   Start = getHighResolutionTime();
   readInput();
   cout << endl << "The program has read input values" << endl;
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
   calcGlobalSys();
   End = getHighResolutionTime();
   printf("Time for Global   = %-.4g seconds.\n", End - Start);

   Start = getHighResolutionTime();
   applyBC();
   End = getHighResolutionTime();
   printf("Time for BC       = %-.4g seconds.\n", End - Start);

   Start = getHighResolutionTime();
   solve();
   End = getHighResolutionTime();
   printf("Time for Solve    = %-.4g seconds.\n", End - Start);

   //postProcess();

   Start = getHighResolutionTime();
   writeTecplotFile();
   End = getHighResolutionTime();
   printf("Time for Tecplot  = %-.4g seconds.\n", End - Start);


   End1 = getHighResolutionTime();

   printf("total Time        = %-.4g seconds.\n", End1 - Start1);

   cout << endl << "The program is terminated successfully.\nPress a key to close this window...\n";

//   cin.get();
   return 0;

} // End of function main()




//-------------------------------------------------------------
void readInput()
//-------------------------------------------------------------
{

   string dummy, dummy2;
   int dummy3;

   cout << "The program is started" << endl ;
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
   meshfile >> dummy >> dummy2 >> NN;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile >> dummy >> dummy2 >> NEN;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile >> dummy >> dummy2 >> NGP;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile >> dummy >> dummy2 >> solverIterMax;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile >> dummy >> dummy2 >> solverTol;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile >> dummy >> dummy2 >> axyFunc;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile >> dummy >> dummy2 >> fxyFunc;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile.ignore(256, '\n'); // Read and ignore the line
   meshfile.ignore(256, '\n'); // Read and ignore the line   
    
   // Read node coordinates
   coord = new double*[NN];
   
   if (eType == 2 || eType == 1) {
      for (int i = 0; i < NN; i++) {
         coord[i] = new double[2];
      }
   	
      for (int i=0; i<NN; i++){
         meshfile >> dummy3 >> coord[i][0] >> coord[i][1] ;
         meshfile.ignore(256, '\n'); // Ignore the rest of the line    
      }  
   } else {
      for (int i = 0; i < NN; i++) {
         coord[i] = new double[3];
      }
   	
      for (int i=0; i<NN; i++){
         meshfile >> dummy3 >> coord[i][0] >> coord[i][1] >> coord[i][2] ;
         meshfile.ignore(256, '\n'); // Ignore the rest of the line    
      }  
   }
   
   meshfile.ignore(256, '\n'); // Read and ignore the line
   meshfile.ignore(256, '\n'); // Read and ignore the line 
    
   // Read element connectivity, i.e. LtoG
   LtoG = new int*[NE];

   for (int i=0; i<NE; i++) {
      LtoG[i] = new int[NEN];
   }

   for (int e = 0; e<NE; e++){
      meshfile >> dummy3;
      for (int i = 0; i<NEN; i++){
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
   BCstrings = new double[nBC];
    
   for (int i = 0; i<nBC; i++){
      meshfile >> dummy >> BCtype[i] >> dummy2 >> dummy3 >> BCstrings[i];
      meshfile.ignore(256, '\n'); // Ignore the rest of the line
   }
    

   // Read EBC data 
    
   meshfile.ignore(256, '\n'); // Read and ignore the line 
   meshfile >> dummy >> dummy2 >> nEBCnodes;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile >> dummy >> dummy2 >> nNBCfaces;
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
   meshfile.ignore(256, '\n'); // Ignore the rest of the line
      
   if (nEBCnodes!=0){
      EBCnodes = new int*[nEBCnodes];
      for (int i = 0; i < nEBCnodes; i++){
         EBCnodes[i] = new int[2];
      }
      for (int i = 0; i < nEBCnodes; i++){
         meshfile >> EBCnodes[i][0] >> EBCnodes[i][1];   
         meshfile.ignore(256, '\n'); // Ignore the rest of the line
      }
   }
   	
   if (nNBCfaces!=0){
      NBCfaces = new double*[nNBCfaces];
      for (int i = 0; i < nNBCfaces; i++){
         NBCfaces[i] = new double[1];
      }
      // ...
      // TODO : Incomplete
      // ...
   }

   cout << "Input file is read" << endl ;
   meshfile.close();

} // End of function readInput()




//---------------------------------------------------------------------------------
void compressedSparseRowStorage()
//---------------------------------------------------------------------------------
{
//GtoL creation
int i, j, k, m, x, y, valGtoL, check, temp, *checkCol, noOfColGtoL, *GtoLCounter;

   GtoL = new int*[NN];          // stores which elements connected to the node
   noOfColGtoL = 8;              // for 3D meshes created by our mesh generator max 8 elements connect to one node,
   GtoLCounter = new int[NN];    // but for real 3D problems this must be a big enough value!
   
   for (i=0; i<NN; i++) {     
      GtoL[i] = new int[noOfColGtoL];   
   }

   for(i=0; i<NN; i++) {
      for(j=0; j<noOfColGtoL; j++) {
		   GtoL[i][j] = -1;      // for the nodes that didn't connect 4 nodes
	   }
      GtoLCounter[i] = 0;
   }
   
   for (i=0; i<NE; i++) {
      for(j=0; j<NEN; j++) {
         GtoL[ LtoG[i][j] ][ GtoLCounter[LtoG[i][j]] ] = i;
         GtoLCounter[ LtoG[i][j] ] += 1;
      }
   } 
   
   delete[] GtoLCounter;
   
//--------------------------------------------------------------------
//finding size of col vector

   rowStarts = new int[NN+1];     // how many non zeros at rows of [K]
   checkCol = new int[1000];      // for checking the non zero column overlaps 
                                  // stores non-zero column number for rows (must be a large enough value)
   rowStarts[0] = 0;

   for(i=0; i<NN; i++) {
      NNZ = 0;
      
      if(GtoL[i][0] == -1) {
         NNZ = 1;
      } else {
      
         for(k=0; k<1000; k++) {      // prepare checkCol for new row
            checkCol[k] = -1;
         }
      
         for(j=0; j<noOfColGtoL; j++) {
            valGtoL = GtoL[i][j];
            if(valGtoL != -1) {
               for(x=0; x<NEN; x++) {
                  check = 1;         // for checking if column overlap occurs or not
                  for(y=0; y<NNZ; y++) {
                     if(checkCol[y] == (LtoG[valGtoL][x])) {   // this column was created
                        check = 0;
                     }
                  }
                  if (check) {
                     checkCol[NNZ]=(LtoG[valGtoL][x]);         // adding new non zero number to checkCol
                     NNZ++;
                  }
               }
            }   
         }
         
      }
      
      rowStarts[i+1] = NNZ + rowStarts[i];
   }
   

   //--------------------------------------------------------------------
   //col creation

   col = new int[rowStarts[NN]];   // stores which non zero columns at which row data

   for(i=0; i<NN; i++) {
      NNZ = 0;

      if(GtoL[i][0] == -1) {
         col[rowStarts[i]] = i;
      } else {
      
         for(j=0; j<noOfColGtoL; j++) {
            valGtoL = GtoL[i][j];
            if(valGtoL != -1) {
               for(x=0; x<NEN; x++) {
                  check = 1;
                  for(y=0; y<NNZ; y++) {
                     if(col[rowStarts[i]+y] == (LtoG[valGtoL][x])) {   // for checking if column overlap occurs or not
                        check = 0;
                     }
                  }
                  if (check) {
                     col[rowStarts[i]+NNZ] = (LtoG[valGtoL][x]);
                     NNZ++;
                  }
               }
            }   
         }
         
         for(k=1; k<NNZ; k++) {           // sorting the column vector values
            for(m=1; m<NNZ; m++) {        // for each row from smaller to bigger
               if(col[rowStarts[i]+m] < col[rowStarts[i]+m-1]) {
                  temp = col[rowStarts[i]+m];
                  col[rowStarts[i]+m] = col[rowStarts[i]+m-1];
                  col[rowStarts[i]+m-1] = temp;
               }
            }   
         }

      }      
   }
   
   NNZ = rowStarts[NN];

   //initializing val vector
   val = new real[rowStarts[NN]];

   for(i=0; i<rowStarts[NN]; i++) {
      val[i] = 0;
   }

} // End of function compressedSparseRowStorage()





//---------------------------------------------------------------------------------
void gaussQuad()
//---------------------------------------------------------------------------------
{
   // Generates the NGP-point Gauss quadrature points and weights.

   GQpoint = new double*[NGP];
   for (int i=0; i<NGP; i++) {
      GQpoint[i] = new double[3];     
   }
   GQweight = new double[NGP];

   if (eType == 3) {    // 3D Hexahedron element
   
      GQpoint[0][0] = -sqrt(1.0/3);   GQpoint[0][1] = -sqrt(1.0/3);  GQpoint[0][2] = -sqrt(1.0/3); 
      GQpoint[1][0] = sqrt(1.0/3);    GQpoint[1][1] = -sqrt(1.0/3);  GQpoint[1][2] = -sqrt(1.0/3);
      GQpoint[2][0] = sqrt(1.0/3);    GQpoint[2][1] = sqrt(1.0/3);   GQpoint[2][2] = -sqrt(1.0/3); 
      GQpoint[3][0] = -sqrt(1.0/3);   GQpoint[3][1] = sqrt(1.0/3);   GQpoint[3][2] = -sqrt(1.0/3); 
      GQpoint[4][0] = -sqrt(1.0/3);   GQpoint[4][1] = -sqrt(1.0/3);  GQpoint[4][2] = sqrt(1.0/3); 
      GQpoint[5][0] = sqrt(1.0/3);    GQpoint[5][1] = -sqrt(1.0/3);  GQpoint[5][2] = sqrt(1.0/3);
      GQpoint[6][0] = sqrt(1.0/3);    GQpoint[6][1] = sqrt(1.0/3);   GQpoint[6][2] = sqrt(1.0/3); 
      GQpoint[7][0] = -sqrt(1.0/3);   GQpoint[7][1] = sqrt(1.0/3);   GQpoint[7][2] = sqrt(1.0/3);      
      GQweight[0] = 1.0;
      GQweight[1] = 1.0;
      GQweight[2] = 1.0;
      GQweight[3] = 1.0;
      GQweight[4] = 1.0;
      GQweight[5] = 1.0;
      GQweight[6] = 1.0;
      GQweight[7] = 1.0;
       
   }
   else if (eType == 1) {       // Quadrilateral element  3 ile 4 un yeri degismeyecek mi sor!
      if (NGP == 1) {      // One-point quadrature
         GQpoint[0][0] = 0.0;            GQpoint[0][1] = 0.0;
         GQweight[0] = 4.0;
      } else if (NGP == 4) {  // Four-point quadrature
         GQpoint[0][0] = -sqrt(1.0/3);   GQpoint[0][1] = -sqrt(1.0/3);
         GQpoint[1][0] = sqrt(1.0/3);    GQpoint[1][1] = -sqrt(1.0/3);
         GQpoint[2][0] = -sqrt(1.0/3);   GQpoint[2][1] = sqrt(1.0/3);
         GQpoint[3][0] = sqrt(1.0/3);    GQpoint[3][1] = sqrt(1.0/3);
         GQweight[0] = 1.0;
         GQweight[1] = 1.0;
         GQweight[2] = 1.0;
         GQweight[3] = 1.0;
      } else if (NGP == 9) {  // Nine-point quadrature
         GQpoint[0][0] = -sqrt(3.0/5);  GQpoint[0][1] = -sqrt(3.0/5);
         GQpoint[1][0] = 0.0;           GQpoint[1][1] = -sqrt(3.0/5);
         GQpoint[2][0] = sqrt(3.0/5);   GQpoint[2][1] = -sqrt(3.0/5);
         GQpoint[3][0] = -sqrt(3.0/5);  GQpoint[3][1] = 0.0;
         GQpoint[4][0] = 0.0;           GQpoint[4][1] = 0.0;
         GQpoint[5][0] = sqrt(3.0/5);   GQpoint[5][1] = 0.0;
         GQpoint[6][0] = -sqrt(3.0/5);  GQpoint[6][1] = sqrt(3.0/5);
         GQpoint[7][0] = 0.0;           GQpoint[7][1] = sqrt(3.0/5);
         GQpoint[8][0] = sqrt(3.0/5);   GQpoint[8][1] = sqrt(3.0/5);

         GQweight[0] = 5.0/9 * 5.0/9;
         GQweight[1] = 8.0/9 * 5.0/9;
         GQweight[2] = 5.0/9 * 5.0/9;
         GQweight[3] = 5.0/9 * 8.0/9;
         GQweight[4] = 8.0/9 * 8.0/9;
         GQweight[5] = 5.0/9 * 8.0/9;
         GQweight[6] = 5.0/9 * 5.0/9;
         GQweight[7] = 8.0/9 * 5.0/9;
         GQweight[8] = 5.0/9 * 5.0/9;
      } else if (NGP == 16) { // Sixteen-point quadrature
         GQpoint[0][0] = -0.8611363116;   GQpoint[0][1] = -0.8611363116;
         GQpoint[1][0] = -0.3399810435;   GQpoint[1][1] = -0.8611363116;
         GQpoint[2][0] =  0.3399810435;   GQpoint[2][1] = -0.8611363116;
         GQpoint[3][0] =  0.8611363116;   GQpoint[3][1] = -0.8611363116;
         GQpoint[4][0] = -0.8611363116;   GQpoint[4][1] = -0.3399810435;
         GQpoint[5][0] = -0.3399810435;   GQpoint[5][1] = -0.3399810435;
         GQpoint[6][0] =  0.3399810435;   GQpoint[6][1] = -0.3399810435;
         GQpoint[7][0] =  0.8611363116;   GQpoint[7][1] = -0.3399810435;
         GQpoint[8][0] = -0.8611363116;   GQpoint[8][1] =  0.3399810435;
         GQpoint[9][0] = -0.3399810435;   GQpoint[9][1] =  0.3399810435;
         GQpoint[10][0]=  0.3399810435;   GQpoint[10][1]=  0.3399810435;
         GQpoint[11][0]=  0.8611363116;   GQpoint[11][1]=  0.3399810435;
         GQpoint[12][0]= -0.8611363116;   GQpoint[12][1]=  0.8611363116;
         GQpoint[13][0]= -0.3399810435;   GQpoint[13][1]=  0.8611363116;
         GQpoint[14][0]=  0.3399810435;   GQpoint[14][1]=  0.8611363116;
         GQpoint[15][0]=  0.8611363116;   GQpoint[15][1]=  0.8611363116;

         GQweight[0] = 0.3478548451 * 0.3478548451;
         GQweight[1] = 0.3478548451 * 0.6521451548;
         GQweight[2] = 0.3478548451 * 0.6521451548;
         GQweight[3] = 0.3478548451 * 0.3478548451;
         GQweight[4] = 0.6521451548 * 0.3478548451;
         GQweight[5] = 0.6521451548 * 0.6521451548;
         GQweight[6] = 0.6521451548 * 0.6521451548;
         GQweight[7] = 0.6521451548 * 0.3478548451;
         GQweight[8] = 0.6521451548 * 0.3478548451;
         GQweight[9] = 0.6521451548 * 0.6521451548;
         GQweight[10] = 0.6521451548 * 0.6521451548;
         GQweight[11] = 0.6521451548 * 0.3478548451;
         GQweight[12] = 0.3478548451 * 0.3478548451;
         GQweight[13] = 0.3478548451 * 0.6521451548;
         GQweight[14] = 0.3478548451 * 0.6521451548;
         GQweight[15] = 0.3478548451 * 0.3478548451;
      }
   } else if (eType == 2) {  // Triangular element
      if (NGP == 1) {          // One-point quadrature
         GQpoint[0][0] = 1.0/3;  GQpoint[0][1] = 1.0/3;
         GQweight[0] = 0.5;
      } else if (NGP == 3) {   // Two-point quadrature
         GQpoint[0][0] = 0.5;   GQpoint[0][1] = 0.0;
         GQpoint[1][0] = 0.0;   GQpoint[1][1] = 0.5;
         GQpoint[2][0] = 0.5;   GQpoint[2][1] = 0.5;

         GQweight[0] = 1.0/6;
         GQweight[1] = 1.0/6;
         GQweight[2] = 1.0/6;
      } else if (NGP == 4) {   // Four-point quadrature
         GQpoint[0][0] = 1.0/3;   GQpoint[0][1] = 1.0/3;
         GQpoint[1][0] = 0.6;     GQpoint[1][1] = 0.2;
         GQpoint[2][0] = 0.2;     GQpoint[2][1] = 0.6;
         GQpoint[3][0] = 0.2;     GQpoint[3][1] = 0.2;

         GQweight[0] = -27.0/96;
         GQweight[1] = 25.0/96;
         GQweight[2] = 25.0/96;
         GQweight[3] = 25.0/96;
      } else if (NGP == 7) {  // Seven-point quadrature
         GQpoint[0][0] = 1.0/3;               GQpoint[0][1] = 1.0/3;
         GQpoint[1][0] = 0.059715871789770;   GQpoint[1][1] = 0.470142064105115;
         GQpoint[2][0] = 0.470142064105115;   GQpoint[2][1] = 0.059715871789770;
         GQpoint[3][0] = 0.470142064105115;   GQpoint[3][1] = 0.470142064105115;
         GQpoint[4][0] = 0.101286507323456;   GQpoint[4][1] = 0.797426985353087;
         GQpoint[5][0] = 0.101286507323456;   GQpoint[5][1] = 0.101286507323456;
         GQpoint[6][0] = 0.797426985353087;   GQpoint[6][1] = 0.101286507323456;

         GQweight[0] = 0.225 / 2;
         GQweight[1] = 0.132394152788 / 2;
         GQweight[2] = 0.132394152788 / 2;
         GQweight[3] = 0.132394152788 / 2;
         GQweight[4] = 0.125939180544 / 2;
         GQweight[5] = 0.125939180544 / 2;
         GQweight[6] = 0.125939180544 / 2;
      }
   }

} // End of function gaussQuad()




//-----------------------------------------------------------------------------------
void calcShape()
//-----------------------------------------------------------------------------------
{
   // Calculates the values of the shape functions and their derivatives with
   // respect to ksi and eta at GQ points.

   double ksi, eta, zeta;
   
   if (eType == 3) { // 3D Hexahedron Element
      S = new double*[8];
      for (int i=0; i<NGP; i++) {
         S[i] = new double[NGP];
      }
      
      DS = new double **[3];
   		
      for (int i=0; i<3; i++) {
         DS[i] = new double *[8];
         for (int j=0; j<8; j++) {
            DS[i][j] = new double[NGP];
         }
      }
      
      for (int k = 0; k<NGP; k++) {
         ksi = GQpoint[k][0];
         eta = GQpoint[k][1];
         zeta = GQpoint[k][2];
         
         S[0][k] = 0.125*(1-ksi)*(1-eta)*(1-zeta);
         S[1][k] = 0.125*(1+ksi)*(1-eta)*(1-zeta);
         S[2][k] = 0.125*(1+ksi)*(1+eta)*(1-zeta);
         S[3][k] = 0.125*(1-ksi)*(1+eta)*(1-zeta);   
         S[4][k] = 0.125*(1-ksi)*(1-eta)*(1+zeta);
         S[5][k] = 0.125*(1+ksi)*(1-eta)*(1+zeta);
         S[6][k] = 0.125*(1+ksi)*(1+eta)*(1+zeta);
         S[7][k] = 0.125*(1-ksi)*(1+eta)*(1+zeta); 

         DS[0][0][k] = -0.125*(1-eta)*(1-zeta);  // ksi derivative of the 1st shape funct. at k-th GQ point.
         DS[1][0][k] = -0.125*(1-ksi)*(1-zeta);  // eta derivative of the 1st shape funct. at k-th GQ point.  
         DS[2][0][k] = -0.125*(1-ksi)*(1-eta);   // zeta derivative of the 1st shape funct. at k-th GQ point.
         DS[0][1][k] =  0.125*(1-eta)*(1-zeta);   
         DS[1][1][k] = -0.125*(1+ksi)*(1-zeta);
         DS[2][1][k] = -0.125*(1+ksi)*(1-eta);         
         DS[0][2][k] =  0.125*(1+eta)*(1-zeta);   
         DS[1][2][k] =  0.125*(1+ksi)*(1-zeta);
         DS[2][2][k] = -0.125*(1+ksi)*(1+eta);         
         DS[0][3][k] = -0.125*(1+eta)*(1-zeta);   
         DS[1][3][k] =  0.125*(1-ksi)*(1-zeta);
         DS[2][3][k] = -0.125*(1-ksi)*(1+eta); 
         DS[0][4][k] = -0.125*(1-eta)*(1+zeta); 
         DS[1][4][k] = -0.125*(1-ksi)*(1+zeta);  
         DS[2][4][k] = 0.125*(1-ksi)*(1-eta);   
         DS[0][5][k] =  0.125*(1-eta)*(1+zeta);   
         DS[1][5][k] = -0.125*(1+ksi)*(1+zeta);
         DS[2][5][k] = 0.125*(1+ksi)*(1-eta);         
         DS[0][6][k] =  0.125*(1+eta)*(1+zeta);   
         DS[1][6][k] =  0.125*(1+ksi)*(1+zeta);
         DS[2][6][k] = 0.125*(1+ksi)*(1+eta);         
         DS[0][7][k] = -0.125*(1+eta)*(1+zeta);   
         DS[1][7][k] =  0.125*(1-ksi)*(1+zeta);
         DS[2][7][k] = 0.125*(1-ksi)*(1+eta);  
           
      }
      
   } else if (eType == 1) { // Quadrilateral element
      if (NEN == 4) {
         S = new double*[4];
         for (int i=0; i<NGP; i++) {
            S[i] = new double[NGP];
         }		
     
         DS = new double **[2];
   		
         for (int i=0; i<2; i++) {
            DS[i] = new double *[4];
            for (int j=0; j<4; j++) {
               DS[i][j] = new double[NGP];
            }
         }
     
         for (int k = 0; k<NGP; k++) {
            ksi = GQpoint[k][0];
            eta = GQpoint[k][1];
           
            S[0][k] = 0.25*(1-ksi)*(1-eta);
            S[1][k] = 0.25*(1+ksi)*(1-eta);
            S[2][k] = 0.25*(1+ksi)*(1+eta);
            S[3][k] = 0.25*(1-ksi)*(1+eta);

            DS[0][0][k] = -0.25*(1-eta);  // ksi derivative of the 1st shape funct. at k-th GQ point.
            DS[1][0][k] = -0.25*(1-ksi);  // eta derivative of the 1st shape funct. at k-th GQ point.
            DS[0][1][k] =  0.25*(1-eta);
            DS[1][1][k] = -0.25*(1+ksi);
            DS[0][2][k] =  0.25*(1+eta);
            DS[1][2][k] =  0.25*(1+ksi);
            DS[0][3][k] = -0.25*(1+eta);
            DS[1][3][k] =  0.25*(1-ksi);
         }
      }
   } else if (eType == 2) {  // Triangular element
      if (NEN == 3) {
         S = new double*[3];
         for (int i=0; i<3; i++) {
               S[i] = new double[NGP];
         }
      
         DS = new double **[2];               
         for (int i=0; i<2; i++) {
            DS[i] = new double *[3];
            for (int j=0; j<3; j++) {
               DS[i][j] = new double[NGP];
            }
         }
     
         for (int k = 0; k<NGP; k++) {
            ksi = GQpoint[k][0];
            eta = GQpoint[k][1];
         
            S[0][k] = 1 - ksi - eta;
            S[1][k] = ksi;
            S[2][k] = eta;

            DS[0][0][k] = -1;    // ksi derivative of the 1st shape funct. at k-th GQ point.
            DS[1][0][k] = -1;    // eta derivative of the 1st shape funct. at k-th GQ point.
            DS[0][1][k] =  1;
            DS[1][1][k] =  0;
            DS[0][2][k] =  0;
            DS[1][2][k] =  1;
         }
      }
   }
} // End of function calcShape()




//-------------------------------------------------------------------------
void calcJacobian()
//-----------------------------------------------------------------------------------
{
   // Calculates the Jacobian matrix, its inverse and determinant for all
   // elements at all GQ points. Also evaluates and stores derivatives of shape
   // functions wrt global coordinates x and y.

   int e, i, j, k, x, iG; 
   double **e_coord;
   double temp;
   
   if (eType == 3) {
   
      e_coord = new double*[NEN];
   
      for (i=0; i<NEN; i++) {
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
      
      gDS = new double***[NE];

      for (i=0; i<NE; i++) {
         gDS[i] = new double**[3];
         for(j=0; j<3; j++) {
            gDS[i][j] = new double*[NEN];
            for(k=0; k<NEN; k++) {
               gDS[i][j][k] = new double[NGP];
            }
         }	
      }
   
      for (e = 0; e<NE; e++){
         // To calculate Jacobian matrix of an element we need e_coord matrix of
         // size NEN*3. Each row of it stores x, y & z coordinates of the nodes of
         // an element.
         for (i = 0; i<NEN; i++){
            iG = LtoG[e][i];
            e_coord[i][0] = coord[iG][0]; 
            e_coord[i][1] = coord[iG][1];
            e_coord[i][2] = coord[iG][2];
         }
      
         // For each GQ point calculate 3*3 Jacobian matrix, its inverse and its
         // determinant. Also calculate derivatives of shape functions wrt global
         // coordinates x and y. These are the derivatives that we'll use in
         // evaluating K and F integrals. 
   	
         for (k = 0; k<NGP; k++) {
            for (i = 0; i<3; i++) {
               for (j = 0; j<3; j++) {
                  temp = 0;
                  for (x = 0; x<NEN; x++) {
                     temp = DS[i][x][k] * e_coord[x][j]+temp;
                  }
                  Jacob[i][j] = temp;
               }
            }
            
            invJacob[0][0] = Jacob[1][1]*Jacob[2][2]-Jacob[2][1]*Jacob[1][2];
            invJacob[0][1] = -(Jacob[0][1]*Jacob[2][2]-Jacob[0][2]*Jacob[2][1]);
            invJacob[0][2] = Jacob[0][1]*Jacob[1][2]-Jacob[1][1]*Jacob[0][2];
            invJacob[1][0] =  -(Jacob[1][0]*Jacob[2][2]-Jacob[1][2]*Jacob[2][0]);
            invJacob[1][1] = Jacob[2][2]*Jacob[0][0]-Jacob[2][0]*Jacob[0][2];
            invJacob[1][2] = -(Jacob[1][2]*Jacob[0][0]-Jacob[1][0]*Jacob[0][2]);
            invJacob[2][0] = Jacob[1][0]*Jacob[2][1]-Jacob[2][0]*Jacob[1][1];
            invJacob[2][1] = -(Jacob[2][1]*Jacob[0][0]-Jacob[2][0]*Jacob[0][1]);
            invJacob[2][2] = Jacob[1][1]*Jacob[0][0]-Jacob[1][0]*Jacob[0][1];

            detJacob[e][k] = Jacob[0][0]*(Jacob[1][1]*Jacob[2][2]-Jacob[2][1]*Jacob[1][2]) +
                            Jacob[0][1]*(Jacob[1][2]*Jacob[2][0]-Jacob[1][0]*Jacob[2][2]) +
                            Jacob[0][2]*(Jacob[1][0]*Jacob[2][1]-Jacob[1][1]*Jacob[2][0]);
            
            for (i = 0; i<3; i++){
               for (j = 0; j<3; j++){
                  invJacob[i][j] = invJacob[i][j]/detJacob[e][k];
               }    
            }
            
            for (i = 0; i<3; i++){
               for (j = 0; j<8; j++){
                  temp = 0;
                  for (x = 0; x<3; x++){
                     temp = invJacob[i][x] * DS[x][j][k]+temp;
                  }
                  gDS[e][i][j][k] = temp;
               }
            }
            
         }
      }
   
   
   
   } else {  // TODO : This part seems to be written for 2D, not 3D.
   e_coord = new double*[NEN];

   for (i=0; i<NEN; i++) {
      e_coord[i] = new double[2];
   }
   	
   Jacob = new double*[2];
   invJacob = new double*[2];
   for (i=0; i<2; i++) {
      Jacob[i] = new double[2];
      invJacob[i] = new double[2];
   }

   detJacob = new double*[NE];
   for (i=0; i<NE; i++) {
      detJacob[i] = new double[NGP];
   }
   	
   gDS = new double***[NE];

   for (i=0; i<NE; i++) {
      gDS[i] = new double**[2];
      for(j=0; j<2; j++) {
         gDS[i][j] = new double*[NEN];
         for(k=0; k<NEN; k++) {
            gDS[i][j][k] = new double[NGP];
		   }
	   }	
   }
   		
   for (e = 0; e<NE; e++){
      // To calculate Jacobian matrix of an element we need e_coord matrix of
      // size NEN*2. Each row of it stores x and y coordinates of the nodes of
      // an element.
      for (i = 0; i<NEN; i++){
         iG = LtoG[e][i];
         e_coord[i][0] = coord[iG][0]; 
         e_coord[i][1] = coord[iG][1];
      }
      
      // For each GQ point calculate 2*2 Jacobian matrix, its inverse and its
      // determinant. Also calculate derivatives of shape functions wrt global
      // coordinates x and y. These are the derivatives that we'll use in
      // evaluating K and F integrals. 
   	
      for (k = 0; k<NGP; k++) {
         for (i = 0; i<2; i++) {
            for (j = 0; j<2; j++) {
               temp = 0;
               for (x = 0; x<NEN; x++) {
                  temp = DS[i][x][k] * e_coord[x][j]+temp;
               }
               Jacob[i][j] = temp;
            }
         }

         detJacob[e][k] = (Jacob[0][0]*Jacob[1][1])-(Jacob[0][1]*Jacob[1][0]);
         
         invJacob[0][0] = Jacob[1][1]/detJacob[e][k];
         invJacob[0][1] = -Jacob[0][1]/detJacob[e][k];
         invJacob[1][0] = -Jacob[1][0]/detJacob[e][k];
         invJacob[1][1] = Jacob[0][0]/detJacob[e][k];

         for (i = 0; i<2; i++){
            for (j = 0; j<3; j++){
               temp = 0;
               for (x = 0; x<2; x++){
                  temp = invJacob[i][x] * DS[x][j][k]+temp;
               }
               gDS[e][i][j][k] = temp;
            }
         }
      }
   }
   }
   
}  // End of function calcJacobian()




//------------------------------------------------------------------------------
void calcGlobalSys()
//------------------------------------------------------------------------------
{
   // Calculates Ke and Fe one by one for each element and assembles them into
   // the global K and F.

   int e, i, j, k, iG;
   double x, y, z, axy, fxy;
   real *Fe, **Ke;

   F = new real[NN];
   //K = new real*[NN];	
   //for (i=0; i<NN; i++) {
   //   K[i] = new real[NN];
   //}

   for (i=0; i<NN; i++) {
      F[i] = 0;
      //for (j=0; j<NN; j++) {
      //   K[i][j] = 0;
      //}
   }

   Fe = new real[NEN];
   Ke = new real*[NEN];	
   for (i=0; i<NEN; i++) {
      Ke[i] = new real[NEN];
   }

   for (e = 0; e<NE; e++) {

      // Intitialize Ke and Fe to zero.
      for (i=0; i<NEN; i++) {
         Fe[i] = 0;
         for (j=0; j<NEN; j++) {
            Ke[i][j] = 0;
         }
      }
     
      for (k = 0; k<NGP; k++) {   // Gauss quadrature loop
      // Calculate global x and y coordinates that correspond to the k-th GQ point.
         x = 0;
         y = 0;
         z = 0;
         for (i = 0; i<NEN; i++) {
            iG = LtoG[e][i];
            x = x + S[i][k]*coord[iG][0];
            y = y + S[i][k]*coord[iG][1];
            z = z + S[i][k]*coord[iG][2];
         }
   	
         axy = 1;
         fxy = 12 * M_PI * M_PI * sin(2*M_PI*x) * sin(2*M_PI*y) * sin(2*M_PI*z);

         for (i = 0; i<NEN; i++) {
            Fe[i] = Fe[i] + S[i][k] * fxy * detJacob[e][k] * GQweight[k];
            for (j = 0; j<NEN; j++) {
               if (eType == 3) {
               Ke[i][j] = Ke[i][j] + (axy * (gDS[e][0][i][k] * gDS[e][0][j][k] +
                          gDS[e][1][i][k] * gDS[e][1][j][k] + gDS[e][2][i][k] * gDS[e][2][j][k]) ) * detJacob[e][k] * GQweight[k];
               } else {
               Ke[i][j] = Ke[i][j] + (axy * (gDS[e][0][i][k] * gDS[e][0][j][k] +
                          gDS[e][1][i][k] * gDS[e][1][j][k]) ) * detJacob[e][k] * GQweight[k];
               }
            }
         }
      }  // End of GQ loop
   		
      assemble(e, Ke, Fe); 	
   }

} // End of function calcGlobalSys()




//-----------------------------------------------------------------------
void assemble(int e, real **Ke, real *Fe)
//-----------------------------------------------------------------------
{
   // Inserts Ke and Fe into proper locations of K and F.
   // Global K in full form is only constructed for GE solver. 

   int i, j, k, iG, jG; 

   for (i = 0; i<NEN; i++) {
      iG = LtoG[e][i];
      F[iG] = F[iG] + Fe[i];  
      //for (j = 0; j<NEN; j++) {
      //   jG = LtoG[e][j];
      //   K[iG][jG] = K[iG][jG] + Ke[i][j];
      //}
   }


   //--------------------------------------------------------------------
   //Assembly process for compressed sparse storage 
   //KeKMapSmall creation

   int *nodeData, pp, q;

   nodeData = new int[NEN];              //stores sorted LtoG data
   for(k=0; k<NEN; k++) {
      nodeData[k] = (LtoG[e][k]);   //takes node data from LtoG
   } 

   KeKMapSmall = new int*[NEN];        //NEN X NEN                                          
   for(j=0; j<NEN; j++) {              //stores map data between K elemental and value vector
      KeKMapSmall[j] = new int[NEN];
   }

   for(i=0; i<NEN; i++) {
      for(j=0; j<NEN; j++) {
         q=0;
         for(pp=rowStarts[nodeData[i]]; pp<rowStarts[nodeData[i]+1]; pp++) {  // p is the location of the col vector(col[x], p=x) 
            if(col[pp] == nodeData[j]) {                                         // Selection process of the KeKMapSmall data from the col vector
               KeKMapSmall[i][j] = q; 
               break;
            }
            q++;
         }
      }
   }

   //--------------------------------------------------------------------
   //creating val vector
   for(i=0; i<NEN; i++) {
      for(j=0; j<NEN; j++) {
         val[ rowStarts[LtoG[ e ][ i ]] + KeKMapSmall[i][j]] += Ke[i][j] ;
      }
   }
   
   for (int i = 0; i<NEN; i++) {
      delete[] KeKMapSmall[i];
   }
   delete[] KeKMapSmall;

   delete[] nodeData;   
      	
} // End of function assemble()




//----------------------------------------------------------------------------
void applyBC()
//----------------------------------------------------------------------------
{
   // For EBCs reduction is not applied. Instead K and F are modified as
   // explained in class, which requires modification of both [K] and {F}.
   // SV values specified for NBCs are added to {F}.

   int i, j, whichBC, node;
   // double x,y,z; 
   int p, colCounter;   
   
   bigNumber=200;          //to make the sparse matrix diagonally dominant

   // Modify K and F for EBCs
   for (i = 0; i<nEBCnodes; i++) {
      node = EBCnodes[i][0];         // Node at which this EBC is specified
   	
      // x = coord[node][0];              // May be necessary for BCstring evaluation
      // y = coord[node][1];
      // z = coord[node][2];
      
      whichBC = EBCnodes[i][1]-1;      // Number of the specified BC
      F[node] = BCstrings[whichBC];    // Specified value of the PV

      //for (j=0; j<NN; j++) {
      //   K[node][j] = 0.0;
      //}
      //K[node][node] = 1.0;
      
      colCounter = 0;
      for(p=rowStarts[node]; p<rowStarts[node+1]; p++) {   // p is the location of the col vector(col[x], p=x)
         if(col[p] == node) {                                   // Selection process of the KeKMapSmall data from the col vector.
            break;
         }
         colCounter++;
      }

      // Modify val and F for the specified u velocity.
      for (j=rowStarts[node]; j<rowStarts[node+1]; j++) {
         val[j] = 0.0;                    // Set non-diagonal entries to 0.
      }
      val[ rowStarts[node] + colCounter ] = 1*bigNumber;     // Set the diagonal entry to 1.  
      
   }
   	
   // Modify F for nonzero NBCs
   // ...
   // ...

} // End of function ApplyBC()




//-------------------------------------------------------------------------
void solve()
//-------------------------------------------------------------------------
{
   // Solves NNxNN global system using Gauss Elimination.

   bool err;

   u = new real[NN];

   #ifdef CUSP
      CUSPsolver();
   #endif

   #ifdef CUSPARSE
      CUSPARSEsolver();
   #endif

   #ifdef MKLCG
      MKLCGsolver();
   #endif
	
   #ifdef CULA
      CULAsolver();
   #endif

}  // End of function solve()




//-----------------------------------------------------------------------------
void postProcess()
//-----------------------------------------------------------------------------
{
   // Write the calculated unknowns on the screen.
   // Actually it is a good idea to write the results to an output file named
   // ProblemName.out too.


/*
   printf("\nCalculated unknowns are \n\n");
   printf(" Node            x                y                 z                 u \n");
   printf("================================================================================\n");
   if ( eType == 3) {
      for (int i = 0; i<NN; i++) { 
         printf("%-5d %16.8f %16.8f %16.8f %18.8f\n", i, coord[i][0], coord[i][1], coord[i][2], u[i]);
      }
   }
   else { 
      for (int i = 0; i<NN; i++) { 
         printf("%-5d %18.8f %18.8f %20.8f\n", i, coord[i][0], coord[i][1], u[i]);
      }
   }   

*/

} // End of function postProcess()



//-----------------------------------------------------------------------------
void writeTecplotFile()
//-----------------------------------------------------------------------------
{
   // Write the calculated unknowns to a Tecplot file
   double x, y, z;
   int i, e;

   outputFile.open((whichProblem + outputExtension).c_str(), ios::out);
   
   outputFile << "TITLE = " << whichProblem << endl;
   outputFile << "VARIABLES = X,  Y,  Z,  U " << endl;
   
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
      outputFile << scientific  << "\t" << x << " "  << y << " "  << z << " " << u[i]  << endl;
   }

   // Print the connectivity list
   for (e = 0; e<NE; e++) {
      outputFile << fixed << "\t";
      for (i = 0; i<NEN; i++) {
         // outputFile.precision(5);
         outputFile << LtoG[e][i]+1 << " " ;
      }
      outputFile<< endl;
   }

   outputFile.close();
}



//-----------------------------------------------------------------------------
void gaussElimination(int N, real **K, real *F, real *u, bool& err)
//-----------------------------------------------------------------------------
{
   // WARNING : In order to be able to use this function full storage of K and
   // all related lines should be uncommented.

   // Solve system of N linear equations with N unknowns using Gaussian
   // Elimination with scaled partial pivoting.
   // err returns true if process fails; false if it is successful.
   
   int *indx=new int[NN];
   real *scale= new real[NN];
   real maxRatio, ratio, sum;
   int maxIndx, tmpIndx;;
    
   for (int i = 0; i < N; i++) {
      indx[i] = i;  // Index array initialization
   }
    
   // Determine scale factors
    
   for (int row = 0; row < N; row++) {
      scale[row] = abs(K[row][0]);
      for (int col = 1; col < N; col++) {
         if (abs(K[row][col]) > scale[row]) {
            scale[row] = abs(K[row][col]);
		   }
	   }
   }
    
   // Forward elimination
    
   for (int k = 0; k < N; k++) {
      maxRatio = abs(K[indx[k]][k])/scale[indx[k]];
      maxIndx = k;
      
      for (int i = k+1; i < N; i++) {
         if (abs(K[indx[i]][k])/scale[indx[i]] > maxRatio) {
            maxRatio = abs(K[indx[i]][k])/scale[indx[i]];
            maxIndx = i;
         }
      }

      if (maxRatio == 0) { // no pivot available
         err = true;
         return;
      }

      tmpIndx =indx[k];
      indx[k]=indx[maxIndx]; 
      indx[maxIndx] = tmpIndx;
    
      // Use pivot row to eliminate kth variable in "lower" rows

      for (int i = k+1; i < N; i++) {
         ratio = -K[indx[i]][k]/K[indx[k]][k];
         for (int col = k; col <= N; col++) {
            if (col==N)
               F[indx[i]] += ratio*F[indx[k]];
            else
               K[indx[i]][col] += ratio*K[indx[k]][col];
         }
      }
   }	
    
   // Back substitution

   for (int k = N-1; k >= 0; k--) {
      sum = 0;
      for (int col = k+1; col < N; col++) {
         sum += K[indx[k]][col] * F[indx[col]];
      }
      F[indx[k]] = (F[indx[k]] - sum)/K[indx[k]][k];
   }

   for (int k = 0; k < N; k++) {
      u[k] = F[indx[k]];
   }

} // End of function gaussElimination()




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

