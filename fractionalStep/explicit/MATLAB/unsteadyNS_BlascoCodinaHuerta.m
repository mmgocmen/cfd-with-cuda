%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
%               3D Unsteady Incompressible Navier Stokes                %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
%                   Middle East Technical University                    %
%                 Department of Mechanical Engineering                  %
%            ME 582 Finite Element Analysis in Thermofluids             %
%                http://www.me.metu.edu.tr/courses/ME582                %
%                                                                       %
%                            Dr. Cuneyt Sert                            %
%                     http://www.metu.edu.tr/~csert                     %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
%                       FEATURES and LIMITATIONS                        %
%                                                                       %
% This code                                                             %
% - is written for educational purposes.                                %
% - solves 3D, unsteady, incompressible Navier-Stokes equations.        %
% - follows Blasco, Codina, Huerta's 1998 paper.                        %
% - uses Lagrange type hexahedral (brick) type elements with 27         %
%   and 8 pressure nodes.                                               %
% - reads the problem data and mesh from an input file, which has the   %
%   coordinates of the corner nodes only.                               %
% - generates an output file to be visualized with the Tecplot software.%
% - generates profile plots for data across a cross section.            %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
%                              HOW TO USE?                              %
%                                                                       %
% This code needs an input file with an extension "inp" to run. Provide %
% its name when asked for.                                              %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
%                               VARIABLES                               %
%                                                                       %
% prName       : Name of the problem                                    %
% NE           : Number of elements                                     %
% NCN          : Number of corner nodes of the mesh                     %
% NN           : Number of both corner and mid-face nodes, if there are %
%                any                                                    %
% NEC          : Number of element corners                              %
%                3: Triangular, 4: Quadrilateral                        %
% NENv         : Number of velocity nodes on an element.                %
% NENp         : Number of pressure nodes on an element.                %
% NEU          : 3*NENv + NENp                                          %
% Ndof         : All velocity and pressures of the solution.            %
% eType        : 1: Quadrilateral, 2: Triangular. Same for all elements %
% funcs        : Structure to hold fx and fy functions                  %
% coord        : Matrix of size NNx2 to hold the x and y coordinates of %
%                the mesh nodes (both corner and mid-face nodes)        %
% NGP          : Number of GQ points.                                   %
% GQ           : Structure to store GQ points and weights.              %
%                                                                       %
% Sv, Sp       : Shape functions for velocity and pressure evaluated at %
%                GQ points. Matrices of size NENvxNGP and NENpxNGP      %
% dSv, dSp     : Derivatives of shape functions for velocity and        %
%                pressure wrt to ksi and eta evaluated at GQ points.    %
%                Matrices of size 2xNENvxNGP and 2xNENpxNGP             %
%                                                                       %
% BC           : Structure for boundary conditions                      %
%  - nBC       : Number of different BCs specified.                     %
%  - type      : Array of size nBC.                                     %
%                1: Given velocity, i.e. inlet or wall                  %
%                2: Outflow                                             %
%  - str       : BC values/functions are stored as strings in this cell %
%                array of size nBCx2. For boundaries of type 1 1st and  %
%                2nd columns store u and v velocity components. For     %
%                boundaries of type 2 1st column stores pressure and    %
%                2nd column is not used.                                %
%  - nVelNodes : Number of mesh nodes where velocity is specified.      %
%  - nOutFaces : Number of element faces where outflow is specified.    %
%  - velNodes  : Array of nVelNodesx2. 1st column stores global node    %
%                numbers for which velocity is provided. 2nd column     %
%                stores the specified BC number.                        %
%  - outFaces  : Array of nOutFacesx3. 1st and 2nd columns store        %
%                element and face numbers for which NBC is given. 3rd   %
%                column is used to store which BC is specified.         %
%  - pNode     : Node number where pressure is fixed to zero. Provide a %
%                negative value in the input file if you do not want to %
%                fix pressure at a point.                               %
%                                                                       %
% time         : Structure for time integration                         %
%  - s         : Current time level                                     %
%  - t         : Current time                                           %
%  - dt        : Time step                                              %
%  - initial   : Initial time                                           %
%  - final     : Final time                                             %
%  - alpha     : Parameter used to select a time integration scheme     %
%                0.0: Forward Difference                                %
%                0.5: Crank Nicolson                                    %
%                1.0: Backward Difference                               %
%  - isRestart : Flag to control restarting from a previous solution    %
%                0 : Restart from zero initial conditions.              %
%                1 : Restart from problemName_restart.dat file, which   %
%                    previously generated Tecplot file, renamed         %
%                    properly.                                          %
%                                                                       %
% elem         : Structure for elements                                 %
%  - he        : Size of the element                                    %
%  - LtoGnode  : Local to global node mapping array of size NENvx1      %
%  - LtoGvel   : Local to global mapping of velocity unknowns (3*NENvx1)%
%  - LtoGpres  : Local to global mapping of pressure unknowns (NENpx1)  %
%  - Ke        : TODO : Change                                          %
%  - Fe        : TODO : Change                                          %
%  - gDSv      : Derivatives of shape functions for velocity wrt x and  %
%                y at GQ points. Size is 2xNENvxNGP                     %
%  - gDSp      : Derivatives of shape functions for pressure wrt x and  %
%                y at GQ points. Size is 2xNENpxNGP                     %
%  - detJacob  : Determinant of the Jacobian matrix evaluated at a      %
%                certain (ksi, eta)                                     %
%                                                                       %
% soln         : Structure for the global system of equations           %
%  - M         : Global mass matrix                                     %
%  - K         : Global stiffness matrix                                %
%  - G         : TODO ...                                               %
%                TODO: Following list is incomplete                     %
%  - F         : Global force vector of size Ndofx1                     %
%  - U         : Global unknown vector of size Ndofx1                   %
%  - Uold      : Solution of the previous time step                     %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
%                               FUNCTIONS                               %
%                           (incomplete list)                           %
%                                                                       %
% steadyNavierStokes2D : Main function                                  %
% readInputFile : Read the input file                                   %
% setupLtoGdof  : Create LtoGvel and LtoGpres for each element based on %
%                 LtoGnode values read from the input file              %
% setupGQ       : Generate variables for GQ points and weights          %
% calcShape     : Evaluate shape functions and their ksi and eta        %
%                 derivatives at GQ points                              %
% calcJacob     : Calculate the Jacobian, and its determinant and       %
%                 derivatives of shape functions wrt x and y at GQ      %
%                 points                                                %
% calcGlobalSys : Calculate global K and F                              %
% calcElemSys   : Calculate elemental Ke and Fe                         %
% assemble      : Assemble elemental systems into a global system       %
% applyBC       : TODO: There are many applyBC functions now.           %
% timeLoop      : The whole solution takes place here                   %
% solve         : Solve the global system of equations                  %
% postProcess   : Generate contour plots of velocity components and     %
%                 pressure                                              %
% createTecplot : Create an output file for Tecplot software            %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% TODO: Interchanging use of NENv and NCN is confusing and maybe wrong also.


%========================================================================
function [] = unsteadyNS_BlascoCodinaHuerta
%========================================================================
clc;
clear all;
close all;

disp('******************************************************************');
disp('***                                                            ***');
disp('***  ME 582 - 3D Unsteady Incompressible Navier-Stokes Solver  ***');
disp('***                                                            ***');
disp('***           Formulation of Blasco, Codina & Huerta           ***');
disp('***                                                            ***');
disp('******************************************************************');

ti = readInputFile();               fprintf('readInputFile()       took %7.3f sec.\n', toc(ti))

time1 = tic;

%calcElemSize();
%findElemNeighbors();
ti = tic;   setupMidFaceNodes();    fprintf('setupMidFaceNodes()   took %7.3f sec.\n', toc(ti))
ti = tic;   setupLtoGdof();         fprintf('setupLtoGdof()        took %7.3f sec.\n', toc(ti))
ti = tic;   determineVelBCnodes();  fprintf('determineVelBCnodes() took %7.3f sec.\n', toc(ti))
ti = tic;   findElemsOfNodes();     fprintf('findElemsOfNodes()    took %7.3f sec.\n', toc(ti))
ti = tic;   findMonitorPoint();     fprintf('findMonitorPoint()    took %7.3f sec.\n', toc(ti))
ti = tic;   setupSparseM();         fprintf('setupSparseM()        took %7.3f sec.\n', toc(ti))
ti = tic;   setupSparseG();         fprintf('setupSparseG()        took %7.3f sec.\n', toc(ti))
ti = tic;   setupGQ();              fprintf('setupGQ()             took %7.3f sec.\n', toc(ti))
ti = tic;   calcShape();            fprintf('calcShape()           took %7.3f sec.\n', toc(ti))
ti = tic;   calcJacob();            fprintf('calcJacob()           took %7.3f sec.\n\n', toc(ti))

time2 = tic;

timeLoop();

time3 = toc(time2);
time4 = toc(time1);

%postProcess();
createTecplot();

fprintf('\nTotal run time is  %f seconds.', time4);
fprintf('\nTime loop took  %f seconds.\n\n', time3);

fprintf('Program is terminated successfully.\n');

% End of function unsteadyNS_BlascoCodinaHuerta()




%========================================================================
function [ti] = readInputFile
%========================================================================
global prName funcs coord elem NE NNp NCN NEC NEF NEE NENv NENp;
global NGP BC eType density viscosity time maxIter tolerance monPoint;

% Ask the user for the input file name
prName = input('\nEnter the name of the input file (without the .inp extension): ','s');
fprintf('\n');
inputFile = fopen(strcat(prName,'.inp'), 'r');

ti = tic;

fgets(inputFile);  % Read the header line of the file
fgets(inputFile);  % Read the next dummy line.


% Read problem parameters
eType  = fscanf(inputFile, 'eType    :%d');     fgets(inputFile);
NE     = fscanf(inputFile, 'NE       :%d');     fgets(inputFile);
NCN    = fscanf(inputFile, 'NCN      :%d');     fgets(inputFile);
NENv   = fscanf(inputFile, 'NENv     :%d');     fgets(inputFile);
NENp   = fscanf(inputFile, 'NENp     :%d');     fgets(inputFile);
NGP    = fscanf(inputFile, 'NGP      :%d');     fgets(inputFile);

% NEU = 3*NENv + NENp;
% Ndof = 3*NN + NCN;    % Velocities are stored at NN nodes and pressures
                      % are stored at NCN corner nodes.

time.alpha     = fscanf(inputFile, 'alpha    :%f');   fgets(inputFile);
time.dt        = fscanf(inputFile, 'dt       :%f');   fgets(inputFile);
time.initial   = fscanf(inputFile, 't_ini    :%f');   fgets(inputFile);
time.final     = fscanf(inputFile, 't_final  :%f');   fgets(inputFile);
maxIter        = fscanf(inputFile, 'maxIter  :%f');   fgets(inputFile);
tolerance      = fscanf(inputFile, 'tolerance:%f');   fgets(inputFile);
time.isRestart = fscanf(inputFile, 'isRestart:%f');   fgets(inputFile);

density        = fscanf(inputFile, 'density  :%f');   fgets(inputFile);
viscosity      = fscanf(inputFile, 'viscosity:%f');   fgets(inputFile);

funcs.fx       = fscanf(inputFile, 'fx       :%s');   fgets(inputFile);
funcs.fy       = fscanf(inputFile, 'fy       :%s');   fgets(inputFile);


% Read corner node coordinates
fgets(inputFile);  % Read the next dummy line.
fgets(inputFile);  % Read the next dummy line.
for i = 1:NCN
  dummy = str2num(fgets(inputFile));
  coord(i,1) = dummy(2);
  coord(i,2) = dummy(3);
  coord(i,3) = dummy(4);
end

if eType == 1      % Hexahedral element
  NEC = 8;      % Number of element corners
  NEF = 6;      % Number of element faces
  NEE = 12;     % Number of element edges
else               % Tetrahedral element
  NEC = 4;
  NEF = 4;
  NEE = 6;
end


% Read connectivity of the elements (LtoGnode). Only the connectivity of
% corner nodes are available in the input file
fgets(inputFile);  % Read the next dummy line.
fgets(inputFile);  % Read the next dummy line.
for e = 1:NE
   dummy = str2num(fgets(inputFile));
   for i = 1:NEC
      elem(e).LtoGnode(i) = dummy(1+i);
   end
end


% Read number of different BC types and details of each BC
fgets(inputFile);  % Read the next dummy line.
fgets(inputFile);  % Read the next dummy line.
nBC   = fscanf(inputFile, 'nBC       :%d');      fgets(inputFile);
BC.type = zeros(nBC,1);
BC.str  = cell(nBC,3);

for i = 1:nBC
  fscanf(inputFile, 'BC %*d      :');
  BC.type(i) = fscanf(inputFile, '%d', 1);
  
  if BC.type(i) == 1  % For velocity BC read 3 strings separated with
                      % semicolons.
    string = fgets(inputFile);
    [str1, str2] = strtok(string, ':');
    [str2, str3] = strtok(str2, ':');
    [str3, str4] = strtok(str3, ':');
    
    BC.str(i,1) = cellstr(str1);
    BC.str(i,2) = cellstr(str2);
    BC.str(i,3) = cellstr(str3);
  else                % For outflow BC read a single string.
    BC.str(i,1) = cellstr(fgets(inputFile));
  end
end

% Read number of velocity faces and number of outflow faces.
dummy = fgets(inputFile);  % Read the next dummy line.
BC.nVelFaces = fscanf(inputFile, 'nVelFaces :%d');     fgets(inputFile);
BC.nOutFaces = fscanf(inputFile, 'nOutFaces :%d');     fgets(inputFile);


% Read velocity BC data
BC.velFaces = zeros(BC.nVelFaces,3);
fgets(inputFile);  % Read the next dummy line.
fgets(inputFile);  % Read the next dummy line.
for i = 1:BC.nVelFaces
  BC.velFaces(i,1) = fscanf(inputFile, '%d', 1);    % Read element no.
  BC.velFaces(i,2) = fscanf(inputFile, '%d', 1);    % Read face no.
  BC.velFaces(i,3) = fscanf(inputFile, '%d', 1);    % Read BC no.
  fgets(inputFile);
end


% Read outflow BC data
BC.Outfaces = zeros(BC.nOutFaces,3);
fgets(inputFile);  % Read the next dummy line.
fgets(inputFile);  % Read the next dummy line.
for i = 1:BC.nOutFaces
  BC.outFaces(i,1) = fscanf(inputFile, '%d', 1);    % Read element no.
  BC.outFaces(i,2) = fscanf(inputFile, '%d', 1);    % Read face no.
  BC.outFaces(i,3) = fscanf(inputFile, '%d', 1);    % Read BC no.
  fgets(inputFile);
end


% Read the node number where pressure is fixed to zero. A negative value
% means that presure will not be fixed at that point.
fgets(inputFile);  % Read the next dummy line.
fgets(inputFile);  % Read the next dummy line.
BC.pNode = str2num(fgets(inputFile));


% Read monitor point coordinates.
fgets(inputFile);  % Read the next dummy line.
fgets(inputFile);  % Read the next dummy line.
monPoint.x = fscanf(inputFile, '%f', 1);
monPoint.y = fscanf(inputFile, '%f', 1);
monPoint.z = fscanf(inputFile, '%f', 1);


fclose(inputFile);


% Determine NNp, number of pressure nodes
if NENp == 1   % Only 1 pressure node at the element center. Not tested at all.   TODO: Either test this element and fully support it or remove details about it.
  NNp = NE;
else
  NNp = NCN;
end

% End of function readInputFile()
 



%========================================================================
function [] = setupMidFaceNodes
%========================================================================
global NE NN NCN NENv NENp NEC NEF NEE elem
global coord eType;
% Calculates coordinates of mid-edge and mid-face nodes and adds these
% nodes to LtoGnode.

if NENv == NENp   % Don't do anything if NENv == NENp
  return;
end

SMALL = 1e-10;         % Value used for coordinate equality check

nodeCount = NCN; % This will be incremented as new mid-edge and mid-face
% nodes are added.


for e = 1:NE
  
  % Go through all edges of the element and check whether there are any new
  % mid-edge nodes or not.
  for ed = 1:NEE
  
    % Determine corner nodes of edge ed
    if eType == 1   % Hexahedral element
      switch ed
        case 1
          n1 = elem(e).LtoGnode(1);
          n2 = elem(e).LtoGnode(2);
        case 2
          n1 = elem(e).LtoGnode(2);
          n2 = elem(e).LtoGnode(3);
        case 3
          n1 = elem(e).LtoGnode(3);
          n2 = elem(e).LtoGnode(4);
        case 4
          n1 = elem(e).LtoGnode(4);
          n2 = elem(e).LtoGnode(1);
        case 5
          n1 = elem(e).LtoGnode(1);
          n2 = elem(e).LtoGnode(5);
        case 6
          n1 = elem(e).LtoGnode(2);
          n2 = elem(e).LtoGnode(6);
        case 7
          n1 = elem(e).LtoGnode(3);
          n2 = elem(e).LtoGnode(7);
        case 8
          n1 = elem(e).LtoGnode(4);
          n2 = elem(e).LtoGnode(8);
        case 9
          n1 = elem(e).LtoGnode(5);
          n2 = elem(e).LtoGnode(6);
        case 10
          n1 = elem(e).LtoGnode(6);
          n2 = elem(e).LtoGnode(7);
        case 11
          n1 = elem(e).LtoGnode(7);
          n2 = elem(e).LtoGnode(8);
        case 12
          n1 = elem(e).LtoGnode(8);
          n2 = elem(e).LtoGnode(5);
      end
    elseif eType == 2   % Tetrahedral element
      fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function setupMidFaceNodes() yet!!!\n\n\n');
    end

    midEdgeCoord(1) = 0.5 * (coord(n1,1) + coord(n2,1));
    midEdgeCoord(2) = 0.5 * (coord(n1,2) + coord(n2,2));
    midEdgeCoord(3) = 0.5 * (coord(n1,3) + coord(n2,3));

    matchFound = 0;

    % Search if this new coordinate was already found previously.
    for i = NCN+1:nodeCount
      if abs(midEdgeCoord(1) - coord(i,1)) < SMALL
         if abs(midEdgeCoord(2) - coord(i,2)) < SMALL
            if abs(midEdgeCoord(3) - coord(i,3)) < SMALL   % Match found, this is not a new node.
              elem(e).LtoGnode(ed+NEC) = i;
              matchFound = 1;
              break;
            end
         end
      end
    end

    if matchFound == 0    % No match found, this is a new node.
      nodeCount = nodeCount + 1;
      elem(e).LtoGnode(ed+NEC) = nodeCount;
      coord(nodeCount,:) = midEdgeCoord(:);
    end
  end  % End of ed (edge) loop
  

  % Go through all faces of the element and check whether there are any new
  % mid-face nodes or not.
  for f = 1:NEF
  
    % Determine corner nodes of face f
    if eType == 1   % Hexahedral element
      switch f
        case 1
          n1 = elem(e).LtoGnode(1);
          n2 = elem(e).LtoGnode(2);
          n3 = elem(e).LtoGnode(3);
          n4 = elem(e).LtoGnode(4);
        case 2
          n1 = elem(e).LtoGnode(1);
          n2 = elem(e).LtoGnode(2);
          n3 = elem(e).LtoGnode(5);
          n4 = elem(e).LtoGnode(6);
        case 3
          n1 = elem(e).LtoGnode(2);
          n2 = elem(e).LtoGnode(3);
          n3 = elem(e).LtoGnode(6);
          n4 = elem(e).LtoGnode(7);
        case 4
          n1 = elem(e).LtoGnode(3);
          n2 = elem(e).LtoGnode(4);
          n3 = elem(e).LtoGnode(7);
          n4 = elem(e).LtoGnode(8);
        case 5
          n1 = elem(e).LtoGnode(1);
          n2 = elem(e).LtoGnode(4);
          n3 = elem(e).LtoGnode(5);
          n4 = elem(e).LtoGnode(8);
        case 6
          n1 = elem(e).LtoGnode(5);
          n2 = elem(e).LtoGnode(6);
          n3 = elem(e).LtoGnode(7);
          n4 = elem(e).LtoGnode(8);
      end
      
      midFaceCoord(1) = 0.25 * (coord(n1,1) + coord(n2,1) + coord(n3,1) + coord(n4,1));
      midFaceCoord(2) = 0.25 * (coord(n1,2) + coord(n2,2) + coord(n3,2) + coord(n4,2));
      midFaceCoord(3) = 0.25 * (coord(n1,3) + coord(n2,3) + coord(n3,3) + coord(n4,3));

    elseif eType == 2   % Tetrahedral element
      fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function setupMidFaceNodes() yet!!!\n\n\n');
    end

    matchFound = 0;

    % Search if this new coordinate was already found previously.
    for i = NCN+1:nodeCount
      if abs(midFaceCoord(1) - coord(i,1)) < SMALL
         if abs(midFaceCoord(2) - coord(i,2)) < SMALL
            if abs(midFaceCoord(3) - coord(i,3)) < SMALL   % Match found, this is not a new node.
              elem(e).LtoGnode(f+NEC+NEE) = i;
              matchFound = 1;
              break;
            end
         end
      end
    end

    if matchFound == 0    % No match found, this is a new node.
      nodeCount = nodeCount + 1;
      elem(e).LtoGnode(f+NEC+NEE) = nodeCount;
      coord(nodeCount,:) = midFaceCoord(:);
    end
  end  % End of f (face) loop


  % Add the mid-element node as a new node.
  if eType == 1   % Hexahedral element
    n1 = elem(e).LtoGnode(1);
    n2 = elem(e).LtoGnode(2);
    n3 = elem(e).LtoGnode(3);
    n4 = elem(e).LtoGnode(4);
    n5 = elem(e).LtoGnode(5);
    n6 = elem(e).LtoGnode(6);
    n7 = elem(e).LtoGnode(7);
    n8 = elem(e).LtoGnode(8);
    
    midElemCoord(1) = 0.125 * (coord(n1,1) + coord(n2,1) + coord(n3,1) + coord(n4,1) + coord(n5,1) + coord(n6,1) + coord(n7,1) + coord(n8,1));
    midElemCoord(2) = 0.125 * (coord(n1,2) + coord(n2,2) + coord(n3,2) + coord(n4,2) + coord(n5,2) + coord(n6,2) + coord(n7,2) + coord(n8,2));
    midElemCoord(3) = 0.125 * (coord(n1,3) + coord(n2,3) + coord(n3,3) + coord(n4,3) + coord(n5,3) + coord(n6,3) + coord(n7,3) + coord(n8,3));
    
    nodeCount = nodeCount + 1;
    elem(e).LtoGnode(1+NEC+NEE+NEF) = nodeCount;
    coord(nodeCount,:) = midElemCoord(:);

  elseif eType == 2   % Tetrahedral element
    fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function setupMidFaceNodes() yet!!!\n\n\n');
  end
  
end  % End of e loop

%   % Do calculations for mid-element nodes. Skip this part for serendipity
%   % elements
%   if eType == 1 && NENv==9
%     % Add mid-element node to LtoGnode and calculate its coordinate
%     nodeCount = nodeCount + 1;
%     elem(e).LtoGnode(NENv) = nodeCount;
% 
%     % Calculate the coordinates of mid-element node as the averages of
%     % coordinates of corner nodes.
%     sumX = 0.0;
%     sumY = 0.0;
%     for i = 1:NEC
%       sumX = sumX + coord(elem(e).LtoGnode(i),1);
%       sumY = sumY + coord(elem(e).LtoGnode(i),2);
%     end
%     coord(nodeCount,1) = sumX / NEC;
%     coord(nodeCount,2) = sumY / NEC;
%   end

% From now on use NN instead of nodeCount
NN = nodeCount;

% End of function setupMidFaceNodes()




%========================================================================
function [] = setupLtoGdof
%========================================================================
global NE NN NENv NENp elem;

% Sets up LtoGvel and LtoGpres for each element using LtoGnode. As an
% example LtoGvel uses the following local unknown ordering for a
% quadrilateral element with NENv = 27
%
% u1, u2, u3, ..., u26, u27, v1, v2, v3, ..., v26, v27, w1, w2, w3, ..., w26, w27

for e = 1:NE
  velCounter = 0;   % Velocity unknown counter
  presCounter = 0;  % Pressure unknown counter
  
  % u velocity unknowns
  for i = 1:NENv
    velCounter = velCounter + 1;
    elem(e).LtoGvel(velCounter) = elem(e).LtoGnode(i);
  end
  
  % v velocity unknowns
  for i = 1:NENv
    velCounter = velCounter + 1;
    elem(e).LtoGvel(velCounter) = NN + elem(e).LtoGnode(i);
  end

  % w velocity unknowns
  for i = 1:NENv
    velCounter = velCounter + 1;
    elem(e).LtoGvel(velCounter) = 2*NN + elem(e).LtoGnode(i);
  end
  
  % pressure unknowns
  % Note that first pressure unknown is numbered as 1, but not 3*NN + 1.
  for i = 1:NENp
    presCounter = presCounter + 1;
    elem(e).LtoGpres(presCounter) = elem(e).LtoGnode(i);
  end

end

% End of function setupLtoGdof()




%========================================================================
function [] = determineVelBCnodes
%========================================================================
global NN NENv NENp elem BC eType

% Element faces where velocity BCs are specified were read from the input
% file. Now let's determine the actual nodes where these BCs are specified.

velBCinfo = zeros(NN,1);   % Dummy variable to store which velocity BC is specified at a node.

velBCinfo(:,:) = -1;    % Initialize to -1.

for i = 1:BC.nVelFaces
  e       = BC.velFaces(i,1);   % Element where velocity BC is specified.
  f       = BC.velFaces(i,2);   % Face where velocity BC is specified.
  whichBC = BC.velFaces(i,3);   % Number of specified BC.
  
  % Consider corner nodes of the face
  if eType == 1   % Hexahedral element
    switch f
      case 1
        n1 = elem(e).LtoGnode(1);
        n2 = elem(e).LtoGnode(2);
        n3 = elem(e).LtoGnode(3);
        n4 = elem(e).LtoGnode(4);
      case 2
        n1 = elem(e).LtoGnode(1);
        n2 = elem(e).LtoGnode(2);
        n3 = elem(e).LtoGnode(5);
        n4 = elem(e).LtoGnode(6);
      case 3
        n1 = elem(e).LtoGnode(2);
        n2 = elem(e).LtoGnode(3);
        n3 = elem(e).LtoGnode(6);
        n4 = elem(e).LtoGnode(7);
      case 4
        n1 = elem(e).LtoGnode(3);
        n2 = elem(e).LtoGnode(4);
        n3 = elem(e).LtoGnode(7);
        n4 = elem(e).LtoGnode(8);
      case 5
        n1 = elem(e).LtoGnode(1);
        n2 = elem(e).LtoGnode(4);
        n3 = elem(e).LtoGnode(5);
        n4 = elem(e).LtoGnode(8);
      case 6
        n1 = elem(e).LtoGnode(5);
        n2 = elem(e).LtoGnode(6);
        n3 = elem(e).LtoGnode(7);
        n4 = elem(e).LtoGnode(8);
    end

    velBCinfo(n1) = whichBC;
    velBCinfo(n2) = whichBC;
    velBCinfo(n3) = whichBC;
    velBCinfo(n4) = whichBC;

  elseif eType == 2   % Tetrahedral element
    fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function determineVelBCnodes() yet!!!\n\n\n');
  end
  
  
  % Consider mid-edge and mid-face if there are any.
  if NENp ~= NENv
    if eType == 1   % Hexahedral element
      switch f
        case 1
          n1 = elem(e).LtoGnode(9);
          n2 = elem(e).LtoGnode(10);
          n3 = elem(e).LtoGnode(11);
          n4 = elem(e).LtoGnode(12);
          n5 = elem(e).LtoGnode(21);
        case 2
          n1 = elem(e).LtoGnode(9);
          n2 = elem(e).LtoGnode(13);
          n3 = elem(e).LtoGnode(14);
          n4 = elem(e).LtoGnode(17);
          n5 = elem(e).LtoGnode(22);
        case 3
          n1 = elem(e).LtoGnode(10);
          n2 = elem(e).LtoGnode(14);
          n3 = elem(e).LtoGnode(15);
          n4 = elem(e).LtoGnode(18);
          n5 = elem(e).LtoGnode(23);
        case 4
          n1 = elem(e).LtoGnode(11);
          n2 = elem(e).LtoGnode(15);
          n3 = elem(e).LtoGnode(16);
          n4 = elem(e).LtoGnode(19);
          n5 = elem(e).LtoGnode(24);
        case 5
          n1 = elem(e).LtoGnode(12);
          n2 = elem(e).LtoGnode(13);
          n3 = elem(e).LtoGnode(16);
          n4 = elem(e).LtoGnode(20);
          n5 = elem(e).LtoGnode(25);
        case 6
          n1 = elem(e).LtoGnode(17);
          n2 = elem(e).LtoGnode(18);
          n3 = elem(e).LtoGnode(19);
          n4 = elem(e).LtoGnode(20);
          n5 = elem(e).LtoGnode(26);
      end

      velBCinfo(n1) = whichBC;
      velBCinfo(n2) = whichBC;
      velBCinfo(n3) = whichBC;
      velBCinfo(n4) = whichBC;
      velBCinfo(n5) = whichBC;

    elseif eType == 2   % Tetrahedral element
      fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function determineVelBCnodes() yet!!!\n\n\n');
    end
    
  end
end


% Count the number of velocity BC nodes
BC.nVelNodes = 0;
for i = 1:NN
  if velBCinfo(i) ~= -1
    BC.nVelNodes = BC.nVelNodes + 1;
  end
end

% Store velBCinfo variable as BC.velNodes
BC.velNodes = zeros(BC.nVelNodes,2);
counter = 0;
for i = 1:NN
  if velBCinfo(i) ~= -1
    counter = counter + 1;
    BC.velNodes(counter,1) = i;
    BC.velNodes(counter,2) = velBCinfo(i);
  end
end

clear velBCinfo BC.nVelFaces BC.velFaces

% End of function determineVelBCnodes()




%========================================================================
function [] = findElemsOfNodes
%========================================================================
global NE NN NNp NENv NENp elem elemsOfVelNodes elemsOfPresNodes
global NelemOfVelNodes NelemOfPresNodes;

% Determines elements connected to velocity nodes (elemsOfVelNodes) and 
% pressure nodes (elemsOfPresNodes). These are necessary for sparse storage.

% They are stored in matrices of size NNx10 and NNpx10, where 10 is a
% number, estimated to be larger than the maximum number of elements
% connected to a node.

% Also two arrays (NelemOfVelNodes & NelemOfPresNodes) store the actual
% number of elements connected to each velocity and pressure node.

LARGE = 10;  % We assume that not more than 10 elements are connected to a node.      % TODO: Define this somewhere else, which will be easy to notice.

elemsOfVelNodes  = zeros(NN,LARGE);
elemsOfPresNodes = zeros(NNp,LARGE);

NelemOfVelNodes  = zeros(NN,1);
NelemOfPresNodes = zeros(NNp,1);

% Form elemsOfVelNodes using LtoGvel of each element
for e = 1:NE
  for i = 1:NENv
    node = elem(e).LtoGvel(i);
    NelemOfVelNodes(node) = NelemOfVelNodes(node) + 1;
    elemsOfVelNodes(node, NelemOfVelNodes(node)) = e;
  end
end

% Form elemsOfPresNodes using LtoGpres of each element
for e = 1:NE
  for i = 1:NENp
    node = elem(e).LtoGpres(i);
    NelemOfPresNodes(node) = NelemOfPresNodes(node) + 1;
    elemsOfPresNodes(node, NelemOfPresNodes(node)) = e;
  end
end

% End of function findElemsOfNodes()




%========================================================================
function [] = findMonitorPoint
%========================================================================
global NCN coord monPoint;

% Find the point that is closest to the monitor point coordinates read
% from the input file.

distance = 1e6;   % Initialize to a large value

for i = 1:NCN
  dx = coord(i,1) - monPoint.x;
  dy = coord(i,2) - monPoint.y;
  dz = coord(i,3) - monPoint.z;
  if sqrt(dx^2 + dy^2 + dz^2) < distance
    distance = sqrt(dx^2 + dy^2 + dz^2);
    monPoint.node = i;
  end
end

% End of function findMonitorPoint()




%========================================================================
function [] = setupSparseM
%========================================================================
global NN NE NENv elem soln elemsOfVelNodes NelemOfVelNodes;

% Sets up row and column arrays of the global mass matrix.

% First work only with the upper left part of the matrix and then extend it
% for the middle and lower right parts, which are identical to the upper left
% part.


% In each row, find the columns with nonzero entries.

LARGE = max(NelemOfVelNodes);  % We know that not more than this many elements are connected to a velocity node.

soln.sparseM.NNZ = 0;  % Counts nonzero entries in sub-mass matrix.

NNZcolInARow = zeros(NN,1);            % Number of nonzero columns in each row. This is nothing but the list of nodes that are in communication with each node.
NZcolsInARow = zeros(NN,LARGE*NENv);   % Nonzero columns in a row.            % TODO: This array may take too much memory in 3D. Instead this information can be stored in a CSR type arrangement.

for r = 1:NN     % Loop over all rows.
  isColNZ = zeros(NN,1);   % A flag array to store whether the column is zero or not.
                           % Stores similar information as NZcolsInARow,
                           % but makes counting nonzeros easier.
  colCount = 0;
  
  for i = 1:NelemOfVelNodes(r)   % NelemOfVelNodes(r) is the number of elements connected to node r
    e = elemsOfVelNodes(r,i);    % This element contributes to row r.
    for j = 1:NENv
      if isColNZ(elem(e).LtoGvel(j)) == 0   % 0 means this column had no previous non zero contribution.
        colCount = colCount + 1;
        isColNZ(elem(e).LtoGvel(j)) = 1;    % 1 means this column is non zero.
        NZcolsInARow(r,colCount) = elem(e).LtoGvel(j);
      end
    end
  end
  
  NNZcolInARow(r) = colCount;
  soln.sparseM.NNZ = soln.sparseM.NNZ + colCount;
end


% Entries in each row of NZcolsInARow are not sorted. Let's sort them out.
for r = 1:NN
  array = NZcolsInARow(r,1:NNZcolInARow(r));   % Nonzero columns in a row (unsorted)
  sortedArray = sort(array);
  NZcolsInARow(r,1:NNZcolInARow(r)) = sortedArray;
end



% Allocate memory for two vectors of soln.sparseM. Thinking about the whole
% mass matrix, let's define the sizes properly by using three times of the
% calculated NNZ.
soln.sparseM.col = zeros(3 * soln.sparseM.NNZ,1);
soln.sparseM.row = zeros(3 * soln.sparseM.NNZ,1);


% Fill in soln.sparseM.col and soln.sparseM.row arrays
% This is done in a row-by-row way.
NNZcounter = 0;
for r = 1:NN     % Loop over all rows.
  for i = 1:NNZcolInARow(r)
    NNZcounter = NNZcounter + 1;
    soln.sparseM.row(NNZcounter) = r;
    soln.sparseM.col(NNZcounter) = NZcolsInARow(r,i);
  end
end



% Actually mass matrix is not NNxNN, but 3NNx3NN. Its middle part and 
% lower right part are the same as its upper right part, and the other 6
% blocks are full of zeros. What we set up above is only the upper right
% part. Let's extend the row, col and value arrays to accomodate the
% middle and lower right parts too.
                                                                                 % TODO: Is this expansion really necessary or can we save some memory by not doing it.
for i = 1:soln.sparseM.NNZ
  soln.sparseM.row(i + soln.sparseM.NNZ)     = soln.sparseM.row(i) + NN;
  soln.sparseM.row(i + 2 * soln.sparseM.NNZ) = soln.sparseM.row(i) + 2*NN;
  soln.sparseM.col(i + soln.sparseM.NNZ)     = soln.sparseM.col(i) + NN;
  soln.sparseM.col(i + 2 * soln.sparseM.NNZ) = soln.sparseM.col(i) + 2*NN;
end

soln.sparseM.NNZ = 3 * soln.sparseM.NNZ;   % Triple the number of nonzeros.
soln.sparseM.value = zeros(soln.sparseM.NNZ,1);  % Allocate the value array




% Sparse storage of the K and A matrices are the same as M. Only extra
% value arrays are necessary.
soln.sparseK.value = zeros(soln.sparseM.NNZ,1);
soln.sparseA.value = zeros(soln.sparseM.NNZ,1);



% Determine local-to-sparse mapping, i.e. find the location of the entries
% of elemental sub-mass matrices in sparse storage. This will be used in
% the assembly process.

% First determine the nonzero entry number at the beginning of each row.
% rowStarts(NN+1) is equal to NNZ+1.
rowStarts = zeros(NN+1,1);
rowStarts(1) = 1;  % First row starts with the first entry
for i = 2:NN+1
  rowStarts(i) = rowStarts(i-1) + NNZcolInARow(i-1);
end


for e = 1:NE
  elem(e).sparseMapM = zeros(NENv,NENv);
  for i = 1:NENv
    r = elem(e).LtoGvel(i);
    col = soln.sparseM.col(rowStarts(r) : rowStarts(r+1) - 1);
    for j = 1:NENv
      % Find the location in the col array
      jj = elem(e).LtoGvel(j);
      for c = 1:size(col,1)
        if jj == col(c)
          break
        end
      end
      elem(e).sparseMapM(i,j) = c + rowStarts(r) - 1;
    end
  end
end

clear NZcolsInARow NNZcolInARow rowStarts;

% End of function setupSparseM()





%========================================================================
function [] = setupSparseG
%========================================================================
global NN NE NNp NENv NENp elem soln elemsOfVelNodes NelemOfVelNodes NelemOfPresNodes;

% Sets up row and column arrays of the global G matrix.

% First work only with the upper part of the matrix and then extend it
% for the lower part, which is identical to the upper part.


% In each row, find the columns with nonzero entries.

LARGE = max(NelemOfPresNodes);  % We know that not more than this many elements are connected to a pressure node.

soln.sparseG.NNZ = 0;  % Counts nonzero entries in sub-G matrix.

NNZcolInARow = zeros(NN,1);            % Number of nonzero columns in each row.
NZcolsInARow = zeros(NN,LARGE*NENp);   % Nonzero columns in a row.            % TODO: This array may take too much memory in 3D. Instead this information can be stored in a CSR type arrangement.

for r = 1:NN     % Loop over all rows.
  isColNZ = zeros(NNp,1);  % A flag array to store whether the column is zero or not.
                           % Stores similar information as NZcolsInARow,
                           % but makes counting nonzeros easier.
  colCount = 0;
  
  for i = 1:NelemOfVelNodes(r)   % NelemOfVelNodes(r) is the number of elements connected to node r
    e = elemsOfVelNodes(r,i);    % This element contributes to row r.
    for j = 1:NENp
      if isColNZ(elem(e).LtoGpres(j)) == 0   % 0 means this column had no previous non zero contribution.
        colCount = colCount + 1;
        isColNZ(elem(e).LtoGpres(j)) = 1;    % 1 means this column is non zero.
        NZcolsInARow(r,colCount) = elem(e).LtoGpres(j);
      end
    end
  end
  
  NNZcolInARow(r) = colCount;
  soln.sparseG.NNZ = soln.sparseG.NNZ + colCount;
end


% Entries in each row of NZcolsInARow are not sorted. Let's sort them out.
for r = 1:NN
  array = NZcolsInARow(r,1:NNZcolInARow(r));   % Nonzero columns in a row (unsorted)
  sortedArray = sort(array);
  NZcolsInARow(r,1:NNZcolInARow(r)) = sortedArray;
end



% Allocate memory for two vectors of soln.sparseG. Thinking about the whole
% G matrix, let's define the sizes properly by using three times of the
% calculated NNZ.
soln.sparseG.col = zeros(3 * soln.sparseG.NNZ,1);
soln.sparseG.row = zeros(3 * soln.sparseG.NNZ,1);


% Fill in soln.sparseG.col and soln.sparseG.row arrays that stores column
% and row numbers of each nonzero entry. This is done in a row-by-row way.
NNZcounter = 0;
for r = 1:NN     % Loop over all rows.
  for i = 1:NNZcolInARow(r)
    NNZcounter = NNZcounter + 1;
    soln.sparseG.row(NNZcounter) = r;
    soln.sparseG.col(NNZcounter) = NZcolsInARow(r,i);
  end
end



% Actually G matrix is not NNxNNp, but 3NNxNNp. Its middle and lower parts
% are the same as its upper part. What we set up above is only the upper
% part. Let's extend the row, col and value arrays to accomodate the middle
% and lower parts too.

for i = 1:soln.sparseG.NNZ
  soln.sparseG.row(i + soln.sparseG.NNZ)     = soln.sparseG.row(i) + NN;
  soln.sparseG.row(i + 2 * soln.sparseG.NNZ) = soln.sparseG.row(i) + 2*NN;
  soln.sparseG.col(i + soln.sparseG.NNZ)     = soln.sparseG.col(i);
  soln.sparseG.col(i + 2 * soln.sparseG.NNZ) = soln.sparseG.col(i);
end
soln.sparseG.NNZ = 3 * soln.sparseG.NNZ;   % Triple the number of nonzeros.
soln.sparseG.value = zeros(soln.sparseG.NNZ,1);  % Allocate the value array





% Determine local-to-sparse mapping, i.e. find the location of the entries
% of elemental sub-G matrices in sparse storage. This will be used in
% the assembly process.

% First determine the nonzero entry number at the beginning of each row.
% rowStarts(NN+1) is equal to NNZ+1.
rowStarts = zeros(NN+1,1);
rowStarts(1) = 1;  % First row starts with the first entry
for i = 2:NN+1
  rowStarts(i) = rowStarts(i-1) + NNZcolInARow(i-1);
end


for e = 1:NE
  elem(e).sparseMapG = zeros(NENv,NENp);
  for i = 1:NENv
    r = elem(e).LtoGvel(i);
    col = soln.sparseG.col(rowStarts(r) : rowStarts(r+1) - 1);
    for j = 1:NENp
      % Find the location in the col array
      jj = elem(e).LtoGpres(j);
      for c = 1:size(col,1)
        if jj == col(c)
          break
        end
      end
      elem(e).sparseMapG(i,j) = c + rowStarts(r) - 1;
    end
  end
end

clear elemsOfVelNodes NelemOfVelNodes elemsOfPresNodes NelemOfPresNodes;
clear NZcolsInARow NNZcolInARow;

% End of function setupSparseG()




%========================================================================
function [] = setupGQ
%========================================================================
global eType NGP GQ;

if eType == 1       % Hexahedral element
  if NGP == 1       % 1 point quadrature
    GQ.point(1,1) = 0.0;  GQ.point(1,2) = 0.0;  GQ.point(1,3) = 0.0;
    GQ.weight(1) = 4.0;                                                           % TODO: Is this correct?
  elseif NGP == 8   % 8 point quadrature
    GQ.point(1,1) = -sqrt(1/3);   GQ.point(1,2) = -sqrt(1/3);   GQ.point(1,3) = -sqrt(1/3);
    GQ.point(2,1) = sqrt(1/3);    GQ.point(2,2) = -sqrt(1/3);   GQ.point(2,3) = -sqrt(1/3);
    GQ.point(3,1) = -sqrt(1/3);   GQ.point(3,2) = sqrt(1/3);    GQ.point(3,3) = -sqrt(1/3);
    GQ.point(4,1) = sqrt(1/3);    GQ.point(4,2) = sqrt(1/3);    GQ.point(4,3) = -sqrt(1/3);
    GQ.point(5,1) = -sqrt(1/3);   GQ.point(5,2) = -sqrt(1/3);   GQ.point(5,3) = sqrt(1/3);
    GQ.point(6,1) = sqrt(1/3);    GQ.point(6,2) = -sqrt(1/3);   GQ.point(6,3) = sqrt(1/3);
    GQ.point(7,1) = -sqrt(1/3);   GQ.point(7,2) = sqrt(1/3);    GQ.point(7,3) = sqrt(1/3);
    GQ.point(8,1) = sqrt(1/3);    GQ.point(8,2) = sqrt(1/3);    GQ.point(8,3) = sqrt(1/3);
    GQ.weight(1) = 1.0;
    GQ.weight(2) = 1.0;
    GQ.weight(3) = 1.0;
    GQ.weight(4) = 1.0;
    GQ.weight(5) = 1.0;
    GQ.weight(6) = 1.0;
    GQ.weight(7) = 1.0;
    GQ.weight(8) = 1.0;
  elseif NGP == 27   % 27 point quadrature
    
    % TODO : ...
    
  end
  
elseif eType == 2   % Tetrahedral element  
  
  % TODO : ...
  
end

% End of function setupGQ()




%========================================================================
function [] = calcShape()
%========================================================================
global eType NGP NENv NENp GQ Sv Sp dSv dSp;

% Calculates the values of the shape functions and their derivatives with
% respect to ksi and eta at GQ points.

% Sv, Sp   : Shape functions for velocity and pressure approximation.
% dSv, dSp : ksi and eta derivatives of Sv and Sp.

if eType == 1  % Hexahedral element
  
  if NENp == 8
    for k = 1:NGP
      ksi  = GQ.point(k,1);
      eta  = GQ.point(k,2);
      zeta = GQ.point(k,3);
      
      Sp(1,k) = 0.125*(1-ksi)*(1-eta)*(1-zeta);
      Sp(2,k) = 0.125*(1+ksi)*(1-eta)*(1-zeta);
      Sp(3,k) = 0.125*(1+ksi)*(1+eta)*(1-zeta);
      Sp(4,k) = 0.125*(1-ksi)*(1+eta)*(1-zeta);
      Sp(5,k) = 0.125*(1-ksi)*(1-eta)*(1+zeta);
      Sp(6,k) = 0.125*(1+ksi)*(1-eta)*(1+zeta);
      Sp(7,k) = 0.125*(1+ksi)*(1+eta)*(1+zeta);
      Sp(8,k) = 0.125*(1-ksi)*(1+eta)*(1+zeta);
      
      % ksi derivatives of Sp
      dSp(1,1,k) = -0.125*(1-eta)*(1-zeta);
      dSp(1,2,k) =  0.125*(1-eta)*(1-zeta);
      dSp(1,3,k) =  0.125*(1+eta)*(1-zeta);
      dSp(1,4,k) = -0.125*(1+eta)*(1-zeta);
      dSp(1,5,k) = -0.125*(1-eta)*(1+zeta);
      dSp(1,6,k) =  0.125*(1-eta)*(1+zeta);
      dSp(1,7,k) =  0.125*(1+eta)*(1+zeta);
      dSp(1,8,k) = -0.125*(1+eta)*(1+zeta);
      
      % eta derivatives of Sp
      dSp(2,1,k) = -0.125*(1-ksi)*(1-zeta);
      dSp(2,2,k) = -0.125*(1+ksi)*(1-zeta);
      dSp(2,3,k) =  0.125*(1+ksi)*(1-zeta);
      dSp(2,4,k) =  0.125*(1-ksi)*(1-zeta);
      dSp(2,5,k) = -0.125*(1-ksi)*(1+zeta);
      dSp(2,6,k) = -0.125*(1+ksi)*(1+zeta);
      dSp(2,7,k) =  0.125*(1+ksi)*(1+zeta);
      dSp(2,8,k) =  0.125*(1-ksi)*(1+zeta);
      
      % zeta derivatives of Sp
      dSp(3,1,k) = -0.125*(1-ksi)*(1-eta);
      dSp(3,2,k) = -0.125*(1+ksi)*(1-eta);
      dSp(3,3,k) = -0.125*(1+ksi)*(1+eta);
      dSp(3,4,k) = -0.125*(1-ksi)*(1+eta);
      dSp(3,5,k) =  0.125*(1-ksi)*(1-eta);
      dSp(3,6,k) =  0.125*(1+ksi)*(1-eta);
      dSp(3,7,k) =  0.125*(1+ksi)*(1+eta);
      dSp(3,8,k) =  0.125*(1-ksi)*(1+eta);
    end
  else
    fprintf('\n\n\n ERROR: Only NENp = 8 is supported for hexahedral elements.\n\n\n');
  end
  
  if NENv == 8
    Sv = Sp;
    dSv = dSp;
  elseif NENv == 27
    for k = 1:NGP
      ksi  = GQ.point(k,1);
      eta  = GQ.point(k,2);
      zeta = GQ.point(k,3);
      
      Sv(1,k) = 0.125 * (ksi^2 - ksi) * (eta^2 - eta) * (zeta^2 - zeta);
      Sv(2,k) = 0.125 * (ksi^2 + ksi) * (eta^2 - eta) * (zeta^2 - zeta);
      Sv(3,k) = 0.125 * (ksi^2 + ksi) * (eta^2 + eta) * (zeta^2 - zeta);
      Sv(4,k) = 0.125 * (ksi^2 - ksi) * (eta^2 + eta) * (zeta^2 - zeta);
      Sv(5,k) = 0.125 * (ksi^2 - ksi) * (eta^2 - eta) * (zeta^2 + zeta);
      Sv(6,k) = 0.125 * (ksi^2 + ksi) * (eta^2 - eta) * (zeta^2 + zeta);
      Sv(7,k) = 0.125 * (ksi^2 + ksi) * (eta^2 + eta) * (zeta^2 + zeta);
      Sv(8,k) = 0.125 * (ksi^2 - ksi) * (eta^2 + eta) * (zeta^2 + zeta);
      
      Sv(9,k)  = 0.25 * (1 - ksi^2) * (eta^2 - eta) * (zeta^2 - zeta);
      Sv(10,k) = 0.25 * (ksi^2 + ksi) * (1 - eta^2) * (zeta^2 - zeta);
      Sv(11,k) = 0.25 * (1 - ksi^2) * (eta^2 + eta) * (zeta^2 - zeta);
      Sv(12,k) = 0.25 * (ksi^2 - ksi) * (1 - eta^2) * (zeta^2 - zeta);
      
      Sv(13,k) = 0.25 * (ksi^2 - ksi) * (eta^2 - eta) * (1 - zeta^2);
      Sv(14,k) = 0.25 * (ksi^2 + ksi) * (eta^2 - eta) * (1 - zeta^2);
      Sv(15,k) = 0.25 * (ksi^2 + ksi) * (eta^2 + eta) * (1 - zeta^2);
      Sv(16,k) = 0.25 * (ksi^2 - ksi) * (eta^2 + eta) * (1 - zeta^2);
      
      Sv(17,k) = 0.25 * (1 - ksi^2) * (eta^2 - eta) * (zeta^2 + zeta);
      Sv(18,k) = 0.25 * (ksi^2 + ksi) * (1 - eta^2) * (zeta^2 + zeta);
      Sv(19,k) = 0.25 * (1 - ksi^2) * (eta^2 + eta) * (zeta^2 + zeta);
      Sv(20,k) = 0.25 * (ksi^2 - ksi) * (1 - eta^2) * (zeta^2 + zeta);
      
      Sv(21,k) = 0.5 * (1 - ksi^2) * (1 - eta^2) * (zeta^2 - zeta);
      Sv(22,k) = 0.5 * (1 - ksi^2) * (eta^2 - eta) * (1 - zeta^2);
      Sv(23,k) = 0.5 * (ksi^2 + ksi) * (1 - eta^2) * (1 - zeta^2);
      Sv(24,k) = 0.5 * (1 - ksi^2) * (eta^2 + eta) * (1 - zeta^2);
      Sv(25,k) = 0.5 * (ksi^2 - ksi) * (1 - eta^2) * (1 - zeta^2);
      Sv(26,k) = 0.5 * (1 - ksi^2) * (1 - eta^2) * (zeta^2 + zeta);

      Sv(27,k) = (1 - ksi^2) * (1 - eta^2) * (1 - zeta^2);

      % ksi derivatives of Sv
      dSv(1,1,k) = 0.125 * (2*ksi - 1) * (eta^2 - eta) * (zeta^2 - zeta);
      dSv(1,2,k) = 0.125 * (2*ksi + 1) * (eta^2 - eta) * (zeta^2 - zeta);
      dSv(1,3,k) = 0.125 * (2*ksi + 1) * (eta^2 + eta) * (zeta^2 - zeta);
      dSv(1,4,k) = 0.125 * (2*ksi - 1) * (eta^2 + eta) * (zeta^2 - zeta);
      dSv(1,5,k) = 0.125 * (2*ksi - 1) * (eta^2 - eta) * (zeta^2 + zeta);
      dSv(1,6,k) = 0.125 * (2*ksi + 1) * (eta^2 - eta) * (zeta^2 + zeta);
      dSv(1,7,k) = 0.125 * (2*ksi + 1) * (eta^2 + eta) * (zeta^2 + zeta);
      dSv(1,8,k) = 0.125 * (2*ksi - 1) * (eta^2 + eta) * (zeta^2 + zeta);
      
      dSv(1,9,k)  = 0.25 * (- 2*ksi) * (eta^2 - eta) * (zeta^2 - zeta);
      dSv(1,10,k) = 0.25 * (2*ksi + 1) * (1 - eta^2) * (zeta^2 - zeta);
      dSv(1,11,k) = 0.25 * (- 2*ksi) * (eta^2 + eta) * (zeta^2 - zeta);
      dSv(1,12,k) = 0.25 * (2*ksi - 1) * (1 - eta^2) * (zeta^2 - zeta);
      
      dSv(1,13,k) = 0.25 * (2*ksi - 1) * (eta^2 - eta) * (1 - zeta^2);
      dSv(1,14,k) = 0.25 * (2*ksi + 1) * (eta^2 - eta) * (1 - zeta^2);
      dSv(1,15,k) = 0.25 * (2*ksi + 1) * (eta^2 + eta) * (1 - zeta^2);
      dSv(1,16,k) = 0.25 * (2*ksi - 1) * (eta^2 + eta) * (1 - zeta^2);
      
      dSv(1,17,k) = 0.25 * (- 2*ksi) * (eta^2 - eta) * (zeta^2 + zeta);
      dSv(1,18,k) = 0.25 * (2*ksi + 1) * (1 - eta^2) * (zeta^2 + zeta);
      dSv(1,19,k) = 0.25 * (- 2*ksi) * (eta^2 + eta) * (zeta^2 + zeta);
      dSv(1,20,k) = 0.25 * (2*ksi - 1) * (1 - eta^2) * (zeta^2 + zeta);
      
      dSv(1,21,k) = 0.5 * (- 2*ksi) * (1 - eta^2) * (zeta^2 - zeta);
      dSv(1,22,k) = 0.5 * (- 2*ksi) * (eta^2 - eta) * (1 - zeta^2);
      dSv(1,23,k) = 0.5 * (2*ksi + 1) * (1 - eta^2) * (1 - zeta^2);
      dSv(1,24,k) = 0.5 * (- 2*ksi) * (eta^2 + eta) * (1 - zeta^2);
      dSv(1,25,k) = 0.5 * (2*ksi - 1) * (1 - eta^2) * (1 - zeta^2);
      dSv(1,26,k) = 0.5 * (- 2*ksi) * (1 - eta^2) * (zeta^2 + zeta);

      dSv(1,27,k) = (- 2*ksi) * (1 - eta^2) * (1 - zeta^2);
      
      
      % eta derivatives of Sv
      dSv(2,1,k) = 0.125 * (ksi^2 - ksi) * (2*eta - 1) * (zeta^2 - zeta);
      dSv(2,2,k) = 0.125 * (ksi^2 + ksi) * (2*eta - 1) * (zeta^2 - zeta);
      dSv(2,3,k) = 0.125 * (ksi^2 + ksi) * (2*eta + 1) * (zeta^2 - zeta);
      dSv(2,4,k) = 0.125 * (ksi^2 - ksi) * (2*eta + 1) * (zeta^2 - zeta);
      dSv(2,5,k) = 0.125 * (ksi^2 - ksi) * (2*eta - 1) * (zeta^2 + zeta);  
      dSv(2,6,k) = 0.125 * (ksi^2 + ksi) * (2*eta - 1) * (zeta^2 + zeta);
      dSv(2,7,k) = 0.125 * (ksi^2 + ksi) * (2*eta + 1) * (zeta^2 + zeta);
      dSv(2,8,k) = 0.125 * (ksi^2 - ksi) * (2*eta + 1) * (zeta^2 + zeta);
      
      dSv(2,9,k)  = 0.25 * (1 - ksi^2) * (2*eta - 1) * (zeta^2 - zeta);
      dSv(2,10,k) = 0.25 * (ksi^2 + ksi) * (- 2*eta) * (zeta^2 - zeta);
      dSv(2,11,k) = 0.25 * (1 - ksi^2) * (2*eta + 1) * (zeta^2 - zeta);
      dSv(2,12,k) = 0.25 * (ksi^2 - ksi) * (- 2*eta) * (zeta^2 - zeta);
      
      dSv(2,13,k) = 0.25 * (ksi^2 - ksi) * (2*eta - 1) * (1 - zeta^2);
      dSv(2,14,k) = 0.25 * (ksi^2 + ksi) * (2*eta - 1) * (1 - zeta^2);
      dSv(2,15,k) = 0.25 * (ksi^2 + ksi) * (2*eta + 1) * (1 - zeta^2);
      dSv(2,16,k) = 0.25 * (ksi^2 - ksi) * (2*eta + 1) * (1 - zeta^2);
      
      dSv(2,17,k) = 0.25 * (1 - ksi^2) * (2*eta - 1) * (zeta^2 + zeta);
      dSv(2,18,k) = 0.25 * (ksi^2 + ksi) * (- 2*eta) * (zeta^2 + zeta);
      dSv(2,19,k) = 0.25 * (1 - ksi^2) * (2*eta + 1) * (zeta^2 + zeta);
      dSv(2,20,k) = 0.25 * (ksi^2 - ksi) * (- 2*eta) * (zeta^2 + zeta);
      
      dSv(2,21,k) = 0.5 * (1 - ksi^2) * (- 2*eta) * (zeta^2 - zeta);
      dSv(2,22,k) = 0.5 * (1 - ksi^2) * (2*eta - 1) * (1 - zeta^2);
      dSv(2,23,k) = 0.5 * (ksi^2 + ksi) * (- 2*eta) * (1 - zeta^2);
      dSv(2,24,k) = 0.5 * (1 - ksi^2) * (2*eta + 1) * (1 - zeta^2);
      dSv(2,25,k) = 0.5 * (ksi^2 - ksi) * (- 2*eta) * (1 - zeta^2);
      dSv(2,26,k) = 0.5 * (1 - ksi^2) * (- 2*eta) * (zeta^2 + zeta);

      dSv(2,27,k) = (1 - ksi^2) * (- 2*eta) * (1 - zeta^2);
     
      
      % zeta derivatives of Sv
      dSv(3,1,k) = 0.125 * (ksi^2 - ksi) * (eta^2 - eta) * (2*zeta - 1);
      dSv(3,2,k) = 0.125 * (ksi^2 + ksi) * (eta^2 - eta) * (2*zeta - 1);
      dSv(3,3,k) = 0.125 * (ksi^2 + ksi) * (eta^2 + eta) * (2*zeta - 1);
      dSv(3,4,k) = 0.125 * (ksi^2 - ksi) * (eta^2 + eta) * (2*zeta - 1);
      dSv(3,5,k) = 0.125 * (ksi^2 - ksi) * (eta^2 - eta) * (2*zeta + 1);
      dSv(3,6,k) = 0.125 * (ksi^2 + ksi) * (eta^2 - eta) * (2*zeta + 1);
      dSv(3,7,k) = 0.125 * (ksi^2 + ksi) * (eta^2 + eta) * (2*zeta + 1);
      dSv(3,8,k) = 0.125 * (ksi^2 - ksi) * (eta^2 + eta) * (2*zeta + 1);
      
      dSv(3,9,k)  = 0.25 * (1 - ksi^2) * (eta^2 - eta) * (2*zeta - 1);
      dSv(3,10,k) = 0.25 * (ksi^2 + ksi) * (1 - eta^2) * (2*zeta - 1);
      dSv(3,11,k) = 0.25 * (1 - ksi^2) * (eta^2 + eta) * (2*zeta - 1);
      dSv(3,12,k) = 0.25 * (ksi^2 - ksi) * (1 - eta^2) * (2*zeta - 1);
      
      dSv(3,13,k) = 0.25 * (ksi^2 - ksi) * (eta^2 - eta) * (- 2*zeta);
      dSv(3,14,k) = 0.25 * (ksi^2 + ksi) * (eta^2 - eta) * (- 2*zeta);
      dSv(3,15,k) = 0.25 * (ksi^2 + ksi) * (eta^2 + eta) * (- 2*zeta);
      dSv(3,16,k) = 0.25 * (ksi^2 - ksi) * (eta^2 + eta) * (- 2*zeta);
      
      dSv(3,17,k) = 0.25 * (1 - ksi^2) * (eta^2 - eta) * (2*zeta + 1);
      dSv(3,18,k) = 0.25 * (ksi^2 + ksi) * (1 - eta^2) * (2*zeta + 1);
      dSv(3,19,k) = 0.25 * (1 - ksi^2) * (eta^2 + eta) * (2*zeta + 1);
      dSv(3,20,k) = 0.25 * (ksi^2 - ksi) * (1 - eta^2) * (2*zeta + 1);
      
      dSv(3,21,k) = 0.5 * (1 - ksi^2) * (1 - eta^2) * (2*zeta - 1);
      dSv(3,22,k) = 0.5 * (1 - ksi^2) * (eta^2 - eta) * (- 2*zeta);
      dSv(3,23,k) = 0.5 * (ksi^2 + ksi) * (1 - eta^2) * (- 2*zeta);
      dSv(3,24,k) = 0.5 * (1 - ksi^2) * (eta^2 + eta) * (- 2*zeta);
      dSv(3,25,k) = 0.5 * (ksi^2 - ksi) * (1 - eta^2) * (- 2*zeta);
      dSv(3,26,k) = 0.5 * (1 - ksi^2) * (1 - eta^2) * (2*zeta + 1);

      dSv(3,27,k) = (1 - ksi^2) * (1 - eta^2) * (- 2*zeta);
    end
  else
    fprintf('\n\n\n ERROR: Only NENv = 8 and 27 are supported for hexahedral elements.\n\n\n');
  end
  
elseif eType == 2  % Tetrahedral element
  
  % TODO : ...
  
end


% % Control of Kronecker-Delta property
% 
% % Corner point coordinates of the master element
% points(1,1) = -1;
% points(4,1) = -1;
% points(5,1) = -1;
% points(8,1) = -1;
% points(12,1) = -1;
% points(13,1) = -1;
% points(16,1) = -1;
% points(20,1) = -1;
% points(25,1) = -1;
% 
% points(9,1) = 0;
% points(11,1) = 0;
% points(17,1) = 0;
% points(19,1) = 0;
% points(21,1) = 0;
% points(22,1) = 0;
% points(24,1) = 0;
% points(26,1) = 0;
% points(27,1) = 0;
% 
% points(2,1) = 1;
% points(3,1) = 1;
% points(6,1) = 1;
% points(7,1) = 1;
% points(10,1) = 1;
% points(14,1) = 1;
% points(15,1) = 1;
% points(18,1) = 1;
% points(23,1) = 1;
% 
% points(1,2) = -1;
% points(2,2) = -1;
% points(5,2) = -1;
% points(6,2) = -1;
% points(9,2) = -1;
% points(13,2) = -1;
% points(14,2) = -1;
% points(17,2) = -1;
% points(22,2) = -1;
% 
% points(10,2) = 0;
% points(12,2) = 0;
% points(18,2) = 0;
% points(20,2) = 0;
% points(21,2) = 0;
% points(23,2) = 0;
% points(25,2) = 0;
% points(26,2) = 0;
% points(27,2) = 0;
% 
% points(3,2) = 1;
% points(4,2) = 1;
% points(7,2) = 1;
% points(8,2) = 1;
% points(11,2) = 1;
% points(15,2) = 1;
% points(16,2) = 1;
% points(19,2) = 1;
% points(24,2) = 1;
% 
% points(1,3) = -1;
% points(2,3) = -1;
% points(3,3) = -1;
% points(4,3) = -1;
% points(9,3) = -1;
% points(10,3) = -1;
% points(11,3) = -1;
% points(12,3) = -1;
% points(21,3) = -1;
% 
% points(13,3) = 0;
% points(14,3) = 0;
% points(15,3) = 0;
% points(16,3) = 0;
% points(22,3) = 0;
% points(23,3) = 0;
% points(24,3) = 0;
% points(25,3) = 0;
% points(27,3) = 0;
% 
% points(5,3) = 1;
% points(6,3) = 1;
% points(7,3) = 1;
% points(8,3) = 1;
% points(17,3) = 1;
% points(18,3) = 1;
% points(19,3) = 1;
% points(20,3) = 1;
% points(26,3) = 1;
% 
% 
% for i = 1:8
%   ksi = points(i,1);
%   eta = points(i,2);
%   zeta = points(i,3);
% 
%   Sp(1) = 0.125*(1-ksi)*(1-eta)*(1-zeta);
%   Sp(2) = 0.125*(1+ksi)*(1-eta)*(1-zeta);
%   Sp(3) = 0.125*(1+ksi)*(1+eta)*(1-zeta);
%   Sp(4) = 0.125*(1-ksi)*(1+eta)*(1-zeta);
%   Sp(5) = 0.125*(1-ksi)*(1-eta)*(1+zeta);
%   Sp(6) = 0.125*(1+ksi)*(1-eta)*(1+zeta);
%   Sp(7) = 0.125*(1+ksi)*(1+eta)*(1+zeta);
%   Sp(8) = 0.125*(1-ksi)*(1+eta)*(1+zeta);
%   
%   if Sp(i) ~= 1
%     fprintf('\n\nERROR 1: %d\n\n', i);
%   end
%   
%   for j = 1:8
%     if i ~= j && Sp(j) ~= 0
%       fprintf('\n\nERROR 2: %d %d\n\n', i, j);
%     end
%   end
% end
% 
% 
% for i = 1:27
%   ksi = points(i,1);
%   eta = points(i,2);
%   zeta = points(i,3);
%   
%   Sv(1) = 0.125 * (ksi^2 - ksi) * (eta^2 - eta) * (zeta^2 - zeta);
%   Sv(2) = 0.125 * (ksi^2 + ksi) * (eta^2 - eta) * (zeta^2 - zeta);
%   Sv(3) = 0.125 * (ksi^2 + ksi) * (eta^2 + eta) * (zeta^2 - zeta);
%   Sv(4) = 0.125 * (ksi^2 - ksi) * (eta^2 + eta) * (zeta^2 - zeta);
%   Sv(5) = 0.125 * (ksi^2 - ksi) * (eta^2 - eta) * (zeta^2 + zeta);
%   Sv(6) = 0.125 * (ksi^2 + ksi) * (eta^2 - eta) * (zeta^2 + zeta);
%   Sv(7) = 0.125 * (ksi^2 + ksi) * (eta^2 + eta) * (zeta^2 + zeta);
%   Sv(8) = 0.125 * (ksi^2 - ksi) * (eta^2 + eta) * (zeta^2 + zeta);
% 
%   Sv(9)  = 0.25 * (1 - ksi^2) * (eta^2 - eta) * (zeta^2 - zeta);
%   Sv(10) = 0.25 * (ksi^2 + ksi) * (1 - eta^2) * (zeta^2 - zeta);
%   Sv(11) = 0.25 * (1 - ksi^2) * (eta^2 + eta) * (zeta^2 - zeta);
%   Sv(12) = 0.25 * (ksi^2 - ksi) * (1 - eta^2) * (zeta^2 - zeta);
% 
%   Sv(13) = 0.25 * (ksi^2 - ksi) * (eta^2 - eta) * (1 - zeta^2);
%   Sv(14) = 0.25 * (ksi^2 + ksi) * (eta^2 - eta) * (1 - zeta^2);
%   Sv(15) = 0.25 * (ksi^2 + ksi) * (eta^2 + eta) * (1 - zeta^2);
%   Sv(16) = 0.25 * (ksi^2 - ksi) * (eta^2 + eta) * (1 - zeta^2);
% 
%   Sv(17) = 0.25 * (1 - ksi^2) * (eta^2 - eta) * (zeta^2 + zeta);
%   Sv(18) = 0.25 * (ksi^2 + ksi) * (1 - eta^2) * (zeta^2 + zeta);
%   Sv(19) = 0.25 * (1 - ksi^2) * (eta^2 + eta) * (zeta^2 + zeta);
%   Sv(20) = 0.25 * (ksi^2 - ksi) * (1 - eta^2) * (zeta^2 + zeta);
% 
%   Sv(21) = 0.5 * (1 - ksi^2) * (1 - eta^2) * (zeta^2 - zeta);
%   Sv(22) = 0.5 * (1 - ksi^2) * (eta^2 - eta) * (1 - zeta^2);
%   Sv(23) = 0.5 * (ksi^2 + ksi) * (1 - eta^2) * (1 - zeta^2);
%   Sv(24) = 0.5 * (1 - ksi^2) * (eta^2 + eta) * (1 - zeta^2);
%   Sv(25) = 0.5 * (ksi^2 - ksi) * (1 - eta^2) * (1 - zeta^2);
%   Sv(26) = 0.5 * (1 - ksi^2) * (1 - eta^2) * (zeta^2 + zeta);
% 
%   Sv(27) = (1 - ksi^2) * (1 - eta^2) * (1 - zeta^2);
%   
%   if Sv(i) ~= 1
%     fprintf('\n\nERROR 3: %d\n\n', i);
%   end
%   
%   for j = 1:27
%     if i ~= j && Sv(j) ~= 0
%       fprintf('\n\nERROR 4: %d %d\n\n', i, j);
%     end
%   end
% end

% End of function calcShape()




%========================================================================
function [] = calcJacob()
%========================================================================
global NE NEC NGP coord dSv dSp elem;

% Calculates Jacobian matrix and its determinant for each element. Shape
% functions for corner nodes (pressure nodes) are used for Jacobian
% calculation.
% Also calculates and stores the derivatives of velocity shape functions
% wrt x and y at GQ points for each element.

for e = 1:NE
  % To calculate the Jacobian matrix first generate e_coord matrix of size
  % NECx2. Each row of it stores x and y coords of the nodes of elem e.
  for i = 1:NEC
    iG = elem(e).LtoGnode(i);
    e_coord(i,:) = coord(iG,:);  % Copy both x and y cooordinates at once.
  end

  % For each GQ point calculate the 2x2 Jacobian matrix, its inverse and
  % determinant. Only store the determinant for each element. Also
  % calculate and store the shape function derivatives wrt x and y.
  for k = 1:NGP
    Jacob(:,:) = dSp(:,:,k) * e_coord(:,:);  % Sp's are used for Jacobian     % TODO : Depending on NENv and NENp Sp or Sv can be used for Jacobian calculation.
                                             % calculation.
    elem(e).detJacob(k) = det(Jacob);
    
    % Calculate x and y derivatives of Sv and Sp.
    elem(e).gDSv(:,:,k) = inv(Jacob(:,:)) * dSv(:,:,k);
    elem(e).gDSp(:,:,k) = inv(Jacob(:,:)) * dSp(:,:,k);
  end
end

% End of function calcJacob()




%========================================================================
function [] = initializeAndAllocate()
%========================================================================
% Do the necessary memory allocations. Apply the initial condition or read
% the restart file.

global NN NNp soln time;

soln.Un           = zeros(3*NN,1);     % x, y and z velocity components of time step n.
soln.Unp1         = zeros(3*NN,1);     % U_i+1^n+1 of the reference paper.
soln.Unp1_prev    = zeros(3*NN,1);     % U_i^n+1 of the reference paper.
soln.UnpHalf      = zeros(3*NN,1);     % U_i+1^n+1/2 of the reference paper.
soln.UnpHalf_prev = zeros(3*NN,1);     % U_i^n+1/2 of the reference paper.

soln.AccHalf      = zeros(3*NN,1);     % A_i+1^n+1/2 of the reference paper.
soln.Acc          = zeros(3*NN,1);     % A_i+1^n+1 of the reference paper.
soln.Acc_prev     = zeros(3*NN,1);     % A_i^n+1 of the reference paper.

soln.Pn        = zeros(NNp,1);         % Pressure of time step n.
soln.Pnp1      = zeros(NNp,1);         % U_i+1^n+1 of the reference paper.
soln.Pnp1_prev = zeros(NNp,1);         % p_i+1^n+1 of the reference paper.
soln.Pdot      = zeros(NNp,1);         % Pdot_i+1^n+1 of the reference paper.

soln.Md        = zeros(3*NN, 1);       % Diagonalized mass matrix with BCs applied
soln.MdOrig    = zeros(3*NN, 1);       % Diagonalized mass matrix without BCs applied
soln.MdOrigInv = zeros(3*NN, 1);       % Inverse of the diagonalized mass matrix without BCs applied

soln.R1 = zeros(3*NN, 1);              % RHS vector of intermediate velocity calculation.
soln.R2 = zeros(NNp, 1);               % RHS vector of pressure calculation.
soln.R3 = zeros(3*NN, 1);              % RHS vector of new velocity calculation.


% Read the restart file if isRestart is equal to 1. If not, apply the
% specified BCs.
if time.isRestart == 1
  readRestartFile();
else
  applyBC_initial();
end

createTecplot();

% Initialize time step counter and time.
time.s = 0;
time.t = time.initial;

% End of function initializeAndAllocate()




%========================================================================
function [] = timeLoop()
%========================================================================
% Main time loop of the solution.
global NN soln time maxIter tolerance monPoint coord;

% Initialize the solution using the specified initial condition and do
% memory allocations.
initializeAndAllocate();


% Calculate certain matrices and their inverses only once before the time loop.
ti = tic;
step0();
fprintf('step0                 took %7.3f sec.\n\n', toc(ti))

updateProfilePlots(1);

fprintf('Monitoring node is %d, with coordinates [%f, %f, %f].\n\n', ...
        monPoint.node, coord(monPoint.node,1), coord(monPoint.node,2), coord(monPoint.node,3));

fprintf('Time step  Iter     Time       u_monitor     v_monitor     w_monitor     p_monitor\n');
fprintf('-------------------------------------------------------------------------------------\n');


while (time.t < time.final)  % Time loop
  time.s = time.s + 1;
  time.t = time.t + time.dt;
  
  % Initialize variables for the first iteration of the following loop.
  soln.UnpHalf_prev = soln.Un;
  soln.Unp1_prev = soln.Un;
  soln.Pnp1_prev = soln.Pn;
  soln.Acc_prev = zeros(3*NN,1);
  
  % Iterations inside a time step
  for iter=1:maxIter
    % Calculate intermediate velocity.
    ti = tic;
    step1(iter);
    fprintf('\nstep1                 took %7.3f sec.', toc(ti))
  
    % Calculate pressure of the new time step
    ti = tic;
    step2();
    fprintf('\nstep2                 took %7.3f sec.', toc(ti))
  
    % Calculate velocity of the new time step
    ti = tic;
    step3();
    fprintf('\nstep3                 took %7.3f sec.\n', toc(ti))
    
    
    % Check for convergence
    if norm(soln.Unp1 - soln.Unp1_prev) / norm(soln.Unp1) < tolerance && ...
       norm(soln.Pnp1 - soln.Pnp1_prev) / norm(soln.Pnp1) < tolerance
      break;
    end
    
    % Get ready for the next iteration
    soln.UnpHalf_prev = soln.UnpHalf;
    soln.Unp1_prev = soln.Unp1;
    soln.Pnp1_prev = soln.Pnp1;
    soln.Acc_prev = soln.Acc;
  end
  
  
  % Get ready for the next time step
  soln.Un = soln.Unp1;
  soln.Pn = soln.Pnp1;
  
  
  if (mod(time.s, 20) == 0 || abs(time.t - time.final) < 1e-10)
    createTecplot();
    updateProfilePlots(2);
  end
  
  
  mP = monPoint.node;    % Monitor point
  fprintf('\n%6d  %6d  %10.5f  %12.5f  %12.5f  %12.5f  %12.5f\n', time.s, iter, ...
       time.t, soln.Un(mP), soln.Un(NN+mP), soln.Un(2*NN+mP), soln.Pn(mP));
  
end  % End of while loop for time

% End of function timeLoop()




%========================================================================
function step0()
%========================================================================
% Calculates [M], [G] and [K] only once, before the time loop.
global NE NENv NENp soln elem NGP GQ viscosity time;
global Sv Sp;

nnzM = soln.sparseM.NNZ / 3;
nnzM2 = 2 * nnzM;
nnzM3 = 3 * nnzM;

nnzG = soln.sparseG.NNZ / 3;
nnzG2 = 2 * nnzG;


% Calculate Me, Ke and Ge, and  them into M, K and G
for e = 1:NE
  %tic
  Me_11 = zeros(NENv, NENv);
  Ke_11 = zeros(NENv, NENv);
  Ge_1  = zeros(NENv, NENp);
  Ge_2  = zeros(NENv, NENp);
  Ge_3  = zeros(NENv, NENp);

  for k = 1:NGP   % Gauss Quadrature loop
    % Define shortcuts
    gDSv(:,:) = elem(e).gDSv(:,:,k);
    GQfactor = elem(e).detJacob(k) * GQ.weight(k);
    
    for i = 1:NENv
      for j = 1:NENv
        Me_11(i,j) = Me_11(i,j) + Sv(i,k) * Sv(j,k) * GQfactor;
        Ke_11(i,j) = Ke_11(i,j) + viscosity * (gDSv(1,i) * gDSv(1,j) + ...
                                               gDSv(2,i) * gDSv(2,j) + ...
                                               gDSv(3,i) * gDSv(3,j)) * GQfactor;
      end
    end

    for i = 1:NENv
      for j = 1:NENp
        Ge_1(i,j) = Ge_1(i,j) - Sp(j,k) * gDSv(1,i) * GQfactor;
        Ge_2(i,j) = Ge_2(i,j) - Sp(j,k) * gDSv(2,i) * GQfactor;
        Ge_3(i,j) = Ge_3(i,j) - Sp(j,k) * gDSv(3,i) * GQfactor;
      end
    end
    
  end  % End of GQ loop
  
  
  % Assemble Me and Ke into sparse M and K.
  for i = 1:NENv
    for j = 1:NENv
      soln.sparseM.value(elem(e).sparseMapM(i,j)) = ...   % Assemble upper left sub-matrix of M
        soln.sparseM.value(elem(e).sparseMapM(i,j)) + Me_11(i,j);
      
      soln.sparseM.value(elem(e).sparseMapM(i,j) + nnzM) = ...   % Assemble middle sub-matrix of M
        soln.sparseM.value(elem(e).sparseMapM(i,j) + nnzM) + Me_11(i,j);

      soln.sparseM.value(elem(e).sparseMapM(i,j) + nnzM2) = ...   % Assemble lower right sub-matrix of M
        soln.sparseM.value(elem(e).sparseMapM(i,j) + nnzM2) + Me_11(i,j);

      soln.sparseK.value(elem(e).sparseMapM(i,j)) = ...   % Assemble upper left sub-matrix of K
        soln.sparseK.value(elem(e).sparseMapM(i,j)) + Ke_11(i,j);
      
      soln.sparseK.value(elem(e).sparseMapM(i,j) + nnzM) = ...   % Assemble middle sub-matrix of K
        soln.sparseK.value(elem(e).sparseMapM(i,j) + nnzM) + Ke_11(i,j);

      soln.sparseK.value(elem(e).sparseMapM(i,j) + nnzM2) = ...   % Assemble lower right sub-matrix of K
        soln.sparseK.value(elem(e).sparseMapM(i,j) + nnzM2) + Ke_11(i,j);
    end
  end
  
%  Following commented lines seems to be more efficient than the above
%  loop, but surprisingly they are not.
  
%  soln.sparseM.value(nnzM+1:nnzM2) = soln.sparseM.value(1:nnzM);
%  soln.sparseM.value(nnzM2+1:nnzM3) = soln.sparseM.value(1:nnzM);

%  soln.sparseK.value(nnzM+1:nnzM2) = soln.sparseK.value(1:nnzM);
%  soln.sparseK.value(nnzM2+1:nnzM3) = soln.sparseK.value(1:nnzM);
  

  % Assemble Ge into sparse G.
  for i = 1:NENv
    for j = 1:NENp
      soln.sparseG.value(elem(e).sparseMapG(i,j)) = ...   % Assemble upper part of G
        soln.sparseG.value(elem(e).sparseMapG(i,j)) + Ge_1(i,j);
      
      soln.sparseG.value(elem(e).sparseMapG(i,j) + nnzG) = ...   % Assemble middle part of G
        soln.sparseG.value(elem(e).sparseMapG(i,j) + nnzG) + Ge_2(i,j);

      soln.sparseG.value(elem(e).sparseMapG(i,j) + nnzG2) = ...   % Assemble lower part of G
        soln.sparseG.value(elem(e).sparseMapG(i,j) + nnzG2) + Ge_3(i,j);
    end
  end
  
end   % Element loop

% CONTROL
%for i = 1:soln.sparseG.NNZ
%   fprintf('%d  %f \n', i, soln.sparseG.value(i));
%end


% For sparse M, K and G using row, col and value vectors.
soln.M = sparse(soln.sparseM.row, soln.sparseM.col, soln.sparseM.value);
soln.K = sparse(soln.sparseM.row, soln.sparseM.col, soln.sparseK.value);
soln.G = sparse(soln.sparseG.row, soln.sparseG.col, soln.sparseG.value);

soln.Gt = transpose(soln.G);


% Find the diagonalized mass matrix
for i = 1:nnzM3
  row = soln.sparseM.row(i);
  soln.Md(row) = soln.Md(row) + soln.sparseM.value(i);
end


% Get a copy of soln.Md before modifying it for the BCs of step 1.
soln.MdOrig = soln.Md;

% Calculate the inverse of Md.
soln.MdOrigInv = 1.0 ./ soln.MdOrig;


% Apply velocity BCs to Md and calculate its inverse.
applyBC_Step1(1);
soln.MdInv = 1.0 ./ soln.Md;


% Calculate inv(Md) * G
soln.MdOrigInvTimesG = bsxfun(@times, soln.MdOrigInv, soln.G);


% CONTROL
%for i = 1:soln.sparseG.NNZ
%   fprintf('%d  %f \n', i, soln.sparseG.value(i));
%end


% Calculate inv(Md) * K
soln.MdOrigInvTimesK = bsxfun(@times, soln.MdOrigInv, soln.K);                                    % TODO: Actually this is not necessary. See the comment in step2()


% Calculate [Z] = dt^2 * transpose(G) * inv(M) * G
soln.Z = time.dt^2 * soln.Gt * soln.MdOrigInvTimesG;

% Apply pressure BCs to [Z] and calculate its Cholesky factorization.
applyBC_Step2(1);

%soln.RCMorderOfZ = symrcm(soln.Z);  % Reverse Cuthill McKee (RCM) ordering of Z.
%soln.Zchol = chol(soln.Z(soln.RCMorderOfZ, soln.RCMorderOfZ));
soln.AMDorderOfZ = symamd(soln.Z);  % Approximate Minimum Degree (AMD) ordering of Z.
soln.Zchol = chol(soln.Z(soln.AMDorderOfZ, soln.AMDorderOfZ));

clearvars -global soln.G soln.M soln.Md soln.Z soln.MdOrigInv soln.MdOrigInvTimesG

% End of function step0()




%========================================================================
function step1(iter)
%========================================================================
% Executes step 1 of the method to determine the intermediate velocity.
global NE NENv soln elem Sv NGP GQ time;


% Calculate Ae and assemble into A.
if iter == 1  
  
  nnzM = soln.sparseM.NNZ / 3;
  nnzM2 = 2 * nnzM;
  
  u0_nodal = zeros(NENv,1);
  v0_nodal = zeros(NENv,1);
  w0_nodal = zeros(NENv,1);
  
  soln.sparseA.value = zeros(soln.sparseM.NNZ,1);   % Initialize A matrix to zero.
    
  for e = 1:NE                                                                % TODO: Bu fonksiyonda zamanin cogu bu eleman dongusunde geciyor.
    elem(e).Ae = zeros(3*NENv, 3*NENv);
  
    Ae_11 = zeros(NENv, NENv);

    % Extract elemental u, v and w velocity values from the global solution
    % solution array of the previous iteration.
    for i = 1:NENv
      iG = elem(e).LtoGvel(i);
      u0_nodal(i,1) = soln.Un(iG,1);
  
      iG = elem(e).LtoGvel(i + NENv);
      v0_nodal(i,1) = soln.Un(iG,1);
      
      iG = elem(e).LtoGvel(i + 2*NENv);
      w0_nodal(i,1) = soln.Un(iG,1);
    end

    for k = 1:NGP   % Gauss Quadrature loop
      % Define shortcuts
      gDSv(:,:) = elem(e).gDSv(:,:,k);
      GQfactor = elem(e).detJacob(k) * GQ.weight(k);

      % Above calculated u0 and v0 values are at the nodes. However in GQ
      % integration we need them at GQ points. Let's calculate them using
      % interpolation based on shape functions.
      u0 = 0.0;
      v0 = 0.0;
      w0 = 0.0;
      for i = 1:NENv
        u0 = u0 + Sv(i,k) * u0_nodal(i,1);
        v0 = v0 + Sv(i,k) * v0_nodal(i,1);
        w0 = w0 + Sv(i,k) * w0_nodal(i,1);
      end
    
%       for i = 1:NENv
%         factor = Sv(i,k) * GQfactor;
%         for j = 1:NENv
%            Ae_11(i,j) = Ae_11(i,j) + factor * (u0 * gDSv(1,j) + v0 * gDSv(2,j) + w0 * gDSv(3,j));
%         end
%       end

      for j = 1:NENv
        sum = u0 * gDSv(1,j) + v0 * gDSv(2,j) + w0 * gDSv(3,j);
        for i = 1:NENv
          factor = Sv(i,k) * GQfactor;
          Ae_11(i,j) = Ae_11(i,j) + factor * sum;
        end
      end
      
    end  % End of GQ loop
  

    for j = 1:NENv
      for i = 1:NENv
        soln.sparseA.value(elem(e).sparseMapM(i,j)) = ...   % Assemble upper left sub-matrix of A
        soln.sparseA.value(elem(e).sparseMapM(i,j)) + Ae_11(i,j);   %elem(e).Ae(i,j);
       
         soln.sparseA.value(elem(e).sparseMapM(i,j) + nnzM) = ...   % Assemble middle sub-matrix of A
           soln.sparseA.value(elem(e).sparseMapM(i,j) + nnzM) + Ae_11(i,j);   %elem(e).Ae(i,j);
 
         soln.sparseA.value(elem(e).sparseMapM(i,j) + nnzM2) = ...   % Assemble lower right sub-matrix of A
           soln.sparseA.value(elem(e).sparseMapM(i,j) + nnzM2) + Ae_11(i,j);   %elem(e).Ae(i,j);
      end
    end

%    Following commented lines seems to be more efficient than the above
%    loop, but surprisingly they are not.
%
%    soln.sparseA.value(nnzM+1:nnzM2) = soln.sparseA.value(1:nnzM);
%    soln.sparseA.value(nnzM2+1:nnzM3) = soln.sparseA.value(1:nnzM);

  end  % End of element loop

  soln.A = sparse(soln.sparseM.row, soln.sparseM.col, soln.sparseA.value);

end  % End of iter ==  1 check 


% Calculate the RHS vector of step 1.
soln.R1 = - soln.K * soln.UnpHalf_prev - soln.A * soln.UnpHalf_prev - soln.G * soln.Pn;

% Modify R1 for velocity BCs
applyBC_Step1(2);

% Calculate AccHalf vector.
soln.AccHalf = soln.R1 .* soln.MdInv;

% Calculate UnpHalf
soln.UnpHalf = soln.Un + time.dt * soln.AccHalf;

% End of function step1()




%========================================================================
function step2()
%========================================================================
% Executes step 2 of the method to determine pressure of the new time step.
global soln time;

% Calculate the RHS vector of step 2.
soln.R2 = soln.Gt * (soln.UnpHalf - time.dt^2 * soln.MdOrigInvTimesK *  soln.Acc_prev);       % TODO: To avoid calculating MdOrigInvTimesK, first perform K*Acc_prev
                                                                                              %       followed by MdOrigInv * (K*Acc_prev).
% Apply BCs for step2
applyBC_Step2(2);

% Solve for Pdot using Cholesky factorization obtained in step 0.
%soln.R2 = soln.R2(soln.RCMorderOfZ);
soln.R2 = soln.R2(soln.AMDorderOfZ);
%R23 = R22';
dummy = soln.Zchol' \ soln.R2;
dummy2 = soln.Zchol \ dummy;   % This is Pdot but ordered according to reverse Cuthill Mckee.

% Convert dummy2 to Pdot using reverse cuthill McKee ordering of the Z matrix
%soln.Pdot(soln.RCMorderOfZ) = dummy2;
soln.Pdot(soln.AMDorderOfZ) = dummy2;


% Solve Pdot without using Cholesky factorization
%soln.Pdot = soln.Zsparse \ soln.R2;


% Calculate Pnp1
soln.Pnp1 = soln.Pn + time.dt * soln.Pdot;

% End of function step2()




%========================================================================
function step3()
%========================================================================
% Executes step 3 of the method to determine the velocity of the new time step.
global soln time;

% Calculate the RHS vector of step 3.
soln.R3 = -time.dt * (soln.G * soln.Pdot + soln.K * soln.Acc_prev);

% Modify R3 for velocity BCs
applyBC_Step3();

% Calculate Acc vector.
soln.Acc = soln.R3 .* soln.MdInv;

% Calculate Unp
soln.Unp1 = soln.UnpHalf + time.dt * soln.Acc;

% End of function step3()




%========================================================================
function [] = applyBC_initial()
%========================================================================
% Apply the specified BCs before the solution starts.
global NN soln BC coord;

% Apply velocity BCs
for i = 1:BC.nVelNodes
  node = BC.velNodes(i,1);     % Node at which this velocity BC is specified.
  whichBC = BC.velNodes(i,2);  % Number of the specified BC

  x = coord(node,1);           % May be necessary for BC.str evaluation
  y = coord(node,2);
  z = coord(node,3);
  
  % Change Un with the given u, v and w velocities.
  soln.Un(node)        = eval(char(BC.str(whichBC,1)));
  soln.Un(node + NN)   = eval(char(BC.str(whichBC,2)));
  soln.Un(node + 2*NN) = eval(char(BC.str(whichBC,3)));
end

% End of function applyBC_initial()




%========================================================================
function [] = applyBC_Step1(flag)
%========================================================================
% When flag=1, modify Md for velocity BCs. When flag=2, modify the right
% hand side vector of step 1 (R1) for velocity BCs.

% WARNING : In step 1 velocity differences between 2 iterations is
% calculated. Therefore when specifying velocity BCs a value of zero is
% specified instead of the original velocity value.

global NN soln BC;

if flag == 1
  for i = 1:BC.nVelNodes
    node = BC.velNodes(i,1);   % Node at which this velocity BC specified.
    soln.Md(node)        = 1.0;    % Change [Md] for the given u velocity.
    soln.Md(node + NN)   = 1.0;    % Change [Md] for the given v velocity.
    soln.Md(node + 2*NN) = 1.0;    % Change [Md] for the given w velocity.
  end
  
elseif flag == 2
  for i = 1:BC.nVelNodes
    node = BC.velNodes(i,1);     % Node at which this velocity BC is specified.
  
    % Change R1 for the given u and v velocities.
    soln.R1(node)        = 0.0;  % This is not velocity, but velocity difference between 2 iterations.    % eval(char(BC.str(whichBC,1)));
    soln.R1(node + NN)   = 0.0;  % This is not velocity, but velocity difference between 2 iterations.    % eval(char(BC.str(whichBC,2)));
    soln.R1(node + 2*NN) = 0.0;  % This is not velocity, but velocity difference between 2 iterations.    % eval(char(BC.str(whichBC,3)));
  end
end
% End of function applyBC_Step1()




%========================================================================
function [] = applyBC_Step2(flag)
%========================================================================
% When flag=1, modify Z for pressure BCs. When flag=2, modify the right
% hand side vector of step 2 (R2) for pressure BCs.

% WARNING : In step 2 pressure differences between 2 iterations is
% calculated. Therefore when specifying pressure BCs a value of zero is
% specified instead of the original pressure value.

% In order not to break down the symmetry of [Z], we use the "LARGE number"
% trick.

global soln BC;

LARGE = 1000;                                                                 % TODO: Implement EBCs without the use of LARGE.

if flag == 1
   node = BC.pNode;     % Node at which pressure is set to zero.
  
   if (node > 0) % If node is negative it means we do not set
                 % pressure to zero at any node.
     %soln.Z(node, :) = 0.0;
     %soln.Z(node, node) = 1.0;
     soln.Z(node, node) = soln.Z(node, node) * LARGE;
   end
elseif flag == 2
   node = BC.pNode;     % Node at which pressure is set to zero.
  
   if (node > 0) % If node is negative it means we do not set pressure
                 % to zero at any node.
     soln.R2(node) = 0.0;  % This is not pressure, but pressure difference between 2 iterations.
   end
end
% End of function applyBC_Step2()




%========================================================================
function [] = applyBC_Step3()
%========================================================================
% Modify the right hand side vector of step 3 (R3) for velocity BCs.
global NN soln BC;

for i = 1:BC.nVelNodes
  node = BC.velNodes(i,1);     % Node at which this velocity BC is specified.
  
  % Change R3 for the given u and v velocities.
  soln.R3(node)        = 0.0;  % This is not velocity, but velocity difference between 2 iterations.    % eval(char(BC.str(whichBC,1)));
  soln.R3(node + NN)   = 0.0;  % This is not velocity, but velocity difference between 2 iterations.    % eval(char(BC.str(whichBC,2)));
  soln.R3(node + 2*NN) = 0.0;  % This is not velocity, but velocity difference between 2 iterations.    % eval(char(BC.str(whichBC,3)));
end
% End of function applyBC_Step3()




%========================================================================
function [] = readRestartFile()
%========================================================================
global prName NE NN NCN NENv NENp soln elem;

% This function is called if time.isRestart is equal to one. It reads the
% restart file.

% Open the restart file
restartFile = fopen(strcat(prName,'_restart.dat'), 'r');

% Read three dumy lines at the beginning of the file
fgets(restartFile);  % Read the next dummy line.
fgets(restartFile);  % Read the next dummy line.
fgets(restartFile);  % Read the next dummy line.

% Read velocity components and pressure at each node.
for i = 1:NCN
  x = fscanf(restartFile, '%f', 1);
  y = fscanf(restartFile, '%f', 1);
  z = fscanf(restartFile, '%f', 1);
  u = fscanf(restartFile, '%f', 1);
  v = fscanf(restartFile, '%f', 1);
  w = fscanf(restartFile, '%f', 1);
  p = fscanf(restartFile, '%f', 1);
  fgets(restartFile);

  soln.Un(i) = u;
  soln.Un(i+NN) = v;
  soln.Un(i+2*NN) = w;
  if NENp ~= 1
     soln.Pn(i) = p;
  else
     pNode(i) = p;
  end
end

for i = NCN+1:NN
  x = fscanf(restartFile, '%f', 1);
  y = fscanf(restartFile, '%f', 1);
  z = fscanf(restartFile, '%f', 1);
  u = fscanf(restartFile, '%f', 1);
  v = fscanf(restartFile, '%f', 1);
  w = fscanf(restartFile, '%f', 1);
  p = fscanf(restartFile, '%f', 1);
  fgets(restartFile);

  soln.Un(i) = u;
  soln.Un(i+NN) = v;
  soln.Un(i+2*NN) = w;
end

fclose(restartFile);

% for NENp=1 case, convert nodal pressures to elemental pressures
if NENp == 1
  for e = 1:NE
    for i = 1:NENv
      node = elem(e).LtoGnode(i);
      soln.Pn(e) = soln.Pn(e) + pNode(node);
    end
    soln.Pn(e) = soln.Pn(e) / NCN;
  end
end

% End of function readRestartFile()




%========================================================================
function [] = postProcess()
%========================================================================
% Generates contour plot of u, v, and pressure and streamline plot. Only
% the data at element corners are used even if there are velocity nodes at
% mid faces.
global NE NENv NENp coord elem soln

% TODO: This is not converted to 3D

% Generate necessary data for MATLAB's patch function that'll be used to
% generate contour plots.
for e = 1:NE
  for i = 1:NENp
    x(i,e) = coord(elem(e).LtoGnode(i),1);
    y(i,e) = coord(elem(e).LtoGnode(i),2);
    
    uVel(i,e) = soln.Un(elem(e).LtoGdof(i));
    vVel(i,e) = soln.Un(elem(e).LtoGdof(i + NENv));
    pressure(i,e) = soln.Un(elem(e).LtoGdof(i + 2*NENv));
  end
end

subplot(2,2,1);
patch(x, y, uVel);
axis equal;
colorbar;    % Put a colorbar next to the graph
xlabel('x');
ylabel('y');
zlabel('u');
title('Contour plot of u velocity');

subplot(2,2,2);
patch(x, y, vVel);
axis equal;
colorbar;    % Put a colorbar next to the graph
xlabel('x');
ylabel('y');
zlabel('v');
title('Contour plot of v velocity');

subplot(2,2,3);
patch(x, y, pressure);
axis equal;
colorbar;    % Put a colorbar next to the graph
xlabel('x');
ylabel('y');
zlabel('p');
title('Contour plot of pressure');

% End of function postProcess()




%========================================================================
function [] = createTecplot()
%========================================================================
global eType NE NN NNp NEC NEE NEF NENv NENp coord elem soln prName;

% This file is used if NENp and NENv are different. In this case each
% hexahedral element is divided into eight sub-elements and Tecplot file is
% created as if there are 8*NE hexahedral elements. Missing mid-edge,
% mid-face and mid-element pressure values are evaluated by linear
% interpolation.

% Call the simple version of this function if NENp and NENv are the same.
if NENp == NENv
  createTecplotSimple();
  return;
elseif NENp == 1
  % ...
  % ...
end

% Write the calculated unknowns to a Tecplot file
outputFile = fopen(strcat(prName, '_tecplot.dat'), 'w');

fprintf(outputFile, '%s %s \n', 'TITLE =', prName);
fprintf(outputFile, '%s \n', 'VARIABLES = x, y, z, u, v, w, p');
if (eType == 1)
  fprintf(outputFile, '%s %i %s %i %s \n', 'ZONE N=', NN, ' E=', 8*NE, ', F=FEPOINT , ET=BRICK');
  
  % New Tecplot 360 documentation has the following format but the above seems to be working also
  % ZONE NODES=..., ELEMENTS=..., DATAPACKING=POINT, ZONETYPE=FEBRICK

else
  fprintf(outputFile, '%s %i %s %i %s \n', 'ZONE N=', NN, ', E=', 8*NE, ', F=FEPOINT , ET=TRIANGLE');
end

% Seperate soln.Un into uNode, vNode and wNode variables
uNode = zeros(NN,1);
vNode = zeros(NN,1);
wNode = zeros(NN,1);

for i = 1:NN
  uNode(i) = soln.Un(i);
  vNode(i) = soln.Un(i+NN);
  wNode(i) = soln.Un(i+2*NN);
end

% Copy pressure solution into pNode array, but the size of pNode is NN,
% because it will also store pressure values at mid-egde, mid-face and
% mid-element nodes.
pNode = zeros(NN,1);

for i = 1:NNp
  pNode(i) = soln.Pn(i);
end


% Interpolate pressure at non-corner nodes.
for e = 1:NE
  % Calculate mid-edge pressures as averages of the corner pressures.
  for ed = 1:NEE
  
    % Determine corner nodes of edge ed
    if eType == 1   % Hexahedral element
      switch ed
        case 1
          n1 = elem(e).LtoGnode(1);
          n2 = elem(e).LtoGnode(2);
        case 2
          n1 = elem(e).LtoGnode(2);
          n2 = elem(e).LtoGnode(3);
        case 3
          n1 = elem(e).LtoGnode(3);
          n2 = elem(e).LtoGnode(4);
        case 4
          n1 = elem(e).LtoGnode(4);
          n2 = elem(e).LtoGnode(1);
        case 5
          n1 = elem(e).LtoGnode(1);
          n2 = elem(e).LtoGnode(5);
        case 6
          n1 = elem(e).LtoGnode(2);
          n2 = elem(e).LtoGnode(6);
        case 7
          n1 = elem(e).LtoGnode(3);
          n2 = elem(e).LtoGnode(7);
        case 8
          n1 = elem(e).LtoGnode(4);
          n2 = elem(e).LtoGnode(8);
        case 9
          n1 = elem(e).LtoGnode(5);
          n2 = elem(e).LtoGnode(6);
        case 10
          n1 = elem(e).LtoGnode(6);
          n2 = elem(e).LtoGnode(7);
        case 11
          n1 = elem(e).LtoGnode(7);
          n2 = elem(e).LtoGnode(8);
        case 12
          n1 = elem(e).LtoGnode(8);
          n2 = elem(e).LtoGnode(5);
      end
    elseif eType == 2   % Tetrahedral element
      fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function setupMidFaceNodes() yet!!!\n\n\n');
    end
    
    node = elem(e).LtoGnode(ed+NEC);

    pNode(node) = 0.5 * (pNode(n1) + pNode(n2));

  end  % End of ed (edge) loop
  
  
  % Calculate mid-face pressures as averages of the corner pressures.
  for f = 1:NEF
  
    % Determine corner nodes of face f
    if eType == 1   % Hexahedral element
      NFC = 4;   % Number of face corners
      switch f
        case 1
          n1 = elem(e).LtoGnode(1);
          n2 = elem(e).LtoGnode(2);
          n3 = elem(e).LtoGnode(3);
          n4 = elem(e).LtoGnode(4);
        case 2
          n1 = elem(e).LtoGnode(1);
          n2 = elem(e).LtoGnode(2);
          n3 = elem(e).LtoGnode(5);
          n4 = elem(e).LtoGnode(6);
        case 3
          n1 = elem(e).LtoGnode(2);
          n2 = elem(e).LtoGnode(3);
          n3 = elem(e).LtoGnode(6);
          n4 = elem(e).LtoGnode(7);
        case 4
          n1 = elem(e).LtoGnode(3);
          n2 = elem(e).LtoGnode(4);
          n3 = elem(e).LtoGnode(7);
          n4 = elem(e).LtoGnode(8);
        case 5
          n1 = elem(e).LtoGnode(1);
          n2 = elem(e).LtoGnode(4);
          n3 = elem(e).LtoGnode(5);
          n4 = elem(e).LtoGnode(8);
        case 6
          n1 = elem(e).LtoGnode(5);
          n2 = elem(e).LtoGnode(6);
          n3 = elem(e).LtoGnode(7);
          n4 = elem(e).LtoGnode(8);
      end
      
      node = elem(e).LtoGnode(f+NEC+NEE);
      
      pNode(node) = 0.25 * (pNode(n1) + pNode(n2) + pNode(n3) + pNode(n4));

    elseif eType == 2   % Tetrahedral element
      fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function setupMidFaceNodes() yet!!!\n\n\n');
    end

  end  % End of f (face) loop


  % Find add the mid-element node pressures.
  if eType == 1   % Hexahedral element
    n1 = elem(e).LtoGnode(1);
    n2 = elem(e).LtoGnode(2);
    n3 = elem(e).LtoGnode(3);
    n4 = elem(e).LtoGnode(4);
    n5 = elem(e).LtoGnode(5);
    n6 = elem(e).LtoGnode(6);
    n7 = elem(e).LtoGnode(7);
    n8 = elem(e).LtoGnode(8);
    
    node = elem(e).LtoGnode(1+NEC+NEE+NEF);
    
    pNode(node) = 0.125 * (pNode(n1) + pNode(n2) + pNode(n3) + pNode(n4) + pNode(n5) + pNode(n6) + pNode(n7) + pNode(n8));

  elseif eType == 2   % Tetrahedral element
    fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function setupMidFaceNodes() yet!!!\n\n\n');
  end

end  % End of element loop




% Print the coordinates and the calculated velocity and pressure values
for i = 1:NN
   x = coord(i,1);
   y = coord(i,2);
   z = coord(i,3);
   fprintf(outputFile, '%15.11f %15.11f %15.11f %15.11f %15.11f %15.11f %15.11f \n', ...
           x, y, z, uNode(i), vNode(i), wNode(i), pNode(i));
end


% Print the connectivity list. We will divide hexahedral elements into 8
% and divide tetrahedral elements into ... elements.                               TODO
if eType == 1  % Hexahedral elements
  for e = 1:NE
    % 1st sub-element of element e
    fprintf(outputFile, '%6i %6i %6i %6i %6i %6i %6i %6i \n', elem(e).LtoGnode(1), elem(e).LtoGnode(9), elem(e).LtoGnode(21), elem(e).LtoGnode(12), elem(e).LtoGnode(13), elem(e).LtoGnode(22), elem(e).LtoGnode(27), elem(e).LtoGnode(25));
    % 2nd sub-element of element e
    fprintf(outputFile, '%6i %6i %6i %6i %6i %6i %6i %6i \n', elem(e).LtoGnode(9), elem(e).LtoGnode(2), elem(e).LtoGnode(10), elem(e).LtoGnode(21), elem(e).LtoGnode(22), elem(e).LtoGnode(14), elem(e).LtoGnode(23), elem(e).LtoGnode(27));
    % 3rd sub-element of element e
    fprintf(outputFile, '%6i %6i %6i %6i %6i %6i %6i %6i \n', elem(e).LtoGnode(12), elem(e).LtoGnode(21), elem(e).LtoGnode(11), elem(e).LtoGnode(4), elem(e).LtoGnode(25), elem(e).LtoGnode(27), elem(e).LtoGnode(24), elem(e).LtoGnode(16));
    % 4th sub-element of element e
    fprintf(outputFile, '%6i %6i %6i %6i %6i %6i %6i %6i \n', elem(e).LtoGnode(21), elem(e).LtoGnode(10), elem(e).LtoGnode(3), elem(e).LtoGnode(11), elem(e).LtoGnode(27), elem(e).LtoGnode(23), elem(e).LtoGnode(15), elem(e).LtoGnode(24));
    % 5th sub-element of element e
    fprintf(outputFile, '%6i %6i %6i %6i %6i %6i %6i %6i \n', elem(e).LtoGnode(13), elem(e).LtoGnode(22), elem(e).LtoGnode(27), elem(e).LtoGnode(25), elem(e).LtoGnode(5), elem(e).LtoGnode(17), elem(e).LtoGnode(26), elem(e).LtoGnode(20));
    % 6th sub-element of element e
    fprintf(outputFile, '%6i %6i %6i %6i %6i %6i %6i %6i \n', elem(e).LtoGnode(22), elem(e).LtoGnode(14), elem(e).LtoGnode(23), elem(e).LtoGnode(27), elem(e).LtoGnode(17), elem(e).LtoGnode(6), elem(e).LtoGnode(18), elem(e).LtoGnode(26));
    % 7th sub-element of element e
    fprintf(outputFile, '%6i %6i %6i %6i %6i %6i %6i %6i \n', elem(e).LtoGnode(25), elem(e).LtoGnode(27), elem(e).LtoGnode(24), elem(e).LtoGnode(16), elem(e).LtoGnode(20), elem(e).LtoGnode(26), elem(e).LtoGnode(19), elem(e).LtoGnode(8));
    % 8th sub-element of element e
    fprintf(outputFile, '%6i %6i %6i %6i %6i %6i %6i %6i \n', elem(e).LtoGnode(27), elem(e).LtoGnode(23), elem(e).LtoGnode(15), elem(e).LtoGnode(24), elem(e).LtoGnode(26), elem(e).LtoGnode(18), elem(e).LtoGnode(7), elem(e).LtoGnode(19));
  end
  
elseif eType == 2  % Tetrahedral elements
  
  % TODO : ...  
  
end

fclose(outputFile);

% End of function createTecplot()




%========================================================================
function [] = createTecplotSimple()
%========================================================================
global NE NN NEC NENv NENp eType coord elem soln prName;

% TODO: This is not converted to 3D yet.

% This function is executed if NENp = NENv

% Write the calculated unknowns to a Tecplot file
outputFile = fopen(strcat(prName, '_tecplot.dat'), 'w');


% Store the unknowns in a way that can be used to generate the Tecplot file
for e = 1:NE
  % Extract elemental u, v, p values for element e.
  for i = 1:NENv
    iG = elem(e).LtoGvel(i);
    ue(i) = soln.Un(iG);
  end
  for i = 1:NENv
    ii = i + NENv;
    iG = elem(e).LtoGvel(ii);
    ve(i) = soln.Un(iG);
  end
  for i = 1:NENv
    ii = i + 2*NENv;
    iG = elem(e).LtoGvel(ii);
    we(i) = soln.Un(iG);
  end
  for i = 1:NENp
    iG = elem(e).LtoGpres(i);
    pe(i) = soln.Pn(iG);
  end

  % Write these elemental values into uNode, vNode, wNode, pNode arrays
  % Note that velocities are also written at element corners only.
  for i = 1:NEC
    node = elem(e).LtoGnode(i);
    uNode(node) = ue(i);
    vNode(node) = ve(i);
    wNode(node) = we(i);
    pNode(node) = pe(i);
  end
end



fprintf(outputFile, '%s %s \n', 'TITLE =', prName);
fprintf(outputFile, '%s \n', 'VARIABLES = x, y, z, u, v, w, p');
if (eType == 1)
   fprintf(outputFile, '%s %i %s %i %s \n', 'ZONE N=', NN, ', E=', NE, ...
            ', F=FEPOINT , ET=BRICK');
else
   fprintf(outputFile, '%s %i %s %i %s \n', 'ZONE N=', NN, ', E=', NE, ...
           ', F=FEPOINT , ET=TRIANGLE');
end

% Print the coordinates and the calculated velocity and pressure values
for i = 1:NN
   x = coord(i,1);
   y = coord(i,2);
   z = coord(i,3);
   fprintf(outputFile, '%15.11f %15.11f %15.11f %15.11f %15.11f %15.11f %15.11f \n', ...
           x, y, z, uNode(i), vNode(i), wNode(i), pNode(i));
end

% Print the connectivity list
for e = 1:NE
   for i = 1:NEC
      fprintf(outputFile, '%5i', elem(e).LtoGnode(i));
   end
   fprintf(outputFile, '\n');
end

fclose(outputFile);

% End of function createTecplotSimple()




%========================================================================
function [] = createTecplotNENp1()
%========================================================================
% Used when NENp=1. Pressure is stored at element center and corner
% pressures are obtained by averaging.

global NE NN NEC NENv NENp eType coord elem soln prName;



% TODO : This function is not converted to 3D yet



% Obtain GtoL mapping;
nElemSurrNode = zeros(NN,1);   % Number of elements surrounding a node.
GtoL = zeros(NN,20);           % 20 is a large enough number. Stores the elements
                               % that are connected to a node.
for e = 1:NE
  for i = 1:NENv
    node = elem(e).LtoGnode(i);
    if ~any(e == GtoL(node,:))   % Check whether e is already in GtoL list or not
      nElemSurrNode(node) = nElemSurrNode(node) + 1;
      GtoL(node,nElemSurrNode(node)) = e;
    end
  end
end



% By averaging the pressures stored at cell centers (soln.P) generate pNode
% for pressures at velocity nodes.
pNode = zeros(NN,1);
for n = 1:NN
  for i = 1:nElemSurrNode(n)
    pNode(n) = pNode(n) + soln.Pn(i);
  end
  pNode(n) = pNode(n) / nElemSurrNode(n);
end



% Store the unknowns in a way that can be used to generate the Tecplot file
for e = 1:NE
  % Extract elemental u, v, p values for element e.
  for i = 1:NENv
    iG = elem(e).LtoGvel(i);
    ue(i) = soln.Un(iG);
  end
  for i = 1:NENv
    ii = i + NENv;
    iG = elem(e).LtoGvel(ii);
    ve(i) = soln.Un(iG);
  end

  % Write these elemental values into uNode, vNode arrays
  % Note that velocities are also written at element corners only.
  for i = 1:NEC
    node = elem(e).LtoGnode(i);
    uNode(node) = ue(i);
    vNode(node) = ve(i);
  end
end



% Write the calculated unknowns to a Tecplot file
outputFile = fopen(strcat(prName, '_tecplot.dat'), 'w');

fprintf(outputFile, '%s %s \n', 'TITLE =', prName);
fprintf(outputFile, '%s \n', 'VARIABLES = x, y, z, u, v, p');
if (eType == 1)
   fprintf(outputFile, '%s %i %s %i %s \n', 'ZONE N=', NN, ', E=', NE, ...
            ', F=FEPOINT , ET=BRICK');
else
   fprintf(outputFile, '%s %i %s %i %s \n', 'ZONE N=', NN, ', E=', NE, ...
           ', F=FEPOINT , ET=TRIANGLE');
end

% Print the coordinates and the calculated velocity and pressure values
for i = 1:NN
   x = coord(i,1);
   y = coord(i,2);
   fprintf(outputFile, '%15.11f %15.11f %15.11f %15.11f %15.11f \n', ...
           x, y, uNode(i), vNode(i), pNode(i));
end

% Print the connectivity list
for e = 1:NE
   for i = 1:NEC
      fprintf(outputFile, '%5i', elem(e).LtoGnode(i));
   end
   fprintf(outputFile, '\n');
end

fclose(outputFile);

% End of function createTecplotNENp1()




%========================================================================
function [] = updateProfilePlots(flag)
%========================================================================
global NN NCN profilePlot1 soln coord pP;      % pP stand for profilePlots

atCoord = [1 2];          % Profile is taken at these coordinate (1:x, 2:y, 3:z)
alongCoord = 3;           % Profile is taken along this coordinate (1:x, 2:y, 3:z)
coordinate = [0.5 0.5];   % Profile is taken on a line passing through these coordinates
whichVariable = 1;        % 1: u velocity, 2: v velocity, 3: w velocity, 4: pressure
unknownLimits = [-0.2,1.2];   % Set these limits so that the limits of the unknown
                              % axis of the plot does not chage continuously.

SMALL = 1e-10;         % Value used for coordinate equality check

if flag == 1   % Create profile plot figures for the first time
  % Determine the nodes across the specified line (x=0.5 for the Cavity problem).
  pP.nodeCount = 0;
  
  if whichVariable == 4
    upperNodeLimit = NCN;  % For pressure maximum node value is NCN
  else
    upperNodeLimit = NN;   % For velocities maximum node value is NN
  end
  
  for i = 1:upperNodeLimit
    if abs(coord(i,atCoord(1)) - coordinate(1)) < SMALL && ...
       abs(coord(i,atCoord(2)) - coordinate(2)) < SMALL
      pP.nodeCount = pP.nodeCount + 1;
      pP.nodes(pP.nodeCount,1) = i;
    end
  end

  % We need to sort these nodes in increasing x or y or z.
  for i = 2:pP.nodeCount
    n1 = pP.nodes(i);
    for j = 1:i-1    % Compare the coordinates of node n1 to all nodes before it
      n2 = pP.nodes(j);
      if coord(n1,alongCoord) < coord(n2,alongCoord)
        for k = i-1:-1:j
          pP.nodes(k+1) = pP.nodes(k);  % Shifts nodes from j to i-1 down 1 place.
        end
        pP.nodes(j) = n1;  % Shift node number i up, in place of node number j.
        break;
      end
    end
  end
    
  figure;
  profilePlot1 = plot(zeros(pP.nodeCount,1), zeros(pP.nodeCount,1));
  grid;
  
else   % Update profile plot figures during the unsteady solution
  
  % Generate variables to plot plotVar1 vs plotVar2.
  plotVar1 = zeros(pP.nodeCount,1);
  plotVar2 = zeros(pP.nodeCount,1);

  for i = 1:pP.nodeCount
    node = pP.nodes(i);
    plotVar2(i) = coord(node,alongCoord);
    if whichVariable == 1
      plotVar1(i) = soln.Un(node);       % u velocity values
    elseif  whichVariable == 2
      plotVar1(i) = soln.Un(node+NN);    % v velocity values
    elseif  whichVariable == 2
      plotVar1(i) = soln.Un(node+2*NN);  % w velocity values
    else
      plotVar1(i) = soln.Pn(node);       % pressure values
    end
  end

  set(profilePlot1,'xdata',plotVar2);
  set(profilePlot1,'ydata',plotVar1);
  
  if alongCoord == 1  % If unknowns are extracted along x direction plot plotVar1 vs plotVar2.
    set(profilePlot1,'xdata',plotVar2);
    set(profilePlot1,'ydata',plotVar1);
    ylim([unknownLimits(1) unknownLimits(2)]);
  else                % If unknowns are extracted along y or z directions plot plotVar2 vs plotVar1.
    set(profilePlot1,'xdata',plotVar1);
    set(profilePlot1,'ydata',plotVar2);
    xlim([unknownLimits(1) unknownLimits(2)]);
  end
  
  drawnow;
  
  % TODO : Set axis names
  
end

% End of function updateProfilePlot()








%{
%========================================================================
function [] = calcElemSize
%========================================================================
global eType NE coord elem;

% Calculates the diameter of the circumcircle around the triangle or the
% quadrilateral. It is used for GLS stabilization.

if eType == 1       % Quadrilateral
  for e = 1:NE
    % Get the x and y coordinates of the corners of the quadrilateral
    p1 = [coord(elem(e).LtoGnode(1),1) coord(elem(e).LtoGnode(1),2)];
    p2 = [coord(elem(e).LtoGnode(2),1) coord(elem(e).LtoGnode(2),2)];
    p3 = [coord(elem(e).LtoGnode(3),1) coord(elem(e).LtoGnode(3),2)];
    p4 = [coord(elem(e).LtoGnode(4),1) coord(elem(e).LtoGnode(4),2)];
    
    % Calculate the lengths of the edges of the quadrilateral
    a = sqrt((p1(1)-p2(1))^2 + (p1(2)-p2(2))^2);
    b = sqrt((p2(1)-p3(1))^2 + (p2(2)-p3(2))^2);
    c = sqrt((p3(1)-p4(1))^2 + (p3(2)-p4(2))^2);
    d = sqrt((p4(1)-p1(1))^2 + (p4(2)-p1(2))^2);

    % Calculate the lengths of the diagonals of the quadrilateral
    f = sqrt((p1(1)-p3(1))^2 + (p1(2)-p3(2))^2);
    g = sqrt((p2(1)-p4(1))^2 + (p2(2)-p4(2))^2);
    
    % Calculate the approximate length of the element by taking the largest
    % of a, b, c, d, f, g.
    elem(e).he = max([a b c d f g]);
  end
  
elseif eType == 2   % Triangle
  for e = 1:NE
    % Get the x and y coordinates of the corners of the triangle
    p1 = [coord(elem(e).LtoGnode(1),1) coord(elem(e).LtoGnode(1),2)];
    p2 = [coord(elem(e).LtoGnode(2),1) coord(elem(e).LtoGnode(2),2)];
    p3 = [coord(elem(e).LtoGnode(3),1) coord(elem(e).LtoGnode(3),2)];
    
    % Calculate the lengths of the edges of the triangle
    a = sqrt((p1(1)-p2(1))^2 + (p1(2)-p2(2))^2);
    b = sqrt((p2(1)-p3(1))^2 + (p2(2)-p3(2))^2);
    c = sqrt((p3(1)-p1(1))^2 + (p3(2)-p1(2))^2);
    
    % Calculate the diameter of the circumcircle
    elem(e).he = 2 * a * b * c / sqrt((a+b+c)*(-a+b+c)*(a-b+c)*(a+b-c));
  end
end

% End of function calcElemSize()
%}




% %========================================================================
% function [] = findFaceNeighbors
% %========================================================================
% global NE NEF NENv NENp elem elemEdgeNeighbors eType;
% % Find face neighbors of elements.
% 
% 
% if NENv == NENp   % Neighboring information is NOT necessary if NENv == NENp
%   return;
% end
% 
% 
% % Allocate space for neihboring element/face of each element/face.
% elemFaceNeighbors = zeros(NE,NEF,2);  % The last index means 1: Neighbor element, 2: Neighbor face
% elemFaceNeighbors(:,:,:) = -1;  % Initialize all edge neighbors of all elements to -1
% 
% 
% % Determine face neighbors for each face of each element
% for e = 1:NE
%   for f = 1:NEF
%     if elemFaceNeighbors(e,f,1) ~= -1
%       continue
%     end
%     
%     matchFound = 0;  % Boolean to check whether a neighboring face is found or not
%     
%     % Determine corner nodes of face f
%     if eType == 1   % Hexahedral element
%       switch f
%         case 1
%           n1 = elem(e).LtoGnode(1);
%           n2 = elem(e).LtoGnode(2);
%           n3 = elem(e).LtoGnode(3);
%           n4 = elem(e).LtoGnode(4);
%         case 2
%           n1 = elem(e).LtoGnode(1);
%           n2 = elem(e).LtoGnode(2);
%           n3 = elem(e).LtoGnode(6);
%           n4 = elem(e).LtoGnode(5);
%         case 3
%           n1 = elem(e).LtoGnode(2);
%           n2 = elem(e).LtoGnode(3);
%           n3 = elem(e).LtoGnode(7);
%           n4 = elem(e).LtoGnode(6);
%         case 4
%           n1 = elem(e).LtoGnode(4);
%           n2 = elem(e).LtoGnode(3);
%           n3 = elem(e).LtoGnode(7);
%           n4 = elem(e).LtoGnode(8);
%         case 5
%           n1 = elem(e).LtoGnode(1);
%           n2 = elem(e).LtoGnode(4);
%           n3 = elem(e).LtoGnode(8);
%           n4 = elem(e).LtoGnode(5);
%         case 6
%           n1 = elem(e).LtoGnode(5);
%           n2 = elem(e).LtoGnode(6);
%           n3 = elem(e).LtoGnode(7);
%           n4 = elem(e).LtoGnode(8);
%       end
%     elseif eType == 2   % Tetrahedral element
%       fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function findElemNeighbors() yet!!!\n\n\n');
%     end
%     
%     % Search through all other elements to see if there is any with a face
%     % with corner nodes n1, n2, n3 and n4.
%     
%     for e2 = 1:NE
%       for f2 = 1:NEF
%         % Determine corner nodes of face f
%         if eType == 1   % Hexahedral element
%           switch f2
%             case 1
%               n12 = elem(e2).LtoGnode(1);
%               n22 = elem(e2).LtoGnode(2);
%               n32 = elem(e2).LtoGnode(3);
%               n42 = elem(e2).LtoGnode(4);
%             case 2
%               n12 = elem(e2).LtoGnode(1);
%               n22 = elem(e2).LtoGnode(2);
%               n32 = elem(e2).LtoGnode(6);
%               n42 = elem(e2).LtoGnode(5);
%             case 3
%               n12 = elem(e2).LtoGnode(2);
%               n22 = elem(e2).LtoGnode(3);
%               n32 = elem(e2).LtoGnode(7);
%               n42 = elem(e2).LtoGnode(6);
%             case 4
%               n12 = elem(e2).LtoGnode(4);
%               n22 = elem(e2).LtoGnode(3);
%               n32 = elem(e2).LtoGnode(7);
%               n42 = elem(e2).LtoGnode(8);
%             case 5
%               n12 = elem(e2).LtoGnode(1);
%               n22 = elem(e2).LtoGnode(4);
%               n32 = elem(e2).LtoGnode(8);
%               n42 = elem(e2).LtoGnode(5);
%             case 6
%               n12 = elem(e2).LtoGnode(5);
%               n22 = elem(e2).LtoGnode(6);
%               n32 = elem(e2).LtoGnode(7);
%               n42 = elem(e2).LtoGnode(8);
%           end
%         elseif eType == 2   % Tetrahedral element
%           fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function findElemNeighbors() yet!!!\n\n\n');
%         end
%         
%         if n1 == n12 || n1 == n22 || n1 == n32 || n1 == n42
%           if n2 == n12 || n2 == n22 || n2 == n32 || n2 == n42
%             if n3 == n12 || n3 == n22 || n3 == n32 || n3 == n42
%               if n4 == n12 || n4 == n22 || n4 == n32 || n4 == n42
%                 if e2 ~= e
%                   elemFaceNeighbors(e,f,1) = e2;      % Found a matching face
%                   elemFaceNeighbors(e,f,2) = f2;
%                   matchFound = 1;
%                   elemFaceNeighbors(e2,f2,1) = e;  % Also set (e,f) pair as a neighbor of (e2,f2)
%                   elemFaceNeighbors(e2,f2,2) = f;           
%                 end
%               end
%             end
%           end
%         end
%         
%         if matchFound
%           break
%         end
%       end   % End of f2 loop
% 
%       if matchFound
%         break
%       end
%     end   % End of e2 loop
%   end   % End of f loop
% end   % End of e loop
% 
% % End of function findFaceNeighbors()
% 
% 
% 
% 
% %========================================================================
% function [] = findEdgeNeighbors
% %========================================================================
% global NE NEF NEE NENv NENp elem elemFaceNeighbors elemEdgeNeighbors eType;
% % Find edge neighbors of elements.
% 
% LARGE = 10;   % Possible maximum edge neighbor.
% 
% if NENv == NENp   % Neighboring information is NOT necessary if NENv == NENp
%   return;
% end
% 
% 
% % Allocate space for neihboring element/edge of each element/edge. Number
% % of edge neighbors is not a fixed number for each element, therefore we
% % also need to store the number of face neighbors for each face of each
% % element.
% NelemEdgeNeighbors = zeros(NE,NEE);
% elemEdgeNeighbors = zeros(NE,NEE,LARGE,2);  % LARGE is a possible maximum edge neighbor
% elemEdgeNeighbors(:,:,:,:) = -1;  % Initialize all edge neighbors of all elements to -1
% 
% 
% % Determine edge neighbors for each edge of each element.
% for e = 1:NE
%   for ed = 1:NEE
%     %if elemEdgeNeighbors(e,ed,1) ~= -1   % Neighbors of this edge were already found.
%     %  continue
%     %end
%     
%     % First determine the face neighbors of face neighbors of this element
%     NfaceFaceNeighbors = 0;
%     faceFaceNeighbors = zeros(NEF*NEF);   % This is actually oversized. Actual number is stored by the previous variable. 
%     faceFaceNeighbors(:,:) = -1;
%     
%     for fN = 1:NEF
%       
%       faceNeighbor
%     
%     
%     
%     
%     matchFound = 0;  % Boolean to check whether a neighboring edge is found or not
%     
%     % Determine corner nodes of edge ed
%     if eType == 1   % Hexahedral element
%       switch ed
%         case 1
%           n1 = elem(e).LtoGnode(1);
%           n2 = elem(e).LtoGnode(2);
%         case 2
%           n1 = elem(e).LtoGnode(2);
%           n2 = elem(e).LtoGnode(3);
%         case 3
%           n1 = elem(e).LtoGnode(3);
%           n2 = elem(e).LtoGnode(4);
%         case 4
%           n1 = elem(e).LtoGnode(4);
%           n2 = elem(e).LtoGnode(1);
%         case 5
%           n1 = elem(e).LtoGnode(1);
%           n2 = elem(e).LtoGnode(5);
%         case 6
%           n1 = elem(e).LtoGnode(2);
%           n2 = elem(e).LtoGnode(6);
%         case 7
%           n1 = elem(e).LtoGnode(3);
%           n2 = elem(e).LtoGnode(7);
%         case 8
%           n1 = elem(e).LtoGnode(4);
%           n2 = elem(e).LtoGnode(8);
%         case 9
%           n1 = elem(e).LtoGnode(5);
%           n2 = elem(e).LtoGnode(6);
%         case 10
%           n1 = elem(e).LtoGnode(6);
%           n2 = elem(e).LtoGnode(7);
%         case 11
%           n1 = elem(e).LtoGnode(7);
%           n2 = elem(e).LtoGnode(8);
%         case 12
%           n1 = elem(e).LtoGnode(8);
%           n2 = elem(e).LtoGnode(5);
%       end
%     elseif eType == 2   % Tetrahedral element
%       fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function findElemNeighbors() yet!!!\n\n\n');
%     end
%     
%     % Search through all other elements to see if there is any with an edge
%     % with corner nodes n1 and n2.
%     
%     for e2 = 1:NE
%       for ed2 = 1:NEE
%         % Determine corner nodes of edge ed2
%         if eType == 1   % Hexahedral element
%           switch ed2
%             case 1
%               n12 = elem(e2).LtoGnode(1);
%               n22 = elem(e2).LtoGnode(2);
%             case 2
%               n12 = elem(e2).LtoGnode(2);
%               n22 = elem(e2).LtoGnode(3);
%             case 3
%               n12 = elem(e2).LtoGnode(3);
%               n22 = elem(e2).LtoGnode(4);
%             case 4
%               n12 = elem(e2).LtoGnode(4);
%               n22 = elem(e2).LtoGnode(1);
%             case 5
%               n12 = elem(e2).LtoGnode(1);
%               n22 = elem(e2).LtoGnode(5);
%             case 6
%               n12 = elem(e2).LtoGnode(2);
%               n22 = elem(e2).LtoGnode(6);
%             case 7
%               n12 = elem(e2).LtoGnode(3);
%               n22 = elem(e2).LtoGnode(7);
%             case 8
%               n12 = elem(e2).LtoGnode(4);
%               n22 = elem(e2).LtoGnode(8);
%             case 9
%               n12 = elem(e2).LtoGnode(5);
%               n22 = elem(e2).LtoGnode(6);
%             case 10
%               n12 = elem(e2).LtoGnode(6);
%               n22 = elem(e2).LtoGnode(7);
%             case 11
%               n12 = elem(e2).LtoGnode(7);
%               n22 = elem(e2).LtoGnode(8);
%             case 12
%               n12 = elem(e2).LtoGnode(8);
%               n22 = elem(e2).LtoGnode(5);
%           end
%         elseif eType == 2   % Tetrahedral element
%           fprintf('\n\n\nERROR: Tetrahedral elements are not implemented in function findElemNeighbors() yet!!!\n\n\n');
%         end
%         
%         if n1 == n12 || n1 == n22
%           if n2 == n12 || n2 == n22
%             if e2 ~= e
%               elemEdgeNeighbors(e,ed,1) = e2;      % Found a matching edge
%               elemEdgeNeighbors(e,ed,2) = ed2;
%               matchFound = 1;
%               elemEdgeNeighbors(e2,ed2,1) = e;  % Also set (e,ed) pair as a neighbor of (e2,ed2)
%               elemEdgeNeighbors(e2,ed2,2) = ed;           
%             end
%           end
%         end
%         
%         if matchFound
%           break
%         end
%       end  % End of f2 loop
% 
%       if matchFound
%         break
%       end
%     end  % End of e2  loop
%   end
% end
% 
% % End of function findEdgeNeighbors()
