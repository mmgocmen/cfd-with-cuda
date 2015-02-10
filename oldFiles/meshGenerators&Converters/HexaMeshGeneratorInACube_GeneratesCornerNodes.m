%**********************************************************************
%                                                                     *
% 3D hexa mesh generator on a cubical domain of size [0,1]x[0,1]x[0,1]*
% It can generate meshes with 8 velocity nodes and 27 velocity        *
% nodes. Only the corner nodes are written, others will be calculated *
% inside the solver.                                                  *
%                                                                     *
% Author: Dr. Cuneyt Sert                                             *
%         http://www.metu.edu.tr/~csert                               *
%                                                                     *
% The user specifies the parameterts in the part labeled as "USER     *
% SPECIFIED PARAMETERS"                                               *
%                                                                     *
% The generated mesh is written to a file called CubeMesh.inp         *
%                                                                     *
%**********************************************************************
%                                                                     *
% Change Log                                                          *
%                                                                     *
% 24/09/2013 : Initial version.                                       *
%                                                                     *
%**********************************************************************

function [] = HexaMeshGeneratorInACube()
clc;
clear all;
close all;

disp('****************************************************');
disp('***       3D Hexa Mesh Generator in a Cube       ***');
disp('****************************************************');

x0 = 0.0;
y0 = 0.0;
z0 = 0.0;
x1 = 1.0;
y1 = 1.0;
z1 = 1.0;



% ********************************************************************
% USER SPECIFIED PARAMETERS
% ********************************************************************

NEx = 4;             % Number of elements in x direction. Select an even number for proper clustering.
NEy = NEx;
NEz = NEx;
clusterCoeff = 2.0;  % Enter 0.0 for no clustering.

eType = 1;           % Do not change this.
NENv = 27;           % Do not change this.
NENp = 8;            % Do not change this.
NGP = 8;             % Can be 1, 8.
density = 1.0;
viscosity = 0.01;
fxFunc = '0.0';
fyFunc = '0.0';

alpha     = 1.0;
dt        = 0.01;
t_ini     = 0.0;
t_final   = 10.0;
isRestart = 0;       % 0 means start from initial condition. 

maxIter   = 3;       % Used in Blasco's fractional step solver.
tolerance = 1e-3;    % Used in Blasco's fractional step solver.

% ********************************************************************
% END OF USER SPECIFIED PARAMETERS
% ********************************************************************



NE = NEx*NEy*NEz;       % Number of elements.

NxC = NEx + 1;          % Number of corner nodes in x direction
NyC = NEy + 1;          % Number of corner nodes in y direction
NzC = NEz + 1;          % Number of corner nodes in z direction

if NENv == 8
  Nx = NEx + 1;
  Ny = NEy + 1;
  Ny = NEz + 1;
  NN = Nx*Ny*Nz;        % Number of total velocity nodes is equal to number of corner nodes when NENv == NENp
  NCN = NN;             % Number of corner nodes.
elseif NENv == 27
  Nx = 2*NEx + 1;
  Ny = 2*NEy + 1;
  Nz = 2*NEz + 1;
  NN = Nx*Ny*Nz;        % Number of total velocity nodes is NOT equal to number of corner nodes when NENv ~= NENp
  NCN = (NEx+1)*(NEy+1)*(NEz+1);
end


Lx = x1 - x0;           % Size of the cavity in x direction.
Ly = y1 - y0;           % Size of the cavity in y direction.
Lz = z1 - z0;           % Size of the cavity in z direction.

file = fopen(strcat('CubeMesh.inp'), 'w');


% Generate dx, dy and dz vectors, which are the corner node spacings in x
% y and z directions. Mid-edge, mid-face and mid-element velocity nodes are
% not accounted for.

if(clusterCoeff == 0) % No clustering
  for i = 1:NxC
     dx(i) = (i-1) * Lx / (NxC-1);
  end

  for i = 1:NyC
     dy(i) = (i-1) * Ly / (NyC-1);
  end

  for i = 1:NzC
     dz(i) = (i-1) * Lz / (NzC-1);
  end
  
else  % Use clustering
  MAX = sinh(clusterCoeff);
  for i = 1:(NxC+1)/2
     xx = (1.0 * (i-1)) / ((NxC+1)/2-1);      % 0 < xx < 1
     xxx = sinh(clusterCoeff * xx);           % 0 < xxx < MAX
     dx(i) = Lx/2.0/MAX * xxx;
  end

  for i = (NxC+1)/2:NxC
     dx(i) = Lx - dx(NxC+1-i);
  end
  
  for j = 1:(NyC+1)/2
     yy = (1.0 * (j-1)) / ((NyC+1)/2-1);      % 0 < yy < 1
     yyy = sinh(clusterCoeff * yy);           % 0 < yyy < MAX
     dy(j) = Ly/2.0/MAX * yyy;
  end

  for j = (NyC+1)/2:NyC
     dy(j) = Ly - dy(NyC+1-j);
  end
  
  for k = 1:(NzC+1)/2
     zz = (1.0 * (k-1)) / ((NzC+1)/2-1);      % 0 < zz < 1
     zzz = sinh(clusterCoeff * zz);           % 0 < zzz < MAX
     dz(k) = Lz/2.0/MAX * zzz;
  end

  for k = (NzC+1)/2:NzC
     dz(k) = Lz - dz(NzC+1-k);
  end
  
end


% Print the header part of the input file
fprintf(file, '3D Lid-driven cavity problem \n');
fprintf(file, '================================================\n');
fprintf(file, 'eType    : %d \n', eType);
fprintf(file, 'NE       : %d \n', NE);
fprintf(file, 'NCN      : %d \n', NCN);
fprintf(file, 'NENv     : %d \n', NENv);
fprintf(file, 'NENp     : %d \n', NENp);
fprintf(file, 'NGP      : %d \n', NGP);
fprintf(file, 'alpha    : %f \n', alpha);
fprintf(file, 'dt       : %f \n', dt);
fprintf(file, 't_ini    : %f \n', t_ini);
fprintf(file, 't_final  : %f \n', t_final);
fprintf(file, 'maxIter  : %d \n', maxIter);
fprintf(file, 'tolerance: %f \n', tolerance);
fprintf(file, 'isRestart: %d \n', isRestart);
fprintf(file, 'density  : %f \n', density);
fprintf(file, 'viscosity: %f \n', viscosity);
fprintf(file, 'fx       : %s \n', fxFunc);
fprintf(file, 'fy       : %s \n', fyFunc);


% Generate and print x and y coordinates of the corner nodes.
fprintf(file, '================================================\n');
fprintf(file, 'Corner Node No         x                y                z\n');

nodeCounter = 0;
x = zeros(NxC,1);
y = zeros(NyC,1);
z = zeros(NzC,1);
for k = 1:NzC
  for j = 1:NyC
    for i = 1:NxC
      nodeCounter = nodeCounter + 1;
      x(i) = x0 + dx(i);
      y(j) = y0 + dy(j);
      z(k) = z0 + dz(k);
      fprintf(file, '%9i %18.7f %16.7f %16.7f\n', nodeCounter, x(i), y(j), z(k));
    end
  end
end


% Generate and print the connectivity list of the elements. Only corner
% nodes are listed in the connectivity.
fprintf(file, '================================================\n');

fprintf(file, 'Elem No   corner1  corner2  corner3  corner4  corner5  corner6  corner7  corner8\n');
  
elemCounter = 0;

for k = 1:NEz
  for j = 1:NEy
    for i = 1:NEx
      n1 = (k-1)*(NEx+1)*(NEy+1) + (j-1)*(NEx+1) + i;
      % All other nodes can be found using n1
      n2 = n1 + 1;
      n3 = n1 + NEx + 2;
      n4 = n3 - 1;
      n5 = n1 + (NEx+1)*(NEy+1);
      n6 = n2 + (NEx+1)*(NEy+1);
      n7 = n3 + (NEx+1)*(NEy+1);
      n8 = n4 + (NEx+1)*(NEy+1);
      
      elemCounter = elemCounter + 1;
      
      fprintf(file, '%6i  %8i %8i %8i %8i %8i %8i %8i %8i \n', elemCounter, n1, n2, n3, n4, n5, n6, n7, n8);
    end
  end
end



% Generate and print BC information.
fprintf(file, '================================================\n');
fprintf(file, 'BCs (Number of specified BCs, their types and strings) \n');
fprintf(file, 'nBC       : 2 \n');
fprintf(file, 'BC 1      : 1  0.0 : 0.0 : 0.0\n');
fprintf(file, 'BC 2      : 1  1.0 : 0.0 : 0.0\n');



fprintf(file, '================================================\n');
fprintf(file, 'nVelFaces : %d \n', 2*(NEx*NEy + NEx*NEz + NEy*NEz));
fprintf(file, 'nOutFaces : %d \n', 0);


fprintf(file, '================================================\n');
fprintf(file, 'Velocity BC (Elem# Face# BC#)\n');

k = 1;           % Bottom boundary
for j = 1:NEy
  for i = 1:NEx
    fprintf(file, '%5d    1    1\n', (k-1)*NEx*NEy + (j-1)*NEx + i);
  end
end

i = 1;           % Left boundary
for k = 1:NEz
  for j = 1:NEy
    fprintf(file, '%5d    5    1\n', (k-1)*NEx*NEy + (j-1)*NEx + i);
  end
end

i = NEx;         % Right boundary
for k = 1:NEz
  for j = 1:NEy
    fprintf(file, '%5d    3    1\n', (k-1)*NEx*NEy + (j-1)*NEx + i);
  end
end

j = 1;           % Front boundary
for k = 1:NEz
  for i = 1:NEx
    fprintf(file, '%5d    2    1\n', (k-1)*NEx*NEy + (j-1)*NEx + i);
  end
end

j = NEy;         % Back boundary
for k = 1:NEz
  for i = 1:NEx
    fprintf(file, '%5d    4    1\n', (k-1)*NEx*NEy + (j-1)*NEx + i);
  end
end

k = NEz;         % Top boundary
for j = 1:NEy
  for i = 1:NEx
    fprintf(file, '%5d    6    2\n', (k-1)*NEx*NEy + (j-1)*NEx + i);
  end
end


fprintf(file, '================================================\n');
fprintf(file, 'Outflow BC (Elem# Face# BC#)\n');


fprintf(file, '================================================\n');
fprintf(file, 'Node number where pressure is taken to be zero\n');
fprintf(file, '%d\n', (NEy/2)*NxC + (NxC+1)/2);    % Set pressure to zero at the mid-point of the bottom wall.

fprintf(file, '================================================\n');
fprintf(file, 'Monitor point coordinates\n');
fprintf(file, '0.5  0.5  0.5\n');

fclose(file);

fprintf('\nA file named CubeMesh.inp is generated. \n');
fprintf('\nProgram is terminated successfully :-) \n\n');






