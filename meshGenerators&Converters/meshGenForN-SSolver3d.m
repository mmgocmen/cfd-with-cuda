clc
clear all
disp('********************************************************');
disp('****   3D Mesh Generator for Navier Stokes Solver   ****');
disp('********************************************************');
disp(' ');
disp('This code generates 3D meshes for Driven-Cavity and ');
disp('Rectangular Duct problems with cubic(hexahedron) element.');
disp(' ');
disp('Please select the problem type;');
disp('for Driven Cavity, type "1", ');
disp('for Rectangular Duct, type "2", ');
problemType = input('1?, 2? = ');
if (problemType == 1 || problemType == 2)
else
    disp('Wrong selection! Select again! ');
    disp('for Driven Cavity, type "1" ');
    disp('for Rectangular Duct, type "2", ');
    problemType = input('1?, 2? =');
end

if (problemType == 1)
    dim_x = input('x dimension of cavity = ');                         
    dim_y = input('y dimension of cavity = ');                         
    dim_z = input('z dimension of cavity = ');
else
    dim_x = input('x dimension of tube = ');                         
    dim_y = input('y dimension of tube = ');                         
    dim_z = input('z dimension of tube = ');
end
dim_el = input('length of a side of a cubic(hexahedron) element = ');

if (dim_el < 1)
    scale = 1/dim_el;
    dim_el=1;
    dim_x=scale*dim_x;
    dim_y=scale*dim_y;
    dim_z=scale*dim_z;
else
    scale = 1;
end

if (problemType ==1)
    disp(' ');
    disp('- One node at corner of the xy surface of the cavity is selected');
    disp('as pressure node. Pressure value of this node is zero(0).');
    pressure_value = 0;    %pressure node definition
    nPressureNodes = 1;    %.
    pressureNodes(1) = 0;  %.
    disp(' ');    
    disp('- Velocity boundary is selected at the z=zmax by default.');
    disp(' ');
    BC_velocity = 6;       %@ z=z_max, boundary condition area = 6
    velocity_value_x(1) = input('x component of velocity at velocity BC = ');
    velocity_value_y(1) = input('y component of velocity at velocity BC = ');
    velocity_value_z(1) = input('z component of velocity at velocity BC = ');
else
    disp(' ');
    disp('@ x=0, boundary condition area = 1');
    disp('@ y=0, boundary condition area = 2');
    disp('@ z=0, boundary condition area = 3');
    disp('@ x=x_max, boundary condition area = 4');
    disp('@ y=y_max, boundary condition area = 5');
    disp('@ z=z_max, boundary condition area = 6');
    disp(' ');
    BC_velocity = input('velocity boundary condition area = ');
    velocity_value_x(1) = input('x component of velocity at velocity boundary condition = ');
    velocity_value_y(1) = input('y component of velocity at velocity boundary condition = ');
    velocity_value_z(1) = input('z component of velocity at velocity boundary condition = ');
    disp(' ');
    BC_pressure = input('pressure boundary condition area = ');
    pressure_value = input('pressure value at pressure boundary condition = ');
end

disp(' ');
disp('********   Inputs are collected successfully   *********');
disp(' ');

%-------------------------------------------------------------------------
%wall BC's
velocity_value_x(2) = 0;
velocity_value_y(2) = 0;
velocity_value_z(2) = 0;

j=0;
for i=1:6
    if (problemType == 1)
        if(BC_velocity~=i)
            j=j+1;
            wallBC(j)=i;
        end
    else
        if(BC_velocity~=i && BC_pressure ~=i)
            j=j+1;
            wallBC(j)=i;
        end
    end
end
%-------------------------------------------------------------------------

a=dim_x/dim_el;
b=dim_y/dim_el;
c=dim_z/dim_el;
counter = 0;
BC_faces_counter=zeros(6,1);

for k=0:dim_el:dim_z
   for j=0:dim_el:dim_y
      for i=0:dim_el:dim_x
         if (i==0)                                      %--------------------------
            BC_faces_counter(1)=BC_faces_counter(1)+1;  %finds and keeps boundaries
            BC_faces(1,BC_faces_counter(1))= counter;   %.
         end                                            %.
         if (j==0)                                      %.
            BC_faces_counter(2)=BC_faces_counter(2)+1;
            BC_faces(2,BC_faces_counter(2))= counter;
         end
         if (k==0)
            BC_faces_counter(3)=BC_faces_counter(3)+1;
            BC_faces(3,BC_faces_counter(3))= counter;
         end
         if (i==dim_x)
            BC_faces_counter(4)=BC_faces_counter(4)+1;
            BC_faces(4,BC_faces_counter(4))= counter;
         end
         if (j==dim_y)
            BC_faces_counter(5)=BC_faces_counter(5)+1;
            BC_faces(5,BC_faces_counter(5))= counter;
         end
         if (k==dim_z)
            BC_faces_counter(6)=BC_faces_counter(6)+1;
            BC_faces(6,BC_faces_counter(6))= counter;
         end                               %---------------------------
            
         node(counter+1,1)=i;              %---------------------------
         node(counter+1,2)=j;              %node coordinates
         node(counter+1,3)=k;              %---------------------------
         
         counter=counter+1;
       end
    end
end

number_of_nodes = counter;
counter = 0;

%-------------------------------------------------------------------------
%to find elemental nodes, code determines element centers and to find the 
%true node, it calculates the "distance" between node and the element
%center.

ini_val = dim_el/2;
distance = ini_val*3;                     

for i=0:a-1
   for j=0:b-1
      for k=0:c-1
         position_x_el = ini_val + i*dim_el;
         position_y_el = ini_val + j*dim_el;
         position_z_el = ini_val + k*dim_el;
         counter = counter + 1;
         t=0;
         for counter_2 = 1:number_of_nodes
            if((abs(position_x_el-node(counter_2,1)) + abs(position_y_el-node(counter_2,2)) + abs(position_z_el-node(counter_2,3)))==distance)
               t=t+1;
               ltog(counter,t) = counter_2-1;
               if (t==8)
                  break;
               end
            end
         end
         temp=ltog(counter,3);                  %-------------------
         ltog(counter,3)=ltog(counter,4);       %order of the nodes  
         ltog(counter,4)=temp;                  %(3&4 and 7&8) are 
         temp=ltog(counter,7);                  %incompatible with 
         ltog(counter,7)=ltog(counter,8);       %gauss quadrature. 
         ltog(counter,8)=temp;                  %correcting it
      end
   end
end
number_of_elements = counter;
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
%arranging velocity boundaries
nVelNodes=0;
for j=1:BC_faces_counter(BC_velocity)
    nVelNodes = nVelNodes + 1;
    velNodes(nVelNodes) = BC_faces(BC_velocity,j);
end
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
%arranging pressure boundaries for rectangular duct problem
if (problemType == 2)
    nPressureNodes=0;
    for j=1:BC_faces_counter(BC_pressure)
        nPressureNodes = nPressureNodes+1;
        pressureNodes(nPressureNodes) = BC_faces(BC_pressure,j);
    end
end
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
%rearranging wall boundaries
nWallNodes=0;
controller = 1;
if (problemType == 1)
    noWall = 5;
else
    noWall = 4;
end

for i=1:noWall
    for j=1:BC_faces_counter(wallBC(i))
        for k=1:nWallNodes
            if (wallNodes(k)== BC_faces(wallBC(i),j))
                controller = 0;
            end
        end
        
        %making inlet edges from wall to velocity nodes 
        for k=1:nVelNodes
            if (BC_faces(wallBC(i),j) == velNodes(k))
                controller = 0;
            end
        end
        %-----------------------------
        
        %extracting pressure node
        if (problemType == 1)
            if (BC_faces(wallBC(i),j)==0)
                controller = 0;
            end
        else
            for k=1:nPressureNodes
                if (BC_faces(wallBC(i),j) == pressureNodes(k))
                    controller = 0;
                end
            end
        end
        %-----------------------------
        
        if (controller == 1)
        	nWallNodes = nWallNodes + 1;
        	wallNodes(nWallNodes) = BC_faces(wallBC(i),j);
        end
        controller = 1;
    end
end
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
%creating the input file for code
disp('********* ...starts to write the input file. ***********');

if (problemType == 1)
    outputFile = fopen('fem3dCavityInput.inp','wt');   
else
    outputFile = fopen('fem3dRectangularDuctInput.inp','wt');  
end


if (problemType == 1)
    fprintf(outputFile,'3D Driven-Cavity Flow Input File\n');    
else
    fprintf(outputFile,'3D Rectangular Duct Input File\n');
end
fprintf(outputFile,'================================================\n');
fprintf(outputFile,'eType     : 3\n');
fprintf(outputFile,'NE        : %d\n', number_of_elements);
fprintf(outputFile,'NCN       : %d\n', number_of_nodes);
fprintf(outputFile,'NN        : %d\n', number_of_nodes);
fprintf(outputFile,'NENv      : 8\n');
fprintf(outputFile,'NENp      : 8\n');
fprintf(outputFile,'NGP       : 8\n');
fprintf(outputFile,'iterMax   : 100\n');
fprintf(outputFile,'tolerance : 1e-5\n');
fprintf(outputFile,'solverIterMax : 1000\n');
fprintf(outputFile,'solverTol     : 1e-5\n');
fprintf(outputFile,'density   : 1.0\n');
fprintf(outputFile,'viscosity : 1.0\n');
fprintf(outputFile,'fx        : 0.0\n');
fprintf(outputFile,'fy        : 0.0\n');

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'Node#           x               y               z\n');
for i=1:number_of_nodes
   fprintf(outputFile,'%5d%16.7f%16.7f%16.7f\n', (i-1), node(i,1)/scale, node(i,2)/scale, node(i,3)/scale);
end

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'Elem#   node1   node2   node3   node4   node5   node6   node7   node8\n');
for i=1:number_of_elements
   fprintf(outputFile,'%5d',(i-1));
   for j=1:8
      fprintf(outputFile,'%11d',ltog(i,j));
   end
   fprintf(outputFile,'\n');
end

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'BCs (Number of specified BCs, their types and strings)\n');
fprintf(outputFile,'nBC       : 3\n');                    
fprintf(outputFile,'BC 1      : 1  %d : %d : %d\n', velocity_value_x(1), velocity_value_y(1), velocity_value_z(1));       
fprintf(outputFile,'BC 2      : 2  %d : %d : %d\n', velocity_value_x(2), velocity_value_y(2), velocity_value_z(2));    
fprintf(outputFile,'BC 3      : %d\n', pressure_value);


fprintf(outputFile,'================================================\n');
fprintf(outputFile,'nVelNodes : %d\n', (nVelNodes + nWallNodes));
fprintf(outputFile,'nPressureNodes : %d\n', nPressureNodes);

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'Velocity BC (Node#  BC No.)\n');

for i=1:nVelNodes
    fprintf(outputFile,'%5d     1\n', velNodes(i));
end

for i=1:nWallNodes
    fprintf(outputFile,'%5d     2\n', wallNodes(i));
end

%-------------------------------------------------------------------------
fprintf(outputFile,'================================================\n');
fprintf(outputFile,'Pressure BC (Node#  BC No.)\n');
fprintf(outputFile,'================================================\n');
for i=1:nPressureNodes
   fprintf(outputFile,'%5d     3\n', pressureNodes(i));
end

fclose(outputFile); 
disp(' ');
disp('************** Input file is created! ******************');
if (problemType == 1)
    disp('(name of input file:"fem3dCavityInput.inp")');   
else
    disp('(name of input file:"fem3dRectangularDuctInput.inp")');
end