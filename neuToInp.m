clear all 
clc

disp('********************************************************');
disp('******   Mesh Converter for Navier-Stokes Solver  ******');
disp('******     (from neutral mesh format("neu"))      ******');
disp('******                (hexahedron)                ******');
disp(' ');

fileName = input('\nEnter the name of the input file (without the .neu extension): ','s');
inputFile = fopen(strcat(fileName,'.neu'), 'r');

velocity_value_x(1) = input('x component of velocity at velocity(inlet) boundary condition = ');
velocity_value_y(1) = input('y component of velocity at velocity(inlet) boundary condition = ');
velocity_value_z(1) = input('z component of velocity at velocity(inlet) boundary condition = ');
disp(' ');
pressure_value = input('pressure value at pressure(outlet) boundary condition = ');

%wall BC's
velocity_value_x(2) = 0;
velocity_value_y(2) = 0;
velocity_value_z(2) = 0;

for i=1:6
    dummy = fgets(inputFile);
end

dummy = str2num(fgets(inputFile)); 
NN=dummy(1);
NE=dummy(2);

dummy = fgets(inputFile);
dummy = fgets(inputFile);

for i=1:NN
   dummy = str2num(fgets(inputFile)); 
   coord(i,1)=dummy(2);
   coord(i,2)=dummy(3);
   coord(i,3)=dummy(4); 
end

dummy = str2num(fgets(inputFile)); 
dummy = str2num(fgets(inputFile)); 

for i=1:NE
       dummy = str2num(fgets(inputFile)); 
       LtoG(i,1)=dummy(4);
       LtoG(i,2)=dummy(5);
       LtoG(i,3)=dummy(6);
       LtoG(i,4)=dummy(7);
       LtoG(i,5)=dummy(8);
       LtoG(i,6)=dummy(9);
       LtoG(i,7)=dummy(10);
       dummy = str2num(fgets(inputFile));        
       LtoG(i,8)=dummy(1);
end

jump=round(NE/10);
jump=7+jump;

for i=1:jump
  dummy = fgets(inputFile);
end

%inletBC nodes
dummy = fgets(inputFile); 
[dummy2] = strread(dummy, '%s', 'delimiter', ' ');
noInletBC = str2num(dummy2{3});

for i=1:noInletBC
    dummy = str2num(fgets(inputFile)); 
    inletBC(i)=dummy(1);      
end

dummy = fgets(inputFile);
dummy = fgets(inputFile);

%outletBC nodes
dummy = fgets(inputFile); 
[dummy2] = strread(dummy, '%s', 'delimiter', ' ');
noOutletBC = str2num(dummy2{3});

for i=1:noOutletBC
    dummy = str2num(fgets(inputFile)); 
    outletBC(i)=dummy(1);      
end

dummy = fgets(inputFile);
dummy = fgets(inputFile);

%wallBC nodes
dummy = fgets(inputFile); 
[dummy2] = strread(dummy, '%s', 'delimiter', ' ');
noWallBC = str2num(dummy2{3});

for i=1:noWallBC
    dummy = str2num(fgets(inputFile)); 
    wallBC(i)=dummy(1);      
end

%making edges of the inlet and outlet regions from wall BC to inlet and
%outlet BC.
counter=0;
for i=1:noWallBC
    check=0;
    for j=1:noInletBC
        if wallBC(i)==inletBC(j)
            check=1;
            break
        end
    end
    for j=1:noOutletBC
        if wallBC(i)==outletBC(j)
            check=1;
            break;
        end
    end
    if check==0
        counter=counter+1;
        realWallBC(counter)=wallBC(i);
    end
end

%-------------------------------------------------------------------------
%creating the input file for code
disp('********* ...starts to write the input file. ***********');

outputFile = fopen(strcat(fileName,'.inp'),'wt');

fprintf(outputFile,'Hexahedron Mesh File for Navier-Stokes Solver\n');
fprintf(outputFile,'================================================\n');
fprintf(outputFile,'eType     : 3\n');
fprintf(outputFile,'NE        : %d\n', NE);
fprintf(outputFile,'NCN       : %d\n', NN);
fprintf(outputFile,'NN        : %d\n', NN);
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
for i=1:NN
   fprintf(outputFile,'%5d%16.7f%16.7f%16.7f\n', (i-1), coord(i,1), coord(i,2), coord(i,3));
end

fprintf(outputFile,'=====================================================================\n');
fprintf(outputFile,'Elem#   node1   node2   node3   node4   node5   node6   node7   node8\n');
for i=1:NE
   fprintf(outputFile,'%5d',(i-1));
   fprintf(outputFile,'%11d%11d%11d%11d%11d%11d%11d%11d',...
           LtoG(i,1)-1, LtoG(i,2)-1, LtoG(i,6)-1, LtoG(i,5)-1, LtoG(i,3)-1, LtoG(i,4)-1, LtoG(i,8)-1, LtoG(i,7)-1);
   fprintf(outputFile,'\n');
end

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'BCs (Number of specified BCs, their types and strings)\n');
fprintf(outputFile,'nBC       : 3\n');                    
fprintf(outputFile,'BC 1      : 1  %d : %d : %d\n', velocity_value_x(1), velocity_value_y(1), velocity_value_z(1));       
fprintf(outputFile,'BC 2      : 2  %d : %d : %d\n', velocity_value_x(2), velocity_value_y(2), velocity_value_z(2));    
fprintf(outputFile,'BC 3      : %d\n', pressure_value);


fprintf(outputFile,'================================================\n');
% fprintf(outputFile,'nVelNodes : %d\n', (noInletBC + noWallBC));
fprintf(outputFile,'nVelNodes : %d\n', (noInletBC + counter));
fprintf(outputFile,'nPressureNodes : %d\n', noOutletBC);

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'Velocity BC (Node#  BC No.)\n');

for i=1:noInletBC
    fprintf(outputFile,'%5d     1\n', inletBC(i)-1);
end

% for i=1:noWallBC
%     fprintf(outputFile,'%5d     2\n', wallBC(i)-1);
% end

for i=1:counter
    fprintf(outputFile,'%5d     2\n', realWallBC(i)-1);
end

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'Pressure BC (Node#  BC No.)\n');
fprintf(outputFile,'================================================\n');
for i=1:noOutletBC
   fprintf(outputFile,'%5d     3\n', outletBC(i)-1);
end

fclose(outputFile); 
disp(' ');
disp('************** Input file is created! ******************');

fprintf('(name of input file : "%s.inp")\n', fileName);

fclose(inputFile);

