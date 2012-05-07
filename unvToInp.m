clear all 
clc

disp('********************************************************');
disp('******   Mesh Converter for Navier-Stokes Solver  ******');
disp('******     (from universal mesh format("unv"))    ******');
disp('******         (works for tetrahedral mesh)       ******');
disp(' ');

fileName = input('\nEnter the name of the input file (without the .unv extension): ','s');
inputFile = fopen(strcat(fileName,'.unv'), 'r');

velocity_value_x(1) = input('x component of velocity at velocity(inlet) boundary condition = ');
velocity_value_y(1) = input('y component of velocity at velocity(inlet) boundary condition = ');
velocity_value_z(1) = input('z component of velocity at velocity(inlet) boundary condition = ');
disp(' ');
pressure_value = input('pressure value at pressure(outlet) boundary condition = ');

%wall BC's
velocity_value_x(2) = 0;
velocity_value_y(2) = 0;
velocity_value_z(2) = 0;

for i=1:282
    dummy = fgets(inputFile);
end

NN=0;

while (dummy~=-1)
   dummy = str2num(fgets(inputFile)); 
   if (dummy~=-1)
       dummy = str2num(fgets(inputFile)); 
       coord(NN+1,1)=dummy(1);
       coord(NN+1,2)=dummy(2);
       coord(NN+1,3)=dummy(3);
       NN=NN+1;
   end
end

dummy = str2num(fgets(inputFile)); 
dummy = str2num(fgets(inputFile)); 
NE=0;
while (dummy~=-1)
   dummy = str2num(fgets(inputFile)); 
   if (dummy~=-1)
       dummy = str2num(fgets(inputFile)); 
       LtoG(NE+1,1)=dummy(1);
       LtoG(NE+1,2)=dummy(2);
       LtoG(NE+1,3)=dummy(3);
       LtoG(NE+1,4)=dummy(4);
       NE=NE+1;
   end
end

dummy = str2num(fgets(inputFile)); 
dummy = str2num(fgets(inputFile));

%inletBC nodes
dummy = str2num(fgets(inputFile)); 
noInletBC = dummy(8);
dummy = str2num(fgets(inputFile)); 

if (mod(noInletBC,2)==0)
    for counter=0:2:(noInletBC-2)
        dummy = str2num(fgets(inputFile)); 
        inletBC(counter+1)=dummy(2);
        inletBC(counter+2)=dummy(6);        
    end
else
    for counter=0:2:(noInletBC-3)
        dummy = str2num(fgets(inputFile)); 
        inletBC(counter+1)=dummy(2);
        inletBC(counter+2)=dummy(6);        
    end   
    dummy = str2num(fgets(inputFile));
    inletBC(counter+3)=dummy(2);      
end


%outletBC nodes
dummy = str2num(fgets(inputFile)); 
noOutletBC = dummy(8);
dummy = str2num(fgets(inputFile)); 

if (mod(noOutletBC,2)==0)
    for counter=0:2:(noOutletBC-2)
        dummy = str2num(fgets(inputFile)); 
        outletBC(counter+1)=dummy(2);
        outletBC(counter+2)=dummy(6);        
    end
else
    for counter=0:2:(noOutletBC-3)
        dummy = str2num(fgets(inputFile)); 
        outletBC(counter+1)=dummy(2);
        outletBC(counter+2)=dummy(6);        
    end   
    dummy = str2num(fgets(inputFile));
    outletBC(counter+3)=dummy(2);      
end

%wallBC nodes
dummy = str2num(fgets(inputFile)); 
noWallBC = dummy(8);
dummy = str2num(fgets(inputFile)); 

if (mod(noWallBC,2)==0)
    for counter=0:2:(noWallBC-2)
        dummy = str2num(fgets(inputFile)); 
        wallBC(counter+1)=dummy(2);
        wallBC(counter+2)=dummy(6);        
    end
else
    for counter=0:2:(noWallBC-3)
        dummy = str2num(fgets(inputFile)); 
        wallBC(counter+1)=dummy(2);
        wallBC(counter+2)=dummy(6);        
    end   
    dummy = str2num(fgets(inputFile));
    wallBC(counter+3)=dummy(2);      
end

%-------------------------------------------------------------------------
%creating the input file for code
disp('********* ...starts to write the input file. ***********');

outputFile = fopen('NavierStokes-tetrahedral-1.inp','wt');

fprintf(outputFile,'Tetrahedral Mesh File for Navier-Stokes Solver\n');
fprintf(outputFile,'================================================\n');
fprintf(outputFile,'eType     : 4\n');
fprintf(outputFile,'NE        : %d\n', NE);
fprintf(outputFile,'NCN       : %d\n', NN);
fprintf(outputFile,'NN        : %d\n', NN);
fprintf(outputFile,'NENv      : 4\n');
fprintf(outputFile,'NENp      : 4\n');
fprintf(outputFile,'NGP       : 4\n');
fprintf(outputFile,'iterMax   : 50\n');
fprintf(outputFile,'tolerance : 1e-6\n');
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

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'Elem#   node1   node2   node3   node4\n');
for i=1:NE
   fprintf(outputFile,'%5d',(i-1));
   for j=1:4
      fprintf(outputFile,'%11d',LtoG(i,j)-1);
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
fprintf(outputFile,'nVelNodes : %d\n', (noInletBC + noWallBC));
fprintf(outputFile,'nPressureNodes : %d\n', noOutletBC);

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'Velocity BC (Node#  BC No.)\n');

for i=1:noInletBC
    fprintf(outputFile,'%5d     1\n', inletBC(i)-1);
end

for i=1:noWallBC
    fprintf(outputFile,'%5d     2\n', wallBC(i)-1);
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
disp('(name of input file : "NavierStokes-tetrahedral-1.inp")');





