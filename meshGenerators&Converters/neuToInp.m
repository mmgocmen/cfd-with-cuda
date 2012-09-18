clear all 
clc

disp('********************************************************');
disp('******   Mesh Converter for Navier-Stokes Solver  ******');
disp('******     (from neutral mesh format("neu"))      ******');
disp('********************************************************');
disp(' ');

fileName = input('\nEnter the name of the input file (without the .neu extension): ','s');
inputFile = fopen(strcat(fileName,'.neu'), 'r');

disp('if mesh contains hexahedron (8 node) elements; type "1" ,');
disp('if mesh contains tetrahedral (4 node) elements; type "2".');
elementType = input('element type: ');

disp(' ');
disp('for example; for "Backward Facing Step" problem there are 3 BCs,');
disp(' 1 inlet, 1 outlet and 1 wall (if all the walls are chosen at once).');
noBC = input('number of boundary conditions (including walls) = ');

disp('type of BC; for velocity BC type:1,');
disp('            for pressure BC type:2,');
disp('            for wall BC     type:3.');
disp(' ');

BCdata = zeros(noBC,4);
velocityBC(1) = 0;
pressureBC(1) = 0;
wallBC(1) = 0;
for i=1:noBC
    fprintf('for %d. BC',i);
    type = input('type of BC: ');
    BCdata(i,1) = type;
    switch type
        case 1
        BCdata(i,2) = input('x component of velocity = ');
        BCdata(i,3) = input('y component of velocity = ');
        BCdata(i,4) = input('z component of velocity = ');
        disp(' ');
        velocityBC(1) = velocityBC(1) + 1;
        velocityBC(velocityBC(1)+1) = i;
        case 2
        BCdata(i,2) = input('pressure value = '); 
        disp(' ');
        pressureBC(1) = pressureBC(1) + 1;
        pressureBC(pressureBC(1)+1) = i;
        case 3
        wallBC(1,1) = wallBC(1,1) + 1; 
        wallBC(wallBC(1)+1) = i;
    end
end
%wall BC's
% velocity_value_x(2) = 0;
% velocity_value_y(2) = 0;
% velocity_value_z(2) = 0;

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

switch elementType
    case 1
        LtoG = zeros(NE,8);
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
    case 2
       LtoG = zeros(NE,4);
       for i=1:NE
               dummy = str2num(fgets(inputFile)); 
               LtoG(i,1)=dummy(4);
               LtoG(i,2)=dummy(5);
               LtoG(i,3)=dummy(6);
               LtoG(i,4)=dummy(7);
       end        
end

jump=round(NE/10);
jump=5+jump;
if mod(NE,10)>0 && mod(NE,10)<5
    jump=jump+1;
end

for i=1:jump
  dummy = fgets(inputFile);
end

for k=1:noBC  
    dummy = fgets(inputFile);
    dummy = fgets(inputFile); 
    dummy = fgets(inputFile);     
    [dummy2] = strread(dummy, '%s', 'delimiter', ' ');
    BCdata(k,5) = str2num(dummy2{3});

    for i=1:BCdata(k,5)
        dummy = str2num(fgets(inputFile)); 
        BCdata(k,5+i)=dummy(1);      
    end
end

%making edges of the inlet and outlet regions from wall BC to inlet and
%outlet BC.
counter=0;
for k=1:noBC
    if (BCdata(k,1) == 3)
        for i=1:BCdata(k,5)
            check=0;
            for kk=1:velocityBC(1)
                for j=1:BCdata(velocityBC(kk+1),5)
                    if BCdata(k,i+5)==BCdata(velocityBC(kk+1),j+5)
                        check=1;
                        break
                    end
                end
            end
            for kk=1:pressureBC(1)
                for j=1:BCdata(pressureBC(kk+1),5)
                    if BCdata(k,i+5)==BCdata(pressureBC(kk+1),j+5)
                        check=1;
                        break
                    end
                end
            end                
            
            if check==0
                counter=counter+1;
                realWallBC(counter)=BCdata(k,i+5);
            end 

        end
    end
end
%-------------------------------------------------------------------------
%creating the input file for code
disp('********* ...starts to write the input file. ***********');

outputFile = fopen(strcat(fileName,'.inp'),'wt');

switch elementType
    case 1
        fprintf(outputFile,'Hexahedron Mesh File for Navier-Stokes Solver\n');
        fprintf(outputFile,'================================================\n');
        fprintf(outputFile,'eType     : 3\n');
        fprintf(outputFile,'NE        : %d\n', NE);
        fprintf(outputFile,'NCN       : %d\n', NN);
        fprintf(outputFile,'NN        : %d\n', NN);
        fprintf(outputFile,'NENv      : 8\n');
        fprintf(outputFile,'NENp      : 8\n');
        fprintf(outputFile,'NGP       : 8\n');
    case 2
        fprintf(outputFile,'Hexahedron Mesh File for Navier-Stokes Solver\n');
        fprintf(outputFile,'================================================\n');
        fprintf(outputFile,'eType     : 4\n');
        fprintf(outputFile,'NE        : %d\n', NE);
        fprintf(outputFile,'NCN       : %d\n', NN);
        fprintf(outputFile,'NN        : %d\n', NN);
        fprintf(outputFile,'NENv      : 4\n');
        fprintf(outputFile,'NENp      : 4\n');
        fprintf(outputFile,'NGP       : 4\n');
end
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

switch elementType
    case 1
        fprintf(outputFile,'=====================================================================\n');
        fprintf(outputFile,'Elem#   node1   node2   node3   node4   node5   node6   node7   node8\n');
        for i=1:NE
           fprintf(outputFile,'%5d',(i-1));
           fprintf(outputFile,'%11d%11d%11d%11d%11d%11d%11d%11d',...
                   LtoG(i,1)-1, LtoG(i,2)-1, LtoG(i,6)-1, LtoG(i,5)-1, LtoG(i,3)-1, LtoG(i,4)-1, LtoG(i,8)-1, LtoG(i,7)-1);
           fprintf(outputFile,'\n');
        end
    case 2
        fprintf(outputFile,'=====================================================================\n');
        fprintf(outputFile,'Elem#   node1   node2   node3   node4 \n');
        for i=1:NE
           fprintf(outputFile,'%5d',(i-1));
           fprintf(outputFile,'%11d%11d%11d%11d%',LtoG(i,1)-1, LtoG(i,2)-1, LtoG(i,3)-1, LtoG(i,4)-1);
           fprintf(outputFile,'\n');
        end        
end

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'BCs (Number of specified BCs, their types and strings)\n');
fprintf(outputFile,'nBC       : %d\n',noBC-wallBC(1)+1);
for i=1:velocityBC(1)
    fprintf(outputFile,'BC %d      : %d  %f : %f : %f\n', i, i , BCdata(velocityBC(i+1),2), BCdata(velocityBC(i+1),3), BCdata(velocityBC(i+1),4)); 
end
fprintf(outputFile,'BC %d      : %d  0.0 : 0.0 : 0.0\n', i+1, i+1);
for j=1:pressureBC(1)
    fprintf(outputFile,'BC %d      : %f \n', i+j+1, BCdata(pressureBC(j)+1,2)); 
end  

noVelocity = 0;
for i=1:velocityBC(1)
   noVelocity = noVelocity + BCdata(velocityBC(i+1),5);
end
noPressure = 0;
for i=1:pressureBC(1)
    noPressure = noPressure + BCdata(pressureBC(i+1),5);
end

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'nVelNodes : %d\n', (noVelocity + counter));
fprintf(outputFile,'nPressureNodes : %d\n', noPressure);

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'Velocity BC (Node#  BC No.)\n');

for i=1:velocityBC(1)
    for j=1:BCdata(velocityBC(i+1),5)
        fprintf(outputFile,'%5d     %d\n', BCdata(velocityBC(i+1),5+j)-1,i);
    end
end

for k=1:counter
    fprintf(outputFile,'%5d     %d\n', realWallBC(k)-1, i+1);
end

fprintf(outputFile,'================================================\n');
fprintf(outputFile,'Pressure BC (Node#  BC No.)\n');
fprintf(outputFile,'================================================\n');
for i=1:pressureBC(1)
    for j=1:BCdata(pressureBC(i+1),5)
        fprintf(outputFile,'%5d     %d\n', BCdata(pressureBC(i+1),5+j)-1,i+velocityBC(1)+1);
    end
end

fclose(outputFile); 
disp(' ');
disp('************** Input file is created! ******************');

fprintf('(name of input file : "%s.inp")\n', fileName);

fclose(inputFile);

