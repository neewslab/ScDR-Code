function [k,ft]=truss2d(x1,y1,x2,y2,el,area)

%--------------------------------------------------------------
%  Purpose:
%     Stiffness and mass matrices for the 2-d truss element
%     nodal dof {u_1 v_1 theta_1 u_2 v_2 theta_2}
%
%     This operation forms the 4x4 stiffness and mass matrix M1 and M2
%     and the 4x4 force transformation matrix M3
%
%  Synopsis:
%     [k,ft]=2dtruss(x1,y1,x2,y2,el,area)
%
%  Variable Description:
%     k  - global element stiffness matrix (size of 4x4)   
%     ft - reduced element force transformation matrix (size of 4x4)
%     x1 - the X coordinate of the node 1
%     x2 - the X coordinate of the node 2
%     y1 - the Y coordinate of the node 1
%     y2 - the Y coordinate of the node 2
%     el - elastic modulus 
%     area - area of beam cross-section
%
%  Internal Variables:
%     kl   - local element stiffness matrix (size of 4x4)
%     leng - element length
%     t    - orientation transformation matrix (size of 4x4)
%     ftb  - element force transformation matrix (size of 4x4)
%     beta - angle between the local and global axes                                 ipt = 1: consistent mass matrix
%            is positive if the local axis is in the ccw direction from
%            the global axis
%--------------------------------------------------------------------------
   
% calculate element length and rotation angle

 leng = sqrt((x2-x1)^2 + (y2-y1)^2);
 
 dx = x2-x1;
 dy = y2-y1;
 if dx == 0 
     if dy > 0
         beta = pi/2.;
     else
         beta = -pi/2.;
     end
 else
     beta = atan (dy/dx);
 end
 
% stiffness matrix at the local axis

 a=el*area/leng;
 kl=[a    0   -a   0;...
     0    0    0   0;...
     -a   0    a   0;...
     0    0    0   0];

% transformation matrix

 t=[ cos(beta)  sin(beta)   0          0;...
    -sin(beta)  cos(beta)   0          0;...
     0          0           cos(beta)  sin(beta);...
     0          0          -sin(beta)  cos(beta)];

% element force transformation matrix

 ft = kl*t;

 % stiffness matrix in the global coordinate system

 k=t'*kl*t;

