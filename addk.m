function [kk]=addk(kk,k,id,icol);

%----------------------------------------------------------
%  Purpose:
%     Assembly of element matrices into the system matrix
%
%  Synopsis:
%     [kk]=addk(kk,k,id,iel)
%
%  Variable Description:
%     kk  - structural system stiffness matrix
%     k   - element stiffness matrix (size of 6x6)
%     id  - connectivity array (size of 6 x nel)
%     icol - column of the connectivity array to be used
%
%  Internal Variables
%     index - d.o.f. vector associated with an element (size of 6x1)
%-----------------------------------------------------------

 
 [m,n] = size(id);
 edof = m;
 index = id(:,icol);
 for i=1:edof
   ii=index(i);
   if ii ~= 0
     for j=1:edof
       jj=index(j);
       if jj ~= 0
         kk(ii,jj)=kk(ii,jj)+k(i,j);
     else end
     end
 else end
 end