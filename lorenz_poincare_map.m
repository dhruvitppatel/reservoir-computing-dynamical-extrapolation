function [loc,map] = lorenz_poincare_map(t,v,beta,L)

% produce surface of section piercings

% inputs:
% t - time vector
% v - [x(t);y(t);z(t)]
% beta - Lorenz beta parameter

% outputs:
% loc - indices for map points
% map - z values when xy - beta*z = 0

ind = 1;

a = v(1,:).*v(2,:) - beta.*v(3,:);
for j = 2:length(v)-1
    if and(a(j-1) > 0, a(j) < 0)
        loc1(ind) = j-1;
        loc2(ind) = j;
        ind=ind+1;
    end
end

if exist('loc1','var') == 0
    
   loc = ones(1,L);
   map = zeros(1,L);
   
else
    % Linear interpolation
    t1 = t(loc1);
    t2 = t(loc2);
    z1 = v(3,loc1);
    z2 = v(3,loc2);

    for i = 1:length(t1)
        t_int = linspace(t1(i),t2(i),100);
        q_1 = interp1([t1(i) t2(i)],[z1(i) z2(i)],t_int);
        [~,id] = min(abs(q_1));
        map(i) = q_1(id);
    end
    
    loc = loc2;
end