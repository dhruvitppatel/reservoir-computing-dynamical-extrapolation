function [loc,map] = ks_map(t,w,n1,n2,L)

% produce surface of section piercings

% inputs:
% w - vectors of [a_i(t)]
% n1 - surface of section defined by a_n1 = 0
% n2 - map by the a_n2 coordinate piercing of surface of section

% outputs:
% map - map of coordinate n2 on surface a_n1=0

ind = 1;

for j = 2:length(w)-1
    if and(w(n1,j-1) < 0, w(n1,j) > 0)
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
    w1_1 = w(n1,loc1);
    w2_1 = w(n1,loc2);
    w1_6 = w(n2,loc1);
    w2_6 = w(n2,loc2);

    for i = 1:length(t1)
        t_int = linspace(t1(i),t2(i),100);
        q_1 = interp1([t1(i) t2(i)],[w1_1(i) w2_1(i)],t_int);
        q_6 = interp1([t1(i) t2(i)],[w1_6(i) w2_6(i)],t_int);
        [~,id] = min(abs(q_1));
        map(i) = q_6(id);
    end
    
    loc = loc2;
end