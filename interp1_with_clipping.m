function y_interp = interp1_with_clipping(x,y,x_q)
% MATLAB interp1 function but with the right-side clipped to 1 and
% left-side to 0

n = length(x_q);
xmin = min(x);
xmax = max(x);
y_interp = zeros(1,n);

for i = 1:n
    
    if x_q(i) < xmin
        y_interp(i) = 0;
    elseif x_q(i) > xmax
        y_interp(i) = 1;
    else
        y_interp(i) = interp1(x,y,x_q(i));
    end
    
end