function [t,v] = get_lorenz(ls,rho,v0)
% Generate Lorenz trajectory using RK4 method 
%
% Inputs:
%   ls: struct with fields
%       dt: integration time step
%       t1: start time
%       trans: transient length
%       t2: end time
%       beta: Lorenz beta parameter 
%       sigma: Lorenz sigma parameter
%   `   eps: max noise [eps_x; eps_y; eps_z]
%   rho: Lorenz rho parameter time series
%   v0: [x0,y0,z0] initial condition
%
% Returns:
%   t: time vector
%   v: [x(t); y(t); z(t); rho(t)] time series

% ----------------------------------------------------------------------- %

% Unpack ls struct
dt = ls.dt;
t1 = ls.t1;
trans = ls.trans;
t2 = ls.t2;
beta = ls.beta;
sigma = ls.sigma;
eps = ls.eps;

% Time vector, initial condition, noise
t = t1:dt:t2;
leng = length(t);
v = [v0 zeros(3,leng-1)];
noise = -eps + 2.*eps.*rand(3,size(v,2));

f =@(v_i,rho) [sigma.*(v_i(2)-v_i(1)); ...
    v_i(1).*(rho-v_i(3))-v_i(2); v_i(1).*v_i(2)-beta.*v_i(3)];

for i = 1:leng-1
    vi = v(:,i);
    k1 = dt.*f(vi,rho(i));
    k2 = dt.*f(vi+0.5*k1,rho(i));
    k3 = dt.*f(vi+0.5*k2,rho(i));
    k4 = dt.*f(vi+k3,rho(i));
    v(:,i+1) = vi + k1/6 + k2/3 + k3/3 + k4/6 + noise(:,i);
end

v = [v; rho];

% Remove transients
t = t(round(trans)/dt:end);
v = v(:,round(trans)/dt:end);

end