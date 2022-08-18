function [t,v] = get_ikeda(ik,eta,v0)
% Generate time series for the Ikeda Map with potentially
%   time-dependent eta:
%   x_n+1 = a + b*(x_n*cos(t_n) - y_n*sin(t_n))
%   y_n+1 = b*(x_n*sin(t_n) + y_n*cos(t_n))
%   theta_n = kappa - eta_n/(1 + x_n^2 + y_n^2)
%
% Inputs:
%   eta: system parameter vector
%   ik: struct with fields
%       a, b, kappa - system parameters
%       trans: num of initial steps to discard as transient
%       eps: [eps0x; eps0y] noise strength
%       t1: "start time"
%   v0: [x0; y0] initial condition
%
% Returns:
%   t: vector of steps (time)
%   v: [x_n; y_n; eta_n] time series
%
% ---------- Unpack struct parameters ---------- %

t1 = ik.t1;
a = ik.a;
b = ik.b;
kappa = ik.kappa;
trans = ik.trans;
eps = ik.eps;

% ---------- Generate trajectory ---------- %

N = length(eta);
t = t1:1:(t1+N-1);
x = zeros(1,N);
y = zeros(1,N);
x(1) = v0(1);
y(1) = v0(2);

noise_x = eps(1)*(-1 + 2*rand(N,1));
noise_y = eps(2)*(-1 + 2*rand(N,1));

for i = 1:N-1
    theta = kappa - eta(i)/(1 + x(i).^2 + y(i).^2);
    x(i+1) = a + b*(x(i)*cos(theta) - y(i)*sin(theta)) + noise_x(i);
    y(i+1) = b*(x(i)*sin(theta) + y(i)*cos(theta)) + noise_y(i);
end

v = [x; y];

t = t(trans+1:end);
v = [v(:,trans+1:end); eta(trans+1:end)];