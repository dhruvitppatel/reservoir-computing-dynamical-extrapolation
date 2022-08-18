function [t,x,u] = get_ks(ks,nu,u0)
% Solve the time evolution of Kuramoto-Sivashinsky equation by ETDRK4 scheme
%
% u_t = -u*u_x - u_xx - nu(t)*u_xxxx, periodic BCs on [-pi,pi]
% see Trefethen, "Spectral Methods in MATLAB", SIAM 2000
% AK Kassam and LN Trefethen, July 2002
%
% Inputs:
%   ks: struct with parameters for system
%     dt: integration time step
%     t1: start time
%     trans: length (in time) of solution to remove as initial transient
%     t2: end time
%     N: number of spatial grid points   
%   nu: damping parameter as a vector of length(t1:dt:t2)
%   u0: initial condition 
%
% Returns:
%   t: time vector
%   x: spatial grid points
%   u: KS solution time series and nu(t)
% 
% ----------------------------------------------------------------------- %

% Unpack ks struct 
dt = ks.dt;
t1 = ks.t1;
trans = ks.trans;
t2 = ks.t2;
N = ks.N;
eps = ks.eps;

% Create time vector, spatial grid, initialize solution array
t = t1:dt:t2;
leng = length(t);
d = 2*pi;
x = (d*(1:N)/N)';
u = zeros(length(x),leng);

% FFT initial condition
v = fft(u0);

% Main Loop
for n = 1:leng-1
    
% Update parameters
% k = ([0:N/2-1 0 -N/2+1:-1]*(2*pi/d))';                                   % wave numbers
k = ([0:N/2-1 -N/2:-1]*(2*pi/d))';
g = -0.5i*k;
L = k.^2 - nu(n)*k.^4;                                                            % Fourier multipliers
E = exp(dt*L); E2 = exp(dt*L/2);
M = 16;                                                                     % no. of points for complex means
r = exp(1i*pi*((1:M)-.5)/M);                                                % roots of unity
LR = dt*L(:,ones(M,1)) + r(ones(N,1),:);
Q = dt*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = dt*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
f2 = dt*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = dt*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop
Nv = g.*fft(real(ifft(v)).^2);
a = E2.*v + Q.*Nv;
Na = g.*fft(real(ifft(a)).^2);
b = E2.*v + Q.*Na;
Nb = g.*fft(real(ifft(b)).^2);
c = E2.*a + Q.*(2*Nb-Nv);
Nc = g.*fft(real(ifft(c)).^2);
v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;

u(:,n+1) = real(ifft(v));

% Dynamical noise
noise = eps.*(-1 + 2*rand(N,1));

v = fft(u(:,n+1) + noise);

end

t = t(round(trans/dt):end);
u = [u(:,round(trans/dt):end); nu(round(trans/dt):end)];