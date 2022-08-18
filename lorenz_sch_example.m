% Example: Lorenz model subcritical Hopf bifurcation from fixed-point 
%   to chaotic motion. Includes ML-only and hybrid cases (comment/uncomment
%   appropriate lines at the end of the code). Script performs ensemble prediction
%   and generates climate error metric (plotted) over prediction window.
%       
% -------------------- Helper functions -------------------- %

addpath('res_functions')

% ------------------ Generate time series --------------------- %

% time and Lorenz parameters 
rho0 = 15;  
rho1 = 7.5; 
rhom = rho0 + 0.001;
rhop = 40; 
tau = 10^2;
tm = round(tau*log((rhom-rho0)/rho1));
tpl = round(tau*log((rhop-rho0)/rho1));
dt = 0.01;
t = tm:dt:tpl;
rho = rho0 + rho1*exp(t./tau);
beta = 8/3;
sigma = 10;

% initial condition
rng(10,'twister')
v0 = zeros(3,1);
xa = -20; xb = 20;  
v0(1) = xa + (xb - xa)*rand();
ya = -30; yb = 30; 
v0(2) = ya + (yb - ya)*rand();
za = 100; zb = 300;  
v0(3) = za + (zb - za)*rand();

% Dynamical noise
eps = 2000*[0.011; 0.011; 0.378]*1e-4;      % Adjust strength of dynamical noise here for [x, y, z] components

% ls struct
ls.dt = dt;
ls.t1 = t(1);
ls.trans = 50;
ls.t2 = t(end);
ls.beta = beta;
ls.sigma = sigma;
ls.eps = eps;

% Generate data
[t,v] = get_lorenz(ls,rho,v0);
v = v(1:3,:);

% Sample original data
samp = 1;
v = v(:,1:samp:end); 
t = t(1:samp:end);

% Training data
train_length = length(t(t<=0));

% ------------------- Set Reservoir Parameters ------------------- %

num_nodes = 2000;                           % Number of reservoir nodes
input_size = 4;                             % Number of inputs to reservoir
seed = 1;                                   % Seed for random generation of reservoir matrices

tt = 30000;                                 % Number of reservoir steps for training
tw = 100;                                   % Number of reservoir steps to discard as transient
mp = 1;                                    % Number of passes during training
eps0 = 0*1e-5;                                % Amplitude of added observational noise during training

q = 2000;                                   % Number of initial conditions in ensemble
delta_t = 500;                              % Number of time steps for interval over which to calculate \Gamma (climate error metric)

% Training/Testing parameters
tp.num_nodes = num_nodes;
tp.input_size = input_size;
tp.seed = seed;
tp.tt = tt;
tp.tw = tw;
tp.TT = length(v) - train_length - 1; 
tp.mp = mp;                                  
tp.eps0 = eps0;                           
tp.q = q;                                 
tp.delta_t = delta_t;                           
        
% Random initial conditions for ensemble prediction
v0 = [xa; ya; za] + ...
    ([xb-xa; yb-ya; zb-za].*rand(3,tp.q));  

% -------------- Set hyperparameters ---------------- %

deg = 3;                                    % Degree of reservoir adj. matrix 
spec_rad = 0.25;                            % Spectral radius of reservoir adj. matrix
input_coupling = 0.1; %0.25                      % Input coupling strength
reg = 1e-8;  %1e-13                               % Tikhonov regularization
leak = 1;                                   % Leakage
bias = 0;                                   % Reservoir activation bias
b = 1;                                      % y-intercept of linear control signal
a = 1.2915e-05;                              % Slope of linear control signal

hp_vector = [deg, spec_rad, input_coupling, reg, leak, ...
    bias, b, a]; 

% ----------------- Ensembel Prediction ---------------- %

start_id = train_length-tp.tt; 

% % ML-Only (Make sure hyperparameters above are correct)
% % Uncomment the next command if running ML-Only case
% [X3,Y3,v4,mean_var_z] = lorenz_mlonly_ensem_pred(ls,v0,rho, ...
%                                             tp,hp_vector,start_id);
                                        
% Hybrid (Make sure hyperparameters above are correct)
% Uncomment the next command if running hybrid case
[X3,Y3,v4,mean_var_z] = lorenz_hybrid_ensem_pred(ls,v0,rho, ...
                                            tp,hp_vector,start_id);

[GAMMA,t_GAMMA, cdf_act, ...
    cdf_pred, z_q] = lorenz_ensem_stats(X3,Y3,v4(1,:), ...
                                            tp.q,tp.delta_t);
                                        
% Plot results
t_GAMMA = t_GAMMA(1:tp.delta_t:end-tp.delta_t);
plot(t_GAMMA,GAMMA,'-ok')
grid on