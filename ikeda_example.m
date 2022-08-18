% Example: Ikeda map interior crisis transition from a smaller chaotic 
%   attractor to a larger chaotic attractor. Script performs ensemble prediction
%   and generates climate error metric (plotted) over prediction window.
%
% -------------------- Helper functions -------------------- %

addpath('res_functions')

% ------------------ Generate time series --------------------- %

% time and Ikeda parameters
eta_0 = 6.75;
eta_1 = 0.5;   
eta_m = eta_0 + 0.0001; 
eta_p = 7.75; 
tau = 20000;
t_m = round(tau*log((eta_m-eta_0)/eta_1));
t_p = round(tau*log((eta_p-eta_0)/eta_1));
t = t_m:1:t_p;
eta = eta_0 + eta_1*exp(t./tau);
a = 0.85;
b = 0.9;
kappa = 0.4;

% Dynamical noise
eps = 0.015*[1; 1];             % Adjust the strength of dynamical noise here

% Ikeda initial condition
rng(10,'twister');
v0 = rand(2,1);

% ik struct
ik.t1 = t(1);
ik.a = a;
ik.b = b;
ik.kappa = kappa;
ik.trans  = 100;
ik.eps = eps;

% Trajectory
[t,v] = get_ikeda(ik,eta,v0);
v = v(1:2,:);

% Training data length
tsl = length(t(t<0)); 

% ---------------------- Set Parameters -------------------- %

num_nodes = 1000;                       % Number of reservoir nodes
input_size = 3;                         % Number of inputs to reservoir
seed = 1;                               % Seed for random generation of reservoir matrices

tt = 20000;                             % Number of reservoir steps for training
tw = 100;                               % Number of reservoir steps to discard as transient
mp = 10;                                % Number of passes during training
eps0 = 5e-2;                            % Amplitude of added observational noise during training

q = 4000;                               % Number of initial conditions in ensemble
delta_t = 500;                          % Number of time steps for interval over which to calculate \Gamma (climate error metric)

% Training/Testing parameters
tp.num_nodes = num_nodes;
tp.input_size = input_size;
tp.seed = seed;
tp.tt = tt;
tp.tw = tw;
tp.TT = length(v) - tsl - 1;
tp.mp = mp;                                                                
tp.eps0 = eps0;                                                        
tp.q = q;                                                                
tp.delta_t = delta_t;                                                  

% Random initial conditions for \Gamma Calculation
v0 = rand(2,tp.q);

% -------------- Set hyperparameters ---------------- %

deg = 3;                                    % Degree of reservoir adj. matrix 
spec_rad = 0.25;                            % Spectral radius of reservoir adj. matrix
input_coupling_1 = 2;                       % Input coupling strength for dynamical variables
input_coupling_2 = 1;                       % Input coupling strength for linear control signal
reg = 7.7426e-08;                           % Tikhonov regularization
leak = 1;                                   % Leakage
bias = 0;                                   % Reservoir activation bias
b = 1;                                      % y-intercept of linear control signal
a = 4.6416e-07;                             % Slope of linear control signal

hp_vector = [deg, spec_rad, input_coupling_1, input_coupling_2, ...
    reg, leak, bias, b, a]; 

% ----------------- Calculate \Gamma ---------------- %

start_id = tsl-tp.tt;

[X1,X2,Y1,Y2,Z1,Z2] = ikeda_ensem_pred(ik,v0,eta,tp,hp_vector,start_id);
t_pred = t(start_id+tp.tt-1:end);

[GAMMA,v4,cdf_act,cdf_pred,z_q, ...
    ignore,ignore_id] = ikeda_ensem_stats(X2,Y2,t_pred,tp.q,tp.delta_t);

% Plot results
t_GAMMA = t_pred(1:delta_t:end-delta_t);
plot(t_GAMMA,GAMMA,'-ok')
grid on