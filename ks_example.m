% Example: Kuramoto-Sivashinsky equation saddle-node bifurcation from
%   period-2 to chaotic motion. Script performs ensemble prediction
%   and generates climate error metric (plotted) over prediction window.
%
% -------------------- Helper functions -------------------- %

addpath('res_functions')

% -------------------- Generate time series ------------------- %

% time and KS parameters
nu_0 = 0.0753;
nu_1 = 0.0034; 
nu_m = nu_0 + 0.0003; 
nu_p = 0.0895; 
tau = 10^3 * nu_0;
t_m = round(tau*log((nu_m-nu_0)/nu_1)); 
t_p = round(tau*log((nu_p-nu_0)/nu_1));
dt = 0.1 * nu_0; 
t = t_m:dt:t_p;
nu = nu_0 + nu_1*exp(t./tau); 

% ks struct
ks.dt = dt;
ks.t1 = t_m;
ks.t2 = t_p;
ks.trans = 100 * nu_0;
ks.N = 64;

% Dynamical noise 
eps = 1e-4*ones(ks.N,1);      % Adjust strength of dynamical noise here
ks.eps = eps;

% initial condition
rng(10,'twister');
v0 = 0.6*(-1+2*rand(ks.N,1)) * (1/nu_0);

% Generate data
[t,x,v] = get_ks(ks,nu,v0);

% Sample original data
samp = 1;
v = v(1:end-1,1:samp:end); 
t = t(1:samp:end);

% Training data length
tsl = length(t(t<0)); 

% ---------------------- Set Reservoir Parameters -------------------- %

num_nodes = 3000;                       % Number of reservoir nodes
input_size = ks.N + 1;                  % Number of inputs to reservoir
seed = 1;                               % Seed for random generation of reservoir matrices

tt = round(60/(ks.dt*samp));            % Number of reservoir steps for training
tw = 50;                                % Number of reservoir steps to discard as transient
mp = 1;                                 % Number of passes during training
eps0 = 0;                               % Amplitude of added observational noise during training

q = 2000;                               % Number of initial conditions in ensemble
delta_t = 1000;                         % Number of time steps for interval over which to calculate \Gamma (climate error metric)

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
v0 = -1 + 2*rand(ks.N,1) + 0.5*(-1 + 2*rand(ks.N,tp.q));

% -------------- Set hyperparameters ---------------- %

deg = 3;                                    % Degree of reservoir adj. matrix 
spec_rad = 1;                               % Spectral radius of reservoir adj. matrix
input_coupling_1 = 1;                       % Input coupling strength for dynamical variables
input_coupling_2 = 0.75;                    % Input coupling strength for linear control signal
reg = 7.7426e-10;                           % Tikhonov regularization
leak = 0.5;                                 % Leakage
bias = 1;                                   % Reservoir activation bias
b = 1;                                      % y-intercept of linear control signal
a = 1e-5;                                   % Slope of linear control signal

hp_vector = [deg, spec_rad, input_coupling_1, input_coupling_2, ...
    reg, leak, bias, b, a]; 

% ----------------- Ensemble Prediction ---------------- %

start_id = tsl-tp.tt;
[t_cont,t_map,map_act,map_pred] = ks_ensem_pred(ks,v0,nu,tp, ...
                                        hp_vector,start_id);
[GAMMA,t_cont,cdf_act,cdf_pred,z_q] = ks_ensem_stats(t_cont,t_map, ...
                                        map_act,map_pred,tp.q,tp.delta_t);

% Plot results
t = t_cont(1:tp.delta_t:end-tp.delta_t);
plot(t,GAMMA(1:end-1),'-ok')
grid on
