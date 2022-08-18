function [X3,Y3,v4,mean_var_z] = lorenz_hybrid_ensem_pred(ls,v0,rho,tp,hp,part_id)
% Ensemble prediction for the Lorenz system using the hybrid method 
%
% Inputs:
%   ls: struct with parameters for solving the Lorenz equation
%   v0: set of q initial conditions 
%   rho: time series of Lorenz bifurcation parameter rho(t)
%   tp: struct with fields
%       num_nodes: number of reservoir nodes
%       input_size: number of inputs to reservoir
%       seed: seed for random number generation
%       tw: transient reservoir length to remove
%       tt: training length
%       TT: testing length
%       mp: number of passes in training
%       eps0: noise level in noisy training
%       q: number of initial conditions in ensemble
%   hp: hyperparameter vector
%   part_id: index at which to start training
%
% Outputs:
%   X3: z(t) of true trajectory
%   Y3: z(t) of predicted trajectory
%   v4: time 
%
% ------------------------- Set Parameters ------------------------------ %

num_nodes = tp.num_nodes;
input_size = tp.input_size;
seed = tp.seed;

tt = tp.tt;
tw = tp.tw;
TT = tp.TT;
mp = tp.mp;
eps0 = tp.eps0;

q = tp.q;
delta_t = tp.delta_t;

% Hyperparameters
g1 = hp(1);
g2 = hp(2);
g3 = hp(3);
g4 = hp(4);
g5 = hp(5);
g6 = hp(6);
g7 = hp(7);
g8 = hp(8);

Y3 = zeros(q,TT);
X3 = zeros(q,TT);
v4 = zeros(q,TT);
mean_var_z = zeros(q,1);

% ---------------------- Setup Network -----------------------%

[A,Win] = res_init(num_nodes,g1,g2,input_size,g3,seed);

% -------------------- Main Loop ----------------------------%

% Make the for loop into a parfor loop for parallel computation
for qq = 1:q
% tic; 

% ------------- Generate Lorenz Data ---------------- %
rho_t = rho;

[t,v] = get_lorenz(ls,rho_t,v0(:,qq));

% Sample
samp = 1;
t = t(1:samp:end);
v = v(:,1:samp:end);
rho_t = v(4,:);

t = t(part_id:(part_id+tt+TT));
v = v(1:3,part_id:(part_id+tt+TT));
rho_t = rho_t(part_id:(part_id+tt+TT));

% Rescale Data
rescaling = rms(v(:,1:tt),2);
v = v./rescaling;

% ------------- Set up Network and Training Data ---------------%

% Set index values and input vectors
p = [g8 g7];
id = 1:1:length(v);
id = polyval(p,id);

vtrain = [v(:,1:tt); id(1:tt)];
vtarget = vtrain(1:3,:);

% ------------------------ Training ----------------------%

[Wout, S, S_avg] = res_train_linear_id_hybrid(A,Win,vtrain,vtarget,rho_t(1:tt),rescaling,tw,mp,eps0,g5,g6,g4); 

% ------------ Determine Noise Distribution -------------%

vtemp = Wout*S_avg(:,tw:end);
samp_noise = vtarget(:,tw:end) - vtemp;

% --------------------- Predict -------------------------%

vprediction = [v(:,tt:end); id(tt:end)];
tprediction = t(:,tt:end);

u = res_predict_linear_id_hybrid(A,Win,Wout,S,id(tt:end),rho_t(tt:end),rescaling,g5,g6,TT,samp_noise);

% -------------------------- Save trajectories -------------------------- %

Y3(qq,:) = u(3,:);
X3(qq,:) = vprediction(3,1:TT);
v4(qq,:) = tprediction(1:TT);

mean_var_z(qq) = mean(movvar(v(3,1:tt),[delta_t 0]));

% toc;
end