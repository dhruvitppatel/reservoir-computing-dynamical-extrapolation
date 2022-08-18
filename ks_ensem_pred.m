function [t_cont, t_map, map_a, map_p] = ks_ensem_pred(ks,v0,nu,tp,hp,part_id)
% Ensemble prediction for the Kuramoto-Sivashinsky equation 
%
% Inputs:
%   ks: struct with parameters for solving the KS equation
%   v0: set of q initial conditions 
%   nu: time series of KS bifurcation parameter nu(t) (called kappa(t) in
%   paper)
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
% Returns:
%   t_cont: time vector over prediction interval
%   t_map: vector of time values corresponding to Poincare surface of
%       section piercings
%   map_act: Poincare maps from ensemble of true trajectories
%   map_pred: Poincare maps from ensemble of predicted trajectories
%
% --------------------- Unpack Parameters ------------------------- %

num_nodes = tp.num_nodes;
input_size = tp.input_size;
seed = tp.seed;

tt = tp.tt;
tw = tp.tw;
TT = tp.TT;
mp = tp.mp;
eps0 = tp.eps0;

q = tp.q;

% Hyperparameters
g1 = hp(1);     % Degree of reservoir adj. matrix
g2 = hp(2);     % Spectral radius of reservoir adj. matrix
g3 = hp(3);     % Input coupling strength for dynamical variables
g4 = hp(4);     % Input coupling strength for linear control signal
g5 = hp(5);     % Tikhonov regularization parameter
g6 = hp(6);     % Leakage
g7 = hp(7);     % Bias
g8 = hp(8);     % y-intercept of linear control signal
g9 = hp(9);     % slope of linear control signal

% ---------------------- Setup Network -----------------------%

[A,Win] = res_init_2(num_nodes,g1,g2,input_size,g3,g4,seed);


% -------------------- Main Loop ----------------------------%

for qq = 1:q
% tic; 

%------------- Generate KS traj ---------------%

[t,x,v] = get_ks(ks,nu,v0(:,qq));   

% Sample original data
samp = 1;
v = v(:,1:samp:end);
t = t(1:samp:end);

t = t(part_id:(part_id+tt+TT));
v = v(1:end-1,part_id:(part_id+tt+TT));

% Rescale Data
v = v./rms(v(:,1:tt),2);

% ------------- Set up Network and Training Data ---------------%

% Set index values and input vectors
p = [g9 g8];
id = 1:1:length(v);
id = polyval(p,id);

vtrain = [v(:,1:tt); id(1:tt)];
vtarget = vtrain(1:ks.N,:);

% ------------------------ Training ----------------------%

[Wout, S, S_avg] = res_train_linear_id(A,Win,vtrain,vtarget,tw,mp,eps0,g6,g7,g5);

% ------------ Determine Noise Distribution -------------%

vtemp = Wout*S_avg(:,tw:end);
samp_noise = vtarget(:,tw:end) - vtemp;

% --------------------- Predict -------------------------%

vprediction = [v(:,tt:end); id(tt:end)];
tprediction = t(:,tt:end);

u = res_predict_linear_id(A,Win,Wout,S,id(tt:end),g6,g7,TT,samp_noise);

% -------------------------- Save trajectories -------------------------- %
% toc;
[loc_act_temp,map_act_temp] = ks_map(tprediction,vprediction,1,40,2500);  
[loc_pred_temp,map_pred_temp] = ks_map(tprediction,u,1,40,2500);
% last arg (2500) is simply the length of filler array if prediction becomes
% unstable

loc_a{qq,:} = loc_act_temp;
loc_p{qq,:} = loc_pred_temp;
map_a{qq,:} = map_act_temp;
map_p{qq,:} = map_pred_temp;
t_cont{qq,:} = tprediction;

if length(loc_a{qq,:}) ~= 2500
    t_map{qq,:} = tprediction(loc_a{qq,:});
end
    
end

t_map = t_map{1,:};
t_cont = t_cont{1,:};