function [X1, X2, Y1, Y2, Z1, Z2] = ikeda_ensem_pred(ik,v0,eta,tp,hp,part_id)
% Ensemble prediction for the Ikeda map
%
% Inputs:
%   ik: struct with parameters for solving Ikeda map
%   v0: set of q initial conditions
%   eta: time series of Ikeda map bifurcation parameter eta(t)
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
%   X1, Y1: true and predicted x-coordinate
%   X2, Y2: true and predicted y-coordinate
%   Z1, Z2: training trajectory x- and y- coordinates
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

X1 = cell(q,1);
X2 = X1;
Y1 = X1;
Y2 = X1;
Z1 = X1;
Z2 = X1;

for qq = 1:q
% tic; 

%------------- Generate system traj ---------------%

[t,v] = get_ikeda(ik,eta,v0(:,qq));   

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
vtarget = vtrain(1:2,:);

% ------------------------ Training ----------------------%

[Wout, S, S_avg] = res_train_linear_id_sq(A,Win,vtrain,vtarget,tw,mp,eps0,g6,g7,g5);

% ------------ Determine Noise Distribution -------------%

vtemp = Wout*S_avg(:,tw:end);
samp_noise = vtarget(:,tw:end) - vtemp;

% --------------------- Predict -------------------------%

vprediction = [v(:,tt:end); id(tt:end)];
tprediction = t(:,tt:end);

u = res_predict_linear_id_sq(A,Win,Wout,S,id(tt:end),g6,g7,TT,samp_noise);

% -------------------------- Save trajectories -------------------------- %
% toc;

X1{qq,:} = vprediction(1,:);
X2{qq,:} = vprediction(2,:);
Y1{qq,:} = u(1,:);
Y2{qq,:} = u(2,:);

Z1{qq,:} = v(1,1:tt);
Z2{qq,:} = v(2,1:tt);

end