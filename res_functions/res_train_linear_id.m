function [Wout, S, S_store] = res_train_linear_id(A,Win,v,vT,trans,q,eps0,leak,bias,reg)
% Train a reservoir using time series data
%
% Inputs:
%   A: reservoir adj matrix
%   Win: input coupling matrix
%   v: driving signal time series
%   vT: target signal to train to
%   trans: number of reservoir steps to discard as transient
%   q: number of passes in open-loop (for noisy)
%   eps0: noise strength for noisy training
%   leak: leakage 
%   bias: constant bias in res. activation
%   reg: Tikhonov regularization parameter
%
% Outputs:
%   Wout: trained output weight matrix
%   S: feature vector of last state, [R(:,tt);v(:,tt-1);1]
%   S_store: feature vectors for all states [num_nodes+dim(v)+1]x[tt]
%
% ----------------------------------------------------------------------- %

% Set parameters
num_nodes = size(A,1);
input_size = size(Win,2);
tt = size(v,2);
vR = zeros(size(vT,1),num_nodes+input_size+1);
RR = zeros(num_nodes+input_size+1);
Savg = zeros(num_nodes+input_size+1,tt);

for i = 1:q
    R = zeros(num_nodes,tt);
    S = zeros(num_nodes+input_size+1,tt);
    
    for j = 1:tt-1
        noise = eps0.*(1 - 2*rand(input_size-1,1));
        u = v(:,j) + [noise;0];
        R(:,j+1) = res_advance(A,Win,R(:,j),u,leak,bias);
        S(:,j+1) = [R(:,j+1);v(:,j);1];
    end
    
    vR = vR + vT(:,trans:end)*S(:,trans:end)';
    RR = RR + S(:,trans:end)*S(:,trans:end)';
    
    Savg = (1/i)*((i-1)*Savg + S);
end

S_store = Savg;
S = S(:,tt);
Wout = (1/(q*(tt-trans)))*vR*pinv((1/(q*(tt-trans)))*RR + reg*eye(size(RR)));
% Wout = (1/q)*vR*pinv((1/q)*RR + reg*eye(size(RR)));
% Wout = mrdivide((1/(q*(tt-trans)))*vR,(1/(q*(tt-trans)))*RR + reg*eye(size(RR)));