function u = res_predict_linear_id(A,Win,Wout,S,id,leak,bias,TT,noise)
% Operate reservoir in closed-loop for prediction
%
% Inputs:
%   A: reservoir adj matrix
%   Win: input coupling matrix
%   Wout: trained output weights
%   S: feature vector [R(n);v(n-1);1] at the end of resync or training
%   id: linear index
%   leak: reservoir leakage parameter
%   bias: reservior activation bias 
%   TT: prediction length
%   noise: noise vector
%
% Outputs:
%   u: predicted time series
% 
% ----------------------------------------------------------------------- %

num_nodes = size(A,1);
input_size = size(Win,2);
R = S(1:num_nodes);
u = zeros(input_size-1,TT);
u(:,1) = Wout*S;

nl = size(noise,2);

for i = 1:TT-1
    n = noise(:,randi(nl));
    v = [u(:,i);id(i)] + [n;0];
    R = res_advance(A,Win,R,v,leak,bias);
    u(:,i+1) = Wout*[R;v;1];
end