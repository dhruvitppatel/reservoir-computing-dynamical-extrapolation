function u = res_predict(A,Win,Wout,S,leak,bias,TT)
% Operate reservoir in closed-loop for prediction
%
% Inputs:
%   A: reservoir adj matrix
%   Win: input coupling matrix
%   Wout: trained output weights
%   S: feature vector [R(n);v(n-1);1] at the end of resync or training
%   leak: reservoir leakage parameter
%   bias: reservior activation bias 
%   TT: prediction length
%
% Outputs:
%   u: predicted time series
% 
% ----------------------------------------------------------------------- %

num_nodes = size(A,1);
input_size = size(Win,2);
R = S(1:num_nodes);
u = zeros(input_size,TT);
u(:,1) = Wout*S;

for i = 1:TT-1
    R = res_advance(A,Win,R,u(:,i),leak,bias);
    u(:,i+1) = Wout*[R;u(:,i);1];
end