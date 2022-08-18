function u = res_predict_linear_id_hybrid(A,Win,Wout,S,id,ph_id,rescaling,leak,bias,TT,noise)
% Operate reservoir in closed-loop for prediction
%
% Inputs:
%   A: reservoir adj matrix
%   Win: input coupling matrix
%   Wout: trained output weights
%   S: feature vector [R(n);v(n-1);1] at the end of resync or training
%   id: linear index
%   ph_id: control signal for physics-based model
%   rescaling: rescaling vector for reservoir inputs
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
    v = [u(:,i);id(i)];
    R = res_advance(A,Win,R,v,leak,bias);
    uh = physics_advance(v(1:input_size-1).*rescaling,ph_id(i))./rescaling;
    u(:,i+1) = Wout*[R;v;1;uh] + n;
end