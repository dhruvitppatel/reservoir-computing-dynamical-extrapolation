function S = res_resync(A,Win,v,leak,bias)
% Operate the reservoir in open-loop to resynchronize to a driving signal
%
% Inputs:
%   A: reservoir adj matrix
%   Win: input coupling matrix
%   v: driving signal of same length as resynchronization period
%   leak: leakage
%   bias: constant bias
%   
% Outputs:
%   S: feature vector, [R(:,trs+1);v(:,trs):1]
%
% ----------------------------------------------------------------------- %

num_nodes = size(A,1);
input_size = size(Win,2);
trs = size(v,2);
R = zeros(num_nodes,1);

for i = 1:trs
    R = res_advance(A,Win,R,v(:,i),leak,bias);
end

S = [R;v(:,trs);1];