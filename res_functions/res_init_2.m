function [A,Win] = res_init_2(num_nodes,deg,spec_rad,input_size,sigma1,sigma2,seed)
% Function to initiate reservoir network with separate input coupling
% strength for system variables and dynamical variables
%
% Inputs:
%   num_nodes: number of reservoir nodes
%   deg: degree of reservoir adjacency matrix
%   spec_rad: spectral radius to rescale adj. matrix to
%   input_size: number of input channels to reservoir
%   sigma1: input coupling strength for sytem variable inputs
%   sigma1: input coupling strength for parameter/linear index
%   seed: random seed 
%
% Outputs:
%   A: reservoir adj. matrix - spectral radius is rescaled to spec_rad
%   Win: input coupling matrix
%
% -------------------------------------------------------------------- %

[A,e1] = adj_mat_2(num_nodes,deg,seed);
A = (spec_rad/e1).*A;

Win = sigma1.*CreateWin_2(num_nodes,input_size-1,seed);
% Win = [Win sigma2*ones(num_nodes,1)];
Win = [Win sigma2*(randi(2,num_nodes,1)-1)];
