function [A,Win] = res_init(num_nodes,deg,spec_rad,input_size,sigma,seed)
% Function to initiate reservoir network
%
% Inputs:
%   num_nodes: number of reservoir nodes
%   deg: degree of reservoir adjacency matrix
%   spec_rad: spectral radius to rescale adj. matrix to
%   input_size: number of input channels to reservoir
%   sigma: input coupling strength
%   seed: random seed 
%
% Outputs:
%   A: reservoir adj. matrix - spectral radius is rescaled to spec_rad
%   Win: input coupling matrix
%
% -------------------------------------------------------------------- %

[A,e1] = adj_mat_2(num_nodes,deg,seed);
A = (spec_rad/e1).*A;

Win = sigma.*CreateWin_2(num_nodes,input_size,seed);

