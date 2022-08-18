function [A,e1] = adj_mat_2(num_nodes,degree,seed)
rng(seed,'twister');

rdensity = degree/num_nodes; 					
A = spfun(@rand_minus_one_to_one,sprand(num_nodes,num_nodes,rdensity));
e1 = max(abs(eigs(A))); 						% eigenvalue with largest magnitude
% Check e1
while isnan(e1)
    A = sprand(num_nodes,num_nodes,rdensity);
    e1 = max(abs(eigs(A)));
end

end