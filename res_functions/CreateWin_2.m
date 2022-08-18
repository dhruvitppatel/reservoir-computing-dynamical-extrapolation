function Win = CreateWin_2(num_nodes,input_size,seed)
rng(seed,'twister');

% Create Win where each node is fed only 1 of the inputs at random

k = randi([1 input_size],num_nodes,1);
Win = zeros(num_nodes,input_size);
for i = 1:num_nodes
    Win(i,k(i)) = 1-2*rand();
end

end
