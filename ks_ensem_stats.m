function [GAMMA,t_cont,cdf_act,cdf_pred,z_q] = ks_ensem_stats(t_cont,t_map,map_a,map_p,q,delta_t)
% Calculate Gamma(t) 
%
% Inputs:
%   t_cont: time vector over prediction interval
%   t_map: vector of time values corresponding to Poincare map of a typical
%       actual trajectory.
%   map_a: Poincare maps for ensemble of actual trajectories
%   map_p: Poincare maps for ensemble of predicted trajectories
%   q: number of trajectories in ensemble
%   delta_t: number of time steps for interval over which to calculate the
%       climate error metric (\Gamma(t))
%
% Outputs:
%   GAMMA: Gamma(t) climate metric
%   t_cont: time vector over prediction interval
%   cdf_act: cumulative distribution of Poincare map points for true 
%       trajectories
%   cdf_pred: cumulative distribution of Poincare map points for predicted  
%       trajectories
%   z_q: query points for calculating CDFs
%
% ----------------------------------------------------------------------- %

id = 1:delta_t:(length(t_cont)); 
leng = length(id);
GAMMA = zeros(1,leng);
z_q = linspace(-3,3,1000);
cdf_act = zeros(leng, length(z_q));
cdf_pred = zeros(leng,length(z_q));

for i = 1:leng-1
    
    id1 = find(t_map >= t_cont(id(i)));
    id2 = find(t_map >= t_cont(id(i+1)));
    
    map_act = cell(q,1);
    map_pred = cell(q,1);
%     disp(i)
    for j = 1:q
        map_aa = [map_a{j,:} zeros(1,1000)];
        map_pp = [map_p{j,:} zeros(1,1000)];
%         disp(j)
        v = map_aa(1,id1(1):id2(1)); 
        u = map_pp(1,id1(1):id2(1)); 
        map_act{j} = v(v~=0);
        map_pred{j} = u(u~=0);
    end
    
    map_act_vec = [map_act{:}];
    map_pred_vec = [map_pred{:}];
    ecdf_act = empirical_cdf(map_act_vec);
    ecdf_pred = empirical_cdf(map_pred_vec);
    cdf_act(i,:)  = ecdf_act(z_q);
    cdf_pred(i,:) = ecdf_pred(z_q);
    norm = (max(map_act_vec) - min(map_act_vec))/2;
    GAMMA(i) = abs(z_q(2)-z_q(1))*trapz(abs(cdf_act(i,:) - cdf_pred(i,:)))./norm;
    
end