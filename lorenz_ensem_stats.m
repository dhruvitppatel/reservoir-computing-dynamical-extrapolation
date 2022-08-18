function [GAMMA,v4,cdf_act,cdf_pred,z_q] = lorenz_ensem_stats(X3,Y3,v4,q,delta_t)
% Calculate Gamma(t) climate metric for the Lorenz system given ensemble
% predictions.
%
% Inputs:
%   X3: z(t) of true system trajectories 
%   Y3: z(t) of ml/ml-hybrid predicted trajectories
%   v4: time vector
%   q: number of trajectories in ensemble
%   delta_t: number of time steps for interval over which to calculate the
%       climate error metric (\Gamma(t))
%
% Outputs:
%   GAMMA: Gamma(t)
%   v4: time vector for predicted interval
%   cdf_act: cumulative distribution of z_m for true trajectories
%   cdf_pred: cumulative distribution of z_m for predicted trajectories
%   z_q: query points for calculating CDFs
%
% ----------------------------------------------------------------------- %

v4 = v4(1,:);
id = [1:delta_t:(length(v4)-delta_t)]; 
leng = length(id);
GAMMA = zeros(1,leng);
cdf_act = zeros(leng,1000);
cdf_pred = cdf_act;

for i = 1:leng
    id1 = id(i);   
    id2 = id(i)+delta_t; 
    
    act_pk = zeros(q,100);
    pred_pk = act_pk;

    for j = 1:q
        act = findpeaks(X3(j,id1(1):id2(1)));
        pred = findpeaks(Y3(j,id1(1):id2(1)));
        act_pk(j,1:length(act)) = act;
        pred_pk(j,1:length(pred)) = pred;
    end
    
    cond = isempty(act_pk(:)) + isempty(pred_pk(:)) + isnan(sum(act_pk(:)) + sum(pred_pk(:))) + isinf(sum(act_pk(:)) + sum(pred_pk(:)));
    if cond > 0
        break;
    end
    
    % Wasserstein
    act_pk = act_pk(act_pk ~= 0);
    pred_pk = pred_pk(pred_pk ~= 0);
    ecdf_act = empirical_cdf(act_pk(:));
    ecdf_pred = empirical_cdf(pred_pk(:));
    z_q = linspace(-2,5,1000);
    cdf_act(i,:) = ecdf_act(z_q);
    cdf_pred(i,:) = ecdf_pred(z_q);
    norm = (max(act_pk,[],'all') - min(act_pk,[],'all'))/2;
    GAMMA(i) = (z_q(2)-z_q(1))*trapz(abs(cdf_act(i,:) - cdf_pred(i,:)))./norm;
end