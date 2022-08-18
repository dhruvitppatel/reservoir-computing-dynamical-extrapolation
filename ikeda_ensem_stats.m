function [GAMMA,v4,cdf_act,cdf_pred,z_q,ignore,ignore_id] = ikeda_ensem_stats(X,Y,v4,q,delta_t)
% Calculate Gamma(t) climate metric. Predicted trajectories which become
% unstable are ignored in calculating statistics.
%
% Inputs:
%   X: y-coordinate of true system trajectories
%   Y: y-coordinate of ml predicted trajectories
%   v4: time vector for predicted interval
%   q: number of trajectories in ensemble
%   delta_t: number of time steps for interval over which to calculate the
%       climate error metric (\Gamma(t))
%
% Outputs:
%   GAMMA: Gamma(t)
%   v4: time vector for predicted interval
%   cdf_act: cumulative distribution of z_m for true trajectories
%   cdf_pred: cumulative distribution of z_m for predicted trajectories
%   z_q: inquiry points for calculating CDFs
%   ignore, ignore_ids: number of predicted trajectories ignored because
%   they became unstable, and their corresponding IDs 
%
% ----------------------------------------------------------------------- %

v4 = v4(1,:);
id = [1:delta_t:(length(v4)-delta_t)]; 
leng = length(id);
GAMMA = zeros(1,leng);
z_q = linspace(-10,10,1001);
cdf_act = zeros(leng,1000);
cdf_pred = cdf_act;

ignore = 0;
ignore_id = [];

for i = 1:leng

    id1 = id(i);   
    id2 = id(i)+delta_t; 
    
    act_pk = zeros(q,delta_t+1);
    pred_pk = act_pk;

    for j = 1:q
        act_temp = X{j}(id1(1):id2(1));
        pred_temp = Y{j}(id1(1):id2(1));
        cond1 = isempty(act_temp) + isempty(pred_temp) + isnan(sum(act_temp) + sum(pred_temp)) + isinf(sum(act_temp) + sum(pred_temp));
        cond2 = sum([(mean(Y{j}(id1(1):id2(1))) < -5) (mean(Y{j}(id1(1):id2(1))) > 4)]);
        if cond1 + cond2 > 0
            if i == leng
                ignore = ignore + 1;
                ignore_id(ignore) = j;
            end
            continue;
        end
        act_pk(j,:) = act_temp;
        pred_pk(j,:) = pred_temp;
    end
    
    % Wasserstein
    act_pk = act_pk(act_pk ~= 0);
    pred_pk = pred_pk(pred_pk ~= 0);
    N1 = histcounts(act_pk(:),z_q,'Normalization','cdf');
    N2 = histcounts(pred_pk(:),z_q,'Normalization','cdf');
    cdf_act(i,:) = N1;
    cdf_pred(i,:) = N2;
    norm = (max(act_pk,[],'all') - min(act_pk,[],'all'))/2;
    GAMMA(i) = (z_q(2)-z_q(1))*trapz(abs(cdf_act(i,:) - cdf_pred(i,:)))./norm;
end