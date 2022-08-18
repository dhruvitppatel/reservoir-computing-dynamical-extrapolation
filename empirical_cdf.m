function f = empirical_cdf(x)
% Given a sequence construct an empirical cdf as per paper by 
% Perez-Cruz 2008 IEEE. 
%
% Output: function handle for a piece-wise continuous (linearly
% interpolated) extension of empirical cdf
%
% ----------------------------------------------------------------------- %

n = length(x);

x = sort(x,'ascend');
[x_u,~,ic] = unique(x);
x_counts = accumarray(ic,1);

y = (1/n)*(cumsum(x_counts) - 0.5);

f =@(x_query) interp1_with_clipping(x_u,y,x_query);