% SUMMARY:  p(xn|zn) of Gaussian, size: N*p
% AUTHOR:   QIUQIANG KONG
% Created:  17-11-2015
% Modified: - 
% -----------------------------------------------------------
% input:
%   X       data, size: N*p
%   phi:    para struct
%      mu     size: p*Q
%      Sigma  size: p*p*Q
% output:
%   p(xn|zn)
% ===========================================================
function p_xn_given_zn = Gauss_p_xn_given_zn(X, phi)
[N,p] = size(X);
[p,Q] = size(phi.mu);
p_xn_given_zn = zeros(N,Q);
for i1 = 1:Q
    p_xn_given_zn(:,i1) = mvnpdf(X, phi.mu(:,i1)', phi.Sigma(:,:,i1));
end
end