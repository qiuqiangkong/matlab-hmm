% SUMMARY:  ln p(xn|zn) of GMM, size: N*p
%           Use log to avoid overflow
% AUTHOR:   QIUQIANG KONG
% Created:  21-11-2015
% Modified: - 
% -----------------------------------------------------------
% input:
%   X       data, size: N*p
%   phi:    para struct
%      B      size: Q*M
%      mu     size: p*Q*M
%      Sigma  size: p*p*Q*M
% output:
%   ln p(xn|zn)
% ===========================================================
function logp_xn_given_zn = Gmm_logp_xn_given_zn(X, phi)
[N,p] = size(X);
[M,Q] = size(phi.B);

logp_xn_given_zn = zeros(N,Q);
for q = 1:Q
    logp_xn_given_zn(:,q) = LogGmmpdf(X, phi.B(:,q)', phi.mu(:,:,q), phi.Sigma(:,:,:,q));
end
end 

function logp_X = LogGmmpdf(X, prior, mu, Sigma)
    N = size(X,1);          % num of data
    [p,M] = size(mu);       % mix num & feature dim
    Tmp = zeros(N, M);
    for m = 1:M
        x_minus_mu = bsxfun(@minus, X, mu(:,m)');
        Tmp(:, m) = log(prior(m)) - 0.5*p*log(2*pi) - 0.5*log(det(Sigma(:,:,m))) - 0.5*sum( x_minus_mu * inv(Sigma(:,:,m)) .* x_minus_mu, 2 );
    end

    log( sum( exp( bsxfun(@minus, Tmp, max(Tmp,[],2) ) ) ) );
    logp_X = log( sum( exp( bsxfun(@minus, Tmp, max(Tmp,[],2) ) ), 2 ) ) + max(Tmp,[],2);
end