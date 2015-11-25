% SUMMARY:  Log Forward Backward, to solve overflow problem
% AUTHOR:   QIUQIANG KONG
% Created:  18-11-2015
% Modified: 19-11-2015 modify max to sum (exact solution)
%           20-11-2015 add a ceiling for logbeta
%           21-11-2015 use log instead of all to avoid overfit
%           25-11-2015 Add annotation
% -----------------------------------------------------------
% input:
%   p_xn_given_zn  p(xn|zn), size: N*Q
%   p_start        p(z1), size: Q
%   A              p(zn|zn-1), size: Q*Q
% output:
%   loggamma       ln p(zn|X), size: N*Q
%   logksi         ln p(zn,zn-1|X), size: N*Q*Q
%   loglik         ln p(X), to monitor convergence
% ===========================================================
function [loggamma, logksi, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A)
    % reserve space
    [N,Q] = size(logp_xn_given_zn);
    logalpha = zeros(N,Q);
    logbeta = zeros(N,Q);
    logc = zeros(N,1);
    loggamma = zeros(N,Q);
    logksi = zeros(N,Q,Q);
    
    % init log alpha(z1), log beta(zN), c(1)
    tmp = logp_xn_given_zn(1,:) + log(p_start);
    logc(1) = log( sum( exp( tmp - max(tmp) ) ) ) + max(tmp);
    logalpha(1,:) = -logc(1) + logp_xn_given_zn(1,:) + log(p_start);
    logbeta(N,:) = 0;

    % calculate logalpha, c
    for n = 2:N
        tmp = bsxfun(@plus, bsxfun(@plus, log(A), logalpha(n-1,:)'), logp_xn_given_zn(n,:));
        logc(n) = log ( sum( sum ( exp ( tmp - max(tmp(:)) ) ) ) ) + max(tmp(:));
        for q = 1:Q
            tmp2 = logalpha(n-1,:) + log(A(:,q)');
            if (isinf(max(tmp2)))
                logalpha(n,q) = -inf;
            else
                logalpha(n,q) = -logc(n) + logp_xn_given_zn(n,q) + log( sum( exp( tmp2 - max(tmp2) ) ) ) + max(tmp2);
            end
        end
    end

    % calculate logbeta
    for n = N-1:-1:1
        for q = 1:Q
            tmp = logbeta(n+1,:) + logp_xn_given_zn(n+1,:) + log(A(q,:));
            logbeta(n,q) = -logc(n+1) + log( sum( exp( tmp - max(tmp) ) ) ) + max(tmp);
        end
    end

    % calculate loggamma
    loggamma = logalpha + logbeta;
    
    % calculate logksi
    for n = 2:N
        logksi(n,:,:) = -logc(n) + bsxfun(@plus, bsxfun(@plus, log(A), logalpha(n-1,:)'), logp_xn_given_zn(n,:) + logbeta(n,:));
    end
    logksi(1,:,:) = [];
    
    % calculate likelihood
    loglik = sum(logc);
end