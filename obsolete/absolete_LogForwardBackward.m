% SUMMARY:  Log Forward Backward, to solve overflow problem
% AUTHOR:   QIUQIANG KONG
% Created:  18-11-2015
% Modified: 19-11-2015 modify max to sum (exact solution)
%           20-11-2015 add a ceiling for logbeta
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
function [loggamma, logksi, loglik] = LogForwardBackward(p_xn_given_zn, p_start, A)
    % reserve space
    [N,Q] = size(p_xn_given_zn);
    logalpha = zeros(N,Q);
    logbeta = zeros(N,Q);
    c = zeros(N,1);
    loggamma = zeros(N,Q);
    logksi = -inf(N,Q,Q);
    
    % init log alpha(z1), log beta(zN), c(1)
    p_x1_z1 = p_xn_given_zn(1,:) .* p_start;    % p(x1,z1)
    p_z1_given_x1 = p_x1_z1 / sum(p_x1_z1);
    logalpha(1,:) = log(p_z1_given_x1);
    c(1) = sum(p_x1_z1);                        % c1 = p(x1) = sum(p(x1,z1))
    logbeta(N,:) = 0;

    % calculate logalpha, c
    for n = 2:N
        c(n) = sum(p_xn_given_zn(n,:) .* (exp(logalpha(n-1,:)) * A));
        logalpha(n,:) = -log(c(n)) + log(p_xn_given_zn(n,:)) + log(exp(logalpha(n-1,:)) * A);
    end

    % calculate logbeta
    for n = N-1:-1:1
        tmp = -log(c(n+1)) + log( p_xn_given_zn(n+1,:) .* exp(logbeta(n+1,:)) * A' );
        tmp(find(tmp>log(realmax)-1)) = log(realmax) - 1;   % prevent from up overflow
        logbeta(n,:) = tmp;
    end

    % calculate loggamma
    loggamma = logalpha + logbeta;
    
    % calculate logksi
    for n = 2:N
        logksi(n,:,:) = -log(c(n)) + bsxfun(@plus, bsxfun(@plus, log(A), logalpha(n-1,:)'), log(p_xn_given_zn(n,:)) + logbeta(n,:));
    end
    logksi(1,:,:) = [];
    
    % calculate likelihood
    loglik = sum(log(c));
    logksi
    pause
end