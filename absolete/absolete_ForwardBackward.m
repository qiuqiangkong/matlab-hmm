% THIS FILE IS ABSOLETE NOW
% SUMMARY:  Calculate gamma=p(zn|X), ksi=p(zn,zn-1|X) using forwardbackward
%           algorithm. This used conditional dependence of PGM. Decompose
%           gamma as alpha.*beta
% AUTHOR:   QIUQIANG KONG
% Created:  30-09-2015
% modified: 14-11-2015 Improved using scaling. Ref: PRML
% Modified: 17-11-2015 Add annotation
%           25-11-2015 This file is absolete and replaced by LogForwardBackward(...)
% -----------------------------------------------------------
% input:
%   p_xn_given_zn  p(xn|zn), size: N*Q
%   p_start        p(z1), size: Q
%   A              p(zn|zn-1), size: Q*Q
% output:
%   gamma          p(zn|X), size: N*Q
%   ksi            p(zn,zn-1|X), size: N*Q*Q
%   loglik         ln p(X), to monitor convergence
% ===========================================================
function [gamma, ksi, loglik] = ForwardBackward(p_xn_given_zn, p_start, A)
    % reserve space
    [N,Q] = size(p_xn_given_zn);
    alpha = zeros(N,Q);
    beta = zeros(N,Q);
    c = zeros(N,1);
    ksi = zeros(N,Q,Q);

    % init alpha(z1), beta(zN), c(1)
    p_x1_z1 = p_xn_given_zn(1,:) .* p_start;    % p(x1,z1)
    alpha(1,:) = p_x1_z1 / sum(p_x1_z1);        % p(x1,z1)/p(x1)
    c(1) = sum(p_x1_z1);                        % c1 = p(x1) = sum(p(x1,z1))
    beta(N,:) = 1;

    % calculate alpha, c
    for i1 = 2:N
        tmp = p_xn_given_zn(i1,:) .* (alpha(i1-1,:)*A);
        c(i1) = sum(tmp);
        alpha(i1,:) = tmp / c(i1);
    end

    % calculate beta
    for i1 = N-1:-1:1
        beta(i1,:) = (beta(i1+1,:).*p_xn_given_zn(i1+1,:)) * A' / c(i1+1);
    end

    % calculate gamma
    gamma = alpha .* beta;

    % calculate ksi
    for i1 = 2:N
        ksi(i1,:,:) = bsxfun(@times, bsxfun(@times, alpha(i1-1,:)', A), p_xn_given_zn(i1,:).*beta(i1,:)) / c(i1);
    end
    ksi(1,:,:) = [];

    % calculate ln p(X)
    loglik = sum(log(c));
end
