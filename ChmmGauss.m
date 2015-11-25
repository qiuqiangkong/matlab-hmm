% SUMMARY:  Train Gauss-HMM model
% AUTHOR:   QIUQIANG KONG
% Created:  17-11-2015
% Modified: - 
% -----------------------------------------------------------
% input:
%   Data        cell of data
%   state_num   state num
% varargin input:
%   p_start0    p(z1), size: Q*1
%   A           p(zn|zn-1), transform matrix, size: Q*Q
%   phi0:       emission probability para 
%       mu        size: p*Q
%       Sigma     size: p*p*Q
%   iter_num    how many time the EM should run (default: 100)
%   converge    (default: 1+1e-4)
% output
%   p_start  p(z1), dim 1: Q
%   A        p(zn|zn-1), transform matrix, size: Q*Q
%   phi0:       emission probability para 
%       mu        size: p*Q
%       Sigma     size: p*p*Q
% ===========================================================
function [p_start, A, phi, loglik] = ChmmGauss(Data, state_num, varargin)

% Init Paras
Q = state_num;
p = size(Data{1},2);
for i1 = 1:2:length(varargin)
    switch varargin{i1}
        case 'p_start0'
            p_start = varargin{i1+1};
        case 'A0'
            A = varargin{i1+1};
        case 'phi0'
            phi = varargin{i1+1};
        case 'cov_type'
            cov_type = varargin{i1+1};
        case 'cov_thresh'
            cov_thresh = varargin{i1+1};
        case 'iter_num'
            iter_num = varargin{i1+1};
        case 'converge'
            converge = varargin{i1+1};
    end
end
if (~exist('p_start'))
    tmp = rand(1,Q);
    p_start = tmp / sum(tmp);
end
if (~exist('A'))
    tmp = rand(Q,Q);
    A = bsxfun(@rdivide, tmp, sum(tmp,2));
end
if (~exist('phi'))
    Xall = cell2mat(Data');
    [~, phi.mu, phi.Sigma] = Gmm(Xall, Q, 'diag');
end
if (~exist('iter_num'))
    iter_num = 100;          % the maximum of EM iteration
end
if (~exist('cov_type'))
    cov_type = 'diag';      % 'full' or 'diag'
end
if (~exist('cov_thresh'))
    cov_thresh = 1e-2;      % the thresh of cov
end
if (~exist('converge'))
    converge = 1 + 1e-4;
end

pre_ll = -inf;
obj_num = length(Data);
for k = 1:iter_num
    % E STEP
    for r = 1:obj_num
        
%         [Gamma{r}, Ksi{r}, Loglik{r}] = ForwardBackward(p_xn_given_zn, p_start, A);
        logp_xn_given_zn = Gauss_logp_xn_given_zn(Data{r}, phi);
        [LogGamma{r}, LogKsi{r}, Loglik{r}] = LogForwardBackward(logp_xn_given_zn, p_start, A);
    end
    
%     logp_xn_given_zn
%     1
%     pause
    
    % convert loggamma to gamma, logksi to ksi, substract the max
    [Gamma, Ksi] = UniformLogGammaKsi(LogGamma, LogKsi);
    
    
    % M STEP common
    [p_start, A] = M_step_common(Gamma, Ksi);
    
    % M STEP for Gauss
    mu_numer = zeros(p,Q);
    mu_denom = zeros(1,Q);
    
    % update phi.mu
    for r = 1:obj_num
        mu_numer = mu_numer + Data{r}' * Gamma{r};
        mu_denom = mu_denom + sum(Gamma{r},1);
    end
    phi.mu = bsxfun(@rdivide, mu_numer, mu_denom);
    
    % update phi.Sigma
    Sigma_numer = zeros(p,p,Q);
    for r = 1:obj_num
        for i1 = 1:Q
            x_minus_mu = bsxfun(@minus, Data{r}, phi.mu(:,i1)');
            Sigma_numer(:,:,i1) = Sigma_numer(:,:,i1) + bsxfun(@times, Gamma{r}(:,i1), x_minus_mu)' * x_minus_mu;
        end
    end
    phi.Sigma = bsxfun(@rdivide, Sigma_numer, reshape(mu_denom,1,1,Q));
%     LogGamma{r}
    
    for i1 = 1:Q
        if (cov_type=='diag')
            phi.Sigma(:,:,i1) = diag(diag(phi.Sigma(:,:,i1)));
        end
        if min(eig(phi.Sigma(:,:,i1))) < cov_thresh    % prevent cov from being too small
            phi.Sigma(:,:,i1) = phi.Sigma(:,:,i1) + cov_thresh * eye(p);
        end
    end
    
    % loglik
    loglik = 0;
    for r = 1:obj_num
        loglik = loglik + Loglik{r};
    end
    if (loglik-pre_ll<log(converge)) break;
    else pre_ll = loglik; end    
    

end

end