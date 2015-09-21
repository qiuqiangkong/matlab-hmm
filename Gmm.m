% SUMMARY: Gmm algorithm, using EM algorithm
% AUTHOR:  QIUQIANG KONG, Queen Mary University of London
% Created: 16-09-2015
% Modified: 18-09-2015 Add varargin parameters
%                      Add minimum of cov
%           19-09-2015 Add some remarks
% -----------------------------------------------------------
% Remarks: need voicebox (kmeans.m), error_ellipse.m
% input:   
%   X      size: N*p
%   mix    num of mix 
% varargin input:
%   restart_num     the times of restart kmeans
%   iter_num        the maximum of EM iteration
%   cov_type        'full' or 'diag'
%   cov_thresh      the minimum of cov
% output:
%   pi              size: M*1; dim 1: mix num 
%   mu              size: p*M; dim 1: feature dim, dim 2: mix num    
%   Sigma           size: p*p*M; dim 1,2: feature dim, dim 3: mix num
% Ref:     Chap 9 <Pattern Analysis and Machine Learning>
%          voicebox toolbox
% ===========================================================
function [pi, mu, Sigma] = Gmm(X, mix, varargin)
restart_num = 10;       % the times of restart kmeans
iter_num = 100;         % the maximum of EM iteration
cov_type = 'diag';      % 'full' or 'diag'
cov_thresh = 1e-4;      % the thresh of cov
for i1 = 1:2:length(varargin)
    switch varargin{i1}
        case 'cov_type'
            cov_type = varargin{i1+1};
        case 'cov_thresh'
            cov_thresh = varargin{i1+1};
        case 'restart_num'
            restart_num = varargin{i1+1};
        case 'iter_num'
            iter_num = varargin{i1+1};
    end
end

% Init gmm by kmeans
[pi0, mu0, Sigma0] = Gmm_init_by_kmeans(X, mix, restart_num, cov_thresh);

% EM algorighm for Gmm
[pi, mu, Sigma] = Gmm_em(X, pi0, mu0, Sigma0, iter_num, cov_type, cov_thresh);

% % Debug
% for i1 = 1:mix
%     error_ellipse(Sigma(:,:,i1), mu(:,i1), 'style', 'r'), hold on
% end


end

% --------- Init gmm by kmeans ---------
function [pi0, mu0, Sigma0] = Gmm_init_by_kmeans(X, mix, restart_num, cov_thresh)
% restart kmeans for several times
err = inf;
for i1 = 1:restart_num
    [mu0_curr, err_curr, indi_curr] = kmeans(X, mix, 'f');   
    mu0_curr = mu0_curr';       % u0: dim 1: feature dim, dim 2: feature dimmix num
    if err_curr < err
        err = err_curr;
        mu0 = mu0_curr;
        indi = indi_curr;
    end
end

% calculate pi0
pi0 = zeros(mix,1);
for i1 = 1:mix
    pi0(i1) = sum(indi==i1);
end
pi0 = pi0 / length(indi);

% calculate Sigma0
for i1 = 1:mix
    X_curr = X(indi==i1, :);
    mu0_curr = mu0(:, i1);
    Sigma0_curr = cov(bsxfun(@minus, X_curr, mu0_curr'));
    Sigma0(:,:,i1) = Sigma0_curr;               % dim 1, dim 2: feature dim, dim 3: mix num
    
    if max(max(Sigma0(:,:,i1))) < cov_thresh    % prevent cov from being too small
        Sigma0(:,:,i1) = cov_thresh * eye(size(Sigma0(:,:,i1),1));
    end
end
end

% --------- EM algorighm for Gmm ---------
function [pi, mu, Sigma] = Gmm_em(X, pi, mu, Sigma, iter_num, cov_type, cov_thresh)
mix = length(pi);
N = size(X, 1);

for k = 1:iter_num
    % E step
    p_mat = zeros(N, mix);
    for i1 = 1:mix
        p_mat(:, i1) = mvnpdf(X, mu(:,i1)', Sigma(:,:,i1));
    end

    numer = bsxfun(@times, p_mat, pi');     % dim 1: N, dim 2: mix num
    denor = sum(numer, 2);                  % dim 1: N
    gamma = bsxfun(@rdivide, numer, denor);  % dim 1: N, dim 2: mix num

    % M step
    Nk = sum(gamma, 1)';     % dim 1: mix num
    mu = bsxfun(@rdivide, (X' * gamma), Nk');

    for i1 = 1:mix
        x_minus_mu = bsxfun(@minus, X, mu(:,i1)');
        Sigma(:,:,i1) = bsxfun(@times, gamma(:,i1), x_minus_mu)' * x_minus_mu / Nk(i1);
        if (cov_type=='diag')
            Sigma(:,:,i1) = diag(diag(Sigma(:,:,i1)));
        end
        if max(max(Sigma(:,:,i1))) < cov_thresh    % prevent cov from being too small
            Sigma(:,:,i1) = cov_thresh * eye(size(Sigma(:,:,i1),1));
        end
        pi = Nk / N;
    end
    
%     % debug
%     for i1 = 1:mix
%         error_ellipse(Sigma(:,:,i1), mu(:,i1), 'style', 'r'), hold on
%     end
%     pause
end

end