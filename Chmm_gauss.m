% continuous hmm, 1 gauss per state
function [p_start, A, Emis] = Chmm_gauss(Data, p_start, A, Emis, varargin)
iter_num = 10;          % the maximum of EM iteration
cov_type = 'diag';      % 'full' or 'diag'
cov_thresh = 1e-4;      % the thresh of cov
for i1 = 1:2:length(varargin)
    switch varargin{i1}
        case 'cov_type'
            cov_type = varargin{i1+1};
        case 'cov_thresh'
            cov_thresh = varargin{i1+1};
        case 'iter_num'
            iter_num = varargin{i1+1};
    end
end

data_num = length(Data);
Q = length(p_start);
sum_p_start = zeros(Q,1);
sum_ita = zeros(Q,Q);

% EM
for k = 1:iter_num
    Xtotal = []; gamma_total = [];
    for i1 = 1:data_num
        X = Data{i1};
        N = size(X,1);
        
        % get B
        Ob = Gauss_p_xn_cond_zn(X, Emis);   % size: N*Q
        
        % E
        [gamma, ita, loglik] = ForwardBackward(p_start,A,Ob);

        % M
        sum_p_start = sum_p_start + gamma(1,:)';
        sum_ita = sum_ita + ita;
        
        Xtotal = [Xtotal; X];
        gamma_total = [gamma_total; gamma];
    end
    
    p_start = normalise(sum_p_start);
    A = mk_stochastic(sum_ita);
    
    Emis = UpdateGaussPara(Xtotal,gamma_total,cov_type,cov_thresh);
end

end

% M
function Emis = UpdateGaussPara(X,gamma,cov_type,cov_thresh)
[N,Q] = size(gamma);
for i1 = 1:Q
    Emis{i1}.mu = X' * gamma(:,i1) / sum(gamma(:,i1));
    diff = bsxfun(@minus, X, Emis{i1}.mu');
    Emis{i1}.Sigma = bsxfun(@times, gamma(:,i1), diff)' * diff / sum(gamma(:,i1));
    if (cov_type=='diag')
        Emis{i1}.Sigma = diag(diag(Emis{i1}.Sigma));
    end
    if max(Emis{i1}.Sigma(:)) < cov_thresh    % prevent cov from being too small
        Emis{i1}.Sigma = cov_thresh * eye(size(Emis{i1}.Sigma, 1));
    end
end
end