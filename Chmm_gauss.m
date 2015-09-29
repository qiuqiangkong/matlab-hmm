% continuous hmm, 1 gauss per state
function [p_start, A, Emis, loglik] = Chmm_gauss(Data, state_num, varargin)

% Init Paras
Q = state_num;
for i1 = 1:2:length(varargin)
    switch varargin{i1}
        case 'p_start0'
            p_start = varargin{i1+1};
        case 'A0'
            A = varargin{i1+1};
        case 'Emis0'
            Emis = varargin{i1+1};
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
    p_start = mk_stochastic(rand(Q,1));  % p(z1), dim 1: Q
end
if (~exist('A'))
    A = mk_stochastic(rand(Q)); 
end
if (~exist('Emis'))
    Xall = cell2mat(Data');
    [prior, mu, Sigma] = Gmm(Xall, Q, 'diag');
    p_start = prior;
    for i1 = 1:Q
        Emis{i1}.mu = mu(:,i1);
        Emis{i1}.Sigma = Sigma(:,:,i1);
    end
end
if (~exist('iter_num'))
    iter_num = 100;          % the maximum of EM iteration
end
if (~exist('cov_type'))
    cov_type = 'diag';      % 'full' or 'diag'
end
if (~exist('cov_thresh'))
    cov_thresh = 1e-4;      % the thresh of cov
end
if (~exist('converge'))
    converge = 1 + 1e-4;
end

data_num = length(Data);
sum_p_start = zeros(Q,1);
sum_ita = zeros(Q,Q);

% ---------- EM -----------
loglik = 0;
pre_ll = -inf;
for k = 1:iter_num
    Xtotal = []; gamma_total = [];
    for i1 = 1:data_num
        X = Data{i1};
        N = size(X,1);
        
        % get Ob
        Ob = Gauss_p_xn_cond_zn(X, Emis);   % size: N*Q
        logOb = Gauss_logp_xn_cond_zn(X, Emis); 
        
        % E
        [gamma, ita, curr_ll] = ForwardBackward(p_start,A,[], logOb);
        loglik = loglik + curr_ll;

        % M
        sum_p_start = sum_p_start + gamma(1,:)';
        sum_ita = sum_ita + ita;
        
        Xtotal = [Xtotal; X];
        gamma_total = [gamma_total; gamma];
    end
    
    p_start = normalise(sum_p_start);
    A = mk_stochastic(sum_ita);
    
    Emis = UpdateGaussPara(Xtotal,gamma_total,cov_type,cov_thresh);
    
    if (loglik-pre_ll<log(converge)) break;
    else pre_ll = loglik; end
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

function Ob = Gauss_p_xn_cond_zn(X, Emis)
N = size(X,1);
Q = length(Emis);
Ob = zeros(N,Q);
for i1 = 1:Q
    Ob(:,i1) = mvnpdf(X, Emis{i1}.mu', Emis{i1}.Sigma);
end
end

function logOb = Gauss_logp_xn_cond_zn(X, Emis)
N = size(X,1);
Q = length(Emis);
logOb = -inf*ones(N,Q);
for i1 = 1:Q
    logOb(:,i1) = Logmvnpdf(X, Emis{i1}.mu, Emis{i1}.Sigma);
end
end

function [p_start0, A0, Emis0, cov_type, cov_thresh, iter_num] = InitPara(state_num, varargin)
Q = state_num;


end