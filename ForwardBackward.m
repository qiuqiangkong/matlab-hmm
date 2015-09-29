function [gamma, sum_ita, loglik] = ForwardBackward(p_start,A, Ob, logOb, varargin)
maximize = 1;
% for i1 = 1:2:length(varargin)
%     switch varargin{i1}
%         case 'maximize'
%             maximize = varargin{i1+1}
%     end
% end

    
    
    if (maximize == 0)
        Ob = exp(logOb);
        % init para
        [N,Q] = size(Ob);
        scale = zeros(N,1);
        alpha = zeros(N,Q);             % dim 1: N, dim 2: Q
        beta = zeros(N,Q);              % dim 1: N, dim 2: Q
        gamma = zeros(N,Q);             % dim 1: N, dim 2: Q
        
        % calculate alpha, scale
        alpha(1,:) = (Ob(1,:).*p_start');
%         log(alpha(1,:))
%         pause
        [alpha(1,:), scale(1)] = normalise(alpha(1,:));
        for i1 = 2:N
            alpha(i1,:) = Ob(i1,:) .* (alpha(i1-1,:)*A);       % (PRML 13.59)
%             log(alpha(i1,:))
%             pause
            [alpha(i1,:), scale(i1)] = normalise(alpha(i1,:));
        end
        loglik = sum(log(scale));

        % calculate beta
        beta(N,:) = 1;
        for i1 = N-1:-1:1
            beta(i1,:) = beta(i1+1,:).*Ob(i1+1,:)*A';
            beta(i1,:) = normalise(beta(i1,:));
        end

        % calculate gamma
        gamma = alpha .* beta;          % dim 1: N, dim 2: Q
        for i1 = 1:N
            gamma(i1,:) = normalise(gamma(i1,:));
        end
        
        % calculate ita
        ita = zeros(N,Q,Q);
        for i1 = 2:N
            ita(i1,:,:) = alpha(i1-1,:)' * (Ob(i1,:).*beta(i1,:)) .* A;
%             reshape(ita(i1,:,:),Q,Q)
            ita(i1,:,:) = normalise(ita(i1,:,:));
            
%             pause
        end
        sum_ita = reshape(sum(ita,1),Q,Q);
        
    elseif (maximize==1)
        [N,Q] = size(logOb);
        gamma = zeros(N,Q);
        logalpha = -inf * ones(N,Q);
        logbeta = -inf * ones(N,Q);
        loggamma = -inf * ones(N,Q);
        

        logalpha(1,:) = logOb(1,:) + log(p_start)';
        for i1 = 2:N
            logalpha(i1,:) = logOb(i1,:) + max(bsxfun(@plus, logalpha(i1-1,:)', log(A)), [], 1);
        end        
        
        logbeta(N,:) = 0;
        for i1 = N-1:-1:1
            logbeta(i1,:) = max(bsxfun(@plus, logbeta(i1+1,:) + logOb(i1+1,:), log(A)), [], 2)';
        end
        
        loggamma = logalpha + logbeta;
        gamma = mk_stochastic(exp(LogNormalize(loggamma)));
        
        sum_ita = zeros(Q,Q);
        for i1 = 2:N
            Tmp = bsxfun(@plus, logalpha(i1-1,:)', logOb(i1,:)+logbeta(i1,:)) + log(A);
            Tmp = Tmp - max(Tmp(:));
            Tmp = normalise(exp(Tmp));
            sum_ita = sum_ita + Tmp;
            
        end
        
        loglik = max(logalpha(N,:));
    end
end

function X = LogNormalize(X)
X = bsxfun(@minus, X, max(X,[],2));
end