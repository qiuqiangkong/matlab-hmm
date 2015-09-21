function [gamma, sum_ita, loglik] = ForwardBackward(p_start,A,Ob)
    % init para
    [N,Q] = size(Ob);
	scale = zeros(N,1);
    alpha = zeros(N,Q);             % dim 1: N, dim 2: Q
    beta = zeros(N,Q);              % dim 1: N, dim 2: Q
    gamma = zeros(N,Q);             % dim 1: N, dim 2: Q
    
    % calculate alpha, scale
    alpha(1,:) = (Ob(1,:).*p_start');
    [alpha(1,:), scale(1)] = normalise(alpha(1,:));
    for i1 = 2:N
        alpha(i1,:) = Ob(i1,:) .* (alpha(i1-1,:)*A);       % (PRML 13.59)
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
%         if isnan(gamma(i1,:))
%             gamma(i1,:) = 1 / Q;
%         end
    end

    alpha
    beta
    gamma
    pause

    % calculate ita
    ita = zeros(N,Q,Q);
    for i1 = 2:N
        ita(i1,:,:) = alpha(i1-1,:)' * (Ob(i1,:).*beta(i1,:)) .* A;
        ita(i1,:,:) = normalise(ita(i1,:,:));
    end
    sum_ita = reshape(sum(ita,1),Q,Q);
end