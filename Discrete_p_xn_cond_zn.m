function Ob = Discrete_p_xn_cond_zn(X, mu)
N = size(X,1);
Q = size(mu,2);
Ob = zeros(N,Q);                 % p(xn|zn), dim 1: N, dim 2: Q
for i1 = 1:Q
    mu_curr = mu(:,i1);
    Ob(:,i1) = mu_curr(X);
end

end