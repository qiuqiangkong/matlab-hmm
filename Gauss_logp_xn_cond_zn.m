function logOb = Gauss_logp_xn_cond_zn(X, Emis)
N = size(X,1);
Q = length(Emis);
logOb = -inf*ones(N,Q);
for i1 = 1:Q
    logOb(:,i1) = Logmvnpdf(X, Emis{i1}.mu, Emis{i1}.Sigma);
end
end