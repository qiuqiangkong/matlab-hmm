function logp = Logmvnpdf(X, mu, Sigma)
[N,p] = size(X);
x_minus_mu = bsxfun(@minus, X, mu');
logp = -0.5 * p * log(2*pi) - 0.5 * log(det(Sigma)) - 0.5 * sum((x_minus_mu*inv(Sigma)) .* x_minus_mu, 2);
end