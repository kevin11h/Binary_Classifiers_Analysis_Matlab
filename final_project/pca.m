function [Z] = pca(X, nd) % My hw7 from Andrew Ng's Machine Learning class
    [X_norm, ~, ~] = featureNormalize(X);
    m = size(X_norm, 1);
    covariance = (1/m)*(X'*X); covariance(~isfinite(covariance)) = 0;
    [U, ~, ~] = svd(covariance);
    
    Z = projectData(X_norm, U, nd);
end

function [X_norm, mu, sigma] = featureNormalize(X)
    mu = mean(X);
    X_norm = bsxfun(@minus, X, mu);

    sigma = std(X_norm);
    X_norm = bsxfun(@rdivide, X_norm, sigma);
    X_norm(~isfinite(X_norm)) = 0;
end

function Z = projectData(X, eigenvectors, Ndimensions)
    m = size(X,1);
    Z = zeros(m, Ndimensions);

    top_k_eigenvectors = eigenvectors(:,1:Ndimensions);
  
    for i = 1:m
        x = X(i, :)';
        projection_k = x' * top_k_eigenvectors; % (1 x n) * (n x k)
        Z(i,:) = projection_k;
    end
end