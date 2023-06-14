load iris.mat
%
% Define the range of KernelScale and sigma^2 values
KernelScale_vals = logspace(-3, 3, 7);
sigma2_vals = logspace(-3, 3, 7);

% Initialize matrices to store the results
accuracy_random_split = zeros(length(KernelScale_vals), length(sigma2_vals));
accuracy_10fold_cv = zeros(length(KernelScale_vals), length(sigma2_vals));
accuracy_loo = zeros(length(KernelScale_vals), length(sigma2_vals));

% Iterate over all combinations of KernelScale and sigma^2 values
for i = 1:length(KernelScale_vals)
    for j = 1:length(sigma2_vals)
        KernelScale = KernelScale_vals(i);
        sigma2 = sigma2_vals(j);
        
        % Train the SVM model with RBF kernel using random split method
        SVMModel = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'rbf', ...
            'BoxConstraint', 1, 'KernelScale', sqrt(1/(2*sigma2)), ...
            'KernelScale', KernelScale);
        Ypred = predict(SVMModel, Xtest);
        accuracy_random_split(i,j) = mean(Ypred == Ytest);
        
        % Perform 10-fold cross validation
        cv = cvpartition(Ytrain, 'KFold', 10);
        accuracy_cv = zeros(cv.NumTestSets, 1);
        for k = 1:cv.NumTestSets
            SVMModel = fitcsvm(Xtrain(cv.training(k),:), Ytrain(cv.training(k)), ...
                'KernelFunction', 'rbf', 'BoxConstraint', 1, 'KernelScale', ...
                sqrt(1/(2*sigma2)), 'KernelScale', KernelScale);
            Ypred = predict(SVMModel, Xtrain(cv.test(k),:));
            accuracy_cv(k) = mean(Ypred == Ytrain(cv.test(k)));
        end
        accuracy_10fold_cv(i,j) = mean(accuracy_cv);
        
        % Perform leave-one-out cross validation
        accuracy_loo(i,j) = crossval('mcr', Xtrain, Ytrain, 'predfun', ...
            @(XTRAIN, YTRAIN, XTEST) predict(fitcsvm(XTRAIN, YTRAIN, ...
            'KernelFunction', 'rbf', 'BoxConstraint', 1, 'KernelScale', ...
            sqrt(1/(2*sigma2)), 'KernelScale', KernelScale), XTEST), 'partition', ...
            cvpartition(Ytrain, 'LeaveOut'));
    end
end

% Visualize the results
figure
subplot(1,3,1)
imagesc(KernelScale_vals, sigma2_vals, accuracy_random_split')
set(gca, 'XScale', 'log', 'YScale', 'log')
xlabel('KernelScale')
ylabel('Sigma^2')
title('Random split method')
colorbar

subplot(1,3,2)
imagesc(KernelScale_vals, sigma2_vals, accuracy_10fold_cv')
set(gca, 'XScale', 'log', 'YScale', 'log')
xlabel('KernelScale')
ylabel('Sigma^2')
title('10-fold cross validation')
colorbar

subplot(1,3,3)
imagesc(KernelScale_vals, sigma2_vals, accuracy_loo')
set(gca, 'XScale', 'log', 'YScale', 'log')
xlabel('KernelScale')
ylabel('Sigma^2')
title('Leave-one-out cross validation')
colorbar
