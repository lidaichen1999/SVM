% Load dataset
load iris.mat

% Define ranges for kernel parameters
sig2_range = logspace(-3, 3, 7);
gam_range = logspace(-3, 3, 7);

% Initialize variables to store accuracy results
sig2_acc = zeros(size(sig2_range));
gam_acc = zeros(size(gam_range));

% Loop over different values of sig2 and assess performance
for i = 1:numel(sig2_range)
    sig2 = sig2_range(i);
    SVMModel = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'rbf', 'BoxConstraint', 1, 'KernelScale', sqrt(1/(2*sig2)), 'Standardize', false);

    Ypred = predict(SVMModel, Xtest);
    accuracy = sum(Ypred == Ytest)/numel(Ytest);
    sig2_acc(i) = accuracy;
end

sig2_values = [0.01, 0.1, 1, 10, 100];
accuracy_values = zeros(size(sig2_values));
for idx = 1:numel(sig2_values)
    sig2 = sig2_values(idx);
    SVMModel = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'rbf', 'BoxConstraint', 1, 'KernelScale', sqrt(1/(2*sig2)), 'Scale', sig2);
    Ypred = predict(SVMModel, Xtest);
    accuracy = sum(Ypred == Ytest)/numel(Ytest);
    accuracy_values(idx) = accuracy;
end

semilogx(sig2_values, accuracy_values, '-o');
xlabel('Squared kernel bandwidth (log scale)');
ylabel('Accuracy');
title('Performance of SVM with RBF kernel');
