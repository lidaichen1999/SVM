% Linear SVM:
% Load the dataset
load('breast.mat');

% Train the linear SVM model
svm_linear = fitcsvm(trainset, labels_train, 'KernelFunction', 'linear');

% Predict labels for the test set
predicted_labels_linear = predict(svm_linear, testset);

% Compute ROC curve
[~, ~, ~, AUC_linear] = perfcurve(labels_test, predicted_labels_linear, '1');

% Plot the ROC curve
figure;
plotroc(labels_test, predicted_labels_linear);
title('ROC Curve (Linear SVM)');
%%
% Polynomial SVM:
% Load the dataset
load('breast.mat');

% Train the polynomial SVM model
svm_poly = fitcsvm(trainset, labels_train, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3);

% Predict labels for the test set
predicted_labels_poly = predict(svm_poly, testset);

% Compute ROC curve
[~, ~, ~, AUC_poly] = perfcurve(labels_test, predicted_labels_poly, '1');

% Plot the ROC curve
figure;
plotroc(labels_test, predicted_labels_poly);
title('ROC Curve (Polynomial SVM)');
%%
%RBF Kernel SVM:
% Load the dataset
load('breast.mat');

% Train the RBF kernel SVM model
svm_rbf = fitcsvm(trainset, labels_train, 'KernelFunction', 'RBF', 'OptimizeHyperparameters', 'auto');

% Predict labels for the test set
predicted_labels_rbf = predict(svm_rbf, testset);

% Compute ROC curve
[~, ~, ~, AUC_rbf] = perfcurve(labels_test, predicted_labels_rbf, '1');

% Plot the ROC curve
figure;
plotroc(labels_test, predicted_labels_rbf);
title('ROC Curve (RBF Kernel SVM)');

