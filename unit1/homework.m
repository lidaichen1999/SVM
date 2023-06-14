% Load the Ripley dataset
load ripley.mat

% Visualize the data
figure
gscatter(Xtrain(:,1), Xtrain(:,2), Ytrain)
title('Ripley Dataset')

% Define a range of polynomial degrees to try
degree_values = 1:5;

% Tune hyperparameters and kernel parameters for each degree of polynomial kernel
gam_poly = zeros(length(degree_values), 1);
sig2_poly = zeros(length(degree_values), 1);
cost_poly = zeros(length(degree_values), 1);

for i = 1:length(degree_values)
    [gam_poly(i), sig2_poly(i), cost_poly(i)] = tunelssvm({Xtrain, Ytrain, 'c', [], [], 'poly_kernel', degree_values(i)}, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
end

% Train and test the models with the tuned parameters
alpha_poly = cell(length(degree_values), 1);
b_poly = zeros(length(degree_values), 1);
Ytest_poly = zeros(size(Xtest, 1), length(degree_values));

for i = 1:length(degree_values)
    [alpha_poly{i}, b_poly(i)] = trainlssvm({Xtrain, Ytrain, 'c', gam_poly(i), sig2_poly(i), 'poly_kernel', degree_values(i)});
    Ytest_poly(:, i) = simlssvm({Xtrain, Ytrain, 'c', gam_poly(i), sig2_poly(i), 'poly_kernel', degree_values(i)}, {alpha_poly{i}, b_poly(i)}, Xtest);
end

% Compute ROC curves and plot them
figure
hold on
for i = 1:length(degree_values)
    [X,Y,T,AUC] = perfcurve(Ytest, Ytest_poly(:, i), 1);
    plot(X,Y, 'LineWidth', 1.5)
end
hold off
legend(string(degree_values), 'Location', 'SouthEast')
title('ROC Curve - Polynomial Kernel with Varying Degree')

%% Q212
% %问题二：威斯康星州乳腺癌数据集数据看起来比较干净，因此我们可以开始使用线性内核
load breast.mat

% Train the model
model = {trainset, labels_train, 'c', [], [], 'lin_kernel'};
[gam, sig2, cost] = tunelssvm(model, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
[alpha, b] = trainlssvm({trainset, labels_train, 'c', gam, sig2, 'lin_kernel'});

% Test the model
Ytest = simlssvm({trainset, labels_train, 'c', gam, sig2, 'lin_kernel'}, {alpha, b}, testset);

% Compute and plot ROC curve
roc(Ytest, labels_test);


%%
% 
load diabetes.mat

% Linear model
model_linear = {trainset, labels_train, 'c', [], [], 'lin_kernel'};
[gam_linear, sig2_linear, ~] = tunelssvm(model_linear, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
[alpha_linear, b_linear] = trainlssvm({trainset, labels_train, 'c', gam_linear, sig2_linear, 'lin_kernel'});
Ylatent_linear = simlssvm({trainset, labels_train, 'c', gam_linear, sig2_linear, 'lin_kernel'}, {alpha_linear, b_linear}, testset);
roc(Ylatent_linear, labels_test);

% Polynomial model
degree = 2; % Modify this value for the desired polynomial degree
model_poly = {trainset, labels_train, 'c', [], [], 'poly_kernel', degree};
[gam_poly, sig2_poly, ~] = tunelssvm(model_poly, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
[alpha_poly, b_poly] = trainlssvm({trainset, labels_train, 'c', gam_poly, sig2_poly, 'poly_kernel', degree});
Ylatent_poly = simlssvm({trainset, labels_train, 'c', gam_poly, sig2_poly, 'poly_kernel', degree}, {alpha_poly, b_poly}, testset);
roc(Ylatent_poly, labels_test);

% RBF kernel model
model_rbf = {trainset, labels_train, 'c', [], [], 'RBF_kernel'};
[gam_rbf, sig2_rbf, ~] = tunelssvm(model_rbf, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
[alpha_rbf, b_rbf] = trainlssvm({trainset, labels_train, 'c', gam_rbf, sig2_rbf, 'RBF_kernel'});
Ylatent_rbf = simlssvm({trainset, labels_train, 'c', gam_rbf, sig2_rbf, 'RBF_kernel'}, {alpha_rbf, b_rbf}, testset);
roc(Ylatent_rbf, labels_test);

fprintf('Performance of Models:\n');
fprintf('Linear Model AUC: %.4f\n', AUC_linear);
fprintf('Polynomial Model AUC: %.4f\n', AUC_poly);
fprintf('RBF Kernel Model AUC: %.4f\n', AUC_rbf);
