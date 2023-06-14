% Set up LS-SVM model
model = initlssvm(Xtrain, Ytrain, 'c', [], [], 'RBF_kernel');

% Set tuning parameters
tuneopts = struct('crossvalidatelssvm', 'kfold', 'k', 10, 'costfun', 'misclass');

% Perform tuning using different algorithms
[gam1, sig2_1, cost1] = tunelssvm({Xtrain, Ytrain, 'c', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', tuneopts);
[gam2, sig2_2, cost2] = tunelssvm({Xtrain, Ytrain, 'c', [], [], 'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm', tuneopts);

% Print results
disp('Results for Nelder-Mead method:');
disp(['Gamma: ' num2str(gam1)]);
disp(['Sigma^2: ' num2str(sig2_1)]);
disp(['Cost: ' num2str(cost1)]);

disp('Results for gridsearch method:');
disp(['Gamma: ' num2str(gam2)]);
disp(['Sigma^2: ' num2str(sig2_2)]);
disp(['Cost: ' num2str(cost2)]);
