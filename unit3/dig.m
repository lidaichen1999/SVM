load digits; clear size
[N, dim] = size(X);
Ntest1 = size(Xtest1, 1);
Ntest2 = size(Xtest2, 1);
minx = min(min(X)); 
maxx = max(max(X));

%
% Add noise to the digit maps
%

noisefactor = 0.3;
noise = noisefactor * maxx; % sd for Gaussian noise

Xn = X; 
for i = 1:N
  randn('state', i);
  Xn(i, :) = X(i, :) + noise * randn(1, dim);
end

Xnt1 = Xtest1; 
for i = 1:Ntest1
  randn('state', N + i);
  Xnt1(i, :) = Xtest1(i, :) + noise * randn(1, dim);
end

Xnt2 = Xtest2; 
for i = 1:Ntest2
  randn('state', N + Ntest1 + i);
  Xnt2(i, :) = Xtest2(i, :) + noise * randn(1, dim);
end

%
% select training set
%
Xtr = X(1:1:end, :);

sigmafactor_values = logspace(-2, 2, 10); % Values for sigmafactor in logarithmic scale
lpcs = length(sigmafactor_values);
reconstruction_error_train = zeros(1, lpcs);
reconstruction_error_test1 = zeros(1, lpcs);
reconstruction_error_test2 = zeros(1, lpcs);

%
% Denoise using the first principal components for different sigmafactor values
%

for k = 1:lpcs
    sigmafactor = sigmafactor_values(k);
    sig2 = dim * mean(var(Xtr)) * sigmafactor;
    
    disp(' ');
    disp(['Denoising with sig2 = ', num2str(sig2)]);
    
    % kernel PCA
    [lam, U] = kpca(Xtr, 'RBF_kernel', sig2, [], 'eig', 240);
    [lam, ids] = sort(-lam); 
    lam = -lam; 
    U = U(:, ids);
    
    Xdt = zeros(N, dim);
    Xdt1 = zeros(Ntest1, dim);
    Xdt2 = zeros(Ntest2, dim);
    
    for i = 1:N
        xt = Xn(i, :);
        Xdt(i, :) = preimage_rbf(Xtr, sig2, U, xt, 'denoise');
    end
    
    for i = 1:Ntest1
        xt1 = Xnt1(i, :);
        Xdt1(i, :) = preimage_rbf(Xtr, sig2, U, xt1, 'denoise');
    end
    
    for i = 1:Ntest2
        xt2 = Xnt2(i, :);
        Xdt2(i, :) = preimage_rbf(Xtr, sig2, U, xt2, 'denoise');
    end
    
    % Compute reconstruction errors
    reconstruction_error_train(k) = norm(X - Xdt, 'fro') / sqrt(N);
    reconstruction_error_test1(k) = norm(Xtest1 - Xdt1, 'fro') / sqrt(Ntest1);
    reconstruction_error_test2(k) = norm(Xtest2 - Xdt2, 'fro') / sqrt(Ntest2);
end

% Plot reconstruction errors as a function of sigmafactor values
figure;
semilogx(sigmafactor_values, reconstruction_error_train, 'b-o', 'LineWidth', 1.5);
hold on;
semilogx(sigmafactor_values, reconstruction_error_test1, 'r-o', 'LineWidth', 1.5);
semilogx(sigmafactor_values, reconstruction_error_test2, 'g-o', 'LineWidth', 1.5);
xlabel('sigmafactor');
ylabel('Reconstruction Error');
legend('Training Set', 'Validation Set 1', 'Validation Set 2');
title('Reconstruction Error vs. sigmafactor');

% Find the optimal sigmafactor value with minimal validation set error
[~, idx] = min(reconstruction_error_test1);
optimal_sigmafactor = sigmafactor_values(idx);
disp(['Optimal sigmafactor value: ', num2str(optimal_sigmafactor)]);

% Denoise using the optimal sigmafactor value
optimal_sig2 = dim * mean(var(Xtr)) * optimal_sigmafactor;
disp(['Denoising with optimal sig2 = ', num2str(optimal_sig2)]);

% kernel PCA with optimal sigmafactor
[lam, U] = kpca(Xtr, 'RBF_kernel', optimal_sig2, [], 'eig', 240);
[lam, ids] = sort(-lam); 
lam = -lam; 
U = U(:, ids);

Xdt_optimal = zeros(N, dim);
Xdt1_optimal = zeros(Ntest1, dim);
Xdt2_optimal = zeros(Ntest2, dim);

for i = 1:N
    xt = Xn(i, :);
    Xdt_optimal(i, :) = preimage_rbf(Xtr, optimal_sig2, U, xt, 'denoise');
end

for i = 1:Ntest1
    xt1 = Xnt1(i, :);
    Xdt1_optimal(i, :) = preimage_rbf(Xtr, optimal_sig2, U, xt1, 'denoise');
end

for i = 1:Ntest2
    xt2 = Xnt2(i, :);
    Xdt2_optimal(i, :) = preimage_rbf(Xtr, optimal_sig2, U, xt2, 'denoise');
end

% % Compute reconstruction errors with optimal sigmafactor
% reconstruction_error_train_optimal = norm(X - Xdt_optimal, 'fro') / sqrt(N);
% reconstruction_error_test1_optimal = norm(Xtest1 - Xdt1_optimal, 'fro') / sqrt(Ntest1);
% reconstruction_error_test2_optimal = norm(Xtest2 - Xdt2_optimal, 'fro') / sqrt(Ntest2);

    % Compute reconstruction errors
    reconstruction_error_train(k) = norm(X - Xdt, 2) / N;
    reconstruction_error_test1(k) = norm(Xtest1 - Xdt1, 2) / Ntest1;
    reconstruction_error_test2(k) = norm(Xtest2 - Xdt2, 2) / Ntest2;
    
disp(' ');
disp('Reconstruction errors with optimal sigmafactor:');
disp(['Training Set: ', num2str(reconstruction_error_train_optimal)]);
disp(['Validation Set 1: ', num2str(reconstruction_error_test1_optimal)]);
disp(['Validation Set 2: ', num2str(reconstruction_error_test2_optimal)]);
