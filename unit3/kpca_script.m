clc;
clear;

nb = 400;
sig = 0.3;

nb = nb / 2;

% construct data
leng = 1;
for t = 1:nb
    yin(t, :) = [2.*sin(t/nb*pi*leng) 2.*cos(.61*t/nb*pi*leng) (t/nb*sig)];
    yang(t, :) = [-2.*sin(t/nb*pi*leng) .45-2.*cos(.61*t/nb*pi*leng) (t/nb*sig)];
    samplesyin(t, :) = [yin(t, 1) + yin(t, 3).*randn   yin(t, 2) + yin(t, 3).*randn];
    samplesyang(t, :) = [yang(t, 1) + yang(t, 3).*randn   yang(t, 2) + yang(t, 3).*randn];
end

% get user-defined parameters
nc_values = [5, 10, 15]; % Different values of nc
sig2_values = [0.01, 0.1, 1]; % Different values of sig2

reconstruction_errors_lanczos = zeros(length(nc_values), length(sig2_values));
reconstruction_errors_nystrom = zeros(length(nc_values), length(sig2_values));

for i = 1:length(nc_values)
    nc = nc_values(i);
    for j = 1:length(sig2_values)
        sig2 = sig2_values(j);
        
        % Denoise the data using Lanczos approximation
        approx = 'lanczos';
        [~, ~, ~, ~, xd_lanczos] = kpca([samplesyin;samplesyang], 'RBF_kernel', sig2, [], approx, nc);
        reconstruction_errors_lanczos(i, j) = norm([samplesyin;samplesyang] - xd_lanczos, 'fro') / sqrt(size([samplesyin;samplesyang], 1));
        
        % Denoise the data using Nystrom approximation
        approx = 'nystrom';
        [~, ~, ~, ~, xd_nystrom] = kpca([samplesyin;samplesyang], 'RBF_kernel', sig2, [], approx, nc);
        reconstruction_errors_nystrom(i, j) = norm([samplesyin;samplesyang] - xd_nystrom, 'fro') / sqrt(size([samplesyin;samplesyang], 1));
    end
end

% Output reconstruction errors
disp('Reconstruction Errors (Lanczos):');
disp(reconstruction_errors_lanczos);
disp('Reconstruction Errors (Nystrom):');
disp(reconstruction_errors_nystrom);
