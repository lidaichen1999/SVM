clear
close all

X = 3.*randn(100,2);
ssize = 10;
sig2_values = [0.01, 0.1, 1];  % 不同的 sig2 值
nc_values = [5, 10, 15];  % 不同的 nc 值

best_reconstruction_error = Inf;
best_sig2 = 0;
best_nc = 0;

for sig2 = sig2_values
    for nc = nc_values
        subset = zeros(ssize,2);
        reconstruction_errors = [];
        for t = 1:100,
            % new candidate subset
            r = ceil(rand*ssize);
            candidate = [subset([1:r-1 r+1:end],:); X(t,:)];

            % is this candidate better than the previous?
            if kentropy(candidate, 'RBF_kernel',sig2) > kentropy(subset, 'RBF_kernel',sig2),
                subset = candidate;
            end

            % calculate reconstruction error
            reconstruction_error = kentropy(subset, 'RBF_kernel',sig2);

            reconstruction_errors = [reconstruction_errors, reconstruction_error];

            % make a figure
            plot(X(:,1),X(:,2),'b*'); hold on;
            plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 
            pause(1)
        end

        % display reconstruction errors for the current sig2 and nc combination
        fprintf('sig2 = %f, nc = %d, Reconstruction Errors: %s\n', sig2, nc, mat2str(reconstruction_errors))

        % update best parameters if current reconstruction error is the lowest
        if reconstruction_errors(end) < best_reconstruction_error
            best_reconstruction_error = reconstruction_errors(end);
            best_sig2 = sig2;
            best_nc = nc;
        end
    end
end

% display the best parameters and reconstruction error
fprintf('Best parameters: sig2 = %f, nc = %d\n', best_sig2, best_nc)
fprintf('Best reconstruction error: %f\n', best_reconstruction_error)
