% Tune the parameters gam and sig2
gam = 0.5; % Modify this value for gamma
sig2 = 1; % Modify this value for sigma squared

% Compute probability estimates
bay_modoutClass({Xtrain, Ytrain, 'c', gam, sig2}, 'figure');
