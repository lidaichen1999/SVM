% Construct artificial dataset
X1 = randn(50,2) + 1;
X2 = randn(51,2) - 1;
Y1 = ones(50,1);
Y2 = -ones(51,1);

% Visualize the dataset
figure;
hold on;
plot(X1(:,1), X1(:,2), 'ro');
plot(X2(:,1), X2(:,2), 'bo');
hold off;

% Create a training set with corresponding labels
X = [X1; X2];
Y = [Y1; Y2];

% Train a linear SVM classifier using fitcsvm function
SVMModel = fitcsvm(X,Y);

% Visualize the decision boundary
figure;
gscatter(X(:,1),X(:,2),Y);
hold on;
h = svmplot(SVMModel);
title('SVM classification with linear kernel');
legend('Positive Class', 'Negative Class', 'Decision Boundary', 'Location', 'Best');
hold off;

% Function svmplot to plot decision boundary
function h = svmplot(SVMModel)
% Get the weights and bias of the hyperplane
w = SVMModel.Beta;
b = SVMModel.Bias;
% Get the x-axis limits of the plot
ax = axis;
xl = ax(1:2);
% Compute the y-axis limits of the plot using the hyperplane equation
yl = (-b-w(1)*xl)/w(2);
% Plot the decision boundary
h = plot(xl, yl, '-k');
end