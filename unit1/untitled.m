% 加载数据集
%Try out a polynomial kernel with degree = 1,2,3,... and t = 1 (fix gam = 1). Assess the performance on the test set. What happens when you change the degree of the polynomial kernel?
load iris.mat

% 定义正则化常数
gam = 1;
num_degrees = 5;


% 定义多项式核函数的次数
degrees = 1:6;

% 初始化性能矩阵
performance = zeros(num_degrees, 3);


for i = 1:length(degrees)
    % 训练SVM分类器
    degree = degrees(i);
    model = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'polynomial', 'PolynomialOrder', degree, 'BoxConstraint', gam);

    % 预测测试集的标签
    Ypred = predict(model, Xtest);

    % 计算混淆矩阵
    C = confusionmat(Ytest, Ypred);

    % 计算准确率、召回率和F1分数
    acc = mean(diag(C)) / sum(C(:));
    rec = diag(C) ./ sum(C, 2);
    f1 = 2 * (rec .* diag(C)) ./ (rec + diag(C));


    % 存储性能指标
    performance(i,:) = [acc, mean(rec), mean(f1)];
end

% 打印性能矩阵
disp(performance);

% 选择最好的多项式核函数次数
[~, best_degree] = max(performance(:, 3));
best_model = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'polynomial', 'PolynomialOrder', degrees(best_degree), 'BoxConstraint', gam);

% 在测试集上评估最好的模型
Ypred = predict(best_model, Xtest);
C = confusionmat(Ytest, Ypred);
acc = mean(diag(C)) / sum(C(:));
rec = diag(C) ./ sum(C, 2);
f1 = 2 * (rec .* diag(C)) ./ (rec + diag(C));

% 打印性能指标
fprintf('Best model performance:\n');
fprintf('Accuracy: %.2f\n', acc);
fprintf('Recall: %.2f\n', mean(rec));
fprintf('F1 score: %.2f\n', mean(f1));

%%

