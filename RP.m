function [ Cu_, N, lossCurve ] = RP( X, Y, labelIndex, param )
% Robust Propagation Algorithm
% X - data matrix, with n * m, n = #inst. m = #feat.
% Y - label matrix, with n * 1, k = #class, 
% labelIndex - logistic matrix, with n * 1, x is labeled if element of labelIndex = 1

%% 1. pick data
[n, m] = size(X);
[n, k] = size(Y);
Xl = X(labelIndex, :);
Yl = Y(labelIndex, :);
Xu = X(~labelIndex, :);
Yu = Y(~labelIndex, :); % groundtruth for unlabeled data
% re-group
X = [Xl; Xu];
Y = [Yl; Yu];
nl = size(Xl, 1);
nu = n - nl;

%% 2. get params
iter = param.iter;
gap = param.gap;
knn = param.k; % knn graph
lambda1 = param.lambda1; % for N
lambda2 = param.lambda2; % for C

%% 3. calculate R
% 3.1 calculate W
KK = pdist2(X, X);
mean_dis = sum(KK(:)) / ((n - 1) * n);
rbf_gamma = 1.0 / (sqrt(mean_dis) * m);  % for real-world
% rbf_gamma = 3.0 / mean_dis  # for synthetic
KK2 = KK.^2;
W = exp(-rbf_gamma * KK2);
W(logical(eye(n))) = 0; % 对角线上元素为0
W_ = W;
for i=1 : n
    [~, index] = sort(W_(i, :));
    W(i, index(1 : n-knn)) = 0;
end
W = max(W, W');

KKlu = KK(labelIndex, ~labelIndex); %pdist2(Xl, Xu);

%3.2 calculate discriminate matrix, k * nu
DM = zeros(k, nu);
for i = 1 : k
    kthIndex = find(Yl(:, i) == 1);
    DM(i, :) = min(KKlu(kthIndex, :), [], 1); % 每列的最小值
end

% 3.3 calcualte R
[largestValue, index] = min(DM, [], 1); % min 作用在col上
DM_ = DM;
DM_(find(DM==min(DM))) = 1008611;
[SecondlargestValue, index] = min(DM_, [], 1);
R = (SecondlargestValue - largestValue) ./ SecondlargestValue;

%% 4. init
%C = zeros(nu, 1);
Cl = ones(nl, 1);
N = W;

%% 5. update
preLoss = 0;
preC = zeros(n, 1);
preN = W;
first = true;
lossCurve = zeros(iter, 1);

for i = 1:iter
    % update C
    L = diag(sum(N)) - N;
    Llu = L(labelIndex, ~labelIndex);
    Luu = L(~labelIndex, ~labelIndex);
    
    cvx_begin
            variable Cu(nu, 1)
            minimize( 2 * Cl' * Llu * Cu + Cu' * Luu * Cu + lambda2 * norm(Cu - R', 2))
            subject to
                Cu <= ones(size(nu, 1))
                Cu >= zeros(size(nu, 1))
    cvx_end
    
    % update N
    C = [Cl; Cu];
    Cu_=Cu;
    Cij = pdist2(C, C);
    N = W - (Cij.^2) / (2 * lambda1);
    N(N<0)=0;
    
    % loss
    loss = C' * L * C + lambda1 * norm(N-W, 2)^2 + lambda2 * norm(Cu - R')^2;
    lossCurve(i) = loss;
    fprintf('%d th iter, current loss=%d, change in N = %d, change in C = %d \n', i, loss, norm(N - preN, 2), norm(preC - C, 2));
    if first
        first = false;
        preLoss = loss;
        preC = C;
        preN = N;
    else
        if abs(loss - preLoss) < gap
            fprintf('\n loss gap = %d is small, iterated %d times..\n', abs(loss - preLoss), i);
            break
        end
    end
end

end

