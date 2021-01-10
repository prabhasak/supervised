clc;
close all;

tic

iterations = 200
letters = {'c','b','a'};
samples = [10 50 100 500]
accuracy_LLS1 = zeros(length(samples),iterations);
accuracy_LLS2 = zeros(length(samples),iterations);
accuracy_LLS3 = zeros(length(samples),iterations);
avg_accuracy_LLS1 = zeros(length(samples),1);
avg_accuracy_LLS2 = zeros(length(samples),1);
avg_accuracy_LLS3 = zeros(length(samples),1);

% C1=A(1:1000,1:2);
% C2=A(1001:2000,1:2);

m=3;

X = csvread('iris_dataset.txt');
D = 4;
ratio = 0.8;

A = X(1:ratio*size(X,1),:);
B = X(ratio*size(X,1)+1:size(X,1),:);


fprintf(strcat('Problem 2',letters{m},', HW3\n'));

M0_1 = fitglm(A(:,1:D),A(:,D+1));
M0_2 = fitcecoc(A(:,1:D),A(:,D+1),'Coding','onevsall');
M0_3 = fitcecoc(A(:,1:D),A(:,D+1),'Coding','ternarycomplete');
threshold_1 = predict(M0_1,A(1,1:D));
threshold_2 = predict(M0_2,A(1,1:D));
threshold_3 = predict(M0_3,A(1,1:D));

for k=1:length(samples)
    
    z1 = zeros(size(B,1),1);
%     z2 = zeros(size(B,1),1);
    
    for j=1:iterations
        
        %         S1=datasample(A(:,1:D+1),samples(k),2);
        %         M1 = fitlm(S1(:,1:D),S1(:,D+1));
        S=datasample(A(:,1:D+1)',samples(k),2)';
        M1 = fitglm(S(:,1:D),S(:,D+1));
        M2 = fitcecoc(S(:,1:D),S(:,D+1),'Coding','onevsall');
        M3 = fitcecoc(S(:,1:D),S(:,D+1),'Coding','ternarycomplete');
        
        y1 = predict(M1,B(:,1:D));
        y2 = predict(M2,B(:,1:D));
        y3 = predict(M3,B(:,1:D));
        
        
        for i=1:size(B,1)
            if sign(y1(i))==sign(threshold_1)
                z1(i) = 1;
            else
                z1(i) = -1;
            end
        end
        
        accuracy_LLS1(k,j) = 1-sum(z1~=B(:,D+1))/size(B,1);
        accuracy_LLS2(k,j) = 1-sum(y2~=B(:,D+1))/size(B,1);
        accuracy_LLS3(k,j) = 1-sum(y3~=B(:,D+1))/size(B,1);
        
    end
    
    avg_accuracy_LLS1(k) = mean(accuracy_LLS1(k,:));
    avg_accuracy_LLS2(k) = mean(accuracy_LLS2(k,:));
    avg_accuracy_LLS3(k) = mean(accuracy_LLS3(k,:));
    %     fprintf('Average linear least squares accuracy for sample size %d is: %f \n',samples(k),avg_accuracy_LS(k));
    %     fprintf('Average linear least squares accuracy for sample size %d is: %f \n',samples(k),avg_accuracy_LR(k));
    
end


filename='P2a.mat';
save(filename,'avg_accuracy_LLS1','avg_accuracy_LLS2','avg_accuracy_LLS3')

avg_accuracy_LLS1
avg_accuracy_LLS2

% load('P2a.mat')
    
    figure1 = figure();
%     h1=plot(samples,avg_accuracy_LLS1,'g-','LineWidth',2,'MarkerSize',6);
%     hold on
    h2=plot(samples,avg_accuracy_LLS2,'r-','LineWidth',2,'MarkerSize',6);
    hold on
    h3=plot(samples,avg_accuracy_LLS3,'c-','LineWidth',2,'MarkerSize',6);
    hold on
    legend1 = legend([h2 h3],'OneVsRest','3-class','Location','southeast');
    title('LLS Classifiers on Test data');
    xlabel('Sample Size','FontSize',12);
    ylabel('Avg accuracy','FontSize',12);
    
        extensions = {'eps','fig','png','pdf'};

    for k = 1:length(extensions)
        saveas(gcf,strcat('P2',letters{m},'_accuracy'), extensions{k})
    end

toc

%
% C1=A(1:2,1:1000);
% C2=A(1:2,1001:2000);
%
%
% x1 = -1:0.01:1;
% x2 = -1:0.01:1;
% [x,y] = meshgrid(x1,x2);
% % p = gamfit(C1);
%
% y1 = gampdf(C1,0.5,1);
% y2 = gampdf(C2,2,2);
% plot(y1)
% hold on
% plot(y2)
% % fitdist(A(1:2,:))
%
%


% X = A(1:2,:)';
% y = A(3,:)';
%
% X_1 = X(y == 1,:);
% X_2 = X(y == -1,:);
%
% y = gampdf(X_1, 0.5, 1);
% y = y(:,1).*y(:,2);
% scatter(X_1(:,1), X_1(:,2))
