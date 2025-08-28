close all
clear all

data=readmatrix("IndividualStockPrices.csv");

rf=data(2:end,end)/100; %risk-free rate
xM=data(2:end,end-3)/100; %market excess return
SMB=data(2:end,end-2)/100; %SMB factor return
HML=data(2:end,end-1)/100; %HML factor return

prices=data(:,2:end-4); %individual stock prices
r=prices(2:end,:)./prices(1:end-1,:)-1; %individual stock returns

x=r-rf; %individual stock excess returns

[T,n]=size(x); %number of observations in the time series T and number of stocks n

%% CAPM test
%% First-pass regression (estimation of the betas)

%estimate betas
for i=1:n
    reg=fitlm(xM,x(:,i));
    beta(i)=reg.Coefficients.Estimate(2);
end

%% Easy way: regress avg(ret) on betas
avgx=mean(x);

figure(1)
scatter(beta,avgx)

reg=fitlm(beta',avgx');
display(reg)

%Comment: 
% 1) intercept not 0 (inconsistent with the CAPM)
% 2)slope >0 and statistically significant (consistent with the CAPM)
% conclusion: reject the CAPM because of 1)


%% Fama-MacBeth procedure
%second-pass regression
for t=1:T
    reg=fitlm(beta',x(t,:)');
    intercept(t,1)=reg.Coefficients.Estimate(1);
    slope(t,1)=reg.Coefficients.Estimate(2);
end

%compute average intercept and slope by regressing these on a constant
% Create a constant independent variable
constant = ones(size(intercept));
% Prepare a table for fitlm
tbl = table(intercept, constant, 'VariableNames', {'intercept', 'constant'});
% Fit the linear model
mdl = fitlm(tbl, 'intercept ~ -1+constant');
display(mdl) %comment: i) we reject that the average intercept is 0


% Prepare a table for fitlm
tbl = table(slope, constant, 'VariableNames', {'slope', 'constant'});
% Fit the linear model
mdl = fitlm(tbl, 'slope ~ -1 + constant');
display(mdl) %comment: ii) we don't reject that the average slope is 0

%conclusion: reject the CAPM because of both i) and ii)



%% Fama-Frech 3-factor model test
%% First-pass regression (estimation of the betas)

%estimate betas
for i=1:n
    reg=fitlm([xM SMB HML],x(:,i));
    betaM(i)=reg.Coefficients.Estimate(2);
    betaSMB(i)=reg.Coefficients.Estimate(3);
    betaHML(i)=reg.Coefficients.Estimate(4);
end

%% Easy way: regress avg(ret) on betas
avgx=mean(x);


reg=fitlm([betaM' betaSMB' betaHML'],avgx');
display(reg)

%Comment: 
% 1) intercept not 0 (inconsistent with the FF 3-factor model)
% 2)slope on betaM >0 and statistically significant (consistent with the FF 3-factor model)
% 3)slope on betaSMB >0 and statistically significant (consistent with the FF 3-factor model)
% 4)slope on betaHML <0 and statistically significant (inconsistent with the FF 3-factor model)
% conclusion: reject the Fama-French 3-factor model because of 1) and 4)


%% Fama-MacBeth procedure
%second-pass regression
for t=1:T
    reg=fitlm([betaM' betaSMB' betaHML'],x(t,:)');
    intercept(t,1)=reg.Coefficients.Estimate(1);
    slopeOnBetaM(t,1)=reg.Coefficients.Estimate(2);
    slopeOnBetaSMB(t,1)=reg.Coefficients.Estimate(3);
    slopeOnBetaHML(t,1)=reg.Coefficients.Estimate(4);
end

%compute average intercept and slope by regressing these on a constant
% Create a constant independent variable
constant = ones(size(intercept));
% Prepare a table for fitlm
tbl = table(intercept, constant, 'VariableNames', {'intercept', 'constant'});
% Fit the linear model
mdl = fitlm(tbl, 'intercept ~ -1 + constant');
display(mdl) %comment: i) we reject that the average intercept is 0


% Prepare a table for fitlm
tbl = table(slopeOnBetaM, constant, 'VariableNames', {'slopeOnBetaM', 'constant'});
% Fit the linear model
mdl = fitlm(tbl, 'slopeOnBetaM ~ -1 + constant');
display(mdl) %comment: ii) we don't reject that the average slope on betaM is 0


% Prepare a table for fitlm
tbl = table(slopeOnBetaSMB, constant, 'VariableNames', {'slopeOnBetaSMB', 'constant'});
% Fit the linear model
mdl = fitlm(tbl, 'slopeOnBetaSMB ~ -1 + constant');
display(mdl) %comment: iii) we don't reject that the average slope on betaSMB is 0


% Prepare a table for fitlm
tbl = table(slopeOnBetaHML, constant, 'VariableNames', {'slopeOnBetaHML', 'constant'});
% Fit the linear model
mdl = fitlm(tbl, 'slopeOnBetaHML ~ -1 + constant');
display(mdl) %comment: iv) we don't reject that the average slope on betaHML is 0

%conclusion: reject the FF 3-factor model because of i), ii), iii), and iv)

