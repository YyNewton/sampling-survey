import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from scipy.optimize import minimize
from scipy.special import expit  

# 非概率样本（问卷星）S_A
n_A = 73

gender_A = np.array([0]*36 + [1]*37)            
grade_A = np.array([1]*25 + [2]*40 + [3]*3 + [4]*3 + [5]*2) 
origin_A = np.array([1]*42 + [0]*31)             
expense_A = np.array([800]*2 + [1250]*14 + [2000]*32 + [3000]*25) 


np.random.seed(2026)
np.random.shuffle(gender_A)
np.random.shuffle(grade_A)
np.random.shuffle(origin_A)
np.random.shuffle(expense_A)

S_A = pd.DataFrame({
    'gender': gender_A,
    'grade': grade_A,
    'origin': origin_A,
    'expense': expense_A
})

# 概率样本 S_B
n_B = 10000
S_B = pd.DataFrame({
    'gender': np.random.binomial(1, 0.50, n_B),       
    'grade': np.random.choice([1, 2, 3, 4, 5], n_B, p=[0.22, 0.22, 0.22, 0.22, 0.12]), 
    'origin': np.random.binomial(1, 0.30, n_B),       
    'expense': np.nan,                                 
    'weight': 1.0  
})

## DR

def doubly_robust_estimate(sample_A, sample_B, x_cols, y_col):
    # propensity score
    X_A = np.c_[np.ones(len(sample_A)), sample_A[x_cols].values]
    X_B = np.c_[np.ones(len(sample_B)), sample_B[x_cols].values]
    

    weights_B = sample_B['weight'].values if 'weight' in sample_B.columns else np.ones(len(sample_B))
    
    # 伪对数似然
    def negative_pseudo_log_likelihood(theta):
        term1 = np.sum(X_A @ theta)
        pi_B = expit(X_B @ theta)
        pi_B = np.clip(pi_B, 1e-15, 1 - 1e-15) # 防止 log(0)
        term2 = np.sum(weights_B * np.log(1 - pi_B))
        return -(term1 + term2)
    
    def gradient(theta):
        pi_B = expit(X_B @ theta)
        grad1 = np.sum(X_A, axis=0)
        grad2 = np.dot(weights_B * pi_B, X_B)
        return -(grad1 - grad2)
    
    initial_theta = np.zeros(X_A.shape[1])
    res = minimize(
        fun=negative_pseudo_log_likelihood, 
        x0=initial_theta, 
        jac=gradient, 
        method='BFGS'
    )
    
    best_theta = res.x
    
    p_A = expit(X_A @ best_theta)
    p_A = np.clip(p_A, 0.01, 0.99) # 截断防止极端权重
    
    # 计算逆概率权重 (IPW)
    weights_A = 1.0 / p_A    
    
    # 结果回归模型
    or_model = LinearRegression()
    or_model.fit(sample_A[x_cols], sample_A[y_col])
    
    y_hat_A = or_model.predict(sample_A[x_cols])
    y_hat_B = or_model.predict(sample_B[x_cols])
    
    # DR 
    mu_B = np.average(y_hat_B, weights=weights_B) 
    
    normalized_weights_A = weights_A / np.mean(weights_A)
    residuals_A = sample_A[y_col] - y_hat_A
    correction = np.mean(normalized_weights_A * residuals_A) 
    
    return mu_B + correction

# Bootstrap
x_columns = ['gender', 'grade', 'origin']
y_column = 'expense'


naive_mean = S_A['expense'].mean()
dr_mean = doubly_robust_estimate(S_A, S_B, x_columns, y_column)

print("="*40)
print(f"【样本量】: {n_A} 人")
print(f"【朴素估计】直接对问卷消费求均值: {naive_mean:.2f} 元")
print(f"【DR 估计】修正样本偏差后的均值: {dr_mean:.2f} 元")
print("="*40)

bootstrap_estimates = []
for i in range(100):
    boot_A = resample(S_A, replace=True, n_samples=len(S_A))
    boot_B = resample(S_B, replace=True, n_samples=len(S_B))
    boot_est = doubly_robust_estimate(boot_A, boot_B, x_columns, y_column)
    bootstrap_estimates.append(boot_est)

lower_bound = np.percentile(bootstrap_estimates, 2.5)
upper_bound = np.percentile(bootstrap_estimates, 97.5)

print(f"【最终结论】上海大学生平均月消费的 DR 估计值为: {dr_mean:.2f} 元")
print(f"【置信区间】95% 的置信区间为: [{lower_bound:.2f} 元, {upper_bound:.2f} 元]")