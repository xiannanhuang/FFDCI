import pandas as pd
from iTransformer import iTransformer
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import os
import predict
import matplotlib.pyplot as plt
import lightgbm as lgb
import yaml
from tqdm import tqdm
def conformal_prediction(predictions, true_values,predictions2, true_values2,alpha,each_feature=False,each_step=False,input_size=96,output_size=96):
    '''
    predictions: prediction in calibaration dataset
    true_values: true value in calibaration dataset
    alpha: 置信水平
    each_feature: 是否对每个特征进行预测
    each_step: 是否对每个时间步进行预测
    predictions2: prediction in test dataset
    true_values2: true value in test dataset
    '''
    feature_size=true_values.shape[-1]
    residuals = abs(true_values - predictions)  #(n,output_size,feature_size)
    if each_feature and each_step:
        q=np.quantile(residuals, 1-alpha, axis=0) #(output_size,feature_size)
        low_bound= predictions2-q.reshape(-1,output_size,feature_size).repeat(predictions2.shape[0],axis=0)
        upper_bound= predictions2+q.reshape(-1,output_size,feature_size).repeat(predictions2.shape[0],axis=0)

    elif each_feature:
        q=np.quantile(residuals, 1-alpha, axis=(0,1)) #(feature_size)
        low_bound= predictions2-q.reshape(-1,1,feature_size).repeat(predictions2.shape[0],axis=0).repeat(output_size,axis=1)
        upper_bound= predictions2+q.reshape(-1,1,feature_size).repeat(predictions2.shape[0],axis=0).repeat(output_size,axis=1)
        
    elif each_step:
        q=np.quantile(residuals, 1-alpha, axis=(0,2)) #(output_size)
        low_bound= predictions2-q.reshape(-1,output_size,1).repeat(predictions2.shape[0],axis=0).repeat(feature_size,axis=2)
        upper_bound= predictions2+q.reshape(-1,output_size,1).repeat(predictions2.shape[0],axis=0).repeat(feature_size,axis=2)
    else:
        q=np.quantile(residuals, 1-alpha)
        low_bound = predictions2 - q
        upper_bound = predictions2 + q
    print(q.shape)
    return low_bound,upper_bound
def report(low_bound,upper_bound,true_values):
    '''
    low_bound: lower bound of prediction interval (n,output_size,feature_size)
    upper_bound: upper bound of prediction interval
    true_values: true value in test dataset
    '''
    # 计算预测区间覆盖真实值的比例
    covered = np.logical_and(true_values >= low_bound, true_values <= upper_bound)
    coverage_all = np.mean(covered)
    print(f'Coverage: {coverage_all:.4f}')
    # 计算平均预测区间长度
    interval_length = np.mean(upper_bound - low_bound)
    print(f'Average Interval Length: {interval_length:.4f}')
    min_cov_feature=1
    for i in range(true_values.shape[-1]):
        covered = np.logical_and(true_values[...,i] >= low_bound[...,i], true_values[...,i] <= upper_bound[...,i])
        coverage = np.mean(covered)
        if coverage<min_cov_feature:
            min_cov_feature=coverage
    print(f'Min Coverage feature: {min_cov_feature:.4f}')
    min_cov_step=1
    for i in range(true_values.shape[1]):
        covered = np.logical_and(true_values[:,i,:] >= low_bound[:,i,:], true_values[:,i,:] <= upper_bound[:,i,:])
        coverage = np.mean(covered)
        if coverage<min_cov_step:
            min_cov_step=coverage
    print(f'Min Coverage step: {min_cov_step:.4f}')
    result={
        'Coverage': coverage_all,
        'Average Interval Length': interval_length,
        'Min Coverage feature': min_cov_feature,
        'Min Coverage step': min_cov_step
    }
    return pd.DataFrame(result, index=[0])
def smooth_residuals(residuals, gamma=0.9):
    """
    Apply exponential smoothing to residuals.

    Parameters:
    - residuals: np.ndarray, shape (n, feature_dim), residuals to be smoothed.
    - gamma: float, smoothing parameter in [0, 1].

    Returns:
    - smoothed_residuals: np.ndarray, shape (n, feature_dim), smoothed residuals.
    """
    n, feature_dim = residuals.shape
    smoothed_residuals = np.zeros_like(residuals)

    for j in range(feature_dim):
        for i in range(1, n):
            smoothed_residuals[i, j] = gamma * smoothed_residuals[i - 1, j] + (1 - gamma) * residuals[i, j]
    
    return smoothed_residuals
  def get_fit_x_y(residuals,feature_identify,step_indentifiy,window_size):
    """
    Get the x and y values for fitting the quantile model.

    Parameters:
    - residuals: np.ndarray, shape (n,n_window, feature_dim), residuals 
    - feature_identify: np.ndarray, shape (feature_dim), residuals 
    - window_size: int, window size for fitting.
    - step_indentifiy: np.ndarray, shape (n_window), residuals

    Returns:
    - x: np.ndarray, shape ((n - window_size)*n_window*feature_dim, window_size+1), x values for fitting.
    - y: np.ndarray, shape (n - window_size)*n_window*feature_dim), y values for fitting.

    """
    n, n_window,feature_dim = residuals.shape
    x = []
    y = []
    for i in range(len(residuals) - window_size):
        x_this=residuals[i:i+window_size,0] # shape (window_size, feature_dim)
        x_this=x_this.reshape(window_size,1,feature_dim).repeat(n_window,1)# shape (window_size,n_window,feature_dim)
        x_this=np.concatenate([x_this,feature_identify.reshape(1,1,-1).repeat(n_window,1)],axis=0) # shape (window_size+1,n_window,feature_dim)
        x_this=np.concatenate([x_this,step_indentifiy.reshape(1,-1,1).repeat(feature_dim,-1)],axis=0).T # shape (feature_dim,n_window,window_size+2)
        y_this=residuals[i+window_size,:].reshape(-1) # shape (n_window*feature_dim,)
        
        x.append(x_this.reshape(-1,x_this.shape[-1]))
        y.append(y_this)
    x=np.concatenate(x,axis=0) # shape ((n - window_size)*feature_dim, window_size+2)
    y=np.concatenate(y,axis=0) # shape (n - window_size)*feature_dim)
    # print(x.shape)
    return x,y

def LPCI_with_dynamic_quantile_update(
    val_predictions, val_true_values, test_predictions, test_true_values,
    quantile_model_upper, quantile_model_lower,
    alpha=0.1, w=10, gamma=0.9, k=5, use_smoothing=False,training_size=100):
    """
    LPCI with optional residual smoothing and dynamic quantile model updates.

    Parameters:
    - val_predictions: np.ndarray, shape (n_val, dim_features), validation predictions.
    - val_true_values: np.ndarray, shape (n_val,), true values for validation data.
    - test_predictions: np.ndarray, shape (n_test, dim_features), test predictions.
    - test_true_values: np.ndarray, shape (n_test,), true values for test data.
    - quantile_model_upper: sklearn-like regressor, for upper quantile estimation.
    - quantile_model_lower: sklearn-like regressor, for lower quantile estimation.
    - alpha: float, significance level (e.g., 0.1 for 90% intervals).
    - w: int, window size for residual history.
    - gamma: float, smoothing parameter in [0, 1].
    - k: int, interval for updating quantile models.
    - use_smoothing: bool, whether to use smoothed residuals for training and prediction.

    Returns:
    - lowbound, upperbound: np.ndarray, shape (n_test,), lower and upper bounds of the 1-alpha confidence interval.
    """
    n_val,n_window, dim_features = val_predictions.shape
    n_test = test_predictions.shape[0]

    # Step 1: Compute residuals for validation data
    val_residuals = val_true_values - val_predictions  # Shape: (n_val, ,n_window,dim_features)
    
    # Apply smoothing if enabled
    if use_smoothing:
        val_residuals = smooth_residuals(val_residuals, gamma=gamma)
    
    # Step 2: Initialize feature identifiers
    # feature_identifiers = np.array([random.random() for _ in range(dim_features*n_window)])  # Shape: (dim_features*n_window,)
    feature_identifiers=np.array([random.random() for _ in range(dim_features)])
    step_identifiers=np.array([random.random() for _ in range(n_window)])
    
    # Step 3: Prepare smoothed residual windows with identifiers for validation data
     # Step 3: Prepare the training dataset for the quantile models
    x,y=get_fit_x_y(val_residuals[-training_size:],feature_identifiers,step_identifiers,w)
    # Step 4: Train quantile models for each feature
    quantile_model_lower.fit(x,y)  # Train the lower quantile model
    quantile_model_upper.fit(x,y)  # Train the upper quantile model
    lowbounds,upperbounds=[],[]
    for i in tqdm(range(n_test), desc="Processing Test Data", ncols=100):  # Add a progress bar with tqdm
       # Step 5: Compute residuals for test data
        test_residuals = test_true_values[i] - test_predictions[i]  # Shape: (dim_features,)
        residual_to_use = val_residuals[-w:,0].reshape(w,1,-1).repeat(n_window,1) # Shape: (w,n_windows, dim_features)
        x_to_use=np.concatenate([residual_to_use,feature_identifiers.reshape(1,1,-1).repeat(n_window,1)],axis=0) # Shape: (w+1,n_window, dim_features)
        x_to_use=np.concatenate([x_to_use,step_identifiers.reshape(1,-1,1).repeat(dim_features,-1)],axis=0).reshape(w+2,-1) # Shape: (w+2,n_window, dim_features)
        x_to_use=x_to_use.T # Shape: ( dim_features,w+2)
        # Step 6: Predict quantiles for the next residual
 
        lowbound = test_predictions[i] + quantile_model_lower.predict(x_to_use).reshape(n_window,-1)  # Shape: (dim_features,)
        upperbound = test_predictions[i] + quantile_model_upper.predict(x_to_use).reshape(n_window,-1)  # Shape: (dim_features,)
        lowbounds.append(lowbound)
        upperbounds.append(upperbound)


        val_residuals= np.concatenate((val_residuals,test_residuals.reshape(1,n_window,dim_features)),axis=0) # Shape: (n_val+1, dim_features)
        if (i+1)%k==0:
            x,y=get_fit_x_y(val_residuals[-training_size:],feature_identifiers,step_identifiers,w)
            quantile_model_lower.fit(x,y)  # Train the lower quantile model
            quantile_model_upper.fit(x,y)  # Train the upper quantile model
    return np.array(lowbounds).reshape(n_test, n_window,-1), np.array(upperbounds).reshape(n_test, n_window,-1)
def adp_inter_with_nn(predictions, true_values, errors, lr):
    # predictions: (b, time_step, series_dim)
    n, t, d =errors.shape

    # 初始化调整矩阵
    adjust_matrix = np.zeros((t, d))
    lowbound_adj = np.zeros_like(errors)
    upperbound_adj = np.zeros_like(errors)
    lowbound_adj[:t,:]=predictions[:t,:]-errors[:t,:]
    upperbound_adj[:t,:]=predictions[:t,:]+errors[:t,:]
    idx = np.arange(t)
    for i in tqdm(range(t,n), desc="Processing Test Data", ncols=100):
        # 当前的调整矩阵广播到 (t, d) 形状
        # adjust_broadcast = np.tile(adjust_matrix, (1, 1))
        adjust_broadcast=adjust_matrix
        
        # 计算新的 lowbound_adj 和 upperbound_adj
        lowbound_adj[i] = predictions[i] - adjust_broadcast-errors[i]
        upperbound_adj[i] =predictions[i] + adjust_broadcast+errors[i]

        # 计算误差范围
        # target = true_values[max(0, i - t + 1):i + 1]  # 只取有效的索引范围
        target =np.flip(true_values[ i - t :i ], axis=0)[idx, idx, :]
        l = lowbound_adj[i - t :i ] 
        l=np.flip(l, axis=0)[idx, idx, :]
        u = upperbound_adj[ i - t :i ] 
        u=np.flip(u, axis=0)[idx, idx, :]

        # 计算 err 矩阵
        err = ~((target > l) & (target < u))

        # 更新 adjust_matrix
        adjust_matrix +=(err * 0.9 - (1 - err) * 0.1) * lr
        # print((true_values[max(0, i - t + 1):i + 1] ).shape)

    return upperbound_adj, lowbound_adj, adjust_matrix
  if __name__=='main':
    data_name='traffic'
    model_name='SOFTS'
    length=96
    device='cuda:0'
    model,train_df,val_df,test_df,scaler,config=predict.init_model(data_name,length,model_name)
    # model.load_state_dict(torch.load(fr'model_saved\{data_name}\model{length}.pth'))
    model.load_state_dict(torch.load(fr'model_saved\{data_name}\{model_name}96.pth'))
    predictions, true_values,features=predict.predict(model,val_df,input_window=96,output_window=length,return_feature=True,device=device)
    predictions2, true_values2,features2=predict.predict(model,test_df,input_window=96,output_window=length,return_feature=True,device=device)
    ####### proposed method
    reg=UncertaintyRNN(input_dim=config['dim_head']*4,hidden_size=config['dim_head']*8,embedding_dim=32,series_dim=config['dim'],quantile_target=[0.9],time_step=length,num_layer=1,regularization= 0.0,epochs=50,device=device)
    reg.fit(features,abs(true_values-predictions),epochs=100,batch_size=16)
    p=reg.predict(features2).reshape(-1,config['dim'],length,1)
    error=abs(p[...,0].transpose(0,2,1))
    p2,l2,adj= adp_inter_with_nn(predictions2,true_values2,error,lr)
    report(l2[:],p2[:],true_values2[:,:,:])
    reg=MLPRegressor(config['dim_head']*4,[config['dim_head']*8,config['dim_head']*2],output_dim=length,embedding_dim=32,series_dim=config['dim'],quantile_target=[0.9],num_layer=1,regularization= 0.0,epochs=100)
    reg.fit(features,abs(true_values-predictions),epochs=100,batch_size=32)
    p=reg.predict(features2).reshape(-1,config['dim'],length,1)
    error=abs(p[...,0].transpose(0,2,1))
    lowbound=predictions2-error
    upperbound=predictions2+error
    p2,l2,adj= adp_inter_with_nn(predictions2,true_values2,error,lr)
    report(l2[:],p2[:],true_values2[:,:,:])
    #####LPCI########
    lowbound2,upbound2=LPCI_with_dynamic_quantile_update(predictions, true_values,predictions2[:], true_values2[:],k=1000,
                                                   quantile_model_lower=lgb.LGBMRegressor(objective='quantile', alpha=0.05),
                                                   quantile_model_upper=lgb.LGBMRegressor(objective='quantile', alpha=0.95),training_size=200,w=20)
    report(lowbound2,upbound2,true_values2[:,:,:])
    ####conformal_prediction
    lowbound,upbound=conformal_prediction(predictions, true_values,predictions2, true_values2,alpha=0.1,each_feature=1,each_step=1,input_size=96,output_size=length)
    report(lowbound[:,:,:],upbound[:,:,:],true_values2[:,:,:])
