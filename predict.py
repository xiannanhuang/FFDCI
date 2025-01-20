import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from typing import Optional, Union
from torch import optim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from iTransformer import iTransformer
import numpy as np
import torch
import os
from models.Leddam import Leddam
import yaml

from models.SOFTS import SOFTS
class TimeSeriesDataset(Dataset):
    def __init__(self, df, input_window, output_window):
        self.input_window = input_window
        self.output_window = output_window

        # 确保时间对齐
        # df = df.sort_values(by=['ds', 'unique_id'])
        # pivot_df = df.pivot(index='ds', columns='unique_id', values='y')
        feature_columns = df.columns.difference(['date'])
        self.series = df[feature_columns].values  # 变成二维矩阵 [时间步数, unique_id 数量]

        # 构造滑动窗口
        self.inputs = []
        self.targets = []
        for i in range(len(self.series) - input_window - output_window + 1):
            self.inputs.append(self.series[i:i+input_window, :])  # 窗口的每一维
            self.targets.append(self.series[i+input_window:i+input_window+output_window, :])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)
def format_predictions(df, predictions, input_window, output_window):
    """
    将预测结果从矩阵形式转换为 DataFrame 格式，使用矢量化操作。

    Parameters:
    -----------
    df : pandas.DataFrame
        输入的时间序列数据，包含 `unique_id` 和 `ds` 列。
    predictions : np.ndarray
        预测结果，形状为 [num_windows, output_window, num_series]。
    input_window : int
        输入窗口大小。
    output_window : int
        输出窗口大小。

    Returns:
    --------
    result_df : pandas.DataFrame
        格式化的预测结果 DataFrame，包含 `unique_id`, `ds`, 和 `y_pred`。
    """
    # 获取 unique_id 和 ds 的映射关系
    unique_ids = df['unique_id'].unique()
    series_lengths = df.groupby('unique_id').size().values

    # 计算每个 unique_id 对应的窗口数量
    num_windows_per_series = series_lengths - input_window - output_window + 1
    valid_series_mask = num_windows_per_series > 0  # 过滤有效的时间序列
    unique_ids = unique_ids[valid_series_mask]
    num_windows_per_series = num_windows_per_series[valid_series_mask]

    # 展平 predictions
    predictions_flat = predictions.reshape(-1)

    # 生成 unique_id 列
    unique_id_repeats = np.repeat(
        np.repeat(unique_ids, num_windows_per_series), output_window
    )

    # 修正时间戳生成，确保只保留与预测值对齐的部分
    ds_flat = np.concatenate([
        df[df['unique_id'] == uid]['ds']
        .iloc[input_window:input_window + num_windows_per_series[i] * output_window]
        .values
        for i, uid in enumerate(unique_ids)
    ])

    # 检查长度一致性
    assert len(unique_id_repeats) == len(ds_flat) == len(predictions_flat), (
        f"Inconsistent lengths: unique_id_repeats={len(unique_id_repeats)}, "
        f"ds_flat={len(ds_flat)}, predictions_flat={len(predictions_flat)}"
    )

    # 构建 DataFrame
    result_df = pd.DataFrame({
        'unique_id': unique_id_repeats,
        'ds': ds_flat,
        'y_pred': predictions_flat
    })

    return result_df
def predict(
    model,
    df: pd.DataFrame,
    input_window: int,
    output_window: int,
    batch_size: int = 32,
    device: str = 'cuda:0',
    return_feature=False
) -> tuple[np.ndarray, np.ndarray]:
    """
    根据输入的 DataFrame 使用滑动窗口生成预测结果，并返回预测值和真实值数组。

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch 模型。
    df : pd.DataFrame
        包含 [unique_id, ds, y] 的 DataFrame。
    input_window : int
        输入窗口大小。
    output_window : int
        输出窗口大小。
    batch_size : int, optional (default=32)
        批量大小。
    device : str, optional (default='cpu')
        模型运行设备 ('cpu' 或 'cuda')。

    Returns
    -------
    predictions : np.ndarray
        模型预测的结果，形状为 [num_samples, output_window, num_series]。
    true_values : np.ndarray
        对应的真实值，形状为 [num_samples, output_window, num_series]。
    """
    # 模型设为评估模式
    model.eval()
    model.to(device)

    # 创建数据集和数据加载器
    dataset = TimeSeriesDataset(df, input_window, output_window)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 保存预测结果和真实值
    predictions = []
    true_values = []
    features=[]

    # 遍历数据加载器并进行预测
    if not return_feature:
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.numpy()  # 将真实值转换为 NumPy 数组
                outputs = model(inputs)[output_window].cpu().numpy()  # 假设模型输出形状为 [batch_size, output_window, num_series]

                predictions.append(outputs)
                true_values.append(targets)
        

    # 将所有批次的结果合并
        predictions = np.concatenate(predictions, axis=0)  # [num_samples, output_window, num_series]
        true_values = np.concatenate(true_values, axis=0)  # [num_samples, output_window, num_series]
        return predictions, true_values
    else:
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.numpy()  # 将真实值转换为 NumPy 数组
                r = model(inputs,return_feature=True)  # 假设模型输出形状为 [batch_size, output_window, num_series]
                outputs,feature = r[0][output_window].cpu().numpy(),r[1]  # 假设模型输出形状为 [batch_size, output_window, num_series]
                features.append(feature)
                predictions.append(outputs)
                true_values.append(targets)
        predictions = np.concatenate(predictions, axis=0)  # [num_samples, output_window, num_series]
        true_values = np.concatenate(true_values, axis=0)  # [num_samples, output_window, num_series]
        features=np.concatenate(features,axis=0)
        return predictions, true_values,features


def train_model(model, lr=0.0001, epochs=10, input_window=96, output_window=96, 
                train_df=None, val_df=None, batch_size=32, device='cuda:0', patience=5,series_dim=21):
    """
    Train the model with real-time progress updates and validation, with early stopping based on validation loss.
    Returns the model from the epoch with the best validation loss.
    
    Args:
        model (torch.nn.Module): The model to be trained.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train.
        input_window (int): Length of input sequence.
        output_window (int): Length of output sequence.
        train_df (DataFrame): The training dataset.
        val_df (DataFrame): The validation dataset.
        batch_size (int): Batch size for training.
        device (str): Device to run the model ('cuda:0' or 'cpu').
        patience (int): Number of epochs with no improvement to wait before stopping.
        
    Returns:
        model (torch.nn.Module): The trained model from the epoch with the best validation loss.
        val_loss_history (list): List of validation losses for each epoch.
    """
    # Move model to the appropriate device
    model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()  # Example loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataset = TimeSeriesDataset(train_df, input_window, output_window)
    val_dataset = TimeSeriesDataset(val_df, input_window, output_window)
    
    # Create DataLoader for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Early stopping setup
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    val_loss_history = []  # To store validation losses for each epoch
    best_model_state = None  # To store the best model state
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Using tqdm for real-time progress bar
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for inputs, targets in train_loader:
                # Move data to the appropriate device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)[output_window]
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                running_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)
        
        # Print loss for the epoch
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}')
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)[output_window]
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Calculate the average validation loss
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss= avg_val_loss/(batch_size)
        val_loss_history.append(avg_val_loss)

        # Print validation loss for the epoch
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0  # Reset the counter
            # Save the model state if the validation loss improves
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break  # Stop training early
    
    # Restore the best model from the saved state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model
def init_model(data_name,predicted_len=96,model_name='iTransformer'):
    path=rf'iTransformer_datasets\{data_name}'
    Y_df=pd.read_csv(rf'{path}\{data_name}.csv')
    config=yaml.safe_load(open(f'{path}/config.yaml'))    
    train_size = config['train_size']
    val_size = config['val_size']
    test_size = config['test_size']

    
    Y_df = Y_df.sort_values(by='date')
    feature_columns = Y_df.columns.difference(['date'])
    scaler=StandardScaler()
    Y_df[feature_columns] = scaler.fit_transform(Y_df[feature_columns])
    train_df = Y_df.iloc[:train_size]
    val_df = Y_df.iloc[train_size:train_size + val_size]
    test_df = Y_df.iloc[train_size + val_size:train_size + val_size + test_size]
    if model_name=='iTransformer':
        model = iTransformer(
        num_variates = config['dim'],
        lookback_len = 96,                  # or the lookback length in the paper
        dim = config['dim_head']*4,                          # model dimensions
        depth = 3,                          # depth
        heads = 4,                          # attention heads
        dim_head = config['dim_head'],                      # head dimension
        pred_length =predicted_len,     # can be one prediction, or many
        num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
        use_reversible_instance_norm = True # use reversible instance normalization, proposed here https://openreview.net/forum?id=cGDAkQo1C0p . may be redundant given the layernorms within iTransformer (and whatever else attention learns emergently on the first layer, prior to the first layernorm). if i come across some time, i'll gather up all the statistics across variates, project them, and condition the transformer a bit further. that makes more sense
        )
    elif model_name=='Leddam':
        model=Leddam.Model(config)
    elif model_name=='SOFTS':
        model=SOFTS.Model(config)
    # elif model_name=='SSCNN':
    #     model=SSCNN.Model(config)

    return model,train_df,val_df,test_df,scaler,config
    
if __name__=='__main__':
    # for model_name in ['exchange_rate','ETTh1','ETTh2','weather','ETTm1','ETTm2','electricity','traffic','solar']:

    for data_name in ['solar','PEMS08','PEMS03','PEMS04','PEMS07','traffic']:
        model_name='SOFTS'
        print(data_name)
        length=96
        model,train_df,val_df,test_df,scaler,config=init_model(data_name,predicted_len=96,model_name=model_name)
        # model=Leddam.Model(config)
        device='cuda:1'
        model=train_model(model, train_df=train_df, val_df=val_df,  device=device,epochs=50,series_dim=config['dim'],output_window=96)
        torch.save(model.state_dict(), fr'model_saved\{data_name}\{model_name}96.pth')
 

    
