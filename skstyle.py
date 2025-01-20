import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from tqdm import tqdm  # 引入 tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.base import BaseEstimator
import numpy as np


class UncertaintyRNN(nn.Module, BaseEstimator):
    def __init__(self, input_dim, hidden_size, activation='relu', 
                 learning_rate=0.001, epochs=100, batch_size=32, val_split=0.2,series_dim=21, device=None,num_layer=2, **kwargs):
        super(UncertaintyRNN, self).__init__()
        self.series_dim=series_dim
        self.learning_rate=learning_rate
        self.kwargs=kwargs
        self.epochs=epochs
        self.batch_size=batch_size
        self.val_split=val_split
        self.activation = activation
        self.quantile_target=kwargs.get('quantile_target', [0.05,0.05])
        self.input_dim = input_dim+kwargs.get('embedding_dim', 2)
        self.time_step = kwargs.get('time_step', 96)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnn = nn.LSTM(self.input_dim , hidden_size, batch_first=True,num_layers=num_layer,dropout=kwargs.get('dropout', 0.2)).to(self.device)
        self.fc = nn.Linear(hidden_size, len(self.quantile_target)).to(self.device)  # 输出上下界或不确定性度量
       
        self.embedding = nn.Embedding(self.series_dim,kwargs.get('embedding_dim', 2)).to(self.device)
    def forward(self, feature):
        # x: 输入 (batch_size, seq_len, input_size)
        # feature : (b,time_step,feature_dim)
        # hidden: 隐藏状态 (1, batch_size, hidden_size)
        b=feature.shape[0]
        emb=self.embedding(torch.linspace(0, self.series_dim-1, self.series_dim).long().to(self.device)).reshape(1,self.series_dim,self.kwargs.get('embedding_dim', 2)).repeat(b,1,1) # (b,series_dim, 2)
        feature=torch.cat([feature,emb],dim=-1)  # (b,time_step,feature_dim+2)
        feature=feature.reshape(b*self.series_dim,1,-1).repeat(1,self.time_step,1)  # (b*series_dim,time_step,feature_dim+2)
        # outputs = []
        # batch_size=self.batch_size
       
        # for i in range(0, len(feature), batch_size):
        #     batch = torch.tensor(feature[i:i + batch_size], dtype=torch.float32).to(self.device)
        #     batch_outputs, hidden = self.rnn(batch)
        #     uncertainty = self.fc(batch_outputs)
        #     outputs.append(uncertainty)
        # return torch.concatenate(outputs, axis=0),hidden 
        output, hidden = self.rnn(feature)
        uncertainty = self.fc(output)  # 输出为每个时间步的上下界
        return uncertainty, hidden
    def fit(self, X, y, **kwargs):
        # 优先使用 fit 传入的参数，覆盖实例化时的默认值
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        val_split = kwargs.get('val_split', self.val_split)

        # 数据移动到设备
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).transpose(1, 2).unsqueeze(-1)
        
        # 创建训练和验证集
        dataset = TensorDataset(X, y)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = QuantileLoss(self.quantile_target)  # 分位数损失
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        best_val_loss = float('inf')  # 保存最佳验证损失
        best_epoch = 0
        best_model_state = None  # 保存最佳模型参数
        patience = 5  # 提前停止的容忍次数
        no_improve_count = 0

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            
            # tqdm 进度条显示
            with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
                for batch_X, batch_y in tepoch:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = self(batch_X)[0].reshape(-1, batch_X.shape[1],self.time_step, len(self.quantile_target))
                    # outputs=outputs.cumsum(dim=-2)
                    loss = criterion(outputs, batch_y)
                    v=smooth_indicator(outputs, batch_y)
                    if len(self.quantile_target)==1:
                        l=abs(outputs)
                    else:
                        l=abs(outputs[...,0]-outputs[...,1])
                    regulart_term=correlation_coefficient(v,l)
                    loss=loss+regulart_term*self.kwargs.get('regularization', 1)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())  # 更新进度条显示
            
            # 验证阶段
            self.eval()
            val_loss = 0.0
            c=0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self(batch_X)[0].reshape(-1, batch_X.shape[1], self.time_step, len(self.quantile_target))
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    c+=smooth_indicator(outputs, batch_y).mean().item()*batch_X.shape[0]

            print(f"cover validation': {c/len(val_dataset)}")
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 检查是否有改进
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = self.state_dict()  # 保存最佳模型参数
                no_improve_count = 0  # 重置计数器
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping triggered. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
                    break
            if best_model_state is not None:
                self.load_state_dict(best_model_state)
                print(f"Model parameters restored to best epoch {best_epoch+1}.")
    def predict(self, X, batch_size=2):
        # X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(self.device)
                batch_outputs = self(batch)
                outputs.append(batch_outputs[0].cpu().numpy())
        return np.concatenate(outputs, axis=0)
def smooth_indicator( pred, target):
    # pred (...,len(quantile_target)), target (...,1) 
    # reture (...)
    """ Given a prediction interval C(Xi) = [low,up] or C(xi)=[0,up] for a point Xi, we approximate its coverage
    indicator 1[Y \in C(Xi)] in the following way:
    Vi = tanh(c min(Y-low, up-Y)) where c \in R+ controls the slope of the step function. In the experiments, we set c to be equal
    to 5 10^3. This approximation is differentiable and used in practice
    """
    c=5*torch.tensor(1e3).to(pred.device)
    target=target.squeeze(-1)
    if pred.shape[-1]==1:
        v=torch.tanh(c * torch.min(target,pred[...,0]-target))
        v=0.5*(v+1)
    else:
        low,up=pred[...,0],pred[...,1]
        v=torch.tanh(c * torch.min(target-low,up-target))
        v=0.5*(v+1)
    return v

def correlation_coefficient(x, y):
    # 确保输入是 PyTorch 张量
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    # 确保输入是 1D 向量
    x = x.flatten()
    y = y.flatten()

    # 计算均值
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # 计算协方差
    covariance = torch.mean((x - mean_x) * (y - mean_y))

    # 计算标准差
    std_x = torch.std(x)
    std_y = torch.std(y)

    # 计算相关系数
    correlation = covariance / (std_x * std_y)

    return correlation
class MLPRegressor(nn.Module, BaseEstimator):
    def __init__(self, input_dim, hidden_layers,output_dim,  activation='relu', 
                 learning_rate=0.001, epochs=100, batch_size=32, val_split=0.2,series_dim=21, device=None, **kwargs):
        super(MLPRegressor, self).__init__()
        self.quantile_target=kwargs.get('quantile_target', [0.05,0.05])
        self.input_dim = input_dim+kwargs.get('embedding_dim', 2)
        self.hidden_layers = hidden_layers
        self.output_dim = len(self.quantile_target)*output_dim # 分位数回归需要输出两个值
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.kwargs = kwargs  # 额外参数
        self.series_dim=series_dim

        # 检测设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 构建模型
        self._build_model()
        self.to(self.device)  # 将模型移动到指定设备
        
        self.embedding = nn.Embedding(self.series_dim,kwargs.get('embedding_dim', 2)).to(self.device)
    
    def _build_model(self):
        layers = []
        in_dim = self.input_dim
        for h_dim in self.hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.output_dim))  # 输出层无激活函数
        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        b=X.shape[0]
        emb=self.embedding(torch.linspace(0, self.series_dim-1, self.series_dim).long().to(self.device)).reshape(1,self.series_dim,self.kwargs.get('embedding_dim', 2)).repeat(b,1,1) # (b,series_dim, 2)
        X=torch.cat([X,emb],dim=-1)
        return self.model(X)
    
    def fit(self, X, y, **kwargs):
        # 优先使用 fit 传入的参数，覆盖实例化时的默认值
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        val_split = kwargs.get('val_split', self.val_split)

        # 数据移动到设备
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).transpose(1, 2).unsqueeze(-1)
        
        # 创建训练和验证集
        dataset = TensorDataset(X, y)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = QuantileLoss(self.quantile_target)  # 分位数损失
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        best_val_loss = float('inf')  # 保存最佳验证损失
        best_epoch = 0
        best_model_state = None  # 保存最佳模型参数
        patience = 5  # 提前停止的容忍次数
        no_improve_count = 0

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            
            # tqdm 进度条显示
            with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
                for batch_X, batch_y in tepoch:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = self(batch_X).reshape(-1, batch_X.shape[1], self.output_dim //  len(self.quantile_target),  len(self.quantile_target))
                    # outputs=outputs.cumsum(dim=-2)
                    loss = criterion(outputs, batch_y)
                    v=smooth_indicator(outputs, batch_y)
                    if len(self.quantile_target)==1:
                        l=abs(outputs)
                    else:
                        l=abs(outputs[...,1]-outputs[...,0])
                    regulart_term=correlation_coefficient(v,l)
                    loss=loss+regulart_term*self.kwargs.get('regularization', 1)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())  # 更新进度条显示
            
            # 验证阶段
            self.eval()
            val_loss = 0.0
            c=0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self(batch_X).reshape(-1, batch_X.shape[1], self.output_dim //  len(self.quantile_target),  len(self.quantile_target))
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    c+=smooth_indicator(outputs, batch_y).mean().item()*batch_X.shape[0]

            print(f"cover validation': {c/len(val_dataset)}")

            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 检查是否有改进
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = self.state_dict()  # 保存最佳模型参数
                no_improve_count = 0  # 重置计数器
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping triggered. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
                    break

        # 恢复最佳模型参数
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"Model parameters restored to best epoch {best_epoch+1}.")
    
    def predict(self, X, batch_size=32):
        # X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(self.device)
                batch_outputs = self(batch)
                outputs.append(batch_outputs.cpu().numpy())
        return np.concatenate(outputs, axis=0)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
    
    def forward(self, preds, target):
        assert preds.shape[-1] == len(self.quantiles), "Output dimension must match the number of quantiles"
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i:i+1]
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss

    
   
if __name__ == '__main__':
    import predict,yaml
    # model,train_df,val_df,test_df,scaler=predict.init_model('weather')
    # model.load_state_dict(torch.load('model_saved\weather\model96.pth'))
    # predictions, true_values,features=predict.predict(model,val_df,input_window=96,output_window=96,reture_feature=True)
    # predictions2, true_values2,features2=predict.predict(model,test_df,input_window=96,output_window=96,reture_feature=True)
    # config=yaml.safe_load(open(r'iTransformer_datasets\weather\config.yaml'))
    # reg=UncertaintyRNN(input_dim=config['dim_head']*4,hidden_size=config['dim_head']*8,embedding_dim=32,series_dim=21,quantile_target=[0.05,0.95],num_layer=1,regularization=0)
    # # reg=MLPRegressor(config['dim_head']*4,[config['dim_head']*8],output_dim=96,embedding_dim=32,series_dim=21,quantile_target=[0.05,0.95],regularization=0)
    # reg.fit(features,(true_values-predictions))
    model,train_df,val_df,test_df,scaler=predict.init_model('electricity')
    config=yaml.safe_load(open(r'iTransformer_datasets\electricity\config.yaml'))
    model.load_state_dict(torch.load(r'model_saved\electricity\model96.pth'))
    predictions, true_values,features=predict.predict(model,val_df,input_window=96,output_window=96,reture_feature=True)
    predictions2, true_values2,features2=predict.predict(model,test_df,input_window=96,output_window=96,reture_feature=True)
    reg=UncertaintyRNN(input_dim=config['dim_head']*4,hidden_size=config['dim_head']*8,embedding_dim=32,series_dim=config['dim'],quantile_target=[0.9],num_layer=1,regularization= 0.0,epochs=50)
    reg.fit(features,abs(true_values-predictions),epochs=100,batch_size=4)
