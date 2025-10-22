import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomAttention(nn.Module):
    def __init__(self, input_dim, return_sequences=True):
        super(CustomAttention, self).__init__()
        self.return_sequences = return_sequences
        self.W = nn.Parameter(torch.Tensor(input_dim[-1], 1))
        self.b = nn.Parameter(torch.Tensor(input_dim[1], 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.b)

    def forward(self, x):
        # print(x.shape)
        e = torch.tanh(torch.matmul(x, self.W) + self.b)
        # print(e.shape)
        a = F.softmax(e, dim=1)
        # print(a.shape)
        output = x * a

        if self.return_sequences:
            return output

        return torch.sum(output, dim=1)

    def extra_repr(self):
        return f'input_dim={self.W.size(0)}, return_sequences={self.return_sequences}'


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers,drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attention = CustomAttention((None, time_step1, 2*hidden_dim), return_sequences=False)  # Use the CustomAttention module
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc1 = nn.Linear(2*hidden_dim, 256)
        self.fc2 = nn.Linear(256,1)
        self.relu = nn.ReLU()

    def forward(self, x):         
        out, (h,c) = self.lstm(x) #, (h0.detach(), c0.detach()))
        out = self.attention(out)
        out = F.relu(out)
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        
        return out



hidden_dim=128
N = 360
L = 288 #segment length
print(N-L)
k = 2
D = (N-L)/(k-1)
print(D)
class LSTMNet2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMNet2, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm_list = nn.ModuleList()
        for i in range(k):
            self.lstm_list.append(nn.Sequential(nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2, bidirectional=True),
                                    CustomAttention((None, L, 2*hidden_dim), return_sequences=False)))
       
        
        self.fusion = FusionNet()
        self.conv1d = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, padding='same')
        self.lstm2 = nn.LSTM(256, 128, 1, batch_first=True, dropout=0.2, bidirectional=True)
        self.attention = CustomAttention((None, k, 2*hidden_dim), return_sequences=False)
        self.fc1 = nn.Linear(2*hidden_dim, 256)
        self.fc2 = nn.Linear(256,1)
        self.bn = nn.BatchNorm1d(2*hidden_dim*k)
        self.relu = nn.ReLU()

   


    def forward(self, x):
        x_k = []
        
        for i in range(k):
            #print(i*D+L)
            x_k.append(x[:,int(i*D):int(i*D+L),:])
        
        outs = []
        for i in range(k):
             out, (h,c) = self.lstm_list[i][0](x_k[i])
            
             out = self.lstm_list[i][1](out)
             out = F.relu(out)
             outs.append(out)
       
        outs = [item.unsqueeze(2) for item in outs]
        out = torch.cat(outs, 2)
        out = out.permute(0,2,1)
        
        out, (h,c) = self.lstm2(out)
        out = self.attention(out)
        out = F.relu(out)
        
        #out = self.fusion(out).permute(0, 2, 1).squeeze(2)
        
        #out = torch.cat(outs,1)
        
        #out = F.relu(self.fc2(F.relu(self.fc1(out))))
        

        return out


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.lstm = nn.LSTM(2*hidden_dim1, hidden_dim1, 1, batch_first=True, dropout=0.2, bidirectional=True)
        self.attention = CustomAttention((None, 2, 2*hidden_dim1), return_sequences=False)
        self.fc1 = nn.Linear(2*128+2*512, 256)
        self.fc2 = nn.Linear(256,1)

    def forward(self, x1, x2):
#         x = torch.cat([x1, x2], dim = 1).view(-1,2,2*hidden_dim1)
#         out, (h,c) = self.lstm(x)
#         out = self.attention(out)
#       out = F.relu(out)
        out = torch.cat([x1, x2], dim = 1)
        out = self.fc2(F.relu(self.fc1(out)))

        return out


time_step1= 360
time_step2= 3

class ParallelNet(nn.Module):
    def __init__(self, input_dim1, hidden_dim1, output_dim, n_layers1, input_dim2, hidden_dim2, n_layers2, drop_prob=0.2):
        super(ParallelNet, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.attention1 = CustomAttention((None, time_step1, 2*hidden_dim1), return_sequences=False)  # Use the CustomAttention module
        self.attention2 = CustomAttention((None, time_step2, 2*hidden_dim2), return_sequences=False)  # Use the CustomAttention module
        self.lstm1 = LSTMNet2(input_dim1, hidden_dim1, output_dim, n_layers1)
        self.lstm2 = LSTMNet(input_dim1, hidden_dim1, output_dim, n_layers1)
        self.lstm3 = LSTMNet(input_dim1, hidden_dim1, output_dim, n_layers1)
        self.lstm4 = LSTMNet(input_dim1, hidden_dim1, output_dim, n_layers1)
        self.fusionNet = FusionNet()
        self.lstm5 = nn.LSTM(input_dim2, hidden_dim2, n_layers2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc1 = nn.Linear(2*hidden_dim2, 256)
        self.fc2 = nn.Linear(256,1)
        self.fc3 = nn.Linear(2,1)
       
    def forward(self, x):
        
        out31 = self.lstm1(x)
        
#         out31 = self.attention1(out31)
#         out31 = F.relu(out31)
        
        out0 = self.lstm2(x)
        out1 = self.lstm3(x)
        out2 = self.lstm4(x)
        
        
        x2 = torch.cat([out0, out1, out2], dim = 1).view(-1,3,1)
        
        out32, (h,c) = self.lstm5(x2)
        out32 = self.attention2(out32)
        out32 = F.relu(out32)
        #out32 = self.fc2(F.relu(self.fc1(out32)))
            
        out3 = self.fusionNet(out31, out32)
        

        return out0,out1,out2,out31,out32,out3



input_dim1 = 300
output_dim = 1
n_layers1 = 1
hidden_dim1=128
hidden_dim2=512
input_dim2 = 1
n_layers2 = 2
model2 = ParallelNet(input_dim1, hidden_dim1, output_dim, n_layers1, input_dim2, hidden_dim2, n_layers2 )
model2 = model2.to(device)
