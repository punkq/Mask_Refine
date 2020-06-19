import torch.nn as nn
import torch

from config import parser

class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, pmode='reflect'):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        
        self.fc = nn.Conv1d(state_dim, out_state_dim, 
                            kernel_size=self.n_adj*2+1,
                            padding=self.n_adj,
                            padding_mode=pmode)

    def forward(self, input):
        # input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)


class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1, pmode='reflect'):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        
        self.fc = nn.Conv1d(state_dim, out_state_dim, 
                            kernel_size=self.n_adj*2+1, 
                            dilation=self.dilation,
                            padding=self.n_adj*self.dilation,
                            padding_mode=pmode)

    def forward(self, input):
        # if self.n_adj != 0:
        #     input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
        return self.fc(input)


_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}


class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1, pmode='reflect'):
        super(BasicBlock, self).__init__()

        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation, pmode)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', pmode='reflect'):
        super(Snake, self).__init__()

        self.head = BasicBlock(feature_dim, state_dim, conv_type, pmode=pmode)

        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[i], pmode=pmode)
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x):
        states = []

        x = self.head(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x
    

if __name__ == "__main__":
    print("test snake network:")
    for k in [16,32,64,128,256]:
        net = Snake(state_dim=3+2, feature_dim=k, conv_type='dgrid')
        print('feature dimension='+str(k)+' parameters:', sum(param.numel() for param in net.parameters()))
    print(Snake(state_dim=3+2, feature_dim=128, conv_type='dgrid'))