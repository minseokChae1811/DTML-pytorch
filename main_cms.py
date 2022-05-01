import numpy as np
from load import load_cla_data
import copy
import argparse
data_path='D:/[99]Codes/Adv-ALSTM-master/Adv-ALSTM-master/data/stocknet-dataset/price/ourpped'
tra_date='2014-01-02'
val_date='2015-08-03'
tes_date='2015-10-01'
desc = 'the lstm model'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-p', '--path', help='path of pv data', type=str,
                    default='D:/[99]Codes/Adv-ALSTM-master/Adv-ALSTM-master/data/stocknet-dataset/price/ourpped')
parser.add_argument('-l', '--seq', help='length of history', type=int,
                    default=5)
parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                    type=int, default=32)
parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-2,
                    help='alpha for l2 regularizer')
parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
                    help='beta for adverarial loss')
parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-2,
                    help='epsilon to control the scale of noise')
parser.add_argument('-s', '--step', help='steps to make prediction',
                    type=int, default=1)
parser.add_argument('-b', '--batch_size', help='batch size', type=int,
                    default=1024)
parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
parser.add_argument('-r', '--learning_rate', help='learning rate',
                    type=float, default=1e-2)
parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
parser.add_argument('-q', '--model_path', help='path to load model',
                    type=str, default='./saved_model/acl18_alstm/exp')
parser.add_argument('-qs', '--model_save_path', type=str, help='path to save model',
                    default='./tmp/model')
parser.add_argument('-o', '--action', type=str, default='train',
                    help='train, test, pred')
parser.add_argument('-m', '--model', type=str, default='pure_lstm',
                    help='pure_lstm, di_lstm, att_lstm, week_lstm, aw_lstm')
parser.add_argument('-f', '--fix_init', type=int, default=0,
                    help='use fixed initialization')
parser.add_argument('-a', '--att', type=int, default=1,
                    help='use attention model')
parser.add_argument('-w', '--week', type=int, default=0,
                    help='use week day data')
parser.add_argument('-v', '--adv', type=int, default=0,
                    help='adversarial training')
parser.add_argument('-hi', '--hinge_lose', type=int, default=1,
                    help='use hinge lose')
parser.add_argument('-rl', '--reload', type=int, default=0,
                    help='use pre-trained parameters')

args = parser.parse_args()

parameters = {
    'seq': int(args.seq),
    'unit': int(args.unit),
    'alp': float(args.alpha_l2),
    'bet': float(args.beta_adv),
    'eps': float(args.epsilon_adv),
    'lr': float(args.learning_rate)
}
paras = copy.copy(parameters)
tra_pv, tra_wd, tra_gt, \
val_pv, val_wd, val_gt, \
tes_pv, tes_wd, tes_gt  = load_cla_data(data_path,
    tra_date, val_date, tes_date, seq=paras['seq'])
    
import torch
import torch.nn as nn
x_train = torch.from_numpy(tra_pv).type(torch.Tensor)
x_test = torch.from_numpy(tes_pv).type(torch.Tensor)
y_train_lstm = torch.from_numpy(tra_gt).type(torch.Tensor)
y_test_lstm = torch.from_numpy(tes_gt).type(torch.Tensor)
# y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
# y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)    

input_dim = 11
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100    
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out    
    
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)   

import time
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train_lstm)
    y_pred_tag = torch.round(y_train_pred)
    correct_results_sum = (y_pred_tag == y_train_lstm).sum().float()
    acc = correct_results_sum/y_train_pred.shape[0]
    acc = torch.round(acc * 100)
    print("Epoch ", t, "MSE: ", loss.item(), "acc", acc.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time)) 
    
    
    