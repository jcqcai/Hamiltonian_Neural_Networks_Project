import torch
import torch.nn as nn
import kan


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, f=torch.tanh, num_hidden_layers=1):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.f = f

        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[0:-1]:
            x = self.f(layer(x))
        x = self.layers[-1](x)
        return x
    
class HNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, f=torch.tanh, num_hidden_layers=1, use_kan=False, cut_kan=False, grid=3):
        super(HNN, self).__init__()
        if use_kan:
            self.differentiable_model = kan.KAN(width=[2,1,2])#, grid=grid)
            if cut_kan:
                self.differentiable_model.remove_edge(1,0,0)
        else:
            self.differentiable_model = MLP(input_dim, hidden_dim, output_dim, f=f, num_hidden_layers=num_hidden_layers)
        self.M = self.permutation_tensor(input_dim)

    def forward(self, x):

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1,1)

    def time_derivative(self, x):

        F1, F2 = self.forward(x)

        #conservative_field = torch.zeros_like(x)
        solenoidal_field = torch.zeros_like(x)

        #dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0] # gradients for conservative field
        #conservative_field = dF1 @ torch.eye(*self.M.shape)

        dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] # gradients for solenoidal field
        solenoidal_field = dF2 @ self.M.t()

        return solenoidal_field #+ conservative_field

    def permutation_tensor(self, n):
        M = torch.eye(n)
        M = torch.cat([M[n//2:], -M[:n//2]])
        return M
