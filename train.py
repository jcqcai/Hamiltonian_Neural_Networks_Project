import torch
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import kan
from models import *
from data import *


def prep_data_and_model(hamiltonian_fn, seed, input_dim=2, hidden_dim=200, output_dim=2, noise_std=0.0, timescale=10, use_kan=False, cut_kan=False, grid=3):
    print("BUILDING DATA")
    print(hamiltonian_fn.__name__)
    data = get_dataset(hamiltonian_fn, seed+53, noise_std=noise_std, timescale=timescale)
    hnn_model = HNN(input_dim, hidden_dim, output_dim, use_kan=use_kan, cut_kan=cut_kan, grid=grid)
    return data, hnn_model

def L2_loss(u, v):
  return (u-v).pow(2).mean()

def train(data, model, seed, learning_rate=1e-3, epochs=2000, use_kan=False, enforce_sparsity=False, lambda_l1=1e-5, use_adam=False, weight_decay=1e-4):

  torch.manual_seed(seed)
  np.random.seed(seed)

  optim = kan.LBFGS(model.parameters(), lr=learning_rate)
  if use_adam:
     optim = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

  x = torch.tensor( data['x'], requires_grad=True, dtype=torch.float32)
  test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dx'])
  test_dxdt = torch.Tensor(data['test_dx'])

  if use_kan:
    model.differentiable_model(x)

  stats = {'train_loss': [], 'test_loss': []}
  for step in range(epochs):
    
    def sparsity_penalty(model, lambda_l1=lambda_l1):
       l1_norm = sum(p.abs().sum() for p in model.parameters())
       return lambda_l1 * l1_norm
############################    
    def reg(acts_scale):
      small_mag_threshold=1e-16
      small_reg_factor=1.
      def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
          return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

      reg_ = 0.
      lamb_entropy=0
      lamb_coef=0
      lamb_coefdiff=0
      lamb_l1 = 0

      for i in range(len(acts_scale)):
          vec = acts_scale[i].reshape(-1, )

          p = vec / torch.sum(vec)
          l1 = torch.sum(nonlinear(vec))
          entropy = - torch.sum(p * torch.log2(p + 1e-4))
          reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

      # regularize coefficient to encourage spline to be zero
      for i in range(len(model.differentiable_model.act_fun)):
          coeff_l1 = torch.sum(torch.mean(torch.abs(model.differentiable_model.act_fun[i].coef), dim=1))
          coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(model.differentiable_model.act_fun[i].coef)), dim=1))
          reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

      return reg_
#################################    
    def closure():
      optim.zero_grad()
      dxdt_hat = model.time_derivative(x)
      train_loss = L2_loss(dxdt, dxdt_hat) 
      if enforce_sparsity and not use_kan:
        loss = train_loss + sparsity_penalty(model)
      elif enforce_sparsity and use_kan:
         loss = train_loss + lambda_l1 * reg(model.differentiable_model.acts_scale)
      else:
         loss = train_loss
      loss.backward()
      return loss
    
    optim.step(closure)
    dxdt_hat = model.time_derivative(x)
    train_loss = L2_loss(dxdt, dxdt_hat)
    if enforce_sparsity and not use_kan:
      train_loss = train_loss + sparsity_penalty(model)
    elif enforce_sparsity and use_kan:
      train_loss = train_loss + lambda_l1 * reg(model.differentiable_model.acts_scale)
    test_dxdt_hat = model.time_derivative(test_x)
    test_loss = L2_loss(test_dxdt, test_dxdt_hat)
    if enforce_sparsity and not use_kan:
      test_loss = test_loss + sparsity_penalty(model)
    elif enforce_sparsity and use_kan:
      test_loss = train_loss + lambda_l1 * reg(model.differentiable_model.acts_scale)
    stats['train_loss'].append(train_loss.item())
    stats['test_loss'].append(test_loss.item())
    if (step+1) % 100 == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step+1, stats['train_loss'][step], stats['test_loss'][step]))

  train_dxdt_hat = model.time_derivative(x)
  train_dist = (dxdt - train_dxdt_hat)**2
  test_dxdt_hat = model.time_derivative(test_x)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

  return model, stats

def integrate_model(model, t_span=[0, 20], t_eval=None, y0=torch.tensor([1.0, 0.0]), rtol=1e-12):
    print("INTEGRATING...")
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,2)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx
    
    return solve_ivp(fun=fun, t_span=t_span, y0=y0.numpy(), t_eval=t_eval, rtol=rtol)

def plot_loss(s, SEED=0):
    x = range(len(s['train_loss']))
    y = s['train_loss']
    plt.plot(x, y)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Training Loss", fontsize=14)
    plt.yscale("log")
    plt.title("Training Loss vs Epoch", pad=10)
    plt.savefig("train: " + str(SEED) + ".pdf", format="pdf", dpi=600)
    plt.tight_layout()
    plt.show()

    x = range(len(s['test_loss']))
    y = s['test_loss']
    plt.plot(x, y)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Test Loss", fontsize=14)
    plt.yscale("log")
    plt.title("Test Loss vs Epoch", pad=10)
    plt.savefig("test: " + str(SEED) + ".pdf", format="pdf", dpi=600)
    plt.tight_layout()
    plt.show()

def plot_model(hnn_model, kan_model, hamiltonian_fn, t_span=[0, 1], y0=torch.tensor([1.0, 0.0]), rtol=1e-12, SEED=0):
    print(hamiltonian_fn.__name__)
    
    t_eval = np.linspace(t_span[0], t_span[1], 10_000)

    def dynamics_fn_wrapper(t, coords):
        return dynamics_fn(hamiltonian_fn, coords)
    
    true_path = solve_ivp(dynamics_fn_wrapper, t_span=np.array(t_span)*10, t_eval=t_eval, y0=y0.numpy(), rtol=rtol)['y']

    t_span=np.array(t_span)*10
    hnn_ivp = integrate_model(hnn_model, t_span=np.array(t_span)*10, t_eval=t_eval, y0=y0, rtol=rtol)['y']
    kan_ivp = integrate_model(kan_model, t_span=np.array(t_span)*10, t_eval=t_eval, y0=y0, rtol=rtol)['y']

    plt.title("Total Energy")
    plt.xlabel('Time Step')
    hnn_e = [hamiltonian_fn(torch.tensor(c)) for c in hnn_ivp.T[:, 0:2]]
    kan_e = [hamiltonian_fn(torch.tensor(c)) for c in kan_ivp.T[:, 0:2]]
    true_e = [hamiltonian_fn(torch.tensor(c)) for c in true_path.T]
    plt.plot(t_eval, true_e, 'k-', label='Ground Truth', linewidth=2)
    plt.plot(t_eval, hnn_e, 'r-', label='Hamiltonian NN', linewidth=2)
    plt.plot(t_eval, kan_e, 'b-', label='KAN', linewidth=2)
    plt.legend(fontsize=7)
    plt.savefig("Energy: "+ hamiltonian_fn.__name__ + " " + str(SEED) + ".pdf", format="pdf", dpi=600)
    plt.tight_layout()
    plt.show()

    plt.title("Trajectory")
    plt.xlabel('q')
    plt.ylabel('p')
    plt.plot(true_path.T[:, 0], true_path.T[:, 1], 'k-', label='Ground Truth', linewidth=2)
    plt.plot(hnn_ivp.T[:, 0], hnn_ivp.T[:, 1], 'r-', label='Hamiltonian NN', linewidth=2)
    plt.plot(kan_ivp.T[:, 0], hnn_ivp.T[:, 1], 'b-', label='KAN', linewidth=2)
    plt.legend(fontsize=7)
    plt.savefig("Trajectory: "+ hamiltonian_fn.__name__ + " " + str(SEED) + ".pdf", format="pdf", dpi=600)
    plt.tight_layout()
    plt.show()

    plt.title("MSE Error")
    plt.xlabel('Time Step')
    plt.plot(t_eval, np.mean((hnn_ivp - true_path)**2, axis=0), 'r-', label='Hamiltonian NN', linewidth=2)
    plt.plot(t_eval, np.mean((kan_ivp - true_path)**2, axis=0), 'b-', label='KAN', linewidth=2)
    plt.legend(fontsize=7)
    plt.savefig("MSE Error: "+ hamiltonian_fn.__name__ + " " + str(SEED) + ".pdf", format="pdf", dpi=600)
    plt.tight_layout()
    plt.show()


