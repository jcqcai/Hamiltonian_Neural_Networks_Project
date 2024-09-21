import torch
import autograd
import autograd.numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def spring_hamiltonian(coords):
    q, p = np.split(coords, 2)
    H = p**2 + q**2 + 1*q
    return H

def dynamics_fn(hamiltonian_fn, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S

def get_trajectory(hamiltonian_fn, t_span=[0, 3], timescale=10, radius=None, y0=None, noise_std=0.0):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    if y0 is None:
        y0 = np.random.rand(2)*2-1
    if radius is None:
        if hamiltonian_fn == spring_hamiltonian:
            radius = np.random.rand()*0.9 + 0.1
    y0 = y0 / np.sqrt((y0**2).sum()) * radius

    def dynamics_fn_wrapper(t, coords):
        return dynamics_fn(hamiltonian_fn, coords)

    spring_ivp = solve_ivp(fun=dynamics_fn_wrapper, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn_wrapper(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)
    
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    return q, p, dqdt, dpdt, t_eval

def get_dataset(hamiltonian_fn, seed=0, samples=50, test_split=0.5, noise_std=0.0, timescale=10):
    data = {}

    np.random.seed(seed)

    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(hamiltonian_fn, timescale=timescale, noise_std=noise_std)
        xs.append( np.stack( [x, y]).T )
        dxs.append( np.stack( [dx, dy]).T )
        
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def show_traj(hamiltonian_fn, seed=0, radius=None, y0=torch.tensor([1.0, 0.0]), noise_std=0.0):
    if hamiltonian_fn == spring_hamiltonian:
        radius = 1.0
    q, p, dqdt, dpdt, t = get_trajectory(hamiltonian_fn, radius=radius, y0=y0.numpy(), noise_std=noise_std)

    x = q
    dx = p

    fig = plt.figure(figsize=(3, 3), facecolor='white')

    plt.scatter(x, dx, c=t, s=14, label='start')
    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("p", rotation=0, fontsize=14)
    plt.title("Phase Space")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def show_pure_traj(hamiltonian_fn, y0=torch.tensor([1.0, 0.0])):
    show_traj(hamiltonian_fn, y0=y0, noise_std=0.0)

def show_noisy_traj(hamiltonian_fn, y0=torch.tensor([1.0, 0.0])):
    show_traj(hamiltonian_fn, y0=y0, noise_std=0.1)
