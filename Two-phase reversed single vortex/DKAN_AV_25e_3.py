import numpy as np
import os
import random
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import scipy.io
from smt.sampling_methods import LHS
from util import fwd_gradients
import time

torch.cuda.empty_cache()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", device)

# 设置随机种子以确保结果可复现
init_seed = 42
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
random.seed(init_seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value

        self.DyT_alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.DyT_weight = nn.Parameter(torch.ones(normalized_shape))
        self.DyT_bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.DyT_alpha * x)
        x = x * self.DyT_weight + self.DyT_bias
        return x

class PirateNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(PirateNetBlock, self).__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)  # First dense layer
        self.dyt1 = DynamicTanh(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)  # Second dense layer
        self.dyt2 = DynamicTanh(hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)  # Third dense layer
        self.dyt3 = DynamicTanh(hidden_dim)
        self.alpha = nn.Parameter(torch.zeros(1))  # Parameter alpha for blending

    def forward(self, x, u, v):
        f = self.dyt1(self.dense1(x))  # Apply activation after first layer
        z1 = f * u + (1 - f) * v  # Blend with u and v
        g = self.dyt2(self.dense2(z1))  # Apply activation after second layer
        z2 = g * u + (1 - g) * v  # Blend again with u and v
        h = self.dyt3(self.dense3(z2))  # Apply activation after third layer
        return self.alpha * h + (1 - self.alpha) * x  # Final output after blending

class PirateNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks, hidden_dim=128, s=1.0):
        super(PirateNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.s = s

        # Embedding matrix B for feature transformation
        self.B = nn.Parameter(torch.randn(input_dim, hidden_dim // 2) * s)
        self.embedding = lambda x: torch.cat(
            [torch.cos(torch.matmul(x, self.B)), torch.sin(torch.matmul(x, self.B))], dim=-1
        )

        # List of PirateNetBlock layers
        self.blocks = nn.ModuleList([PirateNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.U = nn.Linear(hidden_dim, hidden_dim)  # U layer for transformations
        self.V = nn.Linear(hidden_dim, hidden_dim)  # V layer for transformations
        self.dyt_u = DynamicTanh(hidden_dim)
        self.dyt_v = DynamicTanh(hidden_dim)
        # Final output layer
        self.final_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        print(self.final_layer.weight.data.shape)  # Print weight shape for debugging

        self.initialize_weights()  # Initialize weights
        self.visc = nn.Parameter(torch.tensor(0.05))

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)  # Xavier initialization for weights
                if module.bias is not None:
                    module.bias.data.zero_()  # Initialize biases to zero

    def forward(self, x):
        x = self.embedding(x)  # Apply embedding transformation
        u = self.dyt_u(self.U(x))  # Apply transformation for u
        v = self.dyt_v(self.V(x))  # Apply transformation for v
        for block in self.blocks:
            x = block(x, u, v)  # Pass through blocks
        return torch.tanh(self.final_layer(x))  # Return the final output

    def initialize_last_layer(self, Y, input_data):
        phi = self.embedding(input_data)  # Apply embedding to input data
        W = torch.linalg.lstsq(phi, Y).solution  # Solve for weights using least squares
        print(W.shape, self.final_layer.weight.data.shape)  # Debugging shapes
        self.final_layer.weight.data = W.T  # Set final layer weights to the solution

class PINN:
    def __init__(self, device='cuda', seed=42):
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.log = {'loss': [], 'loss_ac_ch': [], 'loss_bc': [], 'loss_ic': [], 'AV': []}

        U0 = 2.0  # 特征速度，可自行修改
        L0 = 1.0  # 特征长度
        T0 = 2.0  # 特征时间
        self.t_start=0.0
        self.t_end=1.0

        self.u_fun = lambda t, x, y: -U0 * torch.sin(torch.pi * x / L0) ** 2 * torch.sin(torch.pi * y / L0) * torch.cos(torch.pi * y / L0) * torch.cos(torch.pi * t / T0)
        self.v_fun = lambda t, x, y: U0 * torch.sin(torch.pi * x / L0) * torch.cos(torch.pi * x / L0) * torch.sin(torch.pi * y / L0) ** 2 * torch.cos(torch.pi * t / T0)

        self.threshold_phi = torch.tensor(0.8)
        self.sample_data0()  # 先pde采样
        self.model = PirateNet(input_dim=3,output_dim=1,num_blocks=1,hidden_dim=128).to(device)

        t_vals = np.linspace(self.t_start, self.t_end, 3)  # 101 points in the t direction
        x_vals = np.linspace(0.0, 1.0, 251)  # 1001 points in the x direction
        y_vals = np.linspace(0.0, 1.0, 251)  # 1001 points in the y direction
        self.y_grid, self.x_grid, self.t_grid = np.meshgrid(y_vals, x_vals, t_vals, indexing="ij") #y,x,t
        val_data_pde = np.vstack([self.t_grid.ravel(), self.x_grid.ravel(), self.y_grid.ravel()]).T
        self.val_data_pde = torch.tensor(val_data_pde, dtype=torch.float32).to(device)

    def sample_data0(self):  # 无量纲化时间空间
        with torch.no_grad():
            t_vals = np.linspace(self.t_start, self.t_end, 101)  # 101 points in the t direction
            x_vals = np.linspace(0.0, 1.0, 201)  # 1001 points in the x direction
            y_vals = np.linspace(0.0, 1.0, 201)  # 1001 points in the y direction

            y_grid, x_grid, t_grid = np.meshgrid(y_vals, x_vals, t_vals, indexing='ij')
            # self.train_data_pde = np.vstack([t_grid.ravel(), x_grid.ravel(), y_grid.ravel()]).T

            # 基于 x_vals 与 y_vals 的 mesh 在 t=0 处生成初值点
            y_grid_ic, x_grid_ic = np.meshgrid(y_vals, x_vals, indexing='ij')
            t_zeros_ic = np.zeros_like(x_grid_ic)
            ic_mesh = np.vstack([t_zeros_ic.ravel(), x_grid_ic.ravel(), y_grid_ic.ravel()]).T
            cx, cy = 0.5, 0.75
            r0 = 0.15
            angles = np.random.uniform(0.0, 2*np.pi, 2000)
            r = r0 + np.random.uniform(-0.02, 0.02, 2000)
            x_circ = cx + r * np.cos(angles)
            y_circ = cy + r * np.sin(angles)
            t_circ = np.zeros_like(x_circ)
            ic_circle = np.vstack([t_circ, x_circ, y_circ]).T
            train_data_ic_np = np.vstack([ic_mesh, ic_circle])
            train_data_ic_np = np.unique(train_data_ic_np, axis=0)
            self.train_data_ic = train_data_ic_np  # 形状为 (n, 3): [t, x, y]，其中 t=0
            # plt.scatter(train_data_ic_np[:,1], train_data_ic_np[:,2], s=1, alpha=0.5)  # s点大小, alpha透明度
            # plt.show()
            # Boundary conditions (merge four sides into one array)
            t_grid_lr, y_grid_lr = np.meshgrid(t_vals, y_vals)
            bc_left_np = np.vstack([t_grid_lr.ravel(),  0.0 * np.ones_like(t_grid_lr).ravel(), y_grid_lr.ravel()]).T
            bc_right_np = np.vstack([t_grid_lr.ravel(), 1.0 * np.ones_like(t_grid_lr).ravel(), y_grid_lr.ravel()]).T

            t_grid_tb, x_grid_tb = np.meshgrid(t_vals, x_vals)
            bc_bottom_np = np.vstack([t_grid_tb.ravel(), x_grid_tb.ravel(), 0.0 * np.ones_like(t_grid_tb).ravel()]).T
            bc_top_np    = np.vstack([t_grid_tb.ravel(), x_grid_tb.ravel(), 1.0 * np.ones_like(t_grid_tb).ravel()]).T

            bound_np = np.vstack([bc_left_np, bc_right_np, bc_bottom_np, bc_top_np])
            bound_np = np.unique(bound_np, axis=0)

            # self.train_data_pde = torch.tensor(self.train_data_pde, dtype=torch.float32).to(device)
            self.train_data_bound = torch.tensor(bound_np, dtype=torch.float32).to(device)
            self.train_data_ic = torch.tensor(self.train_data_ic, dtype=torch.float32).to(device)

            r2 = (self.train_data_ic[:,1:2] - 0.5)**2 + (self.train_data_ic[:,2:3] - 0.75)**2
            self.train_data_ic_phi_exact = torch.where(r2 <= (0.15**2), torch.tensor(-1.0, device=self.device), torch.tensor(1.0, device=self.device))


    # plt.figure(figsize=(6, 4))
    # plt.hist(self.train_data_ic[:, 1], bins=40, edgecolor='k', alpha=0.7)  # bins 可以按需要调整
    # plt.tight_layout()
    # plt.show()

    def Msef_sensor(self,train_data_pde_use):
        xi = torch.tensor(2.5e-3); M0 = torch.tensor(1e-4)
        self.visc_data_phi = torch.pow(2 * self.model.visc, 2)
        u=self.u_fun(train_data_pde_use[:, 0:1], train_data_pde_use[:, 1:2], train_data_pde_use[:, 2:3]) #u_fun(x,y,t)
        v=self.v_fun(train_data_pde_use[:, 0:1], train_data_pde_use[:, 1:2], train_data_pde_use[:, 2:3]) #v_fun(x,y,t)
        train_data_pde_use = train_data_pde_use.requires_grad_(True)

        phi = self.model(train_data_pde_use)
        with torch.no_grad():
            sensor_phi = (phi >= -1.0 + 0.25) & (phi <= 1.0 - 0.25)

        dphi_dtxy = fwd_gradients(phi, train_data_pde_use)
        dphi_dt, dphi_dx, dphi_dy = dphi_dtxy[:,0:1], dphi_dtxy[:,1:2], dphi_dtxy[:,2:3]
        dphi_dx_dx = fwd_gradients(dphi_dx, train_data_pde_use)[:,1:2]
        dphi_dy_dy = fwd_gradients(dphi_dy, train_data_pde_use)[:,2:3]

        dphi_u = phi*(phi**2-1)-xi**2*(dphi_dx_dx+dphi_dy_dy)

        dphi_u_dtxy = fwd_gradients(dphi_u, train_data_pde_use)
        dphi_u_dt, dphi_u_dx, dphi_u_dy = dphi_u_dtxy[:,0:1], dphi_u_dtxy[:,1:2], dphi_u_dtxy[:,2:3]
        dphi_u_dxdx = fwd_gradients(dphi_u_dx, train_data_pde_use)[:,1:2]
        dphi_u_dydy = fwd_gradients(dphi_u_dy, train_data_pde_use)[:,2:3]

        pde_CH = dphi_dt+u*dphi_dx+v*dphi_dy-M0*(dphi_u_dxdx+dphi_u_dydy)-self.visc_data_phi*(dphi_dx_dx+dphi_dy_dy)*sensor_phi
        loss_pde_ch = pde_CH.pow(2).mean()
        return loss_pde_ch

    def Mseb(self,train_bound,train_data_ic,phi_exact):
        phi_bc = self.model(train_bound)
        loss_bc = (phi_bc-1).pow(2).mean()
        phi_ic = self.model(train_data_ic)
        loss_ic = (phi_ic - phi_exact).pow(2).mean()
        return loss_bc, loss_ic

    def train(self, epoch):
        x = np.linspace(0.0, 1.0, 201)
        y = np.linspace(0.0, 1.0, 201)
        t = np.linspace(0.0, 1.0, 101)
        train_pde=32000
        train_init=3200
        train_bc=2000

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epoch, pct_start=0.1, div_factor=100.0, final_div_factor=100)
        self.model.train()
        for i in range(epoch):
            idx_point_t = np.random.randint(0, t.shape[0], train_pde)
            idx_point_x = np.random.randint(0, x.shape[0], train_pde)
            idx_point_y = np.random.randint(0, y.shape[0], train_pde)

            t_eqn = np.expand_dims(t[idx_point_t], axis=1);
            x_eqn = np.expand_dims(x[idx_point_x], axis=1);
            y_eqn = np.expand_dims(y[idx_point_y], axis=1);

            t_sample = torch.tensor(t_eqn, requires_grad=True).float().to(device)
            x_sample = torch.tensor(x_eqn, requires_grad=True).float().to(device)
            y_sample = torch.tensor(y_eqn, requires_grad=True).float().to(device)

            batch_bound = np.random.choice(self.train_data_bound.shape[0], train_bc, replace=False)
            batch_ind_init = np.random.choice(self.train_data_ic.shape[0], train_init, replace=False)

            optimizer.zero_grad(set_to_none=True)
            loss_pde_ch = self.Msef_sensor(torch.cat([t_sample, x_sample, y_sample], dim=1))
            loss_bc, loss_ic = self.Mseb(self.train_data_bound[batch_bound,:],
                                         self.train_data_ic[batch_ind_init,:],
                                         self.train_data_ic_phi_exact[batch_ind_init,:])
            loss = loss_pde_ch + loss_bc + loss_ic
            loss = torch.log(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i%100==0:
                print(f'{i}|{epoch} loss={torch.exp(loss).item():.4e} CH={loss_pde_ch.item():.4e} BC={loss_bc.item():.4E} IC={loss_ic.item():.4e} sensor={self.visc_data_phi.item():.4e}')
                self.log['loss'].append(torch.exp(loss).item())
                self.log['loss_ac_ch'].append(loss_pde_ch.item())
                self.log['loss_bc'].append(loss_bc.item())
                self.log['loss_ic'].append(loss_ic.item())
                self.log['AV'].append(self.visc_data_phi.item())

            if i%2000==0:
                with torch.no_grad():
                    torch.save(self.model.state_dict(), f'DKAN_AV_25e_3/DKAN_AV_25e_3_{i}.pth')
                    phi = self.model(self.val_data_pde).cpu().detach().reshape(251, 251, 3)
                    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4))  # 2行3列，最后一个留空
                    axes = axes.flatten()
                    for k in range(3):
                        axes[k].contourf(self.x_grid[:, :, 0], self.y_grid[:, :, 0], phi[:, :, k],levels=np.linspace(-1, 1, 21), vmin=-1, vmax=1)
                        axes[k].set_title(f'phi_{k}')
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(f'DKAN_AV_25e_3/DKAN_AV_25e_3_{i}.png')
                    plt.close()
        torch.save(self.model.state_dict(), "DKAN_AV_25e_3/DKAN_AV_25e_3_end.pth")
        scipy.io.savemat(f'DKAN_AV_25e_3/DKAN_AV_25e_3_history.mat', {'loss': self.log['loss'],'loss_ac_ch': self.log['loss_ac_ch'],'loss_bc': self.log['loss_bc'],'loss_ic': self.log['loss_ic'],'AV': self.log['AV']})

if __name__ == "__main__":
    # 实例化PINN并开始训练
    torch.set_num_threads(1)
    pinn = PINN(device='cuda', seed=42)
    pinn.train(20001)
