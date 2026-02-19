import numpy as np
import os
import random
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
from util import fwd_gradients
import time
torch.cuda.empty_cache()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", device)

init_seed = 42
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
random.seed(init_seed)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class DNN(nn.Sequential):
    def __init__(self,dim_in,dim_out,dim_hidden,layers_hidden,T_sub,act='tanh',X_all=None):
        super(DNN,self).__init__()
        self.X_min = X_all.min(0, keepdim=True)[0]
        self.X_max = X_all.max(0, keepdim=True)[0]

        for i in range(1,layers_hidden):
            self.add_module('fc{}'.format(i),nn.Linear(dim_hidden, dim_hidden))
            if act == 'tanh':
                self.add_module('act{}'.format(i), nn.Tanh())
            elif act == 'relu':
                self.add_module('act{}'.format(i), nn.ReLU())
            elif act == 'swish':
                self.add_module('act{}'.format(i), Act_op())
            else:
                raise ValueError(f'unknown activation function: {act}')

        self.add_module('fc{}'.format(layers_hidden),nn.Linear(dim_hidden,dim_out))

        self.B = nn.Parameter(torch.randn([2, 64]))
        # self.visc_c = nn.Parameter(torch.tensor(0.1))  # 全局可学习粘性
        T_min_out = (273.15 + T_sub) / 273.15
        T_max_out = (273.15 + 25) / 273.15
        self.T_mid = 0.5 * (T_max_out + T_min_out)
        self.dT =0.5*(T_max_out - T_min_out)
        self.visc = nn.Parameter(torch.tensor(0.01))

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight,gain=1.)

    def forward(self,x):
        x = 2 * (x - self.X_min.to(x.device)) / (self.X_max.to(x.device) - self.X_min.to(x.device)) - 1
        xy = torch.matmul(x[:,1:3], self.B.to(x.device))
        x = torch.cat([x[:,0:1], torch.sin(xy), torch.cos(xy)], dim=-1)
        for name, module in self._modules.items():
            x = module(x)
        # u=0
        v = x[:, 0:1]
        p = x[:, 1:2]
        T = self.T_mid + self.dT*torch.tanh(x[:, 2:3])
        c = 0.5 * torch.tanh(x[:, 3:4]) - 0.5  # c∈(-1,0)
        phi = torch.tanh(x[:, 4:5])
        return torch.cat([v, p, T, c, phi], dim=1)

class PINN:
    def __init__(self, device='cuda', seed=42):
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.t_start_second = 8e-2
        self.t_end_second  = 12e-2
        self.log = {'loss': [], 'loss_ac_ch': [], 'loss_ns': [], 'loss_bc': [], 'loss_ic': []}

        self.L_ref   = torch.tensor(1e-3)
        self.p_ref   = torch.tensor(1e5)
        self.rho_ref = torch.tensor(1e3)
        self.u_ref   = torch.sqrt(self.p_ref/self.rho_ref)
        self.T_ref   = torch.tensor(273.15)
        self.t_ref   = self.L_ref / self.u_ref  # 也是一个 tensor
        self.mu_ref  = torch.tensor(1e-3)
        self.k_ref   = torch.tensor(0.5918)
        self.cp_ref  = torch.tensor(4200.0)

        self.T_sub = -25.0

        self.sample_data0()  # 先pde采样
        # self.model = DNN(3,5,129,5,X_all=torch.tensor([[0, self.x_start, self.y_start], [2e2, self.x_end, self.y_end]], device=device)).to(device)
        # self.model._initialize_weight()
        self.model = torch.load('DKAN_ICE898_T25_T4_8/droplet_40000.pt').to(device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        with torch.no_grad():
            train_data_ic_real_data_right = self.model(self.train_data_ic_right)
            train_data_ic_real_data_right[:, 3] = torch.where(train_data_ic_real_data_right[:, 3] <= -0.5, -1.0, 0.0)
            train_data_ic_real_data_right[:, 4] = torch.where(train_data_ic_real_data_right[:, 4] <= 0.0, -1.0, 1.0)
            train_data_ic_real_data_left = train_data_ic_real_data_right.clone();
            self.train_data_ic_real_data = torch.cat([train_data_ic_real_data_left, train_data_ic_real_data_right], dim=0)

            # mask = (self.model(self.train_data_ic)[:, 4] >= 0.99)
            mask = (self.model(self.train_data_ic)[:, 3] <=-0.99) # 冰下的不动
            P = self.train_data_ic[mask, 1:3].cpu().numpy()
            self.train_data_ic_circle = np.c_[np.ones(len(P)) * self.t_start.item(), P]
            t_circle = np.random.uniform(self.t_start, self.t_end, self.train_data_ic_circle.shape[0])
            self.train_data_random_circle = np.c_[t_circle, P]
            self.train_data_ic_circle = torch.tensor(self.train_data_ic_circle, dtype=torch.float32).to(device)
            self.train_data_random_circle = torch.tensor(self.train_data_random_circle, dtype=torch.float32).to(device)

        t_vals = np.linspace(self.t_start, self.t_end, 3)  # 101 points in the t direction
        x_vals = np.linspace(self.x_start, self.x_end, 201)  # 1001 points in the x direction
        y_vals = np.linspace(self.y_start, self.y_end, 201)  # 1001 points in the x direction
        self.y_grid, self.x_grid, self.t_grid = np.meshgrid(y_vals, x_vals, t_vals, indexing="ij") #y,x,t
        val_data_pde = np.vstack([self.t_grid.ravel(), self.x_grid.ravel(), self.y_grid.ravel()]).T
        self.val_data_pde = torch.tensor(val_data_pde, dtype=torch.float32).to(device)

    def sample_data0(self):  # 无量纲化时间空间
        self.t_start = self.t_start_second / self.t_ref
        self.t_end   = self.t_end_second / self.t_ref
        self.x_start = -0.004 / self.L_ref;
        self.x_end = 0.004 / self.L_ref
        self.y_start = -0.004 / self.L_ref;
        self.y_end = 0.004 / self.L_ref
        self.x0=0.0
        self.y0=(-0.004 + 0.00068) / self.L_ref
        self.r0=0.0016 / self.L_ref
        with torch.no_grad():
            t_vals = np.linspace(self.t_start, self.t_end, 201)  # 101 points in the t direction
            x_vals = np.linspace(self.x_start, self.x_end, 401)  # 1001 points in the x direction
            y_vals = np.linspace(self.y_start, self.y_end, 401)  # 1001 points in the x direction
            y_grid, x_grid, t_grid = np.meshgrid(y_vals, x_vals, t_vals, indexing='ij')

            train_data_ic_right = np.vstack([t_grid[:, 200:, 0].ravel(), x_grid[:, 200:, 0].ravel(), y_grid[:, 200:, 0].ravel()]).T
            self.train_data_ic_right = torch.tensor(train_data_ic_right, dtype=torch.float32).to(device)
            self.train_data_ic_left =  self.train_data_ic_right.clone(); self.train_data_ic_left[:,1]=-self.train_data_ic_left[:,1];
            self.train_data_ic = torch.cat([self.train_data_ic_left,self.train_data_ic_right],dim=0)

            # Boundary conditions
            self.train_data_bc_left = np.vstack([t_grid[:, 0, :].ravel(), x_grid[:, 0, :].ravel(), y_grid[:, 0, :].ravel()]).T  # t in [0, 1], x = 0
            self.train_data_bc_right = self.train_data_bc_left.copy(); self.train_data_bc_right[:, 1] = self.x_end
            self.train_data_bc_down = np.vstack([t_grid[0, :, :].ravel(), x_grid[0, :, :].ravel(), y_grid[0, :, :].ravel()]).T
            self.train_data_bc_up = np.vstack([t_grid[-1, :, :].ravel(), x_grid[-1, :, :].ravel(), y_grid[-1, :, :].ravel()]).T

            self.train_data_bc_down = np.vstack([t_grid[0, :, :].ravel(), x_grid[0, :, :].ravel(), y_grid[0, :, :].ravel()]).T
            self.train_data_bc_up = np.vstack([t_grid[-1, :, :].ravel(), x_grid[-1, :, :].ravel(), y_grid[-1, :, :].ravel()]).T

            self.train_data_bc_left = torch.tensor(self.train_data_bc_left, dtype=torch.float32).to(device)
            self.train_data_bc_right = torch.tensor(self.train_data_bc_right, dtype=torch.float32).to(device)
            self.train_data_bc_down = torch.tensor(self.train_data_bc_down, dtype=torch.float32).to(device)
            self.train_data_bc_up = torch.tensor(self.train_data_bc_up, dtype=torch.float32).to(device)

            train_up_phi = np.array([[self.t_start, self.t_end], [self.x_start, self.x_end], [0.0, self.y_end]])
            lhs_up_phi = LHS(xlimits=train_up_phi, random_state=self.seed)
            lhs_up_phi = lhs_up_phi(10000)
            self.lhs_up_phi = torch.tensor(lhs_up_phi, dtype=torch.float32).to(device)

    def Msef_sensor(self, t_sample, x_sample, y_sample):
        t_sample = t_sample.requires_grad_(True)
        x_sample = x_sample.requires_grad_(True)
        y_sample = y_sample.requires_grad_(True)
        
        M_phi = torch.tensor(2.5e-11)
        sigma_phi = torch.tensor(7.27e-2)
        xi_phi = torch.tensor(4.0e-5)

        M_c = torch.tensor(2.0e-3)
        sigma_c = torch.tensor(3.17e-2)
        xi_c = torch.tensor(4.0e-5)
        L = torch.tensor(3.34e8)
        Wa = 3*sigma_c/xi_c

        rho1 = torch.tensor(1.0);    mu1 = torch.tensor(1e-5);  k1 = torch.tensor(0.0209); cp1 = torch.tensor(1003.0)
        rho2 = torch.tensor(1e3);  mu2 = torch.tensor(1e-3);    k2 = torch.tensor(0.5918); cp2 = torch.tensor(4200.0)
        rho3 = torch.tensor(898.0);  mu3 = torch.tensor(100.0);   k3 = torch.tensor(2.25);   cp3 = torch.tensor(2018.0)

        pred = self.model(torch.cat([t_sample, x_sample, y_sample], dim=1))
        v = pred[:,0:1]; p = pred[:, 1:2]; T = pred[:, 2:3]; c = pred[:, 3:4]; phi = pred[:, 4:5];

        rho=(0.5*(phi+1)*((c+1)*rho2-c*rho3)-0.5*(phi-1)*rho1)/self.rho_ref;
        mu=(0.5*(phi+1)*((c+1)*mu2-c*mu3)-0.5*(phi-1)*mu1)/self.mu_ref;
        k=(0.5*(phi+1)*((c+1)*k2-c*k3)-0.5*(phi-1)*k1)/self.k_ref;
        cp=(0.5*(phi+1)*((c+1)*cp2-c*cp3)-0.5*(phi-1)*cp1)/self.cp_ref;

        dc_t = fwd_gradients(c, t_sample);
        dc_x = fwd_gradients(c, x_sample);
        dc_y = fwd_gradients(c, y_sample);
        dc_xx = fwd_gradients(dc_x, x_sample)
        dc_yy = fwd_gradients(dc_y, y_sample)

        dphi_t = fwd_gradients(phi, t_sample);
        dphi_x = fwd_gradients(phi, x_sample);
        dphi_y = fwd_gradients(phi, y_sample);
        dphi_xx = fwd_gradients(dphi_x, x_sample)
        dphi_yy = fwd_gradients(dphi_y, y_sample)

        phi_u = -3*sigma_phi/(2*(2**0.5)*xi_phi) * (torch.pow(xi_phi/self.L_ref,2)*(dphi_xx+dphi_yy)-torch.pow(phi,3)+phi)
        dphi_u_x = fwd_gradients(phi_u, x_sample)
        dphi_u_y = fwd_gradients(phi_u, y_sample)
        dphi_u_xx = fwd_gradients(dphi_u_x, x_sample)
        dphi_u_yy = fwd_gradients(dphi_u_y, y_sample)

        u, du_t, du_x, du_y = 0,0,0,0 # 设置流向为0
        du_xx = 0.0; du_yy = 0.0
        dv_t = fwd_gradients(v, t_sample); dv_x = fwd_gradients(v, x_sample);dv_y = fwd_gradients(v, y_sample);
        dp_x = fwd_gradients(p, x_sample); dp_y = fwd_gradients(p, y_sample);
        dT_t = fwd_gradients(T, t_sample); dT_x = fwd_gradients(T, x_sample);dT_y = fwd_gradients(T, y_sample);

        dv_xx = fwd_gradients(dv_x, x_sample); dv_yy = fwd_gradients(dv_y, y_sample)
        dT_xx = fwd_gradients(dT_x, x_sample); dT_yy = fwd_gradients(dT_y, y_sample)

        lambda_c = 3 / (2 * 2 ** 0.5) * sigma_c * xi_c
        lambda_phi = 3 / (2 * 2 ** 0.5) * sigma_phi * xi_phi
        # drho_x = (0.5 * (rho2 - rho1 + c*(rho2 - rho3)) * dphi_x + 0.5 * (phi + 1) * (rho2 - rho3) * dc_x)/self.rho_ref
        # drho_y = (0.5 * (rho2 - rho1 + c*(rho2 - rho3)) * dphi_y + 0.5 * (phi + 1) * (rho2 - rho3) * dc_y)/self.rho_ref
        # drho_xx = (0.5*(2.0*(rho2-rho3)*dc_x*dphi_x+(rho2-rho1+c*(rho2-rho3))*dphi_xx+(phi+1.0)*(rho2-rho3)*dc_xx))/self.rho_ref
        # drho_yy = (0.5*(2.0*(rho2-rho3)*dc_y*dphi_y+(rho2-rho1+c*(rho2-rho3))*dphi_yy+(phi+1.0)*(rho2-rho3)*dc_yy))/self.rho_ref

        dmu_x = (0.5 * (mu2 - mu1 + c*(mu2 - mu3)) * dphi_x + 0.5 * (phi + 1) * (mu2 - mu3) * dc_x)/self.mu_ref
        dmu_y = (0.5 * (mu2 - mu1 + c*(mu2 - mu3)) * dphi_y + 0.5 * (phi + 1) * (mu2 - mu3) * dc_y)/self.mu_ref
        dk_x = (0.5 * (k2 - k1 + c*(k2 - k3)) * dphi_x + 0.5 * (phi + 1) * (k2 - k3) * dc_x)/self.k_ref
        dk_y = (0.5 * (k2 - k1 + c*(k2 - k3)) * dphi_y + 0.5 * (phi + 1) * (k2 - k3) * dc_y)/self.k_ref

        visc_data = torch.pow(2 * self.model.module.visc, 2)  # 人工粘性
        with torch.no_grad():
            sensor_c = (c >= -1.0 + 0.4) & (c <= 0.0 - 0.4)
            sensor_phi = (phi >= -1.0 + 0.75) & (phi <= 1.0 - 0.75)
            sensor=(sensor_c+sensor_phi)>1e-4

        pde_AC = dc_t+u*dc_x+v*dc_y-M_c*self.t_ref*(torch.pow(xi_c/self.L_ref,2)*(dc_xx+dc_yy)-(c*(c+1)*(2*c+1)+L/Wa*(1-T)*15*torch.pow(c+1,2)*torch.pow(c,2)))-visc_data*(dc_xx + dc_yy)*sensor
        pde_CH = dphi_t+u*dphi_x+v*dphi_y-M_phi/(self.u_ref*self.L_ref)*(dphi_u_xx+dphi_u_yy)-visc_data*(dphi_xx + dphi_yy)*sensor

        pde_NS1 = (du_x + dv_y)
        pde_NS2 = (rho*(du_t+u*du_x+v*du_y)+dp_x-self.mu_ref/(self.rho_ref*self.u_ref*self.L_ref)*(mu*(du_xx+du_yy)+dmu_x*2*du_x+dmu_y*(du_y+dv_x))
                   +lambda_c/(self.rho_ref*torch.pow(self.u_ref,2)*torch.pow(self.L_ref,2))*(dc_xx+dc_yy)*dc_x
                   +lambda_phi/(self.rho_ref*torch.pow(self.u_ref,2)*torch.pow(self.L_ref,2))*(dphi_xx+dphi_yy)*dphi_x-rho*visc_data*(du_xx+du_yy)*sensor)
        pde_NS3 = (rho*(dv_t+u*dv_x+v*dv_y)+dp_y-self.mu_ref/(self.rho_ref*self.u_ref*self.L_ref)*(mu*(dv_xx+dv_yy)+dmu_y*2*dv_y+dmu_x*(dv_x+du_y))
                   +lambda_c/(self.rho_ref*torch.pow(self.u_ref,2)*torch.pow(self.L_ref,2))*(dc_xx+dc_yy)*dc_y
                   +lambda_phi/(self.rho_ref*torch.pow(self.u_ref,2)*torch.pow(self.L_ref,2))*(dphi_xx+dphi_yy)*dphi_y-rho*visc_data*(dv_xx+dv_yy)*sensor)
        pde_NS4 = (rho*cp*(dT_t+u*dT_x+v*dT_y)+0.5*L/(self.rho_ref*self.cp_ref*self.T_ref)*((1+phi)*dc_t+c*dphi_t)
                   -self.k_ref/(self.rho_ref*self.u_ref*self.cp_ref*self.L_ref)*(k*(dT_xx+dT_yy)+dk_x*dT_x+dk_y*dT_y)
                   -self.mu_ref*self.u_ref/(self.rho_ref*self.cp_ref*self.T_ref*self.L_ref)*mu*(2*(du_x**2+dv_y**2)+(du_y+dv_x)**2)-rho*cp*visc_data*(dT_xx+dT_yy)*sensor)
        loss_pde_ac_ch = 1000.0*(pde_AC.pow(2).mean() + pde_CH.pow(2).mean())
        loss_pde_ns = (pde_NS1.pow(2).mean()+pde_NS2.pow(2).mean()+pde_NS3.pow(2).mean()+pde_NS4.pow(2).mean())
        return loss_pde_ac_ch, loss_pde_ns, visc_data

    # plt.figure()
    # plt.scatter(train_data_pde[:,1:2].cpu().detach(),train_data_pde[:,0:1].cpu().detach(),sensor_c.cpu().detach().numpy().astype(float))
    # # plt.scatter(train_data_pde[:, 1:2].cpu().detach(), train_data_pde[:, 0:1].cpu().detach(),sensor_phi.cpu().detach().numpy().astype(float))
    # plt.show()

    def Mseb(self, train_up_phi, train_data_pde_use_left,train_data_pde_use_right,
             train_data_ic_circle, train_data_random_circle,
             train_data_bc_left, train_data_bc_right, train_data_ic, train_data_ic_real_data, train_data_bc_up, train_data_bc_down):

        pred_phi = self.model(train_up_phi)[:,4:5] # 0上方全是气体
        loss_up_phi=0.01*(pred_phi+1.0).pow(2).mean()

        pred_0 = self.model(train_data_ic_circle)[:,4:5]
        pred_1 = self.model(train_data_random_circle)[:,4:5]
        loss_circle=0.1*(pred_0-pred_1).pow(2).mean()

        pred_0 = self.model(train_data_pde_use_left)
        pred_1 = self.model(train_data_pde_use_right)
        loss_sym=(pred_0-pred_1).pow(2).mean()

        pred_left = self.model(train_data_bc_left)
        pred_right = self.model(train_data_bc_right)
        loss_left_right=(pred_left-pred_right).pow(2).mean()

        pred = self.model(train_data_bc_up)
        v = pred[:,0:1]; p = pred[:, 1:2]; T = pred[:, 2:3]; c = pred[:, 3:4]; phi = pred[:, 4:5];
        loss_up = (v.pow(2).mean()+(T-(273.15+25)/self.T_ref).pow(2).mean()+c.pow(2).mean()+(phi+1).pow(2).mean())/4.0

        pred = self.model(train_data_bc_down)
        v = pred[:,0:1]; p = pred[:, 1:2]; T = pred[:, 2:3]; c = pred[:, 3:4]; phi = pred[:, 4:5];
        phi_down = torch.where((train_data_bc_down[:,1:2]**2+(train_data_bc_down[:,2:3]-self.y0)**2)<=self.r0**2,1.0,-1)
        loss_down = (v.pow(2).mean()+(T-(273.15+self.T_sub)/self.T_ref).pow(2).mean()+(c+1).pow(2).mean()+(phi-phi_down).pow(2).mean())/4.0

        loss_bc = loss_left_right + loss_up + loss_down

        pred = self.model(train_data_ic)
        loss_ic = (pred-train_data_ic_real_data).pow(2).mean()
        return (loss_up_phi + loss_circle + loss_sym + loss_bc + loss_ic)/7.0

    # #
    # plt.figure(figsize=(6, 5))
    # sc = plt.scatter(train_data_bc_down[:,0].cpu(), train_data_bc_down[:,1].cpu(), c=phi_down.cpu(), cmap='viridis', s=20)  # c 指定颜色数据，cmap 指定色图
    # plt.colorbar(sc, label="phi value")  # 添加颜色条
    # # plt.ylim([-4, -3.99])
    # # plt.axis([-4.1, 4.1, -4.01, -3.99])
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()

    def compute_gradient_weight(self, loss_pde_ac_ch, loss_pde_ns, loss_bc_ic):
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_ac_ch = torch.autograd.grad(loss_pde_ac_ch, params, retain_graph=True, create_graph=False, allow_unused=True)
        grad_ns = torch.autograd.grad(loss_pde_ns, params, retain_graph=True, create_graph=False, allow_unused=True)
        grad_bcic = torch.autograd.grad(loss_bc_ic, params, retain_graph=True, create_graph=False, allow_unused=True)

        g_ac_ch = torch.linalg.norm(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p, g in zip(params, grad_ac_ch)],dim=0))
        g_ns = torch.linalg.norm(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p, g in zip(params, grad_ns)],dim=0))
        g_bcic = torch.linalg.norm(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p, g in zip(params, grad_bcic)],dim=0))
        
        weight_ac_ch = (g_ac_ch + g_ns + g_bcic) / (g_ac_ch + 1e-12)
        weight_ns = (g_ac_ch + g_ns + g_bcic) / (g_ns + 1e-12)
        weight_bcic = (g_ac_ch + g_ns + g_bcic) / (g_bcic + 1e-12)
        
        weight_ac_ch = torch.clamp(weight_ac_ch, min=1e-4, max=1e4)
        weight_ns = torch.clamp(weight_ns, min=1e-4, max=1e4)
        weight_bcic = torch.clamp(weight_bcic, min=1e-4, max=1e4)
        return weight_ac_ch.detach(), weight_ns.detach(), weight_bcic.detach()

    def train(self, epoch):
        self.model.train()
        # self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1, max_iter=100, history_size=100, tolerance_grad=1e-8, tolerance_change=1e-10)
        # optimizer = SOAP(self.model.parameters(),lr=1e-4,weight_decay=1e-8)
        # optimizer = SOAP(self.model.parameters(),lr=1e-5,weight_decay=1e-8)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.000025, weight_decay=1e-6)
        # optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.1, max_iter=100, history_size=100,tolerance_grad=1e-8, tolerance_change=1e-10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-7)
        train_num=20000
        train_init=2000
        train_bc=2000

        t = np.linspace(self.t_start, self.t_end, 201)
        x = np.linspace(self.x_start, self.x_end, 401)
        y = np.linspace(self.y_start, self.y_end, 401)

        for i in range(epoch):
            with torch.no_grad():
                idx_point_t = np.random.randint(0, t.shape[0], train_num)
                idx_point_x = np.random.randint(0, x.shape[0], train_num)
                idx_point_y = np.random.randint(0, y.shape[0], train_num)

                t_eqn = np.expand_dims(t[idx_point_t], axis=1);
                x_eqn = np.expand_dims(x[idx_point_x], axis=1);
                y_eqn = np.expand_dims(y[idx_point_y], axis=1);

                t_sample = torch.tensor(t_eqn).float().to(device)
                x_sample = torch.tensor(x_eqn).float().to(device)
                y_sample = torch.tensor(y_eqn).float().to(device)

                train_data_pde_use_left = torch.cat([t_sample, x_sample, y_sample],dim=1)
                train_data_pde_use_right = torch.cat([t_sample, -x_sample, y_sample], dim=1)
                train_data_pde_use = torch.cat([train_data_pde_use_left, train_data_pde_use_right], dim=0)
                train_data_pde_use = train_data_pde_use[torch.randperm(train_data_pde_use.size(0))]

                batch_ind_left_right = np.random.choice(self.train_data_bc_left.shape[0], train_bc, replace=False)
                batch_ind_up_down = np.random.choice(self.train_data_bc_up.shape[0], train_bc, replace=False)
                batch_ind_init = np.random.choice(self.train_data_ic.shape[0], train_init, replace=False)
                batch_ind_circle = np.random.choice(self.train_data_ic_circle.shape[0], train_init, replace=False)
                batch_ind_up_phi = np.random.choice(self.lhs_up_phi.shape[0], train_init, replace=False)

            optimizer.zero_grad(set_to_none=True)
            loss_pde_ac_ch, loss_pde_ns, visc_data = self.Msef_sensor(train_data_pde_use[:,0:1],train_data_pde_use[:,1:2],train_data_pde_use[:,2:3])
            loss_bc_ic = self.Mseb(self.lhs_up_phi[batch_ind_up_phi,:], train_data_pde_use_left,train_data_pde_use_right,
                                   self.train_data_ic_circle[batch_ind_circle, :],self.train_data_random_circle[batch_ind_circle, :],
                                   self.train_data_bc_left[batch_ind_left_right,:],
                                   self.train_data_bc_right[batch_ind_left_right,:],
                                   self.train_data_ic[batch_ind_init,:],
                                   self.train_data_ic_real_data[batch_ind_init, :],
                                   self.train_data_bc_up[batch_ind_up_down,:],
                                   self.train_data_bc_down[batch_ind_up_down,:])
            if i%100 == 0:
                weight_ac_ch, weight_ns, weight_bcic = self.compute_gradient_weight(loss_pde_ac_ch, loss_pde_ns,loss_bc_ic)
            loss = weight_ac_ch * loss_pde_ac_ch + weight_ns * loss_pde_ns + weight_bcic * loss_bc_ic
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                print(f'{i}|{epoch}-- Balance  -- loss={loss.item():.4e} CH={weight_ac_ch * loss_pde_ac_ch.item():.4e} NS={weight_ns * loss_pde_ns.item():.4e} Bound_Initial={weight_bcic * loss_bc_ic.item():.4e}')
                print(f'{i}|{epoch}-- Real     -- loss={loss.item():.4e} CH={loss_pde_ac_ch.item():.4e} NS={loss_pde_ns.item():.4e} Bound_Initial={loss_bc_ic.item():.4e} mu={visc_data.item():.4e}')
                # print(f'{i}|{epoch}--Balance-- loss={loss.item():.4e} CH={(600.0*loss_pde_ac_ch).item():.4e} NS={(1.0*loss_pde_ns).item():.4e} Bound_Initial={(10.0*loss_bc_ic).item():.4e}')
            if i % 1000 == 0:
                torch.save(self.model.module, f'DKAN_ICE898_T25_T8_12/droplet_{i}.pt')
                with torch.no_grad():
                    pred = self.model(self.val_data_pde).cpu().detach().reshape(201, 201, 3, 5)
                    c = pred[:, :,:, 3];phi = pred[:, :, :, 4];
                    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
                    axs = axes.ravel()
                    cs_c = [axs[0].contourf(self.x_grid[:, :, 0], self.y_grid[:, :, 0], c[:, :,  0],levels=np.linspace(-1, 0, 11), vmin=-1, vmax=0),
                            axs[2].contourf(self.x_grid[:, :, 0], self.y_grid[:, :, 0], c[:, :, 1],levels=np.linspace(-1, 0, 11), vmin=-1, vmax=0),
                            axs[4].contourf(self.x_grid[:, :, 0], self.y_grid[:, :, 0], c[:, :, 2],levels=np.linspace(-1, 0, 11), vmin=-1, vmax=0),
                            ]
                    titles_c = [f'c_{i}_t=0', f'c_{i}_t=10', f'c_{i}_t=end']
                    for label_i, ax in enumerate([axs[0], axs[2], axs[4]]): ax.set_title(titles_c[label_i])
                    cs_phi = [axs[1].contourf(self.x_grid[:, :, 0], self.y_grid[:, :, 0], phi[:, :,  0],levels=np.linspace(-1, 1, 21), vmin=-1, vmax=1),
                              axs[3].contourf(self.x_grid[:, :, 0], self.y_grid[:, :, 0], phi[:, :, 1],levels=np.linspace(-1, 1, 21), vmin=-1, vmax=1),
                              axs[5].contourf(self.x_grid[:, :, 0], self.y_grid[:, :, 0], phi[:, :, 2],levels=np.linspace(-1, 1, 21), vmin=-1, vmax=1, ),
                              ]
                    titles_phi = [f'φ_{i}_t=0', f'φ_{i}_t=10', f'φ_{i}_t=end']
                    for label_i, ax in enumerate([axs[1], axs[3], axs[5]]): ax.set_title(titles_phi[label_i])

                    fig.colorbar(cs_c[0], ax=[axs[0], axs[2], axs[4]], shrink=0.8)
                    fig.colorbar(cs_phi[0], ax=[axs[1], axs[3], axs[5]], shrink=0.8)
                    # fig.tight_layout()
                    # plt.show()
                    plt.savefig(f'DKAN_ICE898_T25_T8_12/droplet_{i}.png', dpi=300)
                    plt.close()

if __name__ == "__main__":
    # 实例化PINN并开始训练
    t1 = time.time()
    print(f"device: {device}, GPU_num: {torch.cuda.device_count()}")
    # torch.set_num_threads(1)
    pinn = PINN(device='cuda', seed=42)
    # criteria = torch.nn.MSELoss()
    #pinn.model.load_state_dict(torch.load('mlp_LBFGS_sensor_temp.pth'))
    pinn.train(30001)
    t2 = time.time()
    print('wall time:', t2-t1)