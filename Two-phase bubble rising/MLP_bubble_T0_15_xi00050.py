import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
#import torch.optim as optims
from torch.optim import lr_scheduler
import scipy.io
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
import time
from smt.sampling_methods import LHS
begin=time.time()

np.random.seed(1234)
print('import end')
#一、定义网络，定义神经网络的结构

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
    def __init__(self,dim_in,dim_out,dim_hidden,layers_hidden,act='tanh'):
        super(DNN,self).__init__()

        self.add_module('fc0',nn.Linear(dim_in, dim_hidden))
        self.add_module('act0',Act_op())

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

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight,gain=1.)

    def forward(self,x):
        for name, module in self._modules.items():
            x = module(x)
        phi = torch.tanh(x[:, 0:1])
        u = x[:, 1:2]
        v = x[:, 2:3]
        p = x[:, 3:4] # p+
        return torch.cat([phi, u, v, p], dim=1)

model=DNN(3,4,100,10,act='swish').to(device)
# model=DNN(3,4,100,8,act='swish',X_all=torch.tensor([[0, 0, 0], [1.0, 2.0, 1.0]], device=device)).to(device)
model._initialize_weight()
#model=torch.load('bubble_wall_zeros_init_9_11_final.pt',map_location='cuda')
model = nn.DataParallel(model)

#二、定义计算区域，该计算域定义采用的是规则区域采样方式。根据之前的计算经验，随机采样
#次数趋于无穷时，最终结果与真实值应保持一致。故生成的数据应定义为一个时空区域块。可设
#置为[-1,1]^2*[0,1]
#物理量参数
U0=1.0
eta=0.005
r=0.25
Cx=0.5
Cy=0.5
T=2.0#总时间长度
M0=1.0e-4
Lrho=1000
Grho=1
Lmiu=10
Gmiu=0.1
sigma=1.96
gravity=-0.98

#权重大小，取推荐值100
w_eqn=1
w_init=1

#各个方向上网格点长度，直接使用pytorch格式的数据
num_x=201
num_y=401
num_t=151
L=1.0

N_eqn=32000
N_init=3200
N_cyc=3200


x_vals = np.linspace(0, 1.0, 201)  # 1001 points in the x direction
y_vals = np.linspace(0, 2.0, 201)  # 1001 points in the x direction
t_vals = np.linspace(0.0, 1.5, 4)  # 101 points in the t direction
my_y_grid, my_x_grid, my_t_grid = np.meshgrid(y_vals, x_vals, t_vals, indexing="ij")  # y,x,t
val_data_pde = np.vstack([my_x_grid.ravel(), my_y_grid.ravel(), my_t_grid.ravel()]).T
my_val_data_pde = torch.tensor(val_data_pde, dtype=torch.float32).to(device)

x=np.linspace(0.0,1.0,num_x)
y=np.linspace(0.0,2.0,num_y)
t=np.linspace(0.0,1.5,num_t)

#初始化相分数场，采用文章中的形式对相分数场进行初始化
#将x与y网格化，随后采用公式对C_init进行点对点的计算。
C_init=np.zeros((num_x,num_y))
[x_init,y_init]=np.meshgrid(x,y)
C_init=-np.tanh((r-np.sqrt((x_init-Cx)**2+(y_init-Cy)**2))/np.sqrt(2)/eta)
u_init=np.zeros((num_x,num_y))
v_init=np.zeros((num_x,num_y))

#测试：序参量分布情况
# fig,ax=plt.subplots()
# H = ax.pcolormesh(x_init,y_init, C_init, shading='gouraud', cmap = 'jet')#, vmin=-0.1, vmax=0.1)
# fig.colorbar(H,ax=ax)
# fig.savefig('C_init.png')

#转换为torch格式，训练时使用
x_init=torch.tensor(x_init).float().to(device)
y_init=torch.tensor(y_init).float().to(device)
C_init=torch.tensor(C_init).float().to(device)
u_init=torch.tensor(u_init).float().to(device)
v_init=torch.tensor(v_init).float().to(device)
x_init=torch.reshape(x_init,[num_x*num_y,1])
y_init=torch.reshape(y_init,[num_x*num_y,1])
C_init=torch.reshape(C_init,[num_x*num_y,1])
u_init=torch.reshape(u_init,[num_x*num_y,1])
v_init=torch.reshape(v_init,[num_x*num_y,1])

#计算时的数据采用临时抽样的结果，转换后升维得到希望计算的结果。
#此处先将网络参数导入
optimizer =  torch.optim.AdamW(model.parameters(),lr=5e-3,weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40001, eta_min=1e-5)

for epoch in range(40001):
    idx_point_x=np.random.randint(0,x.shape[0],N_eqn)
    idx_point_y=np.random.randint(0,y.shape[0],N_eqn)
    idx_point_t=np.random.randint(0,t.shape[0],N_eqn)
    
    x_eqn=np.expand_dims(x[idx_point_x], axis=1);
    y_eqn=np.expand_dims(y[idx_point_y], axis=1);
    t_eqn=np.expand_dims(t[idx_point_t], axis=1);
    
    x_sample = torch.tensor(x_eqn, requires_grad=True).float().to(device)
    y_sample = torch.tensor(y_eqn, requires_grad=True).float().to(device)
    t_sample = torch.tensor(t_eqn, requires_grad=True).float().to(device)

    optimizer.zero_grad(set_to_none=True)
    Output_uvp_eqns=model(torch.cat([x_sample, y_sample, t_sample], dim=1))
    C_pred=Output_uvp_eqns[:,0:1]
    u_pred=Output_uvp_eqns[:,1:2]
    v_pred=Output_uvp_eqns[:,2:3]
    p_pred=Output_uvp_eqns[:,3:4]

    #代入初始条件    
    idx_init = np.random.choice(x_init.shape[0], N_init,replace=False)
    x0_init=x_init[idx_init];y0_init=y_init[idx_init];
    C0_init=C_init[idx_init];u0_init=u_init[idx_init];v0_init=v_init[idx_init];
    t0_init=torch.zeros([N_init,1]).float().to(device)
    Output_uvp_init=model(torch.cat([x0_init, y0_init, t0_init], dim=1))
    C_data_init=Output_uvp_init[:,0:1]
    u_data_init=Output_uvp_init[:,1:2]
    v_data_init=Output_uvp_init[:,2:3]
    
    #代入边界条件，边界上所有的点均为C=1的点
    #上下边界
    #x_cyc_up=torch.linspace(0,np.max(x0_star), N_cyc, requires_grad=True).float().to(device).unsqueeze(1);
    x_cyc_up=np.max(x)*torch.rand([N_cyc,1], requires_grad=True).float().to(device)
    y_cyc_up=torch.ones_like(x_cyc_up, requires_grad=True)*np.max(y)
    x_cyc_down=x_cyc_up
    y_cyc_down=torch.zeros_like(x_cyc_down, requires_grad=True)
    t_cyc=torch.rand_like(x_cyc_up, requires_grad=True).float().to(device)*np.max(t);
    #左右循环边界
    x_cyc_left=torch.zeros([N_cyc,1], requires_grad=True).float().to(device);
    y_cyc_left=torch.rand_like(x_cyc_left, requires_grad=True)*np.max(y)
    x_cyc_right=np.max(x)*torch.ones([N_cyc,1], requires_grad=True).float().to(device);
    y_cyc_right=y_cyc_left;

    #边界条件计算
    Output_cyc_up=model(torch.cat([x_cyc_up, y_cyc_up, t_cyc], dim=1))
    Output_cyc_down=model(torch.cat([x_cyc_down, y_cyc_down, t_cyc], dim=1))
    Output_cyc_left=model(torch.cat([x_cyc_left, y_cyc_left, t_cyc], dim=1))
    Output_cyc_right=model(torch.cat([x_cyc_right, y_cyc_right, t_cyc], dim=1))
    C_cyc_up=Output_cyc_up[:,0:1]
    u_cyc_up=Output_cyc_up[:,1:2]
    v_cyc_up=Output_cyc_up[:,2:3]
    
    C_cyc_down=Output_cyc_down[:,0:1]
    u_cyc_down=Output_cyc_down[:,1:2]
    v_cyc_down=Output_cyc_down[:,2:3]
    
    C_cyc_left=Output_cyc_left[:,0:1]
    u_cyc_left=Output_cyc_left[:,1:2]
    v_cyc_left=Output_cyc_left[:,2:3]
    
    C_cyc_right=Output_cyc_right[:,0:1]
    u_cyc_right=Output_cyc_right[:,1:2]
    v_cyc_right=Output_cyc_right[:,2:3]

    v_x_left = torch.autograd.grad(v_cyc_left, x_cyc_left, grad_outputs=torch.ones_like(v_cyc_left),retain_graph=True,create_graph=True)[0]
    v_x_right = torch.autograd.grad(v_cyc_right, x_cyc_right, grad_outputs=torch.ones_like(v_cyc_right),retain_graph=True,create_graph=True)[0]

    #自动微分与中间量计算
    #C的场
    C_t = torch.autograd.grad(
        C_pred, t_sample, 
        grad_outputs=torch.ones_like(C_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    C_x = torch.autograd.grad(
        C_pred, x_sample, 
        grad_outputs=torch.ones_like(C_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    C_y = torch.autograd.grad(
        C_pred, y_sample, 
        grad_outputs=torch.ones_like(C_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    C_xx = torch.autograd.grad(
        C_x, x_sample, 
        grad_outputs=torch.ones_like(C_x),
        retain_graph=True,
        create_graph=True
    )[0]
    C_yy = torch.autograd.grad(
        C_y, y_sample, 
        grad_outputs=torch.ones_like(C_y),
        retain_graph=True,
        create_graph=True
    )[0]
    #u的场
    u_t = torch.autograd.grad(
        u_pred, t_sample, 
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u_pred, x_sample, 
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    u_y = torch.autograd.grad(
        u_pred, y_sample, 
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x_sample, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y_sample, 
        grad_outputs=torch.ones_like(u_y),
        retain_graph=True,
        create_graph=True
    )[0]
    #v的场
    v_t = torch.autograd.grad(
        v_pred, t_sample, 
        grad_outputs=torch.ones_like(v_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    v_x = torch.autograd.grad(
        v_pred, x_sample, 
        grad_outputs=torch.ones_like(v_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    v_y = torch.autograd.grad(
        v_pred, y_sample, 
        grad_outputs=torch.ones_like(v_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    v_xx = torch.autograd.grad(
        v_x, x_sample, 
        grad_outputs=torch.ones_like(v_x),
        retain_graph=True,
        create_graph=True
    )[0]
    v_yy = torch.autograd.grad(
        v_y, y_sample, 
        grad_outputs=torch.ones_like(v_y),
        retain_graph=True,
        create_graph=True
    )[0]
    #p的场
    p_x = torch.autograd.grad(
        p_pred, x_sample, 
        grad_outputs=torch.ones_like(p_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    p_y = torch.autograd.grad(
        p_pred, y_sample, 
        grad_outputs=torch.ones_like(p_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    
    #定义中间变量：phi
    phi=C_pred*(C_pred**2-1)-eta**2*(C_xx+C_yy)
    
    phi_x = torch.autograd.grad(
        phi, x_sample, 
        grad_outputs=torch.ones_like(phi),
        retain_graph=True,
        create_graph=True
    )[0]
    phi_y = torch.autograd.grad(
        phi, y_sample, 
        grad_outputs=torch.ones_like(phi),
        retain_graph=True,
        create_graph=True
    )[0]
    phi_xx = torch.autograd.grad(
        phi_x, x_sample, 
        grad_outputs=torch.ones_like(phi_x),
        retain_graph=True,
        create_graph=True
    )[0]
    phi_yy = torch.autograd.grad(
        phi_y, y_sample, #已修改
        grad_outputs=torch.ones_like(phi_y),
        retain_graph=True,
        create_graph=True
    )[0]
    
    #方程计算时的中间变量准备
    #混合物密度与粘度
    #注意：这两项绝对不能越界，可使用一个函数约束住这两项
    Crho=(1.0+C_pred)/2.0*Lrho+(1.0-C_pred)/2.0*Grho
    Cmiu=(1.0+C_pred)/2.0*Lmiu+(1.0-C_pred)/2.0*Gmiu

    #表面张力
    fsigx=3/4*np.sqrt(2)*sigma/eta*phi*C_x
    fsigy=3/4*np.sqrt(2)*sigma/eta*phi*C_y

    m_loss=u_x+v_y
    #2.相场方程
    C_loss=C_t+u_pred*C_x+v_pred*C_y-M0*(phi_xx+phi_yy)
    #3.动量方程-x方向
    u_loss=(Crho*(u_t+u_pred*u_x+v_pred*u_y)+p_x-0.5*(Lmiu-Gmiu)*C_x*2*u_x-0.5*(Lmiu-Gmiu)*C_y*(u_y+v_x)- \
        Cmiu*(u_xx+u_yy)-fsigx)/Lrho
    #4.动量方程-y方向
    v_loss=(Crho*(v_t+u_pred*v_x+v_pred*v_y)+p_y-0.5*(Lmiu-Gmiu)*C_x*(v_x+u_y)-0.5*(Lmiu-Gmiu)*C_y*2*v_y- \
        Cmiu*(v_xx+v_yy)-fsigy-Crho*gravity)/Lrho

    #方程
    loss_eqns=torch.mean(C_loss**2)+torch.mean(u_loss**2)+torch.mean(v_loss**2)+torch.mean(m_loss**2)

    loss_bd=torch.mean((C_cyc_up-1)**2+(C_cyc_down-1)**2+(C_cyc_left-1)**2+(C_cyc_right-1)**2)+ \
        torch.mean(u_cyc_up**2+v_cyc_up**2+u_cyc_down**2+v_cyc_down**2)+ \
        torch.mean(u_cyc_left**2+u_cyc_right**2)+ torch.mean(v_x_left**2+v_x_right**2)
    loss_init=torch.mean((C0_init-C_data_init)**2)+torch.mean((u0_init-u_data_init)**2)+torch.mean((v0_init-v_data_init)**2)
    loss_total=loss_eqns+loss_bd+loss_init
    loss_total.backward()#retain_graph=True
    optimizer.step()
    scheduler.step()
    end = time.time()

    if np.mod(epoch,100)==0:
        print(f"epoch={epoch}, lr={optimizer.param_groups[0]['lr']:.4e}, loss={loss_total.item():.4e}, PDE_loss={loss_eqns.item():.4e}, BC_IC_loss={(loss_bd+loss_init).item():.4e}" )
    if np.mod(epoch,1000)==0:
        with torch.no_grad():
            pred = model(my_val_data_pde).cpu().detach().reshape(201, 201, 4, 4)
            phi = pred[:, :, :, 0];
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            axs = axes.ravel()
            cs_phi = [
                axs[0].contourf(my_x_grid[:, :, 0], my_y_grid[:, :, 0], phi[:, :, 1], levels=np.linspace(-1, 1, 21),vmin=-1, vmax=1),
                axs[1].contourf(my_x_grid[:, :, 0], my_y_grid[:, :, 0], phi[:, :, 2], levels=np.linspace(-1, 1, 21),vmin=-1, vmax=1),
                axs[2].contourf(my_x_grid[:, :, 0], my_y_grid[:, :, 0], phi[:, :, 3], levels=np.linspace(-1, 1, 21),vmin=-1, vmax=1),
            ]
            titles_phi = [r'$\phi,\ t=0.5$', r'$\phi,\ t=1.0$', r'$\phi,\ t=1.5$']
            for ax, tt in zip(axs, titles_phi):
                ax.set_title(tt, fontsize=12)
                ax.set_aspect("equal")
            plt.axis([0, 1, 0, 2])
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'MLP_bubble_T0_15_xi0005/bubble_{epoch}.png', dpi=300)
            plt.close()
            torch.save(model.module, f'MLP_bubble_T0_15_xi0005/bubble_t0_15_{epoch}.pt')

    
    