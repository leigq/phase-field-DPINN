# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch

def fwd_gradients(Y, x):
    dummy = torch.ones_like(Y)
    G = torch.autograd.grad(Y, x, dummy, create_graph=True, retain_graph=True)[0]
    return G

def coeff(nodes, order=1):
    '''
    This function is used to calculate the differential coefficient of given nodes
    '''
    m = len(nodes)
    device=nodes.device

    factor, pownodes = torch.ones(m, device=device), torch.ones(m, m, device=device)
    b = torch.zeros(m, 1, device=device)
    b[order] = 1

    for i in range(1, m):
        factor[i] = i * factor[i-1]
    for i in range(1,m):
        pownodes[:,i] = nodes * pownodes[:,i-1]
        
    A = pownodes / factor
    A = A.T
    x = torch.linalg.solve(A, b)
    return x

def boundary_2nd(u, nodes, du=0, dim=0, loc=['left', 'right']):
    if dim == 1: 
        u = u.T
    
    if 'left' in loc:
        ind_l = nodes[nodes>=0]
        coeff_l = coeff(ind_l)
        u[0:1] = - (u[ind_l[1:]]*coeff_l[ind_l[1:]]).sum(0, keepdim=True) / coeff_l[0]
    if 'right' in loc:
        ind_r = nodes[nodes<=0]
        coeff_r = coeff(ind_r)
        u[-1:] = - (u[ind_r[:-1]-1] * coeff_r[ind_r[:-1]-1]).sum(0, keepdim=True) / coeff_r[-1]
    
    if dim == 1: 
        u = u.T
    return u

def jacobian_trans(x, y, nodes, boundary=(0,0)):
    Xxi = diff2d(x, nodes, dim=0, boundary=boundary[0])
    Xeta = diff2d(x, nodes, dim=1, boundary=boundary[1])
    Yxi = diff2d(y, nodes, dim=0, boundary=boundary[0])
    Yeta = diff2d(y, nodes, dim=1, boundary=boundary[1])
    
    jac = torch.stack([Xxi, Yxi, Xeta, Yeta], dim=2).view(x.shape[0],x.shape[1],2,2)
    jac_inv = torch.inverse(jac)
    return jac, jac_inv

def diff2d_jac_inv(jac_inv, u, nodes, boundary=(0,0)):
    u_xi = diff2d(u, nodes, dim=0, boundary=boundary[0])
    u_eta = diff2d(u, nodes, dim=1, boundary=boundary[1])
    u_xieta = torch.stack([u_xi, u_eta], dim=2).view(u.shape[0],u.shape[1],2,1)
    u_xy = (jac_inv@u_xieta).squeeze()
    return u_xy

def diff2d_xy(x, y, u, nodes, boundary=(0,0)):
    jac, jac_inv = jacobian_trans(x, y, nodes, boundary=boundary)
    u_xy = diff2d_jac_inv(jac_inv, u, boundary=boundary)
    return u_xy

def diff2d(u, nodes, dim=0, order=1, boundary=0):
    # boundary=0: The boundaries use the same number of nodes
    # boundary=1: The boundaries use smaller number of nodes
    # boundary=2: Periodic boundary
    if dim == 1:
        u = u.T
    (M, N) = u.shape
    coeff_u = coeff(nodes, order)
    m, n = nodes[0], nodes[-1]
    du = torch.zeros_like(u)
    
    if boundary == 0 or boundary == 1:
        for i in range(len(nodes)):
            du[-m:M-n] = du[-m:M-n] + coeff_u[i] * u[-m+nodes[i]:M-n+nodes[i]]
            
        for i in range(len(nodes)):
            if nodes[i] == 0: 
                continue
            elif nodes[i] < 0:
                ind = nodes[i]-m
                if boundary == 0: nodes_bound = nodes - nodes[i]
                if boundary == 1: nodes_bound = nodes[nodes >= -ind]
                coeff_bound = coeff(nodes_bound, order)
                nodes_ind = ind + nodes_bound
                du[ind] = du[ind] + (coeff_bound * u[nodes_ind]).sum(0)
            else:
                ind = nodes[i]-n + M-1
                if boundary == 0: nodes_bound = nodes - nodes[i]
                if boundary == 1: nodes_bound = nodes[nodes <= M-1-ind]
                coeff_bound = coeff(nodes_bound, order)
                nodes_ind = ind + nodes_bound
                du[ind] = du[ind] + (coeff_bound * u[nodes_ind]).sum(0)

    elif boundary == 2:
        u = torch.cat([u[M+m:], u, u[:n]]); (M, N) = u.shape
        du = torch.zeros_like(u)
        for i in range(len(nodes)):
            du[-m:M-n] = du[-m:M-n] + coeff_u[i] * u[-m+nodes[i]:M-n+nodes[i]]
        du = du[-m:M-n]
        
    if dim == 1:
        du = du.T
    return du


def jacobian_trans_my(x, y, nodes, boundary=(0, 0)):
    Xxi = diff2d(x, nodes, dim=0, boundary=boundary[0])
    Xeta = diff2d(x, nodes, dim=1, boundary=boundary[1])
    Yxi = diff2d(y, nodes, dim=0, boundary=boundary[0])
    Yeta = diff2d(y, nodes, dim=1, boundary=boundary[1])

    Xxixi = diff2d(Xxi, nodes, dim=0, boundary=boundary[0])
    Xxieta = diff2d(Xxi, nodes, dim=1, boundary=boundary[1])
    # Xetaxi = diff2d(Xeta, nodes, dim=0, boundary=boundary[0])
    Xetaeta = diff2d(Xeta, nodes, dim=1, boundary=boundary[1])

    Yxixi = diff2d(Yxi, nodes, dim=0, boundary=boundary[0])
    Yxieta = diff2d(Yxi, nodes, dim=1, boundary=boundary[1])
    # Yetaxi = diff2d(Yeta, nodes, dim=0, boundary=boundary[0])
    Yetaeta = diff2d(Yeta, nodes, dim=1, boundary=boundary[1])

    jac = torch.stack([Xxi, Yxi, Xeta, Yeta], dim=2).view(x.shape[0], x.shape[1], 2, 2)
    jac_inv = torch.inverse(jac)

    # 计算行列式J_det
    J_det = torch.det(jac)

    # 计算J_det对ξ和η的偏导
    J_xi = Xxixi * Yeta + Xxi * Yxieta - Xxieta * Yxi - Xeta * Yxixi
    J_eta = Xxieta * Yeta + Xxi * Yetaeta - Xetaeta * Yxi - Xeta * Yxieta

    # 从jac_inv中提取一阶逆导数
    xi_x = jac_inv[:,:,0,0]
    eta_x = jac_inv[:,:,0,1]
    xi_y = jac_inv[:,:,1,0]
    eta_y = jac_inv[:,:,1,1]

    # 计算∂(ξ_x)/∂ξ和∂(ξ_x)/∂η
    # ξ_x = (Yeta/J_det)
    # ∂(ξ_x)/∂ξ = [Yxieta * J_det - Yeta * J_xi] / J_det²
    # ∂(ξ_x)/∂η = [Yetaeta * J_det - Yeta * J_eta] / J_det²
    xi_x_xi = (Yxieta * J_det - Yeta * J_xi) / (J_det ** 2)
    xi_x_eta = (Yetaeta * J_det - Yeta * J_eta) / (J_det ** 2)

    # ξ_xx
    xi_xx = xi_x * xi_x_xi + eta_x * xi_x_eta

    # η_x = (-Yxi/J_det)
    # ∂(η_x)/∂ξ = [(-Yxixi)*J_det + Yxi*J_xi]/J_det²
    #           = (-Yxixi*J_det + Yxi*J_xi)/(J_det²)
    eta_x_xi = (-Yxixi * J_det + Yxi * J_xi) / (J_det ** 2)
    # ∂(η_x)/∂η = (-Yxieta*J_det + Yxi*J_eta)/J_det²
    eta_x_eta = (-Yxieta * J_det + Yxi * J_eta) / (J_det ** 2)

    # η_xx
    eta_xx = xi_x * eta_x_xi + eta_x * eta_x_eta

    # ξ_y = (-Xeta/J_det)
    # ∂(ξ_y)/∂ξ = (-Xxieta*J_det + Xeta*J_xi)/J_det²
    xi_y_xi = (-Xxieta * J_det + Xeta * J_xi) / (J_det ** 2)
    # ∂(ξ_y)/∂η = (-Xetaeta*J_det + Xeta*J_eta)/J_det²
    xi_y_eta = (-Xetaeta * J_det + Xeta * J_eta) / (J_det ** 2)

    # ξ_yy
    xi_yy = xi_y * xi_y_xi + eta_y * xi_y_eta

    # η_y = (Xxi/J_det)
    # ∂(η_y)/∂ξ = (Xxixi*J_det - Xxi*J_xi)/J_det²
    eta_y_xi = (Xxixi * J_det - Xxi * J_xi) / (J_det ** 2)
    # ∂(η_y)/∂η = (Xxieta*J_det - Xxi*J_eta)/J_det²
    eta_y_eta = (Xxieta * J_det - Xxi * J_eta) / (J_det ** 2)

    # η_yy
    eta_yy = xi_y * eta_y_xi + eta_y * eta_y_eta

    M_xx = torch.stack([xi_xx, eta_xx, xi_x ** 2, eta_x ** 2, 2 * xi_x * eta_x], dim=-1)  # [nx, ny, 5]
    M_yy = torch.stack([xi_yy, eta_yy, xi_y ** 2, eta_y ** 2, 2 * xi_y * eta_y], dim=-1)  # [nx, ny, 5]
    M = torch.stack([M_xx, M_yy], dim=-2)

    return jac, jac_inv, M


def jacobian_trans_my2(x, y, nodes, boundary=(0, 0)):
    Xxi = diff2d(x, nodes, dim=0, boundary=boundary[0])
    Xeta = diff2d(x, nodes, dim=1, boundary=boundary[1])
    Yxi = diff2d(y, nodes, dim=0, boundary=boundary[0])
    Yeta = diff2d(y, nodes, dim=1, boundary=boundary[1])

    Xxixi = diff2d(Xxi, nodes, dim=0, boundary=boundary[0])
    Xxieta = diff2d(Xxi, nodes, dim=1, boundary=boundary[1])
    Xetaeta = diff2d(Xeta, nodes, dim=1, boundary=boundary[1])

    Yxixi = diff2d(Yxi, nodes, dim=0, boundary=boundary[0])
    Yxieta = diff2d(Yxi, nodes, dim=1, boundary=boundary[1])
    Yetaeta = diff2d(Yeta, nodes, dim=1, boundary=boundary[1])

    jac = torch.stack([Xxi, Yxi, Xeta, Yeta], dim=2).view(x.shape[0], x.shape[1], 2, 2)
    jac_inv = torch.inverse(jac)

    xi_x = jac_inv[:,:,0,0]
    eta_x = jac_inv[:, :, 0, 1]
    xi_y = jac_inv[:, :, 1, 0]
    eta_y = jac_inv[:, :, 1, 1]

    xi_x_xi = diff2d(xi_x, nodes, dim=0, boundary=boundary[0])
    xi_x_eta = diff2d(xi_x, nodes, dim=1, boundary=boundary[1])
    xi_y_xi = diff2d(xi_y, nodes, dim=0, boundary=boundary[0])
    xi_y_eta = diff2d(xi_y, nodes, dim=1, boundary=boundary[1])

    eta_x_xi = diff2d(eta_x, nodes, dim=0, boundary=boundary[0])
    eta_x_eta = diff2d(eta_x, nodes, dim=1, boundary=boundary[1])
    eta_y_xi = diff2d(eta_y, nodes, dim=0, boundary=boundary[0])
    eta_y_eta = diff2d(eta_y, nodes, dim=1, boundary=boundary[1])

    xi_x_x = xi_x_xi*xi_x+xi_x_eta*eta_x
    xi_y_y = xi_y_xi*xi_y+xi_y_eta*eta_y

    eta_x_x = eta_x_xi * xi_x + eta_x_eta * eta_x
    eta_y_y = eta_y_xi * xi_y + eta_y_eta * eta_y

    M_xx = torch.stack([xi_x_x, eta_x_x, xi_x ** 2, eta_x ** 2, 2 * xi_x * eta_x], dim=-1)  # [nx, ny, 5]
    M_yy = torch.stack([xi_y_y, eta_y_y, xi_y ** 2, eta_y ** 2, 2 * xi_y * eta_y], dim=-1)  # [nx, ny, 5]
    M = torch.stack([M_xx, M_yy], dim=-2)

    return jac, jac_inv, M


def jacobian_trans_my10(x, y, nodes, boundary=(0, 0)):

    Xxi = compute_gradient(x, dim=0, periodic_boundary=1)
    Xeta = compute_gradient(x, dim=1, periodic_boundary=0)
    Yxi = compute_gradient(y, dim=0, periodic_boundary=1)
    Yeta = compute_gradient(y, dim=1, periodic_boundary=0)

    jac = torch.stack([Xxi, Yxi, Xeta, Yeta], dim=2).view(x.shape[0], x.shape[1], 2, 2)
    jac_inv = torch.inverse(jac)

    xi_x = jac_inv[:,:,0,0]
    eta_x = jac_inv[:, :, 0, 1]
    xi_y = jac_inv[:, :, 1, 0]
    eta_y = jac_inv[:, :, 1, 1]

    xi_x_xi = compute_gradient(xi_x, dim=0, periodic_boundary=1)
    xi_x_eta = compute_gradient(xi_x, dim=1, periodic_boundary=0)
    xi_y_xi = compute_gradient(xi_y, dim=0, periodic_boundary=1)
    xi_y_eta = compute_gradient(xi_y, dim=1, periodic_boundary=0)
    #
    eta_x_xi = compute_gradient(eta_x, dim=0, periodic_boundary=1)
    eta_x_eta = compute_gradient(eta_x, dim=1, periodic_boundary=0)
    eta_y_xi = compute_gradient(eta_y, dim=0, periodic_boundary=1)
    eta_y_eta = compute_gradient(eta_y, dim=1, periodic_boundary=0)

    xi_x_x = xi_x_xi*xi_x+xi_x_eta*eta_x
    xi_y_y = xi_y_xi*xi_y+xi_y_eta*eta_y

    eta_x_x = eta_x_xi * xi_x + eta_x_eta * eta_x
    eta_y_y = eta_y_xi * xi_y + eta_y_eta * eta_y

    M_xx = torch.stack([xi_x_x, eta_x_x, xi_x ** 2, eta_x ** 2, 2 * xi_x * eta_x], dim=-1)  # [nx, ny, 5]
    M_yy = torch.stack([xi_y_y, eta_y_y, xi_y ** 2, eta_y ** 2, 2 * xi_y * eta_y], dim=-1)  # [nx, ny, 5]
    M = torch.stack([M_xx, M_yy], dim=-2)

    return jac, jac_inv, M


def compute_gradient(input, dim, periodic_boundary=False): #对于翼型绕流只用dim=0才会用到周期性边界条件
    if periodic_boundary:
        # 周期性边界条件处理
        if dim == 0:
            input = torch.cat((input[-1:, :], input, input[:1, :]), dim=0)  # 周期性处理维度0
        elif dim == 1:
            input = torch.cat((input[:, -1:], input, input[:, :1]), dim=1)  # 周期性处理维度1
    # 计算梯度
    gradient = torch.gradient(input, dim=dim, spacing=1, edge_order=2)[0]

    # 如果使用周期性边界条件，返回时去掉新增的边界元素
    if periodic_boundary:
        if dim == 0:
            gradient = gradient[1:-1, :]  # 去掉边界的部分
        elif dim == 1:
            gradient = gradient[:, 1:-1]  # 去掉边界的部分

    return gradient