import numpy as np
import torch
import random
import math
import matplotlib.pyplot as plt


# load train/val data path as lists
def data_loader(dataset, data_path, multiview=4):
    image_paths = []
    bbox = []
    gt_d3_list = []
    gt_d2_list = []
    for j in range(multiview):
        image_paths.append([])
        bbox.append([])
        gt_d2_list.append([])
        gt_d3_list.append([])

    if dataset == "coco":
        base_path = "./datasets/"
        arr = np.genfromtxt(base_path+"coco_path.txt", dtype=str, delimiter=';')
        n_line = open(base_path+"coco_path.txt").read().count('\n')
        for line in range(n_line):
            for j in range(multiview):
                if int(arr[line][0]) % multiview == j:
                    image_paths[j].append(data_path + "coco/" + arr[line][1])
                    bbox[j].append(torch.tensor([int(arr[line][2]), int(arr[line][3]),
                                                     int(arr[line][4]), int(arr[line][5])]))
                    gt = torch.zeros([17, 2])
                    for i in range(17):
                        for k in range(2):
                            gt[i, k] = float(arr[line][2 * i + k + 6])
                    gt_d2_list[j].append(gt)

        return image_paths, bbox, gt_d2_list


# axisangle to R
def rot_from_axisangle(vec):
    # Convert an axisangle rotation into a 3x3 transformation matrix
    # (adapted from https://github.com/Wallacoloo/printipi)
    # Input 'vec' has to be Nx3

    angle = torch.norm(vec, 2, 1, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 3, 3)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)

    return rot


# R to axisangle
def rot_to_axis_angle(rot):
    # Convert an axisangle rotation from a 3x3 transformation matrix
    # (adapted from https://github.com/Wallacoloo/printipi)
    # Input 'rot' has to be Nx3x3
    # Axes.
    batchsize, _, _ = rot.shape
    axis = torch.zeros(batchsize, 3).to(device=rot.device)
    axis[:, 0] = rot[:, 2, 1] - rot[:, 1, 2]
    axis[:, 1] = rot[:, 0, 2] - rot[:, 2, 0]
    axis[:, 2] = rot[:, 1, 0] - rot[:, 0, 1]

    # Angle.
    r = torch.linalg.norm(axis, dim=1)
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    theta = torch.arctan(r/(trace-1))

    # Normalise the axis.
    axis = axis / r.unsqueeze(1) * theta.unsqueeze(1)

    # Return the data.
    return axis


# boundary values that diffusion can reach
def craft_generation_boundary():
    pi = 3.1416
    rho = [0.15, 0.45, 0.45, 0.25, 0.25, 0.12, 0.2, 0.17, 0.3, 0.25]
    x = torch.tensor([[[rho[0], rho[0]], [0.0, pi], [-pi, pi]],
                      [[rho[0], rho[0]], [0.0, pi*5/6], [-pi, pi]],
                      [[rho[1], rho[1]], [pi/3, pi*2/3], [-pi, pi]],
                      [[rho[2], rho[2]], [0.0, pi*5/6], [-pi, pi]],
                      [[rho[0], rho[0]], [pi/6, pi], [-pi, pi]],
                      [[rho[1], rho[1]], [pi/3, pi*2/3], [-pi, pi]],
                      [[rho[2], rho[2]], [0.0, pi*5/6], [-pi, pi]],
                      [[rho[3], rho[3]], [0.0, pi*3/4], [-pi, pi]],
                      [[rho[4], rho[4]], [0.0, pi*2/5], [-pi, pi]],
                      [[rho[5], rho[5]], [0.0, pi*3/5], [-pi, pi]],
                      [[rho[6], rho[6]], [0.0, pi/2], [-pi, pi]],
                      [[rho[7], rho[7]], [pi*2/5, pi*3/4], [-pi, pi]],
                      [[rho[8], rho[8]], [0.0, pi*3/5], [-pi, pi]],
                      [[rho[9], rho[9]], [0.0, pi*9/10], [-pi, pi]],
                      [[rho[7], rho[7]], [pi*2/5, pi*3/4], [-pi, pi]],
                      [[rho[8], rho[8]], [0.0, pi*3/5], [-pi, pi]],
                      [[rho[9], rho[9]], [0.0, pi*9/10], [-pi, pi]]])
    return x


# 3 equations based on 3 head keypoints to solve lambda_prop
def get_pixel_cm_propotion_size(AB_2d, BC_2d, AC_2d, sign_B, sign_C, AB_p=0.12, BC_p=0.12, AC_p=0.2):
    # A Neck, B Nose, C Top, dB, dC, size unknown. size2=size^2
    # dB^2 = (AB_p*size)^2-AB_2d^2
    # dC^2 = (AC_p*size)^2-AC_2d^2
    # (signB*dB-signC*dC)^2 = (BC_p*size)^2-BC_2d^2

    # a=AB_p^2, b=AC_p^2, c=BC_p^2, d=AB_2D^2, e=AC_2d^2, f=BC_2d^2, s=size^2
    # 2 signB*signC * dB*dC = as-d + bs-e - cs+f
    # 4 (as-d)(bs-e) = (as-d + bs-e - cs+f)^2
    # 4 abs2 - 4(ae+bd)s +4de = (a+b-c)2s2 -2(a+b-c)(d+e-f)s+(d+e-f)2
    a = AB_p ** 2
    b = AC_p ** 2
    c = BC_p ** 2
    d = torch.square(AB_2d)
    e = torch.square(AC_2d)
    f = torch.square(BC_2d)
    A = -4*a*b + ((a+b-c) ** 2)
    B = 4*(a*e+b*d) - 2*(a+b-c)*(d+e-f)
    C = -4*d*e + torch.square(d+e-f)
    size1 = (-B+torch.sqrt(torch.abs(B**2-4*A*C)))/(2*A)
    size2 = (-B-torch.sqrt(torch.abs(B**2-4*A*C)))/(2*A)
    if size2.item()<=0 or (a*size2.item()<d.item()) or (b*size2.item()<e.item()) or (c*size2.item()<f.item()):
        size = torch.sqrt(torch.abs(size1))
    elif (a*size1.item()<d.item()) or (b*size1.item()<e.item()) or (c*size1.item()<f.item()):
        size = torch.sqrt(torch.abs(size2))
    else:
        sign1 = ((a+b-c)*size1-(d+e-f))*sign_B*sign_C
        sign2 = ((a+b-c)*size2-(d+e-f))*sign_B*sign_C
        if sign1.item() <=0:
            size = torch.sqrt(torch.abs(size2))
        else:
            size = torch.sqrt(torch.abs(size1))
    return size


# Semi-automatic algorithm to lift 2D to 3D
def craft_lifted_3d_poses(vec, depth_sign=None, depth_factor=0.8, size_version=2,
                          rho = [0.15, 0.45, 0.45, 0.25, 0.25, 0.12, 0.12, 0.2, 0.17, 0.3, 0.25]):
    N, C, _ = vec.shape
    list_relation = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(8,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
    rho_id = [0,1,2,0,1,2,3,4,5,7,8,9,10,8,9,10]

    output = torch.cat((vec, depth_sign), dim=2)
    for i in range(N):
        if size_version == 2:
            AB_2d = torch.sqrt(torch.sum(torch.square(vec[i, 8, :] - vec[i, 9, :])))
            BC_2d = torch.sqrt(torch.sum(torch.square(vec[i, 9, :] - vec[i, 10, :])))
            AC_2d = torch.sqrt(torch.sum(torch.square(vec[i, 8, :] - vec[i, 10, :])))
            size = get_pixel_cm_propotion_size(AB_2d, BC_2d, AC_2d, output[i, 9, -1], output[i, 10, -1], rho[5],
                                                rho[6], rho[7])
        else:
            size = torch.sqrt(torch.sum(torch.square(vec[i, 0, :] - vec[i, 7, :]))) / rho[3]
        if size.item() == 0:
            size += 0.001
        for j in range(16):
            p,q = list_relation[j]
            z2 = torch.square(size*rho[rho_id[j]]) - torch.sum(torch.square((vec[i, p, :] - vec[i, q, :]))) + 0.000001*random.random()
            if z2<0:
                z2=-z2
            output[i, q, 2] = output[i,p,2]+output[i,q,2]*(torch.sqrt(z2)) * depth_factor
        output[i,...] /= size.item()
    return output


def get_limb(x, y, z=None, id1=0, id2=1):
    if z is not None:
        return np.concatenate((np.expand_dims(x[id1], 0), np.expand_dims(x[id2], 0)), 0), \
               np.concatenate((np.expand_dims(y[id1], 0), np.expand_dims(y[id2], 0)), 0), \
               np.concatenate((np.expand_dims(z[id1], 0), np.expand_dims(z[id2], 0)), 0)
    else:
        return np.concatenate((np.expand_dims(x[id1], 0), np.expand_dims(x[id2], 0)), 0), \
               np.concatenate((np.expand_dims(y[id1], 0), np.expand_dims(y[id2], 0)), 0)


# vec: 1xCx2or3
def draw_skeleton(vec, plt_show=True, save_path=None, inverse_z=True, background=None):
    _, _, d = vec.shape
    X = vec.numpy()
    list_branch_head = [(8, 9), (9, 10)]
    list_branch_left_arm = [(8, 11), (11, 12), (12, 13)]
    list_branch_right_arm = [(8, 14), (14, 15), (15, 16)]
    list_branch_body = [(0, 7), (7, 8)]
    list_branch_right_foot = [(0, 1), (1, 2), (2, 3)]
    list_branch_left_foot = [(0, 4), (4, 5), (5, 6)]

    if d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)

        # joints
        if inverse_z:
            zdata = -X[0, :, 1]
        else:
            zdata = X[0, :, 1]
        xdata = X[0, :, 0]
        ydata = X[0, :, 2]
        ax.scatter(xdata, ydata, zdata, c='r')

        # fake bounding box of image
        max_range = np.array([xdata.max() - xdata.min(), ydata.max() - ydata.min(), zdata.max() - zdata.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xdata.max() + xdata.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ydata.max() + ydata.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (zdata.max() + zdata.min())

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        # background
        if background is not None:
            WidthX = Xb[7] - Xb[0]
            WidthZ = Zb[7] - Zb[0]

            arr = np.array(background.getdata()).reshape(background.size[1], background.size[0], 3).astype(float)
            arr = arr / arr.max()
            stepX, stepZ = WidthX / arr.shape[1], WidthZ / arr.shape[0]

            X1 = np.arange(0, -Xb[0]+Xb[7], stepX)
            Z1 = np.arange(Zb[7], Zb[0], -stepZ)
            X1, Z1 = np.meshgrid(X1, Z1)
            Y1 = Z1 * 0.0 + Zb[7] + 0.01

            ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, facecolors=arr, shade=False)

        # branches
        for (id1, id2) in list_branch_head:
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='red')
        for (id1, id2) in list_branch_body:
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='orange')
        for (id1, id2) in list_branch_left_arm:
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='blue')
        for (id1, id2) in list_branch_right_arm:
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='violet')
        for (id1, id2) in list_branch_left_foot:
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='cyan')
        for (id1, id2) in list_branch_right_foot:
            x, y, z = get_limb(xdata, ydata, zdata, id1, id2)
            ax.plot(x, y, z, color='pink')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if plt_show:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path)

    if d == 2:
        fig = plt.figure()
        ax = plt.axes()

        xdata = X[0, :, 0]
        ydata = -X[0, :, 1]
        ax.scatter(xdata, ydata, c='r')

        for (id1, id2) in list_branch_head:
            x, y = get_limb(xdata, ydata, None, id1, id2)
            ax.plot(x, y, color='red')
        for (id1, id2) in list_branch_body:
            x, y = get_limb(xdata, ydata, None, id1, id2)
            ax.plot(x, y, color='orange')
        for (id1, id2) in list_branch_left_arm:
            x, y = get_limb(xdata, ydata, None, id1, id2)
            ax.plot(x, y, color='blue')
        for (id1, id2) in list_branch_right_arm:
            x, y = get_limb(xdata, ydata, None, id1, id2)
            ax.plot(x, y, color='violet')
        for (id1, id2) in list_branch_left_foot:
            x, y = get_limb(xdata, ydata, None, id1, id2)
            ax.plot(x, y, color='cyan')
        for (id1, id2) in list_branch_right_foot:
            x, y = get_limb(xdata, ydata, None, id1, id2)
            ax.plot(x, y, color='pink')

        max_range = np.array([xdata.max() - xdata.min(), ydata.max() - ydata.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xdata.max() + xdata.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ydata.max() + ydata.min())
        for xb, yb in zip(Xb, Yb):
            ax.plot([xb], [yb], 'w')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if plt_show:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path)


def skeleton_coco_to_h36m(vec, num_keypoint=17, dim=3):  # Linear interpolation for some joints due to layout difference
    N, _, _ = vec.shape
    joints_index = [-1, 12, 14, 16, 11, 13, 15, -2, -3, 0, -4, 5, 7, 9, 6, 8, 10]
    joint = torch.zeros(N, num_keypoint, dim).to(vec.device)
    for j in range(17):
        if joints_index[j] >= 0:
            joint[:, j, :] = vec[:, joints_index[j], :dim]
    joint[:, 0, :] = (joint[:, 1, :] + joint[:, 4, :]) / 2.0
    joint[:, 8, :] = (joint[:, 11, :] + joint[:, 14, :] + joint[:, 9, :]) / 3.0
    joint[:, 7, :] = (joint[:, 0, :] + joint[:, 8, :]) / 2.0
    joint[:, 10, :] = (joint[:, 8, :] * 2.0 + joint[:, 7, :] * 2.0 + joint[:, 9, :]) / 5.0 * 2.5 - joint[:, 7, :] * 1.5
    return joint


# y-axis always horizontal
# from x0-x1 locate next joint, x0,x1: Nx3, rho: N, theta_phi: Nx2
def next_joint(x0, x1, rho, theta_phi):
    out = x1.clone().detach()
    arr_x = rho * torch.sin(theta_phi[..., 0]) * torch.cos(theta_phi[..., 1])
    arr_y = rho*torch.sin(theta_phi[..., 0])*torch.sin(theta_phi[..., 1])
    arr_z = rho * torch.cos(theta_phi[..., 0])
    vec = x1-x0
    rnoz = torch.sqrt(torch.square(vec[..., 0]) + torch.square(vec[..., 1]))

    if torch.min(rnoz).item() > 0:
        r = torch.sqrt(torch.square(vec[..., 0]) + torch.square(vec[..., 1]) + torch.square(vec[..., 2]))
        ax = vec[..., 0]
        ay = vec[..., 1]
        az = vec[..., 2]
        exx = ax * az / (rnoz * r)
        exy = ay * az / (rnoz * r)
        exz = -rnoz / r
        eyx = -ay / rnoz
        eyy = ax / rnoz
        eyz = 0
        ezx = ax / r
        ezy = ay / r
        ezz = az / r
        out[:, 0] += arr_x * exx + arr_y * eyx + arr_z * ezx
        out[:, 1] += arr_x * exy + arr_y * eyy + arr_z * ezy
        out[:, 2] += arr_x * exz + arr_y * eyz + arr_z * ezz
    else:
        N = x1.shape[0]
        for i in range(N):
            if rnoz[i] > 0:
                r = torch.sqrt(torch.square(vec[i:(i+1), 0]) + torch.square(vec[i:(i+1), 1])
                               + torch.square(vec[i:(i+1), 2]))
                ax = vec[i:(i+1), 0]
                ay = vec[i:(i+1), 1]
                az = vec[i:(i+1), 2]
                exx = ax * az / (rnoz[i:(i+1)] * r)
                exy = ay * az / (rnoz[i:(i+1)] * r)
                exz = -rnoz[i:(i+1)] / r
                eyx = -ay / rnoz[i:(i+1)]
                eyy = ax / rnoz[i:(i+1)]
                eyz = 0
                ezx = ax / r
                ezy = ay / r
                ezz = az / r
                out[i:(i+1), 0] += arr_x[i:(i+1)] * exx + arr_y[i:(i+1)] * eyx + arr_z[i:(i+1)] * ezx
                out[i:(i+1), 1] += arr_x[i:(i+1)] * exy + arr_y[i:(i+1)] * eyy + arr_z[i:(i+1)] * ezy
                out[i:(i+1), 2] += arr_x[i:(i+1)] * exz + arr_y[i:(i+1)] * eyz + arr_z[i:(i+1)] * ezz

            else:
                out[i:(i+1), 0] += arr_x[i:(i+1)]
                out[i:(i+1), 1] += arr_y[i:(i+1)]
                out[i:(i+1), 2] += arr_z[i:(i+1)]
    return out


# find angles <ABC from 3 3D points, A,B,C: Nx3
def compute_angle(A, B, C):
    ba = A - B
    bc = C - B

    cosine_angle = torch.sum(ba*bc, dim=-1)/(torch.sqrt(torch.sum(torch.square(ba), dim=-1))
                                             * torch.sqrt(torch.sum(torch.square(bc), dim=-1)))

    return torch.nan_to_num(torch.acos(cosine_angle), nan=0)


# find horizontal y axis from root, A,B: Nx3, out: Nx3, Nx3
def compute_xy_unit_axis(A, B):
    N = A.shape[0]
    BY = next_joint(A, B, torch.ones(N).to(A.device),
                    theta_phi=torch.tensor([[math.pi/2, math.pi/2]]).repeat(N, 1).to(A.device))
    BX = next_joint(A, B, torch.ones(N).to(A.device),
                    theta_phi=torch.tensor([[math.pi/2, 0]]).repeat(N, 1).to(A.device))
    return BX, BY


# find rho, theta phi for a branch which can be created by C=next_joint(A,B,rho,theta,phi), A,B,C: Nx3
def find_rho_theta_phi(A, B, C):
    N = A.shape[0]
    output = torch.zeros(N, 3).to(A.device)
    output[:, 0] = torch.sqrt(torch.sum(torch.square(B-C), dim=1))
    A_bar = 2*B-A
    output[:, 1] = compute_angle(A_bar, B, C)
    BX, BY = compute_xy_unit_axis(A, B)
    C_bar = C + output[:, 0:1]*torch.cos(output[:, 1:2]) *\
            (A-B)/torch.sqrt(torch.sum(torch.square(A-B), dim=1, keepdim=True))
    output[:, 2] = compute_angle(BX, B, C_bar)
    sign = (torch.sum((C_bar-B) * (BY-B), dim=-1) >= 0)*2.0-1.0
    output[:, 2] *= sign
    return output


# find all 17 sets of rho data phi, vec: Nx17x3
def find_17_rho_theta_phi(vec):
    OX = torch.tensor([1, 0, 0]).unsqueeze(0).to(vec.device)
    OZ = torch.tensor([0, 0, 1]).unsqueeze(0).to(vec.device)
    N = vec.shape[0]
    output = torch.zeros(N, 17, 3).to(vec.device)
    output[:, 0, :] = find_rho_theta_phi(vec[:, 0, :]+OX, vec[:, 0, :], vec[:, 4, :])
    output[:, 1, :] = find_rho_theta_phi(vec[:, 0, :]-OZ, vec[:, 0, :], vec[:, 1, :])
    output[:, 2, :] = find_rho_theta_phi(vec[:, 0, :], vec[:, 1, :], vec[:, 2, :])
    output[:, 3, :] = find_rho_theta_phi(vec[:, 1, :], vec[:, 2, :], vec[:, 3, :])
    output[:, 4, :] = find_rho_theta_phi(vec[:, 0, :]-OZ, vec[:, 0, :], vec[:, 4, :])
    output[:, 5, :] = find_rho_theta_phi(vec[:, 0, :], vec[:, 4, :], vec[:, 5, :])
    output[:, 6, :] = find_rho_theta_phi(vec[:, 4, :], vec[:, 5, :], vec[:, 6, :])
    output[:, 7, :] = find_rho_theta_phi(vec[:, 0, :]-OZ, vec[:, 0, :], vec[:, 7, :])
    output[:, 8, :] = find_rho_theta_phi(vec[:, 0, :], vec[:, 7, :], vec[:, 8, :])
    output[:, 9, :] = find_rho_theta_phi(vec[:, 7, :], vec[:, 8, :], vec[:, 9, :])
    output[:, 10, :] = find_rho_theta_phi(vec[:, 7, :], vec[:, 8, :], vec[:, 10, :])
    output[:, 11, :] = find_rho_theta_phi(vec[:, 7, :], vec[:, 8, :], vec[:, 11, :])
    output[:, 12, :] = find_rho_theta_phi(vec[:, 8, :], vec[:, 11, :], vec[:, 12, :])
    output[:, 13, :] = find_rho_theta_phi(vec[:, 11, :], vec[:, 12, :], vec[:, 13, :])
    output[:, 14, :] = find_rho_theta_phi(vec[:, 7, :], vec[:, 8, :], vec[:, 14, :])
    output[:, 15, :] = find_rho_theta_phi(vec[:, 8, :], vec[:, 14, :], vec[:, 15, :])
    output[:, 16, :] = find_rho_theta_phi(vec[:, 14, :], vec[:, 15, :], vec[:, 16, :])

    return output


# rho theta phi to skeleton, inverse of find_17_rho_theta_phi
def rho_theta_phi_to_skeleton(rtp):
    N = rtp.shape[0]
    X = torch.zeros(N, 17, 3).to(rtp.device)
    X[:, 1, :] = next_joint(X[:, 0, :], X[:, 0, :], rtp[:, 1, 0], rtp[:, 1, 1:])
    X[:, 2, :] = next_joint(X[:, 0, :], X[:, 1, :], rtp[:, 2, 0], rtp[:, 2, 1:])
    X[:, 3, :] = next_joint(X[:, 1, :], X[:, 2, :], rtp[:, 3, 0], rtp[:, 3, 1:])
    X[:, 4, :] = next_joint(X[:, 0, :], X[:, 0, :], rtp[:, 4, 0], rtp[:, 4, 1:])
    X[:, 5, :] = next_joint(X[:, 0, :], X[:, 4, :], rtp[:, 5, 0], rtp[:, 5, 1:])
    X[:, 6, :] = next_joint(X[:, 4, :], X[:, 5, :], rtp[:, 6, 0], rtp[:, 6, 1:])
    X[:, 7, :] = next_joint(X[:, 0, :], X[:, 0, :], rtp[:, 7, 0], rtp[:, 7, 1:])
    X[:, 8, :] = next_joint(X[:, 0, :], X[:, 7, :], rtp[:, 8, 0], rtp[:, 8, 1:])
    X[:, 9, :] = next_joint(X[:, 7, :], X[:, 8, :], rtp[:, 9, 0], rtp[:, 9, 1:])
    X[:, 10, :] = next_joint(X[:, 7, :], X[:, 8, :], rtp[:, 10, 0], rtp[:, 10, 1:])
    X[:, 11, :] = next_joint(X[:, 7, :], X[:, 8, :], rtp[:, 11, 0], rtp[:, 11, 1:])
    X[:, 12, :] = next_joint(X[:, 8, :], X[:, 11, :], rtp[:, 12, 0], rtp[:, 12, 1:])
    X[:, 13, :] = next_joint(X[:, 11, :], X[:, 12, :], rtp[:, 13, 0], rtp[:, 13, 1:])
    X[:, 14, :] = next_joint(X[:, 7, :], X[:, 8, :], rtp[:, 14, 0], rtp[:, 14, 1:])
    X[:, 15, :] = next_joint(X[:, 8, :], X[:, 14, :], rtp[:, 15, 0], rtp[:, 15, 1:])
    X[:, 16, :] = next_joint(X[:, 14, :], X[:, 15, :], rtp[:, 16, 0], rtp[:, 16, 1:])
    return X


# sampling according to a distribution, current B with counting existed valued, target B, target_values (B+1)
def sampling(current, target, target_values, batch_size=1):
    output = torch.zeros(batch_size).to(current.device)
    indexs = torch.zeros(batch_size).to(current.device)
    for i in range(batch_size):
        if torch.sum(current).item() > 0.0:
            current_distribution = current / torch.sum(current)
        else:
            current_distribution = current + 1.0
            current_distribution = current_distribution / torch.sum(current_distribution)

        if torch.sum(target).item() > 0.0:
            target_distribution = target / torch.sum(target)
        else:
            target_distribution = target+1.0
            target_distribution = target_distribution/torch.sum(target_distribution)

        can_pick_sample = (target_distribution + 0.0001 >= current_distribution) * (target_distribution > 0.0)
        num_can_pick_sample = torch.sum(can_pick_sample*1.0)
        index_in_num_can_pick_sample = torch.nonzero(can_pick_sample*1.0, as_tuple=True)[0]
        index_0 = (torch.rand(1).to(current.device)*num_can_pick_sample).int().item()
        index = index_in_num_can_pick_sample[index_0]

        indexs[i] = index
        output[i] = torch.rand(1).to(current.device)*(target_values[index+1]-target_values[index])+target_values[index]
        current[index] += 1.0
    return output, indexs


# create a skeleton with a given 2D distribution using markov tree structure
# current_dtb, target_dtb: BxBx(C-1)x3, target_values (B+1)x2x(C-1)x3
# root_current, root_target: Bx3, root_values (B+1)x3
def create_skeleton_17(current_dtb, target_dtb, target_values, root_current, root_target, root_values, batchsize=1):
    list_joint_pairs = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (8, 10), (4, 11),
                        (11, 12), (12, 13), (11, 14), (14, 15), (15, 16)]

    bins, _, num_keypoints, _ = current_dtb.shape
    num_keypoints += 1
    rtp = torch.zeros(batchsize, num_keypoints, 3).to(current_dtb.device)
    indexs = torch.zeros(batchsize, num_keypoints, 3).to(current_dtb.device)

    for j in range(3):
        rtp[:, 0, j], indexs[:, 0, j] = sampling(root_current[:, j], root_target[:, j],
                                                 root_values[:, j], batch_size=batchsize)

    for i in range(len(list_joint_pairs)):
        m, n = list_joint_pairs[i]
        for j in range(3):
            for k in range(batchsize):
                ref_idx = indexs[k, m, j].int()
                if i < 10:
                    rtp[k:(k+1), n, j], indexs[k:(k+1), n, j] = sampling(current_dtb[ref_idx, :, i, j],
                                                                         target_dtb[ref_idx, :, i, j],
                                                                         target_values[:, 1, i, j], batch_size=1)
                else:
                    rtp[k:(k+1), n, j], indexs[k:(k+1), n, j] = sampling(current_dtb[ref_idx, :, i+1, j],
                                                                         target_dtb[ref_idx, :, i+1, j],
                                                                         target_values[:, 1, i+1, j], batch_size=1)

    output = rho_theta_phi_to_skeleton(rtp)

    X_abs = torch.cat((output[:, :, 0:1], -output[:, :, 2:3], output[:, :, 1:2]), 2)

    return X_abs


# 1D or 2D diffusion simulating thermal diffusion equation
def distribution_diffusion(dtb, dim=2, alpha=0):
    if dim == 2:
        N, _, C, D = dtb.shape
        dtb_copy = torch.zeros((N+2), (N+2), C, D).to(dtb.device)
        dtb_copy[1:(N+1), 1:(N+1), ...] = dtb
        dtb_copy[0, 1:(N+1), ...] = dtb_copy[1, 1:(N+1), ...]
        dtb_copy[N+1, 1:(N+1), ...] = dtb_copy[N, 1:(N+1), ...]
        dtb_copy[1:(N + 1), 0, ...] = dtb_copy[1:(N + 1), 1, ...]
        dtb_copy[1:(N + 1), N+1, ...] = dtb_copy[1:(N + 1), N, ...]
        laplacian_dtb = dtb_copy[0:N, 1:(N+1), ...]+dtb_copy[2:(N+2), 1:(N+1), ...] +\
                        dtb_copy[1:(N+1), 0:N, ...]+dtb_copy[1:(N+1), 2:(N+2), ...]-4*dtb_copy[1:(N+1), 1:(N+1), ...]
        if torch.is_tensor(alpha):
            dtb += alpha.unsqueeze(1).unsqueeze(0).unsqueeze(0) * laplacian_dtb
        else:
            dtb += alpha * laplacian_dtb
        dtb /= torch.sum(torch.sum(dtb, dim=0, keepdim=True), dim=1, keepdim=True)
    if dim == 1:
        N, D = dtb.shape
        dtb_copy = torch.zeros((N + 2), D).to(dtb.device)
        dtb_copy[1:(N + 1), ...] = dtb
        dtb_copy[0, ...] = dtb_copy[1, ...]
        dtb_copy[N + 1, ...] = dtb_copy[N, ...]
        laplacian_dtb = dtb_copy[0:N, ...] + dtb_copy[2:(N + 2), ...] - 2 * dtb_copy[1:(N + 1), ...]
        dtb += alpha * laplacian_dtb
        dtb /= torch.sum(dtb, dim=0, keepdim=True)


# pjpe_loss size NxCx3
def pjpe_loss(output, gt, align=False):
    diff = output-gt
    if align:
        diff = diff - diff[..., 0:1, :]
        diff = diff[..., 1:, :]
    return torch.mean(torch.mean(torch.sqrt(torch.sum(torch.square(diff), dim=-1, keepdim=False)),
                                 dim=-1, keepdim=False), dim=-1, keepdim=False)


# pck counts NxCx3
def pck_count(output, gt, align=False, head_id=[8, 10], threshold=None):
    N, C, _ = output.shape
    out = torch.tensor([0, 0])
    diff = output-gt
    if align:
        diff = diff - diff[..., 0:1, :]
        diff = diff[..., 1:, :]
    diff = torch.sqrt(torch.sum(torch.square(diff), dim=-1, keepdim=False))  # NxC
    if threshold is None:
        threshold = gt[:, head_id[0], :] - gt[:, head_id[1], :]
        threshold = torch.sqrt(torch.sum(torch.square(threshold), dim=-1, keepdim=True))/2.0  # Nx1
    count = torch.sum(torch.gt(diff, threshold) * 1)
    if align:
        out[0] = N * (C-1) - count
        out[1] = N * (C-1)
    else:
        out[0] = N*C-count
        out[1] = N*C
    return out


def forbenius_norm_multi_dim(vec):
    d = len(vec.shape)
    output = vec.square()
    for i in range(d-1):
        output = output.sum(axis=i+1, keepdim=True)
    for i in range(d-1):
        output /= float(vec.shape[i+1])
    return torch.sqrt(output)


def loss_no_scale(p2d, p3d, num_keypoint=17, confs=None, keep_batch=False, keep_keypoint=False):
    N = p2d.shape[0]
    if confs is None:
        p2d_copy = p2d
        p3d_copy = p3d
    else:
        useful_joint = torch.repeat_interleave((confs > 0.001)*1.0, 2, dim=1)
        p2d_copy = p2d * useful_joint
        p3d_copy = p3d * useful_joint

    scale_p2d = forbenius_norm_multi_dim(p2d_copy)
    p2d_scaled = p2d_copy/scale_p2d

    scale_p3d = forbenius_norm_multi_dim(p3d_copy)
    p3d_scaled = p3d_copy/scale_p3d

    loss = (p2d_scaled - p3d_scaled).abs().reshape(N, num_keypoint, -1).sum(axis=2)

    if confs is not None:
        loss *= confs
    if keep_batch and keep_keypoint:
        return loss
    elif keep_batch:
        loss = loss.sum(axis=1) / (p2d_scaled.shape[1])
    elif keep_keypoint:
        loss = loss.sum(axis=0) / (p2d_scaled.shape[0])
    else:
        loss = loss.sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

    return loss
