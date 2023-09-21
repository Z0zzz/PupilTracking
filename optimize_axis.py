import torch
from pupil_tracking_optimization import rotate, homogenous_proj
# from test import closest_point_on_elipse, closest_point_on_elipse2, generate_ellipse
# from test import center as c1
# from test import width as w1
# from test import height as h1
# from test import angle as a1
# from test import center2 as c2
# from test import width2 as w2
# from test import height2 as h2
# from test import angle2 as a2
from calculate_closest_point_on_elipse import plot_3d_circle
import numpy as np
from test_drawing import find_major_minor_axis
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import argparse

def generate_ellipse(center, width, height, angle, is_radian=False):

    # Generate angles
    theta = np.linspace(0, 2*np.pi, 50)

    if is_radian:
        x = center[0] + width/2 * np.cos(theta) * np.cos(angle) - height/2 * np.sin(theta) * np.sin(angle)
        y = center[1] + width/2 * np.cos(theta) * np.sin(angle) + height/2 * np.sin(theta) * np.cos(angle)
    else:
        # Parametric equations of the ellipse
        x = center[0] + width/2 * np.cos(theta) * np.cos(np.radians(angle)) - height/2 * np.sin(theta) * np.sin(np.radians(angle))
        y = center[1] + width/2 * np.cos(theta) * np.sin(np.radians(angle)) + height/2 * np.sin(theta) * np.cos(np.radians(angle))

    return (x,y)

def euclidean_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def order_pq_moment(x, y, p, q, center):
    s = torch.sum((x-center[0])**p * (y-center[1])**q)
    return s

def find_major_minor_axis(points,center):
    min_p = list()
    min_dist = float("inf")
    max_p = list()
    max_dist = float("-inf")

    x = list(points[0].clone().detach().numpy())
    y = list(points[1].clone().detach().numpy())

    iterations = points.shape[1]
    for i in range(iterations):
        p = [x[i], y[i]]
        d = euclidean_distance(p, center)
        if d > max_dist:
            max_dist = d
            max_idx = i
            max_p = p
        elif d < min_dist:
            min_dist = d
            min_idx = i
            min_p = p

    return min_idx, min_p, max_idx, max_p


def calculate():
    
    global c1, c2, a1, a2
    global thetax, thetay, thetaz, r, center
    global camMatrix1, camMatrix2
    # ground truth
    thetax1 = torch.tensor(-19.9926)
    thetay1 = torch.tensor(-40.0137)
    thetaz1 = torch.tensor(10.0004)
    r1 = torch.tensor(3.4)
    center1 = torch.tensor([-210.3,-107.5,2406.5])

    # thetax = torch.tensor(-19.9996)
    # thetax.requires_grad = True

    # thetay = torch.tensor(-40.)
    # thetay.requires_grad = True

    # thetaz = torch.tensor(9.9997)
    # thetaz.requires_grad = True

    # r = torch.tensor(3.0004)
    # r.requires_grad = True

    # center = torch.tensor([-210.3566, -107.5195, 2407.0444])
    # center.requires_grad = True

    # thetax = torch.tensor(-20.)
    # thetax.requires_grad = True

    # thetay = torch.tensor(-40.)
    # thetay.requires_grad = True

    # thetaz = torch.tensor(10.)
    # thetaz.requires_grad = True

    # r = torch.tensor(3.)
    # r.requires_grad = True
    
    # # center = torch.tensor([-215., -110., 2409.])
    # center = torch.tensor([-193.,-97.,2300.])
    # center.requires_grad = True

    rotated_points1 = rotate(thetax, thetay, thetaz, r, center)
    rotated_points2 = rotate(thetax1, thetay1, thetaz1, r1, center1)

    rr1 = rotated_points1.detach().numpy()
    rr2 = rotated_points2.detach().numpy()

    plot_3d_circle(rr1[0], rr1[1], rr1[2],rr2)


    rotated_points = rotate(thetax, thetay, thetaz, r, center)

    # camMatrix1 = torch.tensor([
    #     [14563.59278,	0,	1747.722805,	0],
    #     [0,	14418.55592,	896.3267034,	0],
    #     [0,	0,	1,	0],
    # ])

    # camMatrix2 = torch.tensor([
    #     [14911.09787,	-224.3242906,	2180.916566,	-1150110.844],
    #     [106.4257579,	14620.19877,	795.2746281,	220494.7087],
    #     [-0.12459133,	-0.04392728697,	0.9912352869,	-50.84174996]
    # ])
    
    lr = 0.01
    criterion1 = nn.MSELoss()
    optimizer1 = optim.SGD([r, center]+[thetax, thetay, thetaz], lr=lr)
    num_epochs = 10000
    scheduler = CosineAnnealingLR(optimizer1, T_max=num_epochs, eta_min=0.0001)
    
    # c1_trial = torch.tensor([474.9870, 252.2632])
    # c2_trial = torch.tensor([417.1281, 228.3979])

    c1 = torch.tensor(c1)
    c2 = torch.tensor(c2)
    aa1 = torch.tensor(a1)
    aa2 = torch.tensor(a2)

    losses = []
    for epoch in range(num_epochs):

        rotated_points = rotate(thetax, thetay, thetaz, r, center)
        prediction1 = homogenous_proj(rotated_points, camMatrix1)
        prediction1copy = prediction1.clone()
        prediction2 = homogenous_proj(rotated_points, camMatrix2)
        prediction2copy = prediction2.clone()

        if epoch == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].set_title('cam proj 1')
            axes[0].plot(prediction1copy[0].detach().numpy(), prediction1copy[1].detach().numpy(), linewidth=2.5, linestyle='--', label='predicted')
            x,y = generate_ellipse(c1, w1, h1, a1, is_radian=True)
            axes[0].plot(x, y, linewidth=1.5, linestyle='-', label='target')
            axes[0].legend()

            axes[1].set_title('cam proj 2')
            axes[1].plot(prediction2copy[0].detach().numpy(), prediction2copy[1].detach().numpy(), linewidth=2.5, linestyle='--', label='predicted')
            x,y = generate_ellipse(c2, w2, h2, a2, is_radian=True)
            axes[1].plot(x, y, linewidth=1.5, linestyle='-', label='target')
            axes[1].legend()
            
            plt.show()

        trial_center1 = torch.mean(prediction1, axis=1)
        trial_center2 = torch.mean(prediction2, axis=1)

        pred_minor1, minor_p1, pred_maj1, maj_p1 = find_major_minor_axis(prediction1, c1)
        pred_minor2, minor_p2, pred_maj2, maj_p2 = find_major_minor_axis(prediction2, c2)

        # prediction1_maj = torch.norm(prediction1.T[pred_maj1] - c1_trial)
        # prediction2_maj = torch.norm(prediction2.T[pred_maj2] - c2_trial)

        # prediction1_minor = torch.norm(prediction1.T[pred_minor1] - c1_trial)
        # prediction2_minor = torch.norm(prediction2.T[pred_minor2] - c2_trial)

        prediction1_maj = torch.norm(prediction1.T[pred_maj1] - c1)
        prediction2_maj = torch.norm(prediction2.T[pred_maj2] - c2)

        prediction1_minor = torch.norm(prediction1.T[pred_minor1] - c1)
        prediction2_minor = torch.norm(prediction2.T[pred_minor2] - c2)

        target1_maj = torch.tensor(w1) / 2
        target2_maj = torch.tensor(w2) / 2

        target1_minor = torch.tensor(h1) / 2
        target2_minor = torch.tensor(h2) / 2

        # loss from major axis
        loss1_maj = torch.abs(prediction1_maj - target1_maj)
        loss2_maj = torch.abs(prediction2_maj - target2_maj)
        partition1 = loss1_maj.detach() / (loss1_maj.detach()+loss2_maj.detach())
        partition2 = loss2_maj.detach() / (loss1_maj.detach()+loss2_maj.detach())
        total_loss_maj = partition1*loss1_maj + partition2*loss2_maj

        # loss from minor axis
        loss1_minor = torch.abs(prediction1_minor - target1_minor)
        loss2_minor = torch.abs(prediction2_minor - target2_minor)
        partition1 = loss1_minor.detach() / (loss1_minor.detach()+loss2_minor.detach())
        partition2 = loss2_minor.detach() / (loss1_minor.detach()+loss2_minor.detach())
        total_loss_minor = partition1*loss1_minor + partition2*loss2_minor

        # loss from center
        loss1_center = criterion1(trial_center1, c1)
        loss2_center = criterion1(trial_center2, c2)
        partition1 = loss1_center.detach() / (loss1_center.detach()+loss2_center.detach())
        partition2 = loss2_center.detach() / (loss1_center.detach()+loss2_center.detach())
        total_loss_center = partition1*loss1_center + partition2*loss2_center


        # loss from angle
        mu20_1 = order_pq_moment(prediction1[0], prediction1[1], 2, 0, center=trial_center1)
        mu02_1 = order_pq_moment(prediction1[0], prediction1[1], 0, 2, center=trial_center1)
        mu11_1 = order_pq_moment(prediction1[0], prediction1[1], 1, 1, center=trial_center1)

        mu20_2 = order_pq_moment(prediction2[0], prediction2[1], 2, 0, center=trial_center2)
        mu02_2 = order_pq_moment(prediction2[0], prediction2[1], 0, 2, center=trial_center2)
        mu11_2 = order_pq_moment(prediction2[0], prediction2[1], 1, 1, center=trial_center2)

        angle1 = 0.5*torch.arctan(2*mu11_1 / (mu20_1-mu02_1))
        angle2 = 0.5*torch.arctan(2*mu11_2 / (mu20_2-mu02_2))
        if angle1 < 0:
            angle1 = angle1 + torch.tensor(3.14)/2
        else:
            angle1 = angle1 - torch.tensor(3.14)/2
        if angle2 < 0:
            angle2 = angle2 + torch.tensor(3.14)/2
        else:
            angle2 = angle2 - torch.tensor(3.14)/2
    
        loss1_angle = torch.abs(angle1 - aa1)
        loss2_angle = torch.abs(angle2 - aa2)
        partition1 = loss1_angle.detach() / (loss1_angle.detach()+loss2_angle.detach())
        partition2 = loss2_angle.detach() / (loss1_angle.detach()+loss2_angle.detach())
        total_loss_angle = partition1*loss1_angle + partition2*loss2_angle

        # total loss
        total_loss = total_loss_maj + total_loss_minor + total_loss_center + total_loss_angle
        losses.append(total_loss.detach().item())
        if epoch % 1000 == 0:
            print("prediction 1: ", prediction1_maj, "target: ", target1_maj, "point: ", maj_p1, "loss: ", loss1_maj)
            print("prediction 2: ", prediction2_maj, "target: ", target2_maj, "point: ", maj_p2, "loss: ", loss2_maj)
            print("prediction 1: ", prediction1_minor, "target: ", target1_minor, "point: ", minor_p1, "loss: ", loss1_minor)
            print("prediction 2: ", prediction2_minor, "target: ", target2_minor, "point: ", minor_p2, "loss: ", loss2_minor)
            print("prediction 1: ", angle1, "target: ", aa1, "loss: ", loss1_angle)
            print("prediction 2: ", angle2, "target: ", aa2, "loss: ", loss2_angle)
            print("#"*20)


        optimizer1.zero_grad()
        total_loss.backward()
 
        optimizer1.step()
        scheduler.step()

    cc1 = torch.mean(prediction1copy, axis=1)
    cc2 = torch.mean(prediction2copy, axis=1)

    print("current center 1: ", cc1)
    print("current center 2: ", cc2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].set_title('cam proj 1')
    axes[0].plot(prediction1copy[0].detach().numpy(), prediction1copy[1].detach().numpy(), linewidth=2.5, linestyle='--', label='predicted')
    x,y = generate_ellipse(c1, w1, h1, a1, is_radian=True)
    axes[0].plot(x, y, linewidth=1.5, linestyle='-', label='target')
    axes[0].legend()

    axes[1].set_title('cam proj 2')
    axes[1].plot(prediction2copy[0].detach().numpy(), prediction2copy[1].detach().numpy(), linewidth=2.5, linestyle='--', label='predicted')
    x,y = generate_ellipse(c2, w2, h2, a2, is_radian=True)
    axes[1].plot(x, y, linewidth=1.5, linestyle='-', label='target')
    axes[1].legend()
    
    plt.show()

    print(thetax)
    print(thetay)
    print(thetaz)
    print(r)
    print(center)

    rotated_points1 = rotate(thetax, thetay, thetaz, r, center)
    rotated_points2 = rotate(thetax1, thetay1, thetaz1, r1, center1)

    rr1 = rotated_points1.detach().numpy()
    rr2 = rotated_points2.detach().numpy()

    plot_3d_circle(rr1[0], rr1[1], rr1[2],rr2)

    bins = np.linspace(0,num_epochs,num_epochs)
    plt.plot(bins[3:], losses[3:])
    plt.show()


print("------------------optimize_axis.py------------------")
# Initialize the parser
parser = argparse.ArgumentParser(description="An example script with argparse")

# Add arguments
parser.add_argument('-c1', nargs='+', type=float, help="a tuple of 2 numbers representing ")
parser.add_argument('-a1', type=float, help="Path to the input file")
parser.add_argument('-b1', type=float,  help="Path to the output file (default: output.txt)")
parser.add_argument('-theta1', type=float, help="Enable verbose mode")

parser.add_argument('-c2', nargs='+', type=float, help="a tuple of 2 numbers representing ")
parser.add_argument('-a2', type=float, help="Path to the input file")
parser.add_argument('-b2', type=float,  help="Path to the output file (default: output.txt)")
parser.add_argument('-theta2', type=float, help="Enable verbose mode")

# Parse the arguments
args = parser.parse_args()
print(args)

c1 = args.c1
w1 = args.a1
h1 = args.b1
a1 = args.theta1
c2 = args.c2
w2 = args.a2
h2 = args.b2
a2 = args.theta2

# modify here: camera projection matrices
camMatrix1 = torch.tensor([
    [14563.59278,	0,	1747.722805,	0],
    [0,	14418.55592,	896.3267034,	0],
    [0,	0,	1,	0],
])

camMatrix2 = torch.tensor([
    [14911.09787,	-224.3242906,	2180.916566,	-1150110.844],
    [106.4257579,	14620.19877,	795.2746281,	220494.7087],
    [-0.12459133,	-0.04392728697,	0.9912352869,	-50.84174996]
])

# modify here: initial guess of 3D parameters
thetax = torch.tensor(-20.)
thetax.requires_grad = True

thetay = torch.tensor(-40.)
thetay.requires_grad = True

thetaz = torch.tensor(10.)
thetaz.requires_grad = True

r = torch.tensor(3.)
r.requires_grad = True

center = torch.tensor([-215., -110., 2409.])
center.requires_grad = True

calculate()