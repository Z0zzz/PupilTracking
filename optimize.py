import torch
from pupil_tracking_optimization import rotate, homogenous_proj
from test import closest_point_on_elipse, closest_point_on_elipse2, generate_ellipse
from test import center as c1
from test import width as w1
from test import height as h1
from test import angle as a1
from test import center2 as c2
from test import width2 as w2
from test import height2 as h2
from test import angle2 as a2
# width, height, angle, center2, width2, height2, angle2 as c1,w1,h1,a1,c2,w2,h2,a2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import random



def calculate():

    thetax = torch.tensor(-20.)
    thetax.requires_grad = True

    thetay = torch.tensor(-40.)
    thetay.requires_grad = True

    thetaz = torch.tensor(10.)
    thetaz.requires_grad = True

    r = torch.tensor(3.)
    r.requires_grad = True

    center = torch.tensor([-211.,-107.,2407.])
    center.requires_grad = True

    rotated_points = rotate(thetax, thetay, thetaz, r, center)


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

    # target1, target2 = get_data()
    criterion1 = nn.MSELoss()
    # criterion2 = nn.MSELoss()
    optimizer1 = optim.SGD([r, center]+[thetax, thetay, thetaz], lr=0.01)
    # optimizer2 = optim.SGD([thetax, thetay, thetaz], lr=0.01)
    num_epochs = 50
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scheduler = CosineAnnealingLR(optimizer1, T_max=num_epochs, eta_min=0.0001)
    for epoch in range(num_epochs):

        rotated_points = rotate(thetax, thetay, thetaz, r, center)
        prediction1 = homogenous_proj(rotated_points, camMatrix1)
        prediction1copy = prediction1.clone()
        prediction2 = homogenous_proj(rotated_points, camMatrix2)
        prediction2copy = prediction2.clone()
        # axes[1].plot(prediction2[0].detach().numpy(), prediction2[1].detach().numpy())
        # find closest point to ground truth elipse
        cp1 = closest_point_on_elipse(prediction1[0][:500].detach().numpy(), prediction1[1][:500].detach().numpy(), initial_cond = 10)
        cp2 = closest_point_on_elipse(prediction1[0][500:].detach().numpy(), prediction1[1][500:].detach().numpy(), initial_cond = 1)

        target1 = cp1 + cp2

        cp1 = closest_point_on_elipse2(prediction2[0][:500].detach().numpy(), prediction2[1][:500].detach().numpy(), initial_cond = 10)
        cp2 = closest_point_on_elipse2(prediction2[0][500:].detach().numpy(), prediction2[1][500:].detach().numpy(), initial_cond = 1)
        
        target2 = cp1 + cp2
        # plt.plot(target1[0], target1[1])
        # plt.show()

        target1 = torch.tensor(target1).squeeze(-1).float()
        target2 = torch.tensor(target2).squeeze(-1).float()

        prediction1 = prediction1.T
        prediction2 = prediction2.T

        loss1 = criterion1(prediction1, target1)
        loss2 = criterion1(prediction2, target2)
        
        partition1 = loss1.detach() / (loss1.detach()+loss2.detach())
        partition2 = loss2.detach() / (loss1.detach()+loss2.detach())

        total_loss = partition1*loss1 + partition2*loss2
        # total_loss = loss1
        print("loss1: ", loss1)
        print("loss2: ", loss2)

        optimizer1.zero_grad()
        # optimizer2.zero_grad()
        total_loss.backward()
 
        optimizer1.step()
        scheduler.step()
        # optimizer2.step()
        if epoch % 10 == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            pp = prediction1.detach().numpy()
            tt = target1.numpy()
            for i in range(10):
                p = pp[i]
                t = tt[i]
                axes[0].plot([p[0], t[0]], [p[1], t[1]], marker='.', linestyle='-', color='b')    
                axes[0].scatter([p[0], t[0]], [p[1], t[1]], color='r')

            axes[0].set_title('cam proj 1')
            axes[0].plot(prediction1copy[0].detach().numpy(), prediction1copy[1].detach().numpy(), linewidth=2.5, linestyle='--', label='predicted')
            target1copy = target1.T
            axes[0].plot(target1copy[0], target1copy[1], linewidth=1.5, linestyle='-', label='target')
            axes[0].legend()

            axes[1].set_title('cam proj 2')
            axes[1].plot(prediction2copy[0].detach().numpy(), prediction2copy[1].detach().numpy(), linewidth=2.5, linestyle='--', label='predicted')
            target2copy = target2.T
            axes[1].plot(target2copy[0], target2copy[1], linewidth=1.5, linestyle='-', label='target')
            axes[1].legend()

            plt.show() 
            print(f"Epoch {epoch}, Loss: {total_loss.item()}")

    print(thetax)
    print(thetay)
    print(thetaz)
    print(r)
    print(center)
    
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

calculate()