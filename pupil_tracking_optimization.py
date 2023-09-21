import torch
import torch.nn as nn
import torch.optim as optim

def rotx(angle):
    """
    Create a 3x3 rotation matrix around the X-axis.
    :param angle: The rotation angle in radians.
    :return: 3x3 rotation matrix.
    """

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    rot1 = torch.stack((torch.tensor(1), torch.tensor(0), torch.tensor(0)))
    rot2 = torch.stack((torch.tensor(0), torch.cos(angle), -1 * torch.sin(angle)))
    rot3 = torch.stack((torch.tensor(0), torch.sin(angle), torch.cos(angle)))
    rotation_matrix = torch.stack((rot1, rot2, rot3))
    
    # rotation_matrix = torch.tensor([[1, 0, 0],
    #                                 [0, cos_angle, -sin_angle],
    #                                 [0, sin_angle, cos_angle]])
    return rotation_matrix

def roty(angle):
    
    # Compute sin and cos of the angle
    sin_theta = torch.sin(angle)
    cos_theta = torch.cos(angle)

    rot1 = torch.stack((torch.cos(angle), torch.tensor(0), torch.sin(angle)))
    rot2 = torch.stack((torch.tensor(0), torch.tensor(1), torch.tensor(0)))
    rot3 = torch.stack((-1*torch.sin(angle), torch.tensor(0), torch.cos(angle)))
    rotation_matrix = torch.stack((rot1, rot2, rot3))

    # Create the 3x3 rotation matrix for rotation about the Y-axis
    # rotation_matrix = torch.tensor([[cos_theta, 0, sin_theta],
    #                                 [0, 1, 0],
    #                                 [-sin_theta, 0, cos_theta]])
    return rotation_matrix

def rotz(angle):

    # Compute sin and cos of the angle
    sin_theta = torch.sin(angle)
    cos_theta = torch.cos(angle)

    rot1 = torch.stack((torch.cos(angle), -1*torch.sin(angle), torch.tensor(0)))
    rot2 = torch.stack((torch.sin(angle), torch.cos(angle), torch.tensor(0)))
    rot3 = torch.stack((torch.tensor(0), torch.tensor(0), torch.tensor(1)))
    rotation_matrix = torch.stack((rot1, rot2, rot3))

    # Create the 3x3 rotation matrix for rotation about the Y-axis
    # rotation_matrix = torch.tensor([[cos_theta, -sin_theta, 0],
    #                             [sin_theta, cos_theta, 0],
    #                             [0, 0, 1]])
    
    return rotation_matrix


def rotate(thetax, thetay, thetaz, r, center):

    thetas = torch.linspace(0,2*3.14,500)
    xs = center[0] + r * torch.cos(thetas)
    ys = center[1] + r * torch.sin(thetas)
    zs = center[2] + r * torch.zeros(xs.shape)

    points = torch.stack((xs,ys,zs))

    xrotate = rotx(thetax)
    yrotate = roty(thetay)
    zrotate = rotz(thetaz)

    rotation_matrix = torch.matmul(torch.matmul(xrotate,yrotate),zrotate)

    center = torch.transpose(center.unsqueeze(dim=0), 0, 1)

    rotated_points = torch.matmul(rotation_matrix,points-center) + center
    
    return rotated_points

def homogenous_proj(pointsthreeD, camMat):
    pointsthreeD_hom = torch.stack((pointsthreeD[0,:].unsqueeze(dim=0), pointsthreeD[1,:].unsqueeze(dim=0), pointsthreeD[2, :].unsqueeze(dim=0), torch.ones(pointsthreeD.shape[1]).unsqueeze(dim=0)), dim=1).squeeze(dim=0)
    pointstwoD_hom = torch.matmul(camMat, pointsthreeD_hom)
    pointstwoD = pointstwoD_hom[:2, :] / pointstwoD_hom[2, :]
    return pointstwoD


def get_data():
    import pandas as pd

    # Specify the path to the CSV file
    csv_file_path = '~/Downloads/Untitled spreadsheet - Sheet1 (1).csv'

    # Read the CSV file using pandas
    df = pd.read_csv(csv_file_path)

    # Convert DataFrame to a NumPy array
    array_data = df.values

    cam1Data = df.values[1:, 5:7]
    cam2Data = df.values[1:, 1:3]
    cam1Data = torch.tensor(cam1Data.tolist())
    cam2Data = torch.tensor(cam2Data.tolist())
    return cam1Data.T, cam2Data.T



def calculate():
    thetax = torch.tensor(-20.)
    thetax.requires_grad = True

    thetay = torch.tensor(-40.)
    thetay.requires_grad = True

    thetaz = torch.tensor(10.)
    thetaz.requires_grad = True

    r = torch.tensor(3.)
    r.requires_grad = True

    center = torch.tensor([-210.,-106.,2406.])
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

    target1, target2 = get_data()
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer1 = optim.SGD([r, center], lr=0.01)
    optimizer2 = optim.SGD([thetax, thetay, thetaz], lr=0.01)
    num_epochs = 30000
    for epoch in range(num_epochs):
        rotated_points = rotate(thetax, thetay, thetaz, r, center)
        prediction1 = homogenous_proj(rotated_points, camMatrix1)
        # prediction2 = homogenous_proj(rotated_points, camMatrix2)

        # find closest point to ground truth elipse
        cp1 = closest_point_on_elipse(prediction1[0][:500], prediction1[1][:500], initial_cond = 10)
        cp2 = closest_point_on_elipse(prediction1[0][500:], prediction1[1][500:], initial_cond = 0)
        
        target1 = cp1 + cp2
        target1 = torch.tensor(target1)

        loss1 = criterion1(prediction1, target1)
        # loss2 = criterion2(prediction2, target2)

        total_loss = loss1#  + loss2

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        total_loss.backward()
        '''
        print(thetax)
        print(thetay)
        print(thetaz)
        print(r)
        print(center)
        '''
        optimizer1.step()
        optimizer2.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item()}")


    print(thetax)
    print(thetay)
    print(thetaz)
    print(r)
    print(center)