import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from calculate_closest_point_on_elipse import plot_3d_circle, homogenous_proj
from pupil_tracking_optimization import rotate
import torch
import csv

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

def order_pq_moment(x, y, p, q, center):
    s = 0
    for x1,y1 in zip(x,y):
        s += (x1-center[0])**p * (y1-center[1])**q
    return s

def save_data(data,i=0):
    # Specify the CSV file path
    csv_file_path = f'./data{i}.csv'

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        
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

def get_x(theta):
    global center, width, angle, height
    x = center[0] + width/2 * np.cos(theta) * np.cos(angle) - height/2 * np.sin(theta) * np.sin(angle)
    return x

def get_y(theta):
    global center, width, angle, height
    y = center[1] + width/2 * np.cos(theta) * np.sin(angle) + height/2 * np.sin(theta) * np.cos(angle)
    return y

def objective_function(theta):
    global original_point
    point = ((get_y(theta) - original_point[1])**2 + (get_x(theta) - original_point[0])**2)**0.5
    return point

def closest_point_on_elipse(x1,y1,initial_cond):
    closest_points = []
    global original_point
    theta = initial_cond
    for x,y in zip(x1,y1):
        original_point = [x,y]
        result = minimize(objective_function, theta)

        closest_points.append([get_x(result.x), get_y(result.x)])
        
        # plt.plot([original_point[0]], [original_point[1]], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
        # plt.plot([get_x(result.x)], [get_y(result.x)], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
    return closest_points

def get_x2(theta):
    global center2, width2, angle2, height2
    x = center2[0] + width2/2 * np.cos(theta) * np.cos(angle2) - height2/2 * np.sin(theta) * np.sin(angle2)
    return x

def get_y2(theta):
    global center2, width2, angle2, height2
    y = center2[1] + width2/2 * np.cos(theta) * np.sin(angle2) + height2/2 * np.sin(theta) * np.cos(angle2)
    return y

def objective_function2(theta):
    global original_point
    point = ((get_y2(theta) - original_point[1])**2 + (get_x2(theta) - original_point[0])**2)**0.5
    return point

def closest_point_on_elipse2(x1,y1,initial_cond):
    closest_points = []
    global original_point
    theta = initial_cond
    for x,y in zip(x1,y1):
        original_point = [x,y]
        result = minimize(objective_function2, theta)
        closest_points.append([get_x2(result.x), get_y2(result.x)])

        # plt.plot([original_point[0]], [original_point[1]], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
        # plt.plot([get_x2(result.x)], [get_y2(result.x)], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
    return closest_points


center = (0, 0)
width = 4
height = 2
angle = 45

original_point = [1,-1]

x,y = generate_ellipse(center, width, height, angle)

center1 = (0.3, -0.3)
x1,y1 = generate_ellipse(center1, width, height, angle)

theta = 0

thetax = torch.tensor(-20.)
thetay = torch.tensor(-40.)
thetaz = torch.tensor(10.)
r = torch.tensor(3.)
center = torch.tensor([-212.,-105.,2407.])

thetax1 = torch.tensor(-19.9926)
thetay1 = torch.tensor(-40.0137)
thetaz1 = torch.tensor(10.0004)
r1 = torch.tensor(3.4)
center1 = torch.tensor([-210.3,-107.5,2406.5])

rotated_points1 = rotate(thetax, thetay, thetaz, r, center)
rotated_points2 = rotate(thetax1, thetay1, thetaz1, r1, center1)

r1 = rotated_points1.detach().numpy()
r2 = rotated_points2.detach().numpy()

# plot_3d_circle(r1[0], r1[1], r1[2],r2)

prediction1a = homogenous_proj(rotated_points1, camMatrix1).detach().numpy()
prediction1b = homogenous_proj(rotated_points1, camMatrix2).detach().numpy()

prediction2a = homogenous_proj(rotated_points2, camMatrix1).detach().numpy()
prediction2b = homogenous_proj(rotated_points2, camMatrix2).detach().numpy()

save_data(prediction2a, i=10)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))
# ax1.plot(prediction1a[0], prediction1a[1])
# x,y = generate_ellipse([472,260],36.9910,7.0147,-0.8781, is_radian=True)
x,y = generate_ellipse([475.0385,252.2438],40.9194,7.7458,-0.8781, is_radian=True)

# ax1.plot(prediction2a[0], prediction2a[1], color="red")
# ax1.plot(x,y,color="blue")

# ax2.plot(prediction1b[0], prediction1b[1])
x,y = generate_ellipse([417.0743,228.4001],42.6008,7.2318,-0.8537, is_radian=True)

# ax2.plot(x, y, color='blue')
# ax2.plot(prediction2b[0], prediction2b[1],color="red")

# plt.tight_layout()
# plt.show()

center = [475.0385,252.2438]
width = 40.9194
height = 7.7458
angle = -0.8781

center2 = [417.0743,228.4001]
width2 = 42.6008
height2 = 7.2318
angle2 = -0.8537

x1,y1 = generate_ellipse(center2, width2, height2, angle2, is_radian=True)
points = np.array([[x,y] for x, y in zip(x1,y1)])
center_of_mass = np.mean(points, axis=0)


# mu20 = order_pq_moment(x1, y1, 2, 0, center=center_of_mass)
# mu02 = order_pq_moment(x1, y1, 0, 2, center=center_of_mass)
# mu11 = order_pq_moment(x1, y1, 1, 1, center=center_of_mass)

# print(0.5*np.arctan(2*mu11 / (mu20-mu02)))

# x2,y2 = generate_ellipse(center2, width2, height2, angle2, is_radian=True)

# plt.plot(x1,y1)
# closest_points = closest_point_on_elipse([465,466,467,468,469,470],[267,266,265,264,263,262], initial_cond=1)
# xs = [l[0] for l in closest_points]
# ys = [l[1] for l in closest_points]
# plt.plot([465,466,467,468,469,470],[267,266,265,264,263,262], ".", color="red")
# plt.plot(xs,ys,".")
# closest_point_on_elipse2(prediction2a[0][:500], prediction2a[1][:500], initial_cond = 10)
# closest_point_on_elipse2(prediction2a[0][500:], prediction2a[1][500:], initial_cond = 0)

plt.show()
