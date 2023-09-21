## A Pupi Tracking Algorithm
Detailed mathematical description can be found [link].

### 1. How to use
Install relevant libraries using `conda create --name <env> --file requirements.txt`.

To use the algorithm, provide the parameters for 2D ground truth ellipse parameters as arguments for running the script below in the order of c1/c2 = center = (x,y), a1/a2 = major axis(semimajor axis * 2) and b1/b2 = minor axis(semiminor axis * 2), and theta1/theta2 = angle of rotation theta. 

`python optimize_axis.py -c1 475.0385 252.2438 -a1 40.9194 -b1 7.7458 -theta1 -0.8781 -c2 417.0743 228.4001 -a2 42.6008 -b2 7.2318 -theta2 -0.8537`

In the above, `-c1 number1 number2` maps to `(x,y) = (number1, number2)`.

The script will print all the losses/errors peridically, and it will visualize the initial and final 3D circle and its corresponding 2D projections as well as the ground truth 2D projections.

The script will output on the screen the optimized parameters of circle in 3D space.

### 2. Setting hyperparameters
If you wish to use different hyperparameters such as camera projection matrix, number of gradient descent steps(epochs), learning rate, etc., please refer to the corresponding file to modify directly.

Camera projection matrix: `camMatrix1` and `camMatrix2` on the bottom of   `optimize_axis.py`.

Initial guess of 3D parameters: `thetax`, `thetay`, `thetaz`, `r`, `center` on the bottom of `optimize_axis.py`.

Epochs: `num_epochs` in `calculate()` in `optimize_axis.py`.

learning rate: `lr` in `calculate()` in `optimize_axis.py`.


### 3. Code Structure
The main algorithm to calculate loss/error and performs gradient descent is `calculate()` of `optimize_axis.py`, and procedure that does rotation about x, y, z axis, calculates homogeneous transform and projection are in `pupil_tracking_optimization.py`, corresponding methods are `rotx`, `roty`, `rotz`, `rotate` and `homogenous_proj`. 

Iterative algorithm to find the major/minor axis as described in the pdf is found `find_major_minor_axis()` in `optimize_axis.py`.

### 4. Others
The functions for initial optimize-by-point-matching method can be found in `test.py`.