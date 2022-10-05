import numpy as np
import pandas as pd
import itertools
import torch
import matplotlib.pyplot as plt
import math
from SAXSsimulations.create_form import Simulation
from SAXSsimulations.utils import safe_multiplier, safe_dividend


class Cylinder(Simulation):
    """
    Puts polydispersed cylinders into the box.
    """

    def __init__(self, size, nPoints, volFrac=0.05):
        Simulation.__init__(self, size, nPoints, volFrac)
        self.hWidth = None
        self.hMean = None
        self.rWidth = None
        self.rMean = None
        self.theta = None
        self.phi = None
        self.center = None
        self.shape = "cylinder"
        self.rotWidth = 3  # VARIATION IS 3 DEGREES
        self.shapes = 0

    def place_shape(self, single=False, **kwargs):
        """
        Updates the Cylinder attributes and places one or many cylinders into the box. If the standard deviation for hight and radius are
        not defined, sets the default values.
        """
        keys = ["rMean", "rWidth", "hMean", "hWidth", "center"]
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in keys)
        if self.rWidth is None:
            self.rWidth = 0.1
        if self.hWidth is None:
            self.hWidth = 0.15
        self.__cylinder_in_box(single)

    def __cylinder_in_box(self, single):
        """

        Given a box fill it with cylinder(s). Cylinder must fit into box, otherwise it is discarded until a fitting one is sampled.
        Box might fit many cylinder if the volume fractuion is too small, unless the 'single' argument is specified.
        Calls the generate_cylinder function that creates cylinders as slices.
        Radius Mean and Hight Mean are sampled from normal distribution or were specified when instantiating a Cylinder object.
        Cylinder radius and height are sampled from a normal distribution around the respective Means.
        Center is sampled from the uniform distribution or was specified for a single cylinder.
        Theta angle is sampled to indicate rotation between the cylinder axis to the beam  and Phi to indicate the  rotation about the beam.
        input:
            single[boolean] : create a single sphere in a box?
        """
        if self.phi is None:
            self.phi = int(np.random.uniform(low=0, high=360))
        if self.theta is None:
            self.theta = 3
            self.theta_distribution = "gaussian"
            self.phi_distribution = "uniform"

        if self.rMean is None:
            self.rMean = -1
            while self.rMean <= 0:
                self.rMean = np.random.normal(
                    loc=self.box_size * 0.02, scale=self.box_size * 0.05
                )
                print("rMean", self.rMean)

        if self.hMean is None:
            self.hMean = -1
            while self.hMean <= 0:
                self.hMean = np.random.normal(
                    loc=self.box_size * 0.4, scale=self.box_size * 0.1
                )
        attempt = 0
        self.theta_all = []
        self.phi_all = []
        if single:
            success = False
            while success == False and attempt < 100:
                attempt += 1
                if self.center is not None and (
                    not (
                        (self.center > self.box_size / 2)
                        | (self.center < -self.box_size / 2)
                        == True
                    ).any()
                ):
                    continue  # center was passed and is inside box
                else:
                    self.center = np.random.uniform(
                        low=-self.box_size / 2 + self.rMean,
                        high=self.box_size / 2 - self.rMean,
                        size=3,
                    )
                self.__generate_cylinder(
                    self.rMean, self.hMean, self.center, self.theta, self.phi
                )
                success = self.pbc
                if success == False:
                    self.center = None

            self.shapes = 1
            print(
                "volume fraction is {vf:.5f}, height is {h:.2f}, radius is {r:.2f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}), rotation phi is {phi}, rotation theta is {theta} ".format(
                    vf=self.volume_fraction,
                    h=self.hMean,
                    r=self.rMean,
                    cx=self.center[0],
                    cy=self.center[1],
                    cz=self.center[2],
                    phi=self.phi,
                    theta=self.theta,
                )
            )
        else:
            while (
                self.volume_fraction < self.volume_fraction_threshold and attempt < 100
            ):
                success = False
                while success == False:
                    height = np.random.normal(loc=self.hMean, scale=self.hWidth)
                    radius = np.random.normal(loc=self.rMean, scale=self.rWidth)
                    center = np.random.uniform(
                        low=-self.box_size / 2, high=self.box_size / 2, size=3
                    )
                    if self.theta_distribution == "gaussian":
                        theta = int(
                            np.random.normal(loc=self.theta, scale=self.rotWidth)
                        )
                        phi = int(np.random.uniform(low=0, high=45))
                        self.phi_all.append(phi)
                    elif self.phi_distribution == "gaussian":
                        theta = int(np.random.uniform(low=0, high=45))
                        phi = int(np.random.normal(loc=self.phi, scale=self.rotWidth))
                        self.theta_all.append(theta)

                    if (
                        (
                            (center > self.box_size / 2) | (center < -self.box_size / 2)
                            == True
                        ).any()
                        or (radius < 0)
                        or (height < 0)
                    ):
                        continue  # center is outside of box or radius or height is negatibve
                    self.__generate_cylinder(radius, height, center, theta, phi)
                    success = self.pbc

                    if success:
                        self.shapes += 1
                        print(
                            "volume fraction is {vf:.5f}, height is {h:.3f}, radius is {r:.3f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}), rotation phi is {phi}, rotation theta is {theta} ".format(
                                vf=self.volume_fraction,
                                h=height,
                                r=radius,
                                cx=center[0],
                                cy=center[1],
                                cz=center[2],
                                phi=phi,
                                theta=theta,
                            )
                        )
                    attempt += 1

    def __points_in_cylinder(self, pt1, pt2, r, q):
        vec = pt2 - pt1
        const = r * np.linalg.norm(vec)
        return np.array(
            (np.dot(q - pt1, vec) >= 0)
            * (np.dot(q - pt2, vec) <= 0)
            * (np.linalg.norm(np.cross(q - pt1, vec), axis=1) <= const),
            dtype=float,
        )

    def edge_pbc(self, pt1, pt2, r, edge_current):
        edge_plane = (
            np.vstack(np.meshgrid(edge_current[0], edge_current[1], edge_current[2]))
            .reshape(3, -1)
            .T
        )
        edge_density = self.__points_in_cylinder(pt1, pt2, r, edge_plane)
        if (
            edge_density.sum() > 0
        ):  # the edge of the box will contain a density of a cylinder: pbc will not hold!
            self.pbc = False

    def __cylinder_pbc(self, pt1, pt2, r):
        self.pbc = True
        edge = [self.grid, self.grid, self.grid]
        for d in range(3):
            edge_current = edge.copy()
            edge_current[d] = float(self.grid.min())
            self.edge_pbc(pt1, pt2, r, edge_current)
            if not self.pbc:
                break
            edge_current[d] = float(self.grid.max())
            self.edge_pbc(pt1, pt2, r, edge_current)
            if not self.pbc:
                break

    def __generate_cylinder(self, radius, height, center, theta, phi):

        coords = np.array(np.meshgrid(self.grid, self.grid, self.grid))
        coordsArr = np.vstack(coords).reshape(3, -1).T
        pointO = np.array(
            [center[0], center[1], center[2] + height * np.cos(np.deg2rad(theta))]
        )
        cylinder_end = [
            center[0]
            + np.cos(np.deg2rad(phi / 2))
            * 2
            * height
            * np.sin(np.deg2rad(theta))
            * np.sin(np.deg2rad(phi / 2)),
            center[1]
            + np.sin(np.deg2rad(theta)) * height
            - np.sin(np.deg2rad(phi / 2))
            * 2
            * height
            * np.sin(np.deg2rad(theta))
            * np.sin(np.deg2rad(phi / 2)),
            center[2] + np.cos(np.deg2rad(theta)) * height,
        ]
        self.__cylinder_pbc(center, cylinder_end, radius)
        if self.pbc:
            cylinder = self.__points_in_cylinder(
                pt1=center, pt2=np.array(cylinder_end), r=radius, q=coordsArr
            )
            cylinder = np.array(cylinder).reshape(
                self.nPoints, self.nPoints, self.nPoints
            )
            self._box = np.logical_or(
                self._box, torch.from_numpy(cylinder).to(torch.bool)
            )

    def save_data(self, uncertainty="ISigma", directory=".", for_SasView=True):
        """
        Saves .dat file. If slice  of 3D Fourier Transform was created only, operates on that slice, otherwise on whole data.
        input:
            directory to save
            for_SasView: boolean, if True converts Q and I to SASView compartible values: Armstrong^-1 for Q and (m*sr)^-1.
        """
        Q = self.Q[self.nPoints // 2 + 1, :, :].numpy()
        data = pd.DataFrame(
            {
                "Qx": self.binned_slice["qy"],
                "Qy": self.binned_slice["qz"],
                "I": self.binned_slice["I"],
                "ISigma": self.binned_slice[uncertainty],
            }
        )
        if for_SasView:
            data.assign(
                Qx=data.Qx / 10,
                Qy=data.Qy / 10,
                I=data.I / 100,
                ISigma=data.ISigma / 100,
            ).to_csv(
                directory
                + "/polydispersed_cylinders_{r}_{h}.dat".format(
                    r=int(self.rMean * 1000), h=int(self.hMean * 1000)
                ),
                header=None,
                index=None,
                columns=["Qx", "Qy", "I", uncertainty],
            )
        else:
            data.to_csv(
                directory
                + "/polydispersed_cylinders_{r}_{h}.dat".format(
                    r=int(self.rMean * 1000), h=int(self.hMean * 1000)
                ),
                header=None,
                index=None,
                columns=["Qx", "Qy", "I", uncertainty],
            )


'''

    def __create_slice(self, height, r_theta,r_phi, center, theta, phi, d, cap_start,cap_small, direction_right, check):
        """
        Creates a slice of a cyllinder:
        input:
            height : the height of the cylinder
            r_theta: ellipse semi-axis in the direction of z-axis
            r_phi : ellipse semi-axis in the direction of y-axis
            center : center of the current ellipse
            theta: rotation angle of current cylinder in the direction of z-axis
            phi: rotation angle of current cylinder in the direction of y-axis
            d : distance from the center of the cylinder to the curretn slice along the x-axis
            cap_start[boolean] : the cylinder capping began, ellipses will be masked
            cap_small[boolean] : the smaller part of the capping began
            direction_right: the direction of the cylinder with respect to the x-axis: when True is positive,
            check: is the check that th cylinder fits the box needed? only done for the last two slices. They are not allowed to
            "touch" borders of the simulation box or not be present in a box at all.
        """
        if direction_right:
            center_y = center[1] + safe_multiplier(d,np.tan(np.deg2rad(phi))) if phi !=0 else center[1] # because of the theta rotation the y-coordinate of center at fixed x  slice shifts
            center_z = center[2] + safe_multiplier(d,np.tan(np.deg2rad(theta))) if theta !=0 else center[2] # because of phi rotation the z-coordinate of center at fixed x shifts
        else:
            center_y = center[1] - safe_multiplier(d,np.tan(np.deg2rad(phi))) if phi !=0 else center[1]
            center_z = center[2] - safe_multiplier(d,np.tan(np.deg2rad(theta))) if theta !=0 else center[2] 
        x2y = self.grid[None,:]
        x2z = self.grid[:,None]
        mask = ((x2y-center_y)**2/r_phi**2 + (x2z-center_z)**2/r_theta**2 <=1).type(torch.bool)
        if check:
            if float(mask.sum()) ==0 or (mask[0,:] == True).any() or (mask[-1,:] == True).any() or (mask[:,0] == True).any() or (mask[:, -1] == True).any():
                return torch.zeros_like(mask), True # check_failed
        if cap_start:
            d_cap = (height/2- safe_dividend(d,np.cos(np.deg2rad(theta)),np.cos(np.deg2rad(phi)))) 
            c_theta =  safe_dividend(d_cap, np.sin(np.deg2rad(theta))) # will be negative for cap_small cases
            c_phi =  safe_dividend(d_cap, np.sin(np.deg2rad(phi)))
            #print("r_phi is {r1:.2f} r theta is {r2:.2f}, cap at ({cap_phi:.2f},{cap_theta:.2f})".format(r2 = r_theta, r1 = r_phi,cap_theta = float(cap_theta), cap_phi = float(cap_phi)))
            cap_theta = 0 if theta == 0 else (float(r_theta + c_theta) )  # AS IN PAGE 2.1
            cap_phi =  0 if phi == 0 else (float(r_phi + c_phi) )

            first_index_y = center_y - r_phi 
            last_index_y = center_y + r_phi 
            first_index_z = center_z - r_theta 
            last_index_z = center_z + r_theta 
            #print('f_y:({f_y:.2f},{c_z:.2f}), l_y:({l_y:.2f},{c_z:.2f}), f_z:({c_y:.2f},{f_z:.2f}), l_z:({c_y:.2f},{l_z:.2f})'.format(f_y = first_index_y, f_z = first_index_z, l_y = last_index_y, l_z  =last_index_z, c_y = center_y, c_z = center_z))

            if theta ==0:
                cap_mask = (x2y<first_index_y+cap_phi).type(torch.bool) if direction_right else (x2y<last_index_y-cap_phi).type(torch.bool)
                #print("Ellipse equation: (x-({y:+.2f}))^2/{r_phi:.2f}^2 +(y -({z:+.2f}))^2/{r_theta:.2f}^2<1, line equation: x < {x}".format(r_theta = float(r_theta), r_phi = float(r_phi),  y = float(center_y), z = float(center_z), x = A[0]))
            elif phi ==0:
                if direction_right:
                    line_right = np.linalg.solve(np.array([[first_index_y, 1],[last_index_y,1]]), np.array([first_index_z+cap_theta,first_index_z+cap_theta]))
                else:
                    line_right = np.linalg.solve(np.array([[first_index_y, 1],[last_index_y,1]]), np.array([last_index_z-cap_theta,last_index_z-cap_theta]))
                cap_mask = (x2y *line_right[0]+line_right[1]>x2z).type(torch.bool)
            else:
                if  direction_right:
                    A = (float(first_index_y + cap_phi), center_z)
                    C = (center_y, float(first_index_z+cap_theta))
                    #A = (np.sqrt(r_phi**2 * (1 - (float(first_index_z + cap_theta) - center_z)**2/r_theta**2))+center_y,float(first_index_z + cap_theta) )
                    #C = (float(first_index_y+cap_phi), np.sqrt(r_theta**2*(1 - (float(first_index_y+cap_phi) - center_y)**2/r_phi**2))+center_z)
                else:
                    A = (float(last_index_y - cap_phi), center_z)
                    C = (center_y, float(last_index_z-cap_theta))
                    #A = (np.sqrt(r_phi**2 * (1 - (float(last_index_z - cap_theta) - center_z)**2/r_theta**2))+center_y,float(last_index_z - cap_theta) )
                    #C = (float(last_index_y-cap_phi), np.sqrt(r_theta**2*(1 - (float(last_index_y-cap_phi) - center_y)**2/r_phi**2))+center_z)

                #print('A: ({a0:.2f},{a1:.2f}), C: ({c0:.2f},{c1:.2f})'.format(a0 = A[0], a1  =A[1], c0 = C[0], c1 = C[1])) 
                line_right = np.linalg.solve(np.array([[A[0], 1],[C[0],1]]), np.array([A[1],C[1]]))
                cap_mask = (x2y *line_right[0]+line_right[1]>x2z).type(torch.bool)
                #print("Ellipse equation: (x-({y:+.2f}))^2/{r_phi:.2f}^2 +(y -({z:+.2f}))^2/{r_theta:.2f}^2<1, line equation: y > {a:.2f}x{b:+.2f}".format(r_theta = float(r_theta), r_phi = float(r_phi),  y = float(center_y), z = float(center_z), a = self._line_right[0], b = self._line_right[1]))                
            mask = torch.logical_and(mask, cap_mask) if direction_right else (torch.logical_and(mask, torch.logical_not(cap_mask)))
        return mask, False


    def __generate_cylinder(self, radius, height, center, theta, phi):
        """
        Create a 3D cylinder as 2D slices. The slices are placed at the same distance to each other 
        as the grid inside the slices. Updates  the class attributes `box` and `density`.
        input:
            radius: the radius of cylinder in nm
            height: the hight of the cylinder, height/2 when counting from the center
            center: triple that points to the center of the cylinder
            theta: rotation angle of current cylinder in the direction of z-axis
            phi: rotation angle of current cylinder in the direction of y-axis
        output:
            boolean: True if a cylinder was  placed in a box, otherwise, if constraints were not met returns False
        """
        central_axis_cylinder_projection_on_x = safe_multiplier(height/2,np.cos(np.deg2rad(theta)),np.cos(np.deg2rad(phi)))//self.grid_space # projection of central cylinder axis on x-axis
        cylinder_rest_projection_on_x =  safe_multiplier(np.sin(np.deg2rad(theta)),np.sin(np.deg2rad(phi)),radius)//self.grid_space # projection of the rest of the cylinder after the central axis on x-axis
        cylinder_projection_on_x = math.ceil(central_axis_cylinder_projection_on_x+ cylinder_rest_projection_on_x) # projection of whole cylinder on x-axis
        #print(cylinder_projection_on_x, central_axis_cylinder_projection_on_x, cylinder_rest_projection_on_x)
        radius_at_theta = safe_dividend(radius,np.cos(np.deg2rad(theta))) # calculate the radius of ellipse at slice at both rotations
        radius_at_phi = safe_dividend(radius,np.cos(np.deg2rad(phi)))
        #print('radius is {r:.2f}, r_phi is {r_phi:.2f} and r_theta is {r_theta:.2f}'.format(r = radius, r_phi = radius_at_phi, r_theta = radius_at_theta))
        # if central slice on grid:
        if len(self.grid[self.grid == center[0]])==1:
            central_slice = int(torch.argwhere(self.grid==center[0])[0,0])# start with the central slice
            if central_slice+cylinder_projection_on_x >self._box.shape[0] or central_slice-cylinder_projection_on_x <0:
                #print('--->outside of x plane')
                return False
            # check if circles at the ends of cylinder are inside the box  
            d = self.grid_space*cylinder_projection_on_x
            _, check_1 = self.__create_slice(height, radius_at_theta,radius_at_phi, center,theta, phi, d, cap_start = True, cap_small = True, direction_right = True, check = True)# mask the ellipse location
            _, check_2 = self.__create_slice(height, radius_at_theta,radius_at_phi, center,theta, phi, d, cap_start - True, cap_small = True, direction_right = False, check = True)

            if  check_1 or check_2:
                #print('--->outside on yz-plane')
                return  False
            # all checks done, the latest sphere is not touching the edge of the box or is completely outside of it:  cylinder fits the box, fill the densities by slices
            for i in range(cylinder_projection_on_x): # last grid point fully covering the radius is considered , cylinder is symmetric so work in both directions
                d = self.grid_space*i # calculate the distance grom the center to the slice
                cap_start = i> central_axis_cylinder_projection_on_x-cylinder_rest_projection_on_x 
                cap_small = i> central_axis_cylinder_projection_on_x
                mask,_ =self.__create_slice(height, radius_at_theta,radius_at_phi, center, theta, phi, d, cap_start, cap_small, direction_right = True, check = False)
                self._box[central_slice+i,mask] = 1 # density inside cylinder
                mask,_ = self.__create_slice(height, radius_at_theta,radius_at_phi, center, theta, phi, d,  cap_start, cap_small, direction_right = False, check = False)
                self._box[central_slice-i,mask] = 1

        else:
            # if the center of the cylinder in between of two grid points, find those points and do the same in both dierections
            nearest_bigger_ind = int(torch.argwhere(self.grid>center[0])[0,0])
            if nearest_bigger_ind+cylinder_projection_on_x >self._box.shape[0] or nearest_bigger_ind-1-cylinder_projection_on_x <0:
                #print('--->outside of x plane')
                return  False
            # check if circles at the ends of cylinder are inside the box  
            d1 = self.grid_space*cylinder_projection_on_x + self.grid[nearest_bigger_ind]-center[0]
            d2 = self.grid_space*cylinder_projection_on_x + center[0] - self.grid[nearest_bigger_ind-1]
            _,check_1 = self.__create_slice(height, radius_at_theta,radius_at_phi, center, theta, phi, d1, cap_start = True, cap_small = True, direction_right = True, check = True)
            _,check_2 = self.__create_slice(height, radius_at_theta,radius_at_phi, center, theta, phi, d2, cap_start = True, cap_small = True, direction_right = False, check = True)        
            if  check_1 or check_2:
                #print('--->outside on yz-plane')
                return False
            # all checks done, cylinder fits the box, fill the densities by slices
            for i in range(cylinder_projection_on_x):
                d1 = self.grid_space*i + self.grid[nearest_bigger_ind]-center[0]
                cap_start = i> central_axis_cylinder_projection_on_x-cylinder_rest_projection_on_x 
                cap_small = i> central_axis_cylinder_projection_on_x
                mask,_ = self.__create_slice(height, radius_at_theta,radius_at_phi, center, theta, phi, d1,  cap_start, cap_small, direction_right = True, check = False)
                self._box[nearest_bigger_ind+i,mask] = 1 # density inside cylinder
                d2 = self.grid_space*i + center[0] - self.grid[nearest_bigger_ind-1]
                mask,_ = self.__create_slice(height, radius_at_theta,radius_at_phi, center, theta, phi, d2,  cap_start, cap_small, direction_right = False, check = False)
                self._box[nearest_bigger_ind-1-i,mask] = 1
        return True
        
'''
