import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import rosbag
import math
from scipy.spatial.transform import Rotation as R

def load_frame(fname):
    frame = cv2.imread(fname)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    frame = cv2.resize(frame, (1920, 1080))
    return frame

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParams = cv2.aruco.DetectorParameters_create()

def detect(frame):
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    corners = np.array(corners).reshape((4, 2)).astype(int)
    topLeft, topRight, bottomRight, bottomLeft = corners
    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
    center = np.array([cX, cY])
    return center

def parse_log(fname):
    with open(fname) as f:
        lines = f.readlines()
    data = [eval(l.strip()) for l in lines]
    df = pd.DataFrame(data)
    pos = [df['lt'].mean(), df['ln'].mean(), df['al'].mean()]
    cov = [df['lt'].var(), df['ln'].var(), df['al'].var()]
    pos = np.array(pos)
    cov = np.array(cov) * np.eye(3)
    return torch.from_numpy(pos).float(), torch.from_numpy(cov).float()


class Euler2RotationMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, angles):
        # angles is expected to be a batch of N x 3 (roll, pitch, yaw)
        roll, pitch, yaw = angles[:, 0], angles[:, 1], angles[:, 2]

        # Precompute cosines and sines of the angles
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

        # Construct rotation matrix for roll
        r_mat = torch.stack([
            torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll),
            torch.zeros_like(roll), cos_r, -sin_r,
            torch.zeros_like(roll), sin_r, cos_r
        ], dim=-1).view(-1, 3, 3)

        # Construct rotation matrix for pitch
        p_mat = torch.stack([
            cos_p, torch.zeros_like(pitch), sin_p,
            torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
            -sin_p, torch.zeros_like(pitch), cos_p
        ], dim=-1).view(-1, 3, 3)

        # Construct rotation matrix for yaw
        y_mat = torch.stack([
            cos_y, -sin_y, torch.zeros_like(yaw),
            sin_y, cos_y, torch.zeros_like(yaw),
            torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
        ], dim=-1).view(-1, 3, 3)

        # Combine the rotation matrices
        rotation_matrix = torch.bmm(torch.bmm(y_mat, p_mat), r_mat)

        return rotation_matrix

class Local2Pixel(nn.Module):
    def __init__(self, camera_info):
        super().__init__()
        self.camera_info = camera_info
        # self.ox = nn.Parameter(torch.tensor(-30.0))
        # self.oy = nn.Parameter(torch.tensor(-200.0))
        # self.oz = nn.Parameter(torch.tensor(150.0))
        self.ox = nn.Parameter(torch.tensor(1.0))
        self.oy = nn.Parameter(torch.tensor(1.0))
        self.oz = nn.Parameter(torch.tensor(1.0))


    def forward(self, local_points):
        X = (local_points[:, 1] + self.oy)
        Y = (local_points[:, 2] + self.oz)
        Z = (local_points[:, 0]) + self.ox
        u = (X/Z) * self.camera_info['left']['fx'] + self.camera_info['left']['cx']
        v = (Y/Z) * self.camera_info['left']['fy'] + self.camera_info['left']['cy']
        u = u / 1920
        v = v / 1080
        pixels = torch.stack((u, v), dim=1)
        return pixels

class GPS2Cartesian(nn.Module):
    def __init__(self, rad=6371000):
        super().__init__()
        self.rad = rad
    
    def forward(self, gps_coordinates):
        lat_rad = torch.deg2rad(gps_coordinates[:, 0])
        lon_rad = torch.deg2rad(gps_coordinates[:, 1])
        adjusted_radius = self.rad + gps_coordinates[:, 2]
        x = adjusted_radius * torch.cos(lat_rad) * torch.cos(lon_rad)
        y = adjusted_radius * torch.cos(lat_rad) * torch.sin(lon_rad)
        z = adjusted_radius * torch.sin(lat_rad)
        points = torch.stack((x, y, z), dim=1) 
        # points = points * 1000
        return points


class Global2Local(nn.Module):
    def __init__(self, origin):
        super().__init__()
        self.origin = origin
    
    def forward(self, gps_coordinates):
        return gps_coordinates - self.origin

class NodeCalibrator(nn.Module):
    def __init__(self, node_pos, node_rot, camera_info):
        super().__init__()
        self.gps2cart = GPS2Cartesian()
        self.register_buffer('node_pos', self.gps2cart(node_pos))
        #self.register_buffer('node_rot', node_rot)
        self.angles = nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))
        #self.node_rot = nn.Parameter(node_rot)
        self.camera_info = camera_info
        self.global2local = Global2Local(self.node_pos)
        self.local2pixel = Local2Pixel(self.camera_info)
        self.euler2rot = Euler2RotationMatrix()



    def forward(self, obj_pos):
        node_rot = self.euler2rot(self.angles).squeeze(0)
        obj_pos = self.gps2cart(obj_pos)
        obj_pos_local = (obj_pos - self.node_pos) @ node_rot
        pixels = self.local2pixel(obj_pos_local)
        return pixels

def cartesian2spherical(cart_coords):
    x, y, z = cart_coords
    r = np.sqrt(x**2 + y**2 + z**2)  # Radius
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)  # Polar angle

    # Ensure theta is between 0 and 2π
    #theta = theta % (2 * np.pi)

    theta = np.degrees(theta)
    phi = np.degrees(phi)
    if phi > 90:
        phi = 180 - phi
    return np.array([r, theta, phi])


if __name__ == '__main__':
    img_h, img_w = 1080, 1920
    fov_h, fov_v = 84, 53  # degrees

    n_pos, n_cov = parse_log('Node Positioning GPS Data/R00-node1-left.log')
    rot_df = pd.read_csv('20240108-224709_camera_rotation.csv', header=None)
    rot = R.from_euler('xyz', rot_df.mean())
    n_rot = torch.from_numpy(rot.as_matrix()).float()
    n_pos = n_pos.unsqueeze(0)
    #n_rot = n_rot.unsqueeze(0)

    fnames = ['samples/frame_1800.jpg', 'samples/frame_2800.jpg', 'samples/frame_4440.jpg']
    frames = [load_frame(fname) for fname in fnames]
    pixels = np.array([detect(frame) for frame in frames])
    # gt_pixels = np.array([
        # [958, 671],
        # [1475, 905],
        # [521, 779]
    # ])
    frames_with_pixels = [cv2.circle(frame, tuple(p), 10, (0, 0, 255), -1) for frame, p in zip(frames, gt_pixels)]
    for i, frame in enumerate(frames_with_pixels):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite('result/orig_frame_%d.jpg' % i, frame)
    pixels = torch.from_numpy(gt_pixels).float().cuda()
    pixels[:, 0] = pixels[:, 0] / img_w 
    pixels[:, 1] = pixels[:, 1] / img_h

    gps_points = []
    for i in [1,2,3]:
        o_pos, o_cov = parse_log('Node Positioning GPS Data/R00-node1-pos%d.log' % i)
        gps_points.append(o_pos)
    gps_points = torch.stack(gps_points, dim=0).cuda()
    print(gps_points.shape)

    with open('camera_info.json') as f:
        camera_info = json.load(f)
    

    model = NodeCalibrator(n_pos, n_rot, camera_info).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


    for i in range(1000):
        optimizer.zero_grad()
        pixels_pred = model(gps_points)
        loss = torch.mean((pixels_pred - pixels)**2)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if i % 100 == 0:
            print(loss.item())

    pixels_pred = pixels_pred.detach().cpu().numpy()
    pixel_preds = pixels_pred * np.array([img_w, img_h])
    for i, frame, pixel in zip([1,2,3], frames, pixel_preds):
        new_frame = frame.copy()
        gt_pixel = gt_pixels[i-1]
        pixel = (int(pixel[0]), int(pixel[1]))
        gt_pixel = (gt_pixel[0], gt_pixel[1])
        new_frame = cv2.circle(new_frame, tuple(pixel), 10, (0, 255, 0), -1)
        new_frame = cv2.circle(new_frame, tuple(gt_pixel), 10, (255, 0, 0), -1)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite('result/frame_%d.png' % i, new_frame)
