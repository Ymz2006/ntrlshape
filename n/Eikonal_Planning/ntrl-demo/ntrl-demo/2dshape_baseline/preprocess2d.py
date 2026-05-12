import sys
import ezdxf
import numpy as np
from shapely.geometry import Point, Polygon
from parse_shape import dxf_to_shape, shape_to_points
from scipy.spatial import cKDTree
from normaldxf import visualize_shapes_or_files 
import torch 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import get_cmap
from shapely.geometry import Polygon
from descartes import PolygonPatch  # For plotting shapely polygons

from matplotlib.colors import Normalize

import time



# from torch_kdtree import build_kd_tree
# import torch
# from scipy.spatial import KDTree #Reference implementation
# import numpy as np


def rotate_points(points, x, y, theta):


    rot = torch.tensor([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta),  np.cos(theta), y],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    shape_points = torch.tensor(points, dtype=torch.float32)  # n x 2
    new_col = torch.ones((shape_points.shape[0], 1))
    shape_points = torch.cat([shape_points, new_col], dim=1)  # n x 3

    transformed = (rot @ shape_points.T).T  # n x 3

    # Return only x, y as list of tuples for Shapely
    return [tuple(p[:2].numpy()) for p in transformed]

def get_bounding_radius(shape):
    points = np.array(shape.exterior.coords)
    centroid = shape.centroid

    dists = np.linalg.norm(points - np.array([centroid.x, centroid.y]), axis=1)
    radius = np.max(dists)

    return np.array([centroid.x, centroid.y]),radius

#shape_points: nx2
def get_obstacle_dist(shape_points, x_add, y_add, theta, centroid, radius, sampled_points, kdtree):

    rot = torch.tensor([
        [np.cos(theta), -np.sin(theta), x_add],
        [np.sin(theta),  np.cos(theta), y_add],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Convert shape_points to torch if needed

    shape_points = torch.tensor(shape_points, dtype=torch.float32)
    new_col = torch.ones(shape_points.shape[0],1)
    shape_points = torch.cat([shape_points,new_col],1)
    # Apply transformation
    translated_points = ((rot @ shape_points.T).T )[:,0:2] # n x 3

    avg = translated_points.mean(axis=0)  # shape (3,)


    # KDTree query
    candidate_indices = kdtree.query_ball_point(avg, r=radius) 
    candidate_points = sampled_points[candidate_indices]

    # Make polygon
    rotated_shape = Polygon(translated_points[:, :2].numpy())  # only x, y

    collision_detected = False
    mindis = float('inf')
    collided_points = []

    for i, pt in enumerate(candidate_points):
        curr_point = Point(pt[0], pt[1])
        if rotated_shape.contains(curr_point):
            collision_detected = True
            collided_points.append(i)
        mindis = min(mindis, rotated_shape.distance(curr_point))

    if collision_detected:
        return collision_detected, collided_points

    return collided_points, mindis


dots_per_m = 80


def sample_points(msp):

    points = []

    for e in msp:
        t = e.dxftype()
        
        if t == "LINE":
            start = np.array([e.dxf.start.x, e.dxf.start.y])
            end   = np.array([e.dxf.end.x, e.dxf.end.y])
            length = np.linalg.norm(end - start)
            dot_cnt = max(int(length * dots_per_m), 1)
            for i in range(dot_cnt):
                new_point = start + (end - start) * (i / dot_cnt)
                points.append(new_point)
        
        elif t in ["LWPOLYLINE", "POLYLINE"]:
            verts = np.array([[v[0], v[1]] for v in e.get_points()])  # Nx2 array
            # If closed, connect last to first
            closed = e.closed if hasattr(e, 'closed') else False
            for i in range(len(verts)-1):
                start, end = verts[i], verts[i+1]
                length = np.linalg.norm(end - start)
                dot_cnt = max(int(length * dots_per_m), 1)
                for j in range(dot_cnt):
                    new_point = start + (end - start) * (j / dot_cnt)
                    points.append(new_point)
            if closed:
                start, end = verts[-1], verts[0]
                length = np.linalg.norm(end - start)
                dot_cnt = max(int(length * dots_per_m), 1)
                for j in range(dot_cnt):
                    new_point = start + (end - start) * (j / dot_cnt)
                    points.append(new_point)
        
        elif t == "CIRCLE":
            center = np.array([e.dxf.center.x, e.dxf.center.y])
            radius = e.dxf.radius
            num_points = max(int(2*np.pi*radius*dots_per_m), 4)
            for i in range(num_points):
                angle = 2*np.pi*i/num_points
                new_point = center + radius*np.array([np.cos(angle), np.sin(angle)])
                points.append(new_point)
        
        # TODO: add more entity types if needed

    return np.array(points)



def generate_valid_points(number_points, shape_points, msp):

    shape = Polygon(shape_points)
    environment_boundary_points = sample_points(msp=msp)

    bottom_left = np.min(environment_boundary_points, axis=0)
    upper_right = np.max(environment_boundary_points, axis=0)
    xrange = upper_right[0] - bottom_left[0] 
    yrange = upper_right[1] - bottom_left[1] 
    scale = max(xrange,yrange)


    t = torch.rand(number_points*10, 3) - 0.5  # shape: (N, 3)
    row_scale = torch.tensor([xrange/scale - 0.02, yrange/scale - 0.02, np.pi/0.5], dtype=torch.float32)  # shape: (3,), 0.8 HARD CODED
    t = t * row_scale  # element-wise scaling of columns

    centroid,radius = get_bounding_radius(shape=shape)

    kdtree = cKDTree(environment_boundary_points)


    cnt = 0
    valid_indicies = []
    closest_dist = []
    for i in range (0, number_points*10):
        intersect, out = get_obstacle_dist(shape_points=shape_points,x_add=t[i,0],y_add=t[i,1],theta=t[i,2],centroid=centroid,
                             radius=radius,sampled_points=environment_boundary_points,kdtree=kdtree)
        
        if not intersect and out > 0:       # change maybe 
            valid_indicies.append(i)
            closest_dist.append(out)
            cnt +=1
            if cnt%10000 == 0:
                print("sampled points: " + str(cnt) )

            if (cnt >= number_points):
                break
    valid_points = t[valid_indicies]

    valid_points = valid_points[:,0:3]

    return valid_points,closest_dist, environment_boundary_points


def generate_training_data(number_points, shape_points, msp, dmin, dmax):
    start_valid_points, start_closest_dist, env_points = generate_valid_points(number_points, shape_points, msp)

    speed  = np.zeros((start_valid_points.shape[0]))

    start_valid_points = np.array(start_valid_points)
    start_closest_dist = np.array(start_closest_dist)


    speed = np.clip(start_closest_dist/dmax , a_min = dmin/dmax, a_max = 1)
    print("number of points:" + str(start_valid_points.shape[0]))
    return start_valid_points,start_closest_dist,speed, env_points


def visual_training(start, shape_points, env_points, cnt, speed, vmin, vmax=None):
    to_visual_shapes = []
    if vmax is None:
        vmax = max(speed)

    cmap = get_cmap('viridis')
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()

    for i in range(cnt):
        # Rotate points and return as list of (x, y) tuples
        rotated_pts = rotate_points(shape_points, start[i][0], start[i][1], start[i][2])
        
        # Skip if not enough points to make a polygon
        if len(rotated_pts) < 3:
            continue
        
        rotated_shape = Polygon(rotated_pts)
        if not rotated_shape.is_valid:
            continue

        to_visual_shapes.append(rotated_shape)

        # Map speed to color
        color = cmap(norm(speed[i]))
        
        # Use matplotlib Polygon to draw the shape
        patch = plt.Polygon(list(rotated_shape.exterior.coords), facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(patch)

    # Plot environment points
    if len(env_points) > 0:
        ax.scatter(*zip(*env_points), color='grey', s=10)

    ax.set_aspect('equal')



    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Speed", rotation=270, labelpad=15)
    plt.show()





def visual_speed(start, speed, env_points, min):
    plt.figure(figsize=(6,6))
    plt.scatter(env_points[:,0], env_points[:,1], c='black')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Velocity')
    plt.axis('equal')  # equal scaling for x and y
    plt.legend()

    plt.scatter(start[:,0], start[:,1],    s=50, c= speed[:,0], cmap="viridis", vmin=min, vmax=1)
    plt.colorbar(label="s value")
    plt.show()



if __name__ == "__main__":
    Fshape_norm = dxf_to_shape("./datasets/Fshape_norm.dxf")
    Fshape_points = shape_to_points(Fshape_norm)
    dmax = 0.2
    dmin = 0.01


    doc = ezdxf.readfile("./datasets/FmazeEasy_norm.dxf")
    msp = doc.modelspace()



    start_time = time.time()
    start, start_dist, speed, env_points = generate_training_data((int)(4e5),Fshape_points,msp,dmin,dmax)
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


    print(type(env_points))
    visual_training(start,Fshape_points, env_points=env_points, cnt=800,speed=speed,vmin=dmin/dmax)
    #visual_speed(start, speed,env_points, dmin/dmax)

    len = (int)(start.shape[0]/2)

    x0 = start[0:len,:]
    x1 = start[len:2*len,:]
    x = np.concatenate((x0,x1), axis=1)


    len= (int)(speed.shape[0]/2)

    y0 = speed[0:len]
    y1 = speed[len:2*len]
    y = np.column_stack((y0,y1))


    out_path = "./training_data2d/Fshape_FmazeEasy_02"

    print(x.shape)
    print(y.shape)
    np.save('{}/sampled_points'.format(out_path),x)
    np.save('{}/speed'.format(out_path),y)

    np.save('{}/Easyenv'.format(out_path), env_points)
    print(env_points.shape)