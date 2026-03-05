import sys
import ezdxf
import numpy as np
from shapely.geometry import Point, Polygon
from parse_shape import dxf_to_shape, shape_to_points, parse_triangles
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


dots_per_m = 100


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


def calculate_speed(shape_lines, env_points):
    AB = shape_lines[:,:,1:] - shape_lines[:,:,0,:]
    
    env_points_aug = env_points.clone()
    env_points_aug = env_points_aug.unsqueeze(0)

    AP = env_points_aug - shape_lines[:,:,0,:]

    cross = torch.cross(AB,AP)
    cross_mag = torch.linalg.norm(cross, ord=2, dim=2)
    AB_norm = torch.linalg.norm(AB, ord=2, dim=2)

    dis = cross_mag/AB_norm
    result = dis.min(dim=1).values

    return result

# shape tensor points: k x 3 x 2

def generate_valid_points(number_points, shape_tensor_points, msp):


    triangle_cnt = shape_tensor_points.shape[0]
    
    shape_tensor_points = torch.cat([shape_tensor_points, torch.ones(triangle_cnt, 3, 1)], 2)



    environment_boundary_points = sample_points(msp=msp)



    bottom_left = np.min(environment_boundary_points, axis=0)
    upper_right = np.max(environment_boundary_points, axis=0)
    xrange = upper_right[0] - bottom_left[0] 
    yrange = upper_right[1] - bottom_left[1] 
    scale = max(xrange,yrange)




    environment_boundary_points = torch.tensor(environment_boundary_points)
    env_point_cnt = environment_boundary_points.shape[0]

    valid_points = torch.zeros(number_points, 3)
    batch_size = (int)(1e4)
    count = 0


    while True:
        t = torch.rand(batch_size, 3) - 0.5  # shape: (N, 3)
        row_scale = torch.tensor([xrange/scale - 0.02, yrange/scale - 0.02, np.pi/0.5], dtype=torch.float32)  # shape: (3,), 0.8 HARD CODED
        t = t * row_scale  # element-wise scaling of columns



        tx = t[:,0]
        ty = t[:,1]
        cos = torch.cos(t[:,2])
        sin = torch.sin(t[:,2])
        Transform_matrix = torch.zeros(batch_size, 3, 3, device="cuda")
        Transform_matrix[:, 0, 0] = cos
        Transform_matrix[:, 0, 1] = -sin
        Transform_matrix[:, 1, 0] = sin
        Transform_matrix[:, 1, 1] = cos
        Transform_matrix[:, 0, 2] = tx
        Transform_matrix[:, 1, 2] = ty
        Transform_matrix[:, 2, 2] = 1.0



        # transform_mat has dim (n,1,1,3,3)
        Transform_matrix = Transform_matrix.unsqueeze(1)
        Transform_matrix = Transform_matrix.unsqueeze(1)



        # shape_tensor_points has dim (k,3,3,1)

        shape_tensor_points_aug = shape_tensor_points.clone()
        shape_tensor_points_aug = shape_tensor_points_aug.unsqueeze(0)
        shape_tensor_points_aug = shape_tensor_points_aug.unsqueeze(-1)
        

        Transform_matrix = Transform_matrix.cuda()
        shape_tensor_points_aug = shape_tensor_points_aug.cuda()


        #print(Transform_matrix.shape)
        #print(shape_tensor_points_aug.shape)


        transformed_shape_points = torch.matmul(Transform_matrix, shape_tensor_points_aug)
        transformed_shape_points2d = transformed_shape_points[:,:,:,0:2,:]
        

        # re format for cross

        # shape_tensor_points has dim (1,n,k,3,2,1)
        # env has dim (n,1,1,1, 2,1)


        env_points_subtract = environment_boundary_points.clone()
        env_points_subtract = env_points_subtract.unsqueeze(-1)
        env_points_subtract = env_points_subtract.unsqueeze(0)
        env_points_subtract = env_points_subtract.unsqueeze(2)
        env_points_subtract = env_points_subtract.unsqueeze(2)

        transformed_shape_points2d = transformed_shape_points2d.unsqueeze(1)
        

        #print("========")
        #print(env_points_subtract.shape)
        #print(transformed_shape_points2d.shape)



        env_points_subtract = env_points_subtract.cuda()
        transformed_shape_points2d = transformed_shape_points2d.cuda()


        center_to_vertex = env_points_subtract - transformed_shape_points2d 

        
        vertex_to_vertex = transformed_shape_points2d.clone()
        vertex_to_vertex[:,:,:,0,:,:] = transformed_shape_points2d[:,:,:,1,:,:] - transformed_shape_points2d[:,:,:,0,:,:]  # edge v0→v1
        vertex_to_vertex[:,:,:,1,:,:] = transformed_shape_points2d[:,:,:,2,:,:] - transformed_shape_points2d[:,:,:,1,:,:]  # edge v1→v2
        vertex_to_vertex[:,:,:,2,:,:] = transformed_shape_points2d[:,:,:,0,:,:] - transformed_shape_points2d[:,:,:,2,:,:]  # edge v2→v0


        # still need ab, bc, ac 


        # det_mat = torch.zeros(batch_size, env_point_cnt, triangle_cnt, 3, 2, 2)

        # det_mat[:,:,:,:,0,0] = vertex_to_vertex[:,:,:,:,0,0]
        # det_mat[:,:,:,:,1,0] = vertex_to_vertex[:,:,:,:,1,0]
        # det_mat[:,:,:,:,0,1] = center_to_vertex[:,:,:,:,0,0]
        # det_mat[:,:,:,:,1,1] = center_to_vertex[:,:,:,:,1,0]

        a = vertex_to_vertex[..., 0, 0]
        b = center_to_vertex[..., 0, 0]
        c = vertex_to_vertex[..., 1, 0]
        d = center_to_vertex[..., 1, 0]

        cross_mat = a*d - b*c

        # cross_mat = torch.linalg.det(det_mat)
        # print(cross_mat.shape)


        cross_mat = torch.sign(cross_mat)


        cross_result = torch.sum(cross_mat, dim=3)
        cross_result = torch.where((abs(cross_result) == 3), 1, 0)
        
        cross_result = torch.sum(cross_result, dim=2)
        cross_result = torch.sum(cross_result, dim=1)
        
        
        
        shape_intersec = torch.where(cross_result ==0 , True,False)
        curr_valid = shape_intersec.sum()


        shape_intersec = shape_intersec.cpu()

        #print(curr_valid)
        if (count + curr_valid >= number_points):
            valid_points[count:,] = (t[shape_intersec])[:number_points-count]
            break
        else:
            valid_points[count: count + curr_valid] = t[shape_intersec]
            count += shape_intersec.sum()

    
    return valid_points, environment_boundary_points


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
        #vmax = max(speed)
        vmax = 1
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
        #olor = cmap(norm(speed[i]))
        color = cmap(norm(1))

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
    doc = ezdxf.readfile("./datasets/Fmaze3_norm.dxf")
    msp = doc.modelspace()

    Fshape_norm = dxf_to_shape("./datasets/Fshape_norm.dxf")
    Fshape_points = shape_to_points(Fshape_norm)

    Fshape_triangulated = [
        [[0.086,0.171], [0.086, 0.157], [0.014, 0.157]],
        [[0.086,0.171], [0.014, 0.157], [0, 0.171]],

        [[0, 0.171], [0.014, 0.157], [0, 0]],
        [[0.014, 0], [0.014, 0.086], [0, 0]],

        [[0.014, 0.1], [0.071, 0.1], [0.014, 0.086]],
        [[0.014, 0.086], [0.071, 0.1], [0.071, 0.086]]

    ]

    Fshape_triangulated = torch.Tensor(Fshape_triangulated)
    start = time.time()
    valid_points, env = generate_valid_points(400000,Fshape_triangulated,msp)
    end = time.time()
    elapsed = end - start
    print("Elapsed time:", elapsed, "seconds")
    visual_training(valid_points,Fshape_points, env_points=env, cnt=1000,speed=1,vmin=0)



    # start_time = time.time()
    # start, start_dist, speed, env_points = generate_training_data((int)(1e2),Fshape_points,msp,dmin,dmax)
    # end_time = time.time()

    # # Calculate elapsed time
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.2f} seconds")


    # print(type(env_points))

    # #visual_speed(start, speed,env_points, dmin/dmax)

    # len = (int)(start.shape[0]/2)

    # x0 = start[0:len,:]
    # x1 = start[len:2*len,:]
    # x = np.concatenate((x0,x1), axis=1)


    # len= (int)(speed.shape[0]/2)

    # y0 = speed[0:len]
    # y1 = speed[len:2*len]
    # y = np.column_stack((y0,y1))


    # out_path = "./training_data2d/Fshape_Fmaze3"

    # print(x.shape)
    # print(y.shape)
    # # np.save('{}/sampled_points'.format(out_path),x)
    # # np.save('{}/speed'.format(out_path),speed, y)

    # np.save('{}/Fmaze3env'.format(out_path), env_points)
    # print(env_points.shape)