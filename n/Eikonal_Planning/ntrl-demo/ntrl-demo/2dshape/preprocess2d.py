import sys
import ezdxf
import numpy as np
from shapely.geometry import Point, Polygon
from parse_shape import dxf_to_shape, shape_to_points
from scipy.spatial import cKDTree
from normaldxf import visualize_shapes_or_files 
import torch 
import matplotlib.pyplot as plt

def rotate_points(points,x,y,theta):
    rot = torch.tensor([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta),  np.cos(theta), y],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Convert shape_points to torch if needed

    shape_points = torch.tensor(points, dtype=torch.float32)
    new_col = torch.ones(shape_points.shape[0],1)
    shape_points = torch.cat([shape_points,new_col],1)
    # Apply transformation
    translated_points = (rot @ shape_points.T).T  # n x 3
    return translated_points

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
    translated_points = (rot @ shape_points.T).T  # n x 3


    centroid[0] += x_add
    centroid[1] += y_add
    # KDTree query
    # candidate_indices = kdtree.query_ball_point(centroid, r=radius) 
    #print ("candidate_points_len" + str(len(candidate_indices)))
    candidate_points = sampled_points#[candidate_indices]

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
    row_scale = torch.tensor([xrange/scale, yrange/scale, np.pi/0.5], dtype=torch.float32)  # shape: (3,), 0.8 HARD CODED
    t = t * row_scale  # element-wise scaling of columns

    centroid,radius = get_bounding_radius(shape=shape)
    print("centroid, radius" + str(centroid) + str(radius))




    plt.figure(figsize=(6,6))
    plt.scatter(environment_boundary_points[:,0], environment_boundary_points[:,1], c='blue', label='Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Environment Points')
    plt.axis('equal')  # equal scaling for x and y
    plt.legend()
    plt.show()
    kdtree = cKDTree(environment_boundary_points)
    print("kdtree size:" + str(kdtree.n))

    valid_indicies = []
    closest_dist = []
    for i in range (0, number_points*10):
        intersect, out = get_obstacle_dist(shape_points=shape_points,x_add=t[i,0],y_add=t[i,1],theta=t[i,2],centroid=centroid,
                             radius=radius,sampled_points=environment_boundary_points,kdtree=kdtree)
        
        if not intersect:
            valid_indicies.append(i)
            closest_dist.append(out)
    valid_points = t[valid_indicies]

    valid_points = valid_points[0:number_points,0:3]
    closest_dist = closest_dist[0:number_points]
    
    return valid_points,closest_dist


def generate_training_data(number_points, shape_points, msp, dmin, dmax):
    start_valid_points, start_closest_dist = generate_valid_points(number_points, shape_points, msp)
    end_valid_points, end_closest_dist = generate_valid_points(number_points, shape_points, msp)

    speed  = np.zeros((len(start_valid_points),2))

    start_valid_points = np.array(start_valid_points)
    end_valid_points = np.array(end_valid_points)
    start_closest_dist = np.array(start_closest_dist)
    end_closest_dist = np.array(end_closest_dist)


    speed[:,0] = np.clip(start_closest_dist/dmax , a_min = dmin/dmax, a_max = 1)
    speed[:,1] = np.clip(end_closest_dist/dmax , a_min = dmin/dmax, a_max = 1)


    return start_valid_points,end_valid_points,start_closest_dist,end_closest_dist,speed


def visual_training(start,shape_points, cnt):
    to_visual = ["Fmaze_norm.dxf"]
    for i in range (cnt):
        rotated_shape = Polygon(rotate_points(shape_points, start[i][0], start[i][1], start[i][2]))
        to_visual.append(rotated_shape)

    visualize_shapes_or_files(to_visual)



Fshape_norm = dxf_to_shape("Fshape_norm.dxf")
Fshape_points = shape_to_points(Fshape_norm)
dmax = 0.08
dmin = 0.04/0.08


doc = ezdxf.readfile("Fmaze_norm.dxf")
msp = doc.modelspace()





start, end, start_dist, end_dist, speed = generate_training_data(20,Fshape_points,msp,dmin,dmax)


start_test = start[0]
end_test = end[0]

print(start_test)

visual_training(start,Fshape_points, 10)

