import sys
import ezdxf
import numpy as np
from shapely.geometry import Point, Polygon
import torch 

def dxf_to_shape(path):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    
    # Find the first LWPOLYLINE
    for e in msp:
        if e.dxftype() == "LWPOLYLINE":
            points = [tuple(vertex[:2]) for vertex in e.get_points()]  # take only x, y
            polygon = Polygon(points)
            return polygon
    
    raise ValueError("No LWPOLYLINE entities found in DXF file")

def shape_to_points(shape: Polygon):

    # exterior coords of polygon
    points = list(shape.exterior.coords)
    return points

def parse_triangles(path):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    triangles = []

    for e in msp:
        if e.dxftype() == "LWPOLYLINE":
            points = [tuple(vertex[:2]) for vertex in e.get_points()]

            # Check if it is a triangle (3 unique vertices)
            if len(points) == 3:
                triangles.append(points)

            # Sometimes closed triangles repeat first vertex at end
            elif len(points) == 4 and points[0] == points[-1]:
                triangles.append(points[:3])

    if len(triangles) == 0:
        raise ValueError("No triangle LWPOLYLINE entities found in DXF file")

    # Convert to PyTorch tensor (k x 3 x 2)
    triangle_tensor = torch.tensor(triangles, dtype=torch.float32)

    return triangle_tensor