import sys
import ezdxf
import numpy as np
from shapely.geometry import Point, Polygon


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