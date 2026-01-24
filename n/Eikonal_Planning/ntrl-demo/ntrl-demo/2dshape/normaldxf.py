import sys
import ezdxf
import numpy as np
import ezdxf
import matplotlib.pyplot as plt
import numpy as np

import ezdxf
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from matplotlib.patches import Patch





def visualize_shapes_or_files(items, random_color, speed, vmin):


    print("random color: " + str(random_color))
    
    fig, ax = plt.subplots()
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'cyan']

    cnt = 0

    for i, item in enumerate(items):
        color = colors[i % len(colors)]
        cmap = plt.get_cmap("viridis")
        if isinstance(item, str) and item.lower().endswith(".dxf"):
            # Handle DXF file
            try:
                doc = ezdxf.readfile(item)
            except Exception as e:
                print(f"Failed to read {item}: {e}")
                continue

            msp = doc.modelspace()
            all_x, all_y = [], []

            for e in msp:
                etype = e.dxftype()

                if etype == "LINE":
                    x = [e.dxf.start.x, e.dxf.end.x]
                    y = [e.dxf.start.y, e.dxf.end.y]
                    ax.plot(x, y, color='black')
                    all_x.extend(x)
                    all_y.extend(y)

                elif etype == "LWPOLYLINE":
                    points = np.array([[p[0], p[1]] for p in e.get_points()])
                    if e.closed:
                        points = np.vstack([points, points[0]])
                    ax.plot(points[:,0], points[:,1], color='black')
                    all_x.extend(points[:,0])
                    all_y.extend(points[:,1])

                elif etype == "POLYLINE":
                    points = np.array([[v.dxf.location.x, v.dxf.location.y] for v in e.vertices()])
                    if e.is_closed:
                        points = np.vstack([points, points[0]])
                    ax.plot(points[:,0], points[:,1], color='black')
                    all_x.extend(points[:,0])
                    all_y.extend(points[:,1])

                elif etype == "CIRCLE":
                    center = e.dxf.center
                    radius = e.dxf.radius
                    circle = plt.Circle((center.x, center.y), radius, fill=False, color='black')
                    ax.add_patch(circle)
                    all_x.append(center.x)
                    all_y.append(center.y)

                elif etype == "ARC":
                    center = e.dxf.center
                    radius = e.dxf.radius
                    start_angle = e.dxf.start_angle
                    end_angle = e.dxf.end_angle
                    angles = np.radians(np.linspace(start_angle, end_angle, 100))
                    x = center.x + radius * np.cos(angles)
                    y = center.y + radius * np.sin(angles)
                    ax.plot(x, y, color='black')
                    all_x.extend(x)
                    all_y.extend(y)

        elif isinstance(item, Polygon):
            # Handle Shapely Polygon
            x, y = item.exterior.xy
            ax.plot(x, y, color=color)

        else:
            print(f"Skipping unsupported item: {item}")

    ax.set_aspect('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Shapes / DXFs Visualization")

    plt.show()

def scale_doc(doc, scale, xoffset, yoffset, name):
    # Create a new DXF for the normalized shape
    doc_norm = ezdxf.new(dxfversion='R2010')  
    msp_norm = doc_norm.modelspace() 

    msp = doc.modelspace()
    for e in msp:
        if e.dxftype() == "LINE":
            start_point = e.dxf.start
            end_point = e.dxf.end

            # Apply offset and scaling
            scaled_start_x = (start_point.x - xoffset) * scale
            scaled_start_y = (start_point.y - yoffset) * scale
            scaled_end_x   = (end_point.x - xoffset) * scale
            scaled_end_y   = (end_point.y - yoffset) * scale

            # Add line to new DXF
            msp_norm.add_line(
                start=(scaled_start_x, scaled_start_y, start_point.z),
                end=(scaled_end_x, scaled_end_y, end_point.z)
            )

        elif e.dxftype() == "LWPOLYLINE":
            # Get points of the polyline
            points = e.get_points()  # returns list of tuples (x, y, optional attributes)
            scaled_points = []
            for point in points:
                x, y = point[0], point[1]
                z = 0  # LWPOLYLINE has no Z by default
                scaled_points.append((
                    (x - xoffset) * scale,
                    (y - yoffset) * scale,
                    z
                ))
            # Preserve closed property if set
            closed = e.closed
            msp_norm.add_lwpolyline(
                [(p[0], p[1]) for p in scaled_points],
                close=closed
            )

    # Save the normalized DXF
    doc_norm.saveas(name)


if __name__ == "__main__":
    doc = ezdxf.readfile("Fmaze.dxf")
    msp = doc.modelspace()



    minX = 1000000
    maxX = -1000000
    minY = 1000000
    maxY = -1000000

    msp = doc.modelspace()
    for e in msp:
        if e.dxftype() == "LINE":
            minX = min(minX, e.dxf.start.x)
            maxX = max(maxX, e.dxf.start.x)
            minY = min(minY, e.dxf.start.y)
            maxY = max(maxY, e.dxf.start.y)

    print(minX)
    print(maxX)
    print(minY)
    print(maxY)


    xorigin = (minX+maxX)/2
    yorigin = (minY+maxY)/2

    scale = max(maxY-minY, maxX-minX)

    scale_doc(doc=doc,scale=1/scale,xoffset=xorigin,yoffset=yorigin,name="Fmaze_norm.dxf")




    doc = ezdxf.readfile("Fshape.dxf")
    msp = doc.modelspace()


    shape_xorigin =  1000000
    shape_yorigin =  1000000
    for e in msp:
        print(e.dxftype()) 
        if e.dxftype() == "LINE":
            print("Line start:", e.dxf.start, "end:", e.dxf.end)
            shape_xorigin = min(shape_xorigin, e.dxf.start.x, e.dxf.end.x)
            shape_yorigin = min(shape_yorigin, e.dxf.start.y, e.dxf.end.y)
        elif e.dxftype() == "LWPOLYLINE":
            for point in e.get_points():
                x, y = point[0], point[1]
                shape_xorigin = min(shape_xorigin, x)
                shape_yorigin = min(shape_yorigin, y)



    print("fshape:" + str(shape_xorigin) + "," + str(shape_yorigin))
    scale_doc(doc=doc,scale=1/scale,xoffset=shape_xorigin,yoffset=shape_yorigin,name="Fshape_norm.dxf")

    visualize_shapes_or_files(["Fshape_norm.dxf", "Fmaze_norm.dxf"])
