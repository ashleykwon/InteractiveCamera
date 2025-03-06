import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output
import numpy as np
import random
import sympy as sp
import math

# Initialize the Dash app
app = dash.Dash(__name__)


# Apply zoom in/out in 3D
# def zoom(Zminimum, Zmaximum, dZ_slope, zoomScaleFactor, XZpairs):
#     newXZpairs = []
#     # Divide x values in XZpairs by scaleFactor if Z is in the modification range
#     for i in range(len(XZpairs)):
#         x, z = XZpairs[i]
#         if z >= Zminimum and z <= Zmaximum:
#             if x >= -1*dZ_slope*z and x <= dZ_slope*z:
#                 newXZpairs.append([x/zoomScaleFactor, z])
#             else:
#                 newXZpairs.append([x,z])
#         else:
#             newXZpairs.append([x,z])
#     return newXZpairs


# # Increase or decrease foreshortening
# def foreshortening(Zminimum, Zmaximum, foreshorteningFactor, dZ_slope, XZpairs):
#     newXZpairs = []

#     # Check if the foreshortening factor was modified
#     if foreshorteningFactor != 0:
#         # Sort XZ pairs based on their z values
#         XZpairs = sorted(XZpairs, key= lambda x:x[1])

#         # Make a dictionary with z-values as keys and all x-values with that z-value in a list
#         XvalForEachZ = dict()
#         for i in range(len(XZpairs)):
#             x, z = XZpairs[i]
#             if z not in XvalForEachZ:
#                 XvalForEachZ[z] = []
#             XvalForEachZ[z].append(x)
#         XvalForEachZ = dict(sorted(XvalForEachZ.items()))
        
#         # Iterate through each z value and get the smallest z value and its x values that are within the Z value range
#         initialZ = Zminimum
#         initialXvals = []
#         for zVal in XvalForEachZ:
#             if zVal >= Zminimum:
#                 # initialZ = zVal
#                 initialXvals = XvalForEachZ[zVal]

#         # Iterate through the dictionary again and modify x values within the Z value range
#         for zVal in XvalForEachZ:
#             currentXvals = XvalForEachZ[zVal]
#             if zVal >= Zminimum and zVal <= Zmaximum:
#                 for k in range(len(currentXvals)):
#                     if currentXvals[k] >= -1*dZ_slope*zVal and currentXvals[k] <= dZ_slope*zVal:
#                         closestInitialXval = min(initialXvals, key=lambda x: abs(x - currentXvals[k]))
#                         # Transforms x values within the positive x boundary
#                         if currentXvals[k] <= 0:
#                             newX = -1*(dZ_slope + foreshorteningFactor)*(zVal - initialZ) + closestInitialXval
#                             newXZpairs.append([newX, zVal])
#                         # Transforms x values within the negative x boundary
#                         else:
#                             newX = (dZ_slope + foreshorteningFactor)*(zVal - initialZ) + closestInitialXval
#                             newXZpairs.append([newX, zVal])
#                     else:
#                         newXZpairs.append([currentXvals[k], zVal])
#             else:
#                 for k in range(len(currentXvals)):
#                     newXZpairs.append([currentXvals[k], zVal])

#     else:
#         newXZpairs = XZpairs
#     return newXZpairs


# Generate initial dot coordinates (arranged as a rectangle between two lines)
def generate_dots():
    # Create a 2D grid of dots
    x_vals = np.linspace(-5, 5, 6) 
    z_vals = np.linspace(3, 6, 5)
    X, Z = np.meshgrid(x_vals, z_vals)
    
    # Flatten grid
    X_flat = X.flatten()
    Z_flat = Z.flatten()

    return X_flat, Z_flat


# def apply_transformation_3d(Xcoords, Zcoords, zoomScaleFactor, foreshorteningSlope, dZ_slope, section_range):
#     Zminimum, Zmaximum = section_range
#     shape = Xcoords.shape

#     # Apply zoom or foreshortening to points that are in the selected range
#     newXZpairs = zoom(Zminimum, Zmaximum, dZ_slope, zoomScaleFactor, [[X, Z] for X, Z in zip(Xcoords, Zcoords)])
#     newXZpairs = foreshortening(Zminimum, Zmaximum, foreshorteningSlope, dZ_slope, newXZpairs)

#     # Reshape dot coordinate arrays for visualization
#     x_vals, z_vals = zip(*newXZpairs)
#     x_vals = np.array(x_vals)
#     z_vals = np.array(z_vals)
#     X_flat = x_vals.reshape(shape)
#     Z_flat = z_vals.reshape(shape)
#     return X_flat, Z_flat


def apply_transformation_uv(Zminimum, Zmaximum, depthRangeValues, dZ_slope, zoomScaleFactor, foreshorteningFactor, XZpairs):
    # Sort XZ pairs based on their z values
    XZpairs = sorted(XZpairs, key= lambda x:x[1])

    # Define selected minimum and maximum z range values
    selected_Zminimum, selected_Zmaximum = depthRangeValues

    # Make a dictionary with z-values as keys and all x-values with that z-value in a list
    XvalForEachZ = dict()
    for i in range(len(XZpairs)):
        x, z = XZpairs[i]
        if z not in XvalForEachZ:
            XvalForEachZ[z] = []
        XvalForEachZ[z].append(x)
    XvalForEachZ = dict(sorted(XvalForEachZ.items(), reverse=True))
    
    # Define a new slope with foreshorteningFactor
    newSlope = dZ_slope + foreshorteningFactor
    if newSlope == 0:
        newSlope = 0.01

    # Iterate through the dictionary and derive u coordinates
    u_coordinates = []
    u_coordinates_to_visualize = np.zeros((len(XvalForEachZ), len(XvalForEachZ[next(iter(XvalForEachZ))])), dtype=bool) # Boolean mask to decide which u coorinate to render
    for j in range(len(XvalForEachZ)):
        zVal = list(XvalForEachZ.keys())[j]
        currentXvals = sorted(XvalForEachZ[zVal])
        if zVal >= Zminimum and zVal <= Zmaximum:
            # Apply zoom and foreshortening to coordinates within the selected depth range
            if zVal >= selected_Zminimum and zVal <= selected_Zmaximum:
                # Apply foreshortening and zoom and then apply transformation from world coordinates to uv (v is 0 by default, so it's not considered here)
                u_coords_for_currentXs = [x/((newSlope/zoomScaleFactor)*zVal) for x in currentXvals]
                u_coordinates += u_coords_for_currentXs
                # Keep track of coordinates to render
                if newSlope > 0: # Positive slope
                    chosen_coords_for_currentXs = []
                    for k in range(len(currentXvals)):
                        x = currentXvals[k]
                        if x > 0:
                            chosen_coords_for_currentXs.append(bool(x <= (newSlope/zoomScaleFactor)*zVal))
                        else:
                            chosen_coords_for_currentXs.append(bool(x >= (-newSlope/zoomScaleFactor)*zVal)) # Need to flip the direction of the slope for negative x values
                    u_coordinates_to_visualize[j] = chosen_coords_for_currentXs
                elif newSlope < 0: # Negative slope
                    chosen_coords_for_currentXs = []
                    for k in range(len(currentXvals)):
                        x = currentXvals[k]
                        if x > 0:
                            xVal_at_selected_Zminimum = dZ_slope*selected_Zminimum
                            # chosen_coords_for_currentXs.append(bool(x <= (newSlope/zoomScaleFactor)*zVal)) # TODO: Wut?? Why would this work when (newSlope/zoomScaleFactor)*zVal is always negative?
                            chosen_coords_for_currentXs.append(bool(x <= (zVal-selected_Zminimum)*(newSlope/zoomScaleFactor) + xVal_at_selected_Zminimum)) 
                        else:
                            xVal_at_selected_Zminimum = -dZ_slope*selected_Zminimum
                            # chosen_coords_for_currentXs.append(bool(x >= (-newSlope/zoomScaleFactor)*zVal)) # Need to flip the direction of the slope for negative x values
                            chosen_coords_for_currentXs.append(bool(x >= (zVal-selected_Zminimum)*(-newSlope/zoomScaleFactor) + xVal_at_selected_Zminimum)) 
                    u_coordinates_to_visualize[j] = chosen_coords_for_currentXs
                # TODO What to do at 0 slope??
            # Don't apply transformations to values that are outside of the selected depth range, but are between the near and far planes. These coordinates are still rendered
            else:
                # Apply transformation from world coordinates to uv (v is 0 by default, so it's not considered here)
                u_coords_for_currentXs = [x/(dZ_slope*zVal) for x in currentXvals]
                u_coordinates += u_coords_for_currentXs
                # Keep track of coordinates to render
                chosen_coords_for_currentXs = [bool(abs(x) <= abs(dZ_slope)*zVal) for x in currentXvals]
                u_coordinates_to_visualize[j] = chosen_coords_for_currentXs
        else:
            u_coordinates +=  [x/(dZ_slope*zVal) for x in currentXvals]
    # print(u_coordinates_to_visualize)
    u_coordinates_to_visualize = u_coordinates_to_visualize.flatten() # This mask is correct
    return u_coordinates, u_coordinates_to_visualize


# def find_camera_params_after_foreshortening(old_half_img_width, old_focal_length, selected_Zminimum, nonminimum_Z, dZ_slope, zoomScaleFactor, foreshorteningFactor):
#     newSlope = dZ_slope + foreshorteningFactor
#     new_focal_length = sp.symbols('x')
#     nonminimum_Z = float(nonminimum_Z)

#     new_focal_length, new_half_img_width = sp.symbols('x y')
#     equation1 = new_half_img_width - abs(newSlope)*new_focal_length
#     equation2 = ((new_half_img_width/(new_focal_length*zoomScaleFactor))*nonminimum_Z - (new_half_img_width/(new_focal_length*zoomScaleFactor))*selected_Zminimum)/(nonminimum_Z - selected_Zminimum) - abs(newSlope)
#     # equation3 = new_focal_length*math.atan(old_half_img_width/old_focal_length) - new_half_img_width
#     # print(math.atan(old_half_img_width/old_focal_length))
#     # inequality1 = new_focal_length > old_focal_length
#     # inequality2 = new_half_img_width > old_half_img_width
#     # equation3 = new_half_img_width - (new_focal_length*old_half_img_width)/old_focal_length
#     solution = sp.solve((equation1, equation2), (new_focal_length, new_half_img_width)) #TODO Is still underconstrained
#     print(solution)
#     new_focal_length = solution[list(solution.keys())[0]]
#     new_half_img_width = solution[list(solution.keys())[1]]
   
#     return new_focal_length, new_half_img_width


def uv_to_3d(u_coordinates, u_coordinates_to_visualize, Zminimum, Zmaximum, depthRangeValues, bZ_slope, half_img_width, focal_length, zoomScaleFactor, foreshorteningFactor, XZpairs):
    # Sort XZ pairs based on their z values and then extract sorted z values
    XZpairs = sorted(XZpairs, key= lambda x:x[1])
    zVals = [pair[1] for pair in XZpairs]

    # print("UV to 3D called")
    # Make a dictionary with z-values as keys and all x-values with that z-value in a list
    XvalForEachZ = dict()
    for i in range(len(XZpairs)):
        x, z = XZpairs[i]
        if z not in XvalForEachZ:
            XvalForEachZ[z] = []
        XvalForEachZ[z].append(x)
    XvalForEachZ = dict(sorted(XvalForEachZ.items(), reverse=True))

    # Define selected minimum and maximum z range values
    selected_Zminimum, selected_Zmaximum = depthRangeValues

    # Define a new slope with foreshorteningFactor
    newSlope = bZ_slope + foreshorteningFactor

    new_xVals = []

    # Foreshortening  
    # Initial lower and upperbound b(z) values
    lowerbound_bZ = -1
    upperbound_bZ = 1
    bound_distance = 2
    stepsize = 1
    # Iterate through z values
    for z in list(XvalForEachZ.keys()):
    # Get the distance bound_distance between the x coordinates of the lower and upper bounds of the new b(z)
        # Iterate through x values with the same z value
        Xs_at_Z = XvalForEachZ[z]
        if z >= selected_Zminimum and z <= selected_Zmaximum:
            # print(z)
            xVal_at_selected_Zminimum = bZ_slope*selected_Zminimum
            for i in range(len(Xs_at_Z)):
                x = Xs_at_Z[i]
                if newSlope >= 0: # Positive or zero slope
                    lowerbound_b = -xVal_at_selected_Zminimum - (newSlope/zoomScaleFactor)*selected_Zminimum
                    upperbound_b = xVal_at_selected_Zminimum - (newSlope/zoomScaleFactor)*selected_Zminimum
                    upperbound_bZ = z*(newSlope/zoomScaleFactor) + upperbound_b
                    lowerbound_bZ = -upperbound_bZ
                else: # Negative slope
                    lowerbound_b = -xVal_at_selected_Zminimum - (newSlope/zoomScaleFactor)*selected_Zminimum
                    upperbound_b = xVal_at_selected_Zminimum - (newSlope/zoomScaleFactor)*selected_Zminimum
                    upperbound_bZ = -z*(newSlope/zoomScaleFactor) + upperbound_b
                    lowerbound_bZ = -upperbound_bZ
                bound_distance = abs(upperbound_bZ) + abs(lowerbound_bZ)
                stepsize = bound_distance/(len(Xs_at_Z)-1)
                
                if x >= lowerbound_bZ and x <= upperbound_bZ:
                    print("lowerbound")
                    print(lowerbound_bZ)
                    print("upperbound")
                    print(upperbound_bZ)
                    remapped_x = lowerbound_bZ + stepsize*(i+1)
                    new_xVals.append(remapped_x)
                else:
                    new_xVals.append(x)
        else:
            new_xVals += Xs_at_Z
    new_xVals.reverse()
    return np.asarray(new_xVals), np.asarray(zVals)



# Create the layout of the app 
app.layout = html.Div([
    html.H1("Interactive ZoomShop"),

    # Near plane slider
    html.Div([
        html.Label("Near plane z value"),
        dcc.Slider(
            id='nearPlane-slider',
            min=1,
            max=10,
            step=0.01,
            value=2,
            marks={i: f'{i}' for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
    ], style={'padding': '10px'}),


    # Far plane slider
    html.Div([
        html.Label("Far plane z value"),
        dcc.Slider(
            id='farPlane-slider',
            min=1,
            max=10,
            step=0.01,
            value=5.5,
            marks={i: f'{i}' for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
    ], style={'padding': '10px'}),

    
    # Selected depth range double slider
    html.Div([
        html.Label("Selected depth range:"),
        dcc.RangeSlider(
            id='depthRange-slider',
            min=1,
            max=10,
            step=0.01,
            marks={i: f'{i}' for i in range(0, 11)},
            value=[3.5, 5],  # Initial selected range
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
    ], style={'padding': '10px'}),

    # Focal length slider
    html.Div([
        html.Label("Focal length"),
        dcc.Slider(
            id='focalLength-slider',
            min=1,
            max=10,
            step=0.01,
            value=1,
            marks={i: f'{i}' for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
    ], style={'padding': '10px'}),

    # Image plane width / 2 slider
    html.Div([
        html.Label("Image plane width / 2"),
        dcc.Slider(
            id='ImgPlaneWidth-slider',
            min=1,
            max=10,
            step=0.01,
            value=1,
            marks={i: f'{i}' for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
    ], style={'padding': '10px'}),

    # Zoom scale factor slider
    html.Div([
        html.Label("Zoom scale factor (Factor > 1 for zoom in):"),
        dcc.Slider(
            id='zoom-slider',
            min=0,
            max=7,
            step=0.01,
            value=1,
            marks={i: f'{i}' for i in range(0, 7, 1)},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
    ], style={'padding': '10px'}),

    # Foreshortening factor slider
    html.Div([
        html.Label("Foreshortening factor (Smaller value to reduce foreshortening and make background objects appear closer):"),
        dcc.Slider(
            id='foreshortening-slider',
            min=-2,
            max=5,
            step=0.01,
            value=5, 
            marks={i: f'{i}' for i in range(-6, 6, 1)},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
    ], style={'padding': '10px'}),

    # Graph to display the plot
    dcc.Graph(id='line-intersection-graph'),

    dcc.Graph(id='uv-plot')
])

# Define the callback to update the graph based on the sliders
@app.callback(
    Output('line-intersection-graph', 'figure'),
    [
        Input('zoom-slider', 'value'),
        Input('foreshortening-slider', 'value'),
        Input('nearPlane-slider', 'value'), 
        Input('farPlane-slider', 'value'),
        Input('depthRange-slider', 'value'),
        Input('focalLength-slider', 'value'), 
        Input('ImgPlaneWidth-slider', 'value')
       
    ]
)

# Update the camera model with world coordinates
def update_camera_model(zoomScaleFactor, foreshorteningFactor, nearPlaneZValue, farPlaneZvalue, depthRangeValues, focalLength, ImgPlaneWidth):
    z_vals = np.linspace(0, farPlaneZvalue, 100)

    slope = ImgPlaneWidth/focalLength

    # Generate initial world coordinates
    initial_X, initial_Z = generate_dots()

    # Get the selected range from the range slider
    Zminimum, Zmaximum = depthRangeValues
    
    # Apply transformation to world coordinates
    # X, Z = apply_transformation_3d(initial_X, initial_Z, zoomScaleFactor, foreshorteningFactor, slope, depthRangeValues)
    u_coordinates, u_coordinates_to_visualize = apply_transformation_uv(nearPlaneZValue, farPlaneZvalue, depthRangeValues, slope, zoomScaleFactor, foreshorteningFactor, [[x, z] for x, z in zip(initial_X, initial_Z)])
    X, Z = uv_to_3d(u_coordinates, u_coordinates_to_visualize, Zminimum, Zmaximum, depthRangeValues, slope, focalLength, ImgPlaneWidth, zoomScaleFactor, foreshorteningFactor, [[x, z] for x, z in zip(initial_X, initial_Z)])
    
    # Calculate the x-values for both lines 
    x1_vals = slope * z_vals  
    x2_vals = -slope * z_vals

    x_intersect1 = nearPlaneZValue / slope  # Intersection with the first line (upward slope)
    x_intersect2 = -nearPlaneZValue / slope

    # Coodinates to draw near and far planes
    # Far plane
    vertical_line_z =[z_vals[-1], z_vals[-1]]  
    vertical_line_x = [x1_vals[-1], x2_vals[-1]]
    # Near plane
    nearPlanel_line_z =[nearPlaneZValue, nearPlaneZValue]  
    nearPlanel_line_x = [x_intersect1,x_intersect2]

    # Select the part of the first line between the selected range
    selected_z_vals = z_vals[(z_vals >= Zminimum) & (z_vals <= Zmaximum)]
    selected_x1_vals = slope * selected_z_vals
    selected_x2_vals = -slope * selected_z_vals  # Mirrored points on the second line

    # Section of the line with modified slope for foreshortening
    z_section = z_vals[(z_vals >= Zminimum) & (z_vals <= Zmaximum)]
    x_section = (slope+foreshorteningFactor) * (z_section - Zminimum) + slope * Zminimum

    # # For the second line, apply the opposite slope in the selected section
    x_section_second = -(slope+foreshorteningFactor) * (z_section - Zminimum) - slope * Zminimum  # Adjust for starting y-value
    
    # Create a color scale function to render coordinate colors
    # This will map each x value to a color from the colorscale
    color_scale = px.colors.sequential.Rainbow  # Get the color scale as a list
    norm_z = [(val - min(initial_Z)) / (max(initial_Z) - min(initial_Z)) for val in initial_Z]  # Normalize x values to [0, 1]
    color_list = [color_scale[int(val * (len(color_scale) - 1))] for val in norm_z]

    # Define the figure data
    figure = {
        'data': [
            # Solid line along b(z) 
            go.Scatter(x=z_vals, y=x1_vals, mode='lines', name=f'b(z) (Slope={slope:.2f})', line=dict(color='orange', width=3)),

            # Solid line along -b(z) 
            go.Scatter(x=z_vals, y=x2_vals, mode='lines', name=f'-b(z) (Slope={-slope:.2f})', line=dict(color='orange', width=3)),
            
            # Dotted line at far plane
            go.Scatter(x=vertical_line_z, y=vertical_line_x, mode='lines', name="Far plane",
                       line=dict(dash='dot', width=2, color='red')),
            
            # Dotted line at near plane 
            go.Scatter(x=nearPlanel_line_z, y=nearPlanel_line_x, mode='lines', name="Near plane",
                        line=dict(dash='dot', width=2, color='green')),  

            # Highlight the selected range along Line 1 (highlight area)
            go.Scatter(x=selected_z_vals, y=selected_x1_vals, mode='lines', fill='tozeroy', fillcolor='rgba(0, 100, 255, 0.3)', name="Selected depth range for x bound by b(z)"),
            
            # Highlight the mirrored selected range on Line 2 (highlight area)
            go.Scatter(x=selected_z_vals, y=selected_x2_vals, mode='lines', fill='tozeroy', fillcolor='rgba(0, 100, 255, 0.3)', name="Selected depth range for x bound by -b(z)"),

            # Foreshortening slope on b(z)
            go.Scatter(x=z_section, y=x_section, mode='lines', name="Foreshortening Slope for b(z)", line=dict(color='blue', width=3)),

            # Foreshortening slope on -b(z)
            go.Scatter(x=z_section, y=x_section_second, mode='lines', name="Foreshortening Slope for -b(z)", line=dict(color='blue', width=3)),

            # World coordinates
            go.Scatter(x=Z, y=X, mode='markers', name="World coordinates", marker=dict(size=10, color=color_list))

        ],
        'layout': go.Layout(
            title="Camera Model",
            xaxis={
                'title': 'Z-axis',  # Adjusted to represent horizontal axis
                'range': [0, 10],  # Set x-axis range
                'showgrid': True,  # Show gridlines
                'zeroline': True,  # Show the zero line
            },
            yaxis={
                'title': 'X-axis',  # Adjusted to represent vertical axis
                'range': [-10, 10],  # Set y-axis range
                'showgrid': True,  # Show gridlines
                'zeroline': True,  # Show the zero line
            },
            showlegend=True
        )
    }
    
    return figure


# Define the callback to update the graph based on the sliders
@app.callback(
    Output('uv-plot', 'figure'),
    [
        Input('zoom-slider', 'value'),
        Input('foreshortening-slider', 'value'),
        Input('nearPlane-slider', 'value'),
        Input('farPlane-slider', 'value'), 
        Input('depthRange-slider', 'value'),
        Input('focalLength-slider', 'value'), 
        Input('ImgPlaneWidth-slider', 'value')
       
    ]
)


def update_uv_plot(zoomScaleFactor, foreshorteningFactor, nearPlaneZValue, farPlaneZvalue, depthRangeValues, focalLength, ImgPlaneWidth):
    # Generate initial world coordinates
    initial_X, initial_Z = generate_dots()
    slope = ImgPlaneWidth/focalLength
    
    # Create a color scale function for uv coordinates
    color_scale = px.colors.sequential.Rainbow  # Get the color scale as a list
    norm_z = [(val - min(initial_Z)) / (max(initial_Z) - min(initial_Z)) for val in initial_Z]  # Normalize x values to [0, 1]
    color_list = [color_scale[int(val * (len(color_scale) - 1))] for val in norm_z]
    color_list.reverse()

    # Derive u and v coordinates after transformation
    u_coordinates, u_coordinates_to_visualize = apply_transformation_uv(nearPlaneZValue, farPlaneZvalue, depthRangeValues, slope, zoomScaleFactor, foreshorteningFactor, [[x, z] for x, z in zip(initial_X, initial_Z)])
    u_coordinates_selected = [u_coordinates[i] for i in range(len(u_coordinates)) if u_coordinates_to_visualize[i]]
        
    # u for u in u_coordinates if u >= -1 and u <= 1 and u_coordinates_to_visualize[i]]
    v_coordinates = [0]*len(u_coordinates_selected) # set to 0 
    
    # Get colors of chosen coordinates (visible from the camera)
    masked_color_list = [color for color, mask in zip(color_list, u_coordinates_to_visualize) if mask]
    
    figure = {
        'data': [
            # Plot the uv coordinates
            go.Scatter(x=u_coordinates_selected, y=v_coordinates, mode='markers', name="U coordinate", marker=dict(size=30, color=masked_color_list))

        ],
        'layout': go.Layout(
            title="Rendered image",
            xaxis={
                'title': 'U', 
                'range': [-1.05, 1.05],  # Set U-axis range
                'showgrid': True,  # Show gridlines
                'zeroline': True,  # Show the zero line
            },
            yaxis={
                'title': 'V',  
                'range': [-1, 1],  # Set V-axis range
                'showgrid': True,  # Show gridlines
                'zeroline': True,  # Show the zero line
            },
            showlegend=True
        )
    }

    return figure


#Run# Keep track of modified coordinate the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
