import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output
import numpy as np
import random

# Initialize the Dash app
app = dash.Dash(__name__)


# Apply zoom in/out in 3D
def zoom(Zminimum, Zmaximum, dZ_slope, zoomScaleFactor, XZpairs):
    newXZpairs = []
    # Divide x values in XZpairs by scaleFactor if Z is in the modification range
    for i in range(len(XZpairs)):
        x, z = XZpairs[i]
        if z >= Zminimum and z <= Zmaximum:
            if x >= -1*dZ_slope*z and x <= dZ_slope*z:
                newXZpairs.append([x/zoomScaleFactor, z])
            else:
                newXZpairs.append([x,z])
        else:
            newXZpairs.append([x,z])
    return newXZpairs


# Increase or decrease foreshortening
def foreshortening(Zminimum, Zmaximum, foreshorteningFactor, dZ_slope, XZpairs):
    newXZpairs = []

    # Check if the foreshortening factor was modified
    if foreshorteningFactor != 0:
        # Sort XZ pairs based on their z values
        XZpairs = sorted(XZpairs, key= lambda x:x[1])

        # Make a dictionary with z-values as keys and all x-values with that z-value in a list
        XvalForEachZ = dict()
        for i in range(len(XZpairs)):
            x, z = XZpairs[i]
            if z not in XvalForEachZ:
                XvalForEachZ[z] = []
            XvalForEachZ[z].append(x)
        XvalForEachZ = dict(sorted(XvalForEachZ.items()))
        
        # Iterate through each z value and get the smallest z value and its x values that are within the Z value range
        initialZ = Zminimum
        initialXvals = []
        for zVal in XvalForEachZ:
            if zVal >= Zminimum:
                # initialZ = zVal
                initialXvals = XvalForEachZ[zVal]

        # Iterate through the dictionary again and modify x values within the Z value range
        for zVal in XvalForEachZ:
            currentXvals = XvalForEachZ[zVal]
            if zVal >= Zminimum and zVal <= Zmaximum:
                for k in range(len(currentXvals)):
                    if currentXvals[k] >= -1*dZ_slope*zVal and currentXvals[k] <= dZ_slope*zVal:
                        closestInitialXval = min(initialXvals, key=lambda x: abs(x - currentXvals[k]))
                        # Transforms x values within the positive x boundary
                        if currentXvals[k] <= 0:
                            newX = -1*(dZ_slope + foreshorteningFactor)*(zVal - initialZ) + closestInitialXval
                            newXZpairs.append([newX, zVal])
                        # Transforms x values within the negative x boundary
                        else:
                            newX = (dZ_slope + foreshorteningFactor)*(zVal - initialZ) + closestInitialXval
                            newXZpairs.append([newX, zVal])
                    else:
                        newXZpairs.append([currentXvals[k], zVal])
            else:
                for k in range(len(currentXvals)):
                    newXZpairs.append([currentXvals[k], zVal])

    else:
        newXZpairs = XZpairs
    return newXZpairs


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


def apply_transformation_3d(Xcoords, Zcoords, zoomScaleFactor, foreshorteningSlope, dZ_slope, section_range):
    Zminimum, Zmaximum = section_range
    shape = Xcoords.shape

    # Apply zoom or foreshortening to points that are in the selected range
    newXZpairs = zoom(Zminimum, Zmaximum, dZ_slope, zoomScaleFactor, [[X, Z] for X, Z in zip(Xcoords, Zcoords)])
    newXZpairs = foreshortening(Zminimum, Zmaximum, foreshorteningSlope, dZ_slope, newXZpairs)

    # Reshape dot coordinate arrays for visualization
    x_vals, z_vals = zip(*newXZpairs)
    x_vals = np.array(x_vals)
    z_vals = np.array(z_vals)
    X_flat = x_vals.reshape(shape)
    Z_flat = z_vals.reshape(shape)
    return X_flat, Z_flat


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
    chosen_coordinates = np.zeros((len(XvalForEachZ), len(XvalForEachZ[next(iter(XvalForEachZ))])), dtype=bool)

    # Define a new slope with the foreshorteningFactor
    newSlope = dZ_slope + foreshorteningFactor

    # Iterate through the dictionary and append u coordinates in descending depth order to render shorter depths at the front
    u_coordinates = []
    for j in range(len(XvalForEachZ)):
        zVal = list(XvalForEachZ.keys())[j]
        if zVal >= Zminimum and zVal <= Zmaximum:
            currentXvals = sorted(XvalForEachZ[zVal], reverse=True)
            # Apply zoom and foreshortening to coordinates within the selected depth range
            if zVal >= selected_Zminimum and zVal <= selected_Zmaximum:
                # Apply foreshortening and zoom and then apply transformation to uv
                u_coords_for_currentXs = [x/(newSlope*zVal*zoomScaleFactor) for x in currentXvals if abs(x) <= abs(newSlope)*zVal*zoomScaleFactor]
                u_coordinates += u_coords_for_currentXs
                    # Keep track of coordinates to render
                chosen_coords_for_currentXs = [bool(abs(x) <= abs(newSlope)*zVal*zoomScaleFactor) for x in currentXvals]
                chosen_coordinates[j] = chosen_coords_for_currentXs
            # Don't apply transformations to values that are outside of the selected depth range, but are between the near and far planes
            else:
                # Apply transformation to uv
                u_coords_for_currentXs = [x/(dZ_slope*zVal) for x in currentXvals if abs(x) <= abs(dZ_slope)*zVal]
                u_coordinates += u_coords_for_currentXs
                # Keep track of coordinates to render
                chosen_coords_for_currentXs = [bool(abs(x) <= abs(dZ_slope)*zVal) for x in currentXvals]
                chosen_coordinates[j] = chosen_coords_for_currentXs
    chosen_coordinates = chosen_coordinates.flatten()
    return u_coordinates, chosen_coordinates



# Create the layout of the app with one slider for slope
app.layout = html.Div([
    html.H1("Interactive ZoomShop"),

    # Slider to adjust the length of both lines
    html.Div([
        html.Label("Near plane z value"),
        dcc.Slider(
            id='nearPlane-slider',
            min=1,
            max=10,
            step=0.1,
            value=2,
            marks={i: f'{i}' for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '10px'}),


    # Slider to adjust the length of both lines
    html.Div([
        html.Label("Far plane z value"),
        dcc.Slider(
            id='farPlane-slider',
            min=1,
            max=10,
            step=0.1,
            value=6,
            marks={i: f'{i}' for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '10px'}),

    
    # Range slider to select a range along the solid line
    html.Div([
        html.Label("Selected depth range:"),
        dcc.RangeSlider(
            id='depthRange-slider',
            min=1,
            max=10,
            step=0.1,
            marks={i: f'{i}' for i in range(0, 11)},
            value=[3, 5],  # Initial selected range
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '10px'}),

    # Slider to adjust the length of both lines
    html.Div([
        html.Label("Focal length"),
        dcc.Slider(
            id='focalLength-slider',
            min=1,
            max=10,
            step=0.1,
            value=1,
            marks={i: f'{i}' for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '10px'}),

    # Slider to adjust the length of both lines
    html.Div([
        html.Label("Image plane width / 2"),
        dcc.Slider(
            id='ImgPlaneWidth-slider',
            min=1,
            max=10,
            step=0.1,
            value=1,
            marks={i: f'{i}' for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '10px'}),

    # Slider to adjust the slope for both lines (m1 and m2)
    html.Div([
        html.Label("Zoom scale factor (Factor > 1 for zoom in):"),
        dcc.Slider(
            id='zoom-slider',
            min=0,
            max=7,
            step=0.1,
            value=1,
            marks={i: f'{i}' for i in range(0, 7, 1)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '10px'}),

    # Slider to adjust the slope for both lines (m1 and m2)
    html.Div([
        html.Label("Foreshortening factor (Lower slope to reduce foreshortening):"),
        dcc.Slider(
            id='foreshortening-slider',
            min=-2,
            max=5,
            step=0.1,
            value=0,
            marks={i: f'{i}' for i in range(-2, 6, 1)},
            tooltip={"placement": "bottom", "always_visible": True}
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
        Input('farPlane-slider', 'value'), 
        Input('nearPlane-slider', 'value'),
        Input('depthRange-slider', 'value'),
        Input('focalLength-slider', 'value'), 
        Input('ImgPlaneWidth-slider', 'value')
       
    ]
)


def update_camera_model(zoomScaleFactor, foreshorteningFactor, line_length, nearPlaneZValue, depthRangeValues, focalLength, ImgPlaneWidth):
    # Calculate the x-values for both lines (from 0 to line_length)
    z_vals = np.linspace(0, line_length, 100)

    slope = ImgPlaneWidth/focalLength

    # Generate initial dots' coordinates
    # X, Y, xDot_vals, yDot_vals_1, yDot_vals_2 = generate_dots(slope, foreshorteningSlope, depthRangeValues)
    initial_X, initial_Z = generate_dots()
    
    # Apply transformation when button is clicked (e.g., shift the dots)
    X, Z = apply_transformation_3d(initial_X, initial_Z, zoomScaleFactor, foreshorteningFactor, slope, depthRangeValues)
    
    # Calculate the y-values for both lines based on the slopes (swap x and y)
    x1_vals = slope * z_vals  # The line with slope 'm'
    x2_vals = -slope * z_vals  # The line with slope '-m'

    x_intersect1 = nearPlaneZValue / slope  # Intersection with the first line (upward slope)
    x_intersect2 = -nearPlaneZValue / slope

    vertical_line_z =[z_vals[-1], z_vals[-1]]  # x = 0 y1_vals[-1], y2_vals[-1]
    vertical_line_x = [x1_vals[-1], x2_vals[-1]]

    nearPlanel_line_z =[nearPlaneZValue, nearPlaneZValue]  # x = 0 y1_vals[-1], y2_vals[-1]
    nearPlanel_line_x = [x_intersect1,x_intersect2]

    # Get the selected range from the range slider
    Zminimum, Zmaximum = depthRangeValues

    # Select the part of the first line between the selected range
    selected_z_vals = z_vals[(z_vals >= Zminimum) & (z_vals <= Zmaximum)]
    selected_x1_vals = slope * selected_z_vals
    selected_x2_vals = -slope * selected_z_vals  # Mirrored points on the second line

    # # Section of the line with the custom slope
    z_section = z_vals[(z_vals >= Zminimum) & (z_vals <= Zmaximum)]
    x_section = (slope+foreshorteningFactor) * (z_section - Zminimum) + slope * Zminimum

    # # For the second line, apply the opposite slope in the selected section
    x_section_second = -(slope+foreshorteningFactor) * (z_section - Zminimum) - slope * Zminimum  # Adjust for starting y-value
    

    # Create a color scale function
    # This will map each x value to a color from the colorscale
    color_scale = px.colors.sequential.Rainbow  # Get the color scale as a list
    norm_x = [(val - min(initial_Z)) / (max(initial_Z) - min(initial_Z)) for val in initial_Z]  # Normalize x values to [0, 1]
    color_list = [color_scale[int(val * (len(color_scale) - 1))] for val in norm_x]

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

            # Highlight the section with the different slope
            go.Scatter(x=z_section, y=x_section, mode='lines', name="Foreshortening Slope for b(z)", line=dict(color='blue', width=3)),

            # Highlight the mirrored section for the second line
            go.Scatter(x=z_section, y=x_section_second, mode='lines', name="Foreshortening Slope for -b(z)", line=dict(color='blue', width=3)),

            # Plot the dots
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
    # Generate initial dots
    initial_X, initial_Z = generate_dots()
    slope = ImgPlaneWidth/focalLength
    
    # Create a color scale function
    # This will map each x value to a color from the colorscale
    color_scale = px.colors.sequential.Rainbow  # Get the color scale as a list
    norm_x = [(val - min(initial_Z)) / (max(initial_Z) - min(initial_Z)) for val in initial_Z]  # Normalize x values to [0, 1]
    color_list = [color_scale[int(val * (len(color_scale) - 1))] for val in norm_x]
    color_list.reverse()


    # Derive u and v coordinates after transformation
    u_coordinates, chosen_coordinates = apply_transformation_uv(nearPlaneZValue, farPlaneZvalue, depthRangeValues, slope, zoomScaleFactor, foreshorteningFactor, [[x, z] for x, z in zip(initial_X, initial_Z)])
    v_coordinates = [0]*len(u_coordinates) # set to 0 
    
    # Get colors of chosen coordinates (visible from the camera)
    masked_color_list = [color for color, mask in zip(color_list, chosen_coordinates) if mask]
    # if 
    
    figure = {
        'data': [
            # Plot the dots
            go.Scatter(x=u_coordinates, y=v_coordinates, mode='markers', name="U coordinate", marker=dict(size=30, color=masked_color_list))

        ],
        'layout': go.Layout(
            title="Rendered image",
            xaxis={
                'title': 'U', # Adjusted to represent horizontal axis
                'range': [-1.5, 1.5],  # Set x-axis range
                'showgrid': True,  # Show gridlines
                'zeroline': True,  # Show the zero line
            },
            yaxis={
                'title': 'V',  # Adjusted to represent vertical axis
                'range': [-1, 1],  # Set y-axis range
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
