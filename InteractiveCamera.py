import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import numpy as np
import random

# Initialize the Dash app
app = dash.Dash(__name__)


# Function to generate random color
def random_color():
    return f'rgba({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)}, 0.7)'

# Apply zoom in/out
def zoom(Zminimum, Zmaximum, dZ_slope, zoomScaleFactor, XZpairs):
    newXZpairs = []
    # Divide x values in XZpairs by scaleFactor if Z is in the modification range
    for i in range(len(XZpairs)):
        x, z = XZpairs[i]
        if z >= Zminimum and z <= Zmaximum:
            if x >= -1*dZ_slope*z and x <= dZ_slope*z:
                newXZpairs.append([x*zoomScaleFactor, z])
            else:
                newXZpairs.append([x,z])
        else:
            newXZpairs.append([x,z])
    return newXZpairs


# Increase or decrease foreshortening
def foreshortening(Zminimum, Zmaximum, foreshorteningSlope, dZ_slope, XZpairs):
    newXZpairs = []

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
    initialZ = 1000
    initialXvals = []
    for zVal in XvalForEachZ:
        if zVal >= Zminimum and zVal < initialZ:
            initialZ = zVal
            initialXvals = XvalForEachZ[zVal]
    
    # Iterate through the dictionary again and modify x values within the Z value range
    for zVal in XvalForEachZ:
        currentXvals = sorted(XvalForEachZ[zVal])
        if foreshorteningSlope != dZ_slope:
            if zVal >= Zminimum and zVal <= Zmaximum:
                for k in range(len(currentXvals)):
                    if currentXvals[k] >= -1*dZ_slope*zVal and currentXvals[k] <= dZ_slope*zVal:
                        closestInitialXval = min(initialXvals, key=lambda x: abs(x - currentXvals[k]))
                        # Transforms x values within the positive x boundary
                        if currentXvals[k] <= 0:
                            newX = -1*foreshorteningSlope*(zVal - initialZ) + closestInitialXval
                            newXZpairs.append([newX, zVal])
                        # Transforms x values within the negative x boundary
                        else:
                            newX = foreshorteningSlope*(zVal - initialZ) + closestInitialXval
                            newXZpairs.append([newX, zVal])
                    else:
                        newXZpairs.append([currentXvals[k], zVal])
            else:
                for k in range(len(currentXvals)):
                    newXZpairs.append([currentXvals[k], zVal])
        else:
            for k in range(len(currentXvals)):
                newXZpairs.append([currentXvals[k], zVal])

    return newXZpairs


# Generate initial dot coordinates (arranged as a rectangle between two lines)
def generate_dots():
    # Create a 2D grid of dots
    x_vals = np.linspace(3, 6, 5)  # 5 columns of dots
    y_vals = np.linspace(-5, 5, 6)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Flatten grid
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    return X_flat, Y_flat


def apply_transformation(Xcoords, Zcoords, zoomScaleFactor, foreshorteningSlope, dZ_slope, section_range):
    Zminimum, Zmaximum = section_range
    shape = Xcoords.shape

    # Apply zoom or foreshortening to points that are in the selected range
    newXZpairs = zoom(Zminimum, Zmaximum, dZ_slope, zoomScaleFactor, [[X, Z] for X, Z in zip(Zcoords, Xcoords)])
    newXZpairs = foreshortening(Zminimum, Zmaximum, foreshorteningSlope, dZ_slope, newXZpairs)

    # Reshape dot coordinate arrays for visualization
    x_vals, z_vals = zip(*newXZpairs)
    x_vals = np.array(x_vals)
    z_vals = np.array(z_vals)
    X_flat = x_vals.reshape(shape)
    Z_flat = z_vals.reshape(shape)
    return Z_flat, X_flat



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
        html.Label("Foreshortening slope (Lower slope to reduce foreshortening):"),
        dcc.Slider(
            id='foreshortening-slider',
            min=-2,
            max=5,
            step=0.1,
            value=1,
            marks={i: f'{i}' for i in range(-2, 6, 1)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'padding': '10px'}),

     # Button to apply transformation to the dots
    # html.Div([
    #     html.Button('Apply Transformation', id='apply-button', n_clicks=0)
    # ], style={'padding': '10px'}),


    # Graph to display the plot
    dcc.Graph(id='line-intersection-graph')
])

# Define the callback to update the graph based on the sliders
@app.callback(
    Output('line-intersection-graph', 'figure'),
    [
        Input('zoom-slider', 'value'),
        Input('foreshortening-slider', 'value'),
        # Input('apply-button', 'n_clicks'),
        Input('farPlane-slider', 'value'), 
        Input('nearPlane-slider', 'value'),
        Input('depthRange-slider', 'value'),
        Input('focalLength-slider', 'value'), 
        Input('ImgPlaneWidth-slider', 'value')
       
    ]
)




def update_graph(zoomScaleFactor, foreshorteningSlope, line_length, nearPlaneZValue, depthRangeValues, focalLength, ImgPlaneWidth):
    # Calculate the x-values for both lines (from 0 to line_length)
    x_vals = np.linspace(0, line_length, 100)

    slope = ImgPlaneWidth/focalLength

    # Generate initial dots' coordinates
    # X, Y, xDot_vals, yDot_vals_1, yDot_vals_2 = generate_dots(slope, foreshorteningSlope, depthRangeValues)
    initial_X, initial_Y = generate_dots()
    
    # Apply transformation when button is clicked (e.g., shift the dots)
    # TODO: Fix this so that it uses the zoom and foreshortening functions below
    # if n_clicks > 0:
    X, Y = apply_transformation(initial_X, initial_Y, zoomScaleFactor, foreshorteningSlope, slope, depthRangeValues)
    
    # Calculate the y-values for both lines based on the slopes (swap x and y)
    y1_vals = slope * x_vals  # The line with slope 'm'
    y2_vals = -slope * x_vals  # The line with slope '-m'

    x_intersect1 = nearPlaneZValue / slope  # Intersection with the first line (upward slope)
    x_intersect2 = -nearPlaneZValue / slope

    vertical_line_x =[x_vals[-1], x_vals[-1]]  # x = 0 y1_vals[-1], y2_vals[-1]
    vertical_line_y = [y1_vals[-1], y2_vals[-1]]

    nearPlanel_line_x =[nearPlaneZValue, nearPlaneZValue]  # x = 0 y1_vals[-1], y2_vals[-1]
    nearPlanel_line_y = [x_intersect1,x_intersect2]

    # Get the selected range from the range slider
    Zminimum, Zmaximum = depthRangeValues

    # Select the part of the first line between the selected range
    selected_x_vals = x_vals[(x_vals >= Zminimum) & (x_vals <= Zmaximum)]
    selected_y1_vals = slope * selected_x_vals
    selected_y2_vals = -slope * selected_x_vals  # Mirrored points on the second line

    x_before = x_vals[x_vals < Zminimum]
    y_before = slope * x_before

    # Section of the line with the custom slope
    x_section = x_vals[(x_vals >= Zminimum) & (x_vals <= Zmaximum)]
    y_section = foreshorteningSlope * (x_section - Zminimum) + slope * Zminimum

    # After the section
    x_after = x_vals[x_vals > Zmaximum]
    y_after = slope * x_after

    # For the second line, the slope is negative (mirrored)
    y_default_second = -slope * x_vals
    
    # For the second line, apply the opposite slope in the selected section
    y_section_second = -foreshorteningSlope * (x_section - Zminimum) - slope * Zminimum  # Adjust for starting y-value
    
    # After the section, the second line follows the opposite of the default slope
    y_after_second = -slope * x_after

    # Combine all parts together for the second line
    y_all_second = np.concatenate([y_before, y_section_second, y_after_second])
    
    # Combine all parts together
    x_all = np.concatenate([x_before, x_section, x_after])
    y_all = np.concatenate([y_before, y_section, y_after])

    updated_color = random_color()
    
    # Define the figure data
    figure = {
        'data': [
            go.Scatter(x=x_vals, y=y1_vals, mode='lines', name=f'b(z) (Slope={slope:.2f})', line=dict(color='orange', width=3)),
            go.Scatter(x=x_vals, y=y2_vals, mode='lines', name=f'-b(z) (Slope={-slope:.2f})', line=dict(color='orange', width=3)),
            go.Scatter(x=vertical_line_x, y=vertical_line_y, mode='lines', name="Far plane",
                       line=dict(dash='dot', width=2, color='red')),
            go.Scatter(x=nearPlanel_line_x, y=nearPlanel_line_y, mode='lines', name="Near plane",
                        line=dict(dash='dot', width=2, color='green')),  # Dotted line at x = 0

            # Highlight the selected range along Line 1 (highlight area)
            go.Scatter(x=selected_x_vals, y=selected_y1_vals, mode='lines', fill='tozeroy', fillcolor='rgba(0, 100, 255, 0.3)', name="Selected depth range for x bound by b(z)"),
            
            # Highlight the mirrored selected range on Line 2 (highlight area)
            go.Scatter(x=selected_x_vals, y=selected_y2_vals, mode='lines', fill='tozeroy', fillcolor='rgba(0, 100, 255, 0.3)', name="Selected depth range for x bound by -b(z)"),

            # Highlight the section with the different slope
            go.Scatter(x=x_section, y=y_section, mode='lines', name="Foreshortening Slope for b(z)", line=dict(color='blue', width=3)),

            # Highlight the mirrored section for the second line
            go.Scatter(x=x_section, y=y_section_second, mode='lines', name="Foreshortening Slope for -b(z)", line=dict(color='blue', width=3)),

            # Plot the dots
            go.Scatter(x=X, y=Y, mode='markers', name="World coordinates", marker=dict(size=10, color='purple'))

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


#Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

    # XVals = range(-5, 6, 1)
    # ZVals = range(3, 11, 1)
    # XZpairs = [[x, z] for x in XVals for z in ZVals]

    # print("Original XZ pairs")
    # print(XZpairs)

    # newXZPairs = foreshortening(3, 50, 0.2, XZPairs)
    # print("New XZ pairs after foreshortening")
    # print(newXZPairs)

     # print("New XZ pairs after zoom")    
    # print(zoom(3, 5, 2, XZpairs))
