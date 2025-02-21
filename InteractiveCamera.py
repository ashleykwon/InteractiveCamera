import dash
from dash import dcc, html
import plotly.graph_objects as go
import numpy as np

# Initialize Dash app
# app = dash.Dash(__name__)

# # Generate initial data
# x = np.linspace(0, 2 * np.pi, 10)
# y = np.sin(x)

# # Create Plotly figure
# fig = go.Figure()

# # Add scatter trace with markers and lines
# fig.add_trace(go.Scatter(
#     x=x, 
#     y=y,
#     mode='markers+lines',  # Show points and lines
#     name='Editable Points',
#     marker=dict(size=12, color='blue'),
#     line=dict(color='blue', width=2)
# ))

# # Update layout to allow dragging of points
# fig.update_layout(
#     title="Editable Plot with Draggable Points",
#     xaxis_title="X",
#     yaxis_title="Y",
#     dragmode='drawopenpath',  # Allows modification of the plot (e.g., drag points)
# )

# # Dash layout
# app.layout = html.Div([
#     html.H1("Editable Plot with Draggable Points"),
#     dcc.Graph(
#         id='interactive-plot',
#         figure=fig
#     ),
# ])


# Apply zoom in/out
def zoom(Zminimum, Zmaximum, scaleFactor, XZpairs):
    # Divide x values in XZpairs by scaleFactor
    newXZpairs = [[x/scaleFactor, z] for [x,z] in XZpairs if z >= Zminimum and z <= Zmaximum]
    return newXZpairs


# Increase or decrease foreshortening
def foreshortening(Zminimum, Zmaximum, slope, XZpairs):
    newXZpairs = []
    XvalsAtZminimum = []
    XvalsAtZmaximum = []
    # Iterate over xz pairs and add them to newXZpairs if the Z values are within the z range
    for i in range(len(XZpairs)):
        if XZpairs[i][1] >= Zminimum and XZpairs[i][1] <= Zmaximum:
            newXZpairs.append(XZpairs[i])
        # Store in XvalsAtZminimum x values at Zminimum
            if XZpairs[i][1] == Zminimum:
                XvalsAtZminimum.append(XZpairs[i][0])
            if XZpairs[i][1] == Zmaximum:
                XvalsAtZmaximum.append(XZpairs[i][0])

    # Sort pairs in newXZpairs based on z values (ascending)
    newXZpairs = sorted(newXZpairs, key=lambda x: x[1])
    XvalsAtZminimum = sorted(XvalsAtZminimum)

    # Iterate through newXZpairs
    initialZ = newXZpairs[0][1] # smallest z value to begin with
    initialXVals = XvalsAtZminimum # x values at the smallest z value to begin with
    currentZ = initialZ
    currentXVals = []
    pairIdx = 0
    while pairIdx < len(newXZpairs):
        # Initiate a while loop and accumulate x-values with the current z-value in a list called currentXVals
        while pairIdx < len(newXZpairs) and newXZpairs[pairIdx][1] == currentZ: 
            currentXVals.append(newXZpairs[pairIdx][0])
            pairIdx += 1
        # Sort currentXVals based on x values (ascending)
        currentXVals = sorted(currentXVals)

        # Iterate through XatCurrentZ
        newX = 0
        for k in range(0, len(currentXVals)):
            # Find [currentX, currentZ] in newXZVals and replace the current value currentX to slope*(intitialZ-currentZ) + initialX
            pairToModifyIdx = newXZpairs.index([int(currentXVals[k]), currentZ])
            # Find the closest initial X value to the current X value
            closestInitialXval = min(initialXVals, key=lambda x: abs(x - currentXVals[k]))
           
            # Transforms x values within the positive x boundary
            if currentXVals[k] <= 0:
                newX = -1*slope*(currentZ - initialZ) + closestInitialXval
                newXZpairs[pairToModifyIdx] = [newX, currentZ]
            # Transforms x values within the negative x boundary
            else:
                newX = slope*(currentZ - initialZ) + closestInitialXval
                newXZpairs[pairToModifyIdx] = [newX, currentZ]

        # Update the currentZ value for the next layer of Z
        if pairIdx < len(newXZpairs):
            currentZ = newXZpairs[pairIdx][1]

        # Empty the currentXVals list for a new iteration
        currentXVals = []
    return newXZpairs


# Run the Dash app
if __name__ == '__main__':
    #app.run_server(debug=True)
    XVals = range(-5, 6, 1)
    ZVals = range(3, 11, 1)

    # x_values = [-3 + i if i < len(ZVals) // 2 else 0 - (i - len(ZVals) // 2) for i in range(len(ZVals))]
    # print(x_values)

    XZpairs = [[x, z] for x in XVals for z in ZVals]
    # print("Original XZ pairs")
    # print(XZpairs)
    # newXZPairs = foreshortening(3, 50, 0.2, XZPairs)
    # print("New XZ pairs")
    # print(newXZPairs)

    # print(zoom(3, 5, 2, XZpairs))
