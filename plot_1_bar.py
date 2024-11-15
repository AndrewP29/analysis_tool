import sqlite3
import pandas as pd
from IPython.display import display
import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter  # For Savitzky-Golay smoothing
from sklearn.linear_model import LinearRegression  # Import for linear fitting


# Connect to the database
conn = sqlite3.connect('C:/Users/popova/Documents/Bartab/2024-11_MIRL_batch-5176_LIV.db')

# Query to list all tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)

# Loop through each table and display its content
for table_name in tables:
    table_name = table_name[0]  # Extract table name from tuple
    print(f"\nData from table: {table_name}")
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    display(df)


currents = []
powers = []
voltages = []

graph_number=0

# Assuming tables variable is already defined
for table_name in tables:
    table_name = table_name[0]
    query = f"SELECT Current, Power, Voltage FROM {table_name}"
    df = pd.read_sql_query(query, conn)

    # Process Current
    current_str = df['current'].iloc[graph_number].decode('utf-8')
    current_values = [float(x) for x in current_str.strip().split('\t')]
    currents.extend(current_values)

    # Process Power
    power_str = df['power'].iloc[graph_number].decode('utf-8')
    power_values = [float(x) for x in power_str.strip().split('\t')]
    powers.extend(power_values)

    # Process Voltage
    voltage_str = df['voltage'].iloc[graph_number].decode('utf-8')
    voltage_values = [float(x) for x in voltage_str.strip().split('\t')]
    voltages.extend(voltage_values)

def calculate_fwhm(x, y):
    peak_idx = np.argmax(y)
    peak_value = y[peak_idx]
    half_max = peak_value / 2
    
    # Find points where the derivative crosses the half maximum level on both sides of the peak
    left_idx = np.where(y[:peak_idx] <= half_max)[0][-1]
    right_idx = np.where(y[peak_idx:] <= half_max)[0][0] + peak_idx

    # Calculate FWHM as the width between these points
    fwhm = x[right_idx] - x[left_idx]
    return fwhm


# Ensure lists have the same length (handle potential inconsistencies)
min_len = min(len(currents), len(powers), len(voltages))
currents = np.array(currents[:min_len])
powers = np.array(powers[:min_len])
voltages = np.array(voltages[:min_len])

# Calculate first and second derivatives
first_derivative = np.diff(powers) / np.diff(currents)
second_derivative = np.diff(first_derivative) / np.diff(currents[:-1])

# Savitzky-Golay filter parameters
first_window_length = 5  # Adjust as needed
second_window_length = 15  # Larger window for the second derivative
polyorder =3

# Apply Savitzky-Golay filter for the first derivative
first_derivative_smooth = savgol_filter(first_derivative, window_length=first_window_length, polyorder=polyorder)

# Apply Savitzky-Golay filter twice for enhanced smoothing on the second derivative
second_derivative_smooth = savgol_filter(second_derivative, window_length=second_window_length, polyorder=polyorder)
second_derivative_smooth = savgol_filter(second_derivative_smooth, window_length=15, polyorder=polyorder)

# Visible range: exclude first and last 10 points
visible_currents = currents[10:len(second_derivative_smooth) + 10]
first_derivative_smooth_visible = first_derivative_smooth[10:-10]
second_derivative_smooth_visible = second_derivative_smooth[10:-10]

# Scale derivatives based only on visible points
def scale_to_own_range(data, target_min, target_max):
    data_min, data_max = data.min(), data.max()
    return (data - data_min) / (data_max - data_min) * (target_max - target_min) + target_min

first_derivative_smooth_scaled = scale_to_own_range(first_derivative_smooth_visible, powers.min(), powers.max())
second_derivative_smooth_scaled = scale_to_own_range(second_derivative_smooth_visible, powers.min(), powers.max())

# Find the maximum of the second derivative within visible points to determine Ith
max_idx = np.argmax(second_derivative_smooth_scaled)
Ith_current = visible_currents[max_idx]


# Define a fraction of points after Ith to consider in the linear range
linear_fraction = 0.3  # Adjust as needed for your data

# Identify points after Ith up to the linear_fraction limit
Ith_idx = np.where(currents >= Ith_current)[0][0]
linear_end_idx = int(Ith_idx + (len(currents) - Ith_idx) * linear_fraction)
linear_currents = currents[Ith_idx:linear_end_idx]
linear_powers = powers[Ith_idx:linear_end_idx]

# Fit a linear regression model to the identified linear region
model = LinearRegression()
model.fit(linear_currents.reshape(-1, 1), linear_powers)  # Reshape for sklearn

# Extend the line of best fit across the full current range from Ith onward
extended_currents = currents[Ith_idx:]  # From Ith to the end of the current range
extended_fit_powers = model.predict(extended_currents.reshape(-1, 1))


# Define the fractions for the fitting range (30% to 50%)
start_fraction = 0.3
end_fraction = 0.5

# Calculate the indices for 30% and 50% of the data range
start_idx = int(len(currents) * start_fraction)
end_idx = int(len(currents) * end_fraction)

# Select the currents and voltages in the 30-50% range for fitting
fit_currents = currents[start_idx:end_idx]
fit_voltages = voltages[start_idx:end_idx]

# Fit a linear regression model to the selected voltage data
voltage_model = LinearRegression()
voltage_model.fit(fit_currents.reshape(-1, 1), fit_voltages)  # Reshape for sklearn

# Calculate V_on as the y-intercept of the fitted model
V_on = voltage_model.intercept_

# Predict voltage across the full range of currents for visualization
extended_voltage_fit = voltage_model.predict(currents.reshape(-1, 1))


# Calculate the series resistance for each point after V_on
series_resistance = (voltages - V_on) / currents
series_resistance = np.where((currents > 0) & (series_resistance > 0), series_resistance, np.nan)  # Filter to show only positive resistance

# Scale series resistance to the range of powers
def scale_to_match_range(data, target_min, target_max):
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    return (data - data_min) / (data_max - data_min) * (target_max - target_min) + target_min

# Find the maximum value of the original series resistance
max_series_resistance = np.nanmax(series_resistance)

series_resistance_scaled = scale_to_match_range(series_resistance, powers.min(), powers.max())







# Create Plotly figure with dual y-axes
fig = go.Figure()

# Add Power trace (left y-axis)
fig.add_trace(go.Scatter(x=currents, y=powers, mode='markers', name='Power',
                         marker=dict(size=5, color='blue'), yaxis='y1'))

# Add Voltage trace (right y-axis)
fig.add_trace(go.Scatter(x=currents, y=voltages, mode='markers', name='Voltage',
                         marker=dict(size=5, color='red'), yaxis='y2'))

# Add first smoothed and scaled derivative trace, visible points only
fig.add_trace(go.Scatter(
    x=visible_currents, 
    y=first_derivative_smooth_scaled, 
    mode='lines', 
    name='First Derivative (Smoothed)', 
    line=dict(color='green', width=2, dash='dot'), 
    opacity=0.5, 
    yaxis='y1', 
    visible=True))

# Add second smoothed and scaled derivative trace, visible points only
fig.add_trace(go.Scatter(
    x=visible_currents, 
    y=second_derivative_smooth_scaled, 
    mode='lines', 
    name='Second Derivative (Enhanced Smoothing)', 
    line=dict(color='purple', width=2, dash='dash'), 
    opacity=0.3, 
    yaxis='y1', 
    visible=True))

# Add Ith line as a separate trace for toggling
fig.add_trace(go.Scatter(
    x=[Ith_current, Ith_current],
    y=[powers.min(), powers.max()],
    mode="lines",
    name=f'Ith = {Ith_current:.3f} Second derivative method',
    line=dict(color="orange", width=2, dash="dash"),
    yaxis='y1',
    visible=True
))

# Calculate the FWHM as the uncertainty
fwhm_uncertainty = calculate_fwhm(visible_currents, second_derivative_smooth_scaled)
left_bound = Ith_current - fwhm_uncertainty / 2
right_bound = Ith_current + fwhm_uncertainty / 2

# Add the shaded uncertainty region as a toggleable scatter trace
fig.add_trace(go.Scatter(
    x=[left_bound, left_bound, right_bound, right_bound],
    y=[powers.min(), powers.max(), powers.max(), powers.min()],
    fill='toself',
    fillcolor='orange',
    opacity=0.2,
    line=dict(color='orange', width=0),
    mode='lines',
    name=f'Uncertainty (FWHM) = {fwhm_uncertainty:.3f}',
    yaxis='y1'
))



# Update layout with dual y-axes
fig.update_layout(
    title='Power and Voltage vs. Current with Enhanced Smoothed Derivatives (Excluding Edges)',
    xaxis_title='Current',
    yaxis=dict(
        title='Power',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue')
    ),
    yaxis2=dict(
        title='Voltage',
        titlefont=dict(color='red'),
        tickfont=dict(color='red'),
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0.785, y=0),  # Adjust x and y values as needed
    height=800  # Adjust 800 to your desired height
)

# Add extended line of best fit to the plot
fig.add_trace(go.Scatter(
    x=extended_currents,
    y=extended_fit_powers,
    mode="lines",
    name="Best Fit (Extended Linear Region)",
    line=dict(color="blue", width=2, dash="dash"),
    yaxis="y1"
))

# Add the extended line of best fit for voltage to the plot on the right axis
fig.add_trace(go.Scatter(
    x=currents,
    y=extended_voltage_fit,
    mode="lines",
    name="Voltage Best Fit (30%-50% Range)",
    line=dict(color="red", width=2, dash="dash"),
    yaxis="y2"  # Ensure it's plotted on the voltage axis (right side)
))

# Add the scaled series resistance trace to the plot
fig.add_trace(go.Scatter(
    x=currents,
    y=series_resistance_scaled,
    mode="lines",
    name="Scaled Series Resistance (V-V_on)/I",
    line=dict(color="purple", width=2),
    yaxis="y1"  # Plot on the left y-axis with power for direct comparison
))

# Annotate the maximum original series resistance value on the plot
fig.add_annotation(
    x=currents[np.nanargmax(series_resistance)],  # Current value at max resistance
    y=powers.max(),  # Position it at the top of the power axis
    text=f"Max Series Resistance = {max_series_resistance:.3f} Ω",
    showarrow=True,
    arrowhead=1,
    ax=0, ay=-40,
    font=dict(color="purple")
)


# Show the interactive plot
fig.show()