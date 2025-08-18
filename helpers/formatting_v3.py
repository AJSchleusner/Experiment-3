'''

Author: Austin J. Schleusner
Date: 2025-July-5

This is code for accessing the database files and simplifying plotting
of data collected by experiment_v3.py and acquisition_v3.ipynb.  Like
that of the experiment code, this is a development upon previous code 
largely written by Niyaz Beysengulov and Camille Mikolas.  My hope is 
to update our plotting code and merge in the parts of database2.py that
we have relied upon to pull data from the database files.  In the 
creation of plots, I plan to incorporate the table structure I used for 
the meta data of my NV-center experiments.  I think it looks clean and 
if you are reading this and disagree, I am sorry that you have poor 
taste.  Like in the experiment code, Microsoft Copilot is helping me 
with the coding, but not to an extent where I am obsolete.

'''



###########################################################################################
## Imports ----------------------------------------------------------------------------- ##
###########################################################################################

import numpy as np
import sqlite3
import pandas as pd
import os
import re
import math

from datetime import date
from newinstruments.BlueFors import BlueFors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.table import Table
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from functools import partial
from scipy.optimize import curve_fit
from pathlib import Path
import textwrap


###########################################################################################
## Accessing the Database Files -------------------------------------------------------- ##
###########################################################################################

class access_db():
    # Initialization method for the access_db class.  This runs when an instance of the
    # class is created.
    def __init__(self, db_path: str):
        # Connect to the SQLite database file
        self.conn = sqlite3.connect(db_path)
    # Collect a list of all tables in the database
    def list_tables(self):
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        return [row[0] for row in self.conn.execute(query).fetchall()]
    # Get the names of all columns in a specific table
    def get_columns(self, table: str):
        cursor = self.conn.execute(f"SELECT * FROM {table} LIMIT 1")
        return [col[0] for col in cursor.description]
    # Get the data from a specific table as a pandas DataFrame
    def grab_table(self, table: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {table}", self.conn)
    # Close the connection to the database
    def close_db(self):
        self.conn.close()

# Check if a value is a valid S-parameter string (e.g., "S21", "S11")
def is_s_parameter(value):
    return isinstance(value, str) and re.fullmatch(r"S\d{2}", value.upper()) 

# Extracts a tuple of two floats from a string like '(4.4e9, 4.6e9) set_frequency_range'
def extract_freq_tuple(s):
    match = re.match(r"\(([^,]+),\s*([^)]+)\)", s)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None

# Get data from the SQLite database file and return it in a structured format
def load_experiment_data(db_path: str):
    # Automatically append '.db' if it's missing
    if not db_path.endswith(".db"):
        db_path += ".db"
    # Create an instance of the access_db class to connect to the database
    db = access_db(db_path)
    # Check what tables are available in the database
    available_tables = db.list_tables()
    # Define a function to grab data from a specific table
    def grab(name):
        return db.grab_table(name) if name in available_tables else None
    # Access the info table to get metadata about the experiment
    info = grab('table_info')

    # Initialize an empty dictionary to hold metadata
    metadata = {}
    # Extract the headers to set as metadata_col
    for col in info.columns:
        # Get the raw values from the column, dropping any NaN values
        raw_vals = info[col].dropna().tolist()
        # Handle frequency columns: convert to float, convert Hz → GHz
        if "freq_range" in col.lower():
            parsed_vals = [extract_freq_tuple(v) for v in raw_vals]
            parsed_vals = [t for t in parsed_vals if t]  # remove failed parses
            # Flatten each (start, stop) pair into two lines in order
            formatted = []
            for pair in parsed_vals:
                formatted.extend([f"{scale_to_ghz(val)} GHz" for val in pair])
            metadata[col] = formatted
        # If the column is 'set_vna_meas', extract S-parameters
        elif col == "set_vna_meas":
            match = re.search(r"\bS\d{2}\b", str(raw_vals))
            if match:
                metadata.setdefault(col, []).append(match.group())
        # Remove the suffix from the sweep_pts column and just report the value
        elif "sweep_pts" in col.lower():
            # Extract just the numeric portion from each entry
            values = [re.search(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", str(v)) for v in raw_vals]
            cleaned = [match.group() if match else "N/A" for match in values]
            metadata[col] = cleaned
        else:
            # Default: treat values as strings
            values = [str(v) for v in raw_vals]
            metadata[col] = values

    # Access the tables containing different types of data
    data = grab('table_data')
    sweep = grab('table_sweep')
    step = grab('table_step')   # Note step will be empty in a 1D measurement
    # Get the headers for the sweep and step tables, if they exist
    sweep_headers = sweep.columns.tolist() if sweep is not None else []
    step_headers  = step.columns.tolist()  if step is not None else []
    # End the connection to the database
    db.close_db()
    # Return a dictionary containing the data, sweep, step, and metadata
    return {
        "data": data.to_numpy() if data is not None else None,
        "sweep": sweep.to_numpy() if sweep is not None else None,
        "step": step.to_numpy() if step is not None else None,
        "metadata": metadata,
        "sweep_headers": sweep_headers,
        "step_headers": step_headers}

# Generate a unique filename (used for saving figures)
def get_unique_filename(save_dir, base_name, extension=".jpg", use_date=True,
                        custom_date=None):
    save_path = Path(save_dir)
    # Get today's date or use a provided one
    date_str = custom_date if custom_date else date.today().isoformat()
    idx = 1
    while True:
        if use_date:
            candidate = f"{date_str}_{base_name}_{idx}{extension}"
        else:
            candidate = f"{base_name}_{idx}{extension}"
        full_path = save_path / candidate
        if not full_path.exists():
            return full_path
        idx += 1

# Try to pull the format from the metadata for later use
def get_vna_format(metadata):
    # Try to call to the format key in the metadata dictionary
    try:
        return metadata['format'][0].upper()
    # If the format key cannot be retrieved
    except KeyError:
        raise KeyError('Missing format key in metadata')


###########################################################################################
## Figure Formation Helpers ------------------------------------------------------------ ##
###########################################################################################

# Format plots for displaying data and the meta data.
def format_plot(fig_wide = 12, fig_tall = 9, left_width = 2.5, right_width = 1):
    fig = plt.figure(figsize= (fig_wide, fig_tall))
    # Create a GridSpec layout with 1 row and 2 columns
    gs = GridSpec(1, 2, width_ratios=[left_width, right_width], figure=fig)
    # Create subplots using the GridSpec layout
    ax_plot = fig.add_subplot(gs[0, 0])  # Left subplot
    ax_meta = fig.add_subplot(gs[0, 1])  # Right subplot
    # Turn off the right subplot's axis
    ax_meta.axis('off')
    return fig, ax_plot, ax_meta

# Comment wrapping definition
def wrap_comment(text, max_chars=30):
    lines = text.split('\n')
    wrapped = [textwrap.fill(line, max_chars) for line in lines]
    return "\n".join(wrapped)

# Get a list of colors indexed by the number of lines in the plot
def get_indexed_colors(num_lines, cmap_name='viridis'):
    cmap = cm.get_cmap(cmap_name, num_lines)
    norm = mcolors.Normalize(vmin=0, vmax=num_lines - 1)
    return [cmap(norm(i)) for i in range(num_lines)]


# Format plots for displaying data and n metadata tables.
def form_plot(num_meta: int=1, fig_w = 13, fig_h = 9, left_width = 2.5,
              right_width = 1, facecolors=None, titles=None):
    fig = plt.figure(figsize= (fig_w + 2*num_meta - 2, fig_h))
    # Total columns is 1 for data and num_meta for metadata
    total_cols = 1 + num_meta
    right_width = [right_width] * num_meta
    width_ratios = [left_width] + right_width
    # Create a GridSpec layout
    gs = GridSpec(1, total_cols, width_ratios=width_ratios, figure=fig)
    # Create subplots using the GridSpec layout
    ax_plot = fig.add_subplot(gs[0, 0])  # Data subplot
    # Create an empty list to hold metadata axes
    ax_meta_list = []
    # Loop through the number of metadata sections to create subplots
    for i in range(num_meta):
        # Create a new subplot for each metadata section
        ax = fig.add_subplot(gs[0, i + 1])
        ax.axis('off')  # Turn off the axis for metadata subplots
        # Set the face color if provided
        if facecolors and i < len(facecolors):
            ax.set_facecolor(facecolors[i])
        # Title are assigned in the metadata table function
        # Append the axis to the list
        ax_meta_list.append(ax)
    # Return the figure components
    return fig, ax_plot, ax_meta_list


def meta_table(ax, metadata: dict, title: str = 'Metadata', fontsize=12,
             comm_on: bool = False, facecolor='aliceblue', style=None):
    # Make a copy of the metadata to avoid modifying the original
    metadata = metadata.copy()
    # Remove comment fields from metadata
    comment_keys = [k for k in metadata if 'comment' in k.lower() or 'note' in k.lower()]
    comments = [f"{k}: {', '.join(map(str, metadata.pop(k)))}" for k in comment_keys]
    # Clear axes
    ax.clear()
    # Dynamically change box_height based on whether comments are present
    has_comments = comm_on and bool(comments)

    # Estimate comment line count
    comment_lines = wrap_comment("\n".join(comments), max_chars=30).count('\n') + 1 if has_comments else 0
    # Estimate total content height including comments
    metadata_lines = sum(len(v) if isinstance(v, list) else 1 for v in metadata.values())
    total_lines = metadata_lines + comment_lines + 1  # +1 for padding/title

    box_height   = 0.96 
    line_spacing = box_height / total_lines
    # Clamp to avoid overflow
    patch_y = 0.5 - box_height / 2  # Center the box vertically

    # 1. Calculate layout
    y_pos_start = 0.5 + box_height / 2 - line_spacing  # Top of the box

    # 2. Add the patch centered around content
    rect = patches.FancyBboxPatch(
        (0, patch_y), 1.1, box_height,
        boxstyle="round,pad=0.02",
        linewidth=style['tick_width'] if style else 3,
        edgecolor="black",
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False)
    # Add the rectangle to the table
    ax.add_patch(rect)

    # 3. Title and metadata
    # Position title just above the top edge of the patch
    title_y = patch_y + box_height + 0.03  # small margin above patch
    ax.text(0.5, title_y, title,
            fontsize=fontsize + 4,
            ha='center',
            va='bottom',
            transform=ax.transAxes)
    ax.axis('off')

    # 4. Render content at correct y_pos
    key_width = 0.55  # percentage of horizontal space for keys
    value_start = 0.03 + key_width + 0.04  # left padding + key width + spacing
    y_pos = y_pos_start
    for k, v in metadata.items():
        val_lines = v if isinstance(v, list) else [str(v)]
        num_lines = len(val_lines)
        ax.text(0.02, y_pos, k, fontsize=fontsize, fontfamily='monospace',
                ha='left', va='top', transform=ax.transAxes)
        for i, line in enumerate(val_lines):
            ax.text(value_start, y_pos - i * line_spacing, line, fontsize=fontsize,
                    fontfamily='monospace', ha='left', va='top', transform=ax.transAxes,
                    clip_on=True)
        y_pos -= line_spacing * num_lines

    # 5. Render comments below (if enabled)
    if comments and comm_on:
        comment_text = wrap_comment("\n".join(comments), max_chars=30)
        ax.text(
            0.02, patch_y + 0.01,
            comment_text,
            fontsize=fontsize - 2,
            fontstyle='italic',
            fontfamily="monospace",
            ha='left',
            transform=ax.transAxes,
            clip_on=False,
            linespacing=1.3,
            wrap=True)




def qfactor_table(ax, qf_data, title="Quality Factors", fontsize=12,
                  facecolor='bisque', style=None):
    ax.clear()
    ax.axis('off')
    # Layout parameters
    box_height = 0.96
    line_spacing = box_height / (len(qf_data) + 1)
    patch_y = 0.5 - box_height / 2

    # Background patch
    rect = patches.FancyBboxPatch(
        (0, patch_y), 1.1, box_height,
        boxstyle="round,pad=0.02",
        linewidth=style['tick_width'] if style else 3,
        edgecolor="black",
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False)
    ax.add_patch(rect)

    # Total content height
    content_height = line_spacing * len(qf_data)
    # Adjust to center the content block
    y_pos = 0.5 + content_height / 2 
    # Title
    title_y = patch_y + box_height + 0.05
    ax.text(0.5, title_y, title,
            fontsize=fontsize + 4,
            ha='center', va='bottom',
            transform=ax.transAxes)

    # Content rendering
    key_x = 0.04
    val_x = 0.62
    for key, val in qf_data:
        ax.text(key_x, y_pos, key, fontsize=fontsize,
                fontfamily='monospace', ha='left', va='top',
                transform=ax.transAxes)
        ax.text(val_x, y_pos, val, fontsize=fontsize,
                fontfamily='monospace', ha='left', va='top',
                transform=ax.transAxes)
        y_pos -= line_spacing

def scale_to_ghz(val):
    if val > 1e6:  # Assume Hz
        return round(val / 1e9, 3)
    return round(val, 3)  # Already in GHz

def format_val(val, unit="", precision=0, default="—"):
    try:
        if precision == 0:
            return f"{int(round(val))} {unit}".strip()
        return f"{round(val, precision)} {unit}".strip()
    except (TypeError, ValueError):
        return default


def format_scaled(val, unit="GHz", default="—"):
    try:
        return f"{scale_to_ghz(val)} {unit}"
    except (TypeError, ValueError):
        return default


###########################################################################################
## Formatting the Plot ----------------------------------------------------------------- ##
###########################################################################################

# General Format ------------------------------------------------------------------------ #

def reshape_to_2dz(x, y, z):
    # Find the unique x and y values
    unique_x = np.unique(x)
    unique_y = np.sort(np.unique(y))
    # Create a 2D grid for the z values
    out_z = z.reshape(len(unique_y), len(unique_x))
    # If the first y is less than the last, flip the data vertically
    if unique_y[0] < unique_y[-1]:
        out_z = out_z[::-1, :] # Flip vertically
    out_x = unique_x
    out_y = unique_y
    return out_x, out_y, out_z

def to_magnitude(real, imag):
    return np.sqrt(real**2 + imag**2)

def to_phase(real, imag, deg=True):
    phase_rad = np.arctan2(imag, real)
    return np.degrees(phase_rad) if deg else phase_rad


def general_plt(ax, xlabel=None, ylabel=None, title=None, log_y=None, ylims=None,
                style=None, legend=True):
    # Set the axes labels
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=style['fontsize'], labelpad=style['labelpad'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=style['fontsize'], labelpad=style['labelpad'])
    # Set the title of the plot
    if title:
        ax.set_title(title, fontsize=style['title_fontsize'], pad=style['title_pad'])
    # Set the tick parameters
    ax.tick_params(axis='both',
                   length=style['tick_length'],
                   width=style['tick_width'],
                   direction='in',
                   labelsize=style['fontsize'],
                   pad=style['tick_pad'])
    # Set the major locators for x and y axes
    ax.xaxis.set_major_locator(style['x_locator'])
    ax.yaxis.set_major_locator(style['y_locator'])
    # Set the spine parameters
    for spine in ax.spines.values():
        spine.set_linewidth(style['tick_width'])
    # If true, set the y-axis to logarithmic scale
    if log_y:
        ax.set_yscale('log')
    # If provided, set y limits
    if ylims:
        ax.set_ylim(ylims)
    # If labels are provided, add a legend
    if legend:
        ax.legend(fontsize=style['fontsize'] - 1)


# 1D Plot ------------------------------------------------------------------------------- #
def form_1d_plot(ax, sweep, data, meta, filename=None, sweep_type=None, style=None):
    # Clear the axis before plotting
    ax.clear()

    # Customize plot based on sweep_type
    if sweep_type == 'freq_range':
        # Determine which data type was taken from the VNA
        vna_form = get_vna_format(meta)
        if vna_form in ('MLOG',):
            y_label = "S21 (dB)"
        elif vna_form == 'PHAS':
            y_label = "S21 (degrees)"
        elif vna_form == 'IMAG':
            y_label = "Im(S21)"
        elif vna_form == 'REAL':
            y_label = "Re(S21)"
        else:
            print(f"Unknown VNA format: {vna_form}. Update code to include format.")
            return
        # Scale the VNA sweep from Hz to GHz.  Also for a 'freq_range' sweep, the measured
        # amplitude is stored in data[:,4] (column number 5).
        ax.plot(sweep*1e-9, data[:,4], 
                       color=style['line_color'], 
                       linewidth=style['line_width'])
        # Set freq_range plot settings
        general_plt(ax, xlabel="Frequency (GHz)", ylabel=y_label,
                    title=filename, log_y=False, ylims=None,
                    style=style, legend=False)
    # add other sweep_type conditions as needed









# 2D Plot ------------------------------------------------------------------------------- #
def form_2d_plot(ax, data, meta, filename=None, sweep_type=None, 
                vmin=None, vmax=None, style=None, sdata_phase=False):

    # Clear the axis before plotting
    ax.clear()

    # If the sweep_type is one of the vna options
    if sweep_type in ('freq_range', 'power', 'elec_delay'):
            # Determine which data type was taken from the VNA
        vna_form = get_vna_format(meta)
        if vna_form == 'MLOG':
            z_label = "S21 (dB)"
        elif vna_form == 'PHAS':
            z_label = r"$\phi$ (S21) (degrees)"
        elif vna_form == 'IMAG':
            z_label = "Im(S21) (dB)"
        elif vna_form == 'REAL':
            z_label = "Re(S21) (dB)"
        elif vna_form == 'SDATA' and sdata_phase:
            z_label = r"$\phi$ (S21) (degrees)"
        elif vna_form == 'SDATA' and not sdata_phase:
            z_label = "S21 (dB)"
        else:
            print(f"Unknown VNA format: {vna_form}. Update code to include format.")
            return

    # Customize plot based on sweep_type
    if sweep_type == 'power':
        # Select the columns in the data table 
        if vna_form == 'SDATA':
            powers      = data[:,3] 
            real_col    = data[:,4]
            imag_col    = data[:,5]
            frequencies = data[:,6]
            # Convert the real and imaginary parts to magnitude if sdata_phase is False
            if not sdata_phase:
                measurements = 20*np.log10(np.sqrt(real_col**2 + imag_col**2))
            # Convert to phase if sdata_phase is True
            if sdata_phase:
                measurements = np.angle(real_col + 1j*imag_col, deg=True)
        else:
            powers       = data[:,3]
            measurements = data[:,4]
            frequencies  = data[:,5]

        # Preprocess axis data
        data_x, data_y, data_z = reshape_to_2dz(frequencies, powers, measurements)
        data_x *= 1e-9  # Convert Hz to GHz

        # Plot the 2D power sweep
        if vmin is None or vmax is None:
            mesh = ax.pcolormesh(data_x, data_y, data_z, 
                                shading=style['shading'],
                                cmap=style['color_map'],
                                vmin=int(data_z.min()), vmax=int(data_z.max()))
        else:
            mesh = ax.pcolormesh(data_x, data_y, data_z,
                                shading=style['shading'],
                                cmap=style['color_map'], 
                                vmin=vmin, vmax=vmax)
        # Set the axis labels
        ax.set_xlabel("Frequency (GHz)", 
                      fontsize=style['fontsize'], 
                      labelpad=style['labelpad'])
        ax.set_ylabel("Power (dBm)", 
                      fontsize=style['fontsize'], 
                      labelpad=style['labelpad'])
        # Add a colorbar to the plot
        fig  = ax.get_figure()
        cbar = fig.colorbar(mesh, ax=ax)

        # Customize tick font size
        cbar.ax.tick_params(labelsize=style['fontsize'],
                            width=style['tick_width'],
                            length=style['tick_length'],
                            direction='out',
                            pad=style['tick_pad'])

        # Match colorbar spine width
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(style['tick_width'])
        # Style label font
        cbar.set_label(z_label, 
                       fontsize=style['fontsize'], 
                       labelpad=style['labelpad'])

    # General 2D plot settings 
    if filename:
        ax.set_title(filename, 
                     fontsize=style['title_fontsize'], 
                     pad=style['title_pad'])
    # Tick parameters 
    ax.tick_params(axis='both', 
                   length=style['tick_length'], 
                   width=style['tick_width'], 
                   direction='out', 
                   labelsize=style['fontsize'], 
                   pad=style['tick_pad'])
    ax.xaxis.set_major_locator(style['x_locator'])
    ax.yaxis.set_major_locator(style['y_locator'])
    # Spine parameters
    for spine in ax.spines.values():
        spine.set_linewidth(style['tick_width'])
    # Enforce y-limit boundaries
    ax.set_ylim(min(data_y), max(data_y)) 

    # Return the x, y, and z data for further processing if needed
    return data_x, data_y, data_z


# Linecut ------------------------------------------------------------------------------- #
def get_linecut(data_x, data_y, data_z, axis='x', value=None):
    if axis == 'y':
        indx = np.abs(data_y - value).argmin()
        return data_x, data_z[indx, :]
    elif axis == 'x':
        indx = np.abs(data_x - value).argmin()
        return data_y, data_z[:, indx]
def plot_linecut(ax, x, y, xlabel, ylabel, filename=None, style=None, **plot_kwargs):
    ax.clear()
    # Unpack style dict
    base_style = {
        'color': style['line_color'],
        'linewidth': style['line_width'],
        # You can add other defaults here if needed
    }
    # Give kwargs precedence over base_style
    final_style = {**base_style, **plot_kwargs}

    # Plot the line
    ax.plot(x, y, **final_style)
    # Axes labels
    ax.set_xlabel(xlabel, fontsize=style['fontsize'], labelpad=style['labelpad'])
    ax.set_ylabel(ylabel, fontsize=style['fontsize'], labelpad=style['labelpad'])
    # Title
    if filename:
        ax.set_title(filename, fontsize=style['title_fontsize'], pad=style['title_pad'])
    # Tick styling
    ax.tick_params(
        axis='both',
        length=style['tick_length'],
        width=style['tick_width'],
        direction='in',
        labelsize=style['fontsize'],
        pad=style['tick_pad'])
    # Set major locators for x and y axes
    ax.xaxis.set_major_locator(style['x_locator'])
    ax.yaxis.set_major_locator(style['y_locator'])
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(style['tick_width'])
    # Clean y-limits to integer bounds
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(math.floor(ymin), math.ceil(ymax))

 
# Scatter Plot -------------------------------------------------------------------------- #
def scatter_1d_plot(ax, x_series, y_series, labels=None, colors=None, 
                    xlabel=None, ylabel=None, filename=None, 
                    log_y=False, ylims=None, style=None):
    ax.clear()
    # For each series in the x and y data, plot the points
    for i in range(len(x_series)):
        ax.plot(x_series[i], y_series[i], 'o',
                label=labels[i] if labels else None,
                markersize=style['marker_size'],
                color=colors[i] if colors else style['line_color'])
    # Set the general plot settings
    general_plt(ax, xlabel=xlabel, ylabel=ylabel,
                title=filename, log_y=False, ylims=None,
                style=style, legend=False)


# Multivariable Plot -------------------------------------------------------------------- #

def plot_exp_sets(ax, plot_data, xlabel=None, ylabel=None, filename=None,
                  log_y=False, ylims=None, style=None):
    # Clear the axis before plotting
    ax.clear()
    # Plot each set of data
    for entry in plot_data:
        x_val = entry['x']
        y_val = entry['y']
        label = entry.get('label', None)
        color = entry.get('color', style['line_color'])
        marks = entry.get('marks', 'o') # Default to circle marker
        # Make a plot with the correct marker style
        if marks == 'o':
            ax.plot(x_val, y_val, 'o', label=label, 
                    markersize=style['marker_size'], 
                    color=color)
        elif marks == '-':
            ax.plot(x_val, y_val, '-', label=label, 
                    linewidth=style['line_width'], 
                    color=color)
        else:
            print(f"Problem: marks must be 'o' or '-', got {marks}")
            return
    # Set the general plot settings
    general_plt(ax, xlabel=xlabel, ylabel=ylabel,
                title=filename, log_y=log_y, ylims=ylims,
                style=style, legend=True)
    return


# Waterfall Plot ------------------------------------------------------------------------ #

def plot_waterfall(ax, x, y, z, y_indices=None, offset_step=5, 
                   style=None, do_fits=False, user_guess=None,
                   artifact_indices=None, cmap_name='viridis'):
    """
    Plots selected linecuts of a 2D dataset in a waterfall format.
    
    Parameters:
    ax          : matplotlib axis
    x, y, z     : 2D plot data (e.g., from form_2d_plot)
    y_indices   : list of y-axis indices to plot linecuts from
    offset_step : vertical offset between traces
    style       : dict for styling parameters (optional)
    """
    # Default: evenly spaced linecuts across y
    if y_indices is None:
        y_indices = np.linspace(0, len(y)-1, 10, dtype=int)
    # Get indexed colors
    colors = get_indexed_colors(len(y_indices), cmap_name)
    # Create an empty quality factor list
    q_factors = []
    # Loop through the selected y indices
    for idx, i in enumerate(y_indices):
        trace = z[i, :]
        # Plotting parameters 
        label = f"{y[i]:.1f} dBm"
        offset = idx * offset_step
        color = colors[idx] if colors else style.get('color', None)
        # Optionally find the lorz_inv fits and quality factor
        if do_fits:
            try:
                popt, _, model = fit_data(x, trace, 'lorz_inv', artifact_indices, 
                                          user_guess=user_guess)
                q_factors.append(popt[1] / popt[2])
            except Exception:
                q_factors.append(np.nan)
            # Add quality factor to the label if fitting was done
            label = f"{y[i]:.1f} dBm, Q={q_factors[-1]:.2f}"
        # Make the waterfall plot
        ax.plot(x, trace + offset, label=label,
                linewidth=style.get('line_width', 1.5),
                color=color)
        # If fitting was done, plot the fit curve
        if do_fits:
            fit_curve = model(x, *popt)
            # Plot the fit curve with a dashed line
            ax.plot(x, fit_curve + offset, 
                    '--', color='black', linewidth=2)
    # Apply the general_plt function to format the plot
    general_plt(ax,
                xlabel="Frequency (GHz)",
                ylabel="S21 (dBm)",
                title="Waterfall Plot",
                style=style,
                legend=True)
    # Return the quality factors if fits were done
    return np.array(q_factors) if do_fits else None













###########################################################################################
## Fitting Code ------------------------------------------------------------------------ ##
###########################################################################################

def linear(x, slope, intercept):
    return slope * x + intercept

def lorentzian(x, amp, center, width, offset):
    return amp*(0.5*width)**2 / ((x - center)**2 + (0.5*width)**2) + offset

def lorz_inv(x, amp, center, width, offset):
    return -amp*(0.5*width)**2 / ((x - center)**2 + (0.5*width)**2) + offset

def gaussian(x, amp, center, sigma, offset):
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset

# Get the appropriate fitting model and initial guess based on the fit_type
def get_fit_model(fit_type):
    models = {
        'linear': {
            'func': linear,
            'guess': lambda x, y: [np.polyfit(x, y, 1)[0], np.polyfit(x, y, 1)[1]],
            'params': ['slope', 'intercept']
        },
        'lorentzian': {
            'func': lorentzian,
            'guess': lambda x, y: [max(y) - min(y), x[np.argmax(y)], (max(x) - min(x)) / 10, min(y)],
            'params': ['amplitude', 'center', 'width', 'offset']
        },
        'lorz_inv': {
            'func': lorz_inv,
            'guess': lambda x, y: [max(y) - min(y), x[np.argmin(y)], (max(x) - min(x)) / 10, min(y)],
            'params': ['amplitude', 'center', 'width', 'offset']
        },
        'gaussian': {
            'func': gaussian,
            'guess': lambda x, y: [max(y) - min(y), x[np.argmax(y)], (max(x) - min(x)) / 10, min(y)],
            'params': ['amplitude', 'center', 'width', 'offset']
        }
    }
    if fit_type not in models:
        raise ValueError(f"Unsupported fit_type: {fit_type}")
    return models[fit_type]['func'], models[fit_type]['guess'], models[fit_type]['params']

# Help text for each model
model_help = {
    'linear': 'linear(x, m, b): m = slope, b = intercept',
    'lorentzian': 'lorentzian(x, A, x0, w, y0): A = amplitude, x0 = center, w = width, y0 = offset',
    'lorz_inv': 'lorz_inv(x, A, x0, w, y0): inverse form of Lorentzian',
    'gaussian': 'gaussian(x, A, x0, w, y0): standard Gaussian model'
}
def print_model_help():
    print("📘 Fit Models Reference:")
    for name, desc in model_help.items():
        print(f"- {name}: {desc}")

# Fit data using curve_fit from scipy.optimize
def fit_data(x, y, fit_type, artifact_indices=None, user_guess=None):
    if artifact_indices:
        x_new, y_new = exclude_artifacts(x, y, artifact_indices)
    else:
        x_new, y_new = x, y
    # Extract the model function, guess function, and parameter names
    model_func, guess_func, param_names = get_fit_model(fit_type)
    expected_len = len(param_names)
    # If a user guess is provided, check its length
    if user_guess is not None:
        if len(user_guess) != expected_len:
            raise ValueError(f"Incorrect guess length for '{fit_type}'. Expected {expected_len} parameters: {param_names}")
        p0 = user_guess
    else:
        p0 = guess_func(x_new, y_new)
    # Perform the curve fitting
    popt, pcov = curve_fit(model_func, x_new, y_new, p0=p0)
    return popt, pcov, model_func

# Find quality factors using the inverse lorentzian
def quality_fits(freq, power, data, artifact_indices=None, user_guess=None,
                 x_window_min=None, x_window_max=None):
    # Initialize for quality factor sweep
    fit_type = 'lorz_inv'
    q_values = []
    # Reshape to make data 2D
    data_x, data_y, data_z = reshape_to_2dz(freq, power, data)
    # Convert frequencies to GHz
    data_x = data_x * 1e-9
    # Run the fitting loop
    for i in range(data_z.shape[0]):
        # Take the i-th row of the data_z matrix
        trace = data_z[i, :]
        # Apply x-windowing before fitting
        if x_window_min is not None and x_window_max is not None:
            mask = (data_x >= x_window_min) & (data_x <= x_window_max)
            x_fit = data_x[mask]
            z_fit = trace[mask]
        else:
            x_fit = data_x
            z_fit = trace
        # Fit the data
        try:
            popt, _, _ = fit_data(x_fit, z_fit, fit_type, artifact_indices,
                                   user_guess=user_guess)
            # Calculate the quality factor
            q = popt[1] / popt[2]
        except Exception:
            q = np.nan  # If fitting fails, set q to NaN
        # Replace if q is NaN or non-positive
        if np.isnan(q) or q <= 0:
            q = 1e-10
        # Append the quality factor to the list
        q_values.append(q)
    return np.array(data_y), np.array(q_values)

# Exclude artifacts for curve fitting
def exclude_artifacts(x, z, artifact_indices):
    mask = np.ones(len(x), dtype=bool)
    mask[artifact_indices] = False
    return x[mask], z[mask]


###########################################################################################
## Saving Figure Into Data Structure --------------------------------------------------- ##
###########################################################################################

# Create a figure saving path based on the database file name and the existing figures
# in the database's folder.
def create_save_path(db_path: str, fig_type:str = 'png') -> str:
    # The folder where the database file is located
    folder = os.path.dirname(db_path)
    # os.path.basename extracts the file name from the path, while then os.path.splitext
    # removes the file extension to get just the base name of the database file.
    base_name = os.path.splitext(os.path.basename(db_path))[0]
    # Check for names in the existing figures pattern: experiment_1_Fig_1.png, etc.
    pattern = re.compile(rf"{re.escape(base_name)}_Fig_(\d+)\.{fig_type}")
    # List all files (names) in the folder that match the pattern
    existing_files = [f for f in os.listdir(folder) if pattern.match(f)]
    # Extract the numeric figure suffixes and find the highest
    existing_indices = [int(pattern.match(f).group(1)) for f in existing_files]
    # Determine the next index for the new figure file
    next_index = max(existing_indices) + 1 if existing_indices else 1
    # Create the new filename with the next index and the specified figure type
    new_filename = f"{base_name}_Fig_{next_index}.{fig_type}"
    # Join the folder path and the new filename to create the full path
    return os.path.join(folder, new_filename)




















###########################################################################################
## Circle Fitting and Plotting --------------------------------------------------------- ##
###########################################################################################

# Reshape circle fit data into a dictionary format for easier access
def reshape_circle_data(freqs, real, imag, power):
    freqs = np.array(freqs)
    real  = np.array(real)
    imag  = np.array(imag)
    power = np.array(power)
    # Identify unique step values
    unique_steps = np.unique(power)
    # Group indices by step value (as opposed to step index)
    grouped_data = {}
    for p in unique_steps:
        mask = (power == p)
        grouped_data[p] = {
            'freq': freqs[mask],
            'S21': real[mask] + 1j * imag[mask]
        }
    return grouped_data  # Dictionary keyed by step value (usually power in dBm)




def format_circle_data(freqs, real, imag, power,
                       remove_phase_slope=False,
                       reconstruct_complex=False, n=8):
    freqs = np.array(freqs)
    real  = np.array(real)
    imag  = np.array(imag)
    power = np.array(power)

    # Allocate arrays
    powers_1d = np.sort(np.unique(power))
    freqs_1d  = np.sort(np.unique(freqs))
    mag       = {}
    phase     = {}

    for i, p in enumerate(powers_1d):
        mask = (power == p)
        s21  = real[mask] + 1j * imag[mask]
        mag  [p] = np.abs(s21)
        phase[p] = np.unwrap(np.angle(s21))

    # Optionally remove linear slope in phase (vs frequency) for each power slice
    if remove_phase_slope:
        phase_prime = {}
        for p in powers_1d:
            ph = phase[p]

            # First and last n-point averages
            x_start = np.mean(freqs_1d[:n])
            x_end   = np.mean(freqs_1d[-n:])
            y_start = np.mean(ph[:n])
            y_end   = np.mean(ph[-n:])

            # Compute slope from line connecting these averages
            slope = (y_end - y_start) / (x_end - x_start)

            # Construct linear trend and subtract
            linear_trend = slope * (freqs_1d - x_start) + y_start
            phase_prime[p] = ph - linear_trend


    # Optionally reconstruct complex data from mag and phase_prime
    if reconstruct_complex:
        real_prime = {}
        imag_prime = {}
        for p in powers_1d:
            real_prime[p] = mag[p] * np.cos(phase_prime[p])
            imag_prime[p] = mag[p] * np.sin(phase_prime[p])
    else:
        real_prime, imag_prime = None, None

    return {
        'freq': freqs_1d,
        'power': powers_1d,
        'mag': mag,
        'phase': phase,
        'phase_prime': phase_prime,
        'real_prime': real_prime,
        'imag_prime': imag_prime}


# Three panel plotting to show : circle fit, resonance curve, and metadata
def triad_plot(freq, S21, settings, fig_dim = (10, 12), filename = None,
                w_ratio = [2, 1], h_ratio = [1, 1], alpha=1, c_adj=[0,0]):
    fig = plt.figure(figsize=fig_dim)
    gs = GridSpec(2, 2, width_ratios=w_ratio, height_ratios=h_ratio, figure=fig)

    # Axes setup using advanced layout
    ax_iq = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0])
    ax_meta = fig.add_subplot(gs[:, 1])
    ax_meta.axis('off')  # Leave metadata panel blank for external injection

    # Convert x-axis to GHz and y-axis to dBm
    freq_GHz = freq / 1e9
    S21_dBm = 20 * np.log10(np.abs(S21))

    # Tick and spine styling parameters
    tick_kwargs = dict(
        width=settings['tick_width'],
        length=settings['tick_length'],
        pad=settings['tick_pad'],
        direction='in',
        labelsize=settings['fontsize']
    )

    # ax_iq plot formatting
    center_re  = np.mean(S21.real) + c_adj[0]
    center_im  = np.mean(S21.imag) + c_adj[1]
    span_re    = np.max(S21.real) - np.min(S21.real)
    span_im    = np.max(S21.imag) - np.min(S21.imag)
    span_max   = max(span_re, span_im)
    # Add margin (e.g., 10%)
    padding    = 0.2 * span_max
    half_width = 0.5 * span_max + padding

    # Raw IQ plot
    ax_iq.plot(S21.real, S21.imag, '.', color=settings['line_color'], 
               markersize=settings['marker_size'], alpha=alpha)
    ax_iq.set_title(filename, fontsize=settings['title_fontsize']-3, pad=settings['title_pad'])
    ax_iq.set_xlabel("Re(S21) arb. units", fontsize=settings['fontsize'], labelpad=settings['labelpad'])
    ax_iq.set_ylabel("Im(S21) arb. units", fontsize=settings['fontsize'], labelpad=settings['labelpad'])
    ax_iq.xaxis.set_major_locator(settings['x_locator'])
    ax_iq.yaxis.set_major_locator(settings['y_locator'])
    # Add faint x = 0 and y = 0 gridlines to circle plot
    ax_iq.axhline(0, color='lightgray', linestyle='--', linewidth=2, zorder=0)
    ax_iq.axvline(0, color='lightgray', linestyle='--', linewidth=2, zorder=0)
    ax_iq.set_xlim(center_re - half_width, center_re + half_width)
    ax_iq.set_ylim(center_im - half_width, center_im + half_width)
    ax_iq.tick_params(**tick_kwargs)
    ax_iq.set_aspect('equal')
    for spine in ax_iq.spines.values():
        spine.set_linewidth(settings['tick_width'])

    # Resonance plot with GHz and dBm scaling
    ax_res.plot(freq_GHz, S21_dBm, '.', color=settings['line_color'], 
                markersize=settings['marker_size'], alpha=alpha)
    ax_res.set_xlabel("Frequency (GHz)", fontsize=settings['fontsize'], labelpad=settings['labelpad'])
    ax_res.set_ylabel("S21 (dBm)", fontsize=settings['fontsize'], labelpad=settings['labelpad'])
    ax_res.xaxis.set_major_locator(settings['x_locator'])
    ax_res.yaxis.set_major_locator(settings['y_locator'])
    ax_res.tick_params(**tick_kwargs)
    for spine in ax_res.spines.values():
        spine.set_linewidth(settings['tick_width'])

    return fig, (ax_iq, ax_res), ax_meta



# Five panel plotting to show : circle fit, resonance curve, phase fit, metadata, and 
# quality factors.
def circle_plot(freq, S21, settings, fig_dim = (11, 16), filename = None,
                w_ratio = [6.66, 4.33], h_ratio = [1.6, 1, 0.25, 0.75], alpha=1,
                c_adj=[0,0], cf=None, scale=1, fit_color='darkorange',
                p0_guess = [7, 4.5165, 0.005, -50]):
    fig = plt.figure(figsize=fig_dim)
    gs = GridSpec(4, 2, width_ratios=w_ratio, height_ratios=h_ratio, figure=fig)

    # Axes setup using advanced layout
    ax_iq    = fig.add_subplot(gs[  0, 0])
    ax_res   = fig.add_subplot(gs[  1, 0])
    ax_phase = fig.add_subplot(gs[2:4, 0])

    ax_meta  = fig.add_subplot(gs[0:3, 1])
    ax_qfact = fig.add_subplot(gs[3:4, 1])
    ax_meta.axis('off')
    ax_qfact.axis('off')

    # Convert x-axis to GHz and y-axis to dBm
    freq_GHz = freq / 1e9
    S21_dBm = 20 * np.log10(np.abs(S21))
    S21_ang = np.angle(S21)

    # Tick and spine styling parameters
    tick_kwargs = dict(
        width=settings['tick_width'],
        length=settings['tick_length'],
        pad=settings['tick_pad'],
        direction='in',
        labelsize=settings['fontsize'])

    # ax_iq plot formatting
    center_re  = np.mean(S21.real) + c_adj[0]
    center_im  = np.mean(S21.imag) + c_adj[1]
    span_re    = np.max(S21.real) - np.min(S21.real)
    span_im    = np.max(S21.imag) - np.min(S21.imag)
    span_max   = max(span_re, span_im)
    # Add margin (e.g., 10%)
    padding    = 0.2 * span_max
    half_width = 0.5 * span_max + padding

    # IQ plot
    ax_iq.plot(S21.real, S21.imag, '.', color=settings['line_color'], 
               markersize=settings['marker_size'], alpha=alpha)
    ax_iq.set_title(filename, fontsize=settings['title_fontsize']-3, pad=settings['title_pad'])
    ax_iq.set_xlabel("Re(S21) arb. units", fontsize=settings['fontsize'], labelpad=settings['labelpad'])
    ax_iq.set_ylabel("Im(S21) arb. units", fontsize=settings['fontsize'], labelpad=settings['labelpad'])
    # Add faint x = 0 and y = 0 gridlines to circle plot
    ax_iq.axhline(0, color='lightgray', linestyle='--', linewidth=2, zorder=0)
    ax_iq.axvline(0, color='lightgray', linestyle='--', linewidth=2, zorder=0)
    ax_iq.set_xlim(center_re - half_width, center_re + half_width)
    ax_iq.set_ylim(center_im - half_width, center_im + half_width)
    if center_re - half_width < 0 < center_re + half_width:
        ax_iq.set_xticks([0])
    else:
        ax_iq.set_xticks([])
    if center_im - half_width < 0 < center_im + half_width:
        ax_iq.set_yticks([0])
    else:
        ax_iq.set_yticks([])
    ax_iq.tick_params(**tick_kwargs)
    ax_iq.set_aspect('equal')
    for spine in ax_iq.spines.values():
        spine.set_linewidth(settings['tick_width'])

    # Resonance plot
    ax_res.plot(freq_GHz, S21_dBm, '.', color=settings['line_color'], 
                markersize=settings['marker_size'], alpha=alpha)
    ax_res.set_ylabel("S21 (dBm)", fontsize=settings['fontsize'], labelpad=settings['labelpad'])
    ax_res.xaxis.set_major_locator(settings['x_locator'])
    ax_res.tick_params(**tick_kwargs)
    for spine in ax_res.spines.values():
        spine.set_linewidth(settings['tick_width'])
    ax_res.yaxis.set_ticks(rounded_ticks(S21_dBm,step=2))
    ax_res.set_ylim(rounded_ticks(S21_dBm,step=2)[0], rounded_ticks(S21_dBm,step=2)[-1])

    # Phase plot
    ax_phase.plot(freq_GHz, S21_ang, '.', color=settings['line_color'], 
                markersize=settings['marker_size'], alpha=alpha)
    ax_phase.set_xlabel("Frequency (GHz)", fontsize=settings['fontsize'], labelpad=settings['labelpad'])
    ax_phase.set_ylabel("Phase (rad)", fontsize=settings['fontsize'], labelpad=settings['labelpad'])
    ax_phase.xaxis.set_major_locator(settings['x_locator'])
    ax_phase.tick_params(**tick_kwargs)
    for spine in ax_phase.spines.values():
        spine.set_linewidth(settings['tick_width'])

    ax_phase.yaxis.set_ticks(rounded_ticks(S21_ang*10,step=2)/10)
    ax_phase.set_ylim(rounded_ticks(S21_ang*10,step=2)[0]/10, rounded_ticks(S21_ang*10,step=2)[-1]/10)

    # Add fit plot if circle fit data is provided
    if cf:
        ax_iq.plot(cf.z_data_sim.real, cf.z_data_sim.imag, 
                   color=fit_color, lw=settings['line_width'])
        ax_res.plot(freq_GHz, 20*np.log10(cf.z_data_sim_norm*scale), 
                color=fit_color, lw=settings['line_width'])
        ax_phase.plot(freq_GHz, np.angle(cf.z_data_sim_norm), 
                color=fit_color, lw=settings['line_width'])
      
    # Use the Inv_Lor fit for finding a quality factor
    dat   = np.abs(S21)
    popt  = fit_data(freq_GHz, dat, 'lorz_inv', user_guess=p0_guess)[0]
    q_li = popt[1] / popt[2] # Lorentzian inverse quality factor
    q_lor_val = f"{q_li:.0f}" if 'q_li' in locals() and q_li is not None else "—"

    # Quality factor metadata
    qdat = [
        ["f_res", format_scaled(cf.fitresults["fr"]) if cf else "—"],
        ["Q_cf", format_val(cf.fitresults["Ql"], precision=0) if cf else "—"],
        ["Q_lor", format_val(q_li, precision=0) if q_li is not None else "—"],
        ["fwhm_cf", format_val(cf.fitresults["fr"]/cf.fitresults["Ql"]/1e6, "MHz", 3)
            if cf and 'fr' in cf.fitresults and 'Ql' in cf.fitresults else "—"],
        ["Qi_cf", format_val(cf.fitresults["Qi"], precision=0) if cf else "—"],
        ["Qc_cf", format_val(cf.fitresults["Qc"], precision=0) if cf else "—"],
    ]

    return fig, (ax_iq, ax_res, ax_phase), ax_meta, ax_qfact, qdat

# Table populating function for circle fit results (or lack thereof)
def safe_cf_value(cf, key, fmt="{:.3f}", default="—"):
    try:
        val = cf.fitresults[key]
        return fmt.format(val)
    except (TypeError, AttributeError, KeyError):
        return default
    
# Rounded ticks function for setting axis ticks of size 2
def rounded_ticks(data, step=1):
    """Round min down and max up, then return evenly spaced ticks."""
    dmin, dmax = np.min(data), np.max(data)
    lower = np.floor(dmin)
    upper = np.ceil(dmax)
    if upper-lower % step == 0:
        lower -= 1
    count = int( (upper - lower)/ step) + 1  # Inclusive range
    return np.linspace(lower, upper, count)
