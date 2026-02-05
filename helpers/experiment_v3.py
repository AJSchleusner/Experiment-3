"""

Author: Austin J. Schleusner
Date: 2025-July-1

Description: This script was made to assist data acquisition for experiments on the Helios
fridge. This is loosely a development on my "experiment_v2.py" script and Camille Mikolas'
"experiment_acq_CM3.py" script. Both of these scripts in themselves were based on code by
Niyaz Beysengulov. My past script ("experiment_v2.py") was used for the high-frequency
Bragg-Cherenkov experiment but needed updating for collecting data on Camille's channel and
resonator device. My hope is that much of this code can be used again when I start
experiments on my next generation device (the SQUILL device). In general, this script can
run measurments on the Yokogawas and Lock-ins as well as on the VNA. As has beem the case
with the eHe scripts in the past, we use SQLite databases to store data and GPIB
communication to control and read most instruments.  It is worth noting that Copilot was
very helpful in explaining to me different elements of this code and how I could improve
upon the previous scripts. Though I want to be clear that I am not a scrub that blindly
asks AI to write my code for me.  At least not yet.

"""

###########################################################################################
## Imports ----------------------------------------------------------------------------- ##
###########################################################################################

# Here we import the necessary libraries and modules to run these experiments.  If an error
# is thrown here, it likely means the module needs to be added to the kernel.

import numpy as np
import warnings
import os.path
import os
import sys
import io
import logging
import pyvisa
import sqlite3
import json
import pandas as pd
import time
import threading
import itertools


from contextlib import redirect_stdout, redirect_stderr
from time import sleep, strftime, time, localtime
from sklearn import base
from tabulate import tabulate
from tqdm import tqdm
from typing import Dict, Sequence, Callable, Tuple, Optional, List
from newinstruments.BlueFors import BlueFors
bluefors = BlueFors()
from plot_setup import live_plot as lp
from matplotlib import pyplot as plt
from IPython.display import display, HTML, clear_output


###########################################################################################
## Connections ------------------------------------------------------------------------- ##
###########################################################################################

# This is where the code will attempt to connect to the vector network analyzer (VNA), the
# Pfeiffer DPG202 pressure gauge, the two lock-in amplifiers (SR830, SR844), the two
# Agilent 33500 sources, and the many Yokogawas. If these instruments are not connected,
# the code will continue without them, but the absense will be noted in the table.

# Instrument connection summary
instrument_status = []
instruments = {}

# Import the instrument drivers 
from newinstruments.vna_N5230A import *
from newinstruments.DPG202 import DPG202
from newinstruments.BlueFors import BlueFors
from newinstruments.mcc_daq import *
from pymeasure.instruments.srs import sr830, sr844
from pymeasure.instruments.yokogawa import yokogawa7651 as y7651
from pymeasure.instruments.yokogawa import yokogawaGS200 as ygs
from pymeasure.instruments.agilent import Agilent33500
import serial
import sys
import io

# to still implement
#from newinstruments.SignalHound import SignalHoundSA124B
#from newinstruments.SignalCore import SignalCore
#from newinstruments.HP8648B import *
#from newinstruments.bncRF import *

# Check available serial ports
def port_check():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"{port.device}: {port.description}")

# Instantiate function to create a device object and check connectivity
def instantiate(name, cls, address=None, serial=None, test_attr=None, 
                scpi_test=None, printing=False):
    try:
        # Create the device object with the given class, address, and serial
        if cls.__name__ == "N5230A":
            device = cls(serial=serial, address=address)
        else:
            device = cls(address)

        # Force a live SCPI query if provided
        if scpi_test:
            response = device.ask(scpi_test)
            if not response or "ERROR" in response.upper():
                raise ValueError(f"No valid response to SCPI query: {response}")
        # Start-up of the Yokogawa GS200 and 7651 power supplies
        if isinstance(device, (ygs.YokogawaGS200, y7651.Yokogawa7651)):
            try:
                # Check if the device is already enabled
                if device.source_enabled:
                    # If the device is already enabled, preserve the voltage
                    if printing:
                        print(f"[{name}] output already enabled — preserving voltage")
                # If the device is not enabled, set the source voltage to 0.0 V
                else:
                    device.source_voltage = 0.0
                    # If the device is a 7651, enable the source this way
                    if isinstance(device, y7651.Yokogawa7651):
                        device.enable_source()
                    # Otherwise, the device is a GS200 and enable the source this way
                    else:
                        device.source_enabled = True
                    if printing:
                        print(f"[{name}] output was off — set to 0.0 V and enabled output")
            except Exception as e:
                if printing:
                    print(f"[{name}] failed to check or enable output: {e}")
        # Start-up of the Agilent 33500 function generators
        if isinstance(device, Agilent33500):
            # Set the output to 'off'
            device.output = 'off'
            if printing:
                print(f"[{name}] output disabled at startup")

        # If a test attribute is provided, check its connectivity
        if test_attr:
            attr = getattr(device, test_attr)
            if callable(attr):
                response = attr()
                if not response or "ERROR" in str(response).upper():
                    raise ValueError(f"{name} failed test_attr call: {response}")
            else:
                # If it's a property, just access it
                _ = attr

        # If the device is successfully created and connected, append its status as True
        if cls.__name__ == "N5230A":
            instrument_status.append([name, True, device.address])
        else:
            instrument_status.append([name, True, address])
        return device
    except Exception as e:
        # If the device cannot be created or connected, append its status as False
        instrument_status.append([name, False, address])
        # If instantiation fails, close the device if it has a close() method
        if 'device' in locals():
            try:
                device.close()
                if printing:
                    print(f"[{name}] Closed device after failure.")
            except Exception:
                if printing:
                    print(f"[{name}] Could not close device after failure.")

        # If printing is enabled, print the error message
        if printing:
            print(f"[{name}] Connection failed: {e}")
        return None

# This function connects to all the instruments used in the experiment. It creates global
# variables for each instrument so that they can be accessed throughout the script and in
# the Jupyter notebook. There is also the option to print the 
def connect_instruments(printing=False, force_reconnect=False):
    global instruments
    # Close previous instruments if force_reconnect is True
    if force_reconnect:
        for inst in instruments.values():
            if hasattr(inst, "close"):
                try:
                    inst.close()
                except:
                    pass
    # Clear previous instruments and status    
    instruments.clear()
    instrument_status.clear()


    # All of the instruments in this experiment (at the moment)
    instruments["vna13"]     = instantiate(    "VNA 120",           N5230A, serial = "MY46400271",     test_attr="get_id", printing=printing)
    instruments["vna20"]     = instantiate(    "VNA 220",           N5230A, serial = "MY45000241",     test_attr="get_id", printing=printing)
    instruments["lockin_LF"] = instantiate(      "SR830",      sr830.SR830, "GPIB0::10::INSTR",        test_attr="status", printing=printing)
    instruments["lockin_HF"] = instantiate(      "SR844",      sr844.SR844, "GPIB0::11::INSTR",     test_attr="frequency", printing=printing)
    instruments["dpg"]       = instantiate(     "DPG202",           DPG202,     address="COM4",     printing=printing)      # No test_attr needed 
    instruments["daq"]       = instantiate_daq("MCC DAQ",          mcc_daq,                         printing=printing)      # No address needed
    instruments["bluefors"]  = instantiate_bf("BlueFors",         BlueFors,                         printing=printing)
    # Yokogawa GS200 and 7651 power supplies
    instruments["yoko_rch"]  = instantiate( "Yoko (rch)", ygs.YokogawaGS200, "GPIB0::7::INSTR",   test_attr="source_mode", printing=printing)
    instruments["yoko_lgd"]  = instantiate( "Yoko (lgd)", ygs.YokogawaGS200, "GPIB0::1::INSTR",   test_attr="source_mode", printing=printing)
    instruments["yoko_rgt"]  = instantiate( "Yoko (rgt)", ygs.YokogawaGS200,"GPIB0::21::INSTR",   test_attr="source_mode", printing=printing)
    instruments["yoko_mgt"]  = instantiate( "Yoko (mgt)", ygs.YokogawaGS200, "GPIB0::6::INSTR",   test_attr="source_mode", printing=printing)
    instruments["yoko_pin"]  = instantiate( "Yoko (pin)",y7651.Yokogawa7651,"GPIB0::24::INSTR",test_attr="source_voltage", printing=printing)
    instruments["yoko_res"]  = instantiate( "Yoko (res)",y7651.Yokogawa7651,"GPIB0::25::INSTR",test_attr="source_voltage", printing=printing)
    # Agilent 33500 function generators
    instruments["gen_sign"]  = instantiate("33500B (sign)",    Agilent33500,"GPIB0::19::INSTR",         test_attr="shape", printing=printing)
    instruments["gen_fila"]  = instantiate("33500B (fila)",    Agilent33500,"GPIB0::17::INSTR",         test_attr="shape", printing=printing)

    # Inject into notebook globals
    globals().update(instruments)
    # Return the full dictionary for optional use
    return instruments

# This function returns the controls for the sweep and step parameters and is used
# extensively in the init_reads function.
def get_controls(source, keys):
    return {key: source.get(key) for key in keys}

# This function connects to the MCC DAQ device
def instantiate_daq(name, cls, printing=False):
    try:
        device = cls()
        if not printing:
            # Suppress internal prints during device_detect()
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                device.device_detect()
            finally:
                sys.stdout = original_stdout
        else:
            device.device_detect()
        instrument_status.append([name, True, "USB Box"])
        return device
    except Exception as e:
        instrument_status.append([name, False, "USB Box"])
        if printing:
            print(f"[{name}] Connection failed: {e}")
        return None

# This function connects to the BlueFors temperature readout device
def instantiate_bf(name, cls, printing=False):
    try:
        device = cls()
        # Optional: try a temperature read to confirm it's responsive
        try:
            _ = device.get_temperature(1)  # Channel 1 should always exist
        except Exception as e:
            raise Exception(f"{name} failed temperature query: {e}")
        instrument_status.append([name, True, "USB Direct"])
        return device
    except Exception as e:
        instrument_status.append([name, False, "USB Direct"])
        if printing:
            print(f"[{name}] Connection failed: {e}")
        return None

# Global variable for VNA
vna = None

# This function initializes all instruments and displays their connection status
def init_instruments(printing=False, Table=True, force_reconnect=False):
    # Set the instruments as global variables
    global instruments, instrument_status, vna
    # Clear previous instrument status for fresh check
    instrument_status.clear()
    # Connect to all instruments
    instruments = connect_instruments(printing=printing, force_reconnect=force_reconnect)
    # Inject into notebook globals
    globals().update(instruments)
    # Determine which VNA to use (prioritize vna13 over vna20)
    if instruments.get("vna13") and hasattr(instruments["vna13"], 'address'):
        vna = instruments["vna13"]
    elif instruments.get("vna20") and hasattr(instruments["vna20"], 'address'):
        vna = instruments["vna20"]
    else:
        vna = None
    if Table:
        show_connection_table(instrument_status)
    return instruments


###########################################################################################
## Connection Table -------------------------------------------------------------------- ##
###########################################################################################

# This function displays the connection status of the instruments in an HTML table with the
# intention of providing a status update at the end of the ipynb imports.

def show_connection_table(status_list):
    # Create the header row of the HTML table
    html = """
    <table style="font-size: 20px; border-collapse: collapse; font-family: monospace;">
        <tr>
            <th class="header">Instrument</th>
            <th class="header">Connected</th>
            <th class="header">Address</th>
        </tr>

    <style>
        .header {
            border: 1px solid #ccc;
            padding: 8px;
            text-align: center;
            font-weight: bold;
        }
    </style>
    """
    # Loop through the status list and create an HTML row for each instrument
    for name, connected, address in status_list:
        # Set the color to green if connected, red if not
        color = "#4CAF50" if connected else "#F44336"
        # Use a check mark for connected and a cross for not connected
        symbol = "✔" if connected else "✘"
        # Set the style for the HTML table cells
        td_style = 'border:1px solid #ccc; padding:6px; text-align:center;'
        # Create the HTML row with the instrument name, symbol, and address
        html += f"""
        <tr>
            <td style="{td_style}">{name}</td>
            <td style="{td_style}; color: {color}; font-weight: bold;">{symbol}</td>
            <td style="{td_style}">{address}</td>
        </tr>
        """
    html += "</table>"
    display(HTML(html))


###########################################################################################
## Database Class ---------------------------------------------------------------------- ##
###########################################################################################

class generalDB:
    # Initialize the database connection
    def __init__(self, filename: str, schema: dict[str, dict]):
        self.conn    = sqlite3.connect(filename)
        self.cursor  = self.conn.cursor()
        self.schema  = schema
        self._make_tables()
    # Create the individual tables based on the schema dictionary
    def _make_tables(self):
        for table, defn in self.schema.items():
            if not defn['params']:  # Skip tables with no columns
                continue
            cols = ", ".join(f'"{p}" {t}' for p, t in zip(defn['params'], defn['types']))
            self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")
        self.conn.commit()
    # Insert multiple rows into a specified table
    def insert_many(self, table: str, rows: list[tuple]):
        ph = ",".join(["?"] * len(rows[0]))
        self.cursor.executemany(f"INSERT INTO {table} VALUES({ph})", rows)
        self.conn.commit()
    # Select all rows from a specified table
    def select_all(self, table: str):
        self.cursor.execute(f"SELECT * FROM {table}")
        return self.cursor.fetchall()
    # Select with a WHERE clause
    def select_where(self, table: str, where: str, params: tuple = ()):
        self.cursor.execute(f"SELECT * FROM {table} WHERE {where}", params)
        return self.cursor.fetchall()
    # Close the database connection
    def close(self):
        self.conn.close()

def log_filament_metadata(db: generalDB, filament_obj, daq_obj, success: bool = None,
                          deltaT: str = None):
    # Flatten metadata
    filament_meta = {
        "amplitude": filament_obj.amplitude,
        "offset": filament_obj.offset,
        "frequency": filament_obj.frequency,
        "shape": filament_obj.shape,
        "duty_cycle": filament_obj.square_dutycycle,
        "polarity": filament_obj.polarity,
        "burst_mode": filament_obj.burst_mode,
        "burst_ncycles": filament_obj.burst_ncycles,
        "burst_state": filament_obj.burst_state,
        "trigger_source": filament_obj.trigger_source,
        "trigger_delay": filament_obj.trigger_delay,
        "output_polarity": filament_obj.output_polarity,
        "output_state": filament_obj.output
    }

    daq_meta = {
        "measurement_time": daq_obj.measurement_time,
        "sampling_rate": daq_obj.sampling_rate,
        "voltage_range": daq_obj.voltage_range,
        "status": daq_obj.status
    }

    # Format rows for insertion
    rows = []
    rows += [("filament", key, str(val)) for key, val in filament_meta.items()]
    rows += [("daq", key, str(val)) for key, val in daq_meta.items()]
    rows.append(("general", "datetime", strftime("%Y-%m-%d %H:%M:%S", localtime())))
    rows.append(("general", "deltaT", str(deltaT)))
    rows.append(("general", "success", str(success)))

    # Insert into database
    db.insert_many('filament_metadata', rows)


# Schema for filament firing experiments database structure
filament_schema = {
    'filament_metadata': {
        'params': ['category', 'key', 'value'],
         'types': ['TEXT', 'TEXT', 'TEXT']
    },
    'general_metadata': {
        'params': ['category', 'key', 'value'],
        'types': ['TEXT', 'TEXT', 'TEXT']
    },
    'daq_data': {
        'params': ['time', 'filament_volt', 'lockin_x'],
         'types': ['TEXT', 'REAL', 'REAL']
    },
    'transport_test': {
        'params': ['channel_volt', 'lockin_x', 'lockin_y'],
         'types': ['REAL', 'REAL', 'REAL']
    },
    'figures': {
        'params': ['fig_name', 'image_blob'],
         'types': ['TEXT', 'BLOB']
    }
}

def write_to_database(db: generalDB, sweep_data: list, step_indices: Optional[list] = None):
    """
    Efficiently insert sweep_data into the database with optional step indexing.
    Compatible with generalDB.
    """
    def sanitize_row(row):
        clean = []
        for val in row:
            if isinstance(val, complex):
                clean.extend([val.real, val.imag])
            else:
                clean.append(val)
        return tuple(clean)

    all_rows = []
    for sweep_index, row in enumerate(sweep_data):
        step_index = step_indices[sweep_index] if step_indices is not None else 0
        full_row = (step_index, sweep_index) + tuple(row)
        sanitized = sanitize_row(full_row)
        all_rows.append(sanitized)

    # Use generalDB's insert_many method
    db.insert_many('table_data', all_rows)


###########################################################################################
## Data Management --------------------------------------------------------------------- ##
###########################################################################################

# This is where we define the functions that will be used to create the sweep lists and
# store data in the SQLite database.

# This creates the list of values being swept over.  The list can be linear or log.
def create_sweep_list(s1=0, s2=1, num=10, scale='linear'):
    try:
        if scale == 'linear':
            sweep_list = np.linspace(s1, s2, num, endpoint=True)
        elif scale == 'log':
            sweep_list = np.logspace(np.log10(s1), np.log10(s2), num, endpoint=True)
        else:
            raise ValueError('Scale must be linear or log')
        return sweep_list
    except:
        print('Sweep list inputs must be: low bound, high bound, point number, scale')
        return None

# Split the filename and suffix (digit), this is a private variable (__)
def __split_name_suffix(name: str):
    # Start at the last character of the name string
    i = len(name) - 1
    # Loop backwards through the name until we find a character that is not a digit
    while i >= 0 and name[i].isdigit():
        i -= 1
    # Earlier characters are the base name
    base = name[:i+1]
    # Later characters are the suffix
    suffix = name[i+1:]
    # Return the suffix as an integer if it exists, otherwise return 0
    return base, int(suffix) if suffix else 0

# Below creates the path for storing the data and the filename to a SQLite database. This
# will automatically create the filename with the date in a folder with the date.
def create_path_filename(measurement_name: str, overwrite: bool = False) -> str:
    # Get the date-stamp for directory organization
    date_str = strftime("%Y-%m-%d")
    # Join this date-stamped directory with the larger 'data' file
    subdir = os.path.join('data', date_str)
    # Make a new directory only if it does not already exist
    os.makedirs(subdir, exist_ok=True)
    # Split the numeric suffix from the measurement name
    base, suf = __split_name_suffix(measurement_name)

    # If overwrite is True, return the first filename without checking
    if overwrite:
        filename = f"{date_str}_{base}.db" if suf == 0 else f"{date_str}_{base}-{suf}.db"
        filepath = os.path.join(subdir, filename)
        print(f"Overwrite enabled: using {filename}")
        return filepath

    # Otherwise, find the next available filename
    while True:
        filename = f"{date_str}_{base}-{suf}.db" if suf > 0 else f"{date_str}_{base}.db"
        filepath = os.path.join(subdir, filename)
        if not os.path.isfile(filepath):
            break
        suf += 1

    # If the suffix is greater than 0, print a message indicating a filename conflict    
    if suf > 0:
        print(f"Avoiding overwrite, new filename: {filename}")
    # Return the full file path
    return filepath

# Scale units to their more readable form. This is primarily used for the 
# tabulate table that displays the class attributes. It will convert units like Hz to
# MHz, kHz, or GHz depending on the value.
def scale_units_to_readable(value, unit):
    # Define the scaling factors and their corresponding units
    scales = {
        'Hz':   [(1e9,  'GHz'), (1e6, 'MHz'), (1e3, 'kHz')],
        'V':    [(1e-3,  'mV'), (1e0, 'V')],
        'A':    [(1e-3,  'mA'), (1e0, 'A')],
        's':    [(1e-9,  'ns'), (1e-6, 'µs'), (1e-3, 'ms'), (1e0, 's')],
        'K':    [(1e-3,  'mK'), (1e0, 'K')],
        'dBm':  [(1e0,  'dBm')],
        'Vpp':  [(1e-3,'mVpp'), (1e0, 'Vpp')],
        'pts':  [(1e0,  'pts')],
        'avgs': [(1e0, 'avgs')]}
    # If the value is a tuple (like the frequency values)...
    if isinstance(value, tuple):
        try:
            # ... iterate through the scaling factors and their corresponding units.
            for factor, new_unit in scales.get(unit, []):
                # scale the values within the tuple by the factor
                scaled = tuple(v / factor for v in value)
                # If all scaled values are within the range of 1 to 1000...
                if all(1 <= abs(v) < 1000 for v in scaled):
                    # ... return it with the new unit formatted to 3 significant figures
                    return f"({scaled[0]:.3g}, {scaled[1]:.3g}) {new_unit}"
        except Exception:
            return str(value)
        return f"{value} {unit}"
    # If the value is an integer or a float...
    if isinstance(value, (int, float)):
        # ... iterate through the scaling factors and their corresponding units.
        for factor, new_unit in scales.get(unit, []):
            # scale 'value' by the factor
            scaled = value / factor
            # If the scaled value is within the range of 1 to 1000...
            if 1 <= abs(scaled) < 1000:
                # ... return it with the new unit formatted to 3 significant figures
                return f"{scaled:.3g} {new_unit}"
        return f"{value:.3g} {unit}"  # fallback
    # For everything else, just return the value as it was entered
    return value  

# Apply the control to the device whether it is a method or attribute.
def apply_control(device, method_name, value):
    attr_or_method = getattr(device, method_name)
    # If it is calleable, it is a method
    if callable(attr_or_method):
        attr_or_method(value)
    # If it is not callable, it is an attribute
    else:
        setattr(device, method_name, value)


###########################################################################################   
## VNA Format Class -------------------------------------------------------------------- ##
###########################################################################################

class VNAFormatRouter:
    def __init__(self, format_key):
        self.format_key = format_key.strip().upper()
        self.format_map = {
            'MLOG': {
                'scpi': 'MLOG',
                'mode': 'scalar',
                'read_dict': lambda vna: {'vna_mag': [vna, 'read_data_y', 'dB']},
                'reshape': lambda data, vna: np.hstack((
                    np.array(data).T,
                    vna.get_fpoints()[:, None])),
                'columns': ['vna_mag', 'vna_freq'],
                'units':   ['dB', 'Hz']
            },
            'PHAS': {
                'scpi': 'PHAS',
                'mode': 'scalar',
                'read_dict': lambda vna: {'vna_phase': [vna, 'read_data_y', 'deg']},
                'reshape': lambda data, vna: np.hstack((
                    np.array(data).T,
                    vna.get_fpoints()[:, None])),
                'columns': ['vna_phase', 'vna_freq'],
                'units':   ['deg', 'Hz']
            },
            'POL': {
                'scpi': 'POL',
                'mode': 'polar',
                'read_dict': lambda vna: {
                    'vna_mag': [vna, 'read_data_y', 'dB'],
                    'vna_phase': [vna, 'read_data_y', 'deg']
                },
                'reshape': lambda data, vna: np.hstack((
                    np.array(data).T, vna.get_fpoints()[:, None])),
                'columns': ['vna_mag', 'vna_phase', 'vna_freq'],
                'units': ['dB', 'deg', 'Hz']
            },
            'SDATA': {
                'scpi': None,
                'mode': 'complex',
                'read_dict': lambda vna: {
                    'vna_complex': [vna, 'read_complex', '']
                },
                'reshape': lambda data, vna: np.hstack((
                    np.vstack((data.real, data.imag)).T,
                    vna.get_fpoints()[:, None])),
                'columns': ['vna_real', 'vna_imag', 'vna_freq'],
                'units':   ['', '', 'Hz']
            }
        }

        self.config = self.format_map.get(self.format_key, self.format_map['MLOG'])

    # This method applies the SCPI format to the VNA instrument, if one is present
    def apply_format(self, vna):
        scpi_format = self.config['scpi']
        if scpi_format:
            vna.set_format(scpi_format)

    def get_read_dict(self, vna):
        return self.config['read_dict'](vna)

    def reshape_data(self, data, vna):
        return self.config['reshape'](np.asarray(data), vna)

    def get_mode(self):
        return self.config['mode']

    def get_columns(self):
        return self.config.get('columns', ['vna_data'])

    def get_units(self):
        return self.config.get('units', [''])

    def get_metadata_dict(self):
        return {
            'format': self.format_key,
            'mode': self.get_mode(),
            'columns': self.get_columns(),
            'units': self.get_units()
        }

























###########################################################################################   
## Experiment Class -------------------------------------------------------------------- ##
###########################################################################################

# The experiment class is used for controlling instruments and running experiments. As a 
# general note, variables defined in the class are prefixed with "__" to make them private,
# and should not be accessed directly. Instead, we use the class methods to access and
# modify them.

class exp3():
    # Comments show up in the experimental parameters table
    comment1 = None
    comment2 = None
    comment3 = None
    tconst   = 0.5


    #######################################################################################
    ## General Setup Definitions ------------------------------------------------------- ##
    #######################################################################################

    # The "initializer method" is used at the start of every class and sets up instance
    # attributes and other necessary parameters.
    def __init__(self, 
                 ctrl_instrument: dict, 
                  vna_instrument: dict, 
                 read_instrument: dict,
             bluefors_instrument: dict,
             instrument_registry: dict):
        # Set the private variables for the control and readout of instruments.
        self.__reads = read_instrument
        self.__ctrls = ctrl_instrument
        # Set the private variables for the temperature readout.
        self.__bluefors = bluefors_instrument
        # Set the private variable for the VNA, this needs to be before being 
        # called in the reset_instruments method.
        self.__vna = vna_instrument
        # Set the private variable for the instrument registry, this is used to
        # look up instrument objects by name.
        self.__registry = instrument_registry
        # Reset the instruments to their default states
        self.reset_instruments()

    # This method is used to set the class attributes. It overrides the default
    def __setattr__(self, name, value):
        # Route to control dictionary if key exists
        if hasattr(self, '_exp3__ctrls') and name in self.__ctrls:
            self.__ctrls[name][0] = value
            self.__apply_param(self.__ctrls, name)
        # Route to VNA dictionary
        elif hasattr(self, '_exp3__vna') and name in self.__vna:
            self.__vna[name][0] = value
            self.__apply_param(self.__vna, name)
        # Route to a standard attribute
        else:
            super().__setattr__(name, value)

    # This method is used to get the class attributes. It overrides the default
    def __apply_param(self, source: dict, key: str):
        entry = source[key]
        value = entry[0]  # The value to set
        device = entry[1]  # The instrument object
        # Adjust method index based on length
        method = entry[3] if len(entry) > 4 else entry[2]
        if hasattr(device, method):
            try:
                args = value if isinstance(value, (tuple, list)) else (value,)
                    # If the method is 'ramp_to_voltage', check if the device has a
                    # source_voltage method.
                if method == 'ramp_to_voltage' and hasattr(device, 'source_voltage'):
                    # If the device has a source_voltage, check if the current voltage
                    # is already close to the target voltage (args[0]). If so, skip ramp.
                    current_voltage = device.source_voltage
                    if abs(current_voltage - args[0]) < 1e-6:
                        return  # skip redundant ramp
                # Otherwise apply the method with the arguments
                getattr(device, method)(*args)
            except Exception as e:
                print(f"[{key}] Error applying: {value} → {method} on {device}: {e}")

    # This method prints the class attributes in a table format using the tabulate library.
    def table(self):
        print(tabulate(self.get_ClassAttributes(), 
                       headers=['Attributes', 'Values'], 
                       tablefmt='simple'))

    # This method resets the instruments to their default states by pulling the
    # attributes.
    def reset_instruments(self):
        # If the vna is attached...
        if self.__vna:
            try: 
                # Pull the VNA instrument attributes to set as keys named "Vkey"
                for Vkey in self.__vna:
                    setattr(self, Vkey, self.__vna[Vkey][0])
            except:
                # {e} is a placeholder for the exception (error) message
                print('Problem initializing VNA keys: {e}')
        # If the vna is not attached, print the below message
        else:
            print('No VNA instrument to initialize')
        # Search for key in the control instruments dict
        for Ckey in self.__ctrls:
            # create class attributes to store all control parameters
            setattr(self, Ckey, self.__ctrls[Ckey][0])
        self.table()  # Print the class attributes in a table format
        
    # This creates a printable table of all the instrument control attributes.
    def get_ClassAttributes(self) -> list:
        # Create empty attribute list to store the attributes
        attr_list = []
        # To avoid duplicates
        seen = set()  
        # Add units from control instrument and vna instrument dictionary
        for source in [self.__ctrls, self.__vna]:
            for key, item in source.items():
                # Avoid duplicates here
                if key not in seen:
                    try:
                        val = item[0]
                        unit = item[-1]
                        display_val = scale_units_to_readable(val, unit)
                        attr_list.append([key, display_val])
                    except Exception as e:
                        attr_list.append([key, f"Error: {e}"])
        # Include the BlueFors temperature attributes
        try:
            for label, (frid, func, unit) in self.__bluefors.items():
                # Get the value of the BlueFors instrument attribute
                val = func(frid)
                display_val = scale_units_to_readable(val, unit)
                attr_list.append([label, display_val])
        except:
            pass
        # Add the comments to the attribute list if they aren't 'None'
        for comment_field in ['comment1', 'comment2', 'comment3']:
            if hasattr(self, comment_field):
                value = getattr(self, comment_field)
                if value not in [None, "None"]:
                    attr_list.append([comment_field, value])
        # Return the list of attributes
        return attr_list

    # Initialization of instruments with the user defined parameters
    def instr_init(self, ramp_time = 2) -> None:
        # initializing the yokogawas
        for key in list(self.__ctrls.keys()):
            attr_value = getattr(self, key)
            link = self.__ctrls[key]
            if type(attr_value) is dict:
                val = attr_value.get('val') - attr_value.get('off')
            else:
                val = attr_value
            set_instrument = getattr(link[1], link[3])
            try:
                set_instrument(val, ramp_time)
            except:
                set_instrument(val)
        # initializing the vna
        try:
            for key in list(self.__vna.keys()):
                attr_value = getattr(self, key)
                link = self.__vna[key]
                set_instrument = getattr(link[1], link[2])
                if not isinstance(attr_value, (list, tuple)):
                    set_instrument(attr_value)
                else:
                    set_instrument(*attr_value)
        except:
            print('no VNA instrument to initialize')
        sleep(ramp_time)
        try:
            for key in self.__vna:
                getattr(self.__vna[key][0], "auto_scale")(channel=1)
        except:
            pass
        print('instruments are initialized!')



    #######################################################################################
    ## Electron Deposition ------------------------------------------------------------- ##
    #######################################################################################

    # Prepare electrodes for electron deposition
    def deposition_prep(self, Vch=0.8, Vres=1.2, Vgt=-0.4, Vpn=-0.2, Vac=0.1) -> None:
        # Turn on transport drive 
        self.__registry['gen_sign'].output    = 'on'
        self.__registry['gen_sign'].amplitude = Vac
        # Ramp electrode voltages
        self.__registry['yoko_pin'].ramp_to_voltage( Vpn, duration=1)
        self.__registry['yoko_res'].ramp_to_voltage(   0, duration=1)
        self.__registry['yoko_lgd'].ramp_to_voltage( Vgt, duration=1)
        self.__registry['yoko_rgt'].ramp_to_voltage( Vch, duration=1)
        self.__registry['yoko_mgt'].ramp_to_voltage( Vch, duration=1)
        self.__registry['yoko_rch'].ramp_to_voltage( Vch, duration=1)
        # Empty reservoirs
        self.__registry['yoko_res'].enable_source()
        self.__registry['yoko_res'].ramp_to_voltage(-0.8, duration=1)
        sleep(5)
        # Ramp reservoir voltage
        self.__registry['yoko_res'].ramp_to_voltage(Vres, duration=1)
        sleep(1)

    # Prepare electrodes for a transport sweep
    def transport_prep(self, Vch=0.0, Vres=0.7, Vgt=-0.4, Vpn =-0.2) -> None:
        # Ramp electrode voltages
        self.__registry['yoko_pin'].ramp_to_voltage( Vpn, duration=1)
        self.__registry['yoko_res'].ramp_to_voltage(Vres, duration=1)
        self.__registry['yoko_lgd'].ramp_to_voltage( Vgt, duration=1)
        self.__registry['yoko_rgt'].ramp_to_voltage( Vch, duration=1)
        self.__registry['yoko_mgt'].ramp_to_voltage( Vch, duration=1)
        self.__registry['yoko_rch'].ramp_to_voltage( Vch, duration=1)
        sleep(2)

    def deposit_electrons(self) -> tuple:
        self.__registry['gen_fila'].output = 'on'
        print('filament turned on')
        sleep(0.3)
        self.__registry['gen_fila'].trigger()
        print('scan started')
        output = self.__registry['daq'].scan()
        self.__registry['gen_fila'].output = 'off'
        print('filament turned off')
        return output
    
    def zero_instruments(self) -> None:
        self.__registry['gen_sign'].output = 'off'
        self.__registry['gen_sign'].amplitude = 0.01
        # Ramp electrode voltages
        self.__registry['yoko_pin'].ramp_to_voltage( 0, duration=1)
        self.__registry['yoko_res'].ramp_to_voltage( 0, duration=1)
        self.__registry['yoko_lgd'].ramp_to_voltage( 0, duration=1)
        self.__registry['yoko_rgt'].ramp_to_voltage( 0, duration=1)
        self.__registry['yoko_mgt'].ramp_to_voltage( 0, duration=1)
        self.__registry['yoko_rch'].ramp_to_voltage( 0, duration=1)
        

    @staticmethod
    def normalize_daq(output):
        """
        Accepts any of the following and returns (t, ch1, ch2) as float arrays:
        1) ndarray with shape (N, >=3): columns = [time, ch1, ch2, ...]
        2) (t, ch1, ch2): three 1D arrays (same length)
        3) (t, Y): where Y has shape (N, >=2), columns [ch1, ch2, ...]
        """
        # Case 2 or 3: tuple/list forms
        if isinstance(output, (list, tuple)):
            # 2) (t, ch1, ch2)
            if len(output) == 3:
                t, ch1, ch2 = map(np.asarray, output)
                if t.ndim != 1 or ch1.ndim != 1 or ch2.ndim != 1:
                    raise ValueError("Expected (t, ch1, ch2) to be 1D arrays.")
                if not (len(t) == len(ch1) == len(ch2)):
                    raise ValueError("t, ch1, ch2 must have the same length.")
                t = t.astype(float); ch1 = ch1.astype(float); ch2 = ch2.astype(float)
                if not np.all(np.diff(t) > 0):
                    raise ValueError("Time column must be strictly increasing.")
                return t, ch1, ch2
            # 3) (t, Y) where Y is (N, >=2)
            if len(output) == 2:
                t, Y = output
                t = np.asarray(t).astype(float)
                Y = np.asarray(Y)
                if Y.ndim != 2 or Y.shape[1] < 2:
                    raise ValueError("Expected Y with shape (N, >=2) for ch1,ch2 in (t, Y).")
                ch1 = Y[:, 0].astype(float)
                ch2 = Y[:, 1].astype(float)
                if not np.all(np.diff(t) > 0):
                    raise ValueError("Time column must be strictly increasing.")
                if len(t) != len(ch1):
                    raise ValueError("t and Y must have the same number of rows.")
                return t, ch1, ch2
        # 1) Already a 2D array with at least 3 columns
        arr = np.asarray(output)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            t = arr[:, 0].astype(float)
            ch1 = arr[:, 1].astype(float)
            ch2 = arr[:, 2].astype(float)
            if not np.all(np.diff(t) > 0):
                raise ValueError("Time column must be strictly increasing.")
            return t, ch1, ch2
        # If we get here, the shape is unknown → print helpful diagnostics
        shapes = None
        if isinstance(output, (list, tuple)):
            shapes = [(np.shape(x), type(x)) for x in output]
        raise ValueError(f"Unsupported DAQ output format. "
                        f"type={type(output)}, shape={np.shape(output)}, parts={shapes}")


    # Search the phase space of deposition parameters
    def deposition_sweep(
        self,
        *,
        experiment_name: str,
        sweep_axes: Dict[str, Sequence[float]],             # e.g., {"amp": [...], "frq": [...]}
        setters: Dict[str, Callable[[float], None]],        # param -> callable(value) that applies just that param
        datasaver,                                          # <-- REQUIRED: external DataSaver instance
        daq,                                                # <-- REQUIRED: external DAQ handle (if not on self)
        trigger_fn: Optional[Callable[[], object]] = None,  # Default to self.deposit_electrons
        prep_fn: Optional[Callable[[dict], None]] = None,   # Reset the electron system
        settle_s: float =  5.0,                             # Settle after cleaning
        pause_s: float = 100.0,                             # Wait between firing loops
        overwrite: bool = False,
        extra_metadata: Optional[Dict[str, str]] = None,
        verbose: bool = False,                              # show small status in tqdm postfix
        quiet_trigger: bool = True,                         # suppress prints inside trigger_fn
        default_Vres: float = 1.0,                          # used if prep_fn is None and Vres not swept
        ):

        """
        N-D sweep with explicit dependency injection for datasaver and daq.
        Assumes DAQ returns [time, ch1, ch2]. Saves long-form rows:
        [<swept params>..., 'Time', 'Ch1', 'Ch2'].
        """

        # Set default trigger
        if trigger_fn is None:
            if not hasattr(self, "deposit_electrons"):
                raise ValueError("No trigger_fn provided and expr has no deposit_electrons().")
            trigger_fn = self.deposit_electrons

        # Prepare sweep axes
        axes_order  = list(sweep_axes.keys())
        axes_values = [list(sweep_axes[name]) for name in axes_order]

        # Validate that all sweep axes have corresponding setters where
        # setters: param name -> callable(value)
        missing = [p for p in axes_order if p not in setters]
        if missing:
            raise ValueError(f"Missing setters for: {missing}")

        # Prepare to collect all columns and rows
        columns = axes_order + ["Time", "Ch1", "Ch2"]
        all_rows: List[Tuple[float, ...]] = []
        time_axis_ref = None
        ref_len = None

        # Prepare metadata
        metadata = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "axes_order": ",".join(axes_order),
            "daq_sampling_rate": getattr(daq, "sampling_rate", None),
            "daq_measurement_time": getattr(daq, "measurement_time", None),
        }
        # Add extra metadata if provided
        if extra_metadata:
            metadata.update({str(k): str(v) for k, v in extra_metadata.items()})

        # Total number of points in sweep(s)
        total_pts = np.prod([len(v) for v in axes_values])

        # Progress bar
        with tqdm(total=total_pts, desc=experiment_name, unit="pt") as pbar:
             # Iterate over all combinations of sweep axis values
            for idx, point_vals in enumerate(itertools.product(*axes_values), start=1):

                # Compose the current point context: {"amp": 3.4, "frq": 2.0, ...}
                context = {name: float(val) for name, val in zip(axes_order, point_vals)}

                # 1) Apply only the parameters you asked to sweep
                for name, val in zip(axes_order, point_vals):
                    setters[name](float(val))

                # 2) Reset/prepare between iterations
                if prep_fn is not None:
                    prep_fn(context)
                else:
                    Vres_now = context.get("Vres", default_Vres)
                    self.deposition_prep(Vres=Vres_now, Vpn=-0.1, Vac=0.1)

                # 3)  Wait for settling time if specified
                if settle_s > 0:
                    time.sleep(settle_s)

                # 4) Acquire; silence prints inside trigger_fn if requested
                if quiet_trigger:
                    _buf_out, _buf_err = io.StringIO(), io.StringIO()
                    with redirect_stdout(_buf_out), redirect_stderr(_buf_err):
                        out = trigger_fn()
                else:
                    out = trigger_fn()

                # 5) Wait between iterations if specified
                if pause_s > 0 and idx < total_pts:
                    time.sleep(pause_s)

                # 6) Normalize DAQ output
                t, y1, y2 = self.normalize_daq(out)

                # 7) Align time length if needed
                if time_axis_ref is None:
                    time_axis_ref = t
                    ref_len = len(t)
                elif len(t) != ref_len:
                    N = min(len(t), ref_len)
                    t, y1, y2 = t[:N], y1[:N], y2[:N]

                # 8) Append rows
                head = [float(v) for v in point_vals]
                for ti, yi1, yi2 in zip(t, y1, y2):
                    all_rows.append(tuple(head + [float(ti), float(yi1), float(yi2)]))
                
                # 9) Update progress bar
                pbar.update(1)
                if verbose:
                    # show a compact postfix with current params
                    pbar.set_postfix({k: f"{v:.4g}" for k, v in context.items()})


        # Save to database
        filepath = datasaver.save_to_db(
            exp_name=experiment_name,
            columns=columns,
            sweep_data=all_rows,
            sweep_lists=None,
            step_lists=None,
            metadata=metadata,
            scheme=None,
            overwrite=overwrite
        )

        # Build 3D cubes if exactly two parameters swept
        cubes = None
        if len(axes_order) == 2:
            p1_vals, p2_vals = axes_values
            T = len(time_axis_ref)
            Z1 = np.empty((len(p1_vals), len(p2_vals), T), dtype=float)
            Z2 = np.empty_like(Z1)
            block = 0
            for i, _ in enumerate(p1_vals):
                for j, _ in enumerate(p2_vals):
                    start = block * T
                    stop  = start + T
                    sl = np.asarray(all_rows[start:stop], dtype=float)
                    Z1[i, j, :] = sl[:, -2]
                    Z2[i, j, :] = sl[:, -1]
                    block += 1
            cubes = (np.asarray(p1_vals, float),
                     np.asarray(p2_vals, float),
                     np.asarray(time_axis_ref, float),
                     Z1, Z2)

        if verbose:
            tqdm.write(f"Saved sweep to {filepath}")
        return filepath, axes_order, axes_values, time_axis_ref, cubes


















    #######################################################################################
    ## Sweep and Step Definitions ------------------------------------------------------ ##
    #######################################################################################

    # Private variable to return the sweep information as a string
    def __sweep_info(self, sw_type, s1, s2, num, scale, offset):
        return f"{sw_type} / {s1:,} : {s2:,} / num={num} / {scale} / off={offset:,}"
    
    # Private variable to identify the object type and return its value
    def __obj_type(self, obj, index):
        if type(obj) is list:
            obj_value = obj[index]
        else:
            obj_value = obj
        return obj_value

    # Variable to set up the start, end, scale type, and offset for the sweep or step
    def control_variables(self, control_type = 'sweep', var = 'None', s1 = 0, s2 = 1,
                          num = 1, scale = 'linear', offset = 0):
        """
        setting up how the instruments will be controlled throughout an experiment
        loop/step/sweep.
        Parameters: 
        control_type: str
            Either 'sweep' or 'step'
        var: None or single string or list of strings 
            The instrument(s) to control during the sweep or step
        s1: int or float
            Start value of the sweep
        s2: int or float
            End value of the sweep
        num: int
            Number of points in the sweep or step
        scale: str
            Linear or logarithmic scale
        offset: int or float
            Offset value to subtract from the sweep or step values
        """
        # If var is not a list, convert it to a list
        if type(var) is not list:
            var = [var]
        # Make an empty control list
        control_lists = []
        # Check that the number of offsets matches the number of variables
        if var and len(var) != len(offset):
            print('''
                  Problem: The number of offsets does not match the number of variables
                  ''')
            return
        # Append sweep lists to the control list for each variable
        for i in range(len(var)):
            first      = self.__obj_type(s1, i)
            final      = self.__obj_type(s2, i)
            scale_type = self.__obj_type(scale, i)
            off        = self.__obj_type(offset, i)
            sweep_list = create_sweep_list(first, final, num, scale_type) + off
            control_lists.append(sweep_list)
            # Store the sweep information in a separate log
            if not hasattr(self, '_sweep_info_log'):
                self._sweep_info_log = {}
            self._sweep_info_log[var[i]] = self.__sweep_info(control_type, first, final,
                                                              num, scale_type, off)
        # Return the variable names, control lists, and number of points. Since I removed
        # the '__indexing_parameters' function, I need to convert 'num' to a list.
        return var, control_lists, [num]
    
    # Public methods to set sweep parameters
    def sweep_params(self, **kwargs) -> None:
        var, control_lists, num = self.control_variables(control_type='sweep', **kwargs)
        self.__sweep = {
            'variable' : var,
            'sweep lists' : control_lists,
            'num points' : num
        }

    # Public methods to set step parameters
    def step_params(self, **kwargs) -> None:
        if kwargs is None:
           var, control_lists, num = ['None'], [[0]], 1
        else: 
            var,control_lists,num = self.control_variables(control_type='step', **kwargs)
        self.__step = {
            'variable' : var,
            'step lists' : control_lists,
            'num points' : num
        }


    #######################################################################################
    ## SQL Database Definitions -------------------------------------------------------- ##
    #######################################################################################
   

    def build_meas_schema(self, measured_columns: list[str]) -> dict:
        sweep_vars = self.__sweep.get('variable')
        step_vars = self.__step.get('variable') if self.__step.get('variable') else []

        schema = {
            'table_sweep': {
                'params': sweep_vars,
                'types': ['REAL'] * len(sweep_vars)
            },
            'table_step': {
                'params': step_vars,
                'types': ['REAL'] * len(step_vars)
            },
            'table_data': {
                'params': ['step_index', 'sweep_index'] + measured_columns,
                'types': ['INTEGER', 'INTEGER'] + ['REAL'] * len(measured_columns)
            },
            'metadata': {
                'params': ['category', 'key', 'value'],
                'types': ['TEXT', 'TEXT', 'TEXT']
            }
        }
        return schema
    

    def save_experiment(self, exp_name: str, sweep_data: list, step_indices: list, savedata: bool = True):
        if not savedata:
            print("Data collected but not saved to database")
            return

        # Create file path and schema
        filepath = create_path_filename(exp_name)
        schema = self.build_meas_schema()

        # Create the database
        db = generalDB(filepath, schema)

        # Insert sweep data
        sweep_lists = self.sweep.get('sweep lists')
        sweep_rows = [tuple(sweep_lists[j][i] for j in range(len(sweep_lists)))
                    for i in range(len(sweep_lists[0]))]
        db.insert_many('table_sweep', sweep_rows)

        # Insert step data
        step_lists = self.step.get('step lists')
        if step_lists:
            step_rows = [tuple(step_lists[j][i] for j in range(len(step_lists)))
                        for i in range(len(step_lists[0]))]
            db.insert_many('table_step', step_rows)

        # Insert measurement data
        db.insert_many('table_data', sweep_data)

        # Insert metadata
        meta_rows = [('experiment', key, str(value)) for key, value in self.metadata]
        db.insert_many('metadata', meta_rows)

        db.close()
        print("Data saved to database")


    #######################################################################################
    ## VNA ----------------------------------------------------------------------------- ##
    #######################################################################################

    # Function to set the VNA wait time based on averaging state and sweep time
    def vna_wait_time(self, vna=None):
        if vna is None:
            vna = self.__vna['instrument']
        # Check if the vna is averaging (True) or not (False)
        vna_avg = vna.get_average_state()
        # Check how many averages are being taken by the vna
        vna_num_avgs = vna.get_averages()
        # If no averaging is occurring...
        if not vna_avg:
            # ... set the sleep time to the sweep time of the vna
            vsleep  = vna.get_sweep_time(channel = 1)
            avstate = False
        # If averaging is occurring...
        elif vna_avg:
            # ... set the sleep time to the sweep time multiplied by the number of averages
            vsleep  = vna.get_sweep_time(channel = 1) * vna_num_avgs
            avstate = True
        # Automatically scale the VNA y axis 
        vna.auto_scale(channel = 1)
        return avstate, vsleep
    
    # Prepare the VNA for measurement by setting the averaging state and waiting for the
    # sweep to complete. 
    def vna_meas_wait(self,vna=None):
        if vna is None:
            vna = self.__vna['instrument']
        avs, vsleep = self.vna_wait_time(vna)
        # Toggle the average state of the VNA to ensure it is ready for the next
        # sweep.
        vna.set_average_state(not avs)
        vna.set_average_state(avs)
        # Let the VNA collect data for the length of time necessary for a full sweep
        # (with possible averaging). The factor of 1.05 is just a safety factor to ensure
        # the VNA has enough time to collect the data we specifically want. 
        sleep(1.05*vsleep)

    # This function pulls the VNA data from the read_dict and returns it as a list.
    def pull_vna_data(self, read_dict):
        # Create an empty list to store the VNA data
        vna_arr = []
        for key, value in read_dict.items():
            device, method, unit = value
            # Get the method or property from the device
            attr = getattr(device, method)
            # Call it if it's a method
            data = attr() if callable(attr) else attr
            vna_arr.append(data)
        return vna_arr


    # This function performs a single acquisition from the VNA with optional
    # frequency range and number of points settings.
    def _acquire_vna_trace_once(self, vna: N5230A, *, read_dict, format_router,
                                start=None, stop=None, num_pts=None,
                                channel=1, use_averaging=None):
        """
        Deterministic single acquisition:
        - If use_averaging is an int > 1, do an averaged sweep (one trigger runs all N sweeps).
        - If None or 1, do exactly one sweep.
        Configures optional start/stop/points if provided, then triggers and waits.
        Returns: (freq[:,1], vna_arr[:,k]) with k depending on your format_router (e.g., 2 for Re/Im)
        """

        # Optional frequency plan
        if (start is not None) and (stop is not None):
            vna.set_frequency_range(start, stop, channel=channel)
        if num_pts is not None and hasattr(vna, "set_sweep_points"):
            vna.set_sweep_points(int(num_pts), channel=channel)

        # RF on
        if hasattr(vna, "set_output"):
            vna.set_output(True)

        # Hold (no continuous)
        if hasattr(vna, "set_trigger_continuous"):
            vna.set_trigger_continuous(False)  # :INIT:CONT OFF

        # Averaging behavior
        if use_averaging is None:
            # honor current average state/count, but run deterministically
            avg_count = vna.get_averages(channel=channel)
            avg_on = vna.get_average_state(channel=channel) and (avg_count > 1)
        else:
            # force what the caller requested
            if int(use_averaging) > 1:
                vna.set_averages(int(use_averaging), channel=channel)
                vna.set_average_state(True, channel=channel)
                avg_on = True
                avg_count = int(use_averaging)
            else:
                vna.set_average_state(False, channel=channel)
                vna.set_averages(1, channel=channel)
                avg_on = False
                avg_count = 1

        # Clear previous averaging accumulation
        if hasattr(vna, "clear_averages"):
            vna.clear_averages(channel=channel)

        # If averaging is ON, let a single trigger complete the entire average
        if hasattr(vna, "set_trigger_average_mode"):
            vna.set_trigger_average_mode(avg_on)  # :TRIG:AVER ON/OFF

        # One trigger starts either one sweep (avg OFF) or N-sweep average (avg ON)
        vna.trigger_single(channel=channel)

        # Wait for completion deterministically
        if avg_on:
            # Wait until averaging complete bit flips (STAT:OPER:AVER?)
            # (averaging_complete returns True when done)
            while not vna.averaging_complete(channel=channel):
                time.sleep(0.05)
            # As a final fence, OPC ensures everything has settled
            vna.get_operation_completion()
        else:
            # One sweep: just wait OPC
            vna.get_operation_completion()

        # Pull vector data using your existing router+read path
        vna_arr_raw = self.pull_vna_data(read_dict)
        vna_arr = format_router.reshape_data(vna_arr_raw, vna)

        # Frequency axis: prefer driver method if you have it; else compute from start/stop/points
        if hasattr(vna, "get_fpoints"):
            try:
                freq_pts = vna.get_fpoints()  # prefer bound call if available
                freq_index = np.asarray(freq_pts).reshape(-1, 1)
            except TypeError:
                # some drivers require channel or self passed; fallback
                freq_pts = vna.get_fpoints(vna)
                freq_index = np.asarray(freq_pts).reshape(-1, 1)
        else:
            if (start is None) or (stop is None):
                # fall back to channel start/stop getters
                start = vna.get_start_frequency(channel=channel)
                stop = vna.get_stop_frequency(channel=channel)
            if num_pts is None:
                if hasattr(vna, "get_sweep_points"):
                    num_pts = vna.get_sweep_points(channel=channel)
                else:
                    raise RuntimeError("No get_fpoints/get_sweep_points available to build frequency axis.")
            freq_index = np.linspace(start, stop, int(num_pts), endpoint=True)[:, None]

        return freq_index, vna_arr


    #######################################################################################
    ## Transport ----------------------------------------------------------------------- ##
    #######################################################################################

    # This function pulls data from the sr830 (LF) or sr844 (HF) and returns it as a list.
    def pull_lockin_data(self, read_dict, target_labels):
        # Create an empty list to store the lockin data
        lockin_arr = []
        # Loop through all read_dict items
        for label in target_labels:
            instr, method, units = read_dict[label]
            # Call the method on the instrument to get the data
            value = getattr(instr, method)
            # If the value is callable, call it to get the data
            data = value() if callable(value) else value
            # Append the data to the lockin_arr list
            if isinstance(data, (tuple, list)):
                lockin_arr.extend(data)
            else:
                lockin_arr.append(data)
        # Return both the x and y data from the selected lock-in amplifier
        return lockin_arr


    #######################################################################################
    ## Querying Control Instruments ---------------------------------------------------- ##
    #######################################################################################

    # This function checks if the instrument is a transport instrument or a vna instrument.
    def query_control_instr(self, instr):
        # Return 'transport' if the instrument is in the transport control instruments
        # dictionary.
        if instr in self.__ctrls:
            return 'transport'
        # Return 'vna' if the instrument is in the vna instruments dictionary
        elif instr in self.__vna:
            return 'vna'
        # If the instrument is in the reads dictionary, check its type
        elif instr in self.__reads:
            # Create a string on the instrument's address
            inst_repr = repr(self.__reads[instr][0])
            # Check if the string representation contains 'vna'
            if "vna" in inst_repr:
                return 'vna'
            else:
                return 'transport'
        # If not in any dictionary, return: 
        else:
            return 'Instrument not in dictionaries'


    #######################################################################################
    ## Initialize For an Experiment Run ------------------------------------------------ ##
    #######################################################################################

    def init_reads(self, printing=False, format_router=None):
        # Since the instrument 'vna' is called explcitly in this function, this check was
        # added to ensure that the vna instrument is named exactly 'vna' before being
        # called to in this function.  If it is not defined, an error is raised.

        if not self.__vna:
            raise RuntimeError('''
                               Instrument 'vna' must be defined before calling init_reads()
                               ''')
        
        # The first VNA is defined as the variable 'vna' for later use
        vna_key = list(self.__vna.keys())[0]
        vna = self.__vna[vna_key][0]


        ###################################################################################
        ## Load Instruments ------------------------------------------------------------ ##
        ###################################################################################

        # Load in the sweep instrument from the sweep parameters
        try:
            instrument_sw = list(self.__sweep['variable'])
        except AttributeError:
            print('Problem: \'sweep_params\' has not been defined')
            return # End the function here if AttributeError occurs
        # Check whether the sweep instument is a 'transport' or 'vna' instrument
        sweep_type = self.query_control_instr(instrument_sw[0])
        
        # Load in the step instrument from the step parameters
        try:
            instrument_st = list(self.__step['variable'])
        except AttributeError:
            print('Problem: \'step_params\' has not been defined')
            return # End the function here if AttributeError occurs
        # Check whether the step instrument is a 'transport' or 'vna' instrument
        try:
            step_type = self.query_control_instr(instrument_st[0])
        except:  # There may not be a step instrument used
            step_type = None

        # Convert string to a router object
        if format_router:
            format_router = VNAFormatRouter(format_router) 



        ###################################################################################
        ## Sweep and Step Combinations ------------------------------------------------- ##
        ###################################################################################

        # 1D transport sweep ------------------------------------------------------------ #
        if sweep_type == 'transport' and step_type is None:
            # Create a dictionary of only the read instruments that do not include 'vna.'
            read_keys = [key for key in self.__reads if 'vna' not in key]
            read_dict = {key: self.__reads.get(key) for key in read_keys}
            read_list = list(read_dict.keys())
        # 1D vna sweep ------------------------------------------------------------------ #
        elif sweep_type == 'vna' and step_type is None:
            # Create a dictionary of only the read instruments that include 'vna'
            read_keys = [key for key in self.__reads if 'vna' in key]
            read_dict = format_router.get_read_dict(vna)
            read_list = list(read_dict.keys())
        # 2D transport sweep ------------------------------------------------------------ #
        elif sweep_type == 'transport' and step_type == 'transport':
            # Create a dictionary of only the read instruments that do not
            # include 'vna.'
            read_keys = [key for key in self.__reads if 'vna' not in key]
            read_dict = {key: self.__reads.get(key) for key in read_keys}
            read_list = list(read_dict.keys())
        # 2D hybrid sweep --------------------------------------------------------------- #
        elif sweep_type == 'vna' and step_type == 'transport':
            # Create a dictionary of all read instruments
            transport_dict = {key: self.__reads.get(key) for key in self.__reads if 'vna' not in key}
            vna_dict = format_router.get_read_dict(vna)
            read_dict = {**transport_dict, **vna_dict}
            read_list = list(read_dict.keys())
        # Error condition --------------------------------------------------------------- #
        else:
            print('''
            Problem: sweep_type and step_type must be \'transport\', \'vna\' or none.
            Also, \'vna\' is not setup as a step option, only a sweep.
            ''')
            return


        ###################################################################################
        ## Sweep and Step Variables ---------------------------------------------------- ##
        ###################################################################################

        # VNA sweep variable ------------------------------------------------------------ #
        if sweep_type == 'vna':
            # If the sweep variable 'freq_range' appears...
            if 'freq_range' in self.__sweep['variable']:
                # startf is the first entry in the sweep list
                startf  = self.__sweep['sweep lists'][0][0]
                # stopf is the last entry (index = [-1]) in the sweep list
                stopf   = self.__sweep['sweep lists'][0][-1]
                # num_pts is the number of points in the sweep
                num_pts = self.__sweep['num points']
                # Setup frequency range and sweep points for entry to the vna
                self.__vna['freq_range'] = [(startf, stopf), vna, 'set_frequency_range']
                self.__vna['sweep_pts'] = [(num_pts), vna, 'set_sweep_points']
                # sweep_controls is a dictionary of the sweep control inputs
                sweep_controls = get_controls(self.__vna,['freq_range','sweep_pts'])
            else:
                # Otherwise, set the sweep variable as the sweep_control
                sweep_controls = get_controls(self.__vna, self.__sweep['variable'])
            # Get the sweep lists and number of points from the sweep parameters
            sweep_lists = self.__sweep.get('sweep lists')
            num_sweep_points = self.__sweep.get('num points')

        # VNA step variable ------------------------------------------------------------- #
        if step_type == 'vna':
            # If the step variable 'freq_range' appears...
            if 'freq_range' in self.__step['variable']:
                # startf is the first entry in the step list
                startf  = self.__step['sweep lists'][0][0]
                # stopf is the last entry (index = [-1]) in the step list
                stopf   = self.__step['sweep lists'][0][-1]
                # num_pts is the number of points in the step
                num_pts = self.__step['num points']
                # Setup frequency range and step points for entry to the vna
                self.__vna['freq_range'] = [(startf, stopf), vna, 'set_frequency_range']
                # The vna driver uses 'sweep' as opposed to step, so it is kept as
                # 'sweep' here.
                self.__vna['sweep_pts'] = [(num_pts), vna, 'set_sweep_points']
                # step_controls is a dictionary of the step control inputs
                step_controls = get_controls(self.__vna, ['freq_range','sweep_pts'])
            else:
                # Otherwise, set the step variable as the step_control
                step_controls = get_controls(self.__vna, self.__step['variable'])
            # Get the step lists and number of points from the step parameters
            step_lists = self.__step.get('step lists')
            num_step_points = self.__step.get('num points')

        # Transport sweep variable ------------------------------------------------------ #
        if sweep_type == 'transport':
            # Get the sweep controls from the sweep parameters
            sweep_controls = get_controls(self.__ctrls, self.__sweep['variable'])
            # Get the sweep lists and number of points from the sweep parameters
            sweep_lists = self.__sweep.get('sweep lists')
            num_sweep_points = self.__sweep.get('num points')

        # Transport step variable ------------------------------------------------------- #
        if step_type == 'transport':
            # Get the step controls from the step parameters
            step_controls = get_controls(self.__ctrls, self.__step['variable'])
            # Get the step lists and number of points from the step parameters
            step_lists = self.__step.get('step lists')
            num_step_points = self.__step.get('num points')

        # None step variable ------------------------------------------------------------ #
        if step_type is None:
            # If there is no step variable, set the step controls to an empty dictionary
            step_controls = {}
            # Set the step lists and number of points to None
            step_lists = None
            num_step_points = None
       
        # Print variables for diagnositics ---------------------------------------------- #
        if printing:
            print('Read dictionary:', read_dict)
            print('Read list:', read_list)
            print('Sweep type:', sweep_type)
            print('Step type:', step_type)
            print('Sweep controls:', sweep_controls)
            print('Step controls:', step_controls)
            print('Sweep lists:', sweep_lists)
            print('Step lists:', step_lists)

        # Return the variables for run experiment --------------------------------------- #
        return (
            read_dict, 
            read_list,
            sweep_type,
            step_type, 
            sweep_controls, 
            step_controls, 
            sweep_lists, 
            step_lists, 
            num_sweep_points, 
            num_step_points
            )
    

    #######################################################################################         
    ## Generic VNA Sweep --------------------------------------------------------------- ##
    #######################################################################################
    
    def run_vna_sweep(
        self,
        vna,                            # the earlier defined vna instrument
        num_sweep_points,               # number of VNA frequency points
        sweep_var: str,                 # 'power', 'elec_delay', or 'none'
        sweep_list: list = None,        # list of values to sweep over (if applicable)
        sweep_controls: dict = None,    # e.g. {'power': (val0, instr, task)}
        read_dict: dict = None,         # used in pull_vna_data
        step_val=0,                     # optional step tag for outer sweeps
        format_router=None):                   
        # The data from the vna sweep will be stored in this list
        sweep_data = []
        # Unpack the number of frequency points 
        num_pts = num_sweep_points[0]
        # Ensure VNA output is ON before measurement
        if hasattr(vna, "set_output"):
            vna.set_output(True)
        # Turn on the vna averaging
        if hasattr(vna, "set_average_state"):
            vna.set_average_state(True)

        
        # Handle the 'freq_range' keyword as a special no-sweep case
        if sweep_var == 'freq_range':

            # S1 will be the start frequency
            start_freq = sweep_list[0]
            # S2 will be the stop frequency
            stop_freq = sweep_list[-1]

            # Choose: one sweep or averaged result
            #   - One sweep per step: use_averaging=1
            #   - Honor current settings: use_averaging=None
            #   - Force N-sweep average: use_averaging=N

            freq_index, vna_arr = self._acquire_vna_trace_once(
                vna,
                read_dict=read_dict,
                format_router=format_router,
                start=start_freq,
                stop=stop_freq,
                num_pts=num_pts,
                channel=1,
                use_averaging=None)   # or 1 for exactly one sweep, or an int N

            # Build step index column(s)
            if isinstance(step_val, (list, tuple)):
                # 2D array (num_pts, n_step_vars)
                step_index = np.column_stack([np.full(num_pts, val) for val in step_val])
            else:
                # 2D column (num_pts, 1)
                step_index = np.full((num_pts, 1), step_val)

            # Ensure vna_arr is 2D with num_pts rows
            vna_arr = np.asarray(vna_arr)
            if vna_arr.ndim == 1:
                vna_arr = vna_arr.reshape(-1, 1)
            elif vna_arr.shape[0] != num_pts and vna_arr.shape[1] == num_pts:
                # If it came transposed, fix it
                vna_arr = vna_arr.T
            assert vna_arr.shape[0] == num_pts, f"vna_arr.rows={vna_arr.shape[0]} != num_pts={num_pts}"

            # Now safely hstack
            stacked = np.hstack((step_index, freq_index.reshape(-1, 1), vna_arr))

            # Turn off the VNA power
            if hasattr(vna, "set_output"):
                vna.set_output(False)
            return stacked.tolist()
        
        # Sweeping one parameter (power, delay, etc.)
        val0, instr, task, units = sweep_controls[sweep_var]
        for val in tqdm(sweep_list, desc=f'Sweeping {sweep_var}', leave=False):
            apply_control(instr, task, val)

            # Deterministic acquisition (choose averaging policy as above)
            freq_index, vna_arr = self._acquire_vna_trace_once(
                vna,
                read_dict=read_dict,
                format_router=format_router,
                channel=1,
                use_averaging=None,  # 1 for single-sweep per value; N for forced averaging
            )

            freq_index = freq_index[:,None] # Ensure freq_index is 2D
            sweep_index = np.full((len(freq_index), 1), val) 
            stacked = np.hstack((freq_index, sweep_index, vna_arr))
            sweep_data.extend(stacked.tolist())

        # Turn off the VNA power    
        if hasattr(vna, "set_output"):
            vna.set_output(False)
        return sweep_data


    #######################################################################################         
    ## Generic Transport Sweep --------------------------------------------------------- ##
    #######################################################################################

    def run_transport_sweep(
        self, 
        volt_lists, 
        sweep_controls, 
        select_read,
        sweep_order,
        step_val=0,
        hold_time=None
        ):
        # If the experiment isn't to be slowed down, use the time constant
        if hold_time is None:
            hold_time = {var: self.tconst for var in sweep_order}
        # All of the data from the sweep will be stored in this list
        sweep_data = []
        # Check if the 'volt_list' dictionary contains the required variables
        if set(volt_lists) != set(sweep_order):
            print(f"Problem: 'volt_lists' must contain the keys: {set(sweep_order)}")
            return
        # Check if the 'sweep_controls' dictionary contains the required variables
        if set(sweep_controls) != set(sweep_order):
            print(f"Problem: 'sweep_controls' must only contain the keys: {set(sweep_order)}")
            return   
        # Get the sweep voltage lists in the declared 'sweep_order' order
        sweep_lists = [volt_lists[var] for var in sweep_order]
        # Check if all voltage sweep lists are of equal length
        if not all(len(lst) == len(sweep_lists[0]) for lst in sweep_lists):
            print("Problem: All voltage sweep lists must be of equal length")
            return
        # Ramp each variable to its initial value
        for var in sweep_order:
            val0, instr, task, ramp_func, units = sweep_controls[var]
            getattr(instr, ramp_func)(volt_lists[var][0])    
        # Sleep for more than 2 second to allow the above ramping complete
        time.sleep(3)
        # Loop through the paired voltage values in the two lists
        for volts in tqdm(zip(*sweep_lists), desc=f'Sweeping {sweep_order}', leave=False):
            # Apply sweep controls to all sweep variables
            for var, val in zip(sweep_order, volts):
                val0, instr, task, ramp_func, units = sweep_controls[var]
                # Call the method or attribute on the instrument
                apply_control(instr, task, val)
            # Determine max hold time across all variables for this step
            max_hold = max(hold_time.get(var, self.tconst) for var in sweep_order)
            sleep(max_hold)
            # Pull the lock-in (HF) data from the select_read
            hf_values = self.pull_lockin_data(select_read, target_labels=['Vhfx', 'Vhfy'])
            # Normalize step_value: tuple → unpacked list; scalar → wrap as list
            step_tag = list(step_val) if isinstance(step_val, tuple) else [step_val]
            # Append the voltage and lock-in values to the sweep_data list
            sweep_data.append(step_tag + list(volts) + hf_values)
        return sweep_data


    #######################################################################################
    ## Generalized 2D Helper Function -------------------------------------------------- ##
    #######################################################################################

    def run_2d_helper(
        self,     
        step_map: dict,
        step_controls: dict,
        sweep_func: callable,   # Function being swept (e.g. self.run_transport)
        sweep_args: list,       # Arguments for the sweep function
        *,
        step_order: list,       # Order-dependent list of step variable name
        sweep_order: list,      # Order-dependent list of sweep variable name
        format_router=None,
        hold_time=None
        ):     
        # Create an empty list for the sweep data
        sweep_data = []
        # Create an empty list for step indices
        step_indices = []
        # Get the step list(s) for the step variable(s): This gives a list of tuples
        step_lists = [step_map[var] for var in step_order]
        # zip(*step_lists) combines the lists into tuples, where each tuple contains
        # one value from each list.  Use enumerate to index the step_lists.
        for step_idx, step_vals in tqdm(enumerate(zip(*step_lists)),
                                        total=len(step_lists[0]),
                                        desc=f'Stepping {step_order}',
                                        leave=True):
            # Step each variable in step_vars to the corresponding value in step_vals
            for var, val in zip(step_order, step_vals):
                # Unpack the step_controls for the variable
                val0, instr, task, ramp_func, units = step_controls[var]
                # Set the instrument to the value
                apply_control(instr, task, val)
            # Having set all of the instruments to their values for this step,...
            # Sleep to allow the instruments/setup to settle
            sleep(self.tconst)

            # Use format_router only if sweep_func expects it
            if format_router and sweep_func.__name__ == 'run_vna_sweep':
                out_data = sweep_func(
                    *sweep_args,
                    step_val=step_vals[0] if len(step_vals) == 1 else step_vals,
                    format_router=format_router)
            elif hold_time is not None:
                out_data = sweep_func(
                    *sweep_args,
                    step_val=step_vals[0] if len(step_vals) == 1 else step_vals,
                    hold_time=hold_time) 
            else:
                out_data = sweep_func(
                    *sweep_args,
                    step_val=step_vals[0] if len(step_vals) == 1 else step_vals)
                
            # Extend the sweep_data with the out_data
            sweep_data.extend(out_data)
            # Track step index for each row
            step_indices.extend([step_idx] * len(out_data))
        # Return the sweep data
        return sweep_data, step_indices


    #######################################################################################
    ## Run Experiment ------------------------------------------------------------------ ##
    #######################################################################################

    def run_experiment(self, exp_name = 'sweep_NA', savedata = False,
                       format_tag = 'MLOG', return_data = False, **kwargs):

        # Import definitions from the 'init_reads'
        (
        read_dict, 
        read_list,
        sweep_type,
        step_type, 
        sweep_controls, 
        step_controls, 
        sweep_lists, 
        step_lists, 
        num_sweep_points, 
        num_step_points
        ) = self.init_reads(printing=False, format_router=format_tag)

        # Begin with the sweep_data set as None
        sweep_data = None
        # Begin with the step_indices set as None
        step_indices = None
        # Create the ordered list of sweep variables 
        sweep_order = list(self.__sweep['variable'])
        # Create the ordered list of step variables 
        step_order = list(self.__step['variable'])
        # Specify the lock-in read dictionaries being used
        if sweep_type == 'transport':
            select_read = {key: read_dict[key] for key in ['Vhfx', 'Vhfy']}

        # Assign the sweep lists to their variables and then package as a dictionary
        sweep_map = dict(zip(self.__sweep['variable'], self.__sweep['sweep lists']))
        # Assign the step lists to their variables and then package as a dictionary
        step_map = dict(zip(self.__step['variable'], self.__step['step lists']))

        # If the sweep type is 'vna', create a VNAFormatRouter to handle the format
        if sweep_type == 'vna':
            router = VNAFormatRouter(format_tag)
            router.apply_format(vna)
            read_dict = router.get_read_dict(vna)


        ###################################################################################
        ## 1D Sweep Options ------------------------------------------------------------ ##
        ###################################################################################

        if step_type is None:

            # 1D VNA Sweep -------------------------------------------------------------- #
            if sweep_type == 'vna': 
                sweep_data = self.run_vna_sweep(
                    vna=vna,
                    num_sweep_points=num_sweep_points,
                    sweep_var=sweep_order[0],
                    sweep_list=sweep_map.get(sweep_order[0]),
                    sweep_controls=sweep_controls,
                    read_dict=read_dict,
                    format_router=router)
                            
            # 1D Transport Sweep -------------------------------------------------------- #
            elif sweep_type == 'transport':
                sweep_data = self.run_transport_sweep(
                    volt_lists=sweep_map,
                    sweep_controls=sweep_controls,
                    select_read=select_read,
                    sweep_order=sweep_order,
                    hold_time=kwargs.get('hold_time'))
            

        ###################################################################################
        ## VNA Step Options ------------------------------------------------------------ ##
        ###################################################################################

        elif step_type == 'vna':

            # 2D VNA Sweep -------------------------------------------------------------- #

            # Since the 'power' and 'elec_delay' variables are already 2D in the sense that
            # they are also observing a range of frequencies, the only combination that
            # couldbe used for a 2D sweep here is a sweep of power and elec_delay. But this
            # is a combination of things being swept that I will never use.  So there is no
            # need to write any functions for the sweep:VNA, step:VNA combination.

            # 2D Transport Sweep-VNA Step ----------------------------------------------- #

            # As is the general case, whatever instrument is being used as the 'sweep'
            # variable will determine what instrument is being used for data collection. If
            # the sweeping instrument is of the 'transport' type, then the data collection
            # will be done via the lock-in amplifier and if the sweeping instrument is of
            # the 'vna' type, then the data will be collected by the VNA.  This is important
            # here because there is not a use case for using the VNA as an input variable
            # without also wanting it to also collect data.  This is what would happen if 
            # the VNA was used as a step variable and one of the transport instruments was
            # used as the sweep variable.  Thus, this step: vna and sweep: transport
            # combination is not implemented in the code.

            print("""
                  Problem: VNA as a step variable is not presently supported.  The VNA
                  is useful for collecting data but not for stepping through.  As a general
                  rule, the data collection method corresponds to the sweep variable.  For
                  now, there are no known use cases for stepping the VNA and then observing
                  via the transport setup (i.e. the lock-in amplifier).  Similarly, the
                  combination of VNA as both the step and sweep type has not been
                  implemented since there is not a known use case.
                  """)
            return


        ###################################################################################
        ## Transport Step Options ------------------------------------------------------ ##
        ###################################################################################

        elif step_type == 'transport':

            # 2D Transport-VNA Sweep ---------------------------------------------------- #
            if sweep_type == 'vna':
                sweep_var=sweep_order[0]
                sweep_list=sweep_map.get(sweep_order[0])

                sweep_data, step_indices = self.run_2d_helper(
                    step_map=step_map,
                    step_controls=step_controls,
                    sweep_func=self.run_vna_sweep,
                    sweep_args=[vna, num_sweep_points, sweep_var,
                        sweep_list, sweep_controls, read_dict],
                    step_order=step_order,
                    sweep_order=sweep_order,
                    format_router=router)
            
            # 2D Transport-Transport Sweep ---------------------------------------------- #
            elif sweep_type == 'transport':  
                sweep_var=sweep_order
                # Us the run_2D_helper on the generic run_transport_sweep function
                sweep_data, step_indices = self.run_2d_helper(
                    step_map=step_map,
                    step_controls=step_controls,
                    sweep_func=self.run_transport_sweep,
                    sweep_args=[sweep_map, sweep_controls, select_read, sweep_order],
                    step_order=step_order,
                    sweep_order=sweep_order,
                    hold_time=kwargs.get('hold_time'))
 

        # If Unimplemented Step-Sweep is Called ----------------------------------------- #
        else:
            print(f"No sweep logic implemented for step: {step_type}, sweep: {sweep_type}")
            return


        ###################################################################################
        ## Saving Data ----------------------------------------------------------------- ##
        ###################################################################################
        
        # If no sweep_data was collected, print a message and return
        if sweep_data is None:
            print('Problem: No sweep data collected. Check the sweep and step parameters.')
            return

        # Create Column Headers
        if sweep_type == 'vna':
            router_meta = router.get_metadata_dict()
            read_cols = router_meta['columns']    
            col_units = router_meta['units'] 
            columns_with_units = [
                f"{col} [{unit}]" if unit else col
                for col, unit in zip(read_cols, col_units)
            ]
        elif sweep_type == 'transport':
            read_cols = list(select_read.keys())
            columns_with_units = read_cols
        else:
            print('Problem: sweep_type must be \'transport\' or \'vna\'')
            return

        # If sweeping power or elec_delay on the VNA, override the sweep_order
        if sweep_type == 'vna' and sweep_order[0] in ['power', 'elec_delay']:
            # prepend the 'freq_range' to the sweep_order
            sweep_order = ['freq_range'] + sweep_order
        elif sweep_order[0] == 'freq_range' and step_order == []:
            step_order = ['empty step']

        # Construct the full column list
        columns = step_order + sweep_order + columns_with_units

        # If it's a 1D sweep and sweep_data has an extra leading column, strip it
        if step_type is None and len(sweep_data[0]) == len(columns) + 1:
            sweep_data = [row[1:] for row in sweep_data]

        # Check that the sweep_data is a list of lists and that each row has the same
        # length as columns.
        for i, row in enumerate(sweep_data):
            if len(row) != len(columns):
                print(f"Row {i}: expected {len(columns)} cols, got {len(row)} → {row}")
        print(f"Non-index columns: {columns}")

        assert all(
            len(row) == len(columns)
            for row in sweep_data
        ), "Problem: Mismatch in data and column lengths"

        # If savedata is True, create a SQLite database and insert the data
        if savedata:
            # Create the database file path
            filepath = create_path_filename(exp_name)
            # Build the schema dynamically from the experiment class
            schema = self.build_meas_schema(columns)
            # Create the database
            db = generalDB(filepath, schema)

            # Insert sweep data
            sweep_lists = self.__sweep.get('sweep lists')
            sweep_rows = [tuple(sweep_lists[j][i] for j in range(len(sweep_lists)))
                        for i in range(len(sweep_lists[0]))]
            db.insert_many('table_sweep', sweep_rows)

            # Insert step data (if any)
            step_lists = self.__step.get('step lists')
            if step_lists:
                step_rows = [tuple(step_lists[j][i] for j in range(len(step_lists)))
                            for i in range(len(step_lists[0]))]
                db.insert_many('table_step', step_rows)

            # Insert measurement data
            write_to_database(db, sweep_data, step_indices)

            # Insert metadata in flat format
            meta_rows = [('experiment', key, str(value)) for key, value in self.get_ClassAttributes()]
            db.insert_many('metadata', meta_rows)

            # Close the database
            db.close()
            print('Data saved to database')
        else:
            print('Data collected but not saved to database')
        if return_data == True:
            return sweep_lists, step_lists, sweep_data
        return
    

###########################################################################################
## Data Saving Class ------------------------------------------------------------------- ##
###########################################################################################  

class DataSaver:
    def __init__(self, base_path="data"):
        self.base_path = base_path

    def save_to_db(self, exp_name, columns, sweep_data, sweep_lists=None, 
                   step_lists=None, metadata=None, scheme=None, overwrite=False):
        # Create file path
        filepath = create_path_filename(exp_name, overwrite=overwrite)

        # Build schema dynamically
        schema = self._build_schema(columns, scheme=scheme)

        # Create DB
        db = generalDB(filepath, schema)

        # Insert sweep data
        if sweep_lists:
            sweep_rows = [tuple(sweep_lists[j][i] for j in range(len(sweep_lists)))
                          for i in range(len(sweep_lists[0]))]
            db.insert_many('table_sweep', sweep_rows)

        # Insert step data
        if step_lists:
            step_rows = [tuple(step_lists[j][i] for j in range(len(step_lists)))
                         for i in range(len(step_lists[0]))]
            db.insert_many('table_step', step_rows)

        # Insert measurement data
        db.insert_many('table_measurements', sweep_data)

        # Insert metadata
        if metadata:
            meta_rows = [('experiment', key, str(value)) for key, value in metadata.items()]
            db.insert_many('metadata', meta_rows)

        db.close()
        print(f"Data saved to {filepath}")
        return filepath

    # Load data from database
    def load_table_meas(self, filepath):
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM table_measurements")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def _build_schema(self, columns, scheme=None):
        if scheme is None:
            params = columns
            types = ['REAL' for _ in columns]  # Default all REAL
        elif scheme == 'dpg':
            params = columns
            types = [
                'REAL' if col in ['ElapsedSeconds', 'Pressure1', 'Pressure2'] else 'TEXT'
                for col in columns
            ]
        else:
            raise ValueError(f"Select None or 'dpg' for scheme. Got: {scheme}")

        schema = {
            'table_measurements': {
                'params': params,
                'types': types
            },
            'table_sweep': {'params': [], 'types': []},
            'table_step': {'params': [], 'types': []},
            'metadata': {
                'params': ['category', 'key', 'value'],
                'types': ['TEXT', 'TEXT', 'TEXT']
            }
        }
        return schema


###########################################################################################
## VNA class --------------------------------------------------------------------------- ##
###########################################################################################  

class VNALogger:
    def __init__(self, vna, dpg, datasaver, exp_name, columns, save_interval=300,
                 overwrite=False):
        # Initialize instruments
        self.vna = vna
        self.dpg = dpg
        # Initialize parameters
        self.datasaver = datasaver
        self.exp_name = exp_name
        self.columns = columns
        self.save_interval = save_interval
        self.data = []
        self.stop_flag = False
        self.start_time = None
        self.last_save_time = None
        self.loop_counter = 0
        self.filepath = None
        self.overwrite = overwrite 

    # Wait for user input to stop recording
    def wait_for_stop(self):
        input("Press Enter to stop...\n")
        self.stop_flag = True

    # Start recording data
    def vna_log(self, wait_time=10, record_pressure=True):
        self.start_time = time.time()
        self.last_save_time = time.time()
        threading.Thread(target=self.wait_for_stop, daemon=True).start()
        print("Recording started...")

        while not self.stop_flag:
            # Measure VNA data
            time.sleep(wait_time)
            complex_data = self.vna.read_complex(channel=1)
            real_part = np.real(complex_data)
            imag_part = np.imag(complex_data)
            fpts = self.vna.get_fpoints()

            if record_pressure:
                # Measure pressures
                pressure1 = self.dpg.measure('Pressure1')
                pressure2 = self.dpg.measure('Pressure2')
            else:
                pressure1 = None
                pressure2 = None

            timestamp = time.time()
            elapsed = round(timestamp - self.start_time, 2)
            timestamp_str = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')

            # Append sweep rows
            sweep_rows = [
                (elapsed, timestamp_str, pressure1, pressure2, float(f), float(r), float(i))
                for f, r, i in zip(fpts, real_part, imag_part)
            ]
            self.data.extend(sweep_rows)

            # Periodic save
            if time.time() - self.last_save_time >= self.save_interval:
                print("Saving data to database:", len(self.data), "records")
                self.filepath = self.datasaver.save_to_db(
                    self.exp_name, self.columns, self.data,
                    scheme='dpg', overwrite=self.overwrite)

                self.last_save_time = time.time()

        # Final save
        print("Final save...")
        self.filepath = self.datasaver.save_to_db(
            self.exp_name, self.columns, self.data,
            scheme='dpg', overwrite=self.overwrite)
        print("Recording stopped.")
