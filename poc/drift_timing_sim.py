#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 06 3:06 PM 2025
Created in PyCharm
Created as saclay_micromegas/drift_timing_sim.py

@author: Dylan Neff, Dylan
"""


# Creating a modular micromegas drift simulation, and running a short example.
# This code will:
# - Define a MicromegasDriftSimulator class with configurable parameters
# - Provide simulate_event() to simulate one particle track crossing the drift gap
# - Provide batch_simulate() to run many events and collect aggregates
# - Run a short example and show sample results (DataFrame) and two plots:
#     1) Histogram of arrival times (ns)
#     2) Scatter of final x vs arrival time
# Notes on units:
# - distances internally are in microns (µm)
# - diffusion constants are in µm / sqrt(cm)
# - drift velocity is in µm / ns
# - gap input is in mm for user convenience (converted internally)
# The code uses matplotlib, and displays a DataFrame for a sample event.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from scipy.signal import fftconvolve
from scipy.special import gamma
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d

from DreamData import fit_waveform_parabola


def main():
    sim = MicromegasDriftSimulator(
        gap_mm=3.0,
        potential_V=800.0,
        ion_rate_per_mm=6.0,
        theta_deg=0.0,
        drift_velocity_um_per_ns=25.0,
        diff_t_um_per_sqrtcm=450.0,
        diff_l_um_per_sqrtcm=200.0,
        seed=None
    )
    # run_batch(sim)
    # run_event(sim)
    # run_events(sim, n_events=10000)
    # gas_properties_plot()
    gas_comparison_plot()

    print('donzo')


def drift_speed_vs_b():
    base_path = '/local/home/dn277127/Bureau/'
    b_fields = [0.2, 0.4, 2.0]
    gas = 'Ar_Iso_95_5'



def gas_comparison_plot():
    base_path = '/local/home/dn277127/Bureau/'
    gases = ['Ar_Iso_95_5', 'Ar_CO2_Iso_90_7_3', 'Ar_Iso_CO2_95_3_2', 'Ar_CO2_CF4_45_10_45']
    gas_colors = ['blue', 'orange', 'green', 'black']

    drift_voltages = np.arange(10, 2000, 20)  # V
    drift_gap = 0.3  # cm
    drift_fields = drift_voltages / drift_gap  # V/cm

    min_drift_field = 950  # V/cm
    min_drift_filter = drift_fields >= min_drift_field
    drift_fields = drift_fields[min_drift_filter]
    drift_voltages = drift_voltages[min_drift_filter]

    drift_field_measured = np.arange(1000, np.max(drift_fields), 500)  # V/cm

    b_fields = [0.2, 2.0]  # Tesla
    for b_field in b_fields:
        fig, ax = plt.subplots(figsize=(10, 5))
        for gas, color in zip(gases, gas_colors):
            gas_props = GasProperties(base_path=f'{base_path}gas', gas_type=gas, interp_kind='cubic')
            drift_v_e, drift_v_exb, lorentz_angle = gas_props.get_properties(drift_fields, b_field_tesla=b_field)
            d_v_e_m, d_v_exb_m, la_m = gas_props.get_properties(drift_field_measured, b_field_tesla=b_field)

            ax.plot(drift_voltages, drift_v_e, label=f'{gas}', color=color)
            ax.scatter(drift_field_measured * drift_gap, d_v_e_m, color=color, marker='o', s=20)

        ax.set_ylabel('Drift Velocity in E (cm/ns)')
        ax.set_xlabel('Drift Voltage (V)')
        ax.legend()
        fig.suptitle(f'B={b_field}T')
        fig.tight_layout()
    plt.show()


def gas_properties_plot():
    # Example usage of GasProperties class
    gas_type = 'Ar_Iso_95_5'
    # gas_type = 'Ar_CO2_Iso_90_7_3'
    # gas_type = 'Ar_Iso_CO2_95_3_2'
    gas_props = GasProperties(base_path='C:/Users/Dylan/Desktop/gas', gas_type=gas_type, interp_kind='cubic')
    drift_voltages = np.arange(10, 2000, 20)  # V
    drift_gap = 0.3  # cm
    drift_fields = drift_voltages / drift_gap  # V/cm

    min_drift_field = 950  # V/cm
    min_drift_filter = drift_fields >= min_drift_field
    drift_fields = drift_fields[min_drift_filter]
    drift_voltages = drift_voltages[min_drift_filter]

    drift_field_measured = np.arange(1000, np.max(drift_fields), 500)  # V/cm

    b_fields = [0.2, 0.4, 2.0]  # Tesla
    colors = ['blue', 'orange', 'red']
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex='all')
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for b, color in zip(b_fields, colors):
        drift_v_e, drift_v_exb, lorentz_angle = gas_props.get_properties(drift_fields, b_field_tesla=b)
        d_v_e_m, d_v_exb_m, la_m = gas_props.get_properties(drift_field_measured, b_field_tesla=b)

        ax[0].plot(drift_voltages, drift_v_e, label=f'B={b}T', color=color)
        ax[0].scatter(drift_field_measured * drift_gap, d_v_e_m, color=color, marker='o', s=20)
        ax[1].plot(drift_voltages, drift_v_exb, label=f'B={b}T', color=color)
        ax[1].scatter(drift_field_measured * drift_gap, d_v_exb_m, color=color, marker='o', s=20)
        ax[2].plot(drift_voltages, lorentz_angle, label=f'B={b}T', color=color)
        ax[2].scatter(drift_field_measured * drift_gap, la_m, color=color, marker='o', s=20)

        ax2.plot(drift_voltages, drift_v_e, label=f'B={b}T', color=color)
        ax2.scatter(drift_field_measured * drift_gap, d_v_e_m, color=color, marker='o', s=20, zorder=5)

    ax[0].set_ylabel('Drift Velocity in E (cm/ns)')
    ax[1].set_ylabel('Drift Velocity in ExB (μm/ns)')
    ax[2].set_ylabel('Lorentz Angle (degrees)')
    ax[2].set_xlabel('Drift Voltage (V)')
    ax[0].legend()
    fig.tight_layout()

    ax2.set_xlabel('Drift Voltage (V)')
    ax2.set_ylabel('Drift Velocity in E (μm/ns)')
    ax2.axvline(800, color='k', linestyle='--', alpha=0.4, label='800 V', zorder=0)
    ax2.set_title(gas_type)
    ax2.grid(zorder=0)
    ax2.legend()
    fig2.tight_layout()

    plt.show()


def run_event(sim):
    # simulate one event
    event = sim.simulate_event()

    # make a waveform
    t, wf = sim.simulate_waveform(event,
                                  t_min=-10, t_max=2000, dt=0.5,
                                  gain_mean=1e5, theta=2.0,
                                  tau_shaping_ns=80.0)

    sample_period = 60  # ns
    # Sample from waveform with sample_period
    sampled_wf = wf[::int(sample_period / (t[1] - t[0]))]
    sampled_t = t[::int(sample_period / (t[1] - t[0]))]
    y_vertex, x_vertex, success = fit_waveform_parabola(sampled_wf)

    fig, ax = plt.subplots()
    ax.plot(t, wf)
    ax.scatter(sampled_t, sampled_wf)
    ax.axhline(0, color='k')
    ax.axhline(y_vertex, color='r', linestyle='--')
    ax.axvline(x_vertex * sample_period, color='r', linestyle='--')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Charge (a.u.)")
    ax.set_title("Simulated Micromegas waveform")
    ax.annotate(f'n_prim = {event["n_prim"]}', xy=(0.7, 0.8), xycoords='axes fraction')
    # ax.legend(loc='best')
    fig.tight_layout()
    plt.show()


def run_events(sim, n_events=1000):
    trigger_res = 5  # ns, jitter in trigger time (not modeled in sim)

    sample_periods = [10, 30, 60]  # ns
    drift_speeds = np.arange(10, 40, 1)  # um/ns
    gas_props = GasProperties(base_path='C:/Users/Dylan/Desktop/gas', gas_type='Ar_Iso_95_5', interp_kind='cubic')
    drift_voltages = np.arange(10, 2000, 20)  # V
    drift_gap = 0.3  # cm
    drift_fields = drift_voltages / drift_gap  # V/cm

    min_drift_field = 1000  # V/cm
    min_drift_filter = drift_fields >= min_drift_field
    drift_fields = drift_fields[min_drift_filter]
    drift_voltages = drift_voltages[min_drift_filter]

    b_fields = [0.2, 0.4, 2.0]  # Tesla

    resolutions = {}
    for sample_period in sample_periods:
        resolutions[sample_period] = []
        for drift_speed in drift_speeds:
            print('drift speed:', drift_speed)
            sim.drift_velocity_um_per_ns = drift_speed
            inferred_times = []
            for _ in range(n_events):
                # simulate one event
                event = sim.simulate_event()

                # make a waveform
                t, wf = sim.simulate_waveform(event,
                                              t_min=-10, t_max=2000, dt=0.5,
                                              gain_mean=1e5, theta=2.0,
                                              tau_shaping_ns=80.0)

                # Shift t by trigger resolution
                t += np.random.normal(loc=0.0, scale=trigger_res)

                # Sample from waveform with sample_period
                step = int(sample_period / (t[1] - t[0]))
                sampled_wf = wf[::step]

                y_vertex, x_vertex, success = fit_waveform_parabola(sampled_wf)
                if success:  # only collect successful fits
                    inferred_times.append(x_vertex * sample_period)

            inferred_times = np.array(inferred_times)

            # Histogram data (no plotting yet so we can fit to it)
            counts, bin_edges = np.histogram(inferred_times, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Initial guess for fit: amplitude, mean, std
            p0 = [counts.max(), inferred_times.mean(), inferred_times.std()]

            try:
                popt, pcov = cf(gaussian, bin_centers, counts, p0=p0)
                A, mu, sigma = popt
            except RuntimeError:
                A, mu, sigma = np.nan, np.nan, np.nan

            # Plot histogram and fit
            plt.figure(figsize=(8, 6))
            plt.hist(inferred_times, bins=50, histtype='step', color='blue', label="Data")

            if not np.isnan(mu):
                x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
                plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', label="Gaussian fit")
                plt.annotate(
                    f"μ = {mu:.2f} ns\nσ = {sigma:.2f} ns",
                    xy=(0.65, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)
                )

            plt.xlabel("Inferred time of max (ns)")
            plt.ylabel("Counts")
            plt.title(f"Distribution of inferred peak times ({n_events} events)")
            plt.legend()
            plt.tight_layout()
            resolutions[sample_period].append(sigma)

    fig, ax = plt.subplots()
    for sample_period, resolution in resolutions.items():
        ax.plot(drift_speeds, resolution, marker='o', ls='none', alpha=0.8, label=f"Sample Period {sample_period} ns")
    ax.set_xlabel("Drift velocity (µm/ns)")
    ax.set_ylabel("Timing resolution (ns)")
    ax.legend()
    fig.tight_layout()

    plt.show()


def run_batch(sim):
    # simulate 200 events (quick example)
    df, ev_df = sim.batch_simulate(n_events=200)

    print(df.head())

    # Plot 1: histogram of arrival times
    if len(df) > 0:
        plt.figure()
        plt.hist(df['arrival_time_ns'], bins=50)
        plt.xlabel('Arrival time (ns)')
        plt.ylabel('Counts')
        plt.title('Histogram of primary arrival times (example)')

        # Plot 2: scatter final x vs arrival time
        plt.figure()
        plt.scatter(df['arrival_time_ns'], df['final_x_um'], alpha=0.3, s=10)
        plt.xlabel('Arrival time (ns)')
        plt.ylabel('Final x (µm)')
        plt.title('Final x vs arrival time (example)')

        fig, ax = plt.subplots()
        hb = ax.hexbin(df['arrival_time_ns'], df['final_x_um'], gridsize=50, cmap='Blues', mincnt=1)
        plt.colorbar(hb, ax=ax, label='Counts')
        ax.set_xlabel('Arrival time (ns)')
        ax.set_ylabel('Final x (µm)')
        ax.set_title('Final x vs arrival time (hexbin)')

        fig, ax = plt.subplots()
        ax.scatter(df['initial_x_um'], df['initial_z_um'], alpha=0.3, s=10)
        ax.set_xlabel('Initial x (µm)')
        ax.set_ylabel('Initial z (µm)')
        ax.set_title('Initial positions of primaries')
        plt.show()

    # Save a CSV for download
    # csv_path = "/mnt/data/micromegas_simulation_sample.csv"
    # df.to_csv(csv_path, index=False)
    # print(f"[Saved sample CSV to {csv_path}]")


@dataclass
class MicromegasDriftSimulator:
    gap_mm: float = 3.0  # drift gap in mm (user-friendly)
    potential_V: float = 800.0  # potential difference across gap in volts (not directly used except to record)
    ion_rate_per_mm: float = 20.0  # mean number of primary ionizations per mm of track in drift (Poisson rate)
    theta_deg: float = 0.0  # angle of track from normal toward +x in degrees (0 = perpendicular to plane)
    drift_velocity_um_per_ns: float = 50.0  # drift velocity in microns / ns
    diff_t_um_per_sqrtcm: float = 200.0  # transverse diffusion constant (µm / sqrt(cm))
    diff_l_um_per_sqrtcm: float = 300.0  # longitudinal diffusion constant (µm / sqrt(cm))
    seed: Optional[int] = None  # RNG seed for reproducibility

    # internal fields initialized after creation
    rng: np.random.Generator = field(init=False, repr=False)
    gap_um: float = field(init=False)
    tan_theta: float = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.gap_um = self.gap_mm * 1000.0  # mm -> microns
        self.tan_theta = np.tan(np.deg2rad(self.theta_deg))

    def _expected_primaries(self, track_length_mm: float) -> float:
        return self.ion_rate_per_mm * track_length_mm

    def simulate_event(self, return_individual_primaries: bool = True):
        """
        Simulate one particle track crossing the drift gap.
        Returns a dict with arrays for primaries: initial z (um), initial x (um),
        final x (um), arrival time (ns), drift distance (um), etc.
        """
        # Compute track length inside drift region.
        # For a straight track crossing from z=0 to z=gap, length = gap / cos(theta)
        cos_theta = 1.0 / np.sqrt(1.0 + self.tan_theta ** 2)
        track_length_um = self.gap_um / cos_theta  # path length through drift in µm
        track_length_mm = track_length_um / 1000.0

        # sample number of primaries (Poisson)
        mean_prim = self._expected_primaries(track_length_mm)
        n_prim = self.rng.poisson(mean_prim)

        if n_prim == 0:
            # return empty structured result
            return {
                'n_prim': 0,
                'initial_z_um': np.array([]),
                'initial_x_um': np.array([]),
                'final_x_um': np.array([]),
                'arrival_time_ns': np.array([]),
                'drift_distance_um': np.array([]),
                'longitudinal_displacement_um': np.array([]),
                'transverse_displacement_um': np.array([]),
            }

        # sample uniformly positions along path length s in [0, track_length_um)
        s = self.rng.uniform(0.0, track_length_um, size=n_prim)  # distance along the track from entry point
        # convert s -> z position in drift (because track goes from z=0 to z=gap)
        # relationship: z = s * cos(theta) because s * cos(theta) projects onto z
        z = s * cos_theta  # µm, between 0 and gap_um approximately

        # initial x coordinate along track: x = s * sin(theta) = s * tan(theta) * cos(theta) = z * tan(theta)
        x0 = z * self.tan_theta

        # for each primary, compute drift distance to readout plane at z = gap_um
        drift_distance_um = self.gap_um - z  # positive

        # diffusion sigmas (convert drift distance to cm for sqrt)
        drift_distance_cm = drift_distance_um / 10000.0  # 1 cm = 10,000 µm
        sigma_T = self.diff_t_um_per_sqrtcm * np.sqrt(np.maximum(drift_distance_cm, 0.0))
        sigma_L = self.diff_l_um_per_sqrtcm * np.sqrt(np.maximum(drift_distance_cm, 0.0))

        # transverse displacement due to diffusion (Gaussian)
        delta_x = self.rng.normal(loc=0.0, scale=sigma_T)
        # longitudinal displacement due to diffusion (along z) -> converts into arrival time jitter
        delta_z = self.rng.normal(loc=0.0, scale=sigma_L)

        # arrival time ignoring longitudinal diffusion
        drift_time_ns = drift_distance_um / self.drift_velocity_um_per_ns  # ns
        # arrival time adding longitudinal diffusion effect: delta_t = delta_z / v_drift
        arrival_time_ns = drift_time_ns + (delta_z / self.drift_velocity_um_per_ns)

        # final x at readout: initial x projected to readout plus transverse diffusion
        # initial x projected to readout (ignoring diffusion along track): x_at_readout = x0 + tan(theta)*drift_distance_z_component
        # but since track already has slope, moving from z to gap changes x by delta_x_track = (gap - z) * tan(theta)
        # initial x on track at z is x0; when reaching gap at z=gap the purely geometrical endpoint would be:
        x_geom_at_readout = x0 + drift_distance_um * self.tan_theta  # note: tan(theta) * dz
        # but that's equivalent to starting at track origin x=0 and going to z=gap gives x = gap * tan(theta)
        # now add transverse diffusion displacement
        final_x_um = x_geom_at_readout + delta_x

        return {
            'n_prim': n_prim,
            'initial_z_um': z,
            'initial_x_um': x0,
            'final_x_um': final_x_um,
            'arrival_time_ns': arrival_time_ns,
            'drift_distance_um': drift_distance_um,
            'longitudinal_displacement_um': delta_z,
            'transverse_displacement_um': delta_x,
        }

    def batch_simulate(self, n_events: int, max_primaries_to_record_per_event: int = 1000):
        """
        Run simulate_event() many times and aggregate results.
        Returns a DataFrame with one row per primary across all events (may be large).
        Also returns per-event summary.
        """
        rows = []
        event_summaries = []
        for ev in range(n_events):
            res = self.simulate_event()
            n = res['n_prim']
            event_summaries.append({'event': ev, 'n_prim': n})
            if n == 0:
                continue
            # cap recording if extremely large
            if n > max_primaries_to_record_per_event:
                inds = np.arange(max_primaries_to_record_per_event)
            else:
                inds = np.arange(n)
            for i in inds:
                rows.append({
                    'event': ev,
                    'initial_z_um': float(res['initial_z_um'][i]),
                    'initial_x_um': float(res['initial_x_um'][i]),
                    'final_x_um': float(res['final_x_um'][i]),
                    'arrival_time_ns': float(res['arrival_time_ns'][i]),
                    'drift_distance_um': float(res['drift_distance_um'][i]),
                    'longitudinal_displacement_um': float(res['longitudinal_displacement_um'][i]),
                    'transverse_displacement_um': float(res['transverse_displacement_um'][i]),
                })
        df = pd.DataFrame(rows)
        ev_df = pd.DataFrame(event_summaries)
        return df, ev_df

    def simulate_waveform_gaus(self, event_data, t_min, t_max, dt,
                               gain_mean=1e5, gain_sigma=2e4, pulse_sigma_ns=2.0):
        """
        Convert event primary arrivals into a summed waveform.

        Parameters
        ----------
        event_data : dict
            Output from simulate_event() (contains arrival_time_ns).
        t_min, t_max : float
            Time window in ns.
        dt : float
            Time step in ns.
        gain_mean, gain_sigma : float
            Mean and sigma of avalanche gain (number of electrons).
        pulse_sigma_ns : float
            Gaussian width of the single-electron pulse in ns.
        """
        times = np.arange(t_min, t_max, dt)
        waveform = np.zeros_like(times)

        n = event_data['n_prim']
        if n == 0:
            return times, waveform

        # Sample amplification gains
        gains = self.rng.normal(loc=gain_mean, scale=gain_sigma, size=n)
        gains = np.clip(gains, 0, None)  # no negative charge

        # Add each electron’s pulse
        for t0, g in zip(event_data['arrival_time_ns'], gains):
            waveform += gaussian_pulse(times, t0, pulse_sigma_ns, g)

        return times, waveform

    def simulate_waveform(self, event_data, t_min, t_max, dt,
                          gain_mean=1e5, theta=2.0,
                          tau_shaping_ns=10.0):
        """
        Simulate shaped waveform from event primaries.

        Parameters
        ----------
        event_data : dict
            From simulate_event(), with 'arrival_time_ns'.
        t_min, t_max : float
            Time window in ns.
        dt : float
            Time step in ns.
        gain_mean : float
            Mean avalanche gain (electrons).
        theta : float
            Polya shape parameter (0=exponential, higher=more narrow).
        tau_shaping_ns : float
            Shaping time constant of CR-RC filter (ns).
        """
        times = np.arange(t_min, t_max, dt)
        waveform = np.zeros_like(times)

        n = event_data['n_prim']
        if n == 0:
            return times, waveform

        # --- Sample avalanche gains from Polya (Gamma) ---
        k = theta + 1.0
        scale = gain_mean / (theta + 1.0)
        gains = self.rng.gamma(shape=k, scale=scale, size=n)

        # --- Deposit raw current: delta function at each arrival time ---
        for t0, g in zip(event_data['arrival_time_ns'], gains):
            # Find nearest index
            idx = int(round((t0 - t_min) / dt))
            if 0 <= idx < len(waveform):
                waveform[idx] += g

        # --- Build shaping function (CR-RC) ---
        t_sh = np.arange(0, 10 * tau_shaping_ns, dt)  # cutoff at 10 tau
        h = (t_sh / tau_shaping_ns ** 2) * np.exp(-t_sh / tau_shaping_ns)

        # Normalize shaping (area=1) so gain sets amplitude scale
        h /= np.sum(h)

        # --- Convolve raw signal with shaping function ---
        shaped = fftconvolve(waveform, h, mode='full')[:len(times)]

        return times, shaped

class GasProperties:
    """Class to read and interpolate gas properties from Garfield Magboltz output files."""

    def __init__(self, base_path: str, gas_type: str, interp_kind: str = 'linear'):
        self.base_path = base_path
        self.interp_kind = interp_kind
        self.gas_prop_df = None
        self.filename = f"{self.base_path}/{gas_type}/out.txt"
        self.load_data()

    def load_data(self):
        self.gas_prop_df = import_garfield_output(self.filename)

    def get_properties(self, field_v_per_cm, b_field_tesla=0.2):
        """Interpolate properties at given electric field."""
        df_b_field = self.gas_prop_df[self.gas_prop_df['Magnetic Field (Tesla)'] == b_field_tesla]
        if df_b_field.empty:
            raise ValueError(f"No data for B={b_field_tesla} Tesla\nAvailable B fields: {self.gas_prop_df['Magnetic Field (Tesla)'].unique()}")
        electric_fields = df_b_field['Electric field (V/cm)'].values
        drift_velocity_e = df_b_field['vz (cm/ns)'].values
        drift_velocity_e_cross_b = df_b_field['vy'].values
        lorentz_angle = df_b_field['Lorentz Angle (degree)'].values

        dve_interp = interp1d(electric_fields, drift_velocity_e, kind=self.interp_kind, fill_value="extrapolate")
        dvexb_interp = interp1d(electric_fields, drift_velocity_e_cross_b, kind=self.interp_kind, fill_value="extrapolate")
        la_interp = interp1d(electric_fields, lorentz_angle, kind=self.interp_kind, fill_value="extrapolate")

        drift_velocity_e = dve_interp(field_v_per_cm) * 1e4  # convert cm/ns to um/ns
        drift_velocity_e_cross_b = dvexb_interp(field_v_per_cm) * 1e4  # convert cm/ns to um/ns
        lorentz_angle = la_interp(field_v_per_cm)

        return drift_velocity_e, drift_velocity_e_cross_b, lorentz_angle

def import_garfield_output(filepath: str) -> pd.DataFrame:
    """
    Import Garfield++/Magboltz output table into a pandas DataFrame.

    Expected columns:
    Electric field (V/cm), Magnetic Field (Tesla),
    vx, vy, vz (cm/ns), Lorentz Angle (degree), Lorentz Angle (radians)
    """
    # Find the line where the table starts
    start_line = None
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if "====" in line:  # separator line
                start_line = i + 1
                break

    if start_line is None:
        raise ValueError("Could not find start of table in file.")

    # Load numerical data into DataFrame
    df = pd.read_csv(
        filepath,
        delim_whitespace=True,
        skiprows=start_line,
        header=None,
        names=[
            "Electric field (V/cm)",
            "Magnetic Field (Tesla)",
            "vx",
            "vy",
            "vz (cm/ns)",
            "Lorentz Angle (degree)",
            "lorentz angle in radians",
        ],
    )

    return df

def gaussian_pulse(t, t0, sigma, amplitude):
    """Return Gaussian of given amplitude centered at t0."""
    return amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def polya(Q, Qbar, theta, norm):
    return norm * ((1+theta)**(1+theta) / gamma(1+theta)) \
           * (Q/Qbar)**theta * np.exp(-(1+theta)*Q/Qbar)


if __name__ == '__main__':
    main()
