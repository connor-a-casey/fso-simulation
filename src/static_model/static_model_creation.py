import math
import numpy as np
import matplotlib.pyplot as plt

# Function Definitions
def calculate_slant_range(latitude, longitude, ground_elevation, elevation, azimuth):
    elevation_rad = math.radians(elevation)
    ground_elevation_km = ground_elevation / 1000.0  # convert to kilometers
    effective_radius = earth_radius_km + ground_elevation_km
    slant_range = effective_radius * (1 / math.sin(elevation_rad))
    return slant_range

def calculate_free_space_loss(S, f):
    Ls = 20 * math.log10(c / (4 * math.pi * S * f))
    return Ls

def calculate_antenna_gain(D, f, eta):
    lambda_ = c / f
    G = (math.pi * D / lambda_)**2
    G_db = 10 * math.log10(eta * G)
    return G_db

def calculate_pointing_loss(theta_p, f, D):
    theta_3dB = 21 / (f / 10**9 * D)
    Lp = -12 * (theta_p / theta_3dB)**2
    return Lp

def calculate_system_noise_temperature(T):
    Ts_dBi = 10 * math.log10(T)
    return Ts_dBi

def calculate_boltzmann_constant():
    k_B_dBi = 10 * math.log10(k_B)
    return k_B_dBi

def calculate_data_rate(Power, SNR_required, margin, L_imp, L_a, G_t_net, Gr_net, Ls, k_B_dBi, Ts_dBi):
    R_dB = Power - (SNR_required + margin) + L_imp + L_a + G_t_net + Gr_net + Ls - k_B_dBi - Ts_dBi
    data_rate_bps = 10**(R_dB / 10)
    return R_dB, data_rate_bps

# Constants
c = 3 * 10**8  # speed of light in m/s
k_B = 1.38 * 10**-23  # Boltzmann constant in J/K
earth_radius_km = 6371  # Radius of Earth in kilometers

if __name__ == "__main__":
    # INPUT - Link Conditions
    f = 2.1 * 10**9
    D_r = 1
    eta_r = 0.55
    theta_pr = 0.001
    T = 135
    D_t = 1
    eta_t = 0.55
    theta_pt = 1
    G_t = 50
    SNR_required = 10
    margin = 3
    misc_losses = 0
    L_l = 1.96
    L_imp = 0.3
    L_a = 0.3
    Power = 18.65

    # INPUT - Ground Station Location
    latitude = 52.5200
    longitude = 13.4050
    ground_elevation = 34  # in meters

    # INPUT - Satellite Position (Static)
    elevation = 45.0
    azimuth = 135.0

    # Calculating Slant Distance
    S = calculate_slant_range(latitude, longitude, ground_elevation, elevation, azimuth) * 1000  # in meters
    print(f"Slant range: {S:.2f} m")

    Ls = calculate_free_space_loss(S, f)
    print(f"Free space loss (L_s) = {Ls:.2f} dB")

    Gr_db = calculate_antenna_gain(D_r, f, eta_r)
    print(f"Receiver antenna peak gain (G_rp) = {Gr_db:.2f} dBi")

    Lp_r = calculate_pointing_loss(theta_pr, f, D_r)
    print(f"Receiver antenna pointing loss (L_p,r) = {Lp_r:.2f} dBi")

    Gr_net = Gr_db + Lp_r
    print(f"Receiver antenna net gain (G_r) = {Gr_net:.2f} dBi")

    Ts_dBi = calculate_system_noise_temperature(T)
    print(f"System noise temperature (T_s) = {Ts_dBi:.2f} dBi")

    k_B_dBi = calculate_boltzmann_constant()
    print(f"Boltzmann constant (k_B) = {k_B_dBi:.2f} dBi")

    Lp_t = calculate_pointing_loss(theta_pt, f, D_t)
    print(f"Transmit antenna pointing loss (L_p,t) = {Lp_t:.2f} dBi")

    G_tp = calculate_antenna_gain(D_t, f, eta_t)
    print(f"Transmit antenna peak gain (G_tp) = {G_tp:.2f} dBi")

    G_t_net = G_tp + Lp_t - L_l
    print(f"Transmit antenna net gain (G_t) = {G_t_net:.2f} dBi")

    EIRP = Power + G_t_net
    print(f"EIRP = {EIRP:.2f} dB")

    R_dB, data_rate_bps = calculate_data_rate(Power, SNR_required, margin, L_imp, L_a, G_t_net, Gr_net, Ls, k_B_dBi, Ts_dBi)
    print(f"Data Rate = {R_dB:.2f} dB")
    print(f"Data Rate = {data_rate_bps:.2f} bits per second")
