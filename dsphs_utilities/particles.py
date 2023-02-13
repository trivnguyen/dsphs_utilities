
import numpy as np
import scipy

def calc_mass_enc(r, mass, r_min=None, r_max=None, num_bins=None):
    """
    Calculate the enclosed mass from particle data

    Args:
    - r: Radii of each particle as calculated from the center of the galaxy
    - mass: Mass of each particle
    - r_min: Minimum bin radius. Default to minimum value of r
    - r_max: Maximum bin radius. Default to maximum value of r
    - num_bins: number of bins. Default to sqrt(N) where N is the number of particles

    Returns:
    - mass_enc: The enclosed mass
    - logr_bins: Radial bins on log10-scale
    """
    logr = np.log10(r)
    num_bins = np.sqrt(len(r)) if num_bins is None else num_bins
    log_rmin = np.floor(np.min(logr) * 10) / 10 if r_min is None else np.log10(r_min)
    log_rmax = np.floor(np.max(logr) * 10) / 10 if r_max is None else np.log10(r_max)

    # histogram mass profile
    mass_bins, log_rbins = np.histogram(
        logr, num_bins, range=(log_rmin, log_rmax), weights=mass)
    mass_enc = np.cumsum(mass_bins)  # cumulative mass

    return mass_enc, log_rbins

def calc_rho(r, mass, r_min=None, r_max=None, num_bins=None):
    """
    Calculate the density profile

    Args:
    - r: Radii of each particle as calculated from the center of the galaxy
    - mass: Mass of each particle
    - r_min: Minimum bin radius. Default to minimum value of r
    - r_max: Maximum bin radius. Default to maximum value of r
    - num_bins: number of bins. Default to sqrt(N) where N is the number of particles

    Returns:
    - rho: the density profile
    - log_rbins: Radial bins on log10-scale
    """
    logr = np.log10(r)
    num_bins = np.sqrt(len(r)) if num_bins is None else num_bins
    log_rmin = np.floor(np.min(logr) * 10) / 10 if r_min is None else np.log10(r_min)
    log_rmax = np.floor(np.max(logr) * 10) / 10 if r_max is None else np.log10(r_max)

    # histogram mass profile
    mass_bins, log_rbins = np.histogram(
        logr, num_bins, range=(log_rmin, log_rmax), weights=mass)
    r_bins = np.power(10, log_rbins)

    # calculate density profile
    dV = 4 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3) / 3
    rho = mass_bins / dV
    return rho, log_rbins

def calc_rho_bar(r, mass, r_min=None, r_max=None, num_bins=None):
    """
    Calculate the average density profile

    Args:
    - r: Radii of each particle as calculated from the center of the galaxy
    - mass: Mass of each particle
    - r_min: Minimum bin radius. Default to minimum value of r
    - r_max: Maximum bin radius. Default to maximum value of r
    - num_bins: number of bins. Default to sqrt(N) where N is the number of particles

    Returns:
    - rho_bar: The average density profile
    - logr_bins: Radial bins on log10-scale
    """
    mass_enc, log_rbins = calc_mass_enc(
        r, mass, r_min=r_min, r_max=r_max, num_bins=num_bins)
    r_bins = np.power(10, log_rbins)
    rho_bar = 3 * mass_enc / (4 * np.pi * r_bins[1:]**3)
    return rho_bar, log_rbins

def calc_log_rvir(r, mass, r_min=None, r_max=None, num_bins=None, vir=200, n_steps=10000):
    """ Solve for the virial radius. Default to c200, the radius within which the average
    densityis 200 times the critical density of the Universe at redshift z = 0
    """
    # critical density
    rho_c = Planck18.critical_density(0).to_value(u.Msun / u.kpc**3)

    # calculate enclosed mass and average density
    rho_bar, log_rbins = calc_rho_bar(
        r, mass, r_min=r_min, r_max=r_max, num_bins=num_bins)

    try:
        log_rbins_ce = 0.5 * (log_rbins[1:] + log_rbins[:-1])
        dlog_rho = np.log10(rho_bar) - np.log10(rho_c)
        return scipy.interpolate.interp1d(dlog_rho, log_rbins_ce)(np.log10(vir))
    except Exception as e:
        print(e)
        return np.nan

