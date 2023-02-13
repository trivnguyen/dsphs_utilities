#!/usr/bin/env python

import os
import sys
import argparse
import h5py
import numpy as np
import gizmo_analysis as gizmo

DEFAULT_BASE_SIM_DIRECTORY = "/ocean/projects/ast200012p/tvnguyen/FIRE/metal_diffusion"
DEFAULT_OUTPUT_DIRECTORY = "/ocean/projects/ast200012p/tvnguyen/FIRE/particles"

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sim-name', required=True, type=str, help="Simulation name")
    parser.add_argument(
        '--output', required=False, type=str, help="Path to output file"
    )
    parser.add_argument(
        '--base-sim-dir', required=False, type=str,
        default=DEFAULT_BASE_SIM_DIRECTORY, help="Base simulation directory")
    parser.add_argument(
        '--snapshot', required=False, type=int, default=600,
        help="Snapshot number"
    )
    return parser.parse_args()

def main(FLAGS):
    """ Read particle data using gizmo and write the galactocentric coordinates
    to an HDF5 file.
    """

    # read particle data using gizmo.io.Read
    sim_directory = os.path.join(FLAGS.base_sim_dir, FLAGS.sim_name)
    part = gizmo.io.Read.read_snapshots(
        ['dark', 'star'], 'index', FLAGS.snapshot,
        simulation_directory=sim_directory,
    )
    # assign hosts based on DM particle and rotation based on stars
    gizmo.io.Read.assign_hosts_coordinates(part, species_name='dark')

    # get the galactocentric coordinate of all DM and star particles
    if FLAGS.output is None:
        FLAGS.output = os.path.join(
            DEFAULT_OUTPUT_DIRECTORY, FLAGS.sim_name + '.hdf5')
        print(f'output not given. default: {FLAGS.output}')

    print(f'writing data to {FLAGS.output}')
    with h5py.File(FLAGS.output, 'w') as f:
        f.attrs.update({
            'simulation_name': FLAGS.sim_name,
            'simulation_directory': FLAGS.base_sim_dir,
            'snapshot': FLAGS.snapshot
        })
        # f.attrs.update(part.info)
        for species in ['dark', 'star']:
            gr = f.create_group(species)
            gr.create_dataset(
                'position', data=part[species].prop('host.distance')
            )
            gr.create_dataset(
                'velocity', data=part[species].prop('host.velocity')
            )
            gr.create_dataset(
                'mass', data=part[species].prop('mass')
            )

if __name__ == "__main__":
    FLAGS = parse_cmd()
    main(FLAGS)
