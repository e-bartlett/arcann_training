"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2024/03/01
Last modified: 2024/04/01
"""

# TODO: Homogenize the docstrings for this module

# Standard library modules
import logging
import re

# Third-party modules
import numpy as np

# Local imports
from deepmd_iterative.common.utils import catch_errors_decorator


# TODO: Add tests for this function
def extract_and_convert_energy(energy_in, energy_out, system_candidates_not_skipped_counter, factor=1.0, program=None, version=None):
    if program == "cp2k":
        pattern = r"ENERGY\|.*?([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)"

        for line in energy_in:
            match = re.search(pattern, line)
            if match:
                energy_float = float(match.group(1))
                energy_array = np.asarray(energy_float, dtype=np.float64).flatten()
                energy_out[system_candidates_not_skipped_counter - 1] = energy_array[0] * factor
        return energy_out
    elif program == "orca":
        energy_line_index = energy_in.index("# The current total energy in Eh") + 2
        energy = float(energy_in[energy_line_index].strip())
        energy_out[system_candidates_not_skipped_counter - 1] = energy


# TODO: Add tests for this function
def extract_and_convert_coordinates(coordinates_in, coordinates_out, system_candidates_not_skipped_counter, factor=1.0):
    del coordinates_in[0:2]
    coordinates_in = [" ".join(_.replace("\n", "").split()) for _ in coordinates_in]
    coordinates_in = [_.split(" ")[1:] for _ in coordinates_in]
    coord_array = np.asarray(coordinates_in, dtype=np.float64).flatten()
    coordinates_out[system_candidates_not_skipped_counter - 1, :] = coord_array * factor
    return coordinates_out


# TODO: Add tests for this function
def extract_and_convert_box_volume(input, box_out, volume_out, system_candidates_not_skipped_counter, factor=1.0, program=None, version=None):
    if program == "cp2k":
        cell = [float(_) for _ in re.findall(r"\d+\.\d+", [_ for _ in input if "ABC" in _][0])]
        box_out[system_candidates_not_skipped_counter - 1, 0] = cell[0]
        box_out[system_candidates_not_skipped_counter - 1, 4] = cell[1]
        box_out[system_candidates_not_skipped_counter - 1, 8] = cell[2]
        volume_out[system_candidates_not_skipped_counter - 1] = cell[0] * cell[1] * cell[2]
        return box_out, volume_out
    elif program == "orca":
        cell = input
        box_out[system_candidates_not_skipped_counter - 1, 0] = cell[0]
        box_out[system_candidates_not_skipped_counter - 1, 4] = cell[1]
        box_out[system_candidates_not_skipped_counter - 1, 8] = cell[2]
        volume_out[system_candidates_not_skipped_counter - 1] = cell[0] * cell[1] * cell[2]
        return box_out, volume_out


# TODO: Add tests for this function
def extract_and_convert_forces(forces_in, forces_out, system_candidates_not_skipped_counter, factor=1.0, program=None, version=None):
    if program == "cp2k":
        del forces_in[0:4]
        del forces_in[-1]
        forces_in = [" ".join(_.replace("\n", "").split()) for _ in forces_in]
        forces_in = [_.split(" ")[3:] for _ in forces_in]
        forces_array = np.asarray(forces_in, dtype=np.float64).flatten()
        forces_out[system_candidates_not_skipped_counter - 1, :] = forces_array * factor
        return forces_out
    elif program == "orca":
        start_index = forces_in.index("# The current gradient in Eh/bohr") + 2
        try:
            end_index = forces_in[start_index:].index("#") + start_index
        except ValueError:
            end_index = len(forces_in)
        gradient_values = forces_in[start_index:end_index]
        forces_array = np.asarray([float(value.strip()) for value in gradient_values], dytpe=np.float64)
        forces_out[system_candidates_not_skipped_counter - 1, :] = forces_array * factor
        return forces_out


# TODO: Add tests for this function
def extract_and_convert_virial(stress_in, virial_out, system_candidates_not_skipped_counter, volume, factor=1.0, program=None, version=None):
    if program == "cp2k":
        if version < 8 and version >= 6:
            matching_index = None
            for index, line in enumerate(stress_in):
                if re.search(r"\bX\b.*\bY\b.*\bZ\b", line):
                    matching_index = index
                    break
            del index, line
            if matching_index is not None:
                x_values = re.findall(r"X\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)", stress_in[matching_index + 1])
                y_values = re.findall(r"Y\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)", stress_in[matching_index + 2])
                z_values = re.findall(r"Z\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)", stress_in[matching_index + 3])

                tensor_values = np.vstack((x_values, y_values, z_values)).astype(np.float64)

                stress_xyz_array = tensor_values.flatten()
                virial_out[system_candidates_not_skipped_counter - 1, :] = stress_xyz_array * volume[system_candidates_not_skipped_counter - 1] / factor

                return virial_out, True
            else:
                return virial_out, False

        elif version < 2024 and version >= 8:
            matching_index = None
            for index, line in enumerate(stress_in):
                if re.search(r"\bx\b.*\by\b.*\bz\b", line):
                    matching_index = index
                    break
            del index, line
            if matching_index is not None:
                x_values = re.findall(r"x\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)", stress_in[matching_index + 1])
                y_values = re.findall(r"y\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)", stress_in[matching_index + 2])
                z_values = re.findall(r"z\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)\s+(-?\d+\.\d+E[+-]\d+)", stress_in[matching_index + 3])

                tensor_values = np.vstack((x_values, y_values, z_values)).astype(np.float64)
                stress_xyz_array = tensor_values.flatten()
                virial_out[system_candidates_not_skipped_counter - 1, :] = stress_xyz_array * volume[system_candidates_not_skipped_counter - 1] / factor

                return virial_out, True
            else:
                return virial_out, False

        else:
            logging.info(f"This version of CP2K is not supported for tensor: {version}")
            return virial_out, False


# TODO: Add tests for this function
def extract_and_convert_wannier(wannier_in, wannier_out, system_candidates_not_skipped_counter, nb_atom, factor=1.0, program=None, version=None):
    if program == "cp2k":
        del wannier_in[0 : 2 + nb_atom]
        wannier_in = [" ".join(_.replace("\n", "").split()) for _ in wannier_in]
        wannier_in = [_.split(" ")[1:] for _ in wannier_in]
        wannier_array = np.asarray(wannier_in, dtype=np.float64).flatten()
        wannier_out[system_candidates_not_skipped_counter - 1, :] = wannier_array * factor
        return wannier_out, True
