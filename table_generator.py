import csv
import math
import numpy as np
import os
import sys


def generate_table(gamma, mach_range, step_distance, decimal_places, folder_path=None, write_csv=True):
    """
    :param gamma: float ratio of specific heats for a given gas
    :param mach_range: tuple of floats - inclusive range of Mach numbers between which to generate table
    :param step_distance: float step size to evaluate table at
    :param decimal_places: integer number of decimal places for table
    :param folder_path: string folder path for output file
    :param write_csv: boolean to determine whether to write csv
    :return: .csv file containing Mach number, Mach angle, Prandtl-Meyer function
    """
    file_name = 'lookuptable_gamma{}_mach{}to{}_step{}_{}dp.csv'.format(gamma,
                                                                        mach_range[0], mach_range[1],
                                                                        step_distance,
                                                                        decimal_places)
    print('Generating table...')
    gamma = gamma[0] / gamma[1] if type(gamma) is tuple else gamma
    mach_numbers = [round(x, 8) for x in np.arange(mach_range[0], mach_range[1] + step_distance, step=step_distance)]
    mach_angles = [math.degrees(math.asin(1.0 / mach_number)) for mach_number in mach_numbers]
    prandtl_meyer = [math.degrees(math.sqrt((gamma + 1) / (gamma - 1)) *
                                  math.atan(math.sqrt((gamma - 1) / (gamma + 1) * (mach_number ** 2 - 1))) -
                                  math.atan(math.sqrt(mach_number ** 2 - 1))) for mach_number in mach_numbers]

    if write_csv and folder_path is not None:
        file_path = os.path.join(folder_path, file_name)
        print('Writing table to .csv file...')
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Mach number', 'Mach angle', 'Prandtl-Meyer'])
            for row in zip([round(number, decimal_places) for number in mach_numbers],
                           [round(angle, decimal_places) for angle in mach_angles],
                           [round(prandtl, decimal_places) for prandtl in prandtl_meyer]):
                writer.writerow(row)

        print('File saved.')
    else:
        return mach_numbers, mach_angles, prandtl_meyer


if __name__ == '__main__':
    input_folder_path = sys.argv[1]
    input_gamma = tuple(float(x) for x in sys.argv[2].split(',')) if len(sys.argv[2].split(',')) != 1 \
        else float(sys.argv[2])
    input_mach_range = tuple(float(x) for x in sys.argv[3].split(','))
    input_step_distance = float(sys.argv[4])
    input_decimal_places = int(sys.argv[5])

    print('Running generate_table function...')
    generate_table(input_gamma, input_mach_range, input_step_distance, input_decimal_places, input_folder_path)
