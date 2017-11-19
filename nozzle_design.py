import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from table_generator import generate_table


class NozzleDesign(object):
    """
    Minimum-length nozzle design procedure using MoC.
        1. Calculate maximum turning angle from design Mach number.
        2. Decide on number of characteristics to use.
        3. Assign theta for each characteristic line emitted from origin (normally uniformly distributed).
            Starting with a small angle for theta_a to theta_d = theta_max.
        4. Assume R+ = 0 for a starting line which is very close to the sonic line at the nozzle throat.
        5. Calculate all properties at the starting points.
        6. Calculate flow variables (M, nu, mach angle, etc) for downstream points using MoC.
        7. Calculate coordinates for all points and the nozzle contour (the wall).
    """

    def __init__(self, folder_path, design_mach, gamma, characteristics, start_angle):
        self.folder_path = folder_path
        self.design_mach = design_mach
        self.gamma = gamma
        self.characteristics = characteristics
        self.start_angle = start_angle

    def __call__(self, *args, **kwargs):
        self._min_length_nozzle_setup()

        char_idx = range(self.characteristics)
        lines = self.characteristics
        for char in char_idx:
            self._characteristic_tracing(char, char_idx[0], lines)

        char_idx = [char + lines for char in char_idx[1:]]
        for char in char_idx:
            self._characteristic_tracing(char, char_idx[0], lines)

        while lines > 1:
            char_idx = [char + lines for char in char_idx[1:]]
            lines -= 1
            for char in char_idx:
                self._characteristic_tracing(char, char_idx[0], lines)

        self._compute_wall_points()

        self._write_moc_csv()

        plt.plot(self.moc_table['x'], self.moc_table['y'], 'ro')
        plt.show()

    def _number_of_points(self, n):
        """
        For symmetric planar nozzle only

        :param n: integer number of characteristic lines
        :return: integer number of points from those lines
        """
        if n == 0:
            return 0
        elif n == 1:
            return 3
        elif n == 2:
            return 2 * 3 + 1
        else:
            return n * 3 + self._number_of_points(n - 1) - (n - 1) * 3 + (n - 1)

    def _characteristic_tracing(self, char, char_idx_0, lines):
        # Trace R- characteristics down
        self.moc_table['R-'][char + lines] = self.moc_table['R-'][char]
        # Trace R+ characteristics from centreline boundary condition
        self.moc_table['R+'][char + lines] = self.moc_table['R-'][char_idx_0]
        # Calculate theta and nu
        self.moc_table['theta'][char + lines] = (self.moc_table['R-'][char + lines] -
                                                 self.moc_table['R+'][char + lines]) / 2.0
        self.moc_table['nu'][char + lines] = (self.moc_table['R-'][char + lines] +
                                              self.moc_table['R+'][char + lines]) / 2.0

        # Retrieve Mach number and angle from Prandtl-Meyer table using nearest value approx
        nu = self.moc_table['nu'][char + lines]
        nu_idx = min(enumerate(self.prandtl_meyer), key=lambda y: abs(y[1] - nu))
        m_angle = self.mach_angles[nu_idx[0]]
        self.moc_table['M'][char + lines] = self.mach_numbers[nu_idx[0]]
        self.moc_table['m_angle'][char + lines] = m_angle
        self.moc_table['theta+m_angle'][char + lines] = self.moc_table['theta'][char + lines] + m_angle
        self.moc_table['theta-m_angle'][char + lines] = self.moc_table['theta'][char + lines] - m_angle

        # Compute coordinates
        if char != char_idx_0:
            alpha_ap = 0.5 * (self.moc_table['theta+m_angle'][char + lines - 1] +
                              self.moc_table['theta+m_angle'][char + lines])
            alpha_bp = 0.5 * (self.moc_table['theta-m_angle'][char] +
                              self.moc_table['theta-m_angle'][char + lines])

            self.moc_table['x'][char + lines] = ((self.moc_table['x'][char] * math.tan(math.radians(alpha_bp)) -
                                                  self.moc_table['x'][char + lines - 1] *
                                                  math.tan(math.radians(alpha_ap)) +
                                                  self.moc_table['y'][char + lines - 1] - self.moc_table['y'][char]) /
                                                 (math.tan(math.radians(alpha_bp)) - math.tan(math.radians(alpha_ap))))

            self.moc_table['y'][char + lines] = (self.moc_table['y'][char + lines - 1] +
                                                 (self.moc_table['x'][char + lines] -
                                                  self.moc_table['x'][char + lines - 1]) *
                                                 math.tan(math.radians(alpha_ap)))

        # Compute centreline coordinates
        else:
            alpha_bp = 0.5 * (self.moc_table['theta-m_angle'][char] +
                              self.moc_table['theta-m_angle'][char + lines])
            self.moc_table['x'][char + lines] = (self.moc_table['x'][char] -
                                                 self.moc_table['y'][char] / math.tan(math.radians(alpha_bp)))
            self.moc_table['y'][char + lines] = 0

        # Remove idx from points list
        self.points.remove(char + lines)

    def _compute_wall_points(self):
        for idx, point in enumerate(self.points):
            self.moc_table[point] = self.moc_table[point - 1]
            alpha_aw = self.moc_table['theta+m_angle'][point - 1]
            alpha_bw = 0.5 * (self.moc_table['theta'][point - (self.characteristics + 1) + idx] +
                              self.moc_table['theta'][point])

            self.moc_table['x'][point] = (self.moc_table['x'][point - (self.characteristics + 1) + idx] *
                                          math.tan(math.radians(alpha_bw)) - self.moc_table['x'][point - 1] *
                                          math.tan(math.radians(alpha_aw)) +
                                          self.moc_table['y'][point - 1] -
                                          self.moc_table['y'][point - (self.characteristics + 1) + idx]) / \
                                         (math.tan(math.radians(alpha_bw)) - math.tan(math.radians(alpha_aw)))

            self.moc_table['y'][point] = (self.moc_table['y'][point - 1] +
                                          (self.moc_table['x'][point] -
                                           self.moc_table['x'][point - 1]) * math.tan(math.radians(alpha_aw)))

    def _write_moc_csv(self):
        file_name = 'MoC Table - Minimum length nozzle (M_des={},gamma={},no_characteristics={},start_angle={}).csv'.format(
            self.design_mach,
            self.gamma,
            self.characteristics,
            self.start_angle
        )
        file_path = os.path.join(self.folder_path, file_name)
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.moc_table_headers)
            for row in self.moc_table:
                writer.writerow(row)

    def _min_length_nozzle_setup(self):
        # Generate prantl-meyer table for calculations
        self.mach_numbers, self.mach_angles, self.prandtl_meyer = generate_table(self.gamma,
                                                                                 (1.0, self.design_mach),
                                                                                 0.01,
                                                                                 4,
                                                                                 write_csv=False)

        # Calculate theta max
        theta_max = self.prandtl_meyer[self.mach_numbers.index(self.design_mach)] / 2.0

        # Define characteristic line thetas
        self.characteristic_thetas = [round(x, 8) for x in np.linspace(self.start_angle,
                                                                       theta_max, self.characteristics)]

        # Generate list of point idx for given number of characteristic lines
        self.points = range(self._number_of_points(self.characteristics))

        # Create empty array for MoC data
        self.moc_table_headers = ['R+', 'R-', 'theta', 'nu', 'M', 'm_angle', 'theta+m_angle', 'theta-m_angle', 'x', 'y']
        self.moc_table = np.zeros(len(self.points), dtype=[(header, 'f') for header in self.moc_table_headers])

        # Last two points form line that is last expansion before uniform flow
        self.moc_table['M'][self.points[-1]] = self.design_mach
        self.moc_table['M'][self.points[-2]] = self.design_mach

        for idx, theta in enumerate(self.characteristic_thetas):
            # From definitions
            self.moc_table['theta'][idx] = theta
            self.moc_table['R+'][idx] = 0  # characteristic lines start at top of throat (0, 1) - radius 1
            self.moc_table['x'][idx] = 0
            self.moc_table['y'][idx] = 1

            # Calculate nu (nu at throat = theta) and hence R- (R- = nu + theta)
            nu = theta
            self.moc_table['nu'][idx] = nu
            self.moc_table['R-'][idx] = nu + theta

            # Retrieve Mach number and angle from Prandtl-Meyer table
            nu_idx = min(enumerate(self.prandtl_meyer), key=lambda y: abs(y[1] - nu))  # Get nearest idx to nu
            m_angle = self.mach_angles[nu_idx[0]]
            self.moc_table['M'][idx] = self.mach_numbers[nu_idx[0]]
            self.moc_table['m_angle'][idx] = m_angle
            self.moc_table['theta+m_angle'][idx] = theta + m_angle
            self.moc_table['theta-m_angle'][idx] = theta - m_angle

            # Remove idx from points list
            self.points.remove(idx)

if __name__ == '__main__':
    input_folder_path = sys.argv[1]
    input_design_mach = float(sys.argv[2])
    input_gamma = tuple(float(x) for x in sys.argv[3].split(',')) if len(sys.argv[3].split(',')) != 1 \
        else float(sys.argv[3])
    input_characteristics = int(sys.argv[4])
    input_start_angle = float(sys.argv[5])
    nozzle_design = NozzleDesign(input_folder_path,
                                 input_design_mach,
                                 input_gamma,
                                 input_characteristics,
                                 input_start_angle)
    nozzle_design()
