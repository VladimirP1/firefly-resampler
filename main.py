import camio
import sys
import argparse

import resampler

parser = argparse.ArgumentParser()

parser.add_argument('input_file', type = str, help = "CSV file straight from the camera")
parser.add_argument('output_file', type = str, help = "Output file name")
parser.add_argument('--flip-xy', action = 'store_true', help ="Flip the signs of X and Y gyro data")

args = parser.parse_args()

hawk_data = camio.read_hawk(args.input_file)
hawk_fixed = resampler.proc_all(hawk_data)

if (args.flip_xy):
    print("Flipping signs on X and Y gyro data")
    hawk_fixed[1,:] *= -1
    hawk_fixed[2,:] *= -1
else:
    print("Not flipping signs on X and Y gyro data. You may have to do it later yourself. Or try --flip-xy.")

camio.write_hawk(args.output_file, hawk_fixed)
