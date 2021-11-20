import camio
import resampler
import sys


hawk_data = camio.read_hawk(sys.argv[1])
hawk_fixed = resampler.proc_all(hawk_data)
camio.write_hawk(sys.argv[2], hawk_fixed)
