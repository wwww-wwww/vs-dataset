import augments
from util import *

outdir = "dataset/"

ww = 1024


def get_base(clip, n):
  clip = augments.color(clip, n, True)
  clip = augments.invert(clip, n)

  return clip


if __name__ == "__main__":
  import sys

  name = sys.argv[1]
  path1 = sys.argv[2]
  path2 = sys.argv[3]

  src1 = source(path1)
  src2 = source(path2)

  src1 = src1[1::3]
  src2 = src2[1::3]

  src1 = core.std.CropAbs(src1, ww, ww, (1920 - ww) // 2, 0)
  src2 = core.std.CropAbs(src2, ww, ww, (1920 - ww) // 2, 0)

  src1 = core.resize.Point(src1, format=vs.RGBS, matrix_in_s="709")
  src2 = core.resize.Point(src2, format=vs.RGBS, matrix_in_s="709")

  src1 = augments.apply(src1, get_base)
  src2 = augments.apply(src2, get_base)

  frames_in = core.std.SplitPlanes(src1)
  frames_out = core.std.SplitPlanes(src2)

  generate_paired_n(frames_in, frames_out, outdir, name=name)
