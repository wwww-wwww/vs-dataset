import augments
from util import *

outdir = "dataset/"

ww = 1024


def get_base(clip, n):
  clip = augments.color(clip, n, False)
  clip = augments.invert(clip, n)

  return clip


if __name__ == "__main__":
  import sys

  name = sys.argv[1]
  path1 = sys.argv[2]
  path2 = sys.argv[3]

  src1 = source(path1)
  src2 = source(path2)

  src1 = core.std.CropAbs(src1, ww, ww, (1920 - ww) // 2, 0)
  src2 = core.std.CropAbs(src2, ww, ww, (1920 - ww) // 2, 0)

  src1 = core.resize.Point(src1, format=vs.YUV420PS)
  src2 = core.resize.Point(src2, format=vs.YUV420PS)

  src1 = core.std.SplitPlanes(src1)[0]
  src2 = core.std.SplitPlanes(src2)[0]

  frames_in = [
      src1[0::3],
      src1[1::3],
      src1[2::3],
  ]
  frames_in = [augments.apply(a, get_base) for a in frames_in]

  src2 = augments.apply(src2, get_base)
  frames_out = [src2[1::3]]

  generate_paired_n(frames_in, frames_out, outdir, name=name)
