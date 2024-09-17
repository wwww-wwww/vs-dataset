import augments
from util import *

outdir = "dataset/"

ww = 1024
hh = 1024


def get_base(clip: vs.VideoNode, fn: int, n: int):
  clip = augments.color(clip, n)
  clip = augments.noise(clip, "base" + str(n), 2)
  clip = augments.invert(clip, fn)

  return clip


def extra_vs(clip: vs.VideoNode, fn: int, n: int):
  return clip


def extra_with_mask(clip: vs.VideoNode, fn: int, n: int):
  clip = augments.zoom(clip, fn, n, ww, hh)

  return clip


if __name__ == "__main__":
  import sys

  name = sys.argv[1]
  path1 = sys.argv[2]
  path2 = sys.argv[3]

  src1 = source(path1)
  src1 = src1[1::3]
  src1 = core.std.CropAbs(src1, ww, ww, (1920 - ww) // 2, (1080 - ww) // 2)
  src1 = core.resize.Point(src1, format=vs.RGBS)

  src2 = source(path2)
  src2 = src2[1::3]
  src2 = core.std.CropAbs(src2, ww, ww, (1920 - ww) // 2, (1080 - ww) // 2)
  src2 = core.resize.Point(src2, format=vs.RGBS)

  generate_paired_3_3_mask(src1, src2, outdir, name, get_base, None, extra_vs,
                           extra_with_mask, 88)
