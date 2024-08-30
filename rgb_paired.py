import augments
from util import *

outdir = "dataset/"

ww = 1920
hh = 1080


def get_base(clip: vs.VideoNode, fn: int, n: int):
  clip = augments.color(clip, n)
  clip = augments.noise(clip, "base" + str(n), 4)
  clip = augments.invert(clip, fn)

  return clip


def extra1(f: vs.VideoFrame, fn: int, n: int):
  return f


def extra2(clip: vs.VideoNode, fn: int, n: int):
  clip = augments.noise(clip, "extra2" + str(n), 1)

  return clip


def extra_final(clip: vs.VideoNode, fn: int, n: int):
  clip = augments.zoom(clip, fn, n, ww, hh)

  return clip


if __name__ == "__main__":
  import sys

  path = sys.argv[1]
  path2 = sys.argv[2]
  name = sys.argv[3]

  src1 = source(path)
  src1 = src1[1::3]
  src1 = core.dfttest.DFTTest(src1, sigma=2)
  src1 = core.resize.Point(src1, format=vs.RGBS)

  src2 = source(path2)
  src2 = src2[1::3]
  src2 = core.dfttest.DFTTest(src2, sigma=2)
  src2 = core.resize.Point(src2, format=vs.RGBS)

  generate_paired(src1, src2, outdir, name, get_base, None, extra2,
                  extra_final)
