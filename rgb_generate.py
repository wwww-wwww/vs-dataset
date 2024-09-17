from vs import *
import numpy as np
from PIL import Image

import augments
from util import *

outdir = "dataset/"

ww = 1024
hh = 1024


def get_base(clip: vs.VideoNode, fn: int, n: int):
  clip = augments.color(clip, n)
  clip = augments.noise(clip, "base" + str(n), 4)
  clip = augments.invert(clip, fn)

  return clip


minbanding = 1024
maxbanding = 8192


def get_lq(clip: vs.VideoNode, fn: int, n: int):
  clip = core.resize.Point(clip, format=vs.YUV444P16, matrix_s="709")

  planes = core.std.SplitPlanes(clip)

  bstr = random.randint(minbanding, maxbanding) / 2
  planes[0] = core.std.Expr([planes[0]], f"x {bstr // 4} - {bstr} /")
  planes[0] = core.std.Limiter(planes[0])
  planes[0] = core.std.Expr([planes[0]], f"x {bstr} * {bstr // 4} +")
  bstr = random.randint(minbanding, maxbanding) / 2
  planes[1] = core.std.Expr([planes[1]], f"x {bstr // 4} - {bstr} /")
  planes[1] = core.std.Limiter(planes[1])
  planes[1] = core.std.Expr([planes[1]], f"x {bstr} * {bstr // 4} +")
  bstr = random.randint(minbanding, maxbanding) / 2
  planes[2] = core.std.Expr([planes[2]], f"x {bstr // 4} - {bstr} /")
  planes[2] = core.std.Limiter(planes[2])
  planes[2] = core.std.Expr([planes[2]], f"x {bstr} * {bstr // 4} +")

  clip = core.std.ShufflePlanes(planes, [0, 0, 0], vs.YUV)

  clip = core.resize.Point(clip, format=vs.RGBS, matrix_in_s="709")

  return clip


def get_gt(clip: vs.VideoNode, fn: int, n: int):
  return clip


def extra_np(f: vs.VideoFrame, fn: int, n: int):
  return f


def extra_vs(clip: vs.VideoNode, fn: int, n: int):
  clip = augments.noise(clip, "extra_vs" + str(n))

  return clip


def extra_with_mask(clip: vs.VideoNode, fn: int, n: int):
  clip = augments.zoom(clip, fn, n, ww, hh)

  return clip


if __name__ == "__main__":
  import sys

  name = sys.argv[1]
  path = sys.argv[2]

  src = source(path)
  src = src[25 * 24:]
  src = core.std.CropAbs(src, ww, ww, (1920 - ww) // 2, (1080 - ww) // 2)
  src = core.resize.Point(src, format=vs.RGBS)

  generate_mask(src, outdir, name, get_base, get_lq, get_gt, extra_np,
                extra_vs, extra_with_mask, ww, hh)
