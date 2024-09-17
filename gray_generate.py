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
  clip = augments.noise(clip, "base" + str(n))
  clip = augments.invert(clip, fn)

  return clip


def get_lq(clip: vs.VideoNode, fn: int, n: int):
  clip = core.resize.Point(clip, format=vs.GRAY16)

  clip = core.asharp.ASharp(clip, d=4, t=random.random() * 4)

  clip = core.resize.Point(clip, format=vs.GRAYS)

  return clip


def get_gt(clip: vs.VideoNode, fn: int, n: int):
  clip = core.resize.Point(clip, format=vs.GRAY16)
  clip = fine_dehalo(clip, brightstr=0.25, darkstr=0.25, edgeproc=0.25)
  clip = core.resize.Point(clip, format=vs.GRAYS)

  return clip


def extra_np(f: vs.VideoFrame, fn: int, n: int):
  im = np_img_l(f)
  im = Image.fromarray(im, "L")

  augments.text(im, n, ry=9)

  im = np.array(im)
  return ndarray_to_frame(im, f.copy())


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
  src = core.std.SplitPlanes(src)[0]
  src = src[25 * 24:]
  src = core.std.CropAbs(src, ww, ww, (1920 - ww) // 2, (1080 - ww) // 2)
  src = core.resize.Point(src, format=vs.GRAYS)

  generate_mask(src, outdir, name, get_base, get_lq, get_gt, extra_np,
                extra_vs, extra_with_mask, ww, hh)
