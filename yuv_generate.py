import augments
from util import *

outdir = "train/a/"

ww = 1024
hh = 1024


def get_base(clip: vs.VideoNode, frame: int, ri: int):
  clip = augments.color(clip, "base", n)
  if frame % 4 == 0:
    clip = core.std.BoxBlur(clip, [0, 1, 2], hradius=3, vradius=3)
  clip = augments.invert(clip, frame)

  return clip


def get_gt(clip: vs.VideoNode, frame: int, ri: int):
  clip = core.resize.Bicubic(clip, format=vs.YUV444P16)
  return clip


minbanding = 1024
maxbanding = 8192


def get_lq(clip: vs.VideoNode, frame: int, ri: int):
  clip = core.resize.Bicubic(clip, format=vs.YUV444P16)
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
  return clip


def extra_np(f: vs.VideoFrame, frame: int, ri: int):
  #im = np_img_l(f)
  #im = Image.fromarray(im, "L")

  #im = augments.text(im, n)

  #im = np.array(im)
  return f


def extra(clip: vs.VideoNode, frame: int, ri: int):
  clip = augments.color(clip, "extra", ri)

  return clip


def extra_with_mask(clip: vs.VideoNode, frame: int, ri: int):
  clip = augments.zoom(clip, frame, ri, ww, hh)
  return clip


if __name__ == "__main__":
  import sys

  path = sys.argv[1]
  name = sys.argv[3]

  src = source(path)
  src = src[25 * 24:]

  src = core.std.CropAbs(src, ww, hh)

  generate(src,
           outdir,
           name,
           get_base,
           get_lq,
           get_gt,
           extra=extra,
           with_mask=True,
           extra_with_mask=extra_with_mask,
           yuv=True)
