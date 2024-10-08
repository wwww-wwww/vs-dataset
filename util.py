import numpy as np
import os
import random
import vapoursynth as vs
from PIL import Image
import vstools

core = vs.core

base_seed = random.random()


def get_base():
  return f"{base_seed}-"


def source(file, bits=16):
  src = core.lsmas.LWLibavSource(file)
  return vstools.initialize_clip(src, bits=bits)


def randchoice(col, *args):
  return col[randint(args) % len(col)]


def randint(*args):
  return abs(hash(get_base() + "-".join([str(c) for c in args])))


def randrange(a, b, *args):
  h = hash(get_base() + "-".join([str(c) for c in args]))
  h = abs(h) % 100000 / 100000
  return int(a + h * (b - a))


def rand(*args):
  h = hash(get_base() + "-".join([str(c) for c in args]))
  h = abs(h) % 100000 / 100000
  return h


def ndarray_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
  for plane in range(frame.format.num_planes):
    np.copyto(np.asarray(frame[plane]), array[:, :])
  return frame


def gt_lq_mask(dirty, clean):
  noise = core.std.BlankClip(clean, color=0.5, format=vs.GRAYS)
  noise = core.noise.Add(noise,
                         type=2,
                         var=255,
                         xsize=150,
                         ysize=150,
                         seed=random.randint(0, 999999))
  noise = core.std.Expr([noise], "x 0.75 pow 1.25 *")

  return core.std.MaskedMerge(dirty, clean, noise), noise


def np_img_rgb(frame: vs.VideoFrame):
  im = np.ctypeslib.as_array(frame)
  im = np.swapaxes(im, 0, 2)
  im = np.swapaxes(im, 0, 1)
  return im


def np_img_rgba(frame: vs.VideoFrame, alpha: vs.VideoFrame):
  im = np_img_rgb(frame)
  im2 = np_img_l(alpha).reshape((alpha.height, alpha.width, 1))
  return np.append(im, im2, axis=2)


def np_img_l(frame: vs.VideoFrame):
  im = np.ctypeslib.as_array(frame)
  im = np.reshape(im, (frame.height, frame.width))
  return im


def np_img_la(frame: vs.VideoFrame, alpha: vs.VideoFrame):
  im = np_img_l(frame)
  im2 = np_img_l(alpha)
  return np.stack([im, im2], axis=2)


def generate(clip, outdir, name, get_base, get_lq, get_gt, extra_np, extra_vs,
             extra_with_mask, width, height):

  os.makedirs(f"{outdir}gt", exist_ok=True)
  os.makedirs(f"{outdir}lq", exist_ok=True)
  os.makedirs(f"{outdir}val/gt", exist_ok=True)
  os.makedirs(f"{outdir}val/lq", exist_ok=True)

  print(0, len(clip))
  for fn in range(len(clip)):
    out = ""
    if fn == ((len(clip) // 2) - 1):
      out = "val/"

    out_gt = f"{outdir}{out}gt/{name}_{fn:03d}.png"
    out_lq = f"{outdir}{out}lq/{name}_{fn:03d}.png"

    skip = True
    if os.path.exists(out_gt):
      try:
        im = Image.open(out_gt)
        im = np.array(im)
      except:
        skip = False
    else:
      skip = False

    if os.path.exists(out_lq):
      try:
        im = Image.open(out_lq)
        im = np.array(im)
      except:
        skip = False
    else:
      skip = False

    if skip:
      print(fn + 1, len(clip))
      continue

    rgb = clip.format.color_family == vs.ColorFamily.RGB

    nn = random.randint(0, 9999)

    clip = core.resize.Bicubic(clip,
                               width,
                               height,
                               filter_param_a=0,
                               filter_param_b=0)
    base = core.std.FrameEval(clip, lambda n: get_base(clip, fn, nn))

    gt = get_gt(base, fn, nn)
    lq = core.std.FrameEval(gt, lambda n: get_lq(base, fn, nn))

    if extra_np:
      gt = core.resize.Point(gt, format=vs.RGB24 if rgb else vs.GRAY8)
      lq = core.resize.Point(lq, format=vs.RGB24 if rgb else vs.GRAY8)

      gt = core.std.ModifyFrame(gt, gt, lambda n, f: extra_np(f, fn, nn))
      lq = core.std.ModifyFrame(lq, lq, lambda n, f: extra_np(f, fn, nn))

      gt = core.resize.Point(gt, format=vs.RGBS if rgb else vs.GRAYS)
      lq = core.resize.Point(lq, format=vs.RGBS if rgb else vs.GRAYS)

    gt = extra_vs(gt, fn, nn)
    lq = extra_vs(lq, fn, nn)

    lq, _mask = gt_lq_mask(lq, gt)
    gt, mask = gt_lq_mask(lq, gt)

    gtout = extra_with_mask(gt, fn, nn)
    lqout = extra_with_mask(lq, fn, nn)
    mask = extra_with_mask(mask, fn, nn)

    gtout = core.resize.Point(gtout,
                              format=vs.RGB24 if rgb else vs.GRAY8,
                              dither_type="error_diffusion")
    lqout = core.resize.Point(lqout,
                              format=vs.RGB24 if rgb else vs.GRAY8,
                              dither_type="error_diffusion")
    mask = core.resize.Point(mask, format=vs.GRAY8)

    if rgb:
      im_gt = np_img_rgb(gtout.get_frame(fn))
      im_lm = np_img_rgba(lqout.get_frame(fn), mask.get_frame(fn))
    else:
      im_gt = np_img_l(gtout.get_frame(fn))
      im_lm = np_img_la(lqout.get_frame(fn), mask.get_frame(fn))

    im_gt = Image.fromarray(im_gt, "RGB" if rgb else "L")
    im_lm = Image.fromarray(im_lm, "RGBA" if rgb else "LA")

    im_gt.save(out_gt)
    im_lm.save(out_lq)
    print(fn + 1, len(clip))


def generate_paired_3_3_mask(srclq,
                             srcgt,
                             outdir,
                             name,
                             get_base,
                             extra_np,
                             extra_vs,
                             extra_with_mask,
                             max_ssimu2=100):

  os.makedirs(f"{outdir}gt", exist_ok=True)
  os.makedirs(f"{outdir}lq", exist_ok=True)
  os.makedirs(f"{outdir}val/gt", exist_ok=True)
  os.makedirs(f"{outdir}val/lq", exist_ok=True)

  if max_ssimu2 < 100:
    ssim = core.ssimulacra2.SSIMULACRA2(srcgt, srclq)

  print(0, len(srclq))
  for fn in range(len(srclq)):
    if max_ssimu2 < 100:
      score = ssim.get_frame(fn).props["_SSIMULACRA2"]
      if score > max_ssimu2: continue
      if score < 50: continue

    out = ""
    if fn == ((len(srclq) // 2) - 1):
      out = "val/"

    out_lq = f"{outdir}{out}lq/{name}_{fn:03d}.png"
    out_gt = f"{outdir}{out}gt/{name}_{fn:03d}.png"

    skip = True
    if os.path.exists(out_gt):
      try:
        im = Image.open(out_gt)
        im = np.array(im)
      except:
        skip = False
    else:
      skip = False

    if os.path.exists(out_lq):
      try:
        im = Image.open(out_lq)
        im = np.array(im)
      except:
        skip = False
    else:
      skip = False

    if skip:
      print(fn + 1, len(srclq))
      continue

    rgb = srclq.format.color_family == vs.ColorFamily.RGB

    nn = random.randint(0, 9999)

    lq = core.std.FrameEval(srclq, lambda n: get_base(srclq, fn, nn))
    gt = core.std.FrameEval(srcgt, lambda n: get_base(srcgt, fn, nn))

    if extra_np:
      lq = core.resize.Point(lq, format=vs.RGB24 if rgb else vs.GRAY8)
      gt = core.resize.Point(gt, format=vs.RGB24 if rgb else vs.GRAY8)

      lq = core.std.ModifyFrame(lq, lq, lambda n, f: extra_np(f, fn, nn))
      gt = core.std.ModifyFrame(gt, gt, lambda n, f: extra_np(f, fn, nn))

      lq = core.resize.Point(lq, format=vs.RGBS if rgb else vs.GRAYS)
      gt = core.resize.Point(gt, format=vs.RGBS if rgb else vs.GRAYS)

    lq = extra_vs(lq, fn, nn)
    gt = extra_vs(gt, fn, nn)

    lq, _mask = gt_lq_mask(lq, gt)  # mix gt in lq
    gt, mask = gt_lq_mask(lq, gt)

    lqout = extra_with_mask(lq, fn, nn)
    gtout = extra_with_mask(gt, fn, nn)
    mask = extra_with_mask(mask, fn, nn)

    lqout = core.resize.Point(lqout,
                              format=vs.RGB24 if rgb else vs.GRAY8,
                              dither_type="error_diffusion")
    gtout = core.resize.Point(gtout,
                              format=vs.RGB24 if rgb else vs.GRAY8,
                              dither_type="error_diffusion")
    mask = core.resize.Point(mask, format=vs.GRAY8)

    if rgb:
      im_lm = np_img_rgba(lqout.get_frame(fn), mask.get_frame(fn))
      im_gt = np_img_rgb(gtout.get_frame(fn))
    else:
      im_lm = np_img_la(lqout.get_frame(fn), mask.get_frame(fn))
      im_gt = np_img_l(gtout.get_frame(fn))

    im_lm = Image.fromarray(im_lm, "RGBA" if rgb else "LA")
    im_gt = Image.fromarray(im_gt, "RGB" if rgb else "L")

    im_lm.save(out_lq)
    im_gt.save(out_gt)
    print(fn + 1, len(srclq))


def generate_paired_3_1(clip1, clip2, outdir, name, get_base, extra_np,
                        extra_vs, extra_with_mask):

  os.makedirs(f"{outdir}gt", exist_ok=True)
  os.makedirs(f"{outdir}lq", exist_ok=True)
  os.makedirs(f"{outdir}val/gt", exist_ok=True)
  os.makedirs(f"{outdir}val/lq", exist_ok=True)

  print(0, len(clip1))
  for fn in range(len(clip1)):
    out = ""
    if fn == ((len(clip1) // 2) - 1):
      out = "val/"

    out_lq = f"{outdir}{out}lq/{name}_{fn:03d}.png"
    out_gt = f"{outdir}{out}gt/{name}_{fn:03d}.png"

    skip = True
    if os.path.exists(out_gt):
      try:
        im = Image.open(out_gt)
        im = np.array(im)
      except:
        skip = False
    else:
      skip = False

    if os.path.exists(out_lq):
      try:
        im = Image.open(out_lq)
        im = np.array(im)
      except:
        skip = False
    else:
      skip = False

    if skip:
      print(fn + 1, len(clip1))
      continue

    nn = random.randint(0, 9999)

    lq = core.std.FrameEval(clip1, lambda n: get_base(clip1, fn, nn))
    gt = core.std.FrameEval(clip2, lambda n: get_base(clip2, fn, nn))

    if extra_np:
      lq = core.resize.Point(lq, format=vs.RGB24)
      gt = core.resize.Point(gt, format=vs.GRAY8)

      lq = core.std.ModifyFrame(lq, lq, lambda n, f: extra_np(f, fn, nn))
      gt = core.std.ModifyFrame(gt, gt, lambda n, f: extra_np(f, fn, nn))

      lq = core.resize.Point(lq, format=vs.RGBS)
      gt = core.resize.Point(gt, format=vs.GRAYS)

    lq = extra_vs(lq, fn, nn)
    gt = extra_vs(gt, fn, nn)

    lqout = extra_with_mask(lq, fn, nn)
    gtout = extra_with_mask(gt, fn, nn)

    lqout = core.resize.Point(lqout,
                              format=vs.RGB24,
                              dither_type="error_diffusion")
    gtout = core.resize.Point(gtout,
                              format=vs.GRAY8,
                              dither_type="error_diffusion")

    fr = lqout.get_frame(fn)
    im_lq = np_img_rgb(fr)
    im_gt = np_img_l(gtout.get_frame(fn))

    im_lq = Image.fromarray(im_lq, "RGB")
    im_gt = Image.fromarray(im_gt, "L")

    im_lq.save(out_lq)
    im_gt.save(out_gt)
    print(fn + 1, len(clip1))


def generate_paired_n(frames_in, frames_out, outdir, name):
  os.makedirs(f"{outdir}gt", exist_ok=True)
  os.makedirs(f"{outdir}lq", exist_ok=True)
  os.makedirs(f"{outdir}val/gt", exist_ok=True)
  os.makedirs(f"{outdir}val/lq", exist_ok=True)

  if type(frames_out) == list:
    num_frames = len(frames_out[0])
  else:
    num_frames = len(frames_out)

  print(0, num_frames)
  for fn in range(num_frames):
    out = ""
    if fn == ((num_frames // 2) - 1):
      out = "val/"

    out_lq = f"{outdir}{out}lq/{name}_{fn:03d}"
    out_gt = f"{outdir}{out}gt/{name}_{fn:03d}"

    skip = True
    if not os.path.exists(out_gt + ".npz"):
      skip = False

    if not os.path.exists(out_lq + ".npz"):
      skip = False

    if skip:
      print(fn + 1, num_frames)
      continue

    #if extra_np:
    #  lq = core.resize.Point(lq, format=vs.RGB24)
    #  gt = core.resize.Point(gt, format=vs.GRAY8)

    #  lq = core.std.ModifyFrame(lq, lq, lambda n, f: extra_np(f, fn, nn))
    #  gt = core.std.ModifyFrame(gt, gt, lambda n, f: extra_np(f, fn, nn))

    #  lq = core.resize.Point(lq, format=vs.RGBS)
    #  gt = core.resize.Point(gt, format=vs.GRAYS)

    # lq = extra_vs(lq, fn, nn)
    # gt = extra_vs(gt, fn, nn)

    # lqout = extra_with_mask(lq, fn, nn)
    # gtout = extra_with_mask(gt, fn, nn)

    lq = [
        core.resize.Point(clip, format=vs.GRAY8, dither_type="error_diffusion")
        for clip in frames_in
    ]

    gt = [
        core.resize.Point(clip, format=vs.GRAY8, dither_type="error_diffusion")
        for clip in frames_out
    ]

    np_lq = [np_img_l(clip.get_frame(fn)) for clip in lq]
    np_lq = np.stack(np_lq, axis=2)

    np_gt = [np_img_l(clip.get_frame(fn)) for clip in gt]
    np_gt = np.stack(np_gt, axis=2)

    np.savez_compressed(out_lq, np_lq)
    np.savez_compressed(out_gt, np_gt)

    print(fn + 1, num_frames)
