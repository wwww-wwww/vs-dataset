import numpy as np
import os
import random
import vapoursynth as vs
import vstools

core = vs.core

base_seed = random.random()


def get_base():
  return f"{base_seed}-"


def source(file, bits=16):
  src = core.bs.VideoSource(file)
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


def modify_frame(clip, fn, ri):
  return core.std.ModifyFrame(clip, clip, lambda n, f: fn(f, n, ri))


def generate(clip, outdir, name, get_base, get_lq, get_gt, extra, with_mask,
             extra_with_mask, yuv):

  os.makedirs(f"{outdir}gt", exist_ok=True)
  os.makedirs(f"{outdir}lq", exist_ok=True)
  os.makedirs(f"{outdir}val/gt", exist_ok=True)
  os.makedirs(f"{outdir}val/lq", exist_ok=True)

  print(0, len(clip))
  for frame in range(len(clip)):
    out = ""
    if frame == ((len(clip) // 2) - 1):
      out = "val/"

    out_lq = f"{outdir}{out}lq/{name}_{frame:03d}"
    out_gt = f"{outdir}{out}gt/{name}_{frame:03d}"

    skip = True
    if not os.path.exists(out_gt + ".npz"):
      skip = False

    if not os.path.exists(out_lq + ".npz"):
      skip = False

    if skip:
      print(frame + 1, len(clip))
      continue

    ri = random.randint(0, 9999)

    base = core.std.FrameEval(clip, lambda n: get_base(clip, frame, ri))

    gt = get_gt(base, frame, ri)
    lq = core.std.FrameEval(gt, lambda n: get_lq(base, frame, ri))

    gt = extra(gt, frame, ri)
    lq = extra(lq, frame, ri)

    if frame % 5 == 0:
      lq = gt

    #if nn % 3 == 0:
    #lq, _mask = gt_lq_mask(lq, gt)
    #if fn == ((len(clip) // 2) - 1):

    if with_mask:
      gt, mask = gt_lq_mask(lq, gt)

      gt = extra_with_mask(gt, frame, ri)
      lq = extra_with_mask(lq, frame, ri)
      mask = extra_with_mask(mask, frame, ri)
      mask = core.resize.Point(mask, format=vs.GRAYS)

    if gt.format.color_family == vs.YUV and not yuv:
      gt = core.resize.Point(gt, format=vs.RGBS, matrix_in_s="709")
      lq = core.resize.Point(lq, format=vs.RGBS, matrix_in_s="709")
    if gt.format.color_family == vs.RGB and yuv:
      gt = core.resize.Point(gt, format=vs.YUV444P16, matrix_s="709")
      lq = core.resize.Point(lq, format=vs.YUV444P16, matrix_s="709")

    gt = core.std.SplitPlanes(gt)
    lq = core.std.SplitPlanes(lq)

    gt[0] = core.resize.Point(gt[0], format=vs.GRAYS)
    gt[1] = core.resize.Point(gt[1], format=vs.GRAYS)
    gt[2] = core.resize.Point(gt[2], format=vs.GRAYS)

    lq[0] = core.resize.Point(lq[0], format=vs.GRAYS)
    lq[1] = core.resize.Point(lq[1], format=vs.GRAYS)
    lq[2] = core.resize.Point(lq[2], format=vs.GRAYS)

    gt = [
        core.resize.Point(clip, format=vs.GRAY8, dither_type="error_diffusion")
        for clip in gt
    ]

    if with_mask:
      lq = lq + [mask]
    lq = [
        core.resize.Point(clip, format=vs.GRAY8, dither_type="error_diffusion")
        for clip in lq
    ]

    np_gt = [np_img_l(clip.get_frame(frame)) for clip in gt]
    np_gt = np.stack(np_gt, axis=2)

    np_lq = [np_img_l(clip.get_frame(frame)) for clip in lq]
    np_lq = np.stack(np_lq, axis=2)

    np.savez_compressed(out_gt, np_gt)
    np.savez_compressed(out_lq, np_lq)

    print(frame + 1, len(clip))


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
