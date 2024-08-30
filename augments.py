from util import *
from PIL import ImageDraw, ImageFont, Image
from fontTools.ttLib import TTFont

fonts = [
    "FOT-TsukuMinPro-B.otf",
    "FOT-TsukuGoPro-H.otf",
    "FOT-TsukuARdGothicStd-M.otf",
    "FOT-TsukuAOldMinPr6N-L.otf",
    "FOT-TsukuAOldMinPr6N-L.otf",
    "FOT-TsukuAOldMinPr6N-L.otf",
]
fonts = [
    ImageFont.truetype(font, size) for font in fonts for size in range(40, 80)
]

tfont = TTFont("FOT-TsukuMinPro-B.otf")


def get_glyphs(font):
  glyphs = []
  for table in font["cmap"].tables:
    if table.isUnicode():
      glyphs.extend(table.cmap.keys())
  return glyphs


alphabet = get_glyphs(tfont)
alphabet2 = [i for i in range(33, 127)]


def get_random_text(alph, length, seed):
  random.seed(seed)
  chars = [random.choice(alph) for i in range(length)]
  chars = [chr(c) for c in chars]
  return "".join(chars)


def text(im: Image, n, rx=7, ry=7):
  canvas = Image.new("LA", (im.width * 2, im.height * 2), (0, 0))
  draw = ImageDraw.Draw(canvas)
  px = canvas.width // rx
  py = canvas.height // ry
  for y in range(ry):
    for x in range(rx):
      text = get_random_text(alphabet, 5, randint("text1", n, x, y))
      text += get_random_text(alphabet2, 5, randint("text2", n, x, y))
      text = list(text)
      random.shuffle(text)
      text = "".join(text)

      fill = randint("textfill", n, x, y) % 256
      stroke_fill = randint("textstroke", n, x, y) % 256
      stroke_width = max(0, randint("textstrokewidth", n, x, y) % 4 - 1)
      fill2 = randint("boxfill", n, x, y) % 256

      xx = px * x
      yy = py * y
      draw.rectangle((
          px / 4 + xx - 10,
          py / 4 + yy - 10,
          px / 4 + xx + 200,
          py / 4 + yy + 70,
      ),
                     fill=(fill2, 255))

      font = randchoice(fonts, "font", n, x, y)

      if randint("shadow", n, x, y) % 2 == 0:
        sx = randint("shadowx", n, x, y) % 3
        sy = randint("shadowy", n, x, y) % 3

        draw.text((px / 4 + xx + sx, py / 4 + yy + sy),
                  text,
                  font=font,
                  fill=(0, 255),
                  stroke_width=0)

      draw.text((px / 4 + xx, py / 4 + yy),
                text,
                font=font,
                fill=(fill, 255),
                stroke_fill=(stroke_fill, 255),
                stroke_width=stroke_width)

  canvas = canvas.resize((im.width, im.height))
  im.paste(canvas, (0, 0), canvas)
  return im


def color(clip: vs.VideoNode, seed: int):
  rgb = clip.format.color_family == vs.ColorFamily.RGB
  if rgb:
    planes = core.std.SplitPlanes(clip)
    for p in range(len(planes)):
      gam = 0.75 + 0.5 * rand("gam", seed, p)
      mul = 0.75 + 0.5 * rand("mul", seed, p)
      off = -0.25 + 0.5 * rand("off", seed, p)
      planes[p] = core.std.Expr([planes[p]], f"x {gam} pow {mul} * {off} +")
    clip = core.std.ShufflePlanes(planes, [0, 0, 0], vs.RGB)
  else:
    gam = 0.75 + 0.5 * rand("gam", seed)
    mul = 0.75 + 0.5 * rand("mul", seed)
    off = -0.25 + 0.5 * rand("off", seed)
    clip = core.std.Expr([clip], f"x {gam} pow {mul} * {off} +")
  return clip


def noise(clip: vs.VideoNode, seed: str, strength: int = 6):
  rgb = clip.format.color_family == vs.ColorFamily.RGB
  if rgb:
    clip = core.resize.Bicubic(clip, format=vs.YUV444P16, matrix_s="709")
  else:
    clip = core.resize.Bicubic(clip, format=vs.GRAY16)

  noise_str = strength * rand("base noise strength", seed)**3
  noise_xsize = 2 + 14 * rand("base noise xsize", seed)
  noise_ysize = 0.9 + 0.2 * rand("base noise ysize", seed)
  noise_seed = randint("base noise seed", seed)

  clip = core.noise.Add(clip,
                        noise_str,
                        type=2,
                        xsize=noise_xsize,
                        ysize=noise_ysize,
                        seed=noise_seed,
                        constant=1)

  if rgb:
    clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
  else:
    clip = core.resize.Bicubic(clip, format=vs.GRAYS)

  return clip


def invert(clip: vs.VideoNode, fn: int):
  if fn % 9 < 2 and fn != ((len(clip) // 2) - 1):
    clip = core.std.Invert(clip)
  return clip


#apply to lq, gt, mask
def zoom(clip: vs.VideoNode, fn: int, n: int, ww: int, hh: int):
  if fn % 5 == 0:
    height = randrange(720, 900, "zoom", n)
    width = round(height * (16 / 9))
    clip = core.std.CropAbs(clip,
                            width,
                            height,
                            left=(ww - width) // 2,
                            top=(hh - height) // 2)
    clip = core.resize.Bicubic(clip, width=ww, height=hh)

  return clip
