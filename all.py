import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import hashlib

script = "gray_generate.py"

if __name__ == "__main__":
  files = os.listdir("src")
  files = [f for f in files if f.endswith(".mp4") or f.endswith(".mkv")]

  files = [(os.path.join("src", f), "") for f in files]

  #files = [(os.path.join("src", f), os.path.join("gt", f)) for f in files]


  def do(a, b, total, progress):
    n = os.path.basename(a)
    n = n[:2] + n[-6:-4] + hashlib.md5(n.encode("utf-8")).hexdigest()
    t = progress.add_task(n, total=1000)

    cmd = ["python", "-u", script, n, a, b]
    #print(" ".join(cmd))
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL,
                         universal_newlines=True,
                         creationflags=0x00008000)
    while True:
      line = p.stdout.readline().strip()

      if len(line) == 0 and p.poll() is not None:
        break

      line = line.split(" ")
      if len(line) == 2:
        frame = int(line[0])
        frames = int(line[1])
        progress.update(t, completed=frame, total=frames)

    progress.remove_task(t)
    progress.update(total, advance=1)

  with Progress(
      "[progress.description]{task.description}",
      BarColumn(None),
      "[progress.percentage]{task.percentage:>3.0f}%",
      "[progress.download]{task.completed}/{task.total}",
      TimeElapsedColumn(),
      TimeRemainingColumn(),
      expand=True,
  ) as progress:
    total = progress.add_task(script, total=len(files))

    with ThreadPoolExecutor(6) as executor:
      for f in files:
        executor.submit(do, f[0], f[1], total, progress)
