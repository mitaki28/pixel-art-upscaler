from pathlib import Path
from PIL import Image, ImageOps
import fire
from pixcaler.util import transparent_background

def extract(out_dir, *imgs):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    for path in map(Path, imgs):
        out_path = out_dir/path.name
        with Image.open(path) as img:
            transparent_background(img.convert('RGBA')).convert('RGBA').save(out_path)
            print(path, '->', out_path)


if __name__ == '__main__':
    fire.Fire(extract)
