from pathlib import Path
from PIL import Image, ImageOps
import fire

def extract(
    imgs,
    w_out = 80,
    h_out = 80,
    w_out_char = 32,
    h_out_char = 48,

    w_char = 32,
    h_char = 48,
    dirs = ['f', 'l', 'r', 'b'],
    poses = ['r', 'c', 'l'],

    n_row = 2,
    n_col = 4):

    def _convert(img):
        trans = img.getpixel((0, 0))
        allTrans = True
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if img.getpixel((i, j)) == trans:
                    img.putpixel((i, j), (0, 0, 0, 0))
                else:
                    allTrans = False
        if allTrans:
            return None
        img = img.resize((w_out_char, h_out_char), Image.NEAREST)
        w_pad = (w_out - img.size[0]) // 2
        h_pad = (h_out - img.size[1]) // 2
        return ImageOps.expand(img, (w_pad, h_pad, w_pad, h_pad))

    n_row_char = len(dirs)
    n_col_char = len(poses)
    assert(w_out >= w_char)
    assert(h_out >= h_char)

    OUT = Path(__file__).parent.parent/'image'/'fsm'/'main'
    OUT.mkdir(parents=True, exist_ok=True)
    for path_str in imgs:
        print(path_str)
        path = Path(path_str)
        for row in range(n_row):
            for col in range(n_col):
                for d_i, d in enumerate(dirs):
                    for p_i, p in enumerate(poses):
                        with Image.open(path) as f:
                            f = f.convert('RGBA')
                            left = col * n_col_char * w_char + p_i * w_char
                            right = col * n_col_char * w_char + (p_i + 1) * w_char
                            upper = row * n_row_char * h_char + d_i * h_char
                            lower = row * n_row_char * h_char + (d_i + 1) * h_char
                            name = "{}-{}-{}-{}-{}{}".format(path.stem, row, col, p, d, path.suffix)
                            
                            img = _convert(f.crop((left, upper, right, lower)))
                            if img is None:
                                continue
                            assert img.size == (w_out, h_out)
                            img.save(OUT/name)

def extract_fsm(
    *imgs,    
    w_out = 80,
    h_out = 80,
    w_out_char = 32,
    h_out_char = 48):
    extract(
        imgs,        
        w_out,
        h_out,
        w_out_char,
        h_out_char,

        w_char = 32,
        h_char = 48,
        dirs = ['f', 'l', 'r', 'b'],
        poses = ['r', 'c', 'l'],

        n_row = 2,
        n_col = 4
    )

def extract_2000(
    *imgs,
    w_out = 80,
    h_out = 80,
    w_out_char = 48,
    h_out_char = 64):

    extract(
        imgs,
        w_out,
        h_out,
        w_out_char,
        h_out_char,

        w_char = 24,
        h_char = 32,
        dirs = ['b', 'r', 'f', 'l'],
        poses = ['r', 'c', 'l'],

        n_row = 2,
        n_col = 4
    )

if __name__ == '__main__':
    fire.Fire()
