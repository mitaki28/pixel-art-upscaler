import * as Jimp from "jimp";

const TRANSPARENT_COLOR = 0x00000000;

function bilinear(v0: number, v1: number, d: number) {
    return (v0 + (v1 - v0) * d);
}
function bilinearRGBA(v0: Jimp.RGBA, v1: Jimp.RGBA, d: number): Jimp.RGBA {
    return {
        r: bilinear(v0.r, v1.r, d),
        g: bilinear(v0.g, v1.g, d),
        b: bilinear(v0.b, v1.b, d),
        a: bilinear(v0.a, v1.a, d),
    };
}

function intToRgba(c: number) {
    return {
        r: (c >> 24) & 0xff,
        g: (c >> 16) & 0xff,
        b: (c >> 8) & 0xff,
        a: (c >> 0) & 0xff,
    };
}


function rgbaToInt(c: Jimp.RGBA) {
    return (c.r << 24) | (c.g << 16) | (c.b << 8) | (c.a << 0);
}

export function resizeBilinear(img: Jimp.Jimp, w: number, h: number) {
    const ret: Jimp.Jimp = new (Jimp.default as any)(w, h);
    const [fx, fy] = [img.bitmap.width / w, img.bitmap.height / h];
    for (let x = 0; x < w; x++) {
        for (let y = 0; y < h; y++) {
            const [ox, oy] = [fx * (x + 0.5) - 0.5, fy * (y + 0.5) - 0.5];
            const [x0, y0] = [Math.max(Math.floor(ox), 0), Math.max(Math.floor(oy), 0)];
            const [x1, y1] = [Math.min(x0 + 1, img.bitmap.width - 1), Math.min(y0 + 1, img.bitmap.height - 1)];
            const [dx, dy] = [ox - x0, oy - y0];

            const c0 = bilinearRGBA(
                intToRgba(img.getPixelColor(x0, y0)),
                intToRgba(img.getPixelColor(x1, y0)),
                dx,
            );
            const c1 = bilinearRGBA(
                intToRgba(img.getPixelColor(x0, y1)),
                intToRgba(img.getPixelColor(x1, y1)),
                dx,
            );
            const c = bilinearRGBA(c0, c1, dy);
            ret.setPixelColor(rgbaToInt(c), x, y);    
        }
    }
    return ret;
}

export function transparentBackgroundColor(img: Jimp.Jimp) {
    img.background(TRANSPARENT_COLOR);
    const backgroundColor = img.getPixelColor(0, 0);
    for (let x = 0; x < img.bitmap.width; x++) {
        for (let y = 0; y < img.bitmap.height; y++) {
            if (img.getPixelColor(x, y) === backgroundColor) {
                img.setPixelColor(TRANSPARENT_COLOR, x, y);
            }
        }
    }
}

export function adjustSizeToPowerOf2(img: Jimp.Jimp, minimumSize: number) {
    let r = 1;
    while (r < img.bitmap.width || r < img.bitmap.height || r < minimumSize) {
        r *= 2;
    }
    img.contain(r, r);
    return r;
}

export function imageToChwFloat32Array(img: Jimp.Jimp, channel: number): Float32Array {
    const ret = new Float32Array(img.bitmap.height * img.bitmap.width * 4);
    const h = img.bitmap.width;
    const w = img.bitmap.height;
    img.scan(0, 0, w, h, (x, y, idx) => {
        for (let c = 0; c < channel; c++) {
            ret[c * (h * w) + y * w + x] = img.bitmap.data[idx + c];
        }
    });
    return ret;
}

export function chwFloat32ArrayToImage(array: Float32Array, channel: number, h: number, w: number): Jimp.Jimp {
    const ret: Jimp.Jimp = new (Jimp.default as any)(w, h);
    ret.scan(0, 0, w, h, (x, y, idx) => {
        for (let c = 0; c < channel; c++) {
            ret.bitmap.data[idx + c] = Math.round(array[c * (h * w) + y * w + x]);
        }
    });
    return ret;
}


export function imageToHwcFloat32Array(img: Jimp.Jimp, channel: number): Float32Array {
    const ret = new Float32Array(img.bitmap.height * img.bitmap.width * channel);
    const h = img.bitmap.height;
    const w = img.bitmap.width;
    img.scan(0, 0, w, h, (x, y, idx) => {
        for (let c = 0; c < channel; c++) {
            ret[y * (w * channel) + x * channel + c] = img.bitmap.data[idx + c];
        }
    });
    return ret;
}

export function hwcFloat32ArrayToImage(array: Float32Array, channel: number, h: number, w: number): Jimp.Jimp {
    const ret: Jimp.Jimp = new (Jimp.default as any)(w, h);
    ret.scan(0, 0, w, h, (x, y, idx) => {
        for (let c = 0; c < channel; c++) {
            ret.bitmap.data[idx + c] = Math.floor(array[y * (w * channel) + x * channel + c]);
        }
    });
    return ret;
}