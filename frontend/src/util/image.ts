import * as Jimp from "jimp";

const TRANSPARENT_COLOR = 0x00000000;

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
    ret.filterType(Jimp.PNG_FILTER_NONE);
    ret.deflateLevel(0);    
    ret.scan(0, 0, w, h, (x, y, idx) => {
        for (let c = 0; c < channel; c++) {
            ret.bitmap.data[idx + c] = Math.round(array[c * (h * w) + y * w + x]);
        }
    });
    return ret;
}


export function imageToHwcFloat32Array(img: Jimp.Jimp, channel: number): Float32Array {
    const ret = new Float32Array(img.bitmap.height * img.bitmap.width * 4);
    const h = img.bitmap.width;
    const w = img.bitmap.height;
    img.scan(0, 0, w, h, (x, y, idx) => {
        for (let c = 0; c < channel; c++) {
            ret[y * (w * channel) + x * channel + c] = img.bitmap.data[idx + c];
        }
    });
    return ret;
}

export function hwcFloat32ArrayToImage(array: Float32Array, channel: number, h: number, w: number): Jimp.Jimp {
    const ret: Jimp.Jimp = new (Jimp.default as any)(w, h);
    ret.filterType(Jimp.PNG_FILTER_NONE);
    ret.deflateLevel(0);
    ret.scan(0, 0, w, h, (x, y, idx) => {
        for (let c = 0; c < channel; c++) {
            ret.bitmap.data[idx + c] = Math.floor(array[y * (w * channel) + x * channel + c]);
        }
    });
    return ret;
}