import * as Jimp from "jimp";
import * as tf from '@tensorflow/tfjs';
import { imageToHwcFloat32Array, hwcFloat32ArrayToImage } from "../util/image";
export class DataUrlImage {
    private constructor(
        public readonly dataUrl: string,
        public readonly width: number,
        public readonly height: number,
    ) { }
    async toJimp() {
        return await Jimp.read(this.dataUrl);
    }
    async toTf() {
        const img = await this.toJimp();
        return tf.tensor(imageToHwcFloat32Array(img, 4), [img.bitmap.height, img.bitmap.width, 4], "float32");
    }
    static async fromTf(tensor: tf.Tensor) {
        return await this.fromJimp(hwcFloat32ArrayToImage(
            await tensor.flatten().data() as Float32Array,
            tensor.shape[2],
            tensor.shape[0],
            tensor.shape[1],
        ));
    }
    static async create(dataUrl: string) {
        const img = await Jimp.read(dataUrl);
        return new DataUrlImage(dataUrl, img.bitmap.width, img.bitmap.height);
    }
    static fromFile(file: File) {
        return new Promise<DataUrlImage>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                resolve(DataUrlImage.create(reader.result));
            };
            reader.onerror = (ev) => {
                reject((ev as any).error);
            }
            reader.readAsDataURL(file);
        });
    }
    static fromJimp(img: Jimp.Jimp) {
        return new Promise<DataUrlImage>((resolve, reject) => {
            (img as any).getBase64(Jimp.MIME_PNG, (error: any, dst: string) => {
                if (error) {
                    reject(error);
                }
                resolve(new DataUrlImage(dst, img.bitmap.width, img.bitmap.height));
            });
        });
    }
    async withJimp(f: (jimp: Jimp.Jimp) => Promise<void> | void): Promise<DataUrlImage> {
        const jimp = await this.toJimp();
        await f(jimp);
        return DataUrlImage.fromJimp(jimp);
    }
}