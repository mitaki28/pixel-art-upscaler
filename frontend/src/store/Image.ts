import * as Jimp from "jimp";
export class DataUrlImage {
    constructor(public readonly dataUrl: string) {}
    async toJimp() {
        return await Jimp.read(this.dataUrl);
    }
    static fromFile(file: File) {
        return new Promise<DataUrlImage>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                resolve(new DataUrlImage(reader.result));
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
                resolve(new DataUrlImage(dst));
            });
        });
    }
    async withJimp(f: (jimp: Jimp.Jimp) => Promise<void> | void): Promise<DataUrlImage> {
        const jimp = await this.toJimp();
        await f(jimp);
        return DataUrlImage.fromJimp(jimp);
    }
}