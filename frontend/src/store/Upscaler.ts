
import { observable, computed, action } from "mobx";
import { Converter, ConversionError } from "./Converter";
import * as Jimp from "jimp";
import * as WebDNN from "webdnn";
import SymbolicFloat32Array from "webdnn/symbolic_typed_array/symbolic_float32array";
import { Either, left, right } from "fp-ts/lib/Either";
import { transparentBackgroundColor, adjustSizeToPowerOf2, imageToChwFloat32Array, chwFloat32ArrayToImage, imageToHwcFloat32Array, hwcFloat32ArrayToImage } from "../util/image";

const MIN_SIZE = 32;
const MAX_HEIGHT = 32;
const MAX_WIDTH = 32;

export interface Upscaler extends Converter {

}

export class WebDNNUpscaler implements Upscaler {

    private _runner: WebDNN.DescriptorRunner;
    private _inputImage: SymbolicFloat32Array;
    private _outputImage: SymbolicFloat32Array;

    constructor(runner: WebDNN.DescriptorRunner) {
        this._runner = runner;
        this._inputImage = runner.inputs[0];
        this._outputImage = runner.outputs[0];
    }

    @action.bound
    async convert(src: string): Promise<Either<ConversionError, string>> {
        let img: Jimp.Jimp;
        try {
            img = await Jimp.read(src);
        } catch (e) {
            return left(ConversionError.failedToLoad(e));
        }

        const srcHeight = img.bitmap.height;
        const srcWidth = img.bitmap.width;
        if (srcHeight > MAX_HEIGHT || srcWidth > MAX_WIDTH) {
            return left(ConversionError.tooLarge({ width: MAX_WIDTH, height: MAX_HEIGHT }))
        }

        transparentBackgroundColor(img);
        const r = adjustSizeToPowerOf2(img, MIN_SIZE);
        const preprocessedSize = r * 2;
        img.resize(preprocessedSize, preprocessedSize, Jimp.RESIZE_NEAREST_NEIGHBOR);
        img.contain(preprocessedSize + 1, preprocessedSize + 1, Jimp.HORIZONTAL_ALIGN_RIGHT | Jimp.VERTICAL_ALIGN_BOTTOM);
        img.crop(0, 0, preprocessedSize, preprocessedSize);

        const x = imageToChwFloat32Array(img, 4).map((v) => {
            return v / 127.5 - 1.0;
        });

        this._inputImage.set(x);
        try {
            await this._runner.run();
        } catch (e) {
            return left(ConversionError.failedToConvert(e));
        }
        const y = this._outputImage.toActual().map((v) => Math.min(Math.max((v + 1.0) * 127.5, 0.0), 255.0));
        const convertedHeight = 2 * srcHeight;
        const convertedWidth = 2 * srcWidth;
        const convertedImage = chwFloat32ArrayToImage(y, 4, preprocessedSize, preprocessedSize).crop(
            (preprocessedSize - convertedWidth) / 2,
            (preprocessedSize - convertedHeight) / 2,
            convertedWidth,
            convertedHeight,
        );;

        return new Promise<Either<ConversionError, string>>((resolve, reject) => {
            (convertedImage as any).getBase64(Jimp.MIME_PNG, (error: any, dst: string) => {
                if (error) {
                    reject(error);
                }
                resolve(right<ConversionError, string>(dst));
            });
        });
    }
}

export namespace UpscalerLoadingState {
    export const PENDING = Symbol("PENDING");
    export const LOADING = Symbol("LOADING");
    export const LOADED = Symbol("LOADED");
    export const LOAD_FAILURE = Symbol("LOAD_FAILURE");
}
export type UpscalerLoadingState =
    {
        status: typeof UpscalerLoadingState.PENDING;
    } | {
        status: typeof UpscalerLoadingState.LOADING;
    } | {
        status: typeof UpscalerLoadingState.LOADED;
        value: Upscaler;
    } | {
        status: typeof UpscalerLoadingState.LOAD_FAILURE;
        error: Error;
    }

export abstract class UpscalerLoader {

    @observable private _state: UpscalerLoadingState;
    private _upscaler: null | Promise<Upscaler>;

    constructor() {
        this._upscaler = null;
        this._state = {
            status: UpscalerLoadingState.PENDING,
        };
    }

    @computed
    get state() {
        return this._state;
    }

    @action.bound
    private startLoading(): Promise<Upscaler> {
        this._state = {
            status: UpscalerLoadingState.LOADING,
        };
        return this.load().then((upscaler) => {
            this.finishLoading(upscaler);
            return upscaler;
        }).catch((e: any) => {
            this.failLoading(e);
            throw e;
        });
    }

    protected abstract load(): Promise<Upscaler>;

    @action.bound
    private finishLoading(value: Upscaler) {
        this._state = {
            status: UpscalerLoadingState.LOADED,
            value,
        }
    }

    @action.bound
    private failLoading(error: Error) {
        this._state = {
            status: UpscalerLoadingState.LOAD_FAILURE,
            error,
        }
        this._upscaler = null;
    }

    @action.bound
    ready(): Promise<Upscaler> {
        if (this._upscaler !== null) {
            return this._upscaler;
        }
        return this._upscaler = this.startLoading();
    }
}


export class WebDNNUpscalerLoader extends UpscalerLoader {
    constructor() {
        super();
    }
    @action.bound
    protected load(): Promise<Upscaler> {
        return WebDNN.load("./model/webdnn").then((runner) => {
            return new WebDNNUpscaler(runner);
        })
    }
}
