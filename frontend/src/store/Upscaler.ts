
import { observable, computed, action } from "mobx";
import * as Jimp from "jimp";
import * as WebDNN from "webdnn";
import * as KerasJS from "keras-js";
import * as tf from '@tensorflow/tfjs';
import SymbolicFloat32Array from "webdnn/symbolic_typed_array/symbolic_float32array";
import { Either, left, right } from "fp-ts/lib/Either";
import { transparentBackgroundColor, adjustSizeToPowerOf2, imageToChwFloat32Array, chwFloat32ArrayToImage, imageToHwcFloat32Array, hwcFloat32ArrayToImage } from "../util/image";
import { DataUrlImage } from "./Image";

export abstract class Upscaler {

    protected abstract predict(img: Jimp.Jimp): Promise<Jimp.Jimp>;
    @action.bound
    async convert(src: DataUrlImage): Promise<DataUrlImage> {
        return DataUrlImage.fromJimp(await this.predict(await src.toJimp()));
    }
}

export class WebDNNUpscaler extends Upscaler {

    private _runner: WebDNN.DescriptorRunner;
    private _inputImage: SymbolicFloat32Array;
    private _outputImage: SymbolicFloat32Array;

    constructor(runner: WebDNN.DescriptorRunner) {
        super();
        this._runner = runner;
        this._inputImage = runner.inputs[0];
        this._outputImage = runner.outputs[0];
    }

    @action.bound
    protected async predict(img: Jimp.Jimp): Promise<Jimp.Jimp> {
        const width = img.bitmap.width;
        const height = img.bitmap.height;
        const x = imageToChwFloat32Array(img, 4).map((v) => v / 127.5 - 1.0);
        this._inputImage.set(x);
        await this._runner.run();
        const y = this._outputImage.toActual().map((v) => Math.min(Math.max((v + 1.0) * 127.5, 0.0), 255.0));
        return chwFloat32ArrayToImage(y, 4, height, width);
    }
}

export class KerasUpscaler extends Upscaler {

    private _model: any;

    constructor(model: any) {
        super();
        this._model = model;
    }

    @action.bound
    async predict(img: Jimp.Jimp): Promise<Jimp.Jimp> {
        const x = imageToHwcFloat32Array(img, 4).map((v) => v / 127.5 - 1.0);
        const y: Float32Array = (await this._model.predict({ "input_1": x })).output_gen;
        const convertedImage = hwcFloat32ArrayToImage(
            y.map((v) => Math.min(Math.max(v * 127.5 + 127.5, 0.0), 255.0)),
            4,
            img.bitmap.width,
            img.bitmap.height,
        );
        return convertedImage;
    }
}

export class TfjsUpscaler extends Upscaler {

    private _model: tf.Model;

    constructor(model: tf.Model) {
        super();
        this._model = model;
    }

    @action.bound
    async predict(img: Jimp.Jimp): Promise<Jimp.Jimp> {
        const width = img.bitmap.width
        const height = img.bitmap.height
        const x = imageToHwcFloat32Array(img, 4).map((v) => v / 127.5 - 1.0);

        const tensor = ((await this._model.predict(tf.tensor(x, [1, height, width, 4]))) as tf.Tensor<tf.Rank>);
        const y = (await tensor.flatten().data() as Float32Array);
        const convertedImage = hwcFloat32ArrayToImage(
            y.map((v) => Math.min(Math.max(v * 127.5 + 127.5, 0.0), 255.0)),
            4,
            img.bitmap.width,
            img.bitmap.height,
        );
        return convertedImage;
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
    @action.bound
    protected load(): Promise<Upscaler> {
        return WebDNN.load("./model/webdnn").then((runner) => {
            return new WebDNNUpscaler(runner);
        })
    }
}

export class KerasUpscalerLoader extends UpscalerLoader {
    @action.bound
    protected load(): Promise<Upscaler> {
        const model = new KerasJS.Model({
            filepath: './model/keras/model.bin',
            gpu: true,
        });
        return model.ready().then(() => {
            return new KerasUpscaler(model);
        });
    }
}

export class TfjsUpscalerLoader extends UpscalerLoader {
    @action.bound
    protected load(): Promise<Upscaler> {
        return tf.loadModel('./model/tfjs/model.json').then((model) => {
            return new TfjsUpscaler(model);
        });
    }
} 