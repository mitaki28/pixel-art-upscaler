
import { observable, computed, action } from "mobx";
import { Converter, ConversionError } from "./Converter";
import { generateRandomString } from "../util/random";
import * as Jimp from "jimp";

export namespace UpscaleConversionState {
    export const LOADING = Symbol("LOADING");
    export type Loading = {
        status: typeof UpscaleConversionState.LOADING;
    };
    export const loading = (): Loading => ({
        status: LOADING,
    });

    export const LOAD_FAILURE = Symbol("LOAD_FAILURE");
    export type LoadFailure = {
        status: typeof UpscaleConversionState.LOAD_FAILURE;
        error: Error;
    };
    export const loadFailure = (error: Error): LoadFailure => ({
        status: LOAD_FAILURE,
        error,
    });

    export const CONVERTING = Symbol("CONVERTING");
    export type Converting = {
        status: typeof UpscaleConversionState.CONVERTING;
    };
    export const converting = (): Converting => ({
        status: CONVERTING,
    });

    export const CONVERTED = Symbol("CONVERTED");
    export type Converted = {
        status: typeof UpscaleConversionState.CONVERTED;
    };
    export const converted = (value: string): Converted => ({
        status: CONVERTED,
    });

    export const CONVERTION_FAILURE = Symbol("CONVERTION_FAILURE");
    export type ConversionFailure = {
        status: typeof UpscaleConversionState.CONVERTION_FAILURE;
        error: ConversionError;
    }
    export const conversionFailure = (error: ConversionError): ConversionFailure =>({
        status: UpscaleConversionState.CONVERTION_FAILURE,
        error,
    });
}

export type ImageConversionState = 
    UpscaleConversionState.Loading
    | UpscaleConversionState.LoadFailure
    | UpscaleConversionState.Converting
    | UpscaleConversionState.Converted
    | UpscaleConversionState.ConversionFailure;

export class UpscaleConversion {
    @observable _state: ImageConversionState;
    @observable private _inputFile: File;
    @observable inputImage: string | null = null;
    @observable inputImage2x: string | null = null;
    @observable convertedImage: string | null = null;
    

    constructor(
        public readonly id: string,
        inputFile: File,
        private readonly converter: Converter,
        private readonly imageConversionList: UpscaleConversionList,
    ) {
        this._inputFile = inputFile;
        this._state = {
            status: UpscaleConversionState.LOADING,
        }
    }

    @computed
    get state() {
        return this._state;
    }

    @computed
    get isConverting() {
        return this._state.status === UpscaleConversionState.CONVERTING;
    }


    @action.bound
    async start() {
        try {
            const img = await this.loadInputImage();
            const img2x = await this.scaleInputImageByNearestNeighbor(img);
            this.finishLoad(img, img2x);
            return this.startConversion(img);
        } catch (e) {
            this.failLoad(e);
            return;
        }
    }    

    @action.bound
    private loadInputImage() {
        return new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                resolve(reader.result);
            };
            reader.onerror = (ev) => {
                reject(ev.error);
            }
            reader.readAsDataURL(this._inputFile);
        });
    }

    @action.bound
    private async scaleInputImageByNearestNeighbor(src: string): Promise<string> {
        let img: Jimp.Jimp;
        img = await Jimp.read(src);
        img.resize(img.bitmap.width * 2, img.bitmap.height * 2, Jimp.RESIZE_NEAREST_NEIGHBOR);
        return  new Promise<string>((resolve, reject) => {
            (img as any).getBase64(Jimp.MIME_PNG, (error: any, dst: string) => {
                if (error) {
                    reject(error);
                }
                resolve(dst);
            });
        });
    }

    @action.bound
    private failLoad(error: Error) {
        this._state = {
            status: UpscaleConversionState.LOAD_FAILURE,
            error,
        }
    }

    @action.bound
    private finishLoad(inputImage: string, inputImage2x: string) {
        this._state = {
            status: UpscaleConversionState.CONVERTING,
        }
        this.inputImage = inputImage;
        this.inputImage2x = inputImage2x;
    }

    @action.bound
    private finishScalingByNearestNeighbor(inputImage2x: string) {
        this._state = {
            status: UpscaleConversionState.CONVERTING,
        }
        this.inputImage2x = inputImage2x;
    }

    @action.bound
    private async startConversion(inputImage: string) {
        this.converter.convert(inputImage).then(
            (result) =>
                result.bimap(
                    (error) => this.failConversion(error),
                    (img) => this.finishConversion(img),
                ),
        );        
    }

    @action.bound
    private failConversion(error: ConversionError) {
        this._state = {
            status: UpscaleConversionState.CONVERTION_FAILURE,
            error,
        }
    }

    @action.bound
    private finishConversion(convertedImage: string) {
        this._state = {
            status: UpscaleConversionState.CONVERTED,
        }
        this.convertedImage = convertedImage;
    }

    @computed
    get canClose() {
        return this.state.status !== UpscaleConversionState.CONVERTING;
    }

    @action.bound
    close() {
        this.imageConversionList.closeConversion(this);
    }
}

export class UpscaleConversionList {
    @observable private _conversions: UpscaleConversion[]
    constructor() {
        this._conversions = [];
    }

    @computed
    get conversions() {
        return this._conversions;
    }

    @computed
    get isConverting() {
        return this._conversions.some((conversion) => conversion.isConverting);
    }

    @action.bound
    startConversion(inputFile: File, converter: Converter): UpscaleConversion {
        const conversion = new UpscaleConversion(
            generateRandomString(),
            inputFile,
            converter,
            this,
        );
        this._conversions.unshift(conversion);
        conversion.start();
        return conversion;
    }

    @action.bound
    closeConversion(conversion: UpscaleConversion) {
        if (this._conversions.indexOf(conversion) === -1) {
            return;
        }
        if (!conversion.canClose) {
            return;
        }
        this._conversions = this._conversions.filter((c) => c !== conversion)
    }
}