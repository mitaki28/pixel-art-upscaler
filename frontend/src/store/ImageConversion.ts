
import { observable, computed, action } from "mobx";
import { Converter, ConversionError } from "./Converter";

export namespace ImageConversionState {
    export const LOADING = Symbol("LOADING");
    export type Loading = {
        status: typeof ImageConversionState.LOADING;
    };
    export const loading = (): Loading => ({
        status: LOADING,
    });

    export const LOAD_FAILURE = Symbol("LOAD_FAILURE");
    export type LoadFailure = {
        status: typeof ImageConversionState.LOAD_FAILURE;
        error: Error;
    };
    export const loadFailure = (error: Error): LoadFailure => ({
        status: LOAD_FAILURE,
        error,
    });

    export const CONVERTING = Symbol("CONVERTING");
    export type Converting = {
        status: typeof ImageConversionState.CONVERTING;
    };
    export const converting = (): Converting => ({
        status: CONVERTING,
    });

    export const CONVERTED = Symbol("CONVERTED");
    export type Converted = {
        status: typeof ImageConversionState.CONVERTED;
    };
    export const converted = (value: string): Converted => ({
        status: CONVERTED,
    });

    export const CONVERTION_FAILURE = Symbol("CONVERTION_FAILURE");
    export type ConversionFailure = {
        status: typeof ImageConversionState.CONVERTION_FAILURE;
        error: ConversionError;
    }
    export const conversionFailure = (error: ConversionError) =>({
        status: typeof ImageConversionState.CONVERTION_FAILURE,
        error,
    });
}

export type ImageConversionState = 
    ImageConversionState.Loading
    | ImageConversionState.LoadFailure
    | ImageConversionState.Converting
    | ImageConversionState.Converted
    | ImageConversionState.ConversionFailure;

export class ImageConversion {
    @observable _state: ImageConversionState;
    @observable private _inputFile: File;
    @observable inputImage: string | null = null;
    @observable convertedImage: string | null = null;
    

    constructor(
        inputFile: File,
        private readonly converter: Converter,
        private readonly imageConversionList: ImageConversionList,
    ) {
        this._inputFile = inputFile;
        this._state = {
            status: ImageConversionState.LOADING,
        }
    }

    @computed
    get state() {
        return this._state;
    }


    @action.bound
    start() {
        this.loadInputImage().then((image) => {
            this.finishLoad(image);
            return this.startConversion(image);
        }).catch((e) => {
            this.failLoad(e);
        });
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
    private failLoad(error: Error) {
        this._state = {
            status: ImageConversionState.LOAD_FAILURE,
            error,
        }
    }

    @action.bound
    private finishLoad(inputImage: string) {
        this._state = {
            status: ImageConversionState.CONVERTING,
        }
        this.inputImage = inputImage;
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
            status: ImageConversionState.CONVERTION_FAILURE,
            error,
        }
    }

    @action.bound
    private finishConversion(convertedImage: string) {
        this._state = {
            status: ImageConversionState.CONVERTED,
        }
        this.convertedImage = convertedImage;
    }

    @computed
    get canClose() {
        return this.state.status === ImageConversionState.CONVERTING;
    }

    @action.bound
    close() {
        this.imageConversionList.closeConversion(this);
    }
}

export class ImageConversionList {
    @observable private _conversions: ImageConversion[]
    constructor() {
        this._conversions = [];
    }

    @computed
    get conversions() {
        return this._conversions;
    }

    @action.bound
    startConversion(inputFile: File, converter: Converter): ImageConversion {
        const conversion = new ImageConversion(
            inputFile,
            converter,
            this,
        );
        this._conversions.push(conversion);
        conversion.start();
        return conversion;
    }

    @action.bound
    closeConversion(conversion: ImageConversion) {
        if (this._conversions.indexOf(conversion) === -1) {
            return;
        }
        if (!conversion.canClose) {
            return;
        }
        this._conversions = this._conversions.filter((c) => c != conversion)
    }
}