import { UpscalerLoader, UpscalerLoadingState } from "../store/Upscaler";
import { ImageConversionList } from "./ImageConversion";
import { action, computed, observable } from "mobx";

export const generateUploadKey = () => Math.random().toFixed(32).toString();

export class App {

    @observable upscalerLoader: UpscalerLoader;
    @observable imageConversionList: ImageConversionList;
    @observable private _uploadKey: string;

    constructor() {
        this.upscalerLoader = new UpscalerLoader();
        this.imageConversionList = new ImageConversionList();
        this._uploadKey = generateUploadKey();
    }

    @computed
    get uploadKey() {
        return this._uploadKey;
    }
    @computed
    get isLoading() {
        return this.upscalerLoader.state.status === UpscalerLoadingState.LOADING;
    }
    @action.bound
    updateUploadKey() {
        this._uploadKey = generateUploadKey();
    }
    @action.bound
    async upscale(file: File) {
        const converter = await this.upscalerLoader.ready();
        this.imageConversionList.startConversion(file, converter);
        this.updateUploadKey();
    }
}