import { UpscalerLoader, UpscalerLoadingState, WebDNNUpscalerLoader } from "../store/Upscaler";
import { ImageConversionList } from "./ImageConversion";
import { action, computed, observable } from "mobx";
import { generateRandomString } from "../util/random";

export class App {

    @observable upscalerLoader: UpscalerLoader;
    @observable imageConversionList: ImageConversionList;
    @observable private _uploadKey: string;

    constructor() {
        this.upscalerLoader = new WebDNNUpscalerLoader();
        this.imageConversionList = new ImageConversionList();
        this._uploadKey = generateRandomString();
    }

    @computed
    get uploadKey() {
        return this._uploadKey;
    }
    @computed
    get canStartUpscale() {
        return !(this.upscalerLoader.state.status === UpscalerLoadingState.LOADING)
    }
    @action.bound
    updateUploadKey() {
        this._uploadKey = generateRandomString();
    }
    @action.bound
    async upscale(file: File) {
        const converter = await this.upscalerLoader.ready();
        this.imageConversionList.startConversion(file, converter);
        this.updateUploadKey();
    }
}