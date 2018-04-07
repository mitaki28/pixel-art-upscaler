import { UpscalerLoader, UpscalerLoadingState, WebDNNUpscalerLoader, TfjsUpscalerLoader, KerasUpscalerLoader } from "../store/Upscaler";
import { UpscaleConversionList } from "./UpscaleConversion";
import { action, computed, observable } from "mobx";
import { generateRandomString } from "../util/random";

export class App {

    @observable upscalerLoader: UpscalerLoader;
    @observable upscaleConversionList: UpscaleConversionList;
    @observable private _uploadKey: string;
    @observable private _isShowingAbout: boolean;

    constructor() {
        this.upscalerLoader = new KerasUpscalerLoader();
        this.upscaleConversionList = new UpscaleConversionList();
        this._uploadKey = generateRandomString();
        this._isShowingAbout = false;
    }

    @computed
    get uploadKey() {
        return this._uploadKey;
    }
    @computed
    get isShowingAbout() {
        return this._isShowingAbout;
    }

    @computed
    get canStartUpscale() {
        return this.upscalerLoader.state.status !== UpscalerLoadingState.LOADING
            && !this.upscaleConversionList.isConverting;
    }
    @action.bound
    updateUploadKey() {
        this._uploadKey = generateRandomString();
    }
    @computed
    get isConverting() {
        return this.upscaleConversionList.isConverting;
    }
    @action.bound
    async upscale(file: File) {
        if (this.canStartUpscale) {
            const converter = await this.upscalerLoader.ready();
            this.upscaleConversionList.startConversion(file, converter);
            this.updateUploadKey();
        }
    }

    @action.bound
    async showAbout() {
        this._isShowingAbout = true;
    }

    @action.bound
    async hideAbout() {
        this._isShowingAbout = false;
    }
}