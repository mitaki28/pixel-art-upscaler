import { UpscalerLoader, UpscalerLoadingState, WebDNNUpscalerLoader, TfjsUpscalerLoader, KerasUpscalerLoader } from "../store/Upscaler";
import { UpscaleConversionList } from "./UpscaleConversion";
import { action, computed, observable } from "mobx";
import { generateRandomString } from "../util/random";

export type UpscalerLoaderKeys = "stable" | "clear";

export class App {

    @observable private _upscalerLoaders: {readonly [K in UpscalerLoaderKeys]: UpscalerLoader};
    @observable upscaleConversionList: UpscaleConversionList;
    @observable private _uploadKey: string;
    @observable private _isShowingAbout: boolean;
    @observable private _currentLoaderKey: UpscalerLoaderKeys = "stable";

    constructor() {
        this._upscalerLoaders = {
            "stable": new TfjsUpscalerLoader("./model/tfjs/stable-20170425/model.json"),
            "clear": new TfjsUpscalerLoader("./model/tfjs/clear-20170430/model.json"),
        };
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
    get currentUpscalerKey() {
        return this._currentLoaderKey;
    }
    @computed
    get currentUpscalerLoader() {
        return this._upscalerLoaders[this._currentLoaderKey];
    }
    @computed
    get canSelectUpscaler() {
        return this.currentUpscalerLoader.state.status === UpscalerLoadingState.PENDING;
    }
    @action.bound
    selectStableMode() {
        this._currentLoaderKey = "stable";
    }
    @action.bound
    selectClearMode() {
        this._currentLoaderKey = "clear";
    }

    @computed
    get canStartUpscale() {
        return Object.values(this._upscalerLoaders).every((loader) => 
            loader.state.status !== UpscalerLoadingState.LOADING
        ) && !this.upscaleConversionList.isConverting;
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
            const converter = await this.currentUpscalerLoader.ready();
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