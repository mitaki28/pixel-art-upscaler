
import { observable, computed, action } from "mobx";
import { generateRandomString } from "../util/random";
import * as Jimp from "jimp";
import { Upscaler } from "./Upscaler";
import { Task } from "./Task";
import { Either, right, left } from "fp-ts/lib/Either";
import { DataUrlImage } from "./Image";
import { downsample_nearest_neighbor, upsample_nearest_neighbor } from "../util/tensor";

const SIZE_FACTOR = 64;
const PATCH_SIZE = 32;

export class PatchUpscaleTask extends Task<DataUrlImage> {
    @observable _patch: DataUrlImage;
    private constructor(upscaler: Upscaler, patch: DataUrlImage, private readonly _original: DataUrlImage) {
        super(async () => {
            const src = await upscaler.convert(patch);
            return await src.withJimp((img) => {
                img.crop(PATCH_SIZE / 2, PATCH_SIZE / 2, PATCH_SIZE, PATCH_SIZE);
            });
        });
        this._patch = patch;
    }
    @computed
    get original() {
        return this._original;
    }

    public static async create(upscaler: Upscaler, patch: DataUrlImage) {
        return new PatchUpscaleTask(upscaler, patch, await patch.withJimp((img) => {
            img.crop(PATCH_SIZE / 2, PATCH_SIZE / 2, PATCH_SIZE, PATCH_SIZE);
        }));
    }
}

export class UpscaleTask extends Task<DataUrlImage> {
    @observable patchUpscaleTasks: Array<Array<PatchUpscaleTask | null>> = [];
    constructor(upscaler: Upscaler, src: DataUrlImage) {
        super(async () => {
            const img = await src.toJimp();
            const [w, h] = [img.bitmap.width, img.bitmap.height];
            const [dstWidth, dstHeight] = [w - PATCH_SIZE, h - PATCH_SIZE];
            const ni = dstHeight / PATCH_SIZE;
            const nj = dstWidth / PATCH_SIZE;
            this.patchUpscaleTasks = Array(ni).fill([]).map(() => Array(nj).fill(null));
            const dst = new (Jimp.default as any)(dstWidth, dstHeight);
            for (let i = 0; i < ni; i++) {
                for (let j = 0; j < nj; j++) {
                    const y = i * PATCH_SIZE;
                    const x = j * PATCH_SIZE;
                    const patch = img.clone().crop(
                        x,
                        y,
                        2 * PATCH_SIZE,
                        2 * PATCH_SIZE,
                    );
                    const convertedPatchSrc = await (this.patchUpscaleTasks[i][j] = await PatchUpscaleTask.create(
                        upscaler,
                        await DataUrlImage.fromJimp(patch),
                    )).run();
                    const convertedPatch = await convertedPatchSrc.toJimp();
                    dst.composite(convertedPatch, x, y);
                }
            }
            return await DataUrlImage.fromJimp(dst);
        });
    }
}


export class UpscaleConversionFlow {

    @observable private _inputFile: File;
    @observable private _isRunning: boolean = false;
    @observable private _tasks: UpscaleConversionFlow.StageDef = {
        load: null,
        preScale2x: null,
        prePadding: null,
        preAlign: null,
        upscale: null,
        postClip: null,
    };
    @observable private _currentStageId: keyof UpscaleConversionFlow.StageDef | null = null;

    constructor(
        public readonly id: string,
        inputFile: File,
        private readonly upscaler: Upscaler,
        private readonly imageConversionList: UpscaleConversionList,
    ) {
        this._inputFile = inputFile;
    }

    @computed
    get isRunning() {
        return this._isRunning;
    }

    @action.bound
    async run() {
        this._isRunning = true;
        try {
            const inputImage = await this.runTask("load", this.loadInputImage(this._inputFile));
            const inputImage2x = await this.runTask("preScale2x", this.scaleInputImageByNearestNeighbor(inputImage));
            const prePaddedImage = await this.runTask("prePadding", this.prePadding(inputImage2x));
            const preAlignedImage = await this.runTask("preAlign", this.preAlign(prePaddedImage));
            const upscaledImage = await this.runTask("upscale", new UpscaleTask(this.upscaler, preAlignedImage));
            const dstImage = await this.runTask("postClip", this.postCrip(upscaledImage, inputImage));
        } finally {
            this._isRunning = false;
        }
    }

    @computed
    get currentTask(): UpscaleConversionFlow.Stage | null {
        if (this._currentStageId === null) {
            return null;
        }
        return {
            id: this._currentStageId,
            task: this._tasks[this._currentStageId],
        } as UpscaleConversionFlow.Stage;
    }

    @computed
    get startedStages(): UpscaleConversionFlow.Stage[] {
        return (Object.keys(this._tasks) as UpscaleConversionFlow.StageId[])
            .reduce((acc: UpscaleConversionFlow.Stage[], id: UpscaleConversionFlow.StageId) =>
                this._tasks[id] === null
                    ? acc
                    : [...acc, { id, task: this._tasks[id] }] as UpscaleConversionFlow.Stage[],
                []);
    }

    @computed
    get finishedStages(): UpscaleConversionFlow.Stage[] {
        return this.startedStages.filter((stage) => stage.task.state.status === Task.SUCCESS);
    }

    @action.bound
    private async runTask<K extends keyof UpscaleConversionFlow.StageDef>(id: K, task: NonNullable<UpscaleConversionFlow.StageDef[K]>) {
        this._currentStageId = id;
        return await (this._tasks[id] = task).run();
    }

    @action.bound
    private loadInputImage(inputFile: File) {
        return new Task(() => DataUrlImage.fromFile(inputFile));
    }

    @action.bound
    private scaleInputImageByNearestNeighbor(src: DataUrlImage): Task<DataUrlImage> {
        return new Task(async () => {
            return await DataUrlImage.fromTf(upsample_nearest_neighbor(await src.toTf()));
        });
    }

    @action.bound
    private prePadding(src: DataUrlImage): Task<DataUrlImage> {
        return new Task(async () => {
            const t = await src.toTf();
            const [h, w, c] = t.shape;
            const [nh, nw] = [h, w].map((x) => SIZE_FACTOR * Math.ceil(x / SIZE_FACTOR) + PATCH_SIZE);
            const [ph, pw] = [
                Math.floor((nh - h) / 2),
                Math.floor((nw - w) / 2),
            ];
            return await DataUrlImage.fromTf(t.pad([
                [ph, nh - h - ph],
                [pw, nw - w - pw],
                [0, 0],
            ]));
        });
    }

    @action.bound
    private preAlign(src: DataUrlImage): Task<DataUrlImage> {
        return new Task(async () => {
            return await DataUrlImage.fromTf(
                upsample_nearest_neighbor(
                    downsample_nearest_neighbor(await src.toTf()),
                ),
            );
        });
    }

    @action.bound
    private postCrip(convertedSrc: DataUrlImage, originalSrc: DataUrlImage): Task<DataUrlImage> {
        return new Task(async () => {
            const convertedImage = await convertedSrc.toTf();
            const originalImage = await originalSrc.toTf();
            const [dstHeight, dstWidth] = [2 * originalImage.shape[0], 2 * originalImage.shape[1]];
            const [convertedHeight, convertedWidth, convertedChannel] = convertedImage.shape;
            return DataUrlImage.fromTf(convertedImage.slice([
                (convertedHeight - dstHeight) / 2,
                (convertedWidth - dstWidth) / 2,
                0,
            ], [
                dstHeight,
                dstWidth,
                convertedChannel,
            ]));
        });
    }

    @computed
    get canClose() {
        return !this._isRunning;
    }

    @action.bound
    close() {
        this.imageConversionList.closeConversion(this);
    }
}

export namespace UpscaleConversionFlow {
    export interface StageDef {
        load: Task<DataUrlImage> | null;
        preScale2x: Task<DataUrlImage> | null;
        prePadding: Task<DataUrlImage> | null;
        preAlign: Task<DataUrlImage> | null;
        upscale: UpscaleTask | null;
        postClip: Task<DataUrlImage> | null;
    }
    export type StageId = keyof StageDef;
    export type Stage = {
        [K in StageId]: { id: K, task: NonNullable<StageDef[K]> }
    }[StageId];
}

export class UpscaleConversionList {
    @observable private _conversions: UpscaleConversionFlow[]
    constructor() {
        this._conversions = [];
    }

    @computed
    get conversions() {
        return this._conversions;
    }

    @computed
    get isConverting() {
        return this._conversions.some((conversion) => conversion.isRunning);
    }

    @action.bound
    startConversion(inputFile: File, upscaler: Upscaler): UpscaleConversionFlow {
        const conversion = new UpscaleConversionFlow(
            generateRandomString(),
            inputFile,
            upscaler,
            this,
        );
        this._conversions.unshift(conversion);
        conversion.run();
        return conversion;
    }

    @action.bound
    closeConversion(conversion: UpscaleConversionFlow) {
        if (this._conversions.indexOf(conversion) === -1) {
            return;
        }
        if (!conversion.canClose) {
            return;
        }
        this._conversions = this._conversions.filter((c) => c !== conversion)
    }
}