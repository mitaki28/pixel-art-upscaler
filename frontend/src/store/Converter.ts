
import { Either } from "fp-ts/lib/Either";

export namespace ConversionError {
    export const FAILED_TO_LOAD = Symbol("FAILED_TO_LOAD");
    export type FailedToLoad = {
        code: typeof ConversionError.FAILED_TO_LOAD,
        error: Error,
    };
    export const failedToLoad = (error: Error): FailedToLoad => ({
        code: ConversionError.FAILED_TO_LOAD,
        error,
    });


    export const TOO_LARGE_RESOLUTION = Symbol("TOO_LARGE_RESOLUTION");
    export type TooLargeResolution = {
        code: typeof ConversionError.TOO_LARGE_RESOLUTION,
        limit: { width: number, height: number },
    };
    export const tooLargeResolution = (limit: { width: number, height: number }): TooLargeResolution => ({
        code: ConversionError.TOO_LARGE_RESOLUTION,
        limit,
    });

    export const FAILED_TO_CONVERT = Symbol("FAILED_TO_CONVERT");
    export type FailedToConvert = {
        code: typeof ConversionError.FAILED_TO_CONVERT,
        error: Error,
    };
    export const failedToConvert = (error: Error): FailedToConvert => ({
        code: ConversionError.FAILED_TO_CONVERT,
        error,
    });
}

export type ConversionError =
    ConversionError.FailedToLoad
    | ConversionError.TooLargeResolution
    | ConversionError.FailedToConvert;

export interface Converter {
    convert(img: string): Promise<Either<ConversionError, string>>;
}