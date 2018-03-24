
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


    export const TOO_LARGE_RESOLUTION = Symbol("TOO_LARGE");
    export type TooLarge = {
        code: typeof ConversionError.TOO_LARGE_RESOLUTION,
        limit: { width: number, height: number },
    };
    export const tooLarge = (limit: { width: number, height: number }): TooLarge => ({
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
    | ConversionError.TooLarge
    | ConversionError.FailedToConvert;

export interface Converter {
    convert(img: string): Promise<Either<ConversionError, string>>;
}