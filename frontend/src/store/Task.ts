import { observable, action, comparer, computed } from "mobx";
import { Either } from "fp-ts/lib/Either";

export class Task<R> {
    @observable private _state: Task.State<R>;
    constructor(private executor: () => Promise<R>) {
        this._state = Task.pending();
    }

    @computed
    get state() {
        return this._state;
    }


    @action.bound
    async run() {
        this._state = Task.running();
        try {
            this._state = Task.success(await this.executor());
            return this._state.result;
        } catch (e) {
            this._state = Task.failure(e);
            throw e;
        }
    }
}

export namespace Task {
    export const PENDING = Symbol("PENDING");
    export type Pending = {
        status: typeof Task.PENDING;
    };
    export const pending = (): Pending => ({
        status: PENDING,
    });

    export const RUNNING = Symbol("RUNNING");
    export type Running = {
        status: typeof Task.RUNNING;
    };
    export const running = (): Running => ({
        status: RUNNING,
    });

    export const SUCCESS = Symbol("SUCCESS");
    export type Success<R> = {
        status: typeof Task.SUCCESS;
        result: R;
    };
    export const success = <R>(result: R): Success<R> => ({
        status: SUCCESS,
        result,
    });

    export const FAILURE = Symbol("FAILURE");
    export type Failure = {
        status: typeof Task.FAILURE;
        error: Error;
    }
    export const failure = (error: Error): Failure =>({
        status: Task.FAILURE,
        error,
    });

    export type State<R> =  Pending | Running | Success<R> | Failure;
}