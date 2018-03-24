import * as React from "react";
import { observer } from "mobx-react"
import { App } from "../store/App";
import { ImageConversionList } from "../store/ImageConversion";
import { ImageConversionListContainer } from "./ImageConversionList";
import { UpscalerLoadingState } from "../store/Upscaler";

@observer
export class AppContainer extends React.Component<{ store: App }> {

    handleOnFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const fileList: FileList | null = e.currentTarget.files;
        if (fileList === null || fileList.length === 0) {
            return;
        }
        this.props.store.upscale(fileList[0]);
    }

    renderUpscalerLoadingStatus() {
        switch (this.props.store.upscalerLoader.state.status) {
            case UpscalerLoadingState.LOADING:
                return <div>モデルをロード中・・・</div>;
            case UpscalerLoadingState.LOAD_FAILURE:
                return <div>モデルのロードに失敗しました: {this.props.store.upscalerLoader.state.error.message}</div>
            default:
                return <div />
        }
    }

    render() {
        return (
            <div>
                <div>
                    {this.renderUpscalerLoadingStatus()}
                </div>
                <form>
                    <input
                        type="file"
                        key={this.props.store.uploadKey}
                        onChange={this.handleOnFileChange}
                        disabled={this.props.store.isLoading}
                    />
                </form>
                <div>
                    <ImageConversionListContainer store={this.props.store.imageConversionList} />
                </div>
            </div>
        );
    }
}