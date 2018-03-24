import * as React from "react";
import { observer } from "mobx-react"
import { ImageConversionList, ImageConversion, ImageConversionState } from "../store/ImageConversion";
import { ConversionError } from "../store/Converter";


@observer
export class ImageConversionContainer extends React.Component<{ store: ImageConversion }> {

    renderInputImage(): JSX.Element {
        switch (this.props.store.state.status) {
            case ImageConversionState.LOADING:
                return <div>ロード中・・・</div>;
            case ImageConversionState.LOAD_FAILURE:
                return <div>ファイルの読み込みに失敗しました: {this.props.store.state.error.message}</div>
            case ImageConversionState.CONVERTING:
            case ImageConversionState.CONVERTED:
            case ImageConversionState.CONVERTION_FAILURE:
                return <img src={this.props.store.inputImage!} />;
        }
    }

    renderConvertedImage() {
        switch (this.props.store.state.status) {
            case ImageConversionState.LOADING:
            case ImageConversionState.LOAD_FAILURE:
                return null;
            case ImageConversionState.CONVERTING:
                return <div>変換中・・・</div>;
            case ImageConversionState.CONVERTED:
                return <img src={this.props.store.convertedImage!} />;
            case ImageConversionState.CONVERTION_FAILURE:
                return this.renderConversionError(this.props.store.state.error);
            }
    }

    renderConversionError(error: ConversionError) {
        switch (error.code) {
            case ConversionError.FAILED_TO_LOAD:
                return <div>ファイルの読み込みに失敗しました: {error.error.message}</div>;
            case ConversionError.FAILED_TO_CONVERT:
                return <div>ファイルの変換に失敗しました</div>;
            case ConversionError.TOO_LARGE_RESOLUTION:
                return <div>解像度は32x32以下で入力してください</div>;
        }
    }

    render() {
        return (
            <tr>
                <td>{this.renderInputImage()}</td>
                <td>{this.renderConvertedImage()}</td>
            </tr>
        );
    }
}