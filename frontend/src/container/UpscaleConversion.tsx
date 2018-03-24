import * as React from "react";
import { observer } from "mobx-react"
import { UpscaleConversionList, UpscaleConversion, UpscaleConversionState } from "../store/UpscaleConversion";
import { ConversionError } from "../store/Converter";
import { Row, Panel, Col, Well, Glyphicon, Button } from "react-bootstrap";
import { Loading } from "../component/Loading";

@observer
export class UpscaleConversionContainer extends React.Component<{ store: UpscaleConversion }> {

    renderInputImage(): JSX.Element {
        switch (this.props.store.state.status) {
            case UpscaleConversionState.LOADING:
                return <Loading />
            case UpscaleConversionState.LOAD_FAILURE:
                return <div>ファイルの読み込みに失敗しました: {this.props.store.state.error.message}</div>
            case UpscaleConversionState.CONVERTING:
            case UpscaleConversionState.CONVERTED:
            case UpscaleConversionState.CONVERTION_FAILURE:
                return <img
                    src={this.props.store.inputImage!}
                />;
        }
    }

    renderInputImage2x(): JSX.Element {
        switch (this.props.store.state.status) {
            case UpscaleConversionState.LOADING:
                return <Loading />
            case UpscaleConversionState.LOAD_FAILURE:
                return <div>ファイルの読み込みに失敗しました: {this.props.store.state.error.message}</div>
            case UpscaleConversionState.CONVERTING:
            case UpscaleConversionState.CONVERTED:
            case UpscaleConversionState.CONVERTION_FAILURE:
                return <img
                    src={this.props.store.inputImage2x!}
                />;
        }
    }

    renderConvertedImage() {
        switch (this.props.store.state.status) {
            case UpscaleConversionState.LOADING:
            case UpscaleConversionState.LOAD_FAILURE:
            case UpscaleConversionState.CONVERTING:
                return <Loading />
            case UpscaleConversionState.CONVERTED:
                return <img
                    src={this.props.store.convertedImage!}
                />
            case UpscaleConversionState.CONVERTION_FAILURE:
                return this.renderConversionError(this.props.store.state.error);
        }
    }

    renderConversionError(error: ConversionError) {
        switch (error.code) {
            case ConversionError.FAILED_TO_LOAD:
                return <div>ファイルの読み込みに失敗しました: {error.error.message}</div>;
            case ConversionError.FAILED_TO_CONVERT:
                return <div>ファイルの変換に失敗しました: {error.error.message}</div>;
            case ConversionError.TOO_LARGE_RESOLUTION:
                return <div>解像度が{error.limit.width}x{error.limit.height}以下の画像しか変換できません</div>;
        }
    }

    render() {
        return (
            <Panel>
                <Panel.Heading style={{textAlign: "right"}}>
                    <Button bsStyle="danger" onClick={this.props.store.close}><Glyphicon glyph="remove" /></Button>
                </Panel.Heading>
                <Panel.Body style={{ overflow: "hidden" }}>
                    <Col md={4}>
                        <Panel style={{ width: "100%", textAlign: "center" }}>
                            <Panel.Heading>元画像</Panel.Heading>
                            <Panel.Body style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "300px", backgroundColor: "black", color: "white"}}>
                                {this.renderInputImage()}
                            </Panel.Body>
                        </Panel>
                    </Col>
                    <Col md={4}>
                        <Panel style={{ width: "100%", textAlign: "center" }}>
                            <Panel.Heading>元画像(x2)</Panel.Heading>
                            <Panel.Body style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "300px", backgroundColor: "black", color: "white" }}>
                                {this.renderInputImage2x()}
                            </Panel.Body>
                        </Panel>
                    </Col>
                    <Col md={4}>
                        <Panel style={{ width: "100%", textAlign: "center" }}>
                            <Panel.Heading>変換後</Panel.Heading>
                            <Panel.Body style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "300px", backgroundColor: "black", color: "white" }}>
                                {this.renderConvertedImage()}
                            </Panel.Body>
                        </Panel>
                    </Col>
                </Panel.Body>
            </Panel>
        );
    }
}