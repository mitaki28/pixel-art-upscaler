import * as React from "react";
import { observer } from "mobx-react"
import { App } from "../store/App";
import { ImageConversionList } from "../store/ImageConversion";
import { ImageConversionListContainer } from "./ImageConversionList";
import { UpscalerLoadingState } from "../store/Upscaler";
import { Navbar, Nav, NavItem, Jumbotron, FormControl, Modal, ProgressBar } from "react-bootstrap";

@observer
export class AppContainer extends React.Component<{ store: App }> {

    handleOnFileChange = (e: any) => {
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
            <div className="container" style={{height: "100%"}}>
                {this.props.store.upscalerLoader.state.status === UpscalerLoadingState.LOADING &&
                    <Modal container={this} show={this.props.store.upscalerLoader.state.status === UpscalerLoadingState.LOADING} onHide={() => undefined}>
                        <Modal.Body>
                            <p style={{textAlign: "center"}}>【初回のみ】モデルのロード中・・・</p>
                            <ProgressBar active now={100} />
                        </Modal.Body>
                    </Modal>
                }
                <Navbar>
                    <Navbar.Header>
                        <Navbar.Brand>
                        <a href="#">Pixel-Art Upscaler</a>
                        </Navbar.Brand>
                    </Navbar.Header>
                </Navbar>
                <Jumbotron style={{textAlign: "center"}}>
                    <div>
                        <label>
                            <span className="btn btn-primary">
                                変換するファイルを選択
                                <FormControl
                                    style={{display: "none"}}
                                    type="file"
                                    key={this.props.store.uploadKey}
                                    onChange={this.handleOnFileChange}
                                    disabled={!this.props.store.canStartUpscale}
                                />
                            </span>
                        </label>
                    </div>
                    <div>
                        ※ 解像度: 32x32 以下
                    </div>
                </Jumbotron>
                <div>
                    <ImageConversionListContainer store={this.props.store.imageConversionList} />
                </div>
            </div>
        );
    }
}