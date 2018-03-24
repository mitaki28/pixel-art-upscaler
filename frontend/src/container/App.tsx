import * as React from "react";
import { observer } from "mobx-react"
import { App } from "../store/App";
import { UpscaleConversionList } from "../store/UpscaleConversion";
import { UpscaleConversionListContainer } from "./UpscaleConversionList";
import { UpscalerLoadingState } from "../store/Upscaler";
import { Navbar, Nav, NavItem, Jumbotron, FormControl, Modal, ProgressBar, Button } from "react-bootstrap";
import { About } from "../component/About";

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
                return (
                    <div>
                        <p style={{textAlign: "center"}}>【初回のみ】モデルのロード中・・・</p>
                        <ProgressBar active now={100} />
                    </div>
                );
            case UpscalerLoadingState.LOAD_FAILURE:
                return <div>モデルのロードに失敗しました: {this.props.store.upscalerLoader.state.error.message}</div>
            default:
                return <div />
        }
    }

    render() {
        return (
            <div className="container" style={{height: "100%"}}>
                <Modal container={this} show={
                    this.props.store.upscalerLoader.state.status === UpscalerLoadingState.LOADING
                        || this.props.store.upscalerLoader.state.status === UpscalerLoadingState.LOAD_FAILURE
                    } onHide={() => undefined}>
                    <Modal.Body>
                        {this.renderUpscalerLoadingStatus()}
                    </Modal.Body>
                </Modal>
                <Modal container={this} show={this.props.store.isShowingAbout} onHide={this.props.store.hideAbout}>
                    <Modal.Body><About /></Modal.Body>
                    <Modal.Footer><Button onClick={this.props.store.hideAbout}>閉じる</Button></Modal.Footer>
                </Modal>
                <Navbar>
                    <Navbar.Header>
                        <Navbar.Brand>
                        <a href="#">Pixel-Art Upscaler</a>
                        </Navbar.Brand>
                    </Navbar.Header>
                    <Nav>
                        <NavItem onClick={this.props.store.showAbout}>
                            About
                        </NavItem>
                    </Nav>
                </Navbar>
                <Jumbotron style={{textAlign: "center"}}>
                    <div>
                        <label>
                            <span className={"btn btn-primary" + (this.props.store.canStartUpscale ? "" : " disabled")}>
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
                    <UpscaleConversionListContainer store={this.props.store.upscaleConversionList} />
                </div>
            </div>
        );
    }
}