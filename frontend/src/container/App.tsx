import * as React from "react";
import { observer } from "mobx-react"
import { App } from "../store/App";
import { UpscaleConversionList } from "../store/UpscaleConversion";
import { UpscaleConversionListContainer } from "./UpscaleConversionList";
import { UpscalerLoadingState } from "../store/Upscaler";
import { Navbar, Nav, NavItem, Jumbotron, FormControl, Modal, ProgressBar, Button, Alert } from "react-bootstrap";
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
                        <p style={{ textAlign: "center" }}>【初回のみ】モデルのロード中・・・</p>
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
            <div className="container" style={{ height: "100%" }}>
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
                            <a href="#">Pixcaler</a>
                        </Navbar.Brand>
                    </Navbar.Header>
                    <Nav>
                        <NavItem onClick={this.props.store.showAbout}>
                            About
                        </NavItem>
                        <NavItem href="https://github.com/mitaki28/pixel-art-upscaler">
                            GitHub
                        </NavItem>
                    </Nav>
                </Navbar>
                <Alert bsStyle="info">
                    <p><Button bsStyle="link" onClick={this.props.store.showAbout}>利用規約はこちら</Button></p>
                    <p>
                        以下の条件を満たすドット絵が前提です
                    <ul>
                            <li>1ドット1px</li>
                            <li>jpeg 圧縮などでノイズがかかっていない</li>
                            <li>png 形式</li>
                        </ul>
                    </p>
                    <p>
                        画像変換処理はブラウザ上で実行されるため以下の点にご注意ください
                        <ul>
                            <li>あまり解像度の高い画像を入力するとPC・ブラウザが重くなったりクラッシュする可能性があります（600x600px程度に抑えること推奨）</li>
                            <li>ブラウザのタブを切り替えたりブラウザを非表示にしている間は一時的に変換処理が止まります</li>
                        </ul>                        
                    </p>
                </Alert>
                <Jumbotron>
                    <div style={{ textAlign: "center" }}>
                        <label>
                            <span className={"btn btn-primary" + (this.props.store.canStartUpscale ? "" : " disabled")}>
                                変換するファイルを選択
                                <FormControl
                                    style={{ display: "none" }}
                                    type="file"
                                    key={this.props.store.uploadKey}
                                    onChange={this.handleOnFileChange}
                                    disabled={!this.props.store.canStartUpscale}
                                />
                            </span>
                        </label>
                    </div>
                </Jumbotron>
                <div>
                    <UpscaleConversionListContainer store={this.props.store.upscaleConversionList} />
                </div>
            </div>
        );
    }
}