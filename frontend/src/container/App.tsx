import * as React from "react";
import { observer } from "mobx-react"
import { App } from "../store/App";
import { UpscaleConversionList } from "../store/UpscaleConversion";
import { UpscaleConversionListContainer } from "./UpscaleConversionList";
import { UpscalerLoadingState } from "../store/Upscaler";
import { Navbar, Nav, NavItem, Jumbotron, FormControl, Modal, ProgressBar, Button, Alert, ButtonGroup } from "react-bootstrap";
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
        switch (this.props.store.currentUpscalerLoader.state.status) {
            case UpscalerLoadingState.LOADING:
                return (
                    <div>
                        <p style={{ textAlign: "center" }}>【初回のみ】モデルのロード中・・・</p>
                        <ProgressBar active now={100} />
                    </div>
                );
            case UpscalerLoadingState.LOAD_FAILURE:
                return <div>モデルのロードに失敗しました: {this.props.store.currentUpscalerLoader.state.error.message}</div>
            default:
                return <div />
        }
    }

    render() {
        return (
            <div className="container" style={{ height: "100%" }}>
                <Modal container={this} show={
                    this.props.store.currentUpscalerLoader.state.status === UpscalerLoadingState.LOADING
                    || this.props.store.currentUpscalerLoader.state.status === UpscalerLoadingState.LOAD_FAILURE
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
                            <a href="#">Pixcaler(β)</a>
                        </Navbar.Brand>
                    </Navbar.Header>
                    <Nav>
                        <NavItem onClick={this.props.store.showAbout}>
                            利用規約/About
                        </NavItem>
                        <NavItem href="https://github.com/mitaki28/pixcaler" target="_blank">
                            GitHub
                        </NavItem>
                    </Nav>
                </Navbar>
                <Alert bsStyle="info">
                    <p>
                        以下の条件を満たすドット絵が前提としています
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
                    <div style={{ textAlign: "center" }}>
                        {this.props.store.canSelectUpscaler
                        ? <ButtonGroup>
                            <Button onClick={this.props.store.selectStableMode} active={this.props.store.currentUpscalerKey === "stable"}>安定性重視</Button>
                            <Button onClick={this.props.store.selectClearMode} active={this.props.store.currentUpscalerKey === "clear"}>鮮明さ重視</Button>
                        </ButtonGroup>
                        : <ButtonGroup>
                            {this.props.store.currentUpscalerKey === "stable" && <Button onClick={this.props.store.selectStableMode} active={true}>安定性重視</Button>}
                            {this.props.store.currentUpscalerKey === "clear" && <Button onClick={this.props.store.selectClearMode} active={true}>鮮明さ重視</Button>}
                        </ButtonGroup>
                        }
                        <div style={{ paddingTop: "1em" }}>
                            {(() => {
                                switch (this.props.store.currentUpscalerKey) {
                                    case "stable":
                                        return (
                                            <>
                                                <div>安定性を重視して変換します。</div>
                                                <div>出力画像は安定しますが細部がぼやけます。</div>
                                            </>
                                        );
                                    case "clear":
                                        return (
                                            <>
                                                <div>(experimental)鮮明さを重視して変換します。</div>
                                                <div>出力画像は細部までくっきり仕上がりますがノイズが乗ったり、形状が歪むなど不安定になりやすいです。</div>
                                            </>
                                        );
                                }
                            })()}
                        </div>
                    </div>
                </Jumbotron>
                <div>
                    <UpscaleConversionListContainer store={this.props.store.upscaleConversionList} />
                </div>
            </div>
        );
    }
}