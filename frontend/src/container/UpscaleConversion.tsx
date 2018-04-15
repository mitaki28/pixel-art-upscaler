import * as React from "react";
import { observer } from "mobx-react"
import { UpscaleConversionList, UpscaleConversionFlow, UpscaleTask } from "../store/UpscaleConversion";
import { Row, Panel, Col, Well, Glyphicon, Button, ProgressBar, Tabs, Tab, Nav, NavItem, Alert } from "react-bootstrap";
import { Loading } from "../component/Loading";
import { DataUrlImage } from "../store/Image";
import { Task } from "../store/Task";

export const RunningUpscaleTaskPreview = observer(({ task }: { task: UpscaleTask }) => {
    if (task.preprocessedImagePreview === null) {
        return null;
    }
    return (
        <div>
            <div style={{height: 0, overflow: "visible", position: "relative"}}>
                <img style={{ opacity: 0.5, zIndex: 1}} src={task.preprocessedImagePreview.dataUrl} />
            </div>
            <table style={{ position: "relative", top: 0, margin: 0, padding: 0, zIndex: 100 }}>
                {task.patchUpscaleTasks.map((row, i) =>
                    <tr key={i}>
                        {row.map((col, j) => (
                            <td>
                                {col === null
                                    ? <div style={{ width: `${task.patchSize}px`, height: `${task.patchSize}px` }} />
                                    : col.state.status === Task.SUCCESS
                                        ? <img src={col.state.result.dataUrl} />
                                        : <img src={col.original.dataUrl} />
                                }
                            </td>
                        ))}
                    </tr>
                )}
            </table>
        </div>
    );
});

export const LoadTaskPreviewContainer = observer(({ task }: { task: Task<DataUrlImage> | null }) => {
    if (task === null) {
        return null;
    }
    switch (task.state.status) {
        case Task.PENDING:
        case Task.RUNNING:
        case Task.FAILURE:
            return null;
        case Task.SUCCESS:
            return <img src={task.state.result.dataUrl} />;
    }
});

export const Scale2xTaskPreviewContainer = observer(({ task }: { task: Task<DataUrlImage> | null }) => {
    if (task === null) {
        return null;
    }
    switch (task.state.status) {
        case Task.PENDING:
        case Task.RUNNING:
        case Task.FAILURE:
            return null;
        case Task.SUCCESS:
            return <img src={task.state.result.dataUrl} />;
    }
});


export const UpscaleTaskPreviewContainer = observer(({ task }: { task: UpscaleTask | null }) => {
    if (task === null) {
        return null;
    }
    switch (task.state.status) {
        case Task.PENDING:
            return null;
        case Task.RUNNING:
        case Task.FAILURE:
            return <RunningUpscaleTaskPreview task={task} />;
        case Task.SUCCESS:
            return <img src={task.state.result.dataUrl} />;
    }
});

export const LoadTaskProgressContainer = observer(({ task }: { task: Task<DataUrlImage> | null }) => {
    if (task === null) {
        return <ProgressBar active={false} now={0} />;
    }
    switch (task.state.status) {
        case Task.PENDING:
            return <ProgressBar active={true} now={0} label={"待機中・・・"}></ProgressBar>;
        case Task.RUNNING:
            return <ProgressBar active={true} now={100} label={"ファイルの読み込み中・・・"}></ProgressBar>;
        case Task.FAILURE:
            return <Alert bsStyle="danger">ファイルの読み込みに失敗しました: {task.state.error.message} </Alert>
        case Task.SUCCESS:
            return null;
    }
});

export const Scale2xTaskProgressContainer = observer(({ task }: { task: Task<DataUrlImage> | null }) => {
    if (task === null) {
        return <ProgressBar active={false} now={0} />;
    }
    switch (task.state.status) {
        case Task.PENDING:
            return <ProgressBar active={false} now={0} />;
        case Task.RUNNING:
            return <ProgressBar active={true} now={100} label={"準備中・・・"}></ProgressBar>;
        case Task.FAILURE:
            return <Alert bsStyle="danger">処理に失敗しました: {task.state.error.message} </Alert>
        case Task.SUCCESS:
            return null;
    }
});


export const UpscaleTaskProgressContainer = observer(({ task }: { task: UpscaleTask | null }) => {
    if (task === null) {
        return <ProgressBar active={true} now={0} />;
    }
    switch (task.state.status) {
        case Task.PENDING:
            return <ProgressBar active={true} now={0} />;
        case Task.RUNNING:
            if (Number.isNaN(task.progress)) {
                return <ProgressBar active={true} now={100} label={"準備中・・・"} />;
            } else {
                return <ProgressBar active={true} now={task.progress} />;
            }
        case Task.FAILURE:
            return <Alert bsStyle="danger">処理に失敗しました: {task.state.error.message} </Alert>
        case Task.SUCCESS:
            return null;
    }
});

export const UpscaleConversionFlowProgressContainer = observer(({ store }: { store: UpscaleConversionFlow }) => {
    const stage = store.currentStage;
    if (stage === null) {
        return null;
    }
    switch (stage.id) {
        case "load":
            return <LoadTaskProgressContainer task={stage.task} />
        case "scale2x":
            return <Scale2xTaskProgressContainer task={stage.task} />
        case "upscale":
            return <UpscaleTaskProgressContainer task={stage.task} />
    }
});

@observer
export class UpscaleConversionContainer extends React.Component<{ store: UpscaleConversionFlow }> {
    render() {
        return (
            <Panel>
                <Panel.Heading style={{ textAlign: "right" }}>
                    <Button bsStyle="danger" onClick={this.props.store.close} disabled={!this.props.store.canClose}><Glyphicon glyph="remove" /></Button>
                </Panel.Heading>
                <Panel.Body style={{ overflow: "hidden" }}>
                    <Tab.Container
                        activeKey={this.props.store.selectedStageId}
                        onSelect={(key: any) => this.props.store.selectStage(key)}
                    >
                        <div>
                            {!this.props.store.allFinished
                            ? (
                                <UpscaleConversionFlowProgressContainer store={this.props.store} />
                            ) : (
                                <Nav bsStyle="pills" style={{display: "flex", justifyContent: "center", alignItems: "center"}}>
                                    <NavItem eventKey="load">元画像</NavItem>
                                    <NavItem eventKey="scale2x">元画像(2x)</NavItem>
                                    <NavItem eventKey="upscale">変換結果</NavItem>
                                </Nav>
                            )}
                            <Tab.Content animation={false} style={{
                                marginTop: "20px",
                                display: "flex", justifyContent: "center", alignItems: "center",
                                minHeight: `${this.props.store.maxSize.height + 10}px`,
                            }}>
                                <Tab.Pane eventKey={"load"}>
                                    <LoadTaskPreviewContainer task={this.props.store.getTask("load")} />
                                </Tab.Pane>
                                <Tab.Pane eventKey={"scale2x"}>
                                    <Scale2xTaskPreviewContainer task={this.props.store.getTask("scale2x")} />
                                </Tab.Pane>
                                <Tab.Pane eventKey={"upscale"}>
                                    <UpscaleTaskPreviewContainer task={this.props.store.getTask("upscale")} />
                                </Tab.Pane>
                            </Tab.Content>
                        </div>
                    </Tab.Container>
                </Panel.Body>
            </Panel>
        );
    }
}