import * as React from "react";
import { observer } from "mobx-react"
import { UpscaleConversionList, UpscaleConversionFlow, UpscaleTask } from "../store/UpscaleConversion";
import { Row, Panel, Col, Well, Glyphicon, Button } from "react-bootstrap";
import { Loading } from "../component/Loading";
import { DataUrlImage } from "../store/Image";
import { Task } from "../store/Task";

export const SimpleImageComponent = (props: { result: DataUrlImage }) => (
    <img src={props.result.dataUrl} />
);

export const GenericErrorComponent = (props: { message: string }) => (es: { error: Error }) => (
    <div>{props.message}: {es.error.message}</div>
);

export interface TaskContainerProps {
    task: Task<DataUrlImage> | null;
    title: string;
    resultComponent: React.ComponentType<{ result: DataUrlImage }>
    errorComponent: React.ComponentType<{ error: Error }>;
}

@observer
export class TaskContainer extends React.Component<TaskContainerProps> {
    renderTaskStatus() {
        if (this.props.task == null) {
            return <Loading />;
        }
        const ErrorComponent = this.props.errorComponent;
        switch (this.props.task.state.status) {
            case Task.PENDING:
            case Task.RUNNING:
                return <Loading />
            case Task.FAILURE:
                return <ErrorComponent error={this.props.task.state.error} />;
            case Task.SUCCESS:
                return <img src={this.props.task.state.result.dataUrl} />;
        }
    }

    render() {
        return (
            <Panel style={{ width: "100%", textAlign: "center" }}>
                <Panel.Heading>{this.props.title}</Panel.Heading>
                <Panel.Body style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "300px", backgroundColor: "black", color: "white" }}>
                    {this.renderTaskStatus()}
                </Panel.Body>
            </Panel>
        );
    }
}

@observer
export class UpscaleTaskContainer extends React.Component<TaskContainerProps> {
    renderTaskStatus() {
        if (this.props.task == null) {
            return <Loading />;
        }
        const ErrorComponent = this.props.errorComponent;
        switch (this.props.task.state.status) {
            case Task.PENDING:
            case Task.RUNNING:
                return <Loading />
            case Task.FAILURE:
                return <ErrorComponent error={this.props.task.state.error} />;
            case Task.SUCCESS:
                return <img src={this.props.task.state.result.dataUrl} />;
        }
    }

    render() {
        return (
            <Panel style={{ width: "100%", textAlign: "center" }}>
                <Panel.Heading>{this.props.title}</Panel.Heading>
                <Panel.Body style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "300px", backgroundColor: "black", color: "white" }}>
                    {this.renderTaskStatus()}
                </Panel.Body>
            </Panel>
        );
    }
}


@observer
export class UpscaleConversionContainer extends React.Component<{ store: UpscaleConversionFlow }> {
    renderUpscaleTask(upscaleTask: UpscaleTask) {
        return (
            <table style={{margin: 0, padding: 0}}>
                {upscaleTask.patchUpscaleTasks.map((row, i) => 
                    <tr key={i}>
                        {row.map((col, j) => (
                            <td>
                                {col && (
                                    col.state.status === Task.SUCCESS
                                    ? <img src={col.state.result.dataUrl} />
                                    : <img src={col.original.dataUrl} />
                                )}
                            </td>    
                        ))}
                    </tr>
                )}
            </table>
        );
    }
    render() {
        return (
            <Panel>
                <Panel.Heading style={{ textAlign: "right" }}>
                    <Button bsStyle="danger" onClick={this.props.store.close}><Glyphicon glyph="remove" /></Button>
                </Panel.Heading>
                <Panel.Body style={{ overflow: "hidden" }}>
                    {(() => {
                        return this.props.store.startedStages.map(({id, task}) => {
                            return task.state.status === Task.SUCCESS ? <img src={task.state.result.dataUrl} /> : null;
                        })
                    })()}
                    {(() => {
                        const stage = this.props.store.currentTask;
                        if (stage === null) {
                            return null;
                        }
                        switch (stage.id) {
                            case "load":
                                switch (stage.task.state.status) {
                                    case Task.PENDING:
                                    case Task.RUNNING:
                                        return <div>ファイルの読み込み中・・・</div>;
                                    case Task.FAILURE:
                                        return <div>ファイルの読み込みに失敗しました: {stage.task.state.error.message} </div>
                                    case Task.SUCCESS:
                                        return null;
                                }
                            case "preScale2x":
                            case "prePadding":
                            case "preAlign":
                                switch (stage.task.state.status) {
                                    case Task.PENDING:
                                    case Task.RUNNING:
                                        return <div>前処理中・・・</div>;
                                    case Task.FAILURE:
                                        return <div>前処理に失敗しました: {stage.task.state.error.message} </div>
                                    case Task.SUCCESS:
                                        return null;
                                }
                            case "upscale":
                                switch (stage.task.state.status) {
                                    case Task.PENDING:
                                    case Task.RUNNING:
                                        return (
                                            <div>
                                                <div>拡大中・・・</div>
                                                {this.renderUpscaleTask(stage.task as UpscaleTask)}
                                            </div>
                                        );
                                    case Task.FAILURE:
                                        return <div>拡大に失敗しました: {stage.task.state.error.message} </div>
                                    case Task.SUCCESS:
                                        return null;
                                }
                        }
                    })()}
                    {/* <Col md={12}>
                        <TaskContainer
                            title="元画像"
                            task={this.props.store.loadImageTask}
                            resultComponent={SimpleImageComponent}
                            errorComponent={GenericErrorComponent({ message: "ファイルの読み込みに失敗しました" })}
                        />
                    </Col>
                    <Col md={12}>
                        <TaskContainer
                            title="元画像(x2)"
                            task={this.props.store.loadImageTask}
                            resultComponent={SimpleImageComponent}
                            errorComponent={GenericErrorComponent({ message: "画像の拡大に失敗しました" })}
                        />
                    </Col> */}
                    {/* <Col md={12}>
                        <Panel style={{ width: "100%", textAlign: "center" }}>
                            <Panel.Heading>変換後</Panel.Heading>
                            <Panel.Body style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "300px", backgroundColor: "black", color: "white" }}>
                                {this.renderConvertedImage()}
                            </Panel.Body>
                        </Panel>
                    </Col> */}
                </Panel.Body>
            </Panel>
        );
    }
}