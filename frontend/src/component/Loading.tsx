import * as React from "react";

export interface LoadingProps {
    backgroundColor?: string;
}
export const Loading = (props: LoadingProps) => (
    <div className="ball-pulse" style={{backgroundColor: props.backgroundColor}}>
        <div />
        <div />
        <div />
    </div>
);