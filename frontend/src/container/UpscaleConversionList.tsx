import * as React from "react";
import { observer } from "mobx-react"
import { UpscaleConversionList } from "../store/UpscaleConversion";
import { UpscaleConversionContainer } from "./UpscaleConversion";
import { Grid, Row } from "react-bootstrap";

@observer
export class UpscaleConversionListContainer extends React.Component<{ store: UpscaleConversionList }> {
    render() {
        return (
            <div>
                {
                    this.props.store.conversions.map((conversion) => 
                        <UpscaleConversionContainer key={conversion.id} store={conversion} />
                    )
                }
            </div>
        );
    }
}