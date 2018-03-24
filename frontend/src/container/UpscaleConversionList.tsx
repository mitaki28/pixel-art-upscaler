import * as React from "react";
import { observer } from "mobx-react"
import { UpscaleConversionList } from "../store/UpscaleConversion";
import { UpscaleConversionContainer } from "./UpscaleConversion";
import { Grid, Row } from "react-bootstrap";

@observer
export class UpscaleConversionListContainer extends React.Component<{ store: UpscaleConversionList }> {
    render() {
        return (
            <Grid>
                {
                    this.props.store.conversions.map((conversion) => 
                        <Row><UpscaleConversionContainer key={conversion.id} store={conversion} /></Row>
                    )
                }
            </Grid>
        );
    }
}