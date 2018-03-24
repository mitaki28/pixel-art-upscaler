import * as React from "react";
import { observer } from "mobx-react"
import { ImageConversionList } from "../store/ImageConversion";
import { ImageConversionContainer } from "./ImageConversion";

@observer
export class ImageConversionListContainer extends React.Component<{ store: ImageConversionList }> {
    render() {
        return (
            <table>
                {
                    this.props.store.conversions.map((conversion) => 
                        <ImageConversionContainer store={conversion} />
                    )
                }
            </table>
        );
    }
}