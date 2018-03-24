import * as React from "react";
import * as ReactDOM from "react-dom";
import { App } from "./store/App";
import { AppContainer } from "./container/App";
import MobXDevTools from 'mobx-react-devtools';

import { configureDevtool } from 'mobx-react-devtools';

const store = new App();
ReactDOM.render(
    <div>
        <AppContainer store={store} />
        <MobXDevTools />
    </div>
, document.getElementById("app"));