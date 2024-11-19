import React from "react";
import ReactDOM from "react-dom/client";

import { App } from "/App.js";

// await navigator.serviceWorker.ready;
const rootElement = document.getElementById("root");
const root = ReactDOM.createRoot(rootElement);
root.render(React.createElement(App));
