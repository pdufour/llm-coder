import React from "react";
import { Chat } from "./Chat.js";
import { Header } from "./Header.js";

const h = React.createElement;

export const Home = () => h(React.Fragment, null,
  h(Header),
  h('div', { className: 'w-full' },
    h(Chat)
  )
);
