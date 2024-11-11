import React from "react";
import { Chat } from "./Chat.tsx";
import { Header } from "/Header.tsx";

export function Home() {
  return (
    <>
      <Header />
      <div className="w-full">
        <Chat />
      </div>
    </>
  );
}
