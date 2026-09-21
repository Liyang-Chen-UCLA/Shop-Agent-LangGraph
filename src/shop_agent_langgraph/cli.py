from __future__ import annotations

import argparse

from .supervisor import supervisor


EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", "退出"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the Shop Agent supervisor.")
    parser.add_argument(
        "--thread-id",
        default="cli",
        help="Conversation identifier used to retain Supervisor state (default: cli).",
    )
    args = parser.parse_args()

    print("Shop Agent 已启动。输入 exit、quit 或 退出结束对话。")
    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n对话已结束。")
            break

        if not user_input:
            continue
        if user_input.casefold() in EXIT_COMMANDS:
            print("对话已结束。")
            break

        try:
            reply = supervisor.invoke(user_input, thread_id=args.thread_id)
        except Exception as exc:
            print(f"\n系统错误：{exc}")
            continue
        print(f"\nSupervisor：{reply}")
