#!/usr/bin/env python
"""Neo4j 지식베이스를 활용하여 Agent와 대화하는 스크립트.

저장된 Slack 대화 기록을 기반으로 질문에 답변하는 AI 에이전트와 대화합니다.

Usage:
    # 인터랙티브 대화 모드
    uv run python scripts/chat_with_agent.py

    # 단일 질문
    uv run python scripts/chat_with_agent.py "팀에서 최근 논의된 주제가 뭐야?"
"""

import asyncio
import sys

from dotenv import load_dotenv

from doctor_cornelius.agent.manager import AgentManager

load_dotenv()


async def interactive_chat() -> None:
    """인터랙티브 대화 모드."""
    print("=" * 70)
    print("🤖 Doctor Cornelius - 팀 지식베이스 AI 어시스턴트")
    print("=" * 70)
    print("\n저장된 Slack 대화를 기반으로 질문에 답변합니다.")
    print("종료하려면 'exit', 'quit', 또는 '종료'를 입력하세요.\n")

    print("⏳ 에이전트 초기화 중...")
    agent = AgentManager()

    try:
        await agent.initialize()
        print("✅ 에이전트 준비 완료!\n")
        print("-" * 70)

        while True:
            try:
                # 사용자 입력 받기
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                # 종료 명령 확인
                if user_input.lower() in ("exit", "quit", "종료", "q"):
                    print("\n👋 대화를 종료합니다. 안녕히 가세요!")
                    break

                # 특수 명령어
                if user_input.lower() == "/help":
                    print_help()
                    continue

                if user_input.lower() == "/clear":
                    print("\033[2J\033[H")  # 화면 클리어
                    print("🤖 Doctor Cornelius - 대화 계속...")
                    continue

                # 에이전트에게 질문
                print("\n🤔 생각 중...")
                response = await agent.chat(user_input)

                print(f"\n🤖 Agent: {response}")
                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\n👋 대화를 종료합니다.")
                break

            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                print("다시 시도해주세요.")

    finally:
        print("\n⏳ 에이전트 종료 중...")
        await agent.close()
        print("✅ 종료 완료")


async def single_query(question: str) -> None:
    """단일 질문 처리."""
    print("=" * 70)
    print("🤖 Doctor Cornelius - 단일 질문 모드")
    print("=" * 70)

    print(f"\n📝 질문: {question}")
    print("\n⏳ 에이전트 초기화 중...")

    agent = AgentManager()

    try:
        await agent.initialize()
        print("✅ 에이전트 준비 완료")

        print("\n🤔 생각 중...")
        response = await agent.chat(question)

        print("\n" + "=" * 70)
        print("🤖 답변:")
        print("=" * 70)
        print(response)
        print("=" * 70)

    finally:
        await agent.close()


def print_help() -> None:
    """도움말 출력."""
    print("""
📚 사용 가능한 명령어:
  /help   - 이 도움말 보기
  /clear  - 화면 지우기
  exit    - 대화 종료

💡 질문 예시:
  - "최근 팀에서 논의된 주요 주제는 뭐야?"
  - "프로젝트 X에 대해 알려줘"
  - "@홍길동이 언급한 내용 중 중요한 게 있어?"
  - "지난주에 결정된 사항이 있어?"
  - "기술 스택에 대한 논의가 있었어?"

ℹ️  Agent는 Neo4j에 저장된 Slack 대화를 검색하여 답변합니다.
    먼저 ingest_slack_to_neo4j.py로 데이터를 저장해야 합니다.
""")


async def main() -> None:
    """메인 함수."""
    if len(sys.argv) > 1:
        # 명령행 인자로 질문이 주어진 경우
        question = " ".join(sys.argv[1:])
        await single_query(question)
    else:
        # 인터랙티브 모드
        await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())
