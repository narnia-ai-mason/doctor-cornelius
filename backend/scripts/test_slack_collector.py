#!/usr/bin/env python
"""Interactive script to test SlackCollector functionality.

Usage:
    uv run python scripts/test_slack_collector.py channels   # 채널 목록 조회
    uv run python scripts/test_slack_collector.py users      # 사용자 목록 조회
    uv run python scripts/test_slack_collector.py messages   # 특정 날짜 메시지 조회
"""

import asyncio
import sys
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv

from doctor_cornelius.collectors.base import CollectionConfig
from doctor_cornelius.collectors.slack_collector import SlackCollector

load_dotenv()


async def list_channels() -> None:
    """봇이 접근 가능한 채널 목록 조회."""
    print("=" * 60)
    print("📢 채널 목록 조회 (아카이브/외부공유 채널 제외)")
    print("=" * 60)

    collector = SlackCollector()

    # 연결 확인
    if not await collector.validate_connection():
        print("❌ Slack 연결 실패. 토큰을 확인하세요.")
        return

    sources = await collector.list_sources()

    print(f"\n✅ 총 {len(sources)}개 채널 발견\n")

    for i, source in enumerate(sources, 1):
        channel_type = "🔒 비공개" if source.metadata.get("is_private") else "📢 공개"
        print(f"{i:3}. [{channel_type}] #{source.name}")
        print(f"     ID: {source.source_id}")
        print(f"     멤버 수: {source.member_count or 'N/A'}")
        if source.description:
            print(f"     설명: {source.description[:50]}...")
        print()


async def list_users() -> None:
    """유효한 사용자 목록 조회 (삭제/봇/앱 제외)."""
    print("=" * 60)
    print("👥 사용자 목록 조회 (삭제된 사용자/봇/앱 제외)")
    print("=" * 60)

    collector = SlackCollector()

    if not await collector.validate_connection():
        print("❌ Slack 연결 실패. 토큰을 확인하세요.")
        return

    users = await collector.list_users()

    print(f"\n✅ 총 {len(users)}명의 유효한 사용자 발견\n")

    for i, user in enumerate(users, 1):
        display_name = (
            user.get("profile", {}).get("display_name") or user.get("real_name") or user.get("name")
        )
        email = user.get("profile", {}).get("email", "")
        is_restricted = "🔸 게스트" if user.get("is_restricted") else ""

        print(f"{i:3}. {display_name} {is_restricted}")
        print(f"     ID: {user['id']}")
        print(f"     Username: @{user.get('name')}")
        if email:
            print(f"     Email: {email}")
        print()


async def list_messages(
    channel_name: str | None = None,
    date_str: str | None = None,
) -> None:
    """특정 날짜의 메시지와 스레드 조회."""
    print("=" * 60)
    print("💬 메시지 및 스레드 조회")
    print("=" * 60)

    collector = SlackCollector()

    if not await collector.validate_connection():
        print("❌ Slack 연결 실패. 토큰을 확인하세요.")
        return

    # 채널 선택
    sources = await collector.list_sources()

    if channel_name:
        selected = next((s for s in sources if s.name == channel_name), None)
        if not selected:
            print(f"❌ 채널 '{channel_name}'을 찾을 수 없습니다.")
            return
    else:
        print("\n사용 가능한 채널:")
        for i, s in enumerate(sources[:10], 1):
            print(f"  {i}. #{s.name}")

        if len(sources) > 10:
            print(f"  ... 외 {len(sources) - 10}개 채널")

        try:
            choice = input("\n채널 번호를 선택하세요 (1-10): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(sources):
                selected = sources[idx]
            else:
                print("❌ 잘못된 선택입니다.")
                return
        except (ValueError, KeyboardInterrupt):
            print("\n취소되었습니다.")
            return

    # 날짜 선택
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            print(f"❌ 날짜 형식이 잘못되었습니다: {date_str} (YYYY-MM-DD)")
            return
    else:
        date_input = input("\n조회할 날짜 (YYYY-MM-DD, 기본값: 오늘): ").strip()
        if date_input:
            try:
                target_date = datetime.strptime(date_input, "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError:
                print("❌ 날짜 형식이 잘못되었습니다. YYYY-MM-DD 형식으로 입력하세요.")
                return
        else:
            target_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    start_time = target_date
    end_time = target_date + timedelta(days=1)

    print(f"\n📅 조회 범위: {start_time.strftime('%Y-%m-%d')} 00:00 ~ 23:59 UTC")
    print(f"📢 채널: #{selected.name}")
    print("-" * 60)

    config = CollectionConfig(
        source_ids=[selected.source_id],
        start_time=start_time,
        end_time=end_time,
        include_threads=True,
        include_replies=True,
    )

    messages = []
    async for item in collector.collect(config):
        messages.append(item)

    print(f"\n✅ 총 {len(messages)}개 메시지 수집됨\n")

    # 메시지를 스레드별로 그룹화
    threads: dict[str, list] = {}
    standalone = []

    for msg in messages:
        if msg.thread_ts:
            if msg.thread_ts not in threads:
                threads[msg.thread_ts] = []
            threads[msg.thread_ts].append(msg)
        else:
            standalone.append(msg)

    # 출력
    msg_num = 0
    for msg in standalone:
        msg_num += 1
        print(f"[{msg_num}] {msg.author_name} ({msg.timestamp.strftime('%H:%M')})")
        print(f"    {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
        print()

    for thread_ts, thread_msgs in threads.items():
        # 시간순 정렬
        thread_msgs.sort(key=lambda x: x.timestamp)
        parent = thread_msgs[0]
        replies = thread_msgs[1:]

        msg_num += 1
        print(
            f"[{msg_num}] 🧵 {parent.author_name} ({parent.timestamp.strftime('%H:%M')}) - {len(replies)}개 답글"
        )
        print(f"    {parent.content[:100]}{'...' if len(parent.content) > 100 else ''}")

        for reply in replies:
            print(
                f"    └─ {reply.author_name} ({reply.timestamp.strftime('%H:%M')}): {reply.content[:60]}..."
            )

        print()


async def main() -> None:
    """메인 함수."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n사용 가능한 명령어:")
        print("  channels  - 채널 목록 조회")
        print("  users     - 사용자 목록 조회")
        print("  messages  - 메시지/스레드 조회")
        return

    command = sys.argv[1].lower()

    if command == "channels":
        await list_channels()
    elif command == "users":
        await list_users()
    elif command == "messages":
        channel = sys.argv[2] if len(sys.argv) > 2 else None
        date = sys.argv[3] if len(sys.argv) > 3 else None
        await list_messages(channel, date)
    else:
        print(f"❌ 알 수 없는 명령어: {command}")
        print("사용 가능: channels, users, messages")


if __name__ == "__main__":
    asyncio.run(main())
