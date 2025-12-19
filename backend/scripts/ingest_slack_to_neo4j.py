#!/usr/bin/env python
"""Slack 메시지를 Neo4j 지식베이스에 저장하는 스크립트.

특정 날짜의 Slack 대화(스레드 및 댓글 포함)를 수집하고
Graphiti를 통해 Neo4j에 Episode로 저장합니다.

Usage:
    # 오늘 날짜의 메시지 수집
    uv run python scripts/ingest_slack_to_neo4j.py

    # 특정 날짜의 메시지 수집
    uv run python scripts/ingest_slack_to_neo4j.py 2024-12-18

    # 특정 채널만 수집
    uv run python scripts/ingest_slack_to_neo4j.py 2024-12-18 --channel general
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from doctor_cornelius.collectors.base import CollectionConfig
from doctor_cornelius.collectors.slack_collector import SlackCollector
from doctor_cornelius.knowledge.graph_client import GraphitiClientManager
from doctor_cornelius.schemas.episode import Episode
from doctor_cornelius.transformers.slack_transformer import SlackTransformer

load_dotenv()

# KST timezone (UTC+9)
KST = timezone(timedelta(hours=9))


async def ingest_slack_messages(
    target_date: datetime,
    channel_name: str | None = None,
    batch_size: int = 10,
) -> None:
    """특정 날짜의 Slack 메시지를 Neo4j에 저장.

    Args:
        target_date: 수집할 날짜 (KST 기준, 해당 일의 00:00 ~ 23:59 KST)
        channel_name: 특정 채널만 수집할 경우 채널명
        batch_size: 한 번에 저장할 에피소드 수
    """
    print("=" * 70)
    print("🚀 Slack 메시지 → Neo4j 지식베이스 저장")
    print("=" * 70)

    # KST 00:00 ~ 23:59:59를 UTC로 변환
    kst_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=KST)
    kst_end = kst_start + timedelta(days=1)

    # UTC로 변환하여 Slack API에 전달
    start_time = kst_start.astimezone(timezone.utc)
    end_time = kst_end.astimezone(timezone.utc)

    print(f"\n📅 수집 기간: {kst_start.strftime('%Y-%m-%d')} 00:00 ~ 23:59 KST")
    print(f"   (UTC: {start_time.strftime('%Y-%m-%d %H:%M')} ~ {end_time.strftime('%Y-%m-%d %H:%M')})")

    # 1. Slack Collector 초기화
    print("\n[1/5] Slack 연결 확인 중...")
    collector = SlackCollector()

    if not await collector.validate_connection():
        print("❌ Slack 연결 실패. 토큰을 확인하세요.")
        return

    print("✅ Slack 연결 성공")

    # 2. 채널 목록 가져오기
    print("\n[2/5] 채널 목록 조회 중...")
    sources = await collector.list_sources()
    print(f"✅ {len(sources)}개 채널 발견")

    # 특정 채널 필터링
    if channel_name:
        sources = [s for s in sources if s.name == channel_name]
        if not sources:
            print(f"❌ 채널 '{channel_name}'을 찾을 수 없습니다.")
            return
        print(f"📢 선택된 채널: #{channel_name}")
    else:
        print("📢 모든 접근 가능한 채널에서 수집합니다.")
        for s in sources[:5]:
            print(f"   - #{s.name}")
        if len(sources) > 5:
            print(f"   ... 외 {len(sources) - 5}개 채널")

    # 3. 메시지 수집
    print("\n[3/5] 메시지 수집 중...")
    config = CollectionConfig(
        source_ids=[s.source_id for s in sources],
        start_time=start_time,
        end_time=end_time,
        include_threads=True,
        include_replies=True,
    )

    raw_items = []
    async for item in collector.collect(config):
        raw_items.append(item)
        if len(raw_items) % 50 == 0:
            print(f"   📨 {len(raw_items)}개 메시지 수집됨...")

    print(f"✅ 총 {len(raw_items)}개 메시지 수집 완료")

    if not raw_items:
        print("\n⚠️  수집된 메시지가 없습니다.")
        return

    # 4. Episode로 변환
    print("\n[4/5] Episode로 변환 중...")

    # 사용자 이름 조회 함수
    async def user_resolver(user_id: str) -> str | None:
        return await collector._get_user_name(user_id)

    transformer = SlackTransformer(user_resolver=user_resolver)

    episodes: list[Episode] = []
    skipped = 0

    for item in raw_items:
        episode = await transformer.transform(item)
        if episode:
            episodes.append(episode)
        else:
            skipped += 1

    print(f"✅ {len(episodes)}개 에피소드 생성 (시스템 메시지 {skipped}개 제외)")

    if not episodes:
        print("\n⚠️  변환된 에피소드가 없습니다.")
        return

    # 5. Neo4j에 저장
    print("\n[5/5] Neo4j에 저장 중...")
    print(f"   배치 크기: {batch_size}개씩 저장")

    async with GraphitiClientManager() as graph_client:
        total_entities = 0
        total_relationships = 0
        total_saved = 0

        # 배치 단위로 저장
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i : i + batch_size]

            try:
                # 채널별로 그룹화하여 batch 저장
                # group_id는 첫 번째 에피소드의 group_id 사용
                result = await graph_client.ingest_episodes_batch(
                    episodes=batch,
                    group_id=batch[0].group_id,
                )

                total_saved += result["episode_count"]
                total_entities += len(result["entities"])
                total_relationships += len(result["relationships"])

                print(
                    f"   ✅ 배치 {i // batch_size + 1}: "
                    f"{result['episode_count']}개 저장, "
                    f"{len(result['entities'])}개 엔티티, "
                    f"{len(result['relationships'])}개 관계"
                )

            except Exception as e:
                print(f"   ❌ 배치 {i // batch_size + 1} 저장 실패: {e}")

                # 개별 저장 시도
                print("   🔄 개별 저장 모드로 전환...")
                for episode in batch:
                    try:
                        result = await graph_client.ingest_episode(episode)
                        total_saved += 1
                        total_entities += len(result["entities"])
                        total_relationships += len(result["relationships"])
                    except Exception as e2:
                        print(f"      ❌ 에피소드 저장 실패: {episode.name[:30]}... - {e2}")

    # 결과 출력
    print("\n" + "=" * 70)
    print("📊 저장 완료!")
    print("=" * 70)
    print(f"   📝 저장된 에피소드: {total_saved}개")
    print(f"   🔵 추출된 엔티티: {total_entities}개")
    print(f"   🔗 추출된 관계: {total_relationships}개")
    print(f"   📅 수집 날짜: {target_date.strftime('%Y-%m-%d')}")
    print("\n💡 Neo4j Browser에서 확인:")
    print("   MATCH (e:Entity) RETURN e LIMIT 25")
    print("   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")


async def main() -> None:
    """메인 함수."""
    # 날짜 파싱 (KST 기준)
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        try:
            target_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").replace(tzinfo=KST)
        except ValueError:
            print(f"❌ 날짜 형식이 잘못되었습니다: {sys.argv[1]}")
            print("   올바른 형식: YYYY-MM-DD (예: 2024-12-18)")
            return
    else:
        # 현재 KST 날짜
        target_date = datetime.now(KST).replace(hour=0, minute=0, second=0, microsecond=0)

    # 채널 파싱
    channel_name = None
    for i, arg in enumerate(sys.argv):
        if arg == "--channel" and i + 1 < len(sys.argv):
            channel_name = sys.argv[i + 1]
            break

    await ingest_slack_messages(target_date, channel_name)


if __name__ == "__main__":
    asyncio.run(main())
