import sys
import io

# Windows 终端 UTF-8 输出支持
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from recommender_engine import RecommenderEngine


def interactive_recommendation():
    """
    交互式冷启动推荐与 I2I 相似推荐逻辑 (加入 E&E 动态强化探测策略)
    重构为调用 RecommenderEngine，保留原有命令行交互体验。
    """
    engine = RecommenderEngine()
    result = engine.start_session()
    session_id = result["session_id"]
    target_n = result["target_n"]

    print("\n" + "=" * 60)
    print("欢迎使用番剧推荐系统 - 动态强化冷启动推断")
    print("=" * 60)
    print(f"\n[Step 1] 构建初始兴趣序列 (目标: 至少收集 {target_n} 部)")
    print("输入说明: y(看过/喜欢), n(没看过/不喜欢), f(提前结束并推荐)\n")

    # 阶段 1：收集阶段
    while result["current_count"] < target_n:
        candidate = result["candidate"]
        if candidate is None:
            print("  -> 所有候选番剧已浏览完毕")
            break

        tag = candidate["tag"]
        name = candidate["name"]
        item_id = candidate["item_id"]

        ans = input(f"{tag} 《{name}》: ").strip().lower()

        if ans in ["f", "finish", "q", "quit"]:
            if result["current_count"] < target_n:
                confirm = input(
                    f"  -> 仅收集 {result['current_count']} 部可能导致推荐偏差，确认直接推荐？(y/n): "
                ).strip().lower()
                if confirm in ["y", "yes"]:
                    break
                else:
                    continue
            else:
                break
        elif ans in ["y", "yes", "1"]:
            result = engine.submit_feedback(session_id, item_id, "like")
            print(f"  -> [记录成功] 进度: {result['current_count']} / {target_n}")
        elif ans in ["n", "no", "0", ""]:
            result = engine.submit_feedback(session_id, item_id, "dislike")
        else:
            print("  -> 输入无效，请输入 y, n, 或 f。")

    # 阶段 2：自由探索阶段
    if result["current_count"] >= target_n:
        print(
            f"\n[Info] 已达标 ({target_n}部)。您可以继续标记以提升模型精度，或输入 'f' 立即获取最终推荐。"
        )
        while True:
            candidate = result["candidate"]
            if candidate is None:
                print("  -> 所有候选番剧已浏览完毕")
                break

            tag = candidate["tag"]
            name = candidate["name"]
            item_id = candidate["item_id"]

            ans = input(f"{tag} 《{name}》: ").strip().lower()

            if ans in ["f", "finish", "q", "quit", ""]:
                break
            elif ans in ["y", "yes", "1"]:
                result = engine.submit_feedback(session_id, item_id, "like")
                print(f"  -> [记录成功] 总数: {result['current_count']}")
            elif ans in ["n", "no", "0"]:
                result = engine.submit_feedback(session_id, item_id, "dislike")

    # 阶段 3：生成推荐
    try:
        rec = engine.recommend(session_id)
    except ValueError:
        print("\n[Error] 未提供有效序列，程序终止。")
        return

    print(f"\n[System] 正在基于 {rec['liked_count']} 部番剧进行张量推断...")

    print("\n" + "=" * 60)
    print(" [Step 2] 您的专属 Top-10 个性化推荐")
    print("=" * 60)
    for item in rec["top10"]:
        print(f" Top {item['rank']:2d} | 《{item['name']}》")

    print("\n" + "=" * 60)
    print(" [Step 3] 看了又看：深度关联探索 (基于 Top-3 推荐)")
    print("=" * 60)

    for group in rec["related"]:
        source = group["source"]
        print(f"\n >>> 因为为您推荐了: 《{source['name']}》")
        print("     看这部番的用户，强烈关联了以下作品：")
        for sim in group["items"]:
            print(f"      - 《{sim['name']}》")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        interactive_recommendation()
    except Exception as e:
        print(f"[Error] 程序运行异常: {e}")
        sys.exit(1)
