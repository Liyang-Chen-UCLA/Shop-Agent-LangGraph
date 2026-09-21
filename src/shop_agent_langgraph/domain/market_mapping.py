from __future__ import annotations

from typing import Final


NODE_TO_DATASET_CATEGORY: Final[dict[str, str]] = {
    "3375": "乒乓底板",
    "2700": "乳胶枕",
    "5598": "儿童冲锋衣",
    "5322": "儿童冲锋衣",
    "5690": "助听器",
    "264": "手机直播补光灯",
    "2926": "手机直播补光灯",
    "2394": "手机直播补光灯",
    "301": "游戏手柄",
    "3530": "狗全价膨化粮",
    "2844": "男士防晒乳霜",
    "605": "移动空调",
    "1062": "羽毛球包",
    "1868": "胶囊咖啡",
}


def dataset_category_for_node(node_id: str) -> str:
    return NODE_TO_DATASET_CATEGORY[str(node_id)]
