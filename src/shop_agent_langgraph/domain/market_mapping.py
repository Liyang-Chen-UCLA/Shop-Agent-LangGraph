from __future__ import annotations

from typing import Final

from .taxonomy import load_taxonomy


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
    requested_id = str(node_id)
    _, nodes_by_id, _ = load_taxonomy()
    current_id: str | None = requested_id
    visited: set[str] = set()

    while current_id is not None:
        category = NODE_TO_DATASET_CATEGORY.get(current_id)
        if category is not None:
            return category
        if current_id in visited:
            raise ValueError(f"cycle in taxonomy parent chain at node ID: {current_id}")
        visited.add(current_id)

        node = nodes_by_id.get(current_id)
        if node is None:
            raise ValueError(f"unknown taxonomy node ID: {current_id}")
        current_id = node["parent_id"]

    raise ValueError(
        f"no market dataset mapping for taxonomy node ID: {requested_id}"
    )
