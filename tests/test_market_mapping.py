import pytest

from shop_agent_langgraph.domain.market_mapping import dataset_category_for_node


def test_dataset_category_uses_direct_mapping() -> None:
    assert dataset_category_for_node("3530") == "狗全价膨化粮"


def test_dataset_category_inherits_nearest_parent_mapping() -> None:
    assert dataset_category_for_node("543682") == "狗全价膨化粮"


def test_dataset_category_rejects_known_unmapped_tree() -> None:
    with pytest.raises(ValueError, match="no market dataset mapping"):
        dataset_category_for_node("632")


def test_dataset_category_rejects_unknown_node() -> None:
    with pytest.raises(ValueError, match="unknown taxonomy node ID"):
        dataset_category_for_node("not-a-node")
