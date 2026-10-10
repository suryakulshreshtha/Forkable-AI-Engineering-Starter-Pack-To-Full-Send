"""The judge for drill 6.3."""


def test_rrf_rewards_agreement(m):
    keyword = ["d1", "d2", "d3"]
    vector = ["d4", "d2", "d5"]
    fused = m.rrf([keyword, vector])
    ids = [d for d, _ in fused]
    assert ids == ["d2", "d1", "d4", "d3", "d5"], f"got {ids}"
    assert abs(fused[0][1] - 2 / 62) < 1e-12


def test_rrf_k_controls_how_much_the_top_rank_dominates(m):
    lists = [["a", "b", "c"], ["a", "c", "b"]]
    flat = dict(m.rrf(lists, k=1000))
    steep = dict(m.rrf(lists, k=0))
    lead = lambda s: s["a"] - s["b"]
    assert lead(steep) > 100 * lead(flat), "a small k lets first place dominate; a large k flattens"
    assert abs(steep["a"] - 2.0) < 1e-12, "with k=0, first place in both lists scores 1/1 + 1/1"


def test_rrf_of_one_list_keeps_its_order(m):
    assert [d for d, _ in m.rrf([["z", "y", "x"]])] == ["z", "y", "x"]


def test_recall_at_k(m):
    ranked = ["a", "b", "c", "d", "e"]
    assert m.recall_at_k(ranked, {"a", "d"}, 1) == 0.5
    assert m.recall_at_k(ranked, {"a", "d"}, 4) == 1.0
    assert m.recall_at_k(ranked, {"zzz"}, 5) == 0.0
    assert m.recall_at_k(ranked, set(), 3) == 0.0


def test_reciprocal_rank(m):
    assert m.reciprocal_rank(["x", "y", "z"], {"z"}) == 1 / 3
    assert m.reciprocal_rank(["x", "y", "z"], {"y", "z"}) == 1 / 2, "only the first hit counts"
    assert m.reciprocal_rank(["x"], {"q"}) == 0.0


def test_mean_reciprocal_rank(m):
    runs = [(["a", "b"], {"a"}), (["a", "b"], {"b"}), (["a", "b"], {"c"})]
    assert abs(m.mean_reciprocal_rank(runs) - (1 + 0.5 + 0) / 3) < 1e-12
    assert m.mean_reciprocal_rank([]) == 0.0


def test_fusion_can_rescue_recall(m):
    relevant = {"good1", "good2"}
    keyword = ["good1", "noise1", "noise2", "good2"]
    vector = ["good2", "noise3", "good1", "noise4"]
    fused = [d for d, _ in m.rrf([keyword, vector])]
    assert m.recall_at_k(keyword, relevant, 2) == 0.5
    assert m.recall_at_k(fused, relevant, 2) == 1.0, "both good docs rank high in at least one list"
