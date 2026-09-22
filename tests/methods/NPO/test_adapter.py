from dataclasses import replace

import pytest

from experiments.reproductions.NPO.adapter import (
    AdapterSpec,
    collate_examples,
    encode_example,
    encode_rows,
)


class FixtureTokenizer:
    eos_token_id = 9
    pad_token_id = 9  # Deliberate alias: real EOS must remain attended/scored.
    unk_token_id = 99

    def convert_tokens_to_ids(self, token):
        return 9

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, date_string, return_dict
    ):
        assert tokenize and date_string == "10 Apr 2025"
        assert return_dict is False
        if add_generation_prompt:
            return [1, 2, 3]
        answer = messages[-1]["content"]
        return [1, 2, 3] + {
            "ordinary": [4, 5],
            "short": [6],
            "ended": [7, 9],
            "long": [4, 5, 6, 7],
        }[answer]


def spec(**kwargs):
    return AdapterSpec(
        "fixture/model", "a" * 40, "fixture/tokenizer", "b" * 40, "fixture/data", "c" * 40, **kwargs
    )


def example(answer="ordinary", index=0, configuration=None):
    return encode_example(
        {"question": "Q?", "answer": answer},
        index,
        "forget05",
        FixtureTokenizer(),
        configuration or spec(),
    )


def test_hand_constructed_answer_eos_and_metadata():
    x = example()
    assert x["input_ids"].tolist() == [1, 2, 3, 4, 5, 9]
    assert x["labels"].tolist() == [-100, -100, -100, 4, 5, 9]
    assert x["attention_mask"].tolist() == [1] * 6
    assert x["prompt_token_count"] == 3 and x["scored_token_count"] == 3
    assert x["truncation"] == {
        "original_length": 6,
        "length": 6,
        "truncated": False,
        "eos_retained": True,
    }
    assert x["identities"]["tokenizer_revision"] == "b" * 40
    assert x["sample_id"].endswith("/forget05/0")
    assert example()["tensors_sha256"] == x["tensors_sha256"]
    assert example()["source_record_sha256"] == x["source_record_sha256"]
    assert example("ended")["input_ids"].tolist() == [1, 2, 3, 7, 9]


@pytest.mark.parametrize("side", ["right", "left"])
def test_padding_is_ignored_but_real_eos_stays_scored(side):
    cfg = spec(padding_side=side)
    batch = collate_examples(
        [example(configuration=cfg), example("short", 1, cfg)], FixtureTokenizer(), cfg
    )
    t = batch["tensors"]
    if side == "right":
        assert t["labels"][1].tolist() == [-100, -100, -100, 6, 9, -100]
        assert t["attention_mask"][1].tolist() == [1, 1, 1, 1, 1, 0]
    else:
        assert t["labels"][1].tolist() == [-100, -100, -100, -100, 6, 9]
        assert t["attention_mask"][1].tolist() == [0, 1, 1, 1, 1, 1]
    assert t["labels"][t["attention_mask"] == 0].tolist() == [-100]
    assert batch["sample_ids"] == [
        example(configuration=cfg)["sample_id"],
        example("short", 1, cfg)["sample_id"],
    ]


def test_truncation_is_explicit_and_never_replaces_cut_answer_with_eos():
    x = example("long", configuration=spec(max_length=4))
    assert x["input_ids"].tolist() == [1, 2, 3, 4]
    assert x["labels"].tolist() == [-100, -100, -100, 4]
    assert x["truncation"]["truncated"] and not x["truncation"]["eos_retained"]
    with pytest.raises(ValueError, match="fully removed"):
        example(configuration=spec(max_length=3))


@pytest.mark.parametrize("answer", ["", "   ", None, [], 17])
def test_invalid_answer_rejected(answer):
    with pytest.raises(ValueError, match="Nonempty"):
        encode_example(
            {"question": "Q", "answer": answer}, 0, "forget05", FixtureTokenizer(), spec()
        )


def test_no_content_no_causal_target_and_prefix_mismatch_rejected():
    class Bad(FixtureTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3] if kwargs["add_generation_prompt"] else [1, 2, 3, 9]

    with pytest.raises(ValueError, match="no content"):
        encode_example({"question": "Q", "answer": "A"}, 0, "forget05", Bad(), spec())

    class Mismatch(FixtureTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if kwargs["add_generation_prompt"] else [1, 7, 4, 9]

    with pytest.raises(ValueError, match="prefix"):
        encode_example({"question": "Q", "answer": "A"}, 0, "forget05", Mismatch(), spec())


def test_order_hashes_and_mutation_fail_closed():
    cfg = spec()
    tok = FixtureTokenizer()
    rows = [{"question": "Q", "answer": "ordinary"}, {"question": "Q", "answer": "short"}]
    xs = encode_rows(rows, "retain95", tok, cfg)
    a = collate_examples(xs, tok, cfg)
    b = collate_examples(list(reversed(xs)), tok, cfg)
    assert a["sample_ids_sha256"] != b["sample_ids_sha256"]
    xs[0]["labels"][3] = -100
    with pytest.raises(ValueError, match="changed"):
        collate_examples(xs, tok, cfg)
    with pytest.raises(ValueError):
        replace(spec(), model_revision="main").validate()
