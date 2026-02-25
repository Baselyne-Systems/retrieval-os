"""Unit tests for the sparse/BM25 tokeniser and vector builder."""

from __future__ import annotations

import pytest

from retrieval_os.serving.sparse import (
    VOCAB_SIZE,
    text_to_sparse_vector,
    tokenize,
)

# ── tokenize ──────────────────────────────────────────────────────────────────


class TestTokenize:
    def test_lowercase_words(self) -> None:
        assert tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        assert tokenize("foo, bar! baz.") == ["foo", "bar", "baz"]

    def test_digits_included(self) -> None:
        assert "42" in tokenize("answer is 42")

    def test_alphanumeric_mixed(self) -> None:
        tokens = tokenize("gpt4 text-embedding-3-small")
        assert "gpt4" in tokens
        assert "text" in tokens
        assert "embedding" in tokens
        assert "3" in tokens
        assert "small" in tokens

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_whitespace_only(self) -> None:
        assert tokenize("   \t\n  ") == []

    def test_unicode_stripped(self) -> None:
        # Non-ascii chars are not matched by [a-z0-9]+
        tokens = tokenize("café")
        assert "caf" in tokens  # "é" stripped

    def test_repeated_tokens_preserved(self) -> None:
        tokens = tokenize("the cat sat on the mat the")
        assert tokens.count("the") == 3


# ── text_to_sparse_vector ─────────────────────────────────────────────────────


class TestTextToSparseVector:
    def test_empty_returns_empty_dict(self) -> None:
        assert text_to_sparse_vector("") == {}

    def test_all_punctuation_returns_empty_dict(self) -> None:
        assert text_to_sparse_vector("!!!  ???") == {}

    def test_single_token(self) -> None:
        vec = text_to_sparse_vector("hello")
        assert len(vec) == 1
        # Only one token → TF = 1/1 = 1.0
        assert list(vec.values())[0] == pytest.approx(1.0)

    def test_indices_in_vocab_range(self) -> None:
        vec = text_to_sparse_vector("retrieval augmented generation system")
        for idx in vec:
            assert 0 <= idx < VOCAB_SIZE

    def test_tf_weights_sum_to_one(self) -> None:
        """For a string of n unique tokens, each TF = 1/n → sum = 1.0."""
        text = "alpha beta gamma delta epsilon"
        vec = text_to_sparse_vector(text)
        assert sum(vec.values()) == pytest.approx(1.0)

    def test_repeated_token_has_higher_weight(self) -> None:
        """'cat' appears 3× out of 5 tokens → TF = 0.6; 'dog' = 0.4."""
        vec = text_to_sparse_vector("cat cat cat dog dog")
        cat_idx = hash("cat") % VOCAB_SIZE
        dog_idx = hash("dog") % VOCAB_SIZE
        if cat_idx != dog_idx:  # guard against hash collision
            assert vec[cat_idx] == pytest.approx(0.6)
            assert vec[dog_idx] == pytest.approx(0.4)

    def test_deterministic(self) -> None:
        text = "stable hashing trick vector"
        assert text_to_sparse_vector(text) == text_to_sparse_vector(text)

    def test_hash_collision_merges(self) -> None:
        """Two different tokens that collide to same index add their TF weights.
        We can't easily force a collision, so we just verify the output is valid.
        """
        vec = text_to_sparse_vector("the quick brown fox jumps over the lazy dog")
        # All values should be positive
        assert all(v > 0 for v in vec.values())
        # All indices in range
        assert all(0 <= idx < VOCAB_SIZE for idx in vec)

    def test_keys_are_integers(self) -> None:
        vec = text_to_sparse_vector("embedding model latency throughput")
        for k in vec:
            assert isinstance(k, int)

    def test_values_are_floats(self) -> None:
        vec = text_to_sparse_vector("test query")
        for v in vec.values():
            assert isinstance(v, float)


# ── VOCAB_SIZE constant ───────────────────────────────────────────────────────


class TestVocabSize:
    def test_is_power_of_two(self) -> None:
        assert VOCAB_SIZE == 2**16

    def test_is_65536(self) -> None:
        assert VOCAB_SIZE == 65_536
