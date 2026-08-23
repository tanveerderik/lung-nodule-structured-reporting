#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dynamic_template_constraint import (  # noqa: E402
    DynamicTemplateLogitsProcessor,
    DynamicTemplateRuntime,
    LiteralNode,
    SlotNode,
    TemplateStateError,
    TemplateStateMachine,
    validate_dense_prediction,
)


class FakeTokenizer:
    """Deterministic character tokenizer with atomic added special tokens."""

    def __init__(self):
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.additional_special_tokens = []
        self._special_to_id = {}
        self._id_to_special = {}

    def __len__(self):
        return 1000 + len(self._special_to_id)

    def add_special_tokens(self, payload):
        for token in payload.get("additional_special_tokens", []):
            if token not in self._special_to_id:
                token_id = 1000 + len(self._special_to_id)
                self._special_to_id[token] = token_id
                self._id_to_special[token_id] = token
                self.additional_special_tokens.append(token)

    @staticmethod
    def _char_id(char):
        return 10 + ord(char)

    def encode(self, text, add_special_tokens=False):
        out = [self.bos_token_id] if add_special_tokens else []
        i = 0
        specials = sorted(self._special_to_id, key=len, reverse=True)
        while i < len(text):
            matched = None
            for token in specials:
                if text.startswith(token, i):
                    matched = token
                    break
            if matched is not None:
                out.append(self._special_to_id[matched])
                i += len(matched)
            else:
                out.append(self._char_id(text[i]))
                i += 1
        if add_special_tokens:
            out.append(self.eos_token_id)
        return out

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del clean_up_tokenization_spaces
        chars = []
        for token_id in token_ids:
            if token_id in self._id_to_special:
                if not skip_special_tokens:
                    chars.append(self._id_to_special[token_id])
            elif token_id in {self.bos_token_id, self.eos_token_id}:
                continue
            else:
                chars.append(chr(token_id - 10))
        return "".join(chars)


class InvisibleBoundaryTokenizer(FakeTokenizer):
    """Model a tokenizer whose standalone scalars start with a blank token."""

    boundary_token_id = 3

    def encode(self, text, add_special_tokens=False):
        # Placeholder markers must remain atomic for runtime registration.
        if text in self._special_to_id:
            return super().encode(text, add_special_tokens=add_special_tokens)
        encoded = super().encode(text, add_special_tokens=add_special_tokens)
        if text and not add_special_tokens:
            return [self.boundary_token_id, *encoded]
        return encoded

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        visible = [
            token_id for token_id in token_ids
            if token_id != self.boundary_token_id
        ]
        return super().decode(
            visible,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )


def make_runtime(*, legacy_compat=False, max_dynamic_nodules=49):
    tokenizer = FakeTokenizer()
    return DynamicTemplateRuntime(
        tokenizer,
        template_style="canonical",
        include_json_tags=False,
        legacy_compat=legacy_compat,
        max_dynamic_nodules=max_dynamic_nodules,
        original_vocab_size=1000,
    )


def make_special_token_runtime():
    return DynamicTemplateRuntime(
        FakeTokenizer(),
        template_style="legacy_snake_case",
        include_json_tags=False,
        legacy_compat=True,
        template_add_special_tokens=True,
        max_dynamic_nodules=None,
        original_vocab_size=1000,
    )


def find_slot(runtime, marker, count=1):
    program = runtime.get_program(count)
    for index, node in enumerate(program.nodes):
        if isinstance(node, SlotNode) and node.marker == marker:
            next_node = program.nodes[index + 1]
            assert isinstance(next_node, LiteralNode)
            return node, next_node.token_ids[0]
    raise AssertionError(f"Missing slot {marker}")


class DynamicTemplateStateMachineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = make_runtime(legacy_compat=False)
        cls.machine = TemplateStateMachine(cls.runtime)

    def test_forced_literal_token(self):
        replay = self.machine.replay([])
        self.assertEqual(replay.action.kind, "fixed")
        first_node = self.runtime.seed_program.nodes[0]
        self.assertIsInstance(first_node, LiteralNode)
        self.assertEqual(replay.action.token_id, first_node.token_ids[0])

    def test_one_token_candidate_and_terminator(self):
        slot, terminator = find_slot(self.runtime, "<|lung_rads|>")
        tokens = self.runtime.encode("0", add_special_tokens=False)
        self.assertEqual(len(tokens), 1)
        consumed, complete, _, chosen = self.machine._slot_progress(
            slot, terminator, tokens + [terminator]
        )
        self.assertTrue(complete)
        self.assertEqual(consumed, 2)
        self.assertEqual(chosen, "0")

    def test_multi_token_candidate(self):
        slot, terminator = find_slot(self.runtime, "<|lobe|>")
        tokens = self.runtime.encode("right upper lobe", add_special_tokens=False)
        self.assertGreater(len(tokens), 1)
        _, complete, _, chosen = self.machine._slot_progress(
            slot, terminator, tokens + [terminator]
        )
        self.assertTrue(complete)
        self.assertEqual(chosen, "right upper lobe")

    def test_prefix_overlapping_candidates(self):
        slot, terminator = find_slot(self.runtime, "<|lobe|>")
        prefix = self.runtime.encode("right ", add_special_tokens=False)
        _, complete, action, _ = self.machine._slot_progress(
            slot, terminator, prefix
        )
        self.assertFalse(complete)
        expected = {
            self.runtime.encode(letter, add_special_tokens=False)[0]
            for letter in ("u", "m", "l")
        }
        self.assertTrue(expected.issubset(set(action.allowed_token_ids)))

    def test_null_candidate(self):
        slot, terminator = find_slot(self.runtime, "<|margin|>")
        tokens = self.runtime.encode("null", add_special_tokens=False)
        _, complete, _, chosen = self.machine._slot_progress(
            slot, terminator, tokens + [terminator]
        )
        self.assertTrue(complete)
        self.assertEqual(chosen, "null")

    def test_count_zero_omits_nodules_and_is_valid_json(self):
        tokens = self.runtime.render_tokens(0)
        text = self.runtime.decode(tokens)
        obj = json.loads(text)
        self.assertEqual(obj["Number of Nodules"], "0")
        self.assertNotIn("Nodules", obj)
        validation = validate_dense_prediction(obj, raw_text=text)
        self.assertTrue(validation["valid"], validation)
        replay = self.machine.replay(tokens)
        self.assertEqual(replay.action.kind, "eos")

    def test_count_one_traverses_every_slot(self):
        tokens = self.runtime.render_tokens(
            1,
            {"Nodule[0].Lobe": "right upper lobe"},
        )
        text = self.runtime.decode(tokens)
        obj = json.loads(text)
        self.assertEqual(len(obj["Nodules"]), 1)
        self.assertEqual(
            obj["Nodules"][0]["Lobe"],
            "right upper lobe",
        )
        self.assertEqual(obj["Nodules"][0]["Margin"], "null")
        validation = validate_dense_prediction(obj, raw_text=text)
        self.assertTrue(validation["valid"], validation)
        trace = self.machine.trace(tokens)
        slot_names = {
            item.get("slot") for item in trace["trace"] if "slot" in item
        }
        self.assertIn("Nodule[0].Comparison Date", slot_names)
        self.assertNotIn("<|lobe|>", text)

    def test_count_greater_than_one_repeats_full_dictionary(self):
        tokens = self.runtime.render_tokens(
            2,
            {
                "Nodule[0].Lobe": "right upper lobe",
                "Nodule[1].Lobe": "left lower lobe",
            },
        )
        text = self.runtime.decode(tokens)
        obj = json.loads(text)
        self.assertEqual(obj["Number of Nodules"], "2")
        self.assertEqual(len(obj["Nodules"]), 2)
        self.assertEqual(obj["Nodules"][1]["Lobe"], "left lower lobe")
        validation = validate_dense_prediction(obj, raw_text=text)
        self.assertTrue(validation["valid"], validation)

    def test_extra_token_after_template_is_rejected(self):
        tokens = self.runtime.render_tokens(0)
        with self.assertRaises(TemplateStateError):
            self.machine.replay(tokens + self.runtime.encode("x"))

    def test_one_terminal_eos_after_template_is_idempotent(self):
        """Match vLLM V0's one-step-ahead async output call sequence."""
        tokens = self.runtime.render_tokens(0)
        replay = self.machine.replay(tokens + [self.runtime.eos_token_id])
        self.assertTrue(replay.program_complete)
        self.assertEqual(replay.action.kind, "eos")
        self.assertEqual(replay.action.token_id, self.runtime.eos_token_id)

    def test_more_than_one_terminal_eos_after_template_is_rejected(self):
        tokens = self.runtime.render_tokens(0)
        with self.assertRaises(TemplateStateError):
            self.machine.replay(
                tokens
                + [self.runtime.eos_token_id, self.runtime.eos_token_id]
            )

    def test_typed_flattened_count_rejects_out_of_range_composition(self):
        runtime = make_runtime(legacy_compat=True, max_dynamic_nodules=None)
        machine = TemplateStateMachine(runtime)
        slot, terminator = find_slot(runtime, "<|num_nodule|>")
        self.assertEqual(slot.candidates.mode, "typed_flattened")
        tokens = runtime.encode("4949", add_special_tokens=False)
        with self.assertRaisesRegex(TemplateStateError, "Invalid typed prefix"):
            machine._slot_progress(slot, terminator, tokens + [terminator])

    def test_typed_flattened_null_forces_terminator(self):
        runtime = make_runtime(legacy_compat=True)
        machine = TemplateStateMachine(runtime)
        slot, terminator = find_slot(runtime, "<|follow_up_date|>")
        tokens = runtime.encode("null", add_special_tokens=False)
        _, complete, action, _ = machine._slot_progress(slot, terminator, tokens)
        self.assertFalse(complete)
        self.assertEqual(action.kind, "fixed")
        self.assertEqual(action.token_id, terminator)
        with self.assertRaisesRegex(TemplateStateError, "Invalid typed prefix"):
            machine._slot_progress(
                slot,
                terminator,
                tokens + runtime.encode("3", add_special_tokens=False),
            )

    def test_typed_flattened_allows_one_tokenizer_boundary_prefix(self):
        runtime = DynamicTemplateRuntime(
            InvisibleBoundaryTokenizer(),
            template_style="canonical",
            include_json_tags=False,
            legacy_compat=True,
            max_dynamic_nodules=49,
            original_vocab_size=1000,
        )
        machine = TemplateStateMachine(runtime)
        tokens = runtime.render_tokens(0)
        replay = machine.replay(tokens)
        self.assertTrue(replay.program_complete)
        self.assertEqual(replay.action.kind, "eos")

        count_slot, terminator = find_slot(runtime, "<|num_nodule|>")
        boundary = runtime.tokenizer.boundary_token_id
        with self.assertRaises(TemplateStateError):
            machine._slot_progress(
                count_slot,
                terminator,
                [boundary, boundary],
            )

    def test_typed_flattened_accepts_composed_decimal_and_date(self):
        runtime = make_runtime(legacy_compat=True)
        machine = TemplateStateMachine(runtime)
        for marker, value in (
            ("<|long_axis|>", "12.5"),
            ("<|comparison_date|>", "3/10/2022"),
            ("<|comparison_date|>", "7/7/22"),
            ("<|follow_up_date|>", "3/2022"),
        ):
            slot, terminator = find_slot(runtime, marker)
            tokens = runtime.encode(value, add_special_tokens=False)
            _, complete, _, chosen = machine._slot_progress(
                slot, terminator, tokens + [terminator]
            )
            self.assertTrue(complete)
            self.assertEqual(chosen, value)

    def test_composed_id_is_not_limited_by_seed_list_maximum(self):
        runtime = make_runtime(legacy_compat=True)
        machine = TemplateStateMachine(runtime)
        for marker, value in (
            ("<|nodule_id|>", "151"),
            ("<|series|>", "2022"),
            ("<|image|>", "4573"),
        ):
            slot, terminator = find_slot(runtime, marker)
            tokens = runtime.encode(value, add_special_tokens=False)
            _, complete, _, chosen = machine._slot_progress(
                slot, terminator, tokens + [terminator]
            )
            self.assertTrue(complete)
            self.assertEqual(chosen, value)

    def test_malformed_composed_values_are_rejected(self):
        runtime = make_runtime(legacy_compat=True)
        machine = TemplateStateMachine(runtime)
        for marker, value in (
            ("<|nodule_id|>", "3 40"),
            ("<|comparison_date|>", "202205/20/12"),
            ("<|follow_up_date|>", "2022/3/10"),
        ):
            slot, terminator = find_slot(runtime, marker)
            tokens = runtime.encode(value, add_special_tokens=False)
            with self.assertRaises(TemplateStateError):
                machine._slot_progress(slot, terminator, tokens + [terminator])

    def test_semantic_validation_rejects_flattened_garbage(self):
        obj = json.loads(self.runtime.decode(self.runtime.render_tokens(0)))
        obj["Follow-up Date"] = "null333"
        validation = validate_dense_prediction(obj)
        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("invalid Follow-up Date" in error for error in validation["errors"]),
            validation,
        )

    def test_top_p_is_applied_inside_legal_token_set(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is provided by the vLLM runtime environment")

        processor = object.__new__(DynamicTemplateLogitsProcessor)
        processor.legacy_top_p = 0.1
        processor.legacy_top_k = -1
        processor.legacy_min_p = 0.0
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
        # Token 4 dominates globally, but it is illegal. The best legal token
        # is 1 and must not be replaced by the first allowed ID.
        self.assertEqual(processor._select_allowed(logits, [0, 1]), 1)

    def test_explicit_count_cap_is_not_silent(self):
        with self.assertRaisesRegex(TemplateStateError, "explicit dynamic template cap"):
            self.runtime.get_program(50)

    def test_historical_compiler_special_tokens_are_program_tokens(self):
        runtime = make_special_token_runtime()
        tokens = runtime.render_tokens(0)
        replay = TemplateStateMachine(runtime).replay(tokens)
        self.assertTrue(replay.program_complete)
        self.assertEqual(replay.action.kind, "eos")
        self.assertTrue(runtime.compiled_program_ends_with_eos(0))
        parsed = json.loads(runtime.decode_output(tokens))
        self.assertEqual(parsed["num_nodule"], "0")


if __name__ == "__main__":
    unittest.main()
