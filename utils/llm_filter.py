"""
Simple LLM Filter for Scene Graph Enumerations

Uses local Flask API to filter implausible GroupNode enumerations and room combinations.
"""

import json
import logging
import re
from typing import List, Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.meta_func.flask_func import FlaskFunc
from utils.model_client import ModelOutputTruncatedError, ModelRequestError


LOG = logging.getLogger(__name__)


class LLMFilter(FlaskFunc):
    """
    Simple LLM filter using local Flask API for filtering scene graph enumerations.

    This class uses prompt templates to ask the LLM whether object combinations
    are plausible, then filters out implausible results.
    """

    BATCH_SIZE = 32

    def __init__(self, verbose=False):
        """
        Initialize LLM filter.

        Args:
            verbose: Whether to print debug information
        """
        super().__init__()
        self.verbose = verbose

        # Prompt templates
        self.group_filter_template = """You are evaluating whether object groups in indoor scenes are plausible.

Given a room type and a list of objects found together in a group, determine if this combination is reasonable.

Room: {room}
Objects in group: {objects}

Is this a plausible combination? Answer in JSON format:
{{"plausible": true/false, "reason": "brief explanation"}}

Answer:"""

        self.room_filter_template = """You are evaluating whether a room configuration is plausible.

Given a room type and multiple object groups within that room, determine if the overall configuration is reasonable.

Room: {room}
Object groups:
{groups}

Is this a plausible room configuration? Answer in JSON format:
{{"plausible": true/false, "reason": "brief explanation"}}

Answer:"""

        # Statistics
        self.stats = {
            'group_filtered': 0,
            'group_kept': 0,
            'room_filtered': 0,
            'room_kept': 0,
        }

    def filter_enumerated_groups(self, enumerated_groups, min_confidence=0.5):
        """
        Filter implausible GroupNode enumerations (BATCH MODE).

        Args:
            enumerated_groups: EnumeratedGroupNodes instance
            min_confidence: Not used (for API compatibility), kept for consistency

        Returns:
            filtered_enumerated_groups: Filtered EnumeratedGroupNodes instance
        """
        if len(enumerated_groups) == 0:
            return enumerated_groups

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[LLM Filter] GroupNode Enumerations - BEFORE Filtering")
            print(f"{'='*70}")
            print(f"Total: {len(enumerated_groups)} enumerations\n")

            # Show all enumerations before filtering
            for idx, enum_data in enumerate(enumerated_groups.all_enumerations, 1):
                room = enum_data['room_caption']
                objects = enum_data['caption_combo']
                prob = enum_data['probability']
                group_idx = enum_data['group_idx']
                num_objects = len(objects)

                # Show which GroupNode this belongs to
                print(f"{idx:3d}. [{room:12s}] Group#{group_idx} ({num_objects} objs) {objects} (p={prob:.4f})")

        original_count = len(enumerated_groups.all_enumerations)

        # Separate items: 100% certain vs uncertain
        certain_indices = set()  # probability = 1.0, skip LLM
        uncertain_enums = []     # need LLM filtering
        uncertain_indices_map = {}  # maps uncertain enum index to original index

        for idx, enum_data in enumerate(enumerated_groups.all_enumerations):
            if enum_data['probability'] >= 0.9999:  # 100% certain (use 0.9999 to handle float precision)
                certain_indices.add(idx)
            else:
                uncertain_indices_map[len(uncertain_enums)] = idx
                uncertain_enums.append(enum_data)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[LLM Filter] Filtering Strategy")
            print(f"{'='*70}")
            print(f"Total: {original_count} enumerations")
            print(f"  - 100% certain (skip LLM): {len(certain_indices)}")
            print(f"  - Uncertain (send to LLM): {len(uncertain_enums)}")

        # If no uncertain items, keep everything
        if len(uncertain_enums) == 0:
            if self.verbose:
                print(f"\n✓ All enumerations are 100% certain, no LLM filtering needed")
            self.stats['group_kept'] += original_count
            return enumerated_groups

        plausible_uncertain_indices = self._filter_batches(
            uncertain_enums, self._build_batch_group_filter_prompt)

        # Map back to original indices
        plausible_indices = certain_indices.copy()  # All certain items are plausible
        for uncertain_idx in plausible_uncertain_indices:
            original_idx = uncertain_indices_map[uncertain_idx]
            plausible_indices.add(original_idx)

        if self.verbose:
            print(f"{'='*70}")
            print(f"[LLM Filter] Filtering Results")
            print(f"{'='*70}")

        # Filter based on LLM response
        kept = []
        filtered_out = []

        for idx, enum_data in enumerate(enumerated_groups.all_enumerations):
            if idx in plausible_indices:
                kept.append(enum_data)
                self.stats['group_kept'] += 1
            else:
                filtered_out.append((idx, enum_data))
                self.stats['group_filtered'] += 1

        # Print filtered items
        if self.verbose:
            print(f"\n❌ FILTERED OUT ({len(filtered_out)} items):")
            for idx, enum_data in filtered_out:
                room = enum_data['room_caption']
                objects = enum_data['caption_combo']
                prob = enum_data['probability']
                group_idx = enum_data['group_idx']
                num_objects = len(objects)
                print(f"  {idx+1:3d}. [{room:12s}] Group#{group_idx} ({num_objects} objs) {objects} (p={prob:.4f})")

        # Update enumerated_groups
        enumerated_groups.all_enumerations = kept
        enumerated_groups._sort()

        # Rebuild indices
        enumerated_groups.by_room = {}
        enumerated_groups.by_group = {}

        for enum_data in kept:
            room_caption = enum_data['room_caption']
            group_key = (room_caption, enum_data['group_idx'])

            if room_caption not in enumerated_groups.by_room:
                enumerated_groups.by_room[room_caption] = []
            enumerated_groups.by_room[room_caption].append(enum_data)

            if group_key not in enumerated_groups.by_group:
                enumerated_groups.by_group[group_key] = []
            enumerated_groups.by_group[group_key].append(enum_data['group_node'])

        if self.verbose:
            print(f"  Result: Kept {len(kept)}/{original_count} ({len(kept)/original_count*100:.1f}%)")
            
        return enumerated_groups

    def filter_room_combinations(self, room_combinations, min_confidence=0.5):
        """
        Filter implausible room-level GroupNode combinations (BATCH MODE).

        Args:
            room_combinations: RoomGroupCombinations instance
            min_confidence: Not used (for API compatibility)

        Returns:
            filtered_room_combinations: Filtered RoomGroupCombinations instance
        """
        if len(room_combinations) == 0:
            return room_combinations

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[LLM Filter] Room Combinations - BEFORE Filtering")
            print(f"{'='*70}")
            print(f"Total: {len(room_combinations)} combinations\n")

            # Show all combinations before filtering
            for idx, combo_data in enumerate(room_combinations.all_combinations, 1):
                room = combo_data['room_caption']
                prob = combo_data['probability']
                num_groups = combo_data['num_groups']

                print(f"{idx:3d}. [{room:12s}] (p={prob:.6f}, {num_groups} groups)")
                for gidx, group in enumerate(combo_data['group_nodes'], 1):
                    captions = [n.caption for n in group.nodes]
                    print(f"     Group {gidx} ({len(captions)} objs): {captions}")

        original_count = len(room_combinations.all_combinations)

        # Separate items: 100% certain vs uncertain
        certain_indices = set()  # probability = 1.0, skip LLM
        uncertain_combos = []     # need LLM filtering
        uncertain_indices_map = {}  # maps uncertain combo index to original index

        for idx, combo_data in enumerate(room_combinations.all_combinations):
            if combo_data['probability'] >= 0.9999:  # 100% certain
                certain_indices.add(idx)
            else:
                uncertain_indices_map[len(uncertain_combos)] = idx
                uncertain_combos.append(combo_data)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[LLM Filter] Filtering Strategy")
            print(f"{'='*70}")
            print(f"Total: {original_count} combinations")
            print(f"  - 100% certain (skip LLM): {len(certain_indices)}")
            print(f"  - Uncertain (send to LLM): {len(uncertain_combos)}")

        # If no uncertain items, keep everything
        if len(uncertain_combos) == 0:
            if self.verbose:
                print(f"\n✓ All combinations are 100% certain, no LLM filtering needed")
            self.stats['room_kept'] += original_count
            return room_combinations

        plausible_uncertain_indices = self._filter_batches(
            uncertain_combos, self._build_batch_room_filter_prompt)

        # Map back to original indices
        plausible_indices = certain_indices.copy()  # All certain items are plausible
        for uncertain_idx in plausible_uncertain_indices:
            original_idx = uncertain_indices_map[uncertain_idx]
            plausible_indices.add(original_idx)

        if self.verbose:
            print(f"{'='*70}")
            print(f"[LLM Filter] Filtering Results")
            print(f"{'='*70}")

        # Filter based on LLM response
        kept = []
        filtered_out = []

        for idx, combo_data in enumerate(room_combinations.all_combinations):
            if idx in plausible_indices:
                kept.append(combo_data)
                self.stats['room_kept'] += 1
            else:
                filtered_out.append((idx, combo_data))
                self.stats['room_filtered'] += 1

        # Update room_combinations
        room_combinations.all_combinations = kept
        room_combinations._sort()

        # Rebuild index
        room_combinations.by_room = {}
        for combo_data in kept:
            room_caption = combo_data['room_caption']
            if room_caption not in room_combinations.by_room:
                room_combinations.by_room[room_caption] = []
            room_combinations.by_room[room_caption].append(combo_data)

        if self.verbose:
            print(f"  Result: Kept {len(kept)}/{original_count} ({len(kept)/original_count*100:.1f}%)")
        return room_combinations

    def _check_group_plausibility(self, room_caption: str, objects: List[str]) -> Tuple[bool, str]:
        """
        Check if a group of objects is plausible in the given room.

        Args:
            room_caption: Room type (e.g., 'bedroom')
            objects: List of object captions (e.g., ['bed', 'table', 'lamp'])

        Returns:
            (is_plausible, reason)
        """
        # Build prompt
        objects_str = ", ".join(objects)
        prompt = self.group_filter_template.format(
            room=room_caption,
            objects=objects_str
        )

        # Call LLM
        response = self.get_llm_response(prompt)

        # Parse response
        is_plausible, reason = self._parse_response(response)

        return is_plausible, reason

    def _check_room_combination_plausibility(self,
                                            room_caption: str,
                                            group_captions: List[List[str]]) -> Tuple[bool, str]:
        """
        Check if a room configuration with multiple groups is plausible.

        Args:
            room_caption: Room type
            group_captions: List of groups, each group is a list of object captions

        Returns:
            (is_plausible, reason)
        """
        # Build groups text
        groups_text = ""
        for idx, group in enumerate(group_captions, 1):
            objects_str = ", ".join(group)
            groups_text += f"Group {idx}: {objects_str}\n"

        # Build prompt
        prompt = self.room_filter_template.format(
            room=room_caption,
            groups=groups_text.strip()
        )

        # Call LLM
        response = self.get_llm_response(prompt)

        # Parse response
        is_plausible, reason = self._parse_response(response)

        return is_plausible, reason

    def _build_batch_group_filter_prompt(self, all_enumerations: List[Dict]) -> str:
        """
        Build batch prompt for filtering multiple GroupNode enumerations at once.

        Args:
            all_enumerations: List of enumeration dicts

        Returns:
            Prompt string for batch filtering
        """
        prompt = """You are evaluating whether object groups in indoor scenes are plausible.

Below is a list of object groups, each with a room type and objects found together.
For each group, determine if the combination is reasonable.

Return exactly one JSON object with key "plausible": a list of the unique 1-indexed indices of ONLY the PLAUSIBLE groups.
Do not repeat the inputs, explain decisions, use ellipses, or add other keys.

If all groups are plausible, include all indices. If none are plausible, set "plausible" to [].

Groups to evaluate:
"""

        for idx, enum_data in enumerate(all_enumerations, 1):
            room = enum_data['room_caption']
            objects = enum_data['caption_combo']
            objects_str = ", ".join(objects)
            prompt += f"\n{idx}. Room: {room}, Objects: {objects_str}"

        prompt += "\n\nYour answer (JSON only):"
        return prompt

    def _build_batch_room_filter_prompt(self, all_combinations: List[Dict]) -> str:
        """
        Build batch prompt for filtering multiple room combinations at once.

        Args:
            all_combinations: List of combination dicts

        Returns:
            Prompt string for batch filtering
        """
        prompt = """You are evaluating whether room configurations are plausible.

Below is a list of room configurations, each containing multiple object groups.
For each configuration, determine if the overall setup is reasonable for that room type.

Return exactly one JSON object with key "plausible": a list of the unique 1-indexed indices of ONLY the PLAUSIBLE configurations.
Do not repeat the inputs, explain decisions, use ellipses, or add other keys.

If all configurations are plausible, include all indices. If none are plausible, set "plausible" to [].

Configurations to evaluate:
"""

        for idx, combo_data in enumerate(all_combinations, 1):
            room = combo_data['room_caption']
            prompt += f"\n{idx}. Room: {room}"

            # Add groups
            for gidx, group in enumerate(combo_data['group_nodes'], 1):
                captions = [n.caption for n in group.nodes]
                objects_str = ", ".join(captions)
                prompt += f"\n   Group {gidx}: {objects_str}"

        prompt += "\n\nYour answer (JSON only):"
        return prompt

    def _filter_batches(self, items, build_prompt):
        """Bound replies while retaining every candidate and its original index."""
        plausible = set()
        for offset in range(0, len(items), self.BATCH_SIZE):
            batch = items[offset:offset + self.BATCH_SIZE]
            plausible.update(offset + idx for idx in self._filter_batch(batch, build_prompt))
        return plausible

    def _filter_batch(self, items, build_prompt):
        count = len(items)
        prompt = build_prompt(items)
        plausible = {
            'type': 'array', 'maxItems': count,
            'items': {'type': 'integer', 'minimum': 1, 'maximum': count},
        }
        # Parasail's grammar rejects uniqueItems. The prompt and parser still require distinct indices.
        if not (self.model_client.backend == 'openrouter'
                and self.model_client.model == 'qwen/qwen3-vl-8b-instruct'):
            plausible['uniqueItems'] = True
        format_spec = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'plausibility_filter', 'strict': True,
                'schema': {
                    'type': 'object', 'additionalProperties': False,
                    'required': ['plausible'],
                    'properties': {'plausible': plausible},
                },
            },
        }
        for attempt in range(3):
            try:
                response = self.model_client.complete(
                    prompt, response_format=format_spec,
                    validator=lambda value: self._parse_batch_filter_response(value, count))
                return self._parse_batch_filter_response(response, count)
            except ModelOutputTruncatedError:
                if count > 1:
                    middle = count // 2
                    LOG.warning('LLM filter reply truncated; splitting %d items into %d and %d',
                                count, middle, count - middle)
                    left = self._filter_batch(items[:middle], build_prompt)
                    right = self._filter_batch(items[middle:], build_prompt)
                    return left | {middle + idx for idx in right}
                error = 'single-item reply truncated'
            except ValueError as exc:
                error = str(exc)
            LOG.warning('Invalid LLM filter reply for %d items (attempt %d/3): %s',
                        count, attempt + 1, error)
            prompt = (build_prompt(items) + '\nReturn only a JSON object whose sole key is '
                      f'"plausible": a list of unique integer indices from 1 to {count}; use [] if none. '
                      'No ellipses or explanations.')
        raise ModelRequestError(f'Invalid LLM filter JSON for {count} items after 3 format attempts: {error}')

    def _parse_batch_filter_response(self, response: str, total_count: int) -> set:
        """Accept only complete, unambiguous decisions; never infer from prose."""
        text = response.strip()
        if text.startswith('```') and text.endswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)[:-3].strip()
        data = json.loads(text)
        # Qwen may wrap a structured reply in a single-element list.
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        if not isinstance(data, dict) or set(data) != {'plausible'}:
            raise ValueError('expected one object containing only plausible')
        indices = data['plausible']
        if (not isinstance(indices, list)
                or any(type(idx) is not int or not 1 <= idx <= total_count for idx in indices)):
            raise ValueError('plausible must contain integer indices within the batch')
        if len(set(indices)) != len(indices):
            raise ValueError('duplicate plausible indices')
        return {idx - 1 for idx in indices}

    def _parse_response(self, response: str) -> Tuple[bool, str]:
        """
        Parse LLM response to extract plausibility judgment.

        Expected format:
        {"plausible": true/false, "reason": "explanation"}

        Args:
            response: Raw LLM response

        Returns:
            (is_plausible, reason)
        """
        # Try JSON parsing first
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{[^}]*"plausible"[^}]*\}', response, re.IGNORECASE | re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                is_plausible = data.get('plausible', True)
                reason = data.get('reason', '')
                return (is_plausible, reason)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: simple text parsing
        response_lower = response.lower()

        # Check for negative keywords
        if any(word in response_lower for word in ['implausible', 'not plausible', 'unreasonable', 'unlikely', 'false']):
            # Try to extract reason
            if ':' in response:
                reason = response.split(':', 1)[1].strip()
            else:
                reason = response.strip()
            return (False, reason)

        # Default: assume plausible
        return (True, "")

    def get_stats(self) -> Dict:
        """Get filtering statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset filtering statistics."""
        self.stats = {
            'group_filtered': 0,
            'group_kept': 0,
            'room_filtered': 0,
            'room_kept': 0,
        }

    def print_stats(self):
        """Print filtering statistics."""
        print("\n" + "="*60)
        print("LLM Filter Statistics")
        print("="*60)
        print(f"GroupNode Enumerations:")
        print(f"  Kept: {self.stats['group_kept']}")
        print(f"  Filtered: {self.stats['group_filtered']}")

        total_groups = self.stats['group_kept'] + self.stats['group_filtered']
        if total_groups > 0:
            print(f"  Filter rate: {self.stats['group_filtered']/total_groups*100:.1f}%")

        print(f"\nRoom Combinations:")
        print(f"  Kept: {self.stats['room_kept']}")
        print(f"  Filtered: {self.stats['room_filtered']}")

        total_rooms = self.stats['room_kept'] + self.stats['room_filtered']
        if total_rooms > 0:
            print(f"  Filter rate: {self.stats['room_filtered']/total_rooms*100:.1f}%")

        print("="*60)
