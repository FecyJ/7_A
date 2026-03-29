"""自定义 Footer，统一快捷键显示顺序"""

from __future__ import annotations

from collections import defaultdict
from itertools import groupby

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import Footer
from textual.widgets._footer import FooterKey, FooterLabel, KeyGroup


class AgentFooter(Footer):
    """固定关键快捷键在 Footer 中的显示顺序"""

    KEY_ORDER = {
        "ctrl+n": 0,
        "ctrl+b": 1,
        "ctrl+j": 2,
        "ctrl+c": 3,
        "ctrl+d": 4,
    }

    def _sorted_visible_bindings(self) -> list[tuple[Binding, bool, str]]:
        active_bindings = self.screen.active_bindings
        bindings = [
            (binding, enabled, tooltip)
            for (_, binding, enabled, tooltip) in active_bindings.values()
            if binding.show
        ]

        indexed_bindings = list(enumerate(bindings))
        indexed_bindings.sort(
            key=lambda item: (
                0 if item[1][0].key in self.KEY_ORDER else 1,
                self.KEY_ORDER.get(item[1][0].key, 0),
                item[0],
            )
        )
        return [binding_info for _, binding_info in indexed_bindings]

    def compose(self) -> ComposeResult:
        if not self._bindings_ready:
            return

        bindings = self._sorted_visible_bindings()

        action_to_bindings: defaultdict[str, list[tuple[Binding, bool, str]]]
        action_to_bindings = defaultdict(list)
        for binding, enabled, tooltip in bindings:
            action_to_bindings[binding.action].append((binding, enabled, tooltip))

        self.styles.grid_size_columns = len(action_to_bindings)

        for group, multi_bindings_iterable in groupby(
            action_to_bindings.values(),
            lambda multi_bindings_: multi_bindings_[0][0].group,
        ):
            multi_bindings = list(multi_bindings_iterable)
            if group is not None and len(multi_bindings) > 1:
                with KeyGroup(classes="-compact" if group.compact else ""):
                    for binding_group in multi_bindings:
                        binding, enabled, tooltip = binding_group[0]
                        yield FooterKey(
                            binding.key,
                            self.app.get_key_display(binding),
                            "",
                            binding.action,
                            disabled=not enabled,
                            tooltip=tooltip or binding.description,
                            classes="-grouped",
                        ).data_bind(compact=Footer.compact)
                yield FooterLabel(group.description)
            else:
                for binding_group in multi_bindings:
                    binding, enabled, tooltip = binding_group[0]
                    yield FooterKey(
                        binding.key,
                        self.app.get_key_display(binding),
                        binding.description,
                        binding.action,
                        disabled=not enabled,
                        tooltip=tooltip,
                    ).data_bind(compact=Footer.compact)

        if self.show_command_palette and self.app.ENABLE_COMMAND_PALETTE:
            active_bindings = self.screen.active_bindings
            try:
                _node, binding, enabled, _tooltip = active_bindings[
                    self.app.COMMAND_PALETTE_BINDING
                ]
            except KeyError:
                pass
            else:
                yield FooterKey(
                    binding.key,
                    self.app.get_key_display(binding),
                    binding.description,
                    binding.action,
                    classes="-command-palette",
                    disabled=not enabled,
                    tooltip=binding.tooltip or binding.description,
                )
