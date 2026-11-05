# Traceability: MATPOWER reader/writer tests

Spec: `specs/parser_matpower.md` (approved 2026-10-05, by Amir; amendment 1 C1–C6)
Tests: `tests_spec/test_parser_matpower.py`, with `parser_helpers.py`, `conftest.py` and
`data/*.m`

**Size.** 55 test functions, 158 collected cases after parametrization. That includes 2
Hypothesis properties with 60 derandomized examples each.

**Requirements.** `pytest` and `hypothesis`. Run from the repo root:
`python -m pytest tests_spec/test_parser_matpower.py`. It needs pytest's default import mode
(`prepend`) so the tests can `import parser_helpers`.

**Self-check (QA-internal, not delivered).** The suite was run against a throwaway stub
written from the spec only, never from product code: 158/158 passed, and it passed again on
a rerun (deterministic). Then 17 deliberate defects were injected into the stub, one at a
time. Every one made at least one test fail:

- low float precision;
- `inf` spelled lowercase;
- a short row skipped silently;
- `None` padded to `0`;
- CRLF output;
- integer columns read as float;
- a duplicate `function` line;
- `mpc.` prefix kept in keys;
- comments dropped;
- a shared cache between calls;
- the writer mutating its input;
- a missing `gen` treated as an error;
- CRLF not normalized;
- a message without the section name;
- version 1 accepted;
- `NaN` written as `0`;
- `extra` truncated.

## Coverage per spec item

| Spec ID | Tests |
|---|---|
| SPEC-PARSER-01 | `test_core_scalars_are_read`, `test_casename_comes_from_function_line_not_file_name`, `test_missing_version_defaults_to_2` |
| SPEC-PARSER-02 | `test_bus_13_columns_map_in_order_with_int_columns_and_no_results`, `test_bus_result_columns_fill_in_order_rest_none`, `test_bus_17_columns_fixture_maps_all_results` |
| SPEC-PARSER-03 | `test_gen_10_columns_has_empty_extra`, `test_gen_21_columns_keeps_all_11_extra_columns_in_order`, `test_gen_wider_than_21_columns_keeps_every_extra_column` |
| SPEC-PARSER-04 | `test_branch_13_columns_maps_in_order`, `test_branch_extra_columns_kept_in_order` |
| SPEC-PARSER-05 | `test_gencost_polynomial_row`, `test_gencost_4_columns_gives_empty_coeffs`, `test_gencost_rows_read_as_numbers_without_ncost_check` |
| SPEC-PARSER-06 | `test_order_ids_kept_and_no_semantic_checks` |
| SPEC-PARSER-07 | `test_present_but_empty_section_gives_empty_list`, `test_absent_gen_branch_gencost_give_empty_lists`, `test_absent_gencost_alone_gives_empty_list` |
| SPEC-PARSER-08 | `test_extra_section_keys_without_prefix_and_no_core_keys`, `test_extra_sections_stable_from_second_round_trip`, `test_extra_section_elements_survive_in_written_text` |
| SPEC-PARSER-09 | `test_float_fidelity_on_wide_case`, `test_inf_nan_any_case_read_round_trip_and_written_canonically` |
| SPEC-PARSER-10 | `test_syntax_variants_parse_like_plain_case`, `test_integral_floats_in_integer_columns_read_as_int`, `test_scientific_and_negative_numbers`, `test_crlf_file_reads_exactly_like_lf_file` |
| SPEC-PARSER-11 | `test_round_trip_data_equivalent_then_fully_stable`, `test_source_file_bytes_unchanged` |
| SPEC-PARSER-12 | `test_written_file_meets_textual_rules`, `test_cell_arrays_closed_at_first_terminator_with_all_elements`, `test_crlf_input_is_written_with_lf_only`, `test_empty_header_writes_function_line_first`, `test_casename_wins_over_function_name_in_header`, `test_casename_change_is_written` |
| SPEC-PARSER-13 | `test_writer_is_idempotent`, `test_file_does_not_grow_over_repeated_round_trips`, `test_mixed_widths_writer_idempotent` |
| SPEC-PARSER-14 | `test_comment_above_section_stays_with_section`, `test_bus_comment_follows_basemva_line`, `test_header_comments_preserved` |
| SPEC-PARSER-15 | `test_str_and_path_give_same_result`, `test_writer_creates_missing_parent_directories`, `test_writer_overwrites_existing_file` |
| SPEC-PARSER-16 | `test_missing_file_raises_file_not_found` |
| SPEC-PARSER-17 | `test_malformed_input_raises_value_error_naming_section`, `test_empty_file_raises_value_error` |
| SPEC-PARSER-18 | `test_malformed_input_raises_value_error_naming_section` |
| SPEC-PARSER-19 | `test_opaque_extra_sections_are_not_validated` |
| SPEC-PARSER-20 | `test_reading_is_deterministic` |
| SPEC-PARSER-21 | `test_mutating_result_does_not_affect_next_parse`, `test_writer_does_not_mutate_its_input` |
| SPEC-PARSER-22 | `test_property_text_strategy_round_trip`, `test_property_constructed_network_round_trip` |
| SPEC-PARSER-23 | `test_mixed_widths_read_per_row`, `test_mixed_widths_round_trip_without_padding`, `test_mixed_widths_writer_idempotent` |

## Test → spec ID

| Test | Spec ID(s) |
|---|---|
| `test_core_scalars_are_read` | SPEC-PARSER-01 |
| `test_casename_comes_from_function_line_not_file_name` | SPEC-PARSER-01 |
| `test_missing_version_defaults_to_2` | SPEC-PARSER-01 |
| `test_bus_13_columns_map_in_order_with_int_columns_and_no_results` | SPEC-PARSER-02 |
| `test_bus_result_columns_fill_in_order_rest_none` | SPEC-PARSER-02 |
| `test_bus_17_columns_fixture_maps_all_results` | SPEC-PARSER-02 |
| `test_gen_10_columns_has_empty_extra` | SPEC-PARSER-03 |
| `test_gen_21_columns_keeps_all_11_extra_columns_in_order` | SPEC-PARSER-03 |
| `test_gen_wider_than_21_columns_keeps_every_extra_column` | SPEC-PARSER-03 |
| `test_branch_13_columns_maps_in_order` | SPEC-PARSER-04 |
| `test_branch_extra_columns_kept_in_order` | SPEC-PARSER-04 |
| `test_gencost_polynomial_row` | SPEC-PARSER-05 |
| `test_gencost_4_columns_gives_empty_coeffs` | SPEC-PARSER-05 |
| `test_gencost_rows_read_as_numbers_without_ncost_check` | SPEC-PARSER-05 |
| `test_order_ids_kept_and_no_semantic_checks` | SPEC-PARSER-06 |
| `test_present_but_empty_section_gives_empty_list` | SPEC-PARSER-07 |
| `test_absent_gen_branch_gencost_give_empty_lists` | SPEC-PARSER-07 |
| `test_absent_gencost_alone_gives_empty_list` | SPEC-PARSER-07 |
| `test_extra_section_keys_without_prefix_and_no_core_keys` | SPEC-PARSER-08 |
| `test_extra_sections_stable_from_second_round_trip` | SPEC-PARSER-08 |
| `test_extra_section_elements_survive_in_written_text` | SPEC-PARSER-08 |
| `test_float_fidelity_on_wide_case` | SPEC-PARSER-09 |
| `test_inf_nan_any_case_read_round_trip_and_written_canonically` | SPEC-PARSER-09 |
| `test_syntax_variants_parse_like_plain_case` | SPEC-PARSER-10 |
| `test_integral_floats_in_integer_columns_read_as_int` | SPEC-PARSER-10 |
| `test_scientific_and_negative_numbers` | SPEC-PARSER-10 |
| `test_crlf_file_reads_exactly_like_lf_file` | SPEC-PARSER-10 |
| `test_round_trip_data_equivalent_then_fully_stable` | SPEC-PARSER-11 |
| `test_source_file_bytes_unchanged` | SPEC-PARSER-11 |
| `test_written_file_meets_textual_rules` | SPEC-PARSER-12 |
| `test_cell_arrays_closed_at_first_terminator_with_all_elements` | SPEC-PARSER-12 |
| `test_crlf_input_is_written_with_lf_only` | SPEC-PARSER-12 |
| `test_empty_header_writes_function_line_first` | SPEC-PARSER-12 |
| `test_casename_wins_over_function_name_in_header` | SPEC-PARSER-12 |
| `test_casename_change_is_written` | SPEC-PARSER-12 |
| `test_writer_is_idempotent` | SPEC-PARSER-13 |
| `test_file_does_not_grow_over_repeated_round_trips` | SPEC-PARSER-13 |
| `test_comment_above_section_stays_with_section` | SPEC-PARSER-14 |
| `test_bus_comment_follows_basemva_line` | SPEC-PARSER-14 |
| `test_header_comments_preserved` | SPEC-PARSER-14 |
| `test_str_and_path_give_same_result` | SPEC-PARSER-15 |
| `test_writer_creates_missing_parent_directories` | SPEC-PARSER-15 |
| `test_writer_overwrites_existing_file` | SPEC-PARSER-15 |
| `test_missing_file_raises_file_not_found` | SPEC-PARSER-16 |
| `test_malformed_input_raises_value_error_naming_section` | SPEC-PARSER-17, SPEC-PARSER-18 |
| `test_empty_file_raises_value_error` | SPEC-PARSER-17 |
| `test_opaque_extra_sections_are_not_validated` | SPEC-PARSER-19 |
| `test_reading_is_deterministic` | SPEC-PARSER-20 |
| `test_mutating_result_does_not_affect_next_parse` | SPEC-PARSER-21 |
| `test_writer_does_not_mutate_its_input` | SPEC-PARSER-21 |
| `test_property_text_strategy_round_trip` | SPEC-PARSER-22 |
| `test_property_constructed_network_round_trip` | SPEC-PARSER-22 |
| `test_mixed_widths_read_per_row` | SPEC-PARSER-23 |
| `test_mixed_widths_round_trip_without_padding` | SPEC-PARSER-23 |
| `test_mixed_widths_writer_idempotent` | SPEC-PARSER-23, SPEC-PARSER-13 |

## Spec clarifications

Interpretations C1–C6 were accepted by Amir on 2026-10-05 and recorded as amendment 1 of
`specs/parser_matpower.md`. No spec gaps remain open for this area.

## Not tested (out of scope per spec)

- Semantic validation.
- `}`/`};` inside cell strings.
- Comments above `mpc.version`/`mpc.baseMVA`.
- The written layout.
- Row numbers in messages.
- `.gic` files.
- ExaGO execution.
