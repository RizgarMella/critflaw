"""Test-linked oracle evidence for the provenance records (theory manual, section 8.3.1): for each record, the
oracle kinds (theory manual index, section 4) that named tests actually carry. A kind claimed in provenance
but absent here does not count toward a record's status. Test ids are "tests/<file>.py::<function>"; a test
checks that each exists."""

_FAD = "tests/test_fad.py::"
_OPT2 = "tests/test_option2.py::"
_TGH = "tests/test_toughness.py::"
_PLATE = "tests/test_plate_and_assessment.py::"
_P1 = "tests/oracles/test_partner_reference_phase1.py::"
_P2R = "tests/test_prob_edge_cases.py::"
_P2P2 = "tests/test_prob_advanced.py::"
_P3B = "tests/test_cylinders_embedded.py::"
_P3R = "tests/test_fatigue_edge_cases.py::"
_FAT = "tests/test_fatigue_growth.py::"

_P1_HAND = _P1 + "test_partner_hand_cases_agree_with_critflaw"

# "w": the source standard's own worked examples or printed tables reproduced — information only: they are not an
# acceptance basis, so they never count as kind a. No test currently compares
# a primary reference's own published values, so no record has a linked kind a.
ORACLE_TESTS: dict[str, dict[str, tuple[str, ...]]] = {
    "fad.cutoff": {
        "c": (_FAD + "test_cutoff",),
        "d": (_P1 + "test_curve", _P1_HAND),
    },
    "fad.option1.continuous": {
        "c": (
            _FAD + "test_continuous_curve_is_continuous_at_one",
            _FAD + "test_regression_distinguishes_this_edition_from_older_fixed_coefficient_curve",
            _FAD + "test_material_dependence_of_mu_cap",
            _FAD + "test_power_law_branch",
        ),
        "d": (_P1 + "test_curve", _P1_HAND),
    },
    "fad.option1.discontinuous": {
        "c": (_FAD + "test_discontinuous_step",),
        "d": (_P1 + "test_curve", _P1_HAND),
    },
    "material.luders_strain": {
        "c": (_FAD + "test_discontinuous_step",),
        "d": (_P1 + "test_curve",),
    },
    "fad.kr": {
        "c": (
            _FAD + "test_kr_decision_table",
            _PLATE + "test_kp_zero_with_primary_load_takes_the_limiting_interaction",
        ),
        "d": (
            _P1 + "test_fracture_ratio",
            _P1 + "test_addendum_fracture_ratio_rule",
            _P1 + "test_addendum_hand_cases",
            _P1_HAND,
        ),
    },
    "fad.rho.simplified": {
        "c": (_FAD + "test_rho_breakpoints",),
        "d": (_P1 + "test_interaction", _P1_HAND),
    },
    "fad.v.simplified": {
        "c": (_FAD + "test_v_breakpoints",),
        "d": (_P1 + "test_interaction", _P1_HAND),
    },
    "fad.kjs.plastic_zone": {
        "c": (_FAD + "test_kr_decision_table", _PLATE + "test_plane_strain_option_reaches_the_engine"),
        "d": (_P1 + "test_interaction", _P1_HAND),
    },
    "residual.as_welded": {
        "c": (
            _PLATE + "test_residual_stress_and_relaxation",
            _PLATE + "test_room_temperature_yield_is_not_used_in_the_relaxed_term",
        ),
        "d": (_P1 + "test_addendum_residual_stress_through_the_engine",),
    },
    "residual.pwht": {
        "c": (_PLATE + "test_residual_stress_and_relaxation", _PLATE + "test_residual_stress_options_reach_the_engine"),
    },
    "k.general": {
        "c": (
            _PLATE + "test_secondary_stress_has_no_width_factor",
            _PLATE + "test_through_flaw_with_bending_and_secondary_by_hand",
        ),
        "d": (_P1 + "test_through_flaw_and_general_form", _P1_HAND),
    },
    "k.plate.surface": {
        "b": (
            _PLATE + "test_shallow_long_flaw_tends_to_edge_crack_value",
            _PLATE + "test_semicircular_shallow_flaw_matches_penny_shape_factor",
            _PLATE + "test_surface_flaw_pins_depth_and_half_length_definitions",
            _PLATE + "test_finite_width_factors",
        ),
        "c": (_PLATE + "test_angle_simplifications", _PLATE + "test_aspect_ratio_branches_meet_at_a_equals_c"),
        "d": (
            _P1 + "test_ocr_error_is_detected",
            _P1 + "test_surface_flaw_solutions",
            _P1 + "test_full_surface_flaw_assessment",
            _P1_HAND,
            _FAT + "test_surface_flaw_ranges_against_the_partner_k_solution",
            _P3R + "test_partner_fixed_step_integrator_agrees_to_1e_8",
        ),
    },
    "k.plate.through": {
        "b": (_PLATE + "test_through_flaw_in_wide_plate_pins_the_half_length_definition",),
        "c": (_PLATE + "test_finite_width_factors", _PLATE + "test_through_flaw_assessment"),
        "d": (
            _P1 + "test_through_flaw_and_general_form",
            _P1_HAND,
            _P3R + "test_partner_fixed_step_integrator_agrees_to_1e_8",
        ),
    },
    "sref.plate.surface": {
        "b": (
            _PLATE + "test_reference_stress_limits",
            _PLATE + "test_reference_stress_tends_to_net_section_as_flaw_fills_a_narrow_plate",
        ),
        "c": (_P3B + "test_embedded_below_surface_deep_point_and_reference_stress_limits",),
        "d": (_P1 + "test_surface_flaw_solutions", _P1 + "test_full_surface_flaw_assessment", _P1_HAND),
    },
    "sref.plate.through": {
        "b": (_PLATE + "test_reference_stress_limits",),
        "c": (_PLATE + "test_through_flaw_assessment", _PLATE + "test_through_flaw_with_bending_and_secondary_by_hand"),
        "d": (_P1 + "test_through_flaw_and_general_form", _P1_HAND),
    },
    "fad.option2": {
        "b": (_OPT2 + "test_elastic_law_gives_the_elastic_line", _OPT2 + "test_zero_offset_law_does_not_overflow"),
        "c": (
            _OPT2 + "test_power_law_matches_the_source_expression",
            _OPT2 + "test_tabulated_law_reproduces_the_power_law_it_was_sampled_from",
            _OPT2 + "test_equivalent_engineering_and_true_plateau_records_agree",
        ),
    },
    "fad.option2.monotone": {
        "b": (
            _OPT2 + "test_luders_step_line_is_monotone_and_never_above_the_source_expression",
            _OPT2 + "test_validity_condition_rejects_a_law_still_elastic_beyond_root_two",
            _OPT2 + "test_running_maximum_from_knots_is_exact_and_the_line_is_monotone_and_conservative",
            _OPT2 + "test_partner_counter_example_is_rejected_by_the_validity_condition",
            _OPT2 + "test_tabulated_validity_fails_when_a_single_interior_knot_is_too_elastic",
        ),
    },
    "material.power_law": {
        "b": (_OPT2 + "test_one_curve_per_sample_keeps_the_offset_definition_of_yield",),
        "c": (_OPT2 + "test_power_law_matches_the_source_expression", _OPT2 + "test_power_law_details_pinned"),
    },
    "toughness.k_from_j": {
        "c": (_TGH + "test_k_from_j_and_ctod", _TGH + "test_other_charpy_relations"),
    },
    "toughness.k_from_ctod": {
        "c": (_TGH + "test_k_from_j_and_ctod",),
    },
    "toughness.subsize": {
        "b": (_TGH + "test_subsize_correction_and_limit",),
        "c": (_TGH + "test_subsize_correction_and_limit",),
    },
    "toughness.master_curve": {
        "b": (_TGH + "test_master_curve_median_and_crack_front_length",),
        "c": (
            _TGH + "test_master_curve_median_and_crack_front_length",
            _P2R + "test_master_curve_crack_front_length_in_the_engine",
        ),
    },
    "toughness.charpy_t0": {
        "w": (_TGH + "test_source_worked_example_for_the_charpy_correlations",),
        "c": (_TGH + "test_source_worked_example_for_the_charpy_correlations",),
    },
    "toughness.charpy_master_curve": {
        "w": (_TGH + "test_source_worked_example_for_the_charpy_correlations",),
        "c": (_TGH + "test_source_worked_example_for_the_charpy_correlations",),
    },
    "toughness.charpy_lower_bound": {
        "c": (_TGH + "test_other_charpy_relations",),
    },
    "toughness.charpy_upper_limit": {
        "w": (_TGH + "test_source_worked_example_for_the_charpy_correlations",),
    },
    "toughness.charpy_upper_shelf": {
        "c": (_TGH + "test_other_charpy_relations",),
    },
    "toughness.charpy_subsize": {
        "w": (_TGH + "test_other_charpy_relations",),
    },
    "toughness.mote": {
        "c": (_TGH + "test_minimum_of_three_equivalent",),
    },
    "toughness.lower_bound": {
        "w": (_TGH + "test_tolerance_factor_reproduces_the_source_table_except_one_entry",),
        "b": (_TGH + "test_lower_bound_toughness",),
    },
    "partial_factors.table": {
        "c": (_P2P2 + "test_every_cell_of_the_published_table", _P2P2 + "test_published_partial_factors_lookup"),
    },
    "k.plate.embedded": {
        "b": (
            _P3B + "test_embedded_infinite_body_limits_and_branch_continuity",
            _P3B + "test_embedded_fw_limits",
            _P3B + "test_embedded_mb_limits_and_coefficient_set_jumps",
            _P3B + "test_embedded_below_surface_deep_point_and_reference_stress_limits",
            _P3B + "test_embedded_validity_pins_the_nearer_ligament_and_the_limits",
        ),
        "c": (
            _P3B + "test_embedded_infinite_body_limits_and_branch_continuity",
            _P3B + "test_embedded_mb_limits_and_coefficient_set_jumps",
        ),
        "d": (
            _P3B + "test_embedded_mm_mb_fw_against_the_partner_oracle",
            _P3B + "test_partner_fixed_step_integrator_for_the_new_kinds_agrees_to_1e_8",
        ),
    },
    "sref.plate.embedded": {
        "b": (_P3B + "test_embedded_below_surface_deep_point_and_reference_stress_limits",),
        "c": (_P3B + "test_embedded_below_surface_deep_point_and_reference_stress_limits",),
        "d": (_P3B + "test_embedded_mm_mb_fw_against_the_partner_oracle",),
    },
    "sref.plate.embedded_extended": {
        "c": (_P3B + "test_pb_pure_bending_thick_wall_conversion_and_extended_embedded_stability",),
    },
    "k.plate.extended": {
        "b": (_P3B + "test_cylinder_shell_parameters_and_bulging_against_the_oracle",),
    },
    "k.cylinder.bulging_through": {},
    "k.cylinder.bulging_surface": {},
    "k.cylinder.tables.through": {
        "c": (_P3B + "test_cylinder_tables_match_the_second_copy_cell_by_cell",),
    },
    "k.cylinder.tables.axial_surface": {
        "b": (_P3B + "test_cylinder_tables_reproduce_the_plate_at_zero_depth",),
        "c": (_P3B + "test_cylinder_tables_match_the_second_copy_cell_by_cell",),
        "d": (
            _P3B + "test_partner_fixed_step_integrator_for_the_new_kinds_agrees_to_1e_8",
            _P3B + "test_partner_fixed_step_integrator_for_the_tabulated_extended_flaw_agrees_to_1e_8",
        ),
    },
    "k.cylinder.tables.axial_extended_external": {
        "b": (_P3B + "test_table_monotonicity_ordering_and_the_bulging_check",),
        "c": (_P3B + "test_cylinder_tables_match_the_second_copy_cell_by_cell",),
    },
    "k.cylinder.tables.circumferential_surface": {
        "b": (_P3B + "test_cylinder_tables_reproduce_the_plate_at_zero_depth",),
        "c": (_P3B + "test_cylinder_tables_match_the_second_copy_cell_by_cell",),
    },
    "k.cylinder.tables.circumferential_extended": {
        "c": (_P3B + "test_cylinder_tables_match_the_second_copy_cell_by_cell",),
    },
    "k.cylinder.extended_external": {
        "b": (_P3B + "test_cylinder_shell_parameters_and_bulging_against_the_oracle",),
    },
    "sref.cylinder.axial_thin": {
        "d": (_P3B + "test_cylinder_reference_stresses_against_the_oracle",),
    },
    "sref.cylinder.axial_pressure": {
        "d": (_P3B + "test_cylinder_reference_stresses_against_the_oracle",),
    },
    "sref.cylinder.axial_thick_bending": {
        "c": (_P3B + "test_pb_pure_bending_thick_wall_conversion_and_extended_embedded_stability",),
        "d": (_P3B + "test_cylinder_reference_stresses_against_the_oracle",),
    },
    "sref.cylinder.circumferential_thin": {
        "d": (
            _P3B + "test_cylinder_reference_stresses_against_the_oracle",
            _P3B + "test_pb_p10_2_both_branches_and_the_equality_case",
        ),
    },
    "sref.cylinder.circumferential_thick": {
        "c": (_P3B + "test_r3_half_angle_reading_is_the_conservative_one_of_each_pair",),
        "d": (_P3B + "test_circumferential_thick_limit_load_against_the_oracle",),
    },
    "fatigue.paris_law": {
        "w": (_FAT + "test_transition_by_intersection_reproduces_the_printed_points",),
        "c": (_FAT + "test_unit_conversion_gives_the_same_rate_from_both_unit_systems",),
        "d": (
            _FAT + "test_laws_against_the_partner_oracle",
            _FAT + "test_surface_flaw_growth_against_a_cycle_by_cycle_reference",
            _P3R + "test_partner_fixed_step_integrator_agrees_to_1e_8",
        ),
    },
    "fatigue.recommended_laws": {
        "c": (
            _FAT + "test_upper_bound_coefficients_follow_from_mean_and_scatter",
            _FAT + "test_transition_by_intersection_reproduces_the_printed_points",
        ),
    },
    "fatigue.thresholds": {
        "c": (
            _FAT + "test_law_construction_rules_and_thresholds",
            _P3R + "test_catalogue_defaults_and_through_flaw_detection",
            _P3R + "test_catalogue_shallow_cap_depth_boundary",
        ),
    },
}

UNLINKED: dict[str, tuple[str, ...]] = {  # claimed kinds with no test found, per record (the audit list)
    "residual.pwht": ("d",),
    "toughness.k_from_j": ("b",),
    "toughness.charpy_upper_limit": ("c",),
    "toughness.charpy_upper_shelf": ("e",),
    "toughness.charpy_subsize": ("c",),
    "k.plate.extended": ("c",),
    "k.cylinder.bulging_through": ("b", "c"),
    "k.cylinder.bulging_surface": ("b", "c"),
    "k.cylinder.tables.through": ("b", "d"),
    "k.cylinder.tables.axial_extended_external": ("d",),
    "k.cylinder.tables.circumferential_surface": ("d",),
    "k.cylinder.tables.circumferential_extended": ("b", "d"),
    "k.cylinder.extended_external": ("c",),
    "sref.cylinder.axial_thin": ("b", "c"),
    "sref.cylinder.axial_pressure": ("c",),
    "sref.cylinder.circumferential_thin": ("b", "c"),
}
