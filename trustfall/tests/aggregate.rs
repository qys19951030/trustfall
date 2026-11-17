//! Integration tests for the streaming aggregation module.
//!
//! Rows and specs are constructed by hand; no external data source is involved.

use std::collections::BTreeMap;
use std::sync::Arc;

use trustfall::FieldValue;
use trustfall::aggregate::{
    AggregateOperation, AggregationSpec, MeasureSpec, MeasureValue, aggregate,
};

/// Build a row from `(field name, value)` pairs.
fn row(pairs: Vec<(&str, FieldValue)>) -> BTreeMap<Arc<str>, FieldValue> {
    pairs.into_iter().map(|(name, value)| (Arc::from(name), value)).collect()
}

#[test]
fn empty_input_produces_no_groups() {
    let spec = AggregationSpec::new().group_by("g").count("n");
    let rows: Vec<BTreeMap<Arc<str>, FieldValue>> = Vec::new();
    let groups = aggregate(rows, &spec);
    assert!(groups.is_empty());
}

#[test]
fn equal_keys_merge_into_one_group_and_count_rows() {
    let spec = AggregationSpec::new().group_by("g").count("n");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("n", FieldValue::Int64(1))]),
        row(vec![("g", FieldValue::from("a")), ("n", FieldValue::Null)]),
        row(vec![("g", FieldValue::from("b")), ("n", FieldValue::Int64(9))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].key("g"), Some(&FieldValue::from("a")));
    assert_eq!(groups[0].measure("n"), Some(&MeasureValue::Value(FieldValue::Uint64(2))));
    assert_eq!(groups[1].key("g"), Some(&FieldValue::from("b")));
    assert_eq!(groups[1].measure("n"), Some(&MeasureValue::Value(FieldValue::Uint64(1))));
}

#[test]
fn non_contiguous_rows_with_same_key_merge() {
    let spec = AggregationSpec::new().group_by("g").count("n");
    let rows = vec![
        row(vec![("g", FieldValue::from("a"))]),
        row(vec![("g", FieldValue::from("b"))]),
        row(vec![("g", FieldValue::from("a"))]),
        row(vec![("g", FieldValue::from("b"))]),
        row(vec![("g", FieldValue::from("a"))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 2);
    // Groups appear in first-encounter order.
    assert_eq!(groups[0].key("g"), Some(&FieldValue::from("a")));
    assert_eq!(groups[0].measure("n"), Some(&MeasureValue::Value(FieldValue::Uint64(3))));
    assert_eq!(groups[1].key("g"), Some(&FieldValue::from("b")));
    assert_eq!(groups[1].measure("n"), Some(&MeasureValue::Value(FieldValue::Uint64(2))));
}

#[test]
fn missing_and_null_grouping_keys_share_the_null_group() {
    let spec = AggregationSpec::new().group_by("g").count("n");
    let rows = vec![
        row(vec![("g", FieldValue::Null), ("n", FieldValue::Int64(1))]),
        // This row has no "g" field at all: it is treated as a null key.
        row(vec![("n", FieldValue::Int64(2))]),
        row(vec![("g", FieldValue::from("x")), ("n", FieldValue::Int64(3))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].key("g"), Some(&FieldValue::Null));
    assert_eq!(groups[0].measure("n"), Some(&MeasureValue::Value(FieldValue::Uint64(2))));
    assert_eq!(groups[1].key("g"), Some(&FieldValue::from("x")));
}

#[test]
fn multiple_grouping_keys_must_all_match() {
    let spec = AggregationSpec::new().group_by("g1").group_by("g2").count("n");
    let rows = vec![
        row(vec![("g1", FieldValue::from("a")), ("g2", FieldValue::Int64(1))]),
        row(vec![("g1", FieldValue::from("a")), ("g2", FieldValue::Int64(2))]),
        row(vec![("g1", FieldValue::from("a")), ("g2", FieldValue::Int64(1))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].key("g1"), Some(&FieldValue::from("a")));
    assert_eq!(groups[0].key("g2"), Some(&FieldValue::Int64(1)));
    assert_eq!(groups[0].measure("n"), Some(&MeasureValue::Value(FieldValue::Uint64(2))));
    assert_eq!(groups[1].key("g2"), Some(&FieldValue::Int64(2)));
    assert_eq!(groups[1].measure("n"), Some(&MeasureValue::Value(FieldValue::Uint64(1))));
}

#[test]
fn no_grouping_keys_puts_every_row_in_one_total_group() {
    let spec = AggregationSpec::new().count("n");
    let rows = vec![
        row(vec![("n", FieldValue::Int64(1))]),
        row(vec![("n", FieldValue::Int64(2))]),
        row(vec![("n", FieldValue::Int64(3))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 1);
    assert!(groups[0].keys.is_empty());
    assert_eq!(groups[0].measure("n"), Some(&MeasureValue::Value(FieldValue::Uint64(3))));
}

#[test]
fn total_group_is_absent_for_empty_input() {
    let spec = AggregationSpec::new().count("n");
    let rows: Vec<BTreeMap<Arc<str>, FieldValue>> = Vec::new();
    assert!(aggregate(rows, &spec).is_empty());
}

#[test]
fn spec_reports_its_keys_and_measures() {
    let spec = AggregationSpec::new()
        .group_by("g")
        .measure(MeasureSpec::new("m", AggregateOperation::Min));
    assert_eq!(spec.group_by_fields(), &[Arc::from("g")]);
    assert_eq!(spec.measures().len(), 1);
    assert_eq!(spec.measures()[0].field_name(), "m");
    assert_eq!(spec.measures()[0].operation(), AggregateOperation::Min);
}

#[test]
fn nulls_are_skipped_in_sum_and_only_nulls_yield_null() {
    let spec = AggregationSpec::new().group_by("g").sum("m");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Int64(3))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Null)]),
        row(vec![("g", FieldValue::from("a"))]),
        row(vec![("g", FieldValue::from("only_nulls")), ("m", FieldValue::Null)]),
        row(vec![("g", FieldValue::from("only_nulls"))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Int64(3))));
    // Nulls never become zero: with no observed value, the result stays null.
    assert_eq!(groups[1].measure("m"), Some(&MeasureValue::Value(FieldValue::Null)));
}

#[test]
fn count_includes_rows_with_null_and_missing_values() {
    let spec = AggregationSpec::new().group_by("g").count("m");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Int64(3))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Null)]),
        row(vec![("g", FieldValue::from("a"))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Uint64(3))));
}

#[test]
fn mixed_integer_types_merge_with_exact_precision() {
    let spec = AggregationSpec::new().group_by("g").sum("m");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Int64(i64::MIN))]),
        row(vec![
            ("g", FieldValue::from("a")),
            ("m", FieldValue::Uint64(u64::try_from(i64::MAX).unwrap() + 1)),
        ]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Uint64(2))]),
    ];

    let groups = aggregate(rows, &spec);
    // i64::MIN + (i64::MAX + 1) + 2 = 2, computed exactly despite mixed types.
    assert_eq!(groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Int64(2))));
}

#[test]
fn large_unsigned_sum_is_represented_as_uint64() {
    let spec = AggregationSpec::new().group_by("g").sum("m");
    let first = u64::try_from(i64::MAX).unwrap() + 1;
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Uint64(first))]),
        // The sum stays within `u64` but exceeds the `i64` range, so it comes back
        // as `Uint64` rather than `Int64`.
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Uint64(10))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Uint64(first + 10))));
}

#[test]
fn integer_and_float_merge_using_floating_point_arithmetic() {
    let spec = AggregationSpec::new().group_by("g").sum("m");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Int64(1))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Float64(0.5))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Uint64(2))]),
        // A null must not be coerced into zero.
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Null)]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Float64(3.5))));
}

#[test]
fn float_merging_follows_ordinary_floating_point_rules() {
    let spec = AggregationSpec::new().group_by("g").sum("m");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Float64(0.1))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Float64(0.2))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Float64(0.1 + 0.2))));
}

#[test]
fn min_and_max_compare_by_numeric_value_across_integer_types() {
    let min_spec = AggregationSpec::new().group_by("g").min("m");
    let max_spec = AggregationSpec::new().group_by("g").max("m");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Int64(-5))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Uint64(7))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Int64(3))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Null)]),
    ];

    let min_groups = aggregate(rows.clone(), &min_spec);
    let max_groups = aggregate(rows, &max_spec);
    assert_eq!(min_groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Int64(-5))));
    // Pin the exact variant: plain `assert_eq!` would also accept `Uint64(7)` because
    // `FieldValue`'s numeric `Eq` treats the two as equal. Integer results prefer the
    // `Int64` variant whenever they fit; `Uint64` is only used past the `i64` range.
    assert!(matches!(max_groups[0].measure("m"), Some(MeasureValue::Value(FieldValue::Int64(7)))));
}

#[test]
fn non_numeric_measure_value_fails_only_that_measure() {
    let spec = AggregationSpec::new().group_by("g").sum("bad").sum("good").count("rows");
    let rows = vec![
        row(vec![
            ("g", FieldValue::from("a")),
            ("bad", FieldValue::from("not a number")),
            ("good", FieldValue::Int64(4)),
        ]),
        row(vec![
            ("g", FieldValue::from("a")),
            ("bad", FieldValue::Boolean(true)),
            ("good", FieldValue::Int64(5)),
        ]),
        row(vec![
            ("g", FieldValue::from("b")),
            ("bad", FieldValue::Int64(1)),
            ("good", FieldValue::Int64(9)),
        ]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 2);

    // Group "a": the string and boolean values fail the "bad" measure, while the
    // sibling measure and the row count keep computing.
    assert_eq!(groups[0].measure("bad"), Some(&MeasureValue::Failed));
    assert_eq!(groups[0].measure("good"), Some(&MeasureValue::Value(FieldValue::Int64(9))));
    assert_eq!(groups[0].measure("rows"), Some(&MeasureValue::Value(FieldValue::Uint64(2))));

    // Group "b" is completely unaffected by the other group's failure.
    assert_eq!(groups[1].measure("bad"), Some(&MeasureValue::Value(FieldValue::Int64(1))));
    assert_eq!(groups[1].measure("good"), Some(&MeasureValue::Value(FieldValue::Int64(9))));
}

#[test]
fn failed_measure_does_not_recover_on_later_valid_rows() {
    let spec = AggregationSpec::new().group_by("g").sum("m");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Boolean(false))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Int64(10))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups[0].measure("m"), Some(&MeasureValue::Failed));
}

#[test]
fn sums_outside_the_integer_output_range_fail_without_panicking() {
    let spec = AggregationSpec::new().group_by("g").sum("m").count("rows");
    let rows = vec![
        row(vec![("g", FieldValue::from("big")), ("m", FieldValue::Uint64(u64::MAX))]),
        row(vec![("g", FieldValue::from("big")), ("m", FieldValue::Uint64(u64::MAX))]),
        // The negative direction leaves the `i64` output range as well. Note both
        // totals still fit in the internal `i128` accumulator: failure is decided when
        // the final value cannot be represented as `Int64` or `Uint64`.
        row(vec![("g", FieldValue::from("small")), ("m", FieldValue::Int64(i64::MIN))]),
        row(vec![("g", FieldValue::from("small")), ("m", FieldValue::Int64(-1))]),
        row(vec![("g", FieldValue::from("fine")), ("m", FieldValue::Int64(1))]),
    ];

    let groups = aggregate(rows, &spec);
    let by_key = |key: &str| {
        groups
            .iter()
            .find(|group| group.key("g") == Some(&FieldValue::from(key)))
            .expect("group exists")
    };

    assert_eq!(by_key("big").measure("m"), Some(&MeasureValue::Failed));
    assert_eq!(by_key("small").measure("m"), Some(&MeasureValue::Failed));
    // Other groups and the count measure are unaffected.
    assert_eq!(by_key("fine").measure("m"), Some(&MeasureValue::Value(FieldValue::Int64(1))));
    assert_eq!(by_key("big").measure("rows"), Some(&MeasureValue::Value(FieldValue::Uint64(2))));
}

#[test]
fn aggregation_accepts_a_lazy_iterator_and_folds_every_pulled_row() {
    // A custom iterator that produces rows on demand from a single reused slot,
    // exercising the `IntoIterator` entry point without first building a `Vec`.
    // (A black-box test cannot distinguish folding from buffering; the no-buffering
    // guarantee is structural: `aggregate` holds only the `Vec<Group>` state.)
    struct StreamingRows {
        next: usize,
        total: usize,
        slot: BTreeMap<Arc<str>, FieldValue>,
    }

    impl Iterator for StreamingRows {
        type Item = BTreeMap<Arc<str>, FieldValue>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.next >= self.total {
                return None;
            }
            let key = self.next % 3;
            self.slot.clear();
            self.slot.insert(Arc::from("g"), FieldValue::Uint64(u64::try_from(key).unwrap()));
            self.slot.insert(Arc::from("m"), FieldValue::Int64(1));
            self.next += 1;
            Some(self.slot.clone())
        }
    }

    let rows = StreamingRows { next: 0, total: 10_000, slot: BTreeMap::new() };
    let spec = AggregationSpec::new().group_by("g").sum("m").count("rows");

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 3);
    // Over 10_000 rows keyed by `next % 3`, key 0 occurs one more time than the rest.
    let expected = [
        (FieldValue::Uint64(0), 3334u64),
        (FieldValue::Uint64(1), 3333),
        (FieldValue::Uint64(2), 3333),
    ];
    for (key, count) in expected {
        let group = groups.iter().find(|group| group.key("g") == Some(&key)).expect("group exists");
        assert_eq!(
            group.measure("m"),
            Some(&MeasureValue::Value(FieldValue::Int64(i64::try_from(count).unwrap())))
        );
        assert_eq!(group.measure("rows"), Some(&MeasureValue::Value(FieldValue::Uint64(count))));
    }
}

#[test]
fn empty_iterator_is_pulled_lazily_and_yields_nothing() {
    let spec = AggregationSpec::new().count("m");
    let mut pulled = false;
    let groups = aggregate(
        std::iter::from_fn(|| {
            pulled = true;
            None::<BTreeMap<Arc<str>, FieldValue>>
        }),
        &spec,
    );
    assert!(pulled);
    assert!(groups.is_empty());
}

#[test]
fn group_result_exposes_keys_and_measures_field_by_field() {
    let spec = AggregationSpec::new().group_by("g1").group_by("g2").min("m");
    let rows = vec![row(vec![
        ("g1", FieldValue::from("a")),
        ("g2", FieldValue::Boolean(true)),
        ("m", FieldValue::Int64(42)),
    ])];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].keys.len(), 2);
    assert_eq!(groups[0].key("g1"), Some(&FieldValue::from("a")));
    assert_eq!(groups[0].key("g2"), Some(&FieldValue::Boolean(true)));
    assert_eq!(groups[0].key("absent"), None);
    assert_eq!(groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Int64(42))));
    assert_eq!(groups[0].measure("absent"), None);
}

#[test]
fn enum_and_list_values_also_fail_numeric_measures() {
    let spec = AggregationSpec::new().group_by("g").min("enum_field").max("list_field");
    let rows = vec![row(vec![
        ("g", FieldValue::from("a")),
        ("enum_field", FieldValue::Enum(Arc::from("VARIANT"))),
        ("list_field", FieldValue::List(Arc::from([FieldValue::Int64(1)]))),
    ])];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups[0].measure("enum_field"), Some(&MeasureValue::Failed));
    assert_eq!(groups[0].measure("list_field"), Some(&MeasureValue::Failed));
}

#[test]
fn count_ignores_measure_values_including_non_numeric_ones() {
    let spec = AggregationSpec::new().group_by("g").count("label");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("label", FieldValue::from("one"))]),
        row(vec![("g", FieldValue::from("a")), ("label", FieldValue::Boolean(false))]),
        row(vec![("g", FieldValue::from("a"))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups[0].measure("label"), Some(&MeasureValue::Value(FieldValue::Uint64(3))));
}

#[test]
fn min_and_max_follow_floating_point_rules() {
    let min_spec = AggregationSpec::new().group_by("g").min("m");
    let max_spec = AggregationSpec::new().group_by("g").max("m");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Float64(-1.5))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Int64(2))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Float64(0.25))]),
    ];

    let min_groups = aggregate(rows.clone(), &min_spec);
    let max_groups = aggregate(rows, &max_spec);
    assert_eq!(min_groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Float64(-1.5))));
    assert_eq!(max_groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Float64(2.0))));
}

#[test]
fn numeric_measure_over_only_nulls_is_distinct_from_count() {
    let spec =
        AggregationSpec::new().group_by("g").sum("sum_m").min("min_m").max("max_m").count("rows");
    let rows = vec![
        row(vec![
            ("g", FieldValue::from("a")),
            ("sum_m", FieldValue::Null),
            ("min_m", FieldValue::Null),
            ("max_m", FieldValue::Null),
        ]),
        row(vec![("g", FieldValue::from("a"))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups[0].measure("sum_m"), Some(&MeasureValue::Value(FieldValue::Null)));
    assert_eq!(groups[0].measure("min_m"), Some(&MeasureValue::Value(FieldValue::Null)));
    assert_eq!(groups[0].measure("max_m"), Some(&MeasureValue::Value(FieldValue::Null)));
    assert_eq!(groups[0].measure("rows"), Some(&MeasureValue::Value(FieldValue::Uint64(2))));
}

#[test]
fn a_grouping_field_can_also_be_aggregated() {
    let spec = AggregationSpec::new().group_by("g").sum("g").count("rows");
    let rows = vec![row(vec![("g", FieldValue::Int64(1))]), row(vec![("g", FieldValue::Int64(1))])];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].key("g"), Some(&FieldValue::Int64(1)));
    assert_eq!(groups[0].measure("g"), Some(&MeasureValue::Value(FieldValue::Int64(2))));
    assert_eq!(groups[0].measure("rows"), Some(&MeasureValue::Value(FieldValue::Uint64(2))));
}

#[test]
fn string_keys_keep_distinct_groups_and_null_key_alongside() {
    let spec = AggregationSpec::new().group_by("g").count("rows");
    let rows = vec![
        row(vec![("g", FieldValue::from("x")), ("rows", FieldValue::Null)]),
        row(vec![("g", FieldValue::Null), ("rows", FieldValue::Null)]),
        row(vec![("rows", FieldValue::Null)]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].key("g"), Some(&FieldValue::from("x")));
    assert_eq!(groups[0].measure("rows"), Some(&MeasureValue::Value(FieldValue::Uint64(1))));
    assert_eq!(groups[1].key("g"), Some(&FieldValue::Null));
    assert_eq!(groups[1].measure("rows"), Some(&MeasureValue::Value(FieldValue::Uint64(2))));
}

#[test]
fn int64_and_uint64_keys_with_equal_numeric_value_share_a_group() {
    // FieldValue's Eq compares integers by numeric value, so Int64(1) and Uint64(1)
    // are the same grouping key even though their variants differ.
    let spec = AggregationSpec::new().group_by("g").count("rows");
    let rows = vec![
        row(vec![("g", FieldValue::Int64(1))]),
        row(vec![("g", FieldValue::Uint64(1))]),
        row(vec![("g", FieldValue::Int64(2))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].measure("rows"), Some(&MeasureValue::Value(FieldValue::Uint64(2))));
    assert_eq!(groups[1].measure("rows"), Some(&MeasureValue::Value(FieldValue::Uint64(1))));
}

#[test]
fn duplicate_measure_names_overwrite_each_other_in_the_output() {
    // This pins the documented behavior: the output map is keyed by field name, so a
    // second measure over the same field replaces the first one. Both are still
    // computed during the pass; only the final map entry survives.
    let spec = AggregationSpec::new().group_by("g").sum("m").min("m");
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Int64(3))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Int64(5))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups[0].measures.len(), 1);
    // The min measure was declared last, so it wins the single map slot.
    assert_eq!(groups[0].measure("m"), Some(&MeasureValue::Value(FieldValue::Int64(3))));
}

#[test]
fn empty_spec_over_rows_yields_one_total_group_with_empty_maps() {
    let spec = AggregationSpec::new();
    let rows = vec![
        row(vec![("anything", FieldValue::Int64(1))]),
        row(vec![("anything", FieldValue::Int64(2))]),
    ];

    let groups = aggregate(rows, &spec);
    assert_eq!(groups.len(), 1);
    assert!(groups[0].keys.is_empty());
    assert!(groups[0].measures.is_empty());
}

#[test]
fn min_and_max_past_the_i64_range_come_back_as_uint64() {
    let min_spec = AggregationSpec::new().group_by("g").min("m");
    let max_spec = AggregationSpec::new().group_by("g").max("m");
    let beyond_i64 = u64::try_from(i64::MAX).unwrap() + 1;
    let rows = vec![
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Uint64(beyond_i64))]),
        row(vec![("g", FieldValue::from("a")), ("m", FieldValue::Uint64(beyond_i64 + 10))]),
    ];

    let min_groups = aggregate(rows.clone(), &min_spec);
    let max_groups = aggregate(rows, &max_spec);
    assert!(matches!(
        min_groups[0].measure("m"),
        Some(MeasureValue::Value(FieldValue::Uint64(v))) if *v == beyond_i64
    ));
    assert!(matches!(
        max_groups[0].measure("m"),
        Some(MeasureValue::Value(FieldValue::Uint64(v))) if *v == beyond_i64 + 10
    ));
}
