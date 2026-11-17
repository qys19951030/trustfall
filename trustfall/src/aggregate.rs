//! Streaming group-by aggregation over Trustfall query result rows.
//!
//! [`execute_query`](crate::execute_query) returns its results as an iterator of rows,
//! each row mapping field names to [`FieldValue`]s. The query engine does not perform
//! grouping or aggregation: report-style use cases would otherwise have to collect the
//! entire result set into memory and group it themselves.
//!
//! This module consumes a row iterator directly and performs a single streaming pass.
//! Rows are pulled one at a time and only the per-group aggregation state is retained;
//! the original rows are never buffered, so memory use scales with the number of groups
//! rather than the number of rows.
//!
//! # Example
//!
//! ```
//! use std::collections::BTreeMap;
//! use std::sync::Arc;
//!
//! use trustfall::FieldValue;
//! use trustfall::aggregate::{AggregateOperation, AggregationSpec, aggregate};
//!
//! let rows = vec![
//!     BTreeMap::from([
//!         (Arc::from("group"), FieldValue::from("a")),
//!         (Arc::from("score"), FieldValue::Int64(1)),
//!         (Arc::from("votes"), FieldValue::Int64(10)),
//!     ]),
//!     BTreeMap::from([
//!         (Arc::from("group"), FieldValue::from("a")),
//!         (Arc::from("score"), FieldValue::Int64(2)),
//!         (Arc::from("votes"), FieldValue::Int64(20)),
//!     ]),
//!     BTreeMap::from([
//!         (Arc::from("group"), FieldValue::from("b")),
//!         (Arc::from("score"), FieldValue::Int64(5)),
//!         (Arc::from("votes"), FieldValue::Int64(7)),
//!     ]),
//! ];
//!
//! let spec = AggregationSpec::new()
//!     .group_by("group")
//!     .sum("score")
//!     .count("votes");
//! let groups = aggregate(rows.into_iter(), &spec);
//! assert_eq!(groups.len(), 2);
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::FieldValue;

/// How the values of one measure field are combined within a group.
///
/// Every operation ignores rows whose measure value is null or whose row does not
/// contain the measure field, except for [`AggregateOperation::Count`], which counts
/// every row in the group regardless of the field's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOperation {
    /// Sum of the numeric values.
    Sum,

    /// Smallest of the numeric values, compared by numeric value.
    Min,

    /// Largest of the numeric values, compared by numeric value.
    Max,

    /// Number of rows in the group.
    Count,
}

/// One measure: the field to aggregate together with how its values are combined.
///
/// Construct one with [`MeasureSpec::new`] or through the convenience constructors
/// [`MeasureSpec::sum`], [`MeasureSpec::min`], [`MeasureSpec::max`],
/// and [`MeasureSpec::count`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasureSpec {
    field_name: Arc<str>,
    operation: AggregateOperation,
}

impl MeasureSpec {
    /// Aggregate `field_name` using the given [`AggregateOperation`].
    pub fn new(field_name: impl Into<Arc<str>>, operation: AggregateOperation) -> Self {
        Self { field_name: field_name.into(), operation }
    }

    /// Sum the values of `field_name`.
    pub fn sum(field_name: impl Into<Arc<str>>) -> Self {
        Self::new(field_name, AggregateOperation::Sum)
    }

    /// Take the minimum value of `field_name`.
    pub fn min(field_name: impl Into<Arc<str>>) -> Self {
        Self::new(field_name, AggregateOperation::Min)
    }

    /// Take the maximum value of `field_name`.
    pub fn max(field_name: impl Into<Arc<str>>) -> Self {
        Self::new(field_name, AggregateOperation::Max)
    }

    /// Count the rows of the group. The values of `field_name` are not inspected.
    pub fn count(field_name: impl Into<Arc<str>>) -> Self {
        Self::new(field_name, AggregateOperation::Count)
    }

    /// Name of the field this measure reads.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Operation used to combine this measure's values.
    pub fn operation(&self) -> AggregateOperation {
        self.operation
    }
}

/// Description of a streaming aggregation: the grouping keys and the measures to compute.
///
/// Build one with [`AggregationSpec::new`] and the builder methods
/// [`group_by`](AggregationSpec::group_by), [`measure`](AggregationSpec::measure),
/// [`sum`](AggregationSpec::sum), [`min`](AggregationSpec::min),
/// [`max`](AggregationSpec::max), and [`count`](AggregationSpec::count).
///
/// Measure field names should be distinct: the per-group measure result is a map keyed
/// by field name, so two measures over the same field would overwrite each other in the
/// output. Grouping-key fields may legitimately also appear as measure fields.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AggregationSpec {
    group_by_fields: Vec<Arc<str>>,
    measures: Vec<MeasureSpec>,
}

impl AggregationSpec {
    /// Create an empty spec with no grouping keys and no measures.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `field_name` as a grouping key.
    ///
    /// Rows whose key values are all equal land in the same group. A row missing a
    /// grouping-key field, or one whose key value is null, is grouped under the null
    /// key for that field and is never discarded.
    pub fn group_by(mut self, field_name: impl Into<Arc<str>>) -> Self {
        self.group_by_fields.push(field_name.into());
        self
    }

    /// Add a fully described measure.
    pub fn measure(mut self, measure: MeasureSpec) -> Self {
        self.measures.push(measure);
        self
    }

    /// Add a sum measure over `field_name`.
    pub fn sum(self, field_name: impl Into<Arc<str>>) -> Self {
        self.measure(MeasureSpec::sum(field_name))
    }

    /// Add a minimum measure over `field_name`.
    pub fn min(self, field_name: impl Into<Arc<str>>) -> Self {
        self.measure(MeasureSpec::min(field_name))
    }

    /// Add a maximum measure over `field_name`.
    pub fn max(self, field_name: impl Into<Arc<str>>) -> Self {
        self.measure(MeasureSpec::max(field_name))
    }

    /// Add a row-count measure. The values of `field_name` are not inspected.
    pub fn count(self, field_name: impl Into<Arc<str>>) -> Self {
        self.measure(MeasureSpec::count(field_name))
    }

    /// Grouping-key field names, in declaration order.
    pub fn group_by_fields(&self) -> &[Arc<str>] {
        &self.group_by_fields
    }

    /// Measures to compute, in declaration order.
    pub fn measures(&self) -> &[MeasureSpec] {
        &self.measures
    }
}

/// The outcome of one measure within one group.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasureValue {
    /// The measure computed successfully.
    Value(FieldValue),

    /// The measure failed for this group, for example because a non-numeric value
    /// entered a numeric merge or because an integer merge overflowed.
    ///
    /// A failed measure does not affect any other measure in the same group, nor any
    /// other group.
    Failed,
}

/// One output group: its key values and its merged measure values.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupResult {
    /// Maps each grouping-key field name to this group's key value.
    ///
    /// Empty when the spec has no grouping keys, in which case all rows belong to a
    /// single total group.
    pub keys: BTreeMap<Arc<str>, FieldValue>,

    /// Maps each measure field name to its merged value.
    pub measures: BTreeMap<Arc<str>, MeasureValue>,
}

impl GroupResult {
    /// Look up a grouping-key value by field name.
    pub fn key(&self, field_name: &str) -> Option<&FieldValue> {
        self.keys.get(field_name)
    }

    /// Look up a measure result by field name.
    pub fn measure(&self, field_name: &str) -> Option<&MeasureValue> {
        self.measures.get(field_name)
    }
}

/// Accumulator state for one measure within one group.
///
/// Numeric operations keep every observed integer as an exact `i128` until the first
/// `Float64` value arrives; from that point on, values merge using ordinary
/// floating-point arithmetic.
#[derive(Debug, Clone)]
enum MeasureAccumulator {
    /// No non-null value has been observed yet.
    Empty,

    /// Exact integer accumulator (sum, min, or max, depending on the operation).
    Integer(i128),

    /// Floating-point accumulator, used as soon as a `Float64` value is observed.
    Float(f64),

    /// Counts rows independently of the measure field's values.
    Count(u64),

    /// This measure can no longer produce a value for its group.
    Failed,
}

/// A numeric value observed in a measure, with the exact integer representation kept.
#[derive(Debug, Clone, Copy)]
enum NumericInput {
    Integer(i128),
    Float(f64),
}

impl NumericInput {
    /// Interpret a field value as a numeric measure input.
    ///
    /// Returns `Ok(None)` for nulls (and for fields absent from the row, which are
    /// looked up as [`FieldValue::Null`]): they do not participate in numeric merging.
    /// Strings, booleans, enums, lists, and any variant added to [`FieldValue`] later
    /// produce `Err(())`, which fails only the affected measure.
    fn from_field_value(value: &FieldValue) -> Result<Option<Self>, ()> {
        match value {
            FieldValue::Null => Ok(None),
            FieldValue::Int64(v) => Ok(Some(Self::Integer(i128::from(*v)))),
            FieldValue::Uint64(v) => Ok(Some(Self::Integer(i128::from(*v)))),
            FieldValue::Float64(v) => Ok(Some(Self::Float(*v))),
            _ => Err(()),
        }
    }
}

/// Combine two floating-point accumulator values according to `operation`.
///
/// Nulls never reach this function, so a null is never converted into a zero.
fn combine_float(operation: AggregateOperation, accumulator: f64, value: f64) -> f64 {
    match operation {
        AggregateOperation::Sum => accumulator + value,
        AggregateOperation::Min => accumulator.min(value),
        AggregateOperation::Max => accumulator.max(value),
        // Counting does not merge numeric values and never reaches this function.
        AggregateOperation::Count => {
            unreachable!("count measures do not merge numeric values")
        }
    }
}

/// Combine two exact integer accumulator values.
///
/// Returns `None` when a sum overflows the `i128` accumulator. Min and max cannot
/// overflow, since their result is one of the two inputs.
fn combine_integer(operation: AggregateOperation, accumulator: i128, value: i128) -> Option<i128> {
    match operation {
        AggregateOperation::Sum => accumulator.checked_add(value),
        AggregateOperation::Min => Some(accumulator.min(value)),
        AggregateOperation::Max => Some(accumulator.max(value)),
        AggregateOperation::Count => {
            unreachable!("count measures do not merge numeric values")
        }
    }
}

impl MeasureAccumulator {
    /// Fold one observed numeric value into this accumulator.
    ///
    /// `None` means the row's measure value was null or missing: numeric operations
    /// skip such rows. A failed accumulator stays failed, so one bad row permanently
    /// fails just the one measure it belongs to.
    fn update_numeric(&mut self, operation: AggregateOperation, input: Option<NumericInput>) {
        let Some(input) = input else {
            return;
        };

        match self {
            Self::Empty => {
                *self = match input {
                    NumericInput::Integer(value) => Self::Integer(value),
                    NumericInput::Float(value) => Self::Float(value),
                };
            }
            Self::Integer(accumulator) => match input {
                NumericInput::Integer(value) => {
                    match combine_integer(operation, *accumulator, value) {
                        Some(result) => *accumulator = result,
                        None => *self = Self::Failed,
                    }
                }
                NumericInput::Float(value) => {
                    // The exact integer accumulator becomes a float from here on.
                    *self = Self::Float(combine_float(operation, *accumulator as f64, value));
                }
            },
            Self::Float(accumulator) => {
                let value = match input {
                    NumericInput::Integer(integer) => integer as f64,
                    NumericInput::Float(float) => float,
                };
                *accumulator = combine_float(operation, *accumulator, value);
            }
            // A failed measure never recovers; counts advance through a separate path.
            Self::Count(..) | Self::Failed => {}
        }
    }

    /// Convert the accumulator into the value exposed in the group result.
    fn finish(self) -> MeasureValue {
        match self {
            // No non-null value was observed: the measure is null rather than zero.
            Self::Empty => MeasureValue::Value(FieldValue::Null),
            Self::Integer(value) => {
                // The inputs were `i64`/`u64` values; represent the exact result using
                // whichever integer variant fits. A sum can grow beyond both ranges, in
                // which case the measure is reported as failed.
                match i64::try_from(value) {
                    Ok(signed) => MeasureValue::Value(FieldValue::Int64(signed)),
                    Err(_) => match u64::try_from(value) {
                        Ok(unsigned) => MeasureValue::Value(FieldValue::Uint64(unsigned)),
                        Err(_) => MeasureValue::Failed,
                    },
                }
            }
            Self::Float(value) => MeasureValue::Value(FieldValue::Float64(value)),
            Self::Count(value) => MeasureValue::Value(FieldValue::Uint64(value)),
            Self::Failed => MeasureValue::Failed,
        }
    }
}

/// Mutable state of one group while the input iterator is being consumed.
struct Group {
    /// Key values in the same order as the spec's grouping-key fields.
    key: Vec<FieldValue>,
    measures: Vec<MeasureAccumulator>,
}

/// Extract a row's grouping-key values.
///
/// A missing field and an explicit null are both represented as [`FieldValue::Null`],
/// so such rows land in the null-key group rather than being discarded.
fn row_key(row: &BTreeMap<Arc<str>, FieldValue>, group_by_fields: &[Arc<str>]) -> Vec<FieldValue> {
    group_by_fields
        .iter()
        .map(|field_name| row.get(field_name).cloned().unwrap_or(FieldValue::Null))
        .collect()
}

/// Run a streaming group-by aggregation over the rows produced by `rows`.
///
/// The iterator is consumed in a single pass and the input rows are not retained:
/// only one group entry per distinct grouping key is kept in memory. Groups are
/// returned in the order their first row was encountered.
///
/// An empty input produces no groups. With no grouping keys configured, every row
/// belongs to a single total group, which only appears once at least one row has
/// been processed.
///
/// # Measure semantics
///
/// - Every row contributes to its group's row counts, including rows with null or
///   missing measure values.
/// - Null and missing measure values do not participate in numeric merging; a measure
///   that only ever saw nulls comes out as [`FieldValue::Null`], never as zero.
/// - `Int64` and `Uint64` values merge as exact integers even when mixed, as long as
///   no floating-point value enters the measure. Floating-point merging follows
///   ordinary `f64` arithmetic.
/// - A non-numeric value entering a numeric merge, or an integer merge that
///   overflows, marks that measure of that group as [`MeasureValue::Failed`]. For
///   valid [`FieldValue`]s the aggregation never panics: other measures of the same
///   group and all other groups keep computing normally. (NaN and infinite values are
///   not valid `FieldValue::Float64` values and are outside this contract; used as a
///   grouping key they hit the finite-value check in `FieldValue`'s own equality.)
pub fn aggregate<I>(rows: I, spec: &AggregationSpec) -> Vec<GroupResult>
where
    I: IntoIterator<Item = BTreeMap<Arc<str>, FieldValue>>,
{
    let mut groups: Vec<Group> = Vec::new();

    for row in rows {
        let key = row_key(&row, &spec.group_by_fields);

        // `FieldValue` has no `Hash` or `Ord` impl, so groups live in a `Vec` and are
        // located by equality. `FieldValue`'s own `Eq` treats e.g. `Int64(1)` and
        // `Uint64(1)` as equal, which is the intended numeric key semantics.
        let group_index = match groups.iter().position(|group| group.key == key) {
            Some(index) => index,
            None => {
                let measures = spec
                    .measures
                    .iter()
                    .map(|measure| match measure.operation() {
                        AggregateOperation::Count => MeasureAccumulator::Count(0),
                        _ => MeasureAccumulator::Empty,
                    })
                    .collect();
                groups.push(Group { key, measures });
                groups.len() - 1
            }
        };

        for (measure, accumulator) in
            spec.measures.iter().zip(groups[group_index].measures.iter_mut())
        {
            match measure.operation() {
                // Counts include every row, even rows whose measure value is null.
                AggregateOperation::Count => {
                    if let MeasureAccumulator::Count(count) = accumulator {
                        *count = count.saturating_add(1);
                    }
                }
                operation => {
                    let value = row.get(measure.field_name()).unwrap_or(&FieldValue::Null);
                    match NumericInput::from_field_value(value) {
                        Ok(input) => accumulator.update_numeric(operation, input),
                        Err(()) => *accumulator = MeasureAccumulator::Failed,
                    }
                }
            }
        }
    }

    groups
        .into_iter()
        .map(|group| {
            let keys = spec.group_by_fields.iter().cloned().zip(group.key).collect();
            let measures = spec
                .measures
                .iter()
                .zip(group.measures)
                .map(|(measure, accumulator)| (measure.field_name.clone(), accumulator.finish()))
                .collect();

            GroupResult { keys, measures }
        })
        .collect()
}
