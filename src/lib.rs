use std::cell::RefCell;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Mutex, OnceLock};
use std::thread;

use emacs::{defun, Env, IntoLisp, Result, Value};
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32Str};

emacs::plugin_is_GPL_compatible!();

#[emacs::module(
    name = "nucleo-completion-module",
    defun_prefix = "nucleo-completion",
    mod_in_name = false
)]
fn init(_: &Env) -> Result<()> {
    Ok(())
}

struct ScoredIndex {
    index: usize,
    score: u32,
}

const PARALLEL_BATCH_SIZE: usize = 2048;
const MIN_PARALLEL_ITEMS: usize = 8192;
const MIN_PARALLEL_BYTES: usize = 3_000_000;
const MODULE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy)]
struct SortOptions {
    ties_by_length: bool,
    ties_alphabetically: bool,
    ignore_case: bool,
}

impl SortOptions {
    fn from_lisp<'e>(
        sort_ties_by_length: Value<'e>,
        sort_ties_alphabetically: Value<'e>,
        ignore_case: bool,
    ) -> Self {
        Self {
            ties_by_length: sort_ties_by_length.is_not_nil(),
            ties_alphabetically: sort_ties_alphabetically.is_not_nil(),
            ignore_case,
        }
    }
}

fn case_matching(ignore_case: bool) -> CaseMatching {
    if ignore_case {
        CaseMatching::Ignore
    } else {
        CaseMatching::Respect
    }
}

thread_local! {
    /// Matcher cached on the main Emacs thread for serial scoring and
    /// for the small post-sort indices pass.  Reused across module
    /// invocations so that we do not pay for `Matcher::new` per call.
    static MATCHER: RefCell<Matcher> =
        RefCell::new(Matcher::new(Config::DEFAULT.match_paths()));
}

/// Persistent worker-thread matchers.
///
/// `thread::scope` spawns fresh OS threads on every parallel call, so
/// a `thread_local` would re-run `Matcher::new` per scope.  We keep
/// the matchers alive globally and let each worker borrow its
/// dedicated slot via a `Mutex`.  No contention occurs because each
/// worker is assigned a unique slot, but `Mutex` is required because
/// `Matcher` is not `Sync`.
static MATCHER_POOL: OnceLock<Vec<Mutex<Matcher>>> = OnceLock::new();

fn matcher_pool() -> &'static [Mutex<Matcher>] {
    MATCHER_POOL
        .get_or_init(|| {
            let n = thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
                .max(1);
            (0..n)
                .map(|_| Mutex::new(Matcher::new(Config::DEFAULT.match_paths())))
                .collect()
        })
        .as_slice()
}

fn with_matcher<T>(function: impl FnOnce(&mut Matcher) -> T) -> T {
    MATCHER.with(|matcher| function(&mut matcher.borrow_mut()))
}

fn collect_candidates<'e>(candidates: Value<'e>) -> Result<(Vec<Value<'e>>, Vec<String>)> {
    let mut values = Vec::new();
    let mut texts = Vec::new();
    let mut list = candidates;

    while list.is_not_nil() {
        let value: Value<'e> = list.car()?;
        let text: String = value.into_rust()?;
        values.push(value);
        texts.push(text);
        list = list.cdr()?;
    }

    Ok((values, texts))
}

fn collect_history_ranks<'e>(history_ranks: Value<'e>) -> Result<Vec<Option<usize>>> {
    let mut ranks = Vec::new();
    let mut list = history_ranks;

    while list.is_not_nil() {
        let value: Value<'e> = list.car()?;
        ranks.push(if value.is_not_nil() {
            Some(value.into_rust()?)
        } else {
            None
        });
        list = list.cdr()?;
    }

    Ok(ranks)
}

fn score_items_with_sort(
    pattern: &Pattern,
    texts: &[String],
    sort_options: SortOptions,
    history_ranks: Option<&[Option<usize>]>,
) -> Vec<ScoredIndex> {
    if !should_score_parallel_workload(texts) {
        return score_items_serial_with_sort(pattern, texts, sort_options, history_ranks);
    }

    let pool = matcher_pool();
    let workers = parallel_worker_count(texts.len(), pool.len());

    if workers == 1 {
        return score_items_serial_with_sort(pattern, texts, sort_options, history_ranks);
    }

    sort_scored(
        score_items_parallel(pattern, texts, pool, workers),
        texts,
        sort_options,
        history_ranks,
    )
}

fn score_items_serial_with_sort(
    pattern: &Pattern,
    texts: &[String],
    sort_options: SortOptions,
    history_ranks: Option<&[Option<usize>]>,
) -> Vec<ScoredIndex> {
    sort_scored(
        score_items_serial(pattern, texts),
        texts,
        sort_options,
        history_ranks,
    )
}

#[cfg(test)]
fn should_score_parallel(texts: &[String], workers: usize) -> bool {
    workers > 1 && should_score_parallel_workload(texts)
}

fn should_score_parallel_workload(texts: &[String]) -> bool {
    texts.len() >= MIN_PARALLEL_ITEMS
        && texts.iter().map(|text| text.len()).sum::<usize>() >= MIN_PARALLEL_BYTES
}

fn score_items_serial(pattern: &Pattern, texts: &[String]) -> Vec<ScoredIndex> {
    with_matcher(|matcher| score_item_range(pattern, texts, 0, matcher))
}

fn score_items_parallel(
    pattern: &Pattern,
    texts: &[String],
    pool: &[Mutex<Matcher>],
    workers: usize,
) -> Vec<ScoredIndex> {
    let batch_count = texts.len().div_ceil(PARALLEL_BATCH_SIZE);
    let next_batch = AtomicUsize::new(0);
    thread::scope(|scope| {
        let handles = pool
            .iter()
            .take(workers)
            .map(|matcher_mutex| {
                let next_batch = &next_batch;
                scope.spawn(move || {
                    score_worker_batches(pattern, texts, batch_count, next_batch, matcher_mutex)
                })
            })
            .collect::<Vec<_>>();

        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("nucleo worker thread panicked"))
            .collect()
    })
}

fn score_worker_batches(
    pattern: &Pattern,
    texts: &[String],
    batch_count: usize,
    next_batch: &AtomicUsize,
    matcher_mutex: &Mutex<Matcher>,
) -> Vec<ScoredIndex> {
    let mut guard = matcher_mutex
        .lock()
        .expect("nucleo matcher pool mutex poisoned");
    let matcher: &mut Matcher = &mut guard;
    let mut scored = Vec::new();

    while let Some((start, end)) = next_score_batch(texts.len(), batch_count, next_batch) {
        scored.extend(score_item_range(
            pattern,
            &texts[start..end],
            start,
            matcher,
        ));
    }

    scored
}

fn next_score_batch(
    item_count: usize,
    batch_count: usize,
    next_batch: &AtomicUsize,
) -> Option<(usize, usize)> {
    let batch_index = next_batch.fetch_add(1, AtomicOrdering::Relaxed);
    if batch_index >= batch_count {
        return None;
    }

    let start = batch_index * PARALLEL_BATCH_SIZE;
    let end = (start + PARALLEL_BATCH_SIZE).min(item_count);
    Some((start, end))
}

fn parallel_worker_count(item_count: usize, available_parallelism: usize) -> usize {
    let batches = item_count.div_ceil(PARALLEL_BATCH_SIZE).max(1);
    available_parallelism.max(1).min(batches)
}

fn sort_scored(
    mut scored: Vec<ScoredIndex>,
    texts: &[String],
    sort_options: SortOptions,
    history_ranks: Option<&[Option<usize>]>,
) -> Vec<ScoredIndex> {
    let lengths = sort_options.ties_by_length.then(|| {
        texts
            .iter()
            .map(|text| text.chars().count())
            .collect::<Vec<_>>()
    });
    let folded_texts = (sort_options.ties_alphabetically && sort_options.ignore_case).then(|| {
        texts
            .iter()
            .map(|text| text.to_lowercase())
            .collect::<Vec<_>>()
    });

    scored.sort_unstable_by(|a, b| {
        compare_scored(
            a,
            b,
            texts,
            sort_options,
            history_ranks,
            lengths.as_deref(),
            folded_texts.as_deref(),
        )
    });
    scored
}

fn compare_scored(
    a: &ScoredIndex,
    b: &ScoredIndex,
    texts: &[String],
    sort_options: SortOptions,
    history_ranks: Option<&[Option<usize>]>,
    lengths: Option<&[usize]>,
    folded_texts: Option<&[String]>,
) -> Ordering {
    b.score
        .cmp(&a.score)
        .then_with(|| compare_history_tie(a, b, history_ranks))
        .then_with(|| compare_length_tie(a, b, lengths))
        .then_with(|| compare_alphabetical_tie(a, b, texts, folded_texts, sort_options))
        .then_with(|| a.index.cmp(&b.index))
}

fn compare_history_tie(
    a: &ScoredIndex,
    b: &ScoredIndex,
    history_ranks: Option<&[Option<usize>]>,
) -> Ordering {
    let Some(history_ranks) = history_ranks else {
        return Ordering::Equal;
    };

    match (history_ranks[a.index], history_ranks[b.index]) {
        (Some(rank_a), Some(rank_b)) => rank_a.cmp(&rank_b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn compare_length_tie(a: &ScoredIndex, b: &ScoredIndex, lengths: Option<&[usize]>) -> Ordering {
    lengths.map_or(Ordering::Equal, |lengths| {
        lengths[a.index].cmp(&lengths[b.index])
    })
}

fn compare_alphabetical_tie(
    a: &ScoredIndex,
    b: &ScoredIndex,
    texts: &[String],
    folded_texts: Option<&[String]>,
    sort_options: SortOptions,
) -> Ordering {
    if !sort_options.ties_alphabetically {
        return Ordering::Equal;
    }

    match folded_texts {
        Some(folded_texts) => folded_texts[a.index].cmp(&folded_texts[b.index]),
        None => texts[a.index].cmp(&texts[b.index]),
    }
}

fn score_item_range(
    pattern: &Pattern,
    texts: &[String],
    offset: usize,
    matcher: &mut Matcher,
) -> Vec<ScoredIndex> {
    let mut buf = Vec::new();
    texts
        .iter()
        .enumerate()
        .filter_map(|(index, text)| {
            pattern
                .score(Utf32Str::new(text, &mut buf), matcher)
                .map(|score| ScoredIndex {
                    index: offset + index,
                    score,
                })
        })
        .collect()
}

/// Compute the highlight indices for one already-matched candidate.
///
/// `pattern.indices` re-runs the dynamic-programming match in order to
/// record positions, but the call only happens for the at most
/// `highlight_limit` top-ranked candidates and so its cost is bounded
/// by a small constant (default 25) regardless of the candidate-set
/// size.  Score and indices share the underlying algorithm; coalescing
/// them into a single per-candidate call would force allocating a
/// `Vec<u32>` for every match instead of just for the top-ranked few,
/// so the deliberate two-phase shape is retained.
fn matched_indices(pattern: &Pattern, text: &str, matcher: &mut Matcher) -> Option<Vec<u32>> {
    let mut buf = Vec::new();
    let mut indices = Vec::new();
    pattern.indices(Utf32Str::new(text, &mut buf), matcher, &mut indices)?;
    indices.sort_unstable();
    indices.dedup();
    Some(indices)
}

fn indices_to_lisp<'e>(env: &'e Env, indices: Vec<u32>) -> Result<Value<'e>> {
    let mut result = env.intern("nil")?;
    for index in indices.into_iter().rev() {
        result = env.cons(index.into_lisp(env)?, result)?;
    }
    Ok(result)
}

/// Ask Emacs whether new input is waiting.
///
/// The Emacs Lisp `input-pending-p` predicate is the standard way for
/// modules to cooperate with `while-no-input`: when it returns non-nil
/// we abandon the rest of the work and let the caller observe an empty
/// bundle, which the Lisp wrapper interprets as "interrupt and reuse
/// the previous filter result."  The check is cheap.
fn input_pending(env: &Env) -> Result<bool> {
    let args: [Value; 0] = [];
    Ok(env.call("input-pending-p", args)?.is_not_nil())
}

fn build_list_3<'e>(env: &'e Env, a: Value<'e>, b: Value<'e>, c: Value<'e>) -> Result<Value<'e>> {
    let nil = env.intern("nil")?;
    env.cons(a, env.cons(b, env.cons(c, nil)?)?)
}

fn interrupted_bundle<'e>(env: &'e Env) -> Result<Value<'e>> {
    let nil = env.intern("nil")?;
    let sentinel = env.intern("nucleo-completion-interrupted")?;
    build_list_3(env, sentinel, nil, nil)
}

fn build_candidates_list<'e>(
    env: &'e Env,
    texts: &[String],
    values: &[Value<'e>],
    matches: &[ScoredIndex],
    attach_scores: bool,
) -> Result<Value<'e>> {
    let nil = env.intern("nil")?;
    let mut candidates_list = nil;
    for scored in matches.iter().rev() {
        let candidate = candidate_value(env, texts, values, scored, attach_scores)?;
        candidates_list = env.cons(candidate, candidates_list)?;
    }
    Ok(candidates_list)
}

fn candidate_value<'e>(
    env: &'e Env,
    texts: &[String],
    values: &[Value<'e>],
    scored: &ScoredIndex,
    attach_score: bool,
) -> Result<Value<'e>> {
    if attach_score {
        propertized_score_candidate(
            env,
            values[scored.index],
            &texts[scored.index],
            scored.score,
        )
    } else {
        Ok(values[scored.index])
    }
}

fn propertized_score_candidate<'e>(
    env: &'e Env,
    value: Value<'e>,
    text: &str,
    score: u32,
) -> Result<Value<'e>> {
    let copy = env.call("copy-sequence", [value])?;
    let start = 0.into_lisp(env)?;
    let end = text.chars().count().into_lisp(env)?;
    let property = env.intern("nucleo-completion-score")?;
    let score = score.into_lisp(env)?;
    env.call("put-text-property", [start, end, property, score, copy])?;
    Ok(copy)
}

fn build_top_info<'e>(
    env: &'e Env,
    pattern: &Pattern,
    texts: &[String],
    values: &[Value<'e>],
    matches: &[ScoredIndex],
    highlight_limit: usize,
) -> Result<Value<'e>> {
    let nil = env.intern("nil")?;
    let mut top_info = nil;
    for scored in matches.iter().take(highlight_limit).rev() {
        let entry = build_top_info_entry(env, pattern, texts, values, scored, nil)?;
        top_info = env.cons(entry, top_info)?;
    }
    Ok(top_info)
}

fn build_top_info_entry<'e>(
    env: &'e Env,
    pattern: &Pattern,
    texts: &[String],
    values: &[Value<'e>],
    scored: &ScoredIndex,
    nil: Value<'e>,
) -> Result<Value<'e>> {
    let indices_value =
        match with_matcher(|matcher| matched_indices(pattern, &texts[scored.index], matcher)) {
            Some(indices) => indices_to_lisp(env, indices)?,
            None => nil,
        };
    env.cons(
        values[scored.index],
        env.cons(scored.score.into_lisp(env)?, env.cons(indices_value, nil)?)?,
    )
}

fn build_full_scores<'e>(
    env: &'e Env,
    matches: &[ScoredIndex],
    return_all_scores: bool,
) -> Result<Value<'e>> {
    let nil = env.intern("nil")?;
    if !return_all_scores {
        return Ok(nil);
    }

    let mut full_scores = nil;
    for scored in matches.iter().rev() {
        full_scores = env.cons(scored.score.into_lisp(env)?, full_scores)?;
    }
    Ok(full_scores)
}

fn build_candidate_bundle<'e>(
    env: &'e Env,
    pattern: &Pattern,
    values: &[Value<'e>],
    texts: &[String],
    matches: &[ScoredIndex],
    highlight_limit: usize,
    return_all_scores: bool,
) -> Result<Value<'e>> {
    let candidates_list = build_candidates_list(env, texts, values, matches, return_all_scores)?;
    let top_info = build_top_info(env, pattern, texts, values, matches, highlight_limit)?;
    let full_scores = build_full_scores(env, matches, return_all_scores)?;
    build_list_3(env, candidates_list, top_info, full_scores)
}

#[allow(clippy::too_many_arguments, reason = "Emacs module API is positional")]
fn candidates_impl<'e>(
    env: &'e Env,
    pattern: String,
    candidates: Value<'e>,
    ignore_case: Value<'e>,
    sort_ties_by_length: Value<'e>,
    sort_ties_alphabetically: Value<'e>,
    history_ranks: Option<Value<'e>>,
    highlight_limit: Value<'e>,
    return_all_scores: Value<'e>,
) -> Result<Value<'e>> {
    if input_pending(env)? {
        return interrupted_bundle(env);
    }

    let (values, texts) = collect_candidates(candidates)?;
    let ignore_case = ignore_case.is_not_nil();
    let sort_options =
        SortOptions::from_lisp(sort_ties_by_length, sort_ties_alphabetically, ignore_case);
    let history_ranks = match history_ranks {
        Some(value) => {
            let ranks = collect_history_ranks(value)?;
            (ranks.len() == texts.len()).then_some(ranks)
        }
        None => None,
    };
    let pattern = Pattern::parse(&pattern, case_matching(ignore_case), Normalization::Smart);
    let matches = score_items_with_sort(&pattern, &texts, sort_options, history_ranks.as_deref());

    if input_pending(env)? {
        return interrupted_bundle(env);
    }

    let highlight_limit = highlight_limit.into_rust::<usize>()?;
    let return_all_scores = return_all_scores.is_not_nil();
    build_candidate_bundle(
        env,
        &pattern,
        &values,
        &texts,
        &matches,
        highlight_limit,
        return_all_scores,
    )
}

#[defun]
fn module_version<'e>(env: &'e Env) -> Result<Value<'e>> {
    MODULE_VERSION.into_lisp(env)
}

#[allow(clippy::too_many_arguments, reason = "Emacs module API is positional")]
#[defun]
fn candidates<'e>(
    env: &'e Env,
    pattern: String,
    candidates: Value<'e>,
    ignore_case: Value<'e>,
    sort_ties_by_length: Value<'e>,
    sort_ties_alphabetically: Value<'e>,
    highlight_limit: Value<'e>,
    return_all_scores: Value<'e>,
) -> Result<Value<'e>> {
    candidates_impl(
        env,
        pattern,
        candidates,
        ignore_case,
        sort_ties_by_length,
        sort_ties_alphabetically,
        None,
        highlight_limit,
        return_all_scores,
    )
}

#[allow(clippy::too_many_arguments, reason = "Emacs module API is positional")]
#[defun]
fn candidates_with_history<'e>(
    env: &'e Env,
    pattern: String,
    candidates: Value<'e>,
    ignore_case: Value<'e>,
    sort_ties_by_length: Value<'e>,
    sort_ties_alphabetically: Value<'e>,
    history_ranks: Value<'e>,
    highlight_limit: Value<'e>,
    return_all_scores: Value<'e>,
) -> Result<Value<'e>> {
    candidates_impl(
        env,
        pattern,
        candidates,
        ignore_case,
        sort_ties_by_length,
        sort_ties_alphabetically,
        Some(history_ranks),
        highlight_limit,
        return_all_scores,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn parallel_worker_count_caps_to_batches() {
        assert_eq!(parallel_worker_count(PARALLEL_BATCH_SIZE, 8), 1);
        assert_eq!(parallel_worker_count(PARALLEL_BATCH_SIZE + 1, 8), 2);
        assert_eq!(parallel_worker_count(PARALLEL_BATCH_SIZE * 3, 8), 3);
    }

    #[test]
    fn parallel_worker_count_caps_to_available_parallelism() {
        assert_eq!(parallel_worker_count(PARALLEL_BATCH_SIZE * 8, 2), 2);
        assert_eq!(parallel_worker_count(PARALLEL_BATCH_SIZE * 8, 0), 1);
    }

    #[test]
    fn should_score_parallel_requires_multiple_workers() {
        let texts = vec!["x".repeat(768); MIN_PARALLEL_ITEMS];
        assert!(!should_score_parallel(&texts, 1));
        assert!(should_score_parallel(&texts, 2));
    }

    #[test]
    fn should_score_parallel_keeps_small_candidate_sets_serial() {
        let texts = vec!["x".repeat(2000); MIN_PARALLEL_ITEMS - 1];
        assert!(!should_score_parallel(&texts, 8));
    }

    #[test]
    fn should_score_parallel_keeps_short_candidate_sets_serial() {
        let texts = vec!["x".repeat(48); MIN_PARALLEL_ITEMS * 2];
        assert!(!should_score_parallel(&texts, 8));
    }

    #[test]
    fn should_score_parallel_accepts_large_or_long_candidate_sets() {
        let texts = vec!["x".repeat(768); MIN_PARALLEL_ITEMS];
        assert!(should_score_parallel(&texts, 8));

        let many_short = vec!["x".repeat(48); MIN_PARALLEL_ITEMS * 16];
        assert!(should_score_parallel(&many_short, 8));
    }

    fn median_duration(mut durations: Vec<Duration>) -> Duration {
        durations.sort_unstable();
        durations[durations.len() / 2]
    }

    fn measure_duration(mut run: impl FnMut()) -> Duration {
        let mut durations = Vec::new();
        for _ in 0..15 {
            let start = Instant::now();
            run();
            durations.push(start.elapsed());
        }
        median_duration(durations)
    }

    #[test]
    #[ignore = "records local timing evidence for the serial/parallel scoring gate"]
    fn benchmark_parallel_scoring_gate_evidence() {
        let pattern = Pattern::parse("abc", CaseMatching::Ignore, Normalization::Smart);
        let short_texts = (0..(MIN_PARALLEL_ITEMS * 2))
            .map(|index| format!("abc-{:05}-{}", index, "x".repeat(36)))
            .collect::<Vec<_>>();
        let medium_texts = (0..MIN_PARALLEL_ITEMS)
            .map(|index| format!("abc-{:05}-{}", index, "x".repeat(500)))
            .collect::<Vec<_>>();
        let large_texts = (0..(MIN_PARALLEL_ITEMS * 32))
            .map(|index| format!("abc-{:05}-{}", index, "x".repeat(132)))
            .collect::<Vec<_>>();
        let pool = matcher_pool();
        let short_workers = parallel_worker_count(short_texts.len(), pool.len());
        let medium_workers = parallel_worker_count(medium_texts.len(), pool.len());
        let large_workers = parallel_worker_count(large_texts.len(), pool.len());

        let short_serial = measure_duration(|| {
            std::hint::black_box(score_items_serial(&pattern, &short_texts));
        });
        let short_parallel = measure_duration(|| {
            std::hint::black_box(score_items_parallel(
                &pattern,
                &short_texts,
                pool,
                short_workers,
            ));
        });
        let medium_serial = measure_duration(|| {
            std::hint::black_box(score_items_serial(&pattern, &medium_texts));
        });
        let medium_parallel = measure_duration(|| {
            std::hint::black_box(score_items_parallel(
                &pattern,
                &medium_texts,
                pool,
                medium_workers,
            ));
        });
        let large_serial = measure_duration(|| {
            std::hint::black_box(score_items_serial(&pattern, &large_texts));
        });
        let large_parallel = measure_duration(|| {
            std::hint::black_box(score_items_parallel(
                &pattern,
                &large_texts,
                pool,
                large_workers,
            ));
        });

        eprintln!(
            "short candidates: items={} bytes={} workers={} serial={:?} parallel={:?} gate={}",
            short_texts.len(),
            short_texts.iter().map(|text| text.len()).sum::<usize>(),
            short_workers,
            short_serial,
            short_parallel,
            should_score_parallel(&short_texts, short_workers)
        );
        eprintln!(
            "medium candidates: items={} bytes={} workers={} serial={:?} parallel={:?} gate={}",
            medium_texts.len(),
            medium_texts.iter().map(|text| text.len()).sum::<usize>(),
            medium_workers,
            medium_serial,
            medium_parallel,
            should_score_parallel(&medium_texts, medium_workers)
        );
        eprintln!(
            "large candidates: items={} bytes={} workers={} serial={:?} parallel={:?} gate={}",
            large_texts.len(),
            large_texts.iter().map(|text| text.len()).sum::<usize>(),
            large_workers,
            large_serial,
            large_parallel,
            should_score_parallel(&large_texts, large_workers)
        );
    }

    #[test]
    fn sort_scored_can_break_ties_by_length() {
        let texts = vec!["alphabet".into(), "alpha".into(), "alpaca".into()];
        let scored = vec![
            ScoredIndex {
                index: 0,
                score: 10,
            },
            ScoredIndex {
                index: 1,
                score: 10,
            },
            ScoredIndex {
                index: 2,
                score: 11,
            },
        ];

        let sorted = sort_scored(
            scored,
            &texts,
            SortOptions {
                ties_by_length: true,
                ties_alphabetically: false,
                ignore_case: false,
            },
            None,
        );

        assert_eq!(
            sorted
                .into_iter()
                .map(|scored| scored.index)
                .collect::<Vec<_>>(),
            vec![2, 1, 0]
        );
    }

    #[test]
    fn sort_scored_can_break_ties_alphabetically() {
        let texts = vec!["beta".into(), "alpha".into(), "aardvark".into()];
        let scored = vec![
            ScoredIndex {
                index: 0,
                score: 10,
            },
            ScoredIndex {
                index: 1,
                score: 10,
            },
            ScoredIndex { index: 2, score: 9 },
        ];

        let sorted = sort_scored(
            scored,
            &texts,
            SortOptions {
                ties_by_length: false,
                ties_alphabetically: true,
                ignore_case: false,
            },
            None,
        );

        assert_eq!(
            sorted
                .into_iter()
                .map(|scored| scored.index)
                .collect::<Vec<_>>(),
            vec![1, 0, 2]
        );
    }

    #[test]
    fn sort_scored_applies_length_before_alphabetical() {
        let texts = vec!["bbb".into(), "aa".into(), "ccc".into(), "aaa".into()];
        let scored = vec![
            ScoredIndex {
                index: 0,
                score: 10,
            },
            ScoredIndex {
                index: 1,
                score: 10,
            },
            ScoredIndex {
                index: 2,
                score: 10,
            },
            ScoredIndex {
                index: 3,
                score: 10,
            },
        ];

        let sorted = sort_scored(
            scored,
            &texts,
            SortOptions {
                ties_by_length: true,
                ties_alphabetically: true,
                ignore_case: false,
            },
            None,
        );

        assert_eq!(
            sorted
                .into_iter()
                .map(|scored| scored.index)
                .collect::<Vec<_>>(),
            vec![1, 3, 0, 2]
        );
    }

    #[test]
    fn sort_scored_applies_history_before_length_and_alphabetical() {
        let texts = vec!["bbb".into(), "aa".into(), "ccc".into(), "aaa".into()];
        let scored = vec![
            ScoredIndex {
                index: 0,
                score: 10,
            },
            ScoredIndex {
                index: 1,
                score: 10,
            },
            ScoredIndex { index: 2, score: 9 },
            ScoredIndex {
                index: 3,
                score: 10,
            },
        ];
        let history_ranks = vec![Some(1), None, Some(0), Some(0)];

        let sorted = sort_scored(
            scored,
            &texts,
            SortOptions {
                ties_by_length: true,
                ties_alphabetically: true,
                ignore_case: false,
            },
            Some(&history_ranks),
        );

        assert_eq!(
            sorted
                .into_iter()
                .map(|scored| scored.index)
                .collect::<Vec<_>>(),
            vec![3, 0, 1, 2]
        );
    }

    #[test]
    fn matched_indices_returns_sorted_unique_positions() {
        let pattern = Pattern::parse("foo", CaseMatching::Respect, Normalization::Smart);
        let indices = with_matcher(|matcher| matched_indices(&pattern, "xfoox", matcher));

        assert_eq!(indices, Some(vec![1, 2, 3]));
    }
}
