use greeners::Hawkes;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn sorted_event_times(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = make_rng(seed);
    let mut times = Vec::with_capacity(n);
    let mut t = 0.0;
    for _ in 0..n {
        t += rng.gen_range(0.05..0.5);
        times.push(t);
    }
    times
}

/// Hawkes fit returns finite parameters and expected shapes on sorted event times.
#[test]
fn test_hawkes_finite_and_shapes() {
    let events = sorted_event_times(40, 4001);
    let r = Hawkes::fit(&events, Some(30.0)).unwrap();
    assert_eq!(r.n_events, events.len());
    assert_eq!(r.intensity_at_events.len(), events.len());
    assert!(r.mu.is_finite());
    assert!(r.alpha.is_finite());
    assert!(r.beta.is_finite());
    assert!(r.branching_ratio.is_finite());
    assert!(r.log_likelihood.is_finite());
    assert!(r.aic.is_finite());
    assert!(r.bic.is_finite());
    assert!(r.branching_ratio < 0.95);
    assert!(r.beta > 0.0);
    assert!(r.time_window > 0.0);
}

/// When no time_window is supplied, it defaults to the last event time.
#[test]
fn test_hawkes_default_time_window() {
    let events = sorted_event_times(20, 4002);
    let r = Hawkes::fit(&events, None).unwrap();
    assert!((r.time_window - events.last().unwrap()).abs() < 1e-10);
}

/// Input validation rejects too few events, unsorted times, and non-positive windows.
#[test]
fn test_hawkes_input_validation() {
    let short = sorted_event_times(4, 4003);
    assert!(Hawkes::fit(&short, None).is_err());

    let mut unsorted = sorted_event_times(20, 4003);
    unsorted[10] = unsorted[9] - 1.0;
    assert!(Hawkes::fit(&unsorted, None).is_err());

    assert!(Hawkes::fit(&sorted_event_times(20, 4004), Some(-1.0)).is_err());
}
