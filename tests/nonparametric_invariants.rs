use greeners::{KDEMultivariate, KDEUnivariate, Kernel, KernelReg, LocalLevel, Lowess};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::{rngs::StdRng, SeedableRng};

fn make_local_level_data(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let mut y = vec![noise.sample(&mut rng)];
    for _ in 1..n {
        let prev = y.last().copied().unwrap();
        y.push(prev + noise.sample(&mut rng));
    }
    y
}

/// Univariate and multivariate KDE return finite densities and bandwidths with expected shapes.
#[test]
fn test_kde_invariants() {
    let mut rng = StdRng::seed_from_u64(12345);
    let noise = Normal::new(0.0, 1.0).unwrap();

    let data: Array1<f64> = (0..200).map(|_| noise.sample(&mut rng)).collect();
    let u_kde = KDEUnivariate::fit(&data, None, Kernel::Gaussian).unwrap();
    assert_eq!(u_kde.n_obs, 200);
    assert_eq!(u_kde.support.len(), 512);
    assert_eq!(u_kde.density.len(), 512);
    assert!(u_kde.bandwidth > 0.0);
    assert!(u_kde.density.iter().all(|v| v.is_finite() && *v >= 0.0));

    // Density should integrate approximately to 1 over the support (trapezoidal rule)
    let step = (u_kde.support[u_kde.support.len() - 1] - u_kde.support[0])
        / (u_kde.support.len() - 1) as f64;
    let integral: f64 = u_kde.density.iter().copied().sum::<f64>() * step;
    assert!(
        (integral - 1.0).abs() < 0.15,
        "KDE integral off: {}",
        integral
    );

    let mut m_data = Vec::with_capacity(100 * 2);
    for _ in 0..100 {
        let v = noise.sample(&mut rng);
        m_data.push(v);
        m_data.push(v * 0.7 + noise.sample(&mut rng) * 0.5);
    }
    let m_arr = Array2::from_shape_vec((100, 2), m_data).unwrap();
    let m_kde = KDEMultivariate::fit(&m_arr, None, Kernel::Gaussian).unwrap();
    assert_eq!(m_kde.n_obs, 100);
    assert_eq!(m_kde.n_dims, 2);
    assert_eq!(m_kde.bandwidths.len(), 2);
    assert!(m_kde.bandwidths.iter().all(|&v| v > 0.0 && v.is_finite()));

    // Evaluation at the training points returns finite densities
    let eval = m_kde.evaluate(&m_arr);
    assert_eq!(eval.len(), 100);
    assert!(eval.iter().all(|v| v.is_finite() && *v >= 0.0));
}

/// LOWESS and kernel regression return fitted values that reconstruct the response shape.
#[test]
fn test_lowess_and_kernel_reg_invariants() {
    let mut rng = StdRng::seed_from_u64(23456);
    let noise = Normal::new(0.0, 0.5).unwrap();
    let n = 100;
    let x: Array1<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y: Array1<f64> = x.mapv(|v| v.sin() * 2.0)
        + Array1::from_vec((0..n).map(|_| noise.sample(&mut rng)).collect());

    let lowess = Lowess::fit(&y, &x, 0.3, 2).unwrap();
    assert_eq!(lowess.n_obs, n);
    assert_eq!(lowess.smoothed.len(), n);
    assert_eq!(lowess.residuals.len(), n);
    assert!(lowess.smoothed.iter().all(|v| v.is_finite()));
    assert!(lowess.residuals.iter().all(|v| v.is_finite()));

    let kreg = KernelReg::fit(&y, &x, None, Kernel::Gaussian).unwrap();
    assert_eq!(kreg.n_obs, n);
    assert_eq!(kreg.fitted.len(), n);
    assert_eq!(kreg.residuals.len(), n);
    assert!(kreg.fitted.iter().all(|v| v.is_finite()));
    assert!(kreg.residuals.iter().all(|v| v.is_finite()));
    assert!(kreg.bandwidth > 0.0);
}

/// Local-level state-space model estimates finite variances and state paths.
#[test]
fn test_local_level_invariants() {
    let y = make_local_level_data(34567, 120);
    let result = LocalLevel::fit(&y).unwrap();

    assert_eq!(result.n_obs, 120);
    assert!(result.sigma_obs > 0.0 && result.sigma_obs.is_finite());
    assert!(result.sigma_state > 0.0 && result.sigma_state.is_finite());
    assert!(result.log_likelihood.is_finite());
    assert_eq!(result.filtered_states.len(), 120);
    assert_eq!(result.smoothed_states.len(), 120);
    assert!(result.filtered_states.iter().all(|s| s[0].is_finite()));
    assert!(result.smoothed_states.iter().all(|s| s[0].is_finite()));
}

/// Input validation catches mismatched lengths, too few observations and bad bandwidths.
#[test]
fn test_nonparametric_input_validation() {
    let mut rng = StdRng::seed_from_u64(11111);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let _data: Array1<f64> = (0..10).map(|_| noise.sample(&mut rng)).collect();

    assert!(KDEUnivariate::fit(&Array1::from_vec(vec![1.0]), None, Kernel::Gaussian).is_err());

    let x: Array1<f64> = (0..10).map(|i| i as f64).collect();
    let y: Array1<f64> = (0..9).map(|_| noise.sample(&mut rng)).collect();
    assert!(Lowess::fit(&y, &x, 0.5, 0).is_err());
    assert!(KernelReg::fit(&y, &x, None, Kernel::Uniform).is_err());

    let m_data = Array2::from_elem((5, 2), 1.0);
    let bad_bw = Array1::from_vec(vec![1.0]);
    assert!(KDEMultivariate::fit(&m_data, Some(&bad_bw), Kernel::Gaussian).is_err());

    let short_y = make_local_level_data(22222, 3);
    assert!(LocalLevel::fit(&short_y).is_err());
}
