use greeners::VECM;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use rand::prelude::*;

#[test]
fn test_vecm_table_format_is_printed() {
    let t = 200;
    let mut rng = rand::thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut data = Array2::<f64>::zeros((t, 2));

    let mut w = 0.0;
    for tt in 0..t {
        w += norm.sample(&mut rng);
        data[[tt, 0]] = w + norm.sample(&mut rng) * 0.5;
        data[[tt, 1]] = 2.0 * w + norm.sample(&mut rng) * 0.5;
    }

    let model = VECM::fit(&data, 2, 1).unwrap();
    let output = format!("{}", model);

    println!("{}", output);

    assert!(output.contains("VECM (Johansen ML)"));
    assert!(output.contains("Parameters"));
    assert!(output.contains("coef"));
    assert!(output.contains("std err"));
    assert!(output.contains("P>|z|"));
    assert!(output.contains("[0.025"));
    assert!(output.contains("0.975]"));
    assert!(output.contains("beta_1_y1"));
    assert!(output.contains("alpha_1_y1"));
}
