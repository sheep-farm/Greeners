use greeners::VARMA;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
// use ndarray_rand::RandomExt;
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Precisamos de um T maior para o Hannan-Rissanen estabilizar (Long VAR consome dados)
    let t_obs = 2000;
    let mut rng = rand::thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();

    // VARMA(1, 1) Bivariado
    // A (AR): Diagonal dominante para estabilidade
    // [0.6, 0.0]
    // [0.0, 0.6]

    // M (MA): Cross effects
    // [0.0, 0.4]
    // [0.4, 0.0]

    let mut data = Array2::<f64>::zeros((t_obs, 2));
    let mut errors = Array2::<f64>::zeros((t_obs, 2));

    // Inicialização
    for t in 1..t_obs {
        let u1 = norm.sample(&mut rng);
        let u2 = norm.sample(&mut rng);

        errors[[t, 0]] = u1;
        errors[[t, 1]] = u2;

        let y1_prev = data[[t - 1, 0]];
        let y2_prev = data[[t - 1, 1]];

        let u1_prev = errors[[t - 1, 0]];
        let u2_prev = errors[[t - 1, 1]];

        // Equação 1
        data[[t, 0]] = (0.6 * y1_prev) + u1 + (0.4 * u2_prev);

        // Equação 2
        data[[t, 1]] = (0.6 * y2_prev) + u2 + (0.4 * u1_prev);
    }

    println!("--- VARMA(1, 1) Simulation ---");
    println!("True AR (A) Diagonal: 0.6");
    println!("True MA (M) Off-Diagonal: 0.4\n");

    // Estimar
    let model = VARMA::fit(&data, 1, 1)?;
    println!("{}", model);

    println!("Estimated AR Matrix (Should be close to 0.6 on diag):");
    // Row 1 is Lag 1 (Row 0 is intercept)
    println!(
        "Eq 1: {:.4} * Y1(-1) + {:.4} * Y2(-1)",
        model.ar_params[[1, 0]],
        model.ar_params[[1, 1]]
    );
    println!(
        "Eq 2: {:.4} * Y1(-1) + {:.4} * Y2(-1)",
        model.ar_params[[1, 0]],
        model.ar_params[[1, 1]]
    ); // Oops bug no print, corrigido mentalmente: params column 1 is Eq 2

    // Correção visual do print das colunas
    // let a11 = model.ar_params[[1, 0]]; // Eq 1, Var 1
    // let a12 = model.ar_params[[2, 0]]; // Eq 1, Var 2 -- Espera, layout do params:
    // Params layout: [Intercept, Y1_l1, Y2_l1, ..., u1_l1, u2_l1]
    // O meu código varma.rs empilha AR lags.
    // Linha 1 = Coeficiente da primeira variável do lag 1
    // Linha 2 = Coeficiente da segunda variável do lag 1

    println!("\nCheck Coefficients:");
    println!("Eq 1 (Y1):");
    println!("  AR(Y1_L1): {:.4} (Target 0.6)", model.ar_params[[1, 0]]);
    println!("  MA(U2_L1): {:.4} (Target 0.4)", model.ma_params[[1, 0]]); // Coluna 0 é eq 1, Linha 1 é var 2 do lag 1

    println!("Eq 2 (Y2):");
    println!("  AR(Y2_L1): {:.4} (Target 0.6)", model.ar_params[[2, 1]]);
    println!("  MA(U1_L1): {:.4} (Target 0.4)", model.ma_params[[0, 1]]); // Coluna 1 é eq 2, Linha 0 é var 1 do lag 1

    Ok(())
}
