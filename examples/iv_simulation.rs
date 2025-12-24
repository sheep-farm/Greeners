use greeners::{CovarianceType, IV, OLS};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use statrs::distribution::Normal; // Importa Rng e Distribution para .sample() funcionar

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1000;
    let mut rng = rand::thread_rng();

    // 1. Configurar Distribuição Normal (Média 0, Desvio 1)
    let normal = Normal::new(0.0, 1.0).unwrap();

    // 2. Gerar Dados
    let mut z_vec = Vec::with_capacity(n);
    let mut u_vec = Vec::with_capacity(n);
    let mut v_vec = Vec::with_capacity(n);

    for _ in 0..n {
        // Usamos o trait Distribution do rand para amostrar
        z_vec.push(normal.sample(&mut rng));
        u_vec.push(normal.sample(&mut rng));
        v_vec.push(normal.sample(&mut rng));
    }

    let z_array = Array1::from(z_vec);
    let u_array = Array1::from(u_vec);
    let v_array = Array1::from(v_vec);

    // 3. Criar Endogeneidade
    // X depende de Z (relevância) E de u (endogeneidade)
    // Coeficiente real de Z em X = 0.8
    let x_array = 1.0 + &z_array * 0.8 + (&u_array * 0.5 + &v_array * 0.5);

    // 4. Gerar Y (Modelo Verdadeiro)
    // Coeficiente real de X em Y = 3.0
    // Note: 'u_array' aparece aqui e na equação do X acima -> isso causa o viés!
    let y_array = 2.0 + &x_array * 3.0 + &u_array;

    // --- PREPARAR MATRIZES ---

    // Matriz X (Regressores): [1, x]
    let ones = Array2::ones((n, 1));
    // insert_axis(Axis(1)) transforma Array1 (vetor) em Array2 (coluna) para concatenar
    let x_col = x_array.view().insert_axis(Axis(1));
    let x_mat = ndarray::concatenate(Axis(1), &[ones.view(), x_col])?;

    // Matriz Z (Instrumentos): [1, z]
    // O intercepto é instrumento dele mesmo. O Z substitui o X.
    let z_col = z_array.view().insert_axis(Axis(1));
    let z_mat = ndarray::concatenate(Axis(1), &[ones.view(), z_col])?;

    // --- ESTIMAÇÃO ---

    println!("Verdadeiro Beta de X: 3.0");

    // 1. Rodar OLS (Deve ser VIESADO)
    let ols_res = OLS::fit(&y_array, &x_mat, CovarianceType::HC1)?;
    println!("\n--- OLS (Viesado) ---");
    // params[1] é o coeficiente do x (params[0] é o intercepto)
    println!(
        "Estimado Beta de X: {:.4} (Deveria ser 3.0)",
        ols_res.params[1]
    );
    println!("O OLS superestima porque X e erro são correlacionados positivamente.");

    // 2. Rodar IV (Deve ser CONSISTENTE)
    let iv_res = IV::fit(&y_array, &x_mat, &z_mat, CovarianceType::HC1)?;
    println!("\n--- IV / 2SLS (Consistente) ---");
    println!(
        "Estimado Beta de X: {:.4} (Recuperou o valor real!)",
        iv_res.params[1]
    );

    println!("{}", iv_res);

    Ok(())
}
