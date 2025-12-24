use greeners::GMM;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1000;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // 1. Gerar Instrumentos (Exógenos)
    let mut z1_vec = Vec::new();
    let mut z2_vec = Vec::new();
    let mut u_vec = Vec::new(); // Erro estrutural
    let mut v_vec = Vec::new(); // Erro da forma reduzida

    for _ in 0..n {
        z1_vec.push(normal.sample(&mut rng));
        z2_vec.push(normal.sample(&mut rng));
        u_vec.push(normal.sample(&mut rng));
        v_vec.push(normal.sample(&mut rng));
    }

    // Convertendo para vetores ndarray para facilitar a aritmética
    let z1 = Array1::from(z1_vec.clone());
    let z2 = Array1::from(z2_vec.clone());
    let u = Array1::from(u_vec);
    let v = Array1::from(v_vec);

    // 2. Gerar X Endógeno
    // X depende de Z1 e Z2 (Relevante) e de u (Endógeno)
    // Coeficiente real: 0.5 para cada Z
    let x_endog = &z1 * 0.5 + &z2 * 0.5 + &u * 0.5 + &v * 0.5;

    // 3. Gerar Y
    // y = 2.0 + 1.5*x + u
    let y = 2.0 + &x_endog * 1.5 + &u;

    // --- Montar Matrizes ---

    // Matriz X (Regressores): [1, x]
    let mut x_mat = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x_mat[[i, 0]] = 1.0; // Intercepto
        x_mat[[i, 1]] = x_endog[i]; // X Endógeno
    }

    // Matriz Z (Instrumentos): [1, z1, z2]
    // A constante é instrumento dela mesma.
    let mut z_mat = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        z_mat[[i, 0]] = 1.0;
        z_mat[[i, 1]] = z1_vec[i];
        z_mat[[i, 2]] = z2_vec[i];
    }

    println!("--- GMM Estimation (Two-Step Efficient) ---\n");
    println!("Verdadeiro Beta (Inclinação): 1.5");
    println!("Instrumentos Z: [1, z1, z2] (Sobre-identificado, L=3, K=2)");

    // Rodar GMM
    let gmm_res = GMM::fit(&y, &x_mat, &z_mat)?;

    println!("{}", gmm_res);

    println!("Análise:");
    println!(
        "1. Coeficiente Estimado: {:.4} (Esperado: 1.5)",
        gmm_res.params[1]
    );
    println!("2. J-Statistic P-value:  {:.4}", gmm_res.j_p_value);

    if gmm_res.j_p_value > 0.05 {
        println!("   -> P-valor alto (> 0.05). Não rejeitamos H0.");
        println!("   -> Conclusão: Instrumentos são válidos (ortogonais ao erro).");
    } else {
        println!("   -> P-valor baixo. Rejeitamos H0.");
        println!("   -> Conclusão: Instrumentos inválidos ou modelo mal especificado.");
    }

    Ok(())
}
