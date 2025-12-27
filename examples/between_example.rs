use greeners::{BetweenEstimator, FixedEffects};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_entities = 100; // 100 Indivíduos
    let t_periods = 10; // 10 Anos
    let n_obs = n_entities * t_periods;

    let mut rng = rand::thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();

    // Vetores de dados
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    let mut ids_vec = Vec::new();
    let mut x_mat_vec = Vec::new(); // Para Fixed Effects (sem intercepto)
    let mut x_mat_between = Vec::new(); // Para Between (com intercepto)

    // --- GERADOR DE DADOS (O Segredo) ---
    // y_it = 10.0 + 0.5 * x_it + (2.0 * x_base_i) + e_it
    //
    // 1. Beta Within (Temporal): 0.5
    //    Se X varia no tempo, Y varia 0.5.
    //
    // 2. Beta Between (Estrutural): 2.5
    //    Pessoas com X médio alto têm um bônus extra de 2.0 * X_base.
    //    Logo, a relação total nas médias é 0.5 + 2.0 = 2.5.

    for i in 0..n_entities {
        // "Nível Base" do indivíduo (ex: QI, Educação). Fixo no tempo.
        let x_base_i = rng.gen_range(5.0..15.0);

        for _ in 0..t_periods {
            // Variação temporal (ex: horas extras, esforço do ano)
            let x_variation = norm.sample(&mut rng);

            // X observado = Base + Variação
            let x_val = x_base_i + x_variation;

            // Y observado
            let error = norm.sample(&mut rng);
            let y_val = 10.0 + (0.5 * x_val) + (2.0 * x_base_i) + error;

            ids_vec.push(i as i64);
            y_vec.push(y_val);
            x_vec.push(x_val);

            // Montar Matrizes
            x_mat_vec.push(x_val); // Só X para o Fixed Effects

            x_mat_between.push(1.0); // Intercepto
            x_mat_between.push(x_val); // X
        }
    }

    let y = Array1::from(y_vec);
    let ids = Array1::from(ids_vec);

    // Matriz X para Fixed Effects (Sem intercepto, pois FE remove médias)
    let x_fe = Array2::from_shape_vec((n_obs, 1), x_mat_vec)?;

    // Matriz X para Between (Com intercepto, pois é um OLS nas médias)
    let x_be = Array2::from_shape_vec((n_obs, 2), x_mat_between)?;

    println!("--- Comparação: Within vs Between ---");
    println!(
        "Cenário: O 'status' do indivíduo (X médio) afeta Y mais do que a variação de curto prazo."
    );
    println!("Beta Esperado Within (Curto Prazo):  0.5");
    println!("Beta Esperado Between (Longo Prazo): 2.5 (0.5 direto + 2.0 estrutural)\n");

    // 1. Rodar Fixed Effects (Within)
    let fe = FixedEffects::fit(&y, &x_fe, ids.as_slice().unwrap())?;
    println!(">>> FIXED EFFECTS (Within):");
    println!("Beta Estimado: {:.4}", fe.params[0]);
    println!("R2 (Within):   {:.4}", fe.r_squared);

    println!("\n-----------------------------------------------------");

    // 2. Rodar Between Estimator
    let be = BetweenEstimator::fit(&y, &x_be, &ids)?;
    // Nota: O print do BetweenEstimator já formata bonito
    println!("{}", be);

    println!("Conclusão:");
    println!(
        "O Fixed Effects limpou a heterogeneidade e achou o efeito marginal puro ({:.4}).",
        fe.params[0]
    );
    println!(
        "O Between Estimator olhou as médias e achou a relação estrutural de longo prazo ({:.4}).",
        be.params[1]
    );

    Ok(())
}
