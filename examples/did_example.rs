use greeners::{CovarianceType, DiffInDiff};
use ndarray::Array1;
use rand::prelude::*;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_per_group = 500; // 500 obs por célula (2000 total)
    let total_n = n_per_group * 4;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 2.0).unwrap(); // Erro com desvio 2

    let mut y_vec = Vec::with_capacity(total_n);
    let mut treated_vec = Vec::with_capacity(total_n);
    let mut post_vec = Vec::with_capacity(total_n);

    // Parâmetros do "Universo"
    let baseline = 10.0; // Constante
    let group_diff = 2.0; // Grupo Tratado já era 2.0 maior antes (Viés de seleção)
    let time_trend = 1.5; // Tendência temporal (Todo mundo cresce 1.5 no pós)
    let att_real = 5.0; // O EFEITO REAL DO TRATAMENTO (Nosso alvo)

    // Gerar 4 grupos
    for _ in 0..n_per_group {
        // 1. Controle Pré (0, 0) -> y = 10
        y_vec.push(baseline + normal.sample(&mut rng));
        treated_vec.push(0.0);
        post_vec.push(0.0);

        // 2. Controle Pós (0, 1) -> y = 10 + 1.5 = 11.5
        y_vec.push(baseline + time_trend + normal.sample(&mut rng));
        treated_vec.push(0.0);
        post_vec.push(1.0);

        // 3. Tratado Pré (1, 0) -> y = 10 + 2.0 = 12.0
        y_vec.push(baseline + group_diff + normal.sample(&mut rng));
        treated_vec.push(1.0);
        post_vec.push(0.0);

        // 4. Tratado Pós (1, 1) -> y = 10 + 2.0 + 1.5 + 5.0(ATT) = 18.5
        y_vec.push(baseline + group_diff + time_trend + att_real + normal.sample(&mut rng));
        treated_vec.push(1.0);
        post_vec.push(1.0);
    }

    let y = Array1::from(y_vec);
    let treated = Array1::from(treated_vec);
    let post = Array1::from(post_vec);

    println!("--- Causal Inference: Difference-in-Differences ---\n");
    println!("Parâmetros Reais:");
    println!("Tendência Temporal (Counterfactual): +1.5");
    println!("Diferença de Grupo (Viés):           +2.0");
    println!("ATT (Efeito do Tratamento):          +5.0\n");

    // Rodar DiD com Erros Robustos (HC1)
    let res = DiffInDiff::fit(&y, &treated, &post, CovarianceType::HC1)?;

    println!("{}", res);

    // Verificação manual simples
    let diff_control = res.control_post_mean - res.control_pre_mean;
    let diff_treated = res.treated_post_mean - res.treated_pre_mean;
    let manual_att = diff_treated - diff_control;

    println!("Cálculo 'Na Mão' (Diff-in-Diff):");
    println!(
        "Delta Tratado ({:.2} - {:.2}) = {:.4}",
        res.treated_post_mean, res.treated_pre_mean, diff_treated
    );
    println!(
        "Delta Controle ({:.2} - {:.2}) = {:.4}",
        res.control_post_mean, res.control_pre_mean, diff_control
    );
    println!(
        "ATT ({:.4} - {:.4}) = {:.4}",
        diff_treated, diff_control, manual_att
    );

    Ok(())
}
