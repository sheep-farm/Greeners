use greeners::{CovarianceType, FixedEffects, OLS};
use ndarray::{Array1, Array2, Axis};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simulando 2 Empresas (Entities), 5 Anos cada.
    // Modelo: y = alpha_i + 2.0 * x + erro

    // Empresa 1: Alpha = 10 (Alta performance fixa)
    // Empresa 2: Alpha = -10 (Baixa performance fixa)

    // X aumenta com o tempo
    let x_vals = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, // Emp 1
        1.0, 2.0, 3.0, 4.0, 5.0, // Emp 2
    ];

    // Y = Alpha + 2*X
    let y_vals = vec![
        12.0, 14.0, 16.0, 18.0, 20.0, // Emp 1 (10 + 2*x)
        -8.0, -6.0, -4.0, -2.0, 0.0, // Emp 2 (-10 + 2*x)
    ];

    let ids = vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

    let y = Array1::from(y_vals);
    let x_col = Array1::from(x_vals);
    // Transforma X em Matriz Nx1
    let x = x_col.view().insert_axis(Axis(1)).to_owned();

    println!("Verdadeiro Beta (Inclinação): 2.0");
    println!("Diferença de Nível (Efeito Fixo): Empresa 1 é 20 unidades maior que Empresa 2");

    // 1. Rodar Pooled OLS (Ignora a empresa)
    // O Pooled OLS vai tentar traçar uma reta que passa no meio das duas nuvens de pontos.
    // Provavelmente vai achar que X não explica muito, ou achar um Beta estranho se X fosse correlacionado com Alpha.

    // Precisamos adicionar constante manualmente para o Pooled
    let ones = Array2::ones((10, 1));
    let x_pooled = ndarray::concatenate(Axis(1), &[ones.view(), x.view()])?;

    let pooled_res = OLS::fit(&y, &x_pooled, CovarianceType::NonRobust)?;
    println!("\n--- Pooled OLS (Naive) ---");
    println!("R2: {:.4}", pooled_res.r_squared);
    // O R2 será baixo porque a variância "Entre Empresas" (Between) é enorme e não explicada por X.

    // 2. Rodar Fixed Effects
    // Não passamos constante! O FE remove a constante global e os alphas.
    let fe_res = FixedEffects::fit(&y, &x, &ids)?;

    println!("\n--- Fixed Effects (Within) ---");
    println!("Beta Estimado: {:.4}", fe_res.params[0]);
    println!("{}", fe_res);
    println!(
        "Note como o R2 ('Within') deve ser 1.0 ou muito próximo, pois limpamos o efeito fixo."
    );

    Ok(())
}
