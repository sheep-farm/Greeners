use greeners::{PanelThreshold, FixedEffects};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use ndarray_rand::rand_distr::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_entities = 200;
    let t_periods = 20; // T grande ajuda a identificar a quebra
    let n_obs = n_entities * t_periods;
    
    let mut rng = rand::thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();

    let mut y_vec = Vec::new();
    let mut x_vec = Vec::new();
    let mut q_vec = Vec::new(); // Threshold variable (Size)
    let mut id_vec = Vec::new();
    let mut x_mat_flat = Vec::new();

    let gamma_true = 50.0;

    for i in 0..n_entities {
        let alpha_i = norm.sample(&mut rng); 
        
        for _ in 0..t_periods {
            // X: Cash Flow
            let x_val = 1.0 + norm.sample(&mut rng);
            
            // Q: Size (Threshold Variable)
            // Uniforme entre 0 e 100 para garantir busca ampla
            let q_val = rng.gen_range(0.0..100.0);

            // Efeito de X depende de Q
            let beta = if q_val <= gamma_true {
                2.0 // Constraint forte
            } else {
                0.5 // Constraint fraca
            };

            let e_it = norm.sample(&mut rng) * 0.5;
            
            // Modelo
            let y_val = alpha_i + (beta * x_val) + e_it;

            y_vec.push(y_val);
            x_vec.push(x_val);
            q_vec.push(q_val);
            id_vec.push(i as i64);
            
            x_mat_flat.push(x_val);
        }
    }

    let y = Array1::from(y_vec);
    let x = Array2::from_shape_vec((n_obs, 1), x_mat_flat)?;
    let q = Array1::from(q_vec);
    let id = Array1::from(id_vec);

    println!("--- Panel Threshold Model Simulation ---");
    println!("True Threshold (Gamma): 50.0");
    println!("True Beta (Low):  2.0");
    println!("True Beta (High): 0.5\n");

    // 1. Rodar Modelo
    // Note: Isso pode demorar 1-2 segundos dependendo do grid
    let result = PanelThreshold::fit(&y, &x, &q, &id)?;

    println!("{}", result);

    // Comparação com FE Linear Simples (ignorando a quebra)
    let fe_linear = FixedEffects::fit(&y, &x, id.as_slice().unwrap())?;
    println!("Comparação - Fixed Effects Linear (Ignorando a quebra):");
    println!("Beta Estimado (Média): {:.4} (Deveria estar entre 0.5 e 2.0)", fe_linear.params[0]);

    Ok(())
}