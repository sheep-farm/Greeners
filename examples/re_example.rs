use greeners::{RandomEffects};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use statrs::distribution::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_entities = 50;
    let t_periods = 10;
    let n_obs = n_entities * t_periods;
    
    let mut rng = rand::thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();

    // 1. Gerar Efeitos Individuais (Alpha_i)
    // Variância alta (4.0) para justificar Random Effects
    let mut alphas = Vec::new();
    for _ in 0..n_entities {
        alphas.push(norm.sample(&mut rng) * 2.0); 
    }

    // 2. Gerar Dados de Painel
    // y_it = 2.5 * x_it + alpha_i + u_it
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    let mut ids_vec = Vec::new();
    let mut x_flat = Vec::new();

    for i in 0..n_entities {
        let alpha = alphas[i];
        for _ in 0..t_periods {
            let x_val = norm.sample(&mut rng) + 2.0;
            let u_it = norm.sample(&mut rng); // Erro idiossincrático
            
            let y_val = 2.5 * x_val + alpha + u_it;

            ids_vec.push(i as i64);
            y_vec.push(y_val);
            x_vec.push(x_val);
            
            x_flat.push(1.0); // Intercepto
            x_flat.push(x_val);
        }
    }

    let y = Array1::from(y_vec);
    let x = Array2::from_shape_vec((n_obs, 2), x_flat)?;
    let ids = Array1::from(ids_vec);

    println!("--- Random Effects (Swamy-Arora) ---");
    println!("Real Beta: 2.5");
    println!("Real Sigma Alpha: 2.0");
    println!("Real Sigma U: 1.0\n");

    let re = RandomEffects::fit(&y, &x, &ids)?;
    println!("{}", re);

    println!("Diagnóstico:");
    if re.theta > 0.5 {
        println!("Theta alto ({:.4}) indica forte efeito individual.", re.theta);
        println!("O modelo se aproxima do Fixed Effects.");
    } else {
        println!("Theta baixo ({:.4}) indica pouco efeito individual.", re.theta);
        println!("O modelo se aproxima do Pooled OLS.");
    }

    Ok(())
}