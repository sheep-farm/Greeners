use greeners::{ArellanoBond, FixedEffects};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_entities = 500; 
    let t_periods = 10;   
    let n_obs = n_entities * t_periods;
    
    let mut rng = rand::thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();

    let mut y_vec = Vec::new();
    let mut y_lag_vec = Vec::new(); 
    let mut id_vec = Vec::new();
    let mut time_vec = Vec::new();

    let rho_true = 0.8;

    for i in 0..n_entities {
        let alpha_i = norm.sample(&mut rng); 
        let mut y_prev = alpha_i / (1.0 - rho_true) + norm.sample(&mut rng);

        for t in 0..t_periods {
            let e_it = norm.sample(&mut rng);
            let y_curr = rho_true * y_prev + alpha_i + e_it;
            
            y_vec.push(y_curr);
            y_lag_vec.push(y_prev);
            id_vec.push(i as i64);
            time_vec.push(t as i64);

            y_prev = y_curr;
        }
    }

    let y = Array1::from(y_vec);
    let id = Array1::from(id_vec);
    let time = Array1::from(time_vec);
    
    // --- CORREÇÃO AQUI ---
    // Substituímos vec![0.0; n_obs] por números aleatórios
    let x_random: Vec<f64> = (0..n_obs).map(|_| rng.gen()).collect();
    let x_dummy = Array2::from_shape_vec((n_obs, 1), x_random)?;

    println!("--- Dynamic Panel Simulation (AR(1)) ---");
    println!("True Rho: 0.8");
    println!("N: {}, T: {}\n", n_entities, t_periods);

    let y_lag_arr = Array2::from_shape_vec((n_obs, 1), y_lag_vec)?;
    
    // Correção anterior do as_slice() mantida
    let fe = FixedEffects::fit(&y, &y_lag_arr, id.as_slice().unwrap())?;
    println!(">>> Fixed Effects (Biased downwards due to Nickell Bias):");
    println!("Rho Estimado: {:.4} (Esperado < 0.8)", fe.params[0]);

    let ab = ArellanoBond::fit(&y, &x_dummy, &id, &time)?;
    println!("{}", ab);

    println!("Diagnóstico:");
    println!("Fixed Effects deu {:.4} (Viesado)", fe.params[0]);
    println!("Arellano-Bond deu {:.4} (Corrigido)", ab.params[0]);

    Ok(())
}