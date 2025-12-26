use greeners::VAR;
use ndarray::{Array2};
use ndarray_rand::rand_distr::Normal;
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t_obs = 200;
    let mut rng = rand::thread_rng();
    let norm = Normal::new(0.0, 1.0).unwrap();

    // Simulação VAR(1) Bivariado
    // Y1: PIB, Y2: Investimento
    // y1_t = 0.7*y1_{t-1} + 0.2*y2_{t-1} + u1
    // y2_t = 0.4*y1_{t-1} + 0.5*y2_{t-1} + u2
    
    // Matriz A1 verdadeira:
    // [0.7, 0.2]
    // [0.4, 0.5]
    // Autovalores < 1 para estabilidade.

    let mut data = Array2::<f64>::zeros((t_obs, 2));

    // Inicialização
    data[[0, 0]] = 0.0;
    data[[0, 1]] = 0.0;

    for t in 1..t_obs {
        let u1 = norm.sample(&mut rng);
        let u2 = norm.sample(&mut rng) + 0.5 * u1; // Erros correlacionados!
        
        let y1_prev = data[[t-1, 0]];
        let y2_prev = data[[t-1, 1]];

        data[[t, 0]] = 0.7 * y1_prev + 0.2 * y2_prev + u1;
        data[[t, 1]] = 0.4 * y1_prev + 0.5 * y2_prev + u2;
    }

    println!("--- Vector Autoregression (VAR) Simulation ---");
    println!("System: GDP (Var0) and Investment (Var1)");
    println!("True Structure: Feedback loop (GDP <-> Inv)\n");

    // 1. Estimar VAR(1)
    let var = VAR::fit(&data, 1, Some(vec!["GDP".into(), "Inv".into()]))?;
    println!("{}", var);

    // 2. Coeficientes Estimados
    // params layout: [Intercept, Lag1_GDP, Lag1_Inv]
    // Coluna 0 é equação do GDP, Coluna 1 é equação do Inv
    println!("Estimated Matrix A1:");
    println!("GDP eq:  {:.4} * GDP(-1) + {:.4} * Inv(-1)", var.params[[1, 0]], var.params[[2, 0]]);
    println!("Inv eq:  {:.4} * GDP(-1) + {:.4} * Inv(-1)", var.params[[1, 1]], var.params[[2, 1]]);
    
    println!("\nEsperado:");
    println!("GDP eq:  0.7000 * GDP(-1) + 0.2000 * Inv(-1)");
    println!("Inv eq:  0.4000 * GDP(-1) + 0.5000 * Inv(-1)");

    // 3. Impulse Response Function (IRF)
    println!("\n--- Impulse Response Analysis (Orthogonalized) ---");
    println!("Choque no GDP (Var0) -> Efeito no Investment (Var1)");
    
    let steps = 6;
    let irf = var.irf(steps)?;

    println!("{:<5} | {:<15}", "Time", "Response of Inv");
    println!("{:-^25}", "-");
    for h in 0..steps {
        // [h, effect_var, shock_var] -> [h, 1, 0]
        let response = irf[[h, 1, 0]]; 
        println!("{:<5} | {:>10.4}", h, response);
    }
    println!("(Note: Effect should persist and decay over time)");

    Ok(())
}