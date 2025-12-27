use greeners::{Equation, ThreeSLS};
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;
use rand::thread_rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1000;

    // --- 1. Gerar Exógenas (Instrumentos) ---
    // Income (Renda) e Cost (Custo)
    let income = Array1::from_iter(
        (0..n).map(|_| 10.0 + Normal::new(0.0, 2.0).unwrap().sample(&mut thread_rng())),
    );
    let cost = Array1::from_iter(
        (0..n).map(|_| 5.0 + Normal::new(0.0, 1.0).unwrap().sample(&mut thread_rng())),
    );

    // --- 2. Gerar Erros Correlacionados ---
    // Correlação entre erro da demanda e erro da oferta
    let mut rng = thread_rng();
    let dist_u = Normal::new(0.0, 1.0).unwrap();
    let common_shock: Array1<f64> = Array1::from_iter((0..n).map(|_| dist_u.sample(&mut rng)));

    // u (Demanda) e v (Oferta) compartilham o common_shock
    let u = Array1::from_iter((0..n).map(|_| dist_u.sample(&mut rng))) + &common_shock * 0.5;
    let v = Array1::from_iter((0..n).map(|_| dist_u.sample(&mut rng))) + &common_shock * 0.5;

    // --- 3. Forma Estrutural (Resolver P e Q) ---
    // Eq Demanda: Q = 100 - 2*P + 0.5*Income + u
    // Eq Oferta:  Q = 10 + 1*P - 1.0*Cost + v
    // Resolvendo o sistema para P e Q (Algebra):
    // 100 - 2P + 0.5Inc + u = 10 + P - Cost + v
    // 3P = 90 + 0.5Inc + Cost + (u - v)
    // P = 30 + 0.166Inc + 0.33Cost + erro_reduzido

    let p = (&income * 0.1666) + (&cost * 0.3333) + 30.0 + (&u - &v) / 3.0;
    let q_demanda = 100.0 - (&p * 2.0) + (&income * 0.5) + &u;

    // Check consistência (Q oferta deve ser igual Q demanda aproximadamente)
    // let q_oferta = 10.0 + (&p * 1.0) - (&cost * 1.0) + &v;

    // --- 4. Montar Matrizes para o Greeners ---

    // Equação 1: Demanda (Q ~ P + Income)
    // X1 = [Intercept, P, Income]
    let mut x1_vec = Vec::new();
    for i in 0..n {
        x1_vec.push(1.0);
        x1_vec.push(p[i]);
        x1_vec.push(income[i]);
    }
    let x1 = Array2::from_shape_vec((n, 3), x1_vec)?;

    let eq1 = Equation {
        name: "Demand Curve".to_string(),
        y: q_demanda.clone(),
        x: x1,
    };

    // Equação 2: Oferta (Q ~ P + Cost)
    // X2 = [Intercept, P, Cost]
    // Note: Usamos o MESMO vetor Q (equilíbrio de mercado)
    let mut x2_vec = Vec::new();
    for i in 0..n {
        x2_vec.push(1.0);
        x2_vec.push(p[i]);
        x2_vec.push(cost[i]);
    }
    let x2 = Array2::from_shape_vec((n, 3), x2_vec)?;

    let eq2 = Equation {
        name: "Supply Curve".to_string(),
        y: q_demanda.clone(), // Q observado
        x: x2,
    };

    // Instrumentos Globais (Z) = [Intercept, Income, Cost]
    // A união de todas as exógenas do sistema.
    let mut z_vec = Vec::new();
    for i in 0..n {
        z_vec.push(1.0);
        z_vec.push(income[i]);
        z_vec.push(cost[i]);
    }
    let z = Array2::from_shape_vec((n, 3), z_vec)?;

    println!("--- Simulando Sistema de Oferta e Demanda ---");
    println!("Coeficientes Reais Demanda: Intercept=100, P=-2.0, Income=0.5");
    println!("Coeficientes Reais Oferta:  Intercept=10,  P=1.0,  Cost=-1.0\n");

    // RODAR 3SLS
    let result = ThreeSLS::fit(&[eq1, eq2], &z)?;

    println!("{}", result);

    Ok(())
}
