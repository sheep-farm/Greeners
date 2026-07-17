use crate::distributions::{chi2_pvalue, f_pvalue, norm_pdf, t_pvalue_two, t_quantile};
use crate::error::GreenersError;

/// Pure element-wise transforms on numeric slices.
///
/// These functions apply scalar operations to vectors, supporting 1-arg, 2-arg,
/// and 3-arg variants. The interpreter dispatches to these instead of computing inline.
pub struct Transforms;

impl Transforms {
    // ── Column aggregate functions ──────────────────────────────────────────

    /// Rank with average ties, NaN values placed at end.
    pub fn rank(vals: &[f64]) -> Vec<f64> {
        let n = vals.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| match (vals[a].is_nan(), vals[b].is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => vals[a].partial_cmp(&vals[b]).unwrap(),
        });
        let mut ranks = vec![0.0f64; n];
        let mut i = 0;
        while i < n {
            if vals[order[i]].is_nan() {
                for k in i..n {
                    ranks[order[k]] = f64::NAN;
                }
                break;
            }
            let mut j = i;
            while j < n
                && !vals[order[j]].is_nan()
                && (vals[order[j]] - vals[order[i]]).abs() < 1e-10
            {
                j += 1;
            }
            let avg = ((i + 1) as f64 + j as f64) / 2.0;
            for k in i..j {
                ranks[order[k]] = avg;
            }
            i = j;
        }
        ranks
    }

    /// Cumulative sum.
    pub fn cumsum(vals: &[f64]) -> Vec<f64> {
        let mut s = 0.0f64;
        vals.iter()
            .map(|&v| {
                s += v;
                s
            })
            .collect()
    }

    /// Z-score standardization (mean=0, sd=1). NaN-aware.
    pub fn standardize(vals: &[f64]) -> Vec<f64> {
        let clean: Vec<f64> = vals.iter().filter(|v| v.is_finite()).copied().collect();
        let n = clean.len() as f64;
        if n < 2.0 {
            return vals.iter().map(|_| f64::NAN).collect();
        }
        let mean = clean.iter().sum::<f64>() / n;
        let sd = (clean.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        vals.iter()
            .map(|&v| {
                if v.is_finite() && sd > 1e-15 {
                    (v - mean) / sd
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Interquartile range (Q75 - Q25).
    pub fn iqr(vals: &[f64]) -> f64 {
        let mut sorted: Vec<f64> = vals.iter().filter(|v| v.is_finite()).copied().collect();
        if sorted.is_empty() {
            return f64::NAN;
        }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let q25 = sorted[(0.25 * (n - 1) as f64).round() as usize];
        let q75 = sorted[(0.75 * (n - 1) as f64).round() as usize];
        q75 - q25
    }

    /// Assign unique group IDs (1-based) to each distinct value in a string slice.
    /// Values are sorted (numerically if all parseable, otherwise lexicographically).
    pub fn group(strs: &[String]) -> Vec<f64> {
        let mut unique: Vec<String> = strs
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        if unique.iter().all(|s| s.parse::<f64>().is_ok()) {
            unique.sort_by(|a, b| {
                a.parse::<f64>()
                    .unwrap()
                    .partial_cmp(&b.parse::<f64>().unwrap())
                    .unwrap()
            });
        } else {
            unique.sort();
        }
        let lookup: std::collections::HashMap<String, f64> = unique
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, (i + 1) as f64))
            .collect();
        strs.iter()
            .map(|v| *lookup.get(v).unwrap_or(&f64::NAN))
            .collect()
    }

    // ── Row-wise multi-column functions ─────────────────────────────────────

    /// Mean across columns for each row. Returns NaN if any value is non-finite.
    pub fn row_mean(cols: &[Vec<f64>]) -> Vec<f64> {
        if cols.is_empty() {
            return Vec::new();
        }
        let n = cols[0].len();
        let k = cols.len() as f64;
        (0..n)
            .map(|i| {
                let row: Vec<f64> = cols.iter().map(|c| c[i]).collect();
                if row.iter().any(|v| !v.is_finite()) {
                    f64::NAN
                } else {
                    row.iter().sum::<f64>() / k
                }
            })
            .collect()
    }

    /// Sum across columns for each row. Returns NaN if any value is non-finite.
    pub fn row_sum(cols: &[Vec<f64>]) -> Vec<f64> {
        if cols.is_empty() {
            return Vec::new();
        }
        let n = cols[0].len();
        (0..n)
            .map(|i| {
                let row: Vec<f64> = cols.iter().map(|c| c[i]).collect();
                if row.iter().any(|v| !v.is_finite()) {
                    f64::NAN
                } else {
                    row.iter().sum::<f64>()
                }
            })
            .collect()
    }

    /// Min across columns for each row. Returns NaN if any value is non-finite.
    pub fn row_min(cols: &[Vec<f64>]) -> Vec<f64> {
        if cols.is_empty() {
            return Vec::new();
        }
        let n = cols[0].len();
        (0..n)
            .map(|i| {
                let row: Vec<f64> = cols.iter().map(|c| c[i]).collect();
                if row.iter().any(|v| !v.is_finite()) {
                    f64::NAN
                } else {
                    row.iter().cloned().fold(f64::INFINITY, f64::min)
                }
            })
            .collect()
    }

    /// Max across columns for each row. Returns NaN if any value is non-finite.
    pub fn row_max(cols: &[Vec<f64>]) -> Vec<f64> {
        if cols.is_empty() {
            return Vec::new();
        }
        let n = cols[0].len();
        (0..n)
            .map(|i| {
                let row: Vec<f64> = cols.iter().map(|c| c[i]).collect();
                if row.iter().any(|v| !v.is_finite()) {
                    f64::NAN
                } else {
                    row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                }
            })
            .collect()
    }

    /// Sum across columns for each row, treating NaN as 0.
    pub fn row_total(cols: &[Vec<f64>]) -> Vec<f64> {
        if cols.is_empty() {
            return Vec::new();
        }
        let n = cols[0].len();
        (0..n)
            .map(|i| {
                cols.iter()
                    .map(|c| if c[i].is_finite() { c[i] } else { 0.0 })
                    .sum::<f64>()
            })
            .collect()
    }

    /// Count of NaN/non-finite values per row.
    pub fn row_miss(cols: &[Vec<f64>]) -> Vec<f64> {
        if cols.is_empty() {
            return Vec::new();
        }
        let n = cols[0].len();
        (0..n)
            .map(|i| cols.iter().filter(|c| !c[i].is_finite()).count() as f64)
            .collect()
    }

    // ── Random generators ───────────────────────────────────────────────────

    /// Generate `n` uniform random values in [0, 1).
    pub fn uniform(n: usize) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen::<f64>()).collect()
    }

    /// Generate `n` standard normal random values.
    pub fn rnormal(n: usize) -> Vec<f64> {
        use rand::distributions::Distribution;
        let mut rng = rand::thread_rng();
        let normal = rand::distributions::Standard;
        (0..n)
            .map(|_| {
                let v: f64 = normal.sample(&mut rng);
                v
            })
            .collect()
    }

    /// Generate `n` Bernoulli random values with probability `p`.
    pub fn rbernoulli(n: usize, p: f64) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| if rng.gen::<f64>() < p { 1.0 } else { 0.0 })
            .collect()
    }

    // ── Element-wise scalar functions (1-arg) ───────────────────────────────

    /// Apply a named unary function element-wise to a slice.
    pub fn apply(vals: &[f64], func: &str) -> Result<Vec<f64>, GreenersError> {
        let f: fn(f64) -> f64 = match func {
            "log" | "ln" => f64::ln,
            "log2" => f64::log2,
            "log10" => f64::log10,
            "exp" => f64::exp,
            "sqrt" => f64::sqrt,
            "abs" => f64::abs,
            "floor" => f64::floor,
            "ceil" => f64::ceil,
            "round" => f64::round,
            "sin" => f64::sin,
            "cos" => f64::cos,
            "tan" => f64::tan,
            "asin" => f64::asin,
            "acos" => f64::acos,
            "atan" => f64::atan,
            "sign" | "signum" => f64::signum,
            "factorial" => |x: f64| {
                let n = x as u64;
                (1..=n).product::<u64>() as f64
            },
            "normal" | "normalden" => norm_pdf,
            "invnormal" | "qnorm" => |p: f64| t_quantile(p, 1e12),
            _ => {
                return Err(GreenersError::InvalidOperation(format!(
                    "unknown unary transform '{func}'"
                )));
            }
        };
        Ok(vals.iter().map(|&v| f(v)).collect())
    }

    // ── Element-wise scalar functions (2-arg) ───────────────────────────────

    /// Apply a named binary function element-wise to two slices.
    pub fn apply2(a: &[f64], b: &[f64], func: &str) -> Result<Vec<f64>, GreenersError> {
        if func == "round" {
            if b.len() != 1 && b.len() != a.len() {
                return Err(GreenersError::ShapeMismatch(format!(
                    "round(x, digits): digits must be scalar or same length as x ({} vs {})",
                    a.len(),
                    b.len()
                )));
            }
            let digits: Vec<f64> = if b.len() == 1 {
                vec![b[0]; a.len()]
            } else {
                b.to_vec()
            };
            return Ok(a
                .iter()
                .zip(digits.iter())
                .map(|(&x, &d)| {
                    if d.is_infinite() || d.is_nan() {
                        return x;
                    }
                    let mult = 10.0_f64.powf(d);
                    (x * mult).round() / mult
                })
                .collect());
        }
        if a.len() != b.len() {
            return Err(GreenersError::ShapeMismatch(format!(
                "apply2: mismatched lengths {} vs {}",
                a.len(),
                b.len()
            )));
        }
        let f: fn(f64, f64) -> f64 = match func {
            "pow" => f64::powf,
            "mod" | "fmod" => f64::rem_euclid,
            "atan2" => f64::atan2,
            "max" => f64::max,
            "min" => f64::min,
            "comb" => |n: f64, k: f64| {
                let (n, k) = (n as u64, k as u64);
                if k > n {
                    return 0.0;
                }
                let k = k.min(n - k);
                (1..=k).fold(1u64, |acc, i| acc * (n - k + i) / i) as f64
            },
            "ttail" => |df_v: f64, x: f64| 1.0 - t_pvalue_two(x, df_v) / 2.0,
            "invttail" => |df_v: f64, p: f64| t_quantile(1.0 - p, df_v),
            "chi2tail" => |df_v: f64, x: f64| chi2_pvalue(x, df_v),
            "Ftail" | "ftail" => |df1: f64, x: f64| f_pvalue(x, df1, 1000.0),
            _ => {
                return Err(GreenersError::InvalidOperation(format!(
                    "unknown binary transform '{func}'"
                )));
            }
        };
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect())
    }

    // ── Element-wise scalar functions (3-arg) ───────────────────────────────

    /// Apply a named ternary function element-wise to three slices.
    pub fn apply3(a: &[f64], b: &[f64], c: &[f64], func: &str) -> Result<Vec<f64>, GreenersError> {
        if a.len() != b.len() || b.len() != c.len() {
            return Err(GreenersError::ShapeMismatch(format!(
                "apply3: mismatched lengths {}, {}, {}",
                a.len(),
                b.len(),
                c.len()
            )));
        }
        match func {
            "cond" => Ok(a
                .iter()
                .zip(b.iter().zip(c.iter()))
                .map(|(&cond, (&t, &f))| if cond != 0.0 { t } else { f })
                .collect()),
            "Ftail" | "ftail" => Ok(a
                .iter()
                .zip(b.iter().zip(c.iter()))
                .map(|(&df1, (&df2, &x))| f_pvalue(x, df1, df2))
                .collect()),
            "binomial" | "binomialp" => Ok(a
                .iter()
                .zip(b.iter().zip(c.iter()))
                .map(|(&n, (&k, &p))| {
                    let (n, k) = (n as u64, k as u64);
                    if k > n {
                        return 0.0;
                    }
                    let comb = {
                        let kk = k.min(n - k);
                        (1..=kk).fold(1u64, |acc, i| acc * (n - kk + i) / i) as f64
                    };
                    comb * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
                })
                .collect()),
            _ => Err(GreenersError::InvalidOperation(format!(
                "unknown ternary transform '{func}'"
            ))),
        }
    }

    // ── Regex ────────────────────────────────────────────────────────────────

    /// `regexm(s, pattern)` — true if pattern matches anywhere in s (Stata: `regexm()`)
    pub fn regexm(s: &str, pattern: &str) -> bool {
        regex::Regex::new(pattern)
            .map(|re| re.is_match(s))
            .unwrap_or(false)
    }

    /// `regexr(s, pattern, replacement)` — replace first match (Stata: `regexr()`)
    pub fn regexr(s: &str, pattern: &str, replacement: &str) -> String {
        regex::Regex::new(pattern)
            .map(|re| re.replace(s, replacement).into_owned())
            .unwrap_or_else(|_| s.to_string())
    }

    /// `regexra(s, pattern, replacement)` — replace all matches
    pub fn regexra(s: &str, pattern: &str, replacement: &str) -> String {
        regex::Regex::new(pattern)
            .map(|re| re.replace_all(s, replacement).into_owned())
            .unwrap_or_else(|_| s.to_string())
    }

    /// `regexs(s, pattern)` — extract first capture group (or full match if no group)
    pub fn regexs(s: &str, pattern: &str) -> Option<String> {
        regex::Regex::new(pattern).ok().and_then(|re| {
            re.captures(s).map(|caps| {
                caps.get(1)
                    .or_else(|| caps.get(0))
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default()
            })
        })
    }

    /// Apply regex match to a vector of strings, returning 1.0/0.0
    pub fn regexm_vec(strings: &[String], pattern: &str) -> Vec<f64> {
        let re = regex::Regex::new(pattern).ok();
        strings
            .iter()
            .map(|s| {
                re.as_ref()
                    .map(|r| if r.is_match(s) { 1.0 } else { 0.0 })
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Apply regex replace to a vector of strings
    pub fn regexr_vec(strings: &[String], pattern: &str, replacement: &str) -> Vec<String> {
        let re = regex::Regex::new(pattern).ok();
        strings
            .iter()
            .map(|s| {
                re.as_ref()
                    .map(|r| r.replace(s, replacement).into_owned())
                    .unwrap_or_else(|| s.clone())
            })
            .collect()
    }

    /// Apply regex extract to a vector of strings
    pub fn regexs_vec(strings: &[String], pattern: &str) -> Vec<String> {
        let re = regex::Regex::new(pattern).ok();
        strings
            .iter()
            .map(|s| {
                re.as_ref()
                    .and_then(|r| {
                        r.captures(s).and_then(|caps| {
                            caps.get(1)
                                .or_else(|| caps.get(0))
                                .map(|m| m.as_str().to_string())
                        })
                    })
                    .unwrap_or_default()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_basic() {
        let vals = vec![3.0, 1.0, 2.0];
        let r = Transforms::rank(&vals);
        assert_eq!(r, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rank_ties() {
        let vals = vec![1.0, 2.0, 2.0, 3.0];
        let r = Transforms::rank(&vals);
        assert_eq!(r, vec![1.0, 2.5, 2.5, 4.0]);
    }

    #[test]
    fn test_rank_nan() {
        let vals = vec![3.0, f64::NAN, 1.0];
        let r = Transforms::rank(&vals);
        assert_eq!(r[0], 2.0);
        assert!(r[1].is_nan());
        assert_eq!(r[2], 1.0);
    }

    #[test]
    fn test_cumsum() {
        let vals = vec![1.0, 2.0, 3.0];
        assert_eq!(Transforms::cumsum(&vals), vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_standardize() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let z = Transforms::standardize(&vals);
        let mean: f64 = z.iter().sum::<f64>() / z.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_iqr() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = Transforms::iqr(&vals);
        assert!(result > 0.0);
    }

    #[test]
    fn test_group() {
        let strs = vec![
            "B".to_string(),
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
        ];
        let g = Transforms::group(&strs);
        assert_eq!(g, vec![2.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_row_mean() {
        let cols = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(Transforms::row_mean(&cols), vec![2.0, 3.0]);
    }

    #[test]
    fn test_row_sum() {
        let cols = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(Transforms::row_sum(&cols), vec![4.0, 6.0]);
    }

    #[test]
    fn test_row_total_nan() {
        let cols = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
        assert_eq!(Transforms::row_total(&cols), vec![4.0, 4.0]);
    }

    #[test]
    fn test_row_miss() {
        let cols = vec![vec![1.0, f64::NAN], vec![f64::NAN, 4.0]];
        assert_eq!(Transforms::row_miss(&cols), vec![1.0, 1.0]);
    }

    #[test]
    fn test_apply_log() {
        let vals = vec![1.0, std::f64::consts::E];
        let result = Transforms::apply(&vals, "ln").unwrap();
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_abs() {
        let vals = vec![-3.0, 2.0, -1.0];
        let result = Transforms::apply(&vals, "abs").unwrap();
        assert_eq!(result, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_apply_unknown() {
        let vals = vec![1.0];
        assert!(Transforms::apply(&vals, "nonexistent").is_err());
    }

    #[test]
    fn test_apply2_pow() {
        let a = vec![2.0, 3.0];
        let b = vec![3.0, 2.0];
        let result = Transforms::apply2(&a, &b, "pow").unwrap();
        assert!((result[0] - 8.0).abs() < 1e-10);
        assert!((result[1] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply2_mismatched() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(Transforms::apply2(&a, &b, "pow").is_err());
    }

    #[test]
    fn test_apply2_round() {
        let a = vec![3.14159, 1234.5678, 1.2345];
        let b = vec![2.0, -2.0, 3.0];
        let result = Transforms::apply2(&a, &b, "round").unwrap();
        assert!((result[0] - 3.14).abs() < 1e-10);
        assert!((result[1] - 1200.0).abs() < 1e-10);
        assert!((result[2] - 1.235).abs() < 1e-10);

        let scalar = vec![2.0];
        let result = Transforms::apply2(&a, &scalar, "round").unwrap();
        assert!((result[0] - 3.14).abs() < 1e-10);
        assert!((result[1] - 1234.57).abs() < 1e-10);
        assert!((result[2] - 1.23).abs() < 1e-10);
    }

    #[test]
    fn test_apply3_cond() {
        let cond = vec![1.0, 0.0, 1.0];
        let t = vec![10.0, 20.0, 30.0];
        let f = vec![100.0, 200.0, 300.0];
        let result = Transforms::apply3(&cond, &t, &f, "cond").unwrap();
        assert_eq!(result, vec![10.0, 200.0, 30.0]);
    }

    #[test]
    fn test_uniform_length() {
        let v = Transforms::uniform(100);
        assert_eq!(v.len(), 100);
        assert!(v.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_rnormal_length() {
        let v = Transforms::rnormal(100);
        assert_eq!(v.len(), 100);
    }

    #[test]
    fn test_rbernoulli() {
        let v = Transforms::rbernoulli(1000, 0.5);
        assert_eq!(v.len(), 1000);
        assert!(v.iter().all(|&x| x == 0.0 || x == 1.0));
    }

    #[test]
    fn test_apply2_comb() {
        let a = vec![5.0, 10.0];
        let b = vec![2.0, 3.0];
        let result = Transforms::apply2(&a, &b, "comb").unwrap();
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply3_binomial() {
        let n = vec![5.0];
        let k = vec![2.0];
        let p = vec![0.5];
        let result = Transforms::apply3(&n, &k, &p, "binomial").unwrap();
        // C(5,2) * 0.5^2 * 0.5^3 = 10 * 0.03125 = 0.3125
        assert!((result[0] - 0.3125).abs() < 1e-10);
    }

    #[test]
    fn test_row_min_max() {
        let cols = vec![vec![3.0, 1.0], vec![1.0, 5.0], vec![2.0, 3.0]];
        assert_eq!(Transforms::row_min(&cols), vec![1.0, 1.0]);
        assert_eq!(Transforms::row_max(&cols), vec![3.0, 5.0]);
    }

    #[test]
    fn test_regexm() {
        assert!(Transforms::regexm("hello world", "wor"));
        assert!(!Transforms::regexm("hello world", "^wor"));
        assert!(Transforms::regexm("abc123", r"\d+"));
    }

    #[test]
    fn test_regexr() {
        assert_eq!(
            Transforms::regexr("hello world", "world", "rust"),
            "hello rust"
        );
        assert_eq!(Transforms::regexr("aaa bbb aaa", "aaa", "x"), "x bbb aaa");
    }

    #[test]
    fn test_regexra() {
        assert_eq!(Transforms::regexra("aaa bbb aaa", "aaa", "x"), "x bbb x");
    }

    #[test]
    fn test_regexs() {
        assert_eq!(
            Transforms::regexs("price: $42.50", r"\$(\d+\.\d+)"),
            Some("42.50".to_string())
        );
        assert_eq!(Transforms::regexs("no match", r"\d+"), None);
    }
}
