//! Small CLI helpers.

/// The value after `--name`, if present.
pub fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

pub fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// A numeric flag with a default (accepts 1e6-style and 1_000_000 forms).
pub fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    match flag(args, name) {
        None => default,
        Some(v) => {
            let v = v.replace('_', "");
            v.parse::<T>()
                .ok()
                .or_else(|| v.parse::<f64>().ok().and_then(|f| format!("{}", f as u64).parse::<T>().ok()))
                .unwrap_or_else(|| panic!("bad value for {name}: {v}"))
        }
    }
}

/// A comma-separated list of numbers, e.g. "2000,2000,1000".
pub fn list(args: &[String], name: &str) -> Option<Vec<u32>> {
    flag(args, name).map(|v| v.split(',').map(|x| x.trim().replace('_', "").parse().expect("number list")).collect())
}

pub fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}
