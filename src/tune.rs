// Stockcrab, a Rust port of Stockfish, a UCI chess playing engine derived from Glaurung 2.1
// Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
//
// Stockcrab is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Stockcrab is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

//! Tuning-parameter scaffolding mirroring upstream `tune.cpp`.
//!
//! **Unsupported development tooling.** Nothing in the engine registers
//! entries (`Tune::add` is never called outside tests) and the entries are not
//! connected to `OptionsMap`: `init_option` only prints a fishtest-style line
//! and `read_option` is a placeholder. Wiring real SPSA tuning needs mutable
//! parameter storage in the search plus option callbacks, which the port does
//! not have. Do not rely on this module.

use crate::ucioption::OptionsMap;
use std::sync::{Arc, Mutex, OnceLock};

/// A range of integer values `(min, max)`.
pub type Range = (i32, i32);

/// Default range: `[0, 2*v]` if `v > 0`, else `[2*v, 0]`.
///
/// Default Range function, to calculate Option's min-max values
pub const fn default_range(v: i32) -> Range {
    if v > 0 { (0, 2 * v) } else { (2 * v, 0) }
}

/// Encapsulates a range generator or a fixed range.
#[derive(Clone)]
pub struct SetRange {
    fun: Option<Arc<dyn Fn(i32) -> Range + Send + Sync>>,
    fixed: Range,
}

impl SetRange {
    /// Construct from a function or closure.
    ///
    /// Note: stored as a shared trait object to keep `SetRange` clonable.
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(i32) -> Range + Send + Sync + 'static,
    {
        SetRange {
            fun: Some(Arc::new(f)),
            fixed: (0, 0),
        }
    }

    /// Construct from a fixed min/max.
    pub fn new(min: i32, max: i32) -> Self {
        SetRange {
            fun: None,
            fixed: (min, max),
        }
    }

    /// Evaluate the range for value `v`.
    pub fn eval(&self, v: i32) -> Range {
        if let Some(ref f) = self.fun {
            f(v)
        } else {
            self.fixed
        }
    }
}

impl From<Range> for SetRange {
    fn from(r: Range) -> Self {
        SetRange {
            fun: None,
            fixed: r,
        }
    }
}

/// Entry in the tuning list: name, current value, and range.
pub struct TuneEntry {
    pub name: String,
    pub value: i32,
    pub range: SetRange,
}

impl TuneEntry {
    /// Initialize the option (placeholder for OptionsMap integration).
    pub fn init_option(&self) {
        let v = self.value;
        let (mn, mx) = self.range.eval(v);
        let step = (f64::from(mx) - f64::from(mn)) / 20.0;
        // Match C++ tune.cpp formatted line for fishtest sessions.
        // name,value,min,max,step,0.0020
        println!("{},{},{},{},{:.4},0.0020", self.name, v, mn, mx, step);
    }

    /// Read back the option (placeholder for OptionsMap integration).
    pub fn read_option(&self) {
        println!("Read option {}", self.name);
    }
}

/// Singleton tuning manager.
pub struct Tune {
    entries: Vec<TuneEntry>,
    pub update_on_last: bool,
}

impl Tune {
    fn instance() -> &'static Mutex<Tune> {
        static INSTANCE: OnceLock<Mutex<Tune>> = OnceLock::new();
        INSTANCE.get_or_init(|| {
            Mutex::new(Tune {
                entries: Vec::new(),
                update_on_last: false,
            })
        })
    }

    /// Add parameters to tune: names, values, and a range.
    pub fn add<N, V>(names: N, values: V, range: SetRange)
    where
        N: IntoIterator,
        N::Item: AsRef<str>,
        V: IntoIterator<Item = i32>,
    {
        let mut tune = Tune::instance().lock().unwrap();
        for (n, v) in names.into_iter().zip(values) {
            tune.entries.push(TuneEntry {
                name: n.as_ref().to_string(),
                value: v,
                range: range.clone(),
            });
        }
    }

    /// Initialize all tuning options.
    pub fn init(_options: &OptionsMap) {
        // Options are not used: SPSA tuning is not wired to OptionsMap (see module docs).
        let tune = Tune::instance().lock().unwrap();
        for entry in &tune.entries {
            entry.init_option();
        }
    }

    /// Read back options after user input.
    pub fn read() {
        let tune = Tune::instance().lock().unwrap();
        if tune.update_on_last {
            if let Some(last) = tune.entries.last() {
                last.read_option();
            }
        } else {
            for entry in &tune.entries {
                entry.read_option();
            }
        }
    }

    /// Enable or disable deferred updates (mimics UPDATE_ON_LAST in C++).
    pub fn set_update_on_last(enable: bool) {
        let mut tune = Tune::instance().lock().unwrap();
        tune.update_on_last = enable;
    }
}

// ---------------------------------------------------------------------------
// UNIT TESTS
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_range() {
        assert_eq!(default_range(5), (0, 10));
        assert_eq!(default_range(-3), (-6, 0));
    }

    #[test]
    fn test_setrange_fixed() {
        let sr = SetRange::new(2, 8);
        assert_eq!(sr.eval(0), (2, 8));
    }

    #[test]
    fn test_setrange_fun() {
        let sr = SetRange::from_fn(|v| (v - 1, v + 1));
        assert_eq!(sr.eval(10), (9, 11));
    }

    #[test]
    fn test_tune_add_and_init_read() {
        let x = 7;
        Tune::add(["param"], [x], SetRange::new(0, 14));
        let options = OptionsMap::new();
        Tune::init(&options);
        Tune::read();
    }

    #[test]
    fn test_update_on_last() {
        Tune::add(["a", "b"], [1, 2], (0, 4).into());
        Tune::set_update_on_last(true);
        // Should only print once (for last entry). We just exercise the path.
        Tune::read();
        Tune::set_update_on_last(false);
    }
}
