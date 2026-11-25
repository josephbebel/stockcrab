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

//! UCI option handling – Rust port of C++ Option and OptionsMap
//!
//! Case-insensitive option names, default/current values, bounds checking,
//! callbacks on change, and UCI-style printing.

use core::fmt;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Callback invoked when an option changes. Return `Some(info)` to notify listener.
/// Must be Send + Sync for thread safety
pub type OnChange = Arc<dyn Fn(&UciOption) -> Option<String> + Send + Sync>;

#[derive(Clone, PartialEq, Eq, Debug)]
enum OptionType {
    Check,
    Spin,
    Combo(Vec<String>),
    String,
    Button,
}

/// Single UCI option: default/current values, type, bounds, and callback.
pub struct UciOption {
    pub name: String,
    default: String,
    current: String,
    opt_type: OptionType,
    min: i32,
    max: i32,
    pub idx: usize,
    on_change: Option<OnChange>,
}

impl UciOption {
    /// Boolean (check) option
    pub fn new_check(name: &str, v: bool, cb: Option<OnChange>) -> Self {
        let val = if v { "true" } else { "false" }.to_string();
        UciOption {
            name: name.to_string(),
            default: val.clone(),
            current: val,
            opt_type: OptionType::Check,
            min: 0,
            max: 0,
            idx: 0,
            on_change: cb,
        }
    }

    /// Spin (integer) option with bounds
    pub fn new_spin(name: &str, v: i32, min: i32, max: i32, cb: Option<OnChange>) -> Self {
        let val = v.to_string();
        UciOption {
            name: name.to_string(),
            default: val.clone(),
            current: val,
            opt_type: OptionType::Spin,
            min,
            max,
            idx: 0,
            on_change: cb,
        }
    }

    /// Combo option: stores the default value as a space-separated string like C++
    /// The current value is set to cur.
    pub fn new_combo(name: &str, default_values: &str, cur: &str, cb: Option<OnChange>) -> Self {
        // Parse the default value string to get the combo options
        let values: Vec<String> = default_values
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        UciOption {
            name: name.to_string(),
            default: default_values.to_string(),
            current: cur.to_string(),
            opt_type: OptionType::Combo(values),
            min: 0,
            max: 0,
            idx: 0,
            on_change: cb,
        }
    }

    /// String option
    pub fn new_string(name: &str, v: &str, cb: Option<OnChange>) -> Self {
        let val = v.to_string();
        UciOption {
            name: name.to_string(),
            default: val.clone(),
            current: val,
            opt_type: OptionType::String,
            min: 0,
            max: 0,
            idx: 0,
            on_change: cb,
        }
    }

    /// Button option (no stored state)
    pub fn new_button(name: &str, cb: Option<OnChange>) -> Self {
        UciOption {
            name: name.to_string(),
            default: String::new(),
            current: String::new(),
            opt_type: OptionType::Button,
            min: 0,
            max: 0,
            idx: 0,
            on_change: cb,
        }
    }

    /// Assign a new value; return optional info message from callback.
    /// Matches C++ operator= behavior.
    ///
    /// Returns None if the value was rejected or the callback produced no message;
    /// Some(text) (possibly empty) means the value was accepted, and the text is
    /// forwarded as an info string.
    pub fn set(&mut self, v: &str) -> Option<String> {
        self.assign(v).flatten()
    }

    /// Outer `None` means rejection; an accepted callback may return no info.
    fn assign(&mut self, v: &str) -> Option<Option<String>> {
        // Validate based on type
        match &self.opt_type {
            OptionType::Check => {
                if v != "true" && v != "false" {
                    return None;
                }
                self.current = v.to_string();
            }
            OptionType::Spin => {
                // Validate via value_in_range (integer parse with range/overflow
                // checks).
                if !value_in_range(v, self.min, self.max) {
                    return None;
                }
                self.current = v.to_string();
            }
            OptionType::Combo(valid_options) => {
                // Use the already-parsed valid options from the enum variant
                if !valid_options.iter().any(|opt| case_insensitive_eq(opt, v)) {
                    return None;
                }
                self.current = v.to_string();
            }
            OptionType::String => {
                self.current = if v == "<empty>" {
                    String::new()
                } else {
                    v.to_string()
                };
            }
            OptionType::Button => {
                // Button type doesn't store state but still processes callbacks
            }
        }

        // Invoke callback if present and return message
        if let Some(cb) = &self.on_change {
            let info = cb(self);
            return Some(info);
        }
        Some(Some(String::new()))
    }

    /// Retrieve integer value for check/spin options
    pub fn get_int(&self) -> Option<i32> {
        match self.opt_type {
            OptionType::Check => Some(if self.current == "true" { 1 } else { 0 }),
            OptionType::Spin => self.current.parse().ok(),
            _ => None,
        }
    }

    /// Retrieve string value for string options
    pub fn get_str(&self) -> Option<&str> {
        if let OptionType::String = self.opt_type {
            Some(&self.current)
        } else {
            None
        }
    }

    /// Get the current value as a string
    pub fn value(&self) -> &str {
        &self.current
    }

    /// Check if the option is a boolean set to true
    pub fn is_true(&self) -> bool {
        matches!(self.opt_type, OptionType::Check) && self.current == "true"
    }

    /// Cast to integer (implements C++ operator int())
    pub fn as_int(&self) -> i32 {
        match self.opt_type {
            OptionType::Check => {
                if self.current == "true" {
                    1
                } else {
                    0
                }
            }
            OptionType::Spin => self.current.parse().unwrap_or(0),
            _ => panic!("Option type is not check or spin"),
        }
    }

    /// Cast to string (implements C++ operator std::string())
    pub fn as_string(&self) -> &str {
        assert!(
            matches!(self.opt_type, OptionType::String),
            "Option type is not string"
        );
        &self.current
    }

    /// Get minimum value for spin options
    pub fn get_min(&self) -> Option<i32> {
        if matches!(self.opt_type, OptionType::Spin) {
            Some(self.min)
        } else {
            None
        }
    }

    /// Get maximum value for spin options
    pub fn get_max(&self) -> Option<i32> {
        if matches!(self.opt_type, OptionType::Spin) {
            Some(self.max)
        } else {
            None
        }
    }

    /// Equality check for combo options (implements C++ operator==)
    pub fn equals_combo(&self, s: &str) -> bool {
        assert!(
            matches!(self.opt_type, OptionType::Combo(_)),
            "Option type is not combo"
        );
        case_insensitive_eq(&self.current, s)
    }
}

/// Case-insensitive string comparison
fn case_insensitive_eq(s1: &str, s2: &str) -> bool {
    s1.eq_ignore_ascii_case(s2)
}

/// Validate that `v` parses as a base-10 integer within `[min, max]`.
///
/// Mirrors C++ `value_in_range`: rejects empty strings, trailing
/// garbage, and out-of-range/overflowing values. Rust's `i64::parse`
/// rejects trailing characters and reports overflow as an error, matching
/// the C++ `*end != '\0'` and `errno == ERANGE` checks respectively.
fn value_in_range(v: &str, min: i32, max: i32) -> bool {
    if v.is_empty() {
        return false;
    }
    match v.parse::<i64>() {
        Ok(result) => result >= min as i64 && result <= max as i64,
        Err(_) => false,
    }
}

/// Container of named options with case-insensitive keys and insertion order.
pub struct OptionsMap {
    options: Vec<UciOption>,
    lookup: BTreeMap<String, usize>, // lowercase name -> index in `options`
    on_info: Option<Arc<dyn Fn(Option<String>) + Send + Sync>>,
    insert_order: usize, // Tracks insertion order like C++ static variable
}

impl OptionsMap {
    pub fn new() -> Self {
        OptionsMap {
            options: Vec::new(),
            lookup: BTreeMap::new(),
            on_info: None,
            insert_order: 0,
        }
    }

    /// Create with a specified info listener
    pub fn with_listener<F>(listener: F) -> Self
    where
        F: Fn(Option<String>) + Send + Sync + 'static,
    {
        let mut map = Self::new();
        map.on_info = Some(Arc::new(listener));
        map
    }

    /// Add info listener that will be called when options change
    pub fn add_info_listener<F>(&mut self, f: F)
    where
        F: Fn(Option<String>) + Send + Sync + 'static,
    {
        self.on_info = Some(Arc::new(f));
    }

    /// Get integer value directly by name, with default
    pub fn get_int(&self, name: &str, default: i32) -> i32 {
        self.get(name).and_then(|o| o.get_int()).unwrap_or(default)
    }

    /// Get string value directly by name, with default
    pub fn get_string(&self, name: &str, default: &str) -> String {
        self.get(name)
            .and_then(|o| o.get_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| default.to_string())
    }
    /// Inits options and assigns idx in the correct printing order
    pub fn add(&mut self, mut opt: UciOption) {
        let key = opt.name.to_ascii_lowercase();
        if self.lookup.contains_key(&key) {
            eprintln!("Option \"{}\" was already added!", opt.name);
            std::process::exit(1);
        }
        opt.idx = self.insert_order;
        self.insert_order += 1;
        let index = self.options.len();
        self.lookup.insert(key, index);
        self.options.push(opt);
    }
    pub fn count(&self, name: &str) -> usize {
        self.lookup.get(&name.to_ascii_lowercase()).map_or(0, |_| 1)
    }
    pub fn get_mut(&mut self, name: &str) -> Option<&mut UciOption> {
        let key = name.to_ascii_lowercase();
        if let Some(&idx) = self.lookup.get(&key) {
            self.options.get_mut(idx)
        } else {
            None
        }
    }
    pub fn get(&self, name: &str) -> Option<&UciOption> {
        self.lookup
            .get(&name.to_ascii_lowercase())
            .and_then(|&i| self.options.get(i))
    }
    /// Parse `setoption name <name> value <value>` matching C++ behavior
    /// Returns the canonical option name only when the update was accepted.
    pub fn setoption(&mut self, line: &str) -> Option<String> {
        let mut parts = crate::misc::split_cpp_whitespace(line);
        let mut name = String::new();

        // Consume the "name" token. C++ discards whatever token comes first, so
        // `setoption value 3` looks up the option "3".
        parts.next();

        // Read option name (can contain spaces)
        for token in parts.by_ref() {
            if token == "value" {
                break;
            }
            if !name.is_empty() {
                name.push(' ');
            }
            name.push_str(token);
        }

        // Read option value (can contain spaces)
        let remaining: Vec<&str> = parts.collect();
        let value: String = remaining.join(" ");

        if self.count(&name) > 0 {
            if let Some(opt) = self.get_mut(&name) {
                let info = opt.assign(&value)?;
                let accepted_name = opt.name.clone();
                if let Some(callback) = &self.on_info {
                    callback(info);
                }
                return Some(accepted_name);
            }
        } else {
            crate::uci::UciEngine::write_stdout_bytes(
                format!("No such option: {name}\n").as_bytes(),
            );
        }
        None
    }

    /// Get option by name (implements C++ operator[])
    pub fn get_option(&self, name: &str) -> &UciOption {
        self.get(name).expect("Option not found")
    }
}

impl fmt::Display for OptionsMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print options in insertion order, matching C++ behavior
        for idx in 0..self.insert_order {
            if let Some(opt) = self.options.iter().find(|o| o.idx == idx) {
                write!(f, "\noption name {} type {}", opt.name, opt.type_name())?;
                match &opt.opt_type {
                    OptionType::Check | OptionType::Combo(_) => {
                        write!(f, " default {}", opt.default)?;
                    }
                    OptionType::String => {
                        let def = if opt.default.is_empty() {
                            "<empty>"
                        } else {
                            &opt.default
                        };
                        write!(f, " default {}", def)?;
                    }
                    OptionType::Spin => {
                        // Parse as float and convert to int for display, matching C++
                        let val: i32 = opt.default.parse::<f32>().unwrap_or(0.0) as i32;
                        write!(f, " default {} min {} max {}", val, opt.min, opt.max)?;
                    }
                    OptionType::Button => {}
                }
            }
        }
        Ok(())
    }
}

impl UciOption {
    fn type_name(&self) -> &'static str {
        match self.opt_type {
            OptionType::Check => "check",
            OptionType::Spin => "spin",
            OptionType::Combo(_) => "combo",
            OptionType::String => "string",
            OptionType::Button => "button",
        }
    }
}

// ---------------------------------------------------------------------------
// UNIT TESTS
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_insensitive_add_and_get() {
        let mut m = OptionsMap::new();
        m.add(UciOption::new_check("Test", true, None));
        assert!(m.get("test").is_some());
        assert!(m.get("TeSt").is_some());
        assert_eq!(m.count("test"), 1);
    }

    #[test]
    fn test_setoption_simple() {
        let mut m = OptionsMap::new();
        m.add(UciOption::new_spin("Depth", 5, 1, 10, None));
        m.setoption("name Depth value 7");
        assert_eq!(m.get("depth").unwrap().get_int(), Some(7));
    }

    #[test]
    fn accepted_update_is_distinct_from_absent_callback_info() {
        let mut options = OptionsMap::new();
        options.add(UciOption::new_spin(
            "Threads",
            1,
            1,
            8,
            Some(Arc::new(|_| None)),
        ));
        assert_eq!(
            options.setoption("name threads value 2"),
            Some("Threads".into())
        );
        for invalid in ["0", "9", "bad", "2 extra"] {
            assert_eq!(
                options.setoption(&format!("name Threads value {invalid}")),
                None
            );
            assert_eq!(options.get_int("Threads", 0), 2);
        }
        assert_eq!(
            options.setoption("name Threads value 2"),
            Some("Threads".into())
        );
    }

    #[test]
    fn test_display_order_and_format() {
        let mut m = OptionsMap::new();
        m.add(UciOption::new_check("OptA", false, None));
        m.add(UciOption::new_spin("OptB", 3, 0, 5, None));
        let s = format!("{}", m);
        assert!(s.contains("name OptA"));
        assert!(s.contains("type check"));
        assert!(s.contains("name OptB"));
        assert!(s.contains("type spin"));
        assert!(s.contains("min 0 max 5"));
        let a = s.find("name OptA").unwrap();
        let b = s.find("name OptB").unwrap();
        assert!(a < b);
    }

    #[test]
    fn test_combo_validation() {
        let mut m = OptionsMap::new();
        m.add(UciOption::new_combo("Mode", "one two", "one", None));
        assert!(m.get_mut("Mode").unwrap().set("invalid").is_none());
        assert!(m.get_mut("mode").unwrap().set("two").is_some());
    }

    #[test]
    fn test_on_change_info_forwarding() {
        use std::sync::{Arc, Mutex};

        let mut m = OptionsMap::new();
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen_clone = seen.clone();

        m.add_info_listener(move |s| {
            if let Some(msg) = s {
                seen_clone.lock().unwrap().push(msg);
            }
        });

        m.add(UciOption::new_spin(
            "Depth",
            5,
            1,
            10,
            Some(Arc::new(|o: &UciOption| {
                Some(format!("changed {}", o.name))
            })),
        ));
        m.setoption("name Depth value 7");
        assert_eq!(m.get("depth").unwrap().get_int(), Some(7));
        assert_eq!(seen.lock().unwrap().len(), 1);
        assert_eq!(seen.lock().unwrap()[0], "changed Depth");
    }
}
