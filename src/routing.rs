//! Exact Rust port of Haidass-Image-Instruct prompt routing and spelling annotation
//! (`runtime/routing.py`, `runtime/add_spelling.py`, `runtime/add_spelling_v2.py`, `runtime/spelling.py`).

use regex::Regex;
use std::sync::OnceLock;

fn quote_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#""([^"\n]+)"|“([^”\n]+)”|「([^」\n]+)」|『([^』\n]+)』|'([^'\n]+)'|‘([^’\n]+)’"#,
        )
        .unwrap()
    })
}

fn quoted_double() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#""([^"\n]*)"|“([^”\n]*)”"#).unwrap())
}

fn english_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z]+(?:\s+[A-Za-z]+)*$").unwrap())
}

fn spelling_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*\(spelled: [^\n)]*\)").unwrap())
}

fn annotation_v1() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*\(spelled: [A-Za-z](?: [A-Za-z]| / [A-Za-z])*\)").unwrap()
    })
}

fn annotation_aug() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*\(spelled: [^()\n]+\)").unwrap())
}

fn annotation_v2() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*\(spelled:\s*[^)\n]*\)").unwrap())
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Finds non-overlapping quote matches following `routing.py`'s `QUOTE_PATTERN`
/// (including `(?<!\w)'...'(?!\w)` check since Rust `regex` omits lookarounds).
fn find_quote_matches(prompt: &str) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    for m in quote_pattern().find_iter(prompt) {
        let s = m.as_str();
        if s.starts_with('\'') {
            let prev_word = prompt[..m.start()]
                .chars()
                .next_back()
                .map(is_word_char)
                .unwrap_or(false);
            let next_word = prompt[m.end()..]
                .chars()
                .next()
                .map(is_word_char)
                .unwrap_or(false);
            if prev_word || next_word {
                continue;
            }
        }
        matches.push((m.start(), m.end()));
    }
    matches
}

pub fn triggers_writing(prompt: &str) -> bool {
    for (start, end) in find_quote_matches(prompt) {
        let quoted = &prompt[start..end];
        let mut chars = quoted.chars();
        chars.next();
        chars.next_back();
        if !chars.as_str().trim().is_empty() {
            return true;
        }
    }
    false
}

fn spell_words(target: &str) -> String {
    target
        .split_whitespace()
        .map(|w| {
            let mut s = String::new();
            for (i, ch) in w.chars().enumerate() {
                if i > 0 {
                    s.push(' ');
                }
                s.push(ch);
            }
            s
        })
        .collect::<Vec<_>>()
        .join(" / ")
}

fn add_spelling_v1(prompt: &str) -> String {
    let mut out = String::with_capacity(prompt.len() + 64);
    let mut last = 0;
    for caps in quoted_double().captures_iter(prompt) {
        let m = caps.get(0).unwrap();
        out.push_str(&prompt[last..m.start()]);
        let target = caps
            .get(1)
            .or_else(|| caps.get(2))
            .map(|x| x.as_str())
            .unwrap_or("");
        if !english_re().is_match(target) || annotation_v1().is_match(&prompt[m.end()..]) {
            out.push_str(m.as_str());
        } else {
            out.push_str(m.as_str());
            out.push_str(" (spelled: ");
            out.push_str(&spell_words(target));
            out.push(')');
        }
        last = m.end();
    }
    out.push_str(&prompt[last..]);
    out
}

fn augment_caption(caption: &str) -> String {
    if !quoted_double().is_match(caption) {
        return caption.to_string();
    }
    let augmented = add_spelling_v1(caption);
    let mut out = String::with_capacity(augmented.len() + 64);
    let mut last = 0;
    for caps in quoted_double().captures_iter(&augmented) {
        let m = caps.get(0).unwrap();
        out.push_str(&augmented[last..m.start()]);
        let target = caps
            .get(1)
            .or_else(|| caps.get(2))
            .map(|x| x.as_str())
            .unwrap_or("");
        if target.trim().is_empty() || annotation_aug().is_match(&augmented[m.end()..]) {
            out.push_str(m.as_str());
        } else {
            out.push_str(m.as_str());
            out.push_str(" (spelled: ");
            out.push_str(&spell_words(target));
            out.push(')');
        }
        last = m.end();
    }
    out.push_str(&augmented[last..]);
    out
}

fn spell_prompt(prompt: &str) -> String {
    let matches = find_quote_matches(prompt);
    let mut out = String::with_capacity(prompt.len() + 64);
    let mut last = 0;
    for (start, end) in matches {
        out.push_str(&prompt[last..start]);
        let quoted = &prompt[start..end];
        if spelling_pattern().is_match(&prompt[end..]) {
            out.push_str(quoted);
        } else {
            let first_ch = quoted.chars().next().unwrap();
            if first_ch == '"' || first_ch == '\u{201c}' {
                out.push_str(&augment_caption(quoted));
            } else {
                let mut inner = quoted.chars();
                inner.next();
                inner.next_back();
                let canonical = format!("\"{}\"", inner.as_str());
                let aug = augment_caption(&canonical);
                out.push_str(quoted);
                out.push_str(&aug[canonical.len()..]);
            }
        }
        last = end;
    }
    out.push_str(&prompt[last..]);
    out
}

fn is_escaped(chars: &[char], pos: usize) -> bool {
    let mut start = pos;
    while start > 0 && chars[start - 1] == '\\' {
        start -= 1;
    }
    (pos - start) % 2 == 1
}

fn is_apostrophe(chars: &[char], pos: usize) -> bool {
    pos > 0
        && pos + 1 < chars.len()
        && chars[pos - 1].is_alphanumeric()
        && chars[pos + 1].is_alphanumeric()
}

fn closer_for(c: char) -> Option<char> {
    match c {
        '"' => Some('"'),
        '\'' => Some('\''),
        '\u{201c}' => Some('\u{201d}'),
        '\u{2018}' => Some('\u{2019}'),
        _ => None,
    }
}

/// Returns `(char_start, char_end, inner_string)` spans matching `add_spelling_v2._quoted_spans`.
fn quoted_spans_v2(prompt: &str) -> Vec<(usize, usize, String)> {
    let chars: Vec<char> = prompt.chars().collect();
    let mut spans = Vec::new();
    let mut pos = 0;
    while pos < chars.len() {
        if chars[pos..].starts_with(&['(', 's', 'p', 'e', 'l', 'l', 'e', 'd', ':']) {
            if let Some(rel) = chars[pos + 8..].iter().position(|&c| c == ')') {
                pos = pos + 8 + rel + 1;
                continue;
            }
        }
        let opener = chars[pos];
        let Some(closer) = closer_for(opener) else {
            pos += 1;
            continue;
        };
        if is_escaped(&chars, pos)
            || ((opener == '\'' || opener == '\u{2018}') && is_apostrophe(&chars, pos))
        {
            pos += 1;
            continue;
        }
        if (opener == '\'' || opener == '\u{2018}') && pos > 0 && chars[pos - 1].is_alphanumeric() {
            pos += 1;
            continue;
        }
        let mut end = pos + 1;
        let mut found = false;
        while end < chars.len() && chars[end] != '\r' && chars[end] != '\n' {
            if chars[end] == closer && !is_escaped(&chars, end) {
                if (closer == '\'' || closer == '\u{2019}') && is_apostrophe(&chars, end) {
                    end += 1;
                    continue;
                }
                let inner: String = chars[pos + 1..end].iter().collect();
                spans.push((pos, end + 1, inner));
                pos = end + 1;
                found = true;
                break;
            }
            end += 1;
        }
        if !found {
            pos += 1;
        }
    }
    spans
}

fn unescape_quote_chars(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&nxt) = chars.peek() {
                if nxt == '"' || nxt == '\'' || nxt == '\\' {
                    out.push(nxt);
                    chars.next();
                    continue;
                }
            }
        }
        out.push(c);
    }
    out
}

fn is_unicode_punctuation(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(
            c,
            '\u{2010}'..='\u{2027}'
                | '\u{2030}'..='\u{205E}'
                | '\u{3001}'..='\u{301F}'
                | '\u{FF01}'..='\u{FF0F}'
                | '\u{FF1A}'..='\u{FF20}'
                | '\u{FF3B}'..='\u{FF40}'
                | '\u{FF5B}'..='\u{FF65}'
        )
}

fn add_spelling_v2(prompt: &str) -> String {
    let chars: Vec<char> = prompt.chars().collect();
    // Map char indices to byte offsets
    let mut byte_offs: Vec<usize> = prompt.char_indices().map(|(b, _)| b).collect();
    byte_offs.push(prompt.len());

    let mut out = String::with_capacity(prompt.len() + 64);
    let mut cursor = 0usize;
    for (_start, end, target) in quoted_spans_v2(prompt) {
        if target.trim().is_empty() || annotation_v2().is_match(&prompt[byte_offs[end]..]) {
            continue;
        }
        let clean = unescape_quote_chars(&target);
        let spelled = spell_words(&clean);
        out.push_str(&prompt[byte_offs[cursor]..byte_offs[end]]);
        out.push_str(" (spelled: ");
        out.push_str(&spelled);
        out.push(')');
        cursor = end;
    }
    out.push_str(&prompt[byte_offs[cursor]..]);
    let _ = chars;
    out
}

/// Exact equivalent of `inference.py::route_prompt(prompt)`.
/// Returns `(processed_prompt, use_writing_lora)`.
pub fn route_prompt(prompt: &str) -> (String, bool) {
    if !triggers_writing(prompt) {
        return (prompt.to_string(), false);
    }
    let chars: Vec<char> = prompt.chars().collect();
    let spans = quoted_spans_v2(prompt);
    let needs_v2 = spans.iter().any(|(start, _, target)| {
        let opener = chars[*start];
        opener == '\''
            || opener == '\u{2018}'
            || target
                .chars()
                .any(|ch| ch.is_numeric() || is_unicode_punctuation(ch))
    });
    let processed = if needs_v2 {
        add_spelling_v2(prompt)
    } else {
        spell_prompt(prompt)
    };
    (processed, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_prompt_plain() {
        let (p, w) = route_prompt("A red fox in snow.");
        assert!(!w);
        assert_eq!(p, "A red fox in snow.");
    }

    #[test]
    fn test_route_prompt_quoted() {
        let (p, w) = route_prompt(r#"A teal enamel sign reading "OPEN LATE" in a window"#);
        assert!(w);
        assert_eq!(
            p,
            r#"A teal enamel sign reading "OPEN LATE" (spelled: O P E N / L A T E) in a window"#
        );
    }
}
